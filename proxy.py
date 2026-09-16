#!/usr/bin/env python3
"""OpenAI-compat sidecar that forges llama.cpp `timings` on Halogen streams.

llama-swap (v141+) reads the last SSE chunk's
`timings.predicted_per_second` / `timings.prompt_per_second`. Halogen's
stop chunk has neither timings nor usage. This process sits in front of
Halogen, stamps TTFT vs. decode on the wire, and leaves every other
byte alone.

Clocking is client-visible wall time (HTTP + template + engine), not
Halogen's internal kernel clocks. Prefill tok/s is therefore a hair
low vs. `sweep`; decode tok/s is close.

Token counts prefer, in order: a parsed `serve_api:` log line (same
process; Halogen already prints prompt/cached/prefill/decode), then
upstream `usage`, then a text estimate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import OrderedDict, deque
from urllib.parse import urlsplit

log = logging.getLogger("halogen-sidecar")

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

STREAM_PATHS = (
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/completions",
    "/completions",
)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = 0
    other = 0
    for ch in text:
        o = ord(ch)
        if (
            0x4E00 <= o <= 0x9FFF
            or 0x3400 <= o <= 0x4DBF
            or 0x3040 <= o <= 0x30FF
            or 0xAC00 <= o <= 0xD7AF
        ):
            cjk += 1
        elif ch.isspace():
            continue
        else:
            other += 1
    return max(1, cjk + max(1, other // 4)) if (cjk or other) else 0


def content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text") or "")
                elif "text" in p:
                    parts.append(str(p.get("text") or ""))
        return "".join(parts)
    return str(content)


def estimate_prompt_tokens(req: dict) -> int:
    n = 8  # chat-template slop
    if isinstance(req.get("prompt"), str):
        n += estimate_tokens(req["prompt"])
    for msg in req.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        n += estimate_tokens(content_to_text(msg.get("content")))
        n += estimate_tokens(content_to_text(msg.get("reasoning_content") or ""))
    return n


def _msg_fp(msg: dict) -> str:
    blob = json.dumps(
        {
            "role": msg.get("role") or "",
            "content": content_to_text(msg.get("content")),
            "reasoning": content_to_text(msg.get("reasoning_content") or ""),
            "name": msg.get("name") or "",
            "tool_call_id": msg.get("tool_call_id") or "",
            "tool_calls": msg.get("tool_calls") or None,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def prefix_fingerprint(model: str, messages: list, upto: int) -> str:
    parts = [_msg_fp(m) for m in messages[:upto] if isinstance(m, dict)]
    return hashlib.sha256(f"{model}|{'|'.join(parts)}".encode()).hexdigest()


def prompt_fingerprint(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}|prompt|{prompt}".encode()).hexdigest()


class PrefixMemory:
    """Best-effort KV-cache accounting across turns.

    Halogen does not send cache_n. Chat clients resubmit the full history,
    and Halogen's prompt cache (default mode 2) resumes the shared prefix.
    Remember how many tokens each seen prefix occupied after the last
    request that ended on that prefix; the next turn's leftover is new prefill.
    """

    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._data: OrderedDict[str, int] = OrderedDict()

    def _get(self, key: str) -> int | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def _put(self, key: str, tokens: int) -> None:
        self._data[key] = max(0, int(tokens))
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def lookup(self, req: dict, total_prompt: int) -> int:
        model = str(req.get("model") or "")
        messages = [m for m in (req.get("messages") or []) if isinstance(m, dict)]
        best = 0
        if messages:
            # Longest stored prefix of the incoming history.
            for i in range(len(messages), 0, -1):
                hit = self._get(prefix_fingerprint(model, messages, i))
                if hit is not None:
                    best = max(best, hit)
                    break
        prompt = req.get("prompt")
        if isinstance(prompt, str) and prompt:
            hit = self._get(prompt_fingerprint(model, prompt))
            if hit is not None:
                best = max(best, hit)
        return min(best, max(0, total_prompt - 1))

    def remember(
        self, req: dict, prompt_tokens: int, completion_tokens: int, gen_text: str
    ) -> None:
        model = str(req.get("model") or "")
        messages = [m for m in (req.get("messages") or []) if isinstance(m, dict)]
        if messages:
            # Prefix as submitted (after prefill, before this completion).
            self._put(prefix_fingerprint(model, messages, len(messages)), prompt_tokens)
            # Prefix as the next turn will resubmit it: history + this assistant.
            grown = list(messages) + [{"role": "assistant", "content": gen_text}]
            self._put(
                prefix_fingerprint(model, grown, len(grown)),
                prompt_tokens + completion_tokens,
            )
        prompt = req.get("prompt")
        if isinstance(prompt, str) and prompt:
            self._put(prompt_fingerprint(model, prompt), prompt_tokens)
            self._put(
                prompt_fingerprint(model, prompt + gen_text),
                prompt_tokens + completion_tokens,
            )


# Served HTTP prefill on this hardware tops out around 1.0–1.4k tok/s.
# Anything far above that on a long prompt is cache, not GEMM.
PREFILL_CEILING = float(os.environ.get("HALOGEN_PREFILL_CEILING", "1800"))
UPSTREAM_LOG = os.environ.get("HALOGEN_UPSTREAM_LOG", "/tmp/halogen-upstream.log")

# serve_api: mtp 250 tok in 6.75s = 37.05 t/s | 153 rounds, commit 1.63/round | prompt 1287 (1035 cached), prefill 1.26s | detok 31us/tok
SERVE_API_RE = re.compile(
    r"serve_api:\s+(\S+)\s+(\d+)\s+tok\s+in\s+([\d.]+)s\s+=\s+([\d.]+)\s+t/s"
    r"(?:\s+\|\s+(\d+)\s+rounds,\s+commit\s+([\d.]+)/round)?"
    r".*?\|\s+prompt\s+(\d+)(?:\s+\((\d+)\s+cached\))?\s*,\s*prefill\s+([\d.]+)s"
)


def parse_serve_api_line(line: str) -> dict | None:
    m = SERVE_API_RE.search(line)
    if not m:
        return None
    drafter = m.group(1)
    predicted_n = int(m.group(2))
    decode_s = float(m.group(3))
    decode_tps = float(m.group(4))
    rounds = int(m.group(5) or 0)
    commit_per_round = float(m.group(6) or 0.0)
    prompt_total = int(m.group(7))
    cached = int(m.group(8) or 0)
    prefill_s = float(m.group(9))
    prompt_n = max(0, prompt_total - cached)
    # Each MTP round always emits one target-model token. Everything above
    # `rounds` is accepted draft. commit 2.04/round means the head proposed
    # at least two extras that round, so draft_n is rounds × that width,
    # not rounds itself (which is what made accept look like 104%).
    extras = max(0, predicted_n - rounds) if rounds else 0
    extra_per_round = (commit_per_round - 1.0) if commit_per_round else 0.0
    width = max(1, int(math.ceil(extra_per_round - 1e-9))) if extra_per_round > 0 else 1
    if drafter == "serial" or rounds <= 0:
        draft_n = 0
        draft_n_accepted = 0
    else:
        draft_n = rounds * width
        draft_n_accepted = extras
    return {
        "drafter": drafter,
        "predicted_n": predicted_n,
        "decode_s": decode_s,
        "decode_tps": decode_tps,
        "prompt_total": prompt_total,
        "cache_n": cached,
        "prompt_n": prompt_n,
        "prefill_s": prefill_s,
        "prompt_tps": (prompt_n / prefill_s) if prefill_s > 0 and prompt_n else 0.0,
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "commit_per_round": commit_per_round,
    }


class ServeApiLog:
    """Follow Halogen's own request log and hand the matching line back."""

    def __init__(self, path: str):
        self.path = path
        self._fh = None
        self._buf: deque = deque(maxlen=48)

    def ingest(self) -> None:
        if not self.path:
            return
        try:
            if self._fh is None:
                if not os.path.exists(self.path):
                    return
                # wrap.sh truncates this file on boot. Read from the start so
                # the first request's serve_api line is not skipped by a
                # seek(END) that races the tee.
                # Persistent handle: kept open across ingest() calls to
                # stream-follow the tee'd log.
                self._fh = open(  # noqa: SIM115
                    self.path, encoding="utf-8", errors="replace"
                )
        except OSError:
            return
        while True:
            line = self._fh.readline()
            if not line:
                break
            parsed = parse_serve_api_line(line)
            if parsed:
                parsed["t"] = time.monotonic()
                self._buf.append(parsed)

    async def take(
        self,
        ttft_s: float,
        est_pred: int,
        started_mono: float | None = None,
    ) -> dict | None:
        # serve_api prefill is engine time only. Sidecar TTFT also counts
        # slot-wait, so a 0.4s clock gate drops the right line on a long
        # queue (13s TTFT vs 5.3s prefill). Take the unpaired line that
        # appeared during this request; use the clock only as a tie-break.
        deadline = time.monotonic() + 0.8
        origin = (
            (started_mono - 1.0)
            if started_mono is not None
            else time.monotonic() - 180.0
        )
        while True:
            self.ingest()
            now = time.monotonic()
            cands = [
                row
                for row in list(self._buf)
                if row["t"] >= origin and now - row["t"] < 180.0
            ]
            pick = None
            if len(cands) == 1:
                pick = cands[0]
            elif len(cands) > 1:

                def score(row: dict) -> float:
                    gap = ttft_s - row["prefill_s"]
                    if gap >= -0.4:
                        clock = min(abs(gap), 8.0) * 0.02
                    else:
                        clock = 20.0 + abs(gap)
                    if est_pred:
                        clock += 0.0004 * abs(row["predicted_n"] - est_pred)
                    return clock

                pick = min(cands, key=score)
            if pick is not None:
                try:
                    self._buf.remove(pick)
                except ValueError:
                    pass
                return pick
            if time.monotonic() >= deadline:
                return None
            await asyncio.sleep(0.05)


def split_cached_prompt(
    total_prompt: int,
    cache_hint: int,
    prompt_ms: float,
    ceiling: float = PREFILL_CEILING,
) -> tuple[int, int]:
    """Return (cache_n, prompt_n) with prompt_n = tokens actually prefills."""
    total_prompt = max(0, int(total_prompt))
    cache_n = (
        min(max(0, int(cache_hint)), max(0, total_prompt - 1)) if total_prompt else 0
    )
    prompt_n = max(0, total_prompt - cache_n)
    if prompt_ms > 150.0 and prompt_n > 64:
        implied = prompt_n / (prompt_ms / 1000.0)
        if implied > ceiling > 0:
            processed = max(1, int(ceiling * prompt_ms / 1000.0))
            processed = min(processed, prompt_n)
            cache_n = total_prompt - processed
            prompt_n = processed
    if total_prompt and prompt_n <= 0:
        prompt_n = 1
        cache_n = total_prompt - 1
    return cache_n, prompt_n


def cached_tokens_from_usage(usage: dict | None) -> int:
    if not isinstance(usage, dict):
        return 0
    for key in ("cache_tokens", "cached_tokens", "prompt_cache_tokens"):
        if usage.get(key):
            try:
                return int(usage[key])
            except (TypeError, ValueError):
                pass
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens"):
        try:
            return int(details["cached_tokens"])
        except (TypeError, ValueError):
            pass
    return 0


def delta_text(delta: dict) -> str:
    if not isinstance(delta, dict):
        return ""
    return content_to_text(delta.get("content")) + content_to_text(
        delta.get("reasoning_content")
    )


def finish_reason_of(obj: dict) -> str | None:
    for ch in obj.get("choices") or []:
        if isinstance(ch, dict) and ch.get("finish_reason"):
            return str(ch["finish_reason"])
    return None


def usage_of(obj: dict) -> dict | None:
    u = obj.get("usage")
    return u if isinstance(u, dict) else None


def build_timings(
    *,
    t0: float,
    t_first: float | None,
    t_end: float,
    prompt_n: int,
    predicted_n: int,
    cache_n: int = 0,
) -> dict:
    if t_first is None:
        t_first = t_end
    prompt_ms = max(0.0, (t_first - t0) * 1000.0)
    predicted_ms = max(0.0, (t_end - t_first) * 1000.0)
    # llama.cpp still reports a rate with n=0 as 0
    prompt_ps = (prompt_n / prompt_ms * 1000.0) if prompt_ms > 0 else 0.0
    pred_ps = (predicted_n / predicted_ms * 1000.0) if predicted_ms > 0 else 0.0
    return {
        "cache_n": int(cache_n),
        "prompt_n": int(prompt_n),
        "prompt_ms": prompt_ms,
        "prompt_per_token_ms": (prompt_ms / prompt_n) if prompt_n else 0.0,
        "prompt_per_second": prompt_ps,
        "predicted_n": int(predicted_n),
        "predicted_ms": predicted_ms,
        "predicted_per_token_ms": (predicted_ms / predicted_n) if predicted_n else 0.0,
        "predicted_per_second": pred_ps,
    }


def attach_metrics(obj: dict, timings: dict, usage: dict | None) -> dict:
    # Do not clobber a real upstream timings object if Halogen grows one.
    existing = obj.get("timings")
    if not (isinstance(existing, dict) and existing.get("predicted_per_second")):
        obj["timings"] = timings
    if usage:
        obj["usage"] = usage
    elif "usage" not in obj:
        obj["usage"] = {
            "prompt_tokens": timings["prompt_n"],
            "completion_tokens": timings["predicted_n"],
            "total_tokens": timings["prompt_n"] + timings["predicted_n"],
        }
    return obj


async def read_headers(reader: asyncio.StreamReader) -> list[tuple[str, str]]:
    headers: list[tuple[str, str]] = []
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            break
        raw = line.decode("latin1").rstrip("\r\n")
        if ":" not in raw:
            continue
        k, v = raw.split(":", 1)
        headers.append((k.strip(), v.strip()))
    return headers


def header_map(headers: list[tuple[str, str]]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers}


async def read_body(reader: asyncio.StreamReader, hmap: dict[str, str]) -> bytes:
    if hmap.get("transfer-encoding", "").lower() == "chunked":
        chunks = []
        while True:
            size_line = await reader.readline()
            if not size_line:
                break
            size = int(size_line.split(b";", 1)[0].strip() or b"0", 16)
            if size == 0:
                await reader.readline()
                break
            chunks.append(await reader.readexactly(size))
            await reader.readline()
        return b"".join(chunks)
    n = hmap.get("content-length")
    if n:
        return await reader.readexactly(int(n))
    return b""


def filter_headers(
    headers: list[tuple[str, str]], extra_drop: set[str] | None = None
) -> list[tuple[str, str]]:
    drop = HOP_BY_HOP | {x.lower() for x in (extra_drop or ())}
    return [(k, v) for k, v in headers if k.lower() not in drop]


def encode_headers(status_line: str, headers: list[tuple[str, str]]) -> bytes:
    out = [status_line.rstrip() + "\r\n"]
    for k, v in headers:
        out.append(f"{k}: {v}\r\n")
    out.append("\r\n")
    return "".join(out).encode("latin1")


def maybe_augment_request(path: str, method: str, body: bytes) -> tuple[bytes, bool]:
    """Return (body, want_sse_rewrite)."""
    path_only = path.split("?", 1)[0]
    if method != "POST" or path_only.rstrip("/") not in {
        p.rstrip("/") for p in STREAM_PATHS
    }:
        return body, False
    try:
        req = json.loads(body.decode("utf-8"))
    except Exception:
        return body, False
    if not isinstance(req, dict):
        return body, False
    streaming = bool(req.get("stream"))
    if streaming:
        so = req.get("stream_options")
        if not isinstance(so, dict):
            so = {}
        so.setdefault("include_usage", True)
        req["stream_options"] = so
        body = json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    return body, streaming


async def write_simple(
    writer: asyncio.StreamWriter,
    status: str,
    body: bytes,
    extra: list[tuple[str, str]] | None = None,
) -> None:
    headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(body))),
        ("Connection", "close"),
        ("X-Halogen-Sidecar", "timings-1"),
    ]
    if extra:
        headers.extend(extra)
    writer.write(f"HTTP/1.1 {status}\r\n".encode("latin1"))
    writer.write(encode_headers("", headers)[2:])
    writer.write(body)
    await writer.drain()


class Sidecar:
    def __init__(self, upstream_host: str, upstream_port: int):
        self.upstream_host = upstream_host
        self.upstream_port = upstream_port
        self._upstream_up = False
        self._last_down_log = 0.0
        self._down_hits = 0
        self.prefixes = PrefixMemory()
        self.api_log = ServeApiLog(UPSTREAM_LOG)
        self.last_engine_prompt_total = 0

    async def watch_upstream(self) -> None:
        while True:
            ok = await self._probe()
            if ok and not self._upstream_up:
                log.info(
                    "upstream ready on %s:%s",
                    self.upstream_host,
                    self.upstream_port,
                )
                self._upstream_up = True
                self._down_hits = 0
            elif not ok and self._upstream_up:
                log.warning(
                    "upstream lost on %s:%s",
                    self.upstream_host,
                    self.upstream_port,
                )
                self._upstream_up = False
            await asyncio.sleep(2.0)

    async def _probe(self) -> bool:
        try:
            r, w = await asyncio.wait_for(
                asyncio.open_connection(self.upstream_host, self.upstream_port),
                timeout=1.0,
            )
        except Exception:
            return False
        try:
            w.close()
            await w.wait_closed()
        except Exception:
            pass
        return True

    def _note_down(self) -> None:
        self._down_hits += 1
        now = time.monotonic()
        if now - self._last_down_log < 10.0:
            return
        self._last_down_log = now
        log.info(
            "halogen API %s:%s not listening yet "
            "(%s probes while engine cold-loads; answering 503)",
            self.upstream_host,
            self.upstream_port,
            self._down_hits,
        )

    async def open_upstream(self):
        # Short retry only covers the API process coming up a moment
        # after the engine banner. Cold load is minutes; we 503 for that.
        last: Exception | None = None
        for attempt in range(4):
            try:
                return await asyncio.wait_for(
                    asyncio.open_connection(self.upstream_host, self.upstream_port),
                    timeout=0.75,
                )
            except (
                ConnectionRefusedError,
                ConnectionResetError,
                OSError,
                asyncio.TimeoutError,
            ) as e:
                last = e
                await asyncio.sleep(0.25 * (attempt + 1))
        assert last is not None
        raise last

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername")
        try:
            req_line = await reader.readline()
            if not req_line:
                return
            parts = req_line.decode("latin1").rstrip("\r\n").split()
            if len(parts) < 2:
                return
            method, path = parts[0], parts[1]
            headers = await read_headers(reader)
            hmap = header_map(headers)
            body = await read_body(reader, hmap)
            body, rewrite = maybe_augment_request(path, method, body)

            req: dict = {}
            if rewrite:
                try:
                    req = json.loads(body.decode("utf-8"))
                except Exception:
                    req = {}

            fwd = filter_headers(headers, {"content-length", "host"})
            fwd.append(("Host", f"{self.upstream_host}:{self.upstream_port}"))
            fwd.append(("Content-Length", str(len(body))))
            fwd.append(("Connection", "close"))

            try:
                up_r, up_w = await self.open_upstream()
            except (
                ConnectionRefusedError,
                ConnectionResetError,
                OSError,
                asyncio.TimeoutError,
            ):
                self._note_down()
                payload = json.dumps(
                    {
                        "error": {
                            "message": (
                                "halogen API is not listening yet; "
                                "engine cold-load can take several minutes"
                            ),
                            "type": "upstream_unavailable",
                            "code": "engine_loading",
                        }
                    }
                ).encode("utf-8")
                await write_simple(
                    writer,
                    "503 Service Unavailable",
                    payload,
                    extra=[("Retry-After", "5")],
                )
                return

            self._upstream_up = True
            try:
                up_w.write(f"{method} {path} HTTP/1.1\r\n".encode("latin1"))
                up_w.write(encode_headers("", fwd)[2:])  # headers + CRLF only
                if body:
                    up_w.write(body)
                await up_w.drain()

                status = await up_r.readline()
                up_headers = await read_headers(up_r)
                up_map = header_map(up_headers)
                ctype = up_map.get("content-type", "")
                is_sse = rewrite and "text/event-stream" in ctype

                out_headers = filter_headers(up_headers, {"content-length"})
                out_headers.append(("X-Halogen-Sidecar", "timings-1"))
                if is_sse:
                    # We rewrite the last event; length is unknown.
                    out_headers = filter_headers(out_headers, {"content-length"})
                    out_headers.append(("Cache-Control", "no-cache"))

                writer.write(status if status.endswith(b"\n") else status + b"\r\n")
                writer.write(encode_headers("", out_headers)[2:])
                await writer.drain()

                if is_sse:
                    await self._pipe_sse(up_r, writer, req)
                else:
                    if rewrite and "application/json" in ctype:
                        raw = await self._read_rest(up_r, up_map)
                        raw = self._maybe_patch_json(raw, req)
                        writer.write(raw)
                    else:
                        await self._pipe_raw(up_r, writer)
                await writer.drain()
            finally:
                up_w.close()
                try:
                    await up_w.wait_closed()
                except Exception:
                    pass
        except (
            asyncio.IncompleteReadError,
            ConnectionResetError,
            BrokenPipeError,
        ) as e:
            log.debug("peer %s closed: %s", peer, e)
        except (ConnectionRefusedError, asyncio.TimeoutError):
            self._note_down()
        except Exception:
            log.exception("proxy error for %s", peer)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_rest(
        self, reader: asyncio.StreamReader, hmap: dict[str, str]
    ) -> bytes:
        te = hmap.get("transfer-encoding", "").lower()
        if "chunked" in te:
            return await read_body(reader, {"transfer-encoding": "chunked"})
        n = hmap.get("content-length")
        if n:
            return await reader.readexactly(int(n))
        return await reader.read()

    async def _pipe_raw(
        self, src: asyncio.StreamReader, dst: asyncio.StreamWriter
    ) -> None:
        while True:
            chunk = await src.read(64 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            await dst.drain()

    def _maybe_patch_json(self, raw: bytes, req: dict) -> bytes:
        """Non-stream: we cannot split prefill/decode. Leave body alone.

        llama-swap will show tok/s as unknown unless timings exist; inventing
        a single-bucket rate would put prefill into decode and lie.
        """
        return raw

    async def _pipe_sse(
        self,
        src: asyncio.StreamReader,
        dst: asyncio.StreamWriter,
        req: dict,
    ) -> None:
        t0 = time.perf_counter()
        started_mono = time.monotonic()
        t_first: float | None = None
        gen_text: list[str] = []
        seen_usage: dict | None = None
        prompt_n_est = estimate_prompt_tokens(req)
        buf = b""

        async def emit(blob: bytes) -> None:
            dst.write(blob)
            await dst.drain()

        while True:
            chunk = await src.read(64 * 1024)
            if not chunk:
                if buf:
                    await emit(buf)
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                raw_line = line.decode("utf-8", errors="replace")
                stripped = raw_line.rstrip("\r")

                if stripped == "" or stripped.startswith(":"):
                    await emit(line + b"\n")
                    continue

                if not stripped.startswith("data:"):
                    await emit(line + b"\n")
                    continue

                payload = stripped[5:].lstrip()
                if payload == "[DONE]":
                    await emit(line + b"\n")
                    if buf:
                        await emit(buf)
                        buf = b""
                    return

                try:
                    obj = json.loads(payload)
                except Exception:
                    await emit(line + b"\n")
                    continue

                if not isinstance(obj, dict):
                    await emit(line + b"\n")
                    continue

                u = usage_of(obj)
                if u:
                    seen_usage = u

                for ch in obj.get("choices") or []:
                    if not isinstance(ch, dict):
                        continue
                    piece = delta_text(ch.get("delta") or {})
                    if piece:
                        if t_first is None:
                            t_first = time.perf_counter()
                        gen_text.append(piece)

                if finish_reason_of(obj) is None and not (
                    isinstance(obj.get("timings"), dict)
                    and obj["timings"].get("predicted_per_second")
                ):
                    await emit(line + b"\n")
                    continue

                t_end = time.perf_counter()
                if seen_usage:
                    total_prompt = int(
                        seen_usage.get("prompt_tokens")
                        or seen_usage.get("input_tokens")
                        or prompt_n_est
                    )
                    predicted_n = int(
                        seen_usage.get("completion_tokens")
                        or seen_usage.get("output_tokens")
                        or 0
                    )
                else:
                    total_prompt = prompt_n_est
                    predicted_n = estimate_tokens("".join(gen_text))

                predicted_n = max(predicted_n, 0)
                ttft_s = (t_first - t0) if t_first is not None else (t_end - t0)
                engine = await self.api_log.take(
                    ttft_s, predicted_n, started_mono=started_mono
                )

                if engine:
                    cache_n = engine["cache_n"]
                    prompt_n = engine["prompt_n"]
                    total_prompt = engine["prompt_total"]
                    predicted_n = engine["predicted_n"]
                    timings = {
                        "cache_n": cache_n,
                        "prompt_n": prompt_n,
                        "prompt_ms": engine["prefill_s"] * 1000.0,
                        "prompt_per_token_ms": (
                            (engine["prefill_s"] * 1000.0 / prompt_n)
                            if prompt_n
                            else 0.0
                        ),
                        "prompt_per_second": engine["prompt_tps"],
                        "predicted_n": predicted_n,
                        "predicted_ms": engine["decode_s"] * 1000.0,
                        "predicted_per_token_ms": (
                            (engine["decode_s"] * 1000.0 / predicted_n)
                            if predicted_n
                            else 0.0
                        ),
                        "predicted_per_second": engine["decode_tps"],
                    }
                    if engine.get("draft_n"):
                        timings["draft_n"] = engine["draft_n"]
                        timings["draft_n_accepted"] = engine["draft_n_accepted"]
                    source = "serve_api"
                    self.last_engine_prompt_total = total_prompt
                else:
                    cache_hint = max(
                        cached_tokens_from_usage(seen_usage),
                        self.prefixes.lookup(req, total_prompt),
                        self.last_engine_prompt_total,
                    )
                    prompt_ms = max(0.0, ttft_s * 1000.0)
                    cache_n, prompt_n = split_cached_prompt(
                        total_prompt, cache_hint, prompt_ms
                    )
                    timings = build_timings(
                        t0=t0,
                        t_first=t_first,
                        t_end=t_end,
                        prompt_n=prompt_n,
                        predicted_n=predicted_n,
                        cache_n=cache_n,
                    )
                    source = "estimate"

                seen_usage = {
                    "prompt_tokens": total_prompt,
                    "completion_tokens": predicted_n,
                    "total_tokens": total_prompt + predicted_n,
                    "prompt_tokens_details": {"cached_tokens": cache_n},
                }
                attach_metrics(obj, timings, seen_usage)
                self.prefixes.remember(
                    req, total_prompt, predicted_n, "".join(gen_text)
                )
                new_line = "data: " + json.dumps(
                    obj, ensure_ascii=False, separators=(",", ":")
                )
                await emit(new_line.encode("utf-8") + b"\n")
                draft_n = timings.get("draft_n") or 0
                accepted = timings.get("draft_n_accepted") or 0
                accept = (accepted / draft_n) if draft_n else 0.0
                log.info(
                    "timings [%s] cache_n=%s prompt_n=%s prompt=%.1f tok/s "
                    "predicted_n=%s decode=%.1f tok/s ttft=%.3fs"
                    "%s",
                    source,
                    timings["cache_n"],
                    timings["prompt_n"],
                    timings["prompt_per_second"],
                    timings["predicted_n"],
                    timings["predicted_per_second"],
                    ttft_s,
                    (
                        f" draft_n={draft_n} accepted={accepted} accept={accept:.2f}"
                        if draft_n
                        else ""
                    ),
                )


async def main() -> None:
    p = argparse.ArgumentParser(description="Halogen → llama.cpp timings sidecar")
    p.add_argument("--listen", default="0.0.0.0:8731")
    p.add_argument("--upstream", default="127.0.0.1:18731")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="halogen-sidecar: %(message)s",
    )

    lh, lp = _hostport(args.listen, default_port=8731)
    uh, up = _hostport(args.upstream, default_port=18731)
    sidecar = Sidecar(uh, up)
    asyncio.create_task(sidecar.watch_upstream())

    server = await asyncio.start_server(sidecar.handle, lh, int(lp))
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets or [])
    log.info("listening on %s → %s:%s", addrs, uh, up)
    log.info(
        "503s until Halogen's API binds are expected "
        "(engine cold-load reads tens of GiB before serve_api.py starts)"
    )
    async with server:
        await server.serve_forever()


def _hostport(spec: str, default_port: int) -> tuple[str, int]:
    if spec.startswith("http://") or spec.startswith("https://"):
        u = urlsplit(spec)
        return u.hostname or "127.0.0.1", u.port or default_port
    if spec.startswith("["):
        host, _, rest = spec[1:].partition("]")
        port = int(rest[1:]) if rest.startswith(":") and rest[1:] else default_port
        return host, port
    if spec.count(":") == 1:
        h, p = spec.split(":")
        return h or "0.0.0.0", int(p)
    return spec, default_port


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
