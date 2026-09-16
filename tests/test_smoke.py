"""End-to-end smoke tests: the Sidecar in front of a mock Halogen SSE upstream.

No GPU, no Halogen. A stub server emits exactly what halogen-flash-server
emits today (content deltas, a bare finish_reason stop chunk, [DONE]),
and we assert the sidecar forges llama.cpp-shaped timings/usage on the
stop chunk and keeps the stream otherwise intact.

The proxy runs in-process (via Sidecar.handle) so coverage sees the SSE
plumbing; one subprocess test covers the CLI entrypoint.
"""

import asyncio
import json
import os
import socket
import sys

# Isolate from any real wrap.sh log on this machine before proxy imports.
DEAD_LOG = "/tmp/halogen-sidecar-test-does-not-exist.log"
os.environ["HALOGEN_UPSTREAM_LOG"] = DEAD_LOG

import proxy as proxy_mod  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROXY = os.path.join(ROOT, "proxy.py")

MOCK_EVENTS = [
    {"choices": [{"delta": {"role": "assistant"}}]},
    {"choices": [{"delta": {"content": "Hello "}}]},
    {"choices": [{"delta": {"content": "world!"}}]},
    {"choices": [{"finish_reason": "stop", "delta": {}}]},
]

# Same line Halogen's serve_api prints; drives the engine-log timing path.
SERVE_API_MTP = (
    "serve_api: mtp 250 tok in 6.75s = 37.05 t/s "
    "| 153 rounds, commit 1.63/round "
    "| prompt 1287 (1035 cached), prefill 1.26s"
)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def mock_upstream(reader, writer):
    headers = {}
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode("latin1").partition(":")
        headers[key.strip().lower()] = value.strip()
    body = b""
    if headers.get("content-length"):
        body = await reader.readexactly(int(headers["content-length"]))
    try:
        req = json.loads(body)
    except Exception:
        req = {}

    if not req.get("stream"):
        resp = json.dumps(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ]
            }
        ).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: %d\r\n\r\n" % len(resp) + resp
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        return

    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Cache-Control: no-cache\r\n\r\n"
    )
    await writer.drain()
    for event in MOCK_EVENTS:
        writer.write(f"data: {json.dumps(event)}\n\n".encode())
        await writer.drain()
        await asyncio.sleep(0.03)
    writer.write(b"data: [DONE]\n\n")
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def post_chat(port: int, stream: bool = True):
    body = json.dumps(
        {
            "model": "mock-halogen",
            "stream": stream,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Say hi."}],
        }
    ).encode()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    request = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode() + body
    writer.write(request)
    await writer.drain()
    data = await reader.read()
    writer.close()
    await writer.wait_closed()

    head, _, rest = data.partition(b"\r\n\r\n")
    lines = head.decode("latin1").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        key, _, value = line.partition(":")
        headers[key.strip().lower()] = value.strip()
    payloads = [
        ln[5:].strip()
        for ln in rest.decode("utf-8", errors="replace").split("\n")
        if ln.startswith("data:")
    ]
    return status, headers, rest, payloads


async def start_sidecar_proxy(upstream_port: int):
    sidecar = proxy_mod.Sidecar("127.0.0.1", upstream_port)
    server = await asyncio.start_server(sidecar.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return sidecar, server, port


async def scenario_upstream_down():
    dead_port = free_port()
    _, server, proxy_port = await start_sidecar_proxy(dead_port)
    try:
        status, headers, rest, _ = await post_chat(proxy_port)
        assert status == 503, f"expected 503 during cold-load, got {status}"
        assert headers.get("retry-after") == "5"
        body = json.loads(rest.decode())
        assert body["error"]["code"] == "engine_loading"
    finally:
        server.close()
        await server.wait_closed()


async def scenario_timings_forged():
    mock = await asyncio.start_server(mock_upstream, "127.0.0.1", 0)
    mock_port = mock.sockets[0].getsockname()[1]
    _, server, proxy_port = await start_sidecar_proxy(mock_port)
    try:
        status, headers, _, payloads = await post_chat(proxy_port)
        assert status == 200, f"expected 200, got {status}"
        assert headers.get("x-halogen-sidecar") == "timings-1"
        assert payloads[-1] == "[DONE]", "stream must still end with [DONE]"

        stop = json.loads(payloads[-2])
        assert stop["choices"][0]["finish_reason"] == "stop"

        timings = stop["timings"]
        assert timings["predicted_per_second"] > 0
        assert timings["prompt_per_second"] >= 0
        assert timings["prompt_n"] > 0
        assert timings["predicted_n"] > 0
        assert timings["prompt_ms"] > 0
        assert timings["predicted_ms"] > 0
        assert timings["cache_n"] >= 0

        usage = stop["usage"]
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )
        assert usage["completion_tokens"] == timings["predicted_n"]

        # Earlier chunks must pass through untouched.
        first = json.loads(payloads[0])
        assert "timings" not in first
    finally:
        server.close()
        await server.wait_closed()
        mock.close()
        await mock.wait_closed()


async def scenario_timings_from_engine_log():
    """When the tee'd serve_api log has the request's line, its numbers win."""
    import tempfile

    fd, logpath = tempfile.mkstemp(prefix="halogen-upstream-test-")
    with os.fdopen(fd, "w") as fh:
        fh.write(SERVE_API_MTP + "\n")
    mock = await asyncio.start_server(mock_upstream, "127.0.0.1", 0)
    mock_port = mock.sockets[0].getsockname()[1]
    sidecar, server, proxy_port = await start_sidecar_proxy(mock_port)
    sidecar.api_log = proxy_mod.ServeApiLog(logpath)
    try:
        status, _, _, payloads = await post_chat(proxy_port)
        assert status == 200
        stop = json.loads(payloads[-2])
        timings = stop["timings"]
        assert timings["prompt_n"] == 252
        assert timings["cache_n"] == 1035
        assert timings["prompt_per_second"] == 200.0
        assert timings["predicted_n"] == 250
        assert timings["predicted_per_second"] == 37.05
        assert timings["draft_n"] == 153
        assert timings["draft_n_accepted"] == 97
        assert stop["usage"]["prompt_tokens"] == 1287
    finally:
        server.close()
        await server.wait_closed()
        mock.close()
        await mock.wait_closed()
        os.unlink(logpath)


async def scenario_non_stream_passthrough():
    mock = await asyncio.start_server(mock_upstream, "127.0.0.1", 0)
    mock_port = mock.sockets[0].getsockname()[1]
    _, server, proxy_port = await start_sidecar_proxy(mock_port)
    try:
        status, _, rest, _ = await post_chat(proxy_port, stream=False)
        assert status == 200
        obj = json.loads(rest.decode())
        assert "timings" not in obj, "non-stream bodies must pass through alone"
        assert obj["choices"][0]["message"]["content"] == "hi"
    finally:
        server.close()
        await server.wait_closed()
        mock.close()
        await mock.wait_closed()


async def scenario_cli_subprocess():
    """The proxy.py CLI must boot and 503 while its upstream is dead."""
    proxy_port, dead_port = free_port(), free_port()
    env = dict(os.environ, HALOGEN_UPSTREAM_LOG=DEAD_LOG)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        PROXY,
        "--listen",
        f"127.0.0.1:{proxy_port}",
        "--upstream",
        f"127.0.0.1:{dead_port}",
        "--log-level",
        "WARNING",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 15.0
        while loop.time() < deadline:
            try:
                _, w = await asyncio.open_connection("127.0.0.1", proxy_port)
                w.close()
                await w.wait_closed()
                break
            except OSError:
                await asyncio.sleep(0.1)
        else:
            raise RuntimeError("proxy CLI did not bind")
        status, headers, rest, _ = await post_chat(proxy_port)
        assert status == 503
        assert headers.get("retry-after") == "5"
    finally:
        proc.terminate()
        await proc.wait()


def test_upstream_down_returns_503_with_retry_after():
    asyncio.run(scenario_upstream_down())


def test_stop_chunk_gets_forged_timings():
    asyncio.run(scenario_timings_forged())


def test_engine_log_numbers_take_priority():
    asyncio.run(scenario_timings_from_engine_log())


def test_non_stream_passthrough_untimed():
    asyncio.run(scenario_non_stream_passthrough())


def test_cli_boots_and_serves():
    asyncio.run(scenario_cli_subprocess())
