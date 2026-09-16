# halogen-timings-sidecar

[![build](https://github.com/syakyr/halogen-timings-sidecar/actions/workflows/build.yml/badge.svg)](https://github.com/syakyr/halogen-timings-sidecar/actions/workflows/build.yml)
[![watch-halogen](https://github.com/syakyr/halogen-timings-sidecar/actions/workflows/watch-halogen.yml/badge.svg)](https://github.com/syakyr/halogen-timings-sidecar/actions/workflows/watch-halogen.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)![uv](https://img.shields.io/badge/uv-managed-blueviolet)![tests](https://img.shields.io/badge/tests-55%20passing-brightgreen)![coverage](https://img.shields.io/badge/coverage-proxy.py%2078%25-green)

Drop-in front-end for [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) that forges a llama.cpp `timings` object onto the final SSE chunk so [llama-swap](https://github.com/mostlygeek/llama-swap) can show prefill and decode tok/s.

Halogen’s last chunk today:

```json
{"choices":[{"finish_reason":"stop","delta":{}}]}
```

After the sidecar:

```json
{"choices":[{"finish_reason":"stop","delta":{}}],
 "usage":{"prompt_tokens":502,"completion_tokens":3220,"total_tokens":3722},
 "timings":{
   "cache_n":0,
   "prompt_n":502,"prompt_ms":2845.2,"prompt_per_second":176.4,
   "predicted_n":3220,"predicted_ms":137786.7,"predicted_per_second":23.36
 }}
```

Same shape llama.cpp emits. llama-swap v141+ reads `timings.predicted_per_second`.

## Pull the prebuilt image (GHCR)

Prebuilt images live at `ghcr.io/syakyr/halogen-timings-sidecar`. A GitHub Actions
watcher checks upstream every 6 hours and builds automatically on each new Halogen
release — no manual build needed.

```bash
# Newest sidecar for a given Halogen version (recommended, moves with rebuilds):
docker pull ghcr.io/syakyr/halogen-timings-sidecar:0.11.1

# Immutable, reproducible pin:
docker pull ghcr.io/syakyr/halogen-timings-sidecar:0.11.1-sidecar1

# Newest of everything:
docker pull ghcr.io/syakyr/halogen-timings-sidecar:latest
```

Then run it exactly as you would the official image — same devices, same env,
same published port 8731:

```bash
podman run --rm -p 8731:8731 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1:-1 \
  -e HALOGEN_DOWNLOAD=peonist-ai/halogen-qwen3.8-flash-next \
  -v ~/halogen-models:/models \
  ghcr.io/syakyr/halogen-timings-sidecar:0.11.1
```

### Tag scheme

| Tag | Meaning |
|---|---|
| `<halogen>` | latest sidecar build for that Halogen version (moves on rebuild) |
| `<halogen>-sidecar<N>` | immutable build N for that Halogen version |
| `latest` | newest Halogen version with a built sidecar |

## Build locally (one-container, same `podman run` you already use)

```bash
cd halogen-timings-sidecar
podman build -t halogen-flash-timed:local .

podman run --rm -p 8731:8731 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1:-1 \
  -e HALOGEN_DOWNLOAD=peonist-ai/halogen-qwen3.8-flash-next \
  -v ~/halogen-models:/models \
  halogen-flash-timed:local
```

Point llama-swap at `http://127.0.0.1:8731`. No config change other than the image name.

Give llama-swap a long `healthCheckTimeout` (10–20 minutes). The sidecar binds `:8731` immediately, but Halogen’s API only appears on loopback `:18731` after the engine finishes reading weights. During that window the sidecar answers `503` with `Retry-After: 5`. A flood of `ConnectionRefusedError: 127.0.0.1:18731` on older builds was that wait, not a broken proxy.

`engine` / `bench` / `sweep` still bypass the sidecar (those modes are not what llama-swap talks to).

## Two-container official compose + sidecar

Keeps the 115 GiB engine process untouched when you iterate on the proxy:

```bash
HALOGEN_MODELS=~/halogen-models docker compose up --build
```

## What is measured

| Field | Source |
|---|---|
| `prompt_ms` | request start → first non-empty `delta.content` / `reasoning_content` (TTFT) |
| `predicted_ms` | first content → `finish_reason` chunk |
| `prompt_n` / `predicted_n` | upstream `usage` if present, else a cheap CJK/latin estimate |
| `cache_n` | longest previously seen message prefix, else “too fast to be prefill” vs `HALOGEN_PREFILL_CEILING` (default 1800 tok/s) |

This is HTTP-visible wall time, not Halogen’s internal GEMM clocks. Prefill tok/s will sit a few percent under `sweep`. Decode tok/s is close. `cache_n` is inferred — Halogen does not send it. Prefix memory needs the sidecar process to have seen the earlier turn (same container). The rate ceiling still fires on a cache hit even if the prefix table missed (restart, rewritten history). Override the ceiling with `-e HALOGEN_PREFILL_CEILING=1500` if a cold prefill is being marked cached.

Non-stream responses are passed through unchanged. llama-swap would otherwise attribute the whole wait to decode.

## Check

```bash
curl -Ns http://127.0.0.1:8731/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"halogen-qwen3.8-flash-next","stream":true,"max_tokens":32,
       "messages":[{"role":"user","content":"Say hi."}]}' \
  | tail -n 4
```

The `finish_reason` object should contain `timings.predicted_per_second`. The next line should still be `data: [DONE]`.

## Development (uv-first)

All dev tooling (pytest, pytest-cov, ruff) is pinned in `pyproject.toml` /
`uv.lock`; the proxy itself is stdlib-only.

```bash
uv sync                      # create .venv with dev tools
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest --cov=proxy --cov-branch --cov-fail-under=75
```

Tests need no GPU: `tests/test_unit.py` covers the pure timing/estimation
functions, and `tests/test_smoke.py` runs the Sidecar against a mock Halogen
SSE upstream (forged timings, engine-log priority, non-stream passthrough,
503-during-cold-load, CLI boot).

### Releasing a sidecar change on top of a given Halogen version

```bash
git tag 0.11.1-sidecar2 && git push origin 0.11.1-sidecar2
# → build.yml runs lint/tests, then publishes
#   :0.11.1-sidecar2 (immutable) and moves :0.11.1 to it
```

`workflow_dispatch` on build.yml does the same interactively. The
watch-halogen workflow only ever creates `sidecar1` tags automatically.
