<div align="center">

# ⚡ Halogen Timings Sidecar

### Real prefill / decode tok/s for Halogen — inside llama-swap.

A drop-in front-end for [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server)
that forges a llama.cpp-shaped `timings` object onto the final SSE chunk,
so [llama-swap](https://github.com/mostlygeek/llama-swap) finally knows
what your GPU has been doing.

[![build](https://github.com/syakyr/halogen-timings-sidecar/actions/workflows/build.yml/badge.svg)](https://github.com/syakyr/halogen-timings-sidecar/actions/workflows/build.yml)
[![watch-halogen](https://github.com/syakyr/halogen-timings-sidecar/actions/workflows/watch-halogen.yml/badge.svg)](https://github.com/syakyr/halogen-timings-sidecar/actions/workflows/watch-halogen.yml)
![latest build](https://img.shields.io/github/v/tag/Syakyr/halogen-timings-sidecar?label=latest%20build&color=brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![uv](https://img.shields.io/badge/uv-managed-blueviolet)
![tests](https://img.shields.io/badge/tests-55%20passing-brightgreen)
![coverage](https://img.shields.io/badge/coverage-proxy.py%2078%25-green)

**Pull · Run · Point llama-swap at `:8731` — done.**

</div>

---

## The problem, in one chunk

Halogen's final SSE chunk today carries no metrics:

```json
{"choices":[{"finish_reason":"stop","delta":{}}]}
```

After the sidecar, it carries exactly what llama.cpp would have sent
(llama-swap v141+ reads `timings.predicted_per_second`):

```json
{"choices":[{"finish_reason":"stop","delta":{}}],
 "usage":{"prompt_tokens":502,"completion_tokens":3220,"total_tokens":3722},
 "timings":{
   "cache_n":0,
   "prompt_n":502,"prompt_ms":2845.2,"prompt_per_second":176.4,
   "predicted_n":3220,"predicted_ms":137786.7,"predicted_per_second":23.36
 }}
```

Every other byte of the stream passes through untouched.

## Quick start

```bash
podman run --rm -p 8731:8731 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1:-1 \
  -e HALOGEN_DOWNLOAD=peonist-ai/halogen-qwen3.8-flash-next \
  -v ~/halogen-models:/models \
  ghcr.io/syakyr/halogen-timings-sidecar:latest
```

Point llama-swap at `http://127.0.0.1:8731`. Same devices, same env,
same port as the official image — the only change is the image name.

> **Set a long `healthCheckTimeout` (10–20 min).** The sidecar binds `:8731`
> instantly, but Halogen's API only appears on loopback `:18731` after the
> engine finishes reading weights. During that window the sidecar answers
> `503` with `Retry-After: 5` — that wait is expected, not a broken proxy.
>
> `engine` / `bench` / `sweep` modes bypass the sidecar entirely.

Verify it's working:

```bash
curl -Ns http://127.0.0.1:8731/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"halogen-qwen3.8-flash-next","stream":true,"max_tokens":32,
       "messages":[{"role":"user","content":"Say hi."}]}' \
  | tail -n 4
```

The `finish_reason` object should carry `timings.predicted_per_second`,
and the stream should still end with `data: [DONE]`.

## Images & tags

Prebuilt images: `ghcr.io/syakyr/halogen-timings-sidecar`. A GitHub Actions
watcher checks upstream every 6 hours and builds automatically on each new
Halogen release — no manual build needed, ever.

The watcher also keeps this repo's own default base pins (`Dockerfile` ARG
and `compose.yml` defaults) in sync with the newest upstream release via a
`chore: pin default base` commit, so bare local builds never drift onto an
old base. Manual runs pinned to an older `halogen_version` skip the sync —
defaults track newest-only.

<details>
<summary>Don't want to wait up to 6 hours for a fresh upstream tag?</summary>

Run **watch-halogen** manually (`Actions → watch-halogen → Run workflow`) and
leave `halogen_version` blank to build the newest upstream release now, or pin
an exact one. The same workflow also runs on schedule.

```bash
# same thing from the CLI
gh workflow run watch-halogen.yml -f halogen_version=0.11.3 -f sidecar_number=1
```

Note that the watcher triggers `build.yml` with an explicit `workflow_dispatch`
rather than relying on its own tag push: a tag pushed with `GITHUB_TOKEN` does
not start other workflows (only `workflow_dispatch`/`repository_dispatch`
cross that boundary), so the tag is kept purely as provenance.

</details>

| Tag | Meaning |
|---|---|
| `latest` | newest Halogen version with a built sidecar |
| `<halogen>` | latest sidecar build for that Halogen version (moves on rebuild) |
| `<halogen>-sidecar<N>` | immutable build N — pin this for reproducibility |

<details>
<summary>List available tags without pulling</summary>

```bash
curl -s "https://ghcr.io/token?scope=repository:syakyr/halogen-timings-sidecar:pull" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])' \
  | xargs -I{} curl -s -H "Authorization: Bearer {}" \
      https://ghcr.io/v2/syakyr/halogen-timings-sidecar/tags/list \
  | python3 -m json.tool
```

</details>

## What is measured

| Field | Source |
|---|---|
| `prompt_ms` | request start → first non-empty `delta.content` / `reasoning_content` (TTFT) |
| `predicted_ms` | first content → `finish_reason` chunk |
| `prompt_n` / `predicted_n` | tee'd `serve_api` log line if present, else upstream `usage`, else a cheap CJK/latin estimate |
| `cache_n` | `serve_api` log, else longest previously seen message prefix, else "too fast to be prefill" vs `HALOGEN_PREFILL_CEILING` (default 1800 tok/s) |

This is HTTP-visible wall time, not Halogen's internal GEMM clocks.
Prefill tok/s sits a few percent under `sweep`; decode tok/s is close.
`cache_n` is inferred — Halogen does not send it. Prefix memory needs the
sidecar process to have seen the earlier turn (same container); the rate
ceiling still fires on a cache hit even if the prefix table missed
(restart, rewritten history). If a cold prefill is being marked cached,
override with `-e HALOGEN_PREFILL_CEILING=1500`.

Non-stream responses pass through unchanged — inventing a single-bucket
rate would silently attribute prefill to decode.

## Two-container topology (optional)

Keeps the 115 GiB engine process untouched when you iterate on the proxy:

```bash
HALOGEN_MODELS=~/halogen-models docker compose up --build
```

The `sidecar` service runs the same wrapped image in standalone-proxy mode
(`HALOGEN_SIDECAR_UPSTREAM=api:8731`); llama-swap targets `:8731` as always.

## Build locally

```bash
podman build -t halogen-flash-timed:local \
  --build-arg HALOGEN_IMAGE=ghcr.io/peonist-ai/halogen-flash-server:0.11.4 .
```

Then `podman run` it exactly as in the Quick start, swapping the image name.

## Development (uv-first)

All dev tooling (pytest, pytest-cov, ruff) is pinned in `pyproject.toml` /
`uv.lock`; the proxy itself is stdlib-only.

```bash
uv sync                                            # create .venv with dev tools
uv run ruff check .                                # lint
uv run ruff format .                               # format
uv run pytest --cov=proxy --cov-branch --cov-fail-under=75
```

No GPU required: `tests/test_unit.py` covers the pure timing/estimation
functions, and `tests/test_smoke.py` runs the Sidecar against a mock
Halogen SSE upstream (forged timings, engine-log priority, non-stream
passthrough, 503-during-cold-load, CLI boot).

### Releasing a sidecar change

```bash
git tag <halogen>-sidecar<N> && git push origin <halogen>-sidecar<N>
# e.g. git tag 0.11.2-sidecar2 && git push origin 0.11.2-sidecar2
# → build.yml runs lint/tests, then publishes
#   :<halogen>-sidecarN (immutable) and moves :<halogen> to it
```

`workflow_dispatch` on `build.yml` does the same interactively. The
`watch-halogen` workflow only ever creates `sidecar1` tags automatically.
