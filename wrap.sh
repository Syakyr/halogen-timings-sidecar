#!/bin/bash
# Drop-in replacement for the Halogen image entrypoint.
#
# `engine` / `bench` / `sweep` pass straight through.
# `all` / `api` (and the image default) move Halogen's OpenAI port to
# 127.0.0.1:18731 and publish the timings sidecar on HALOGEN_API_PORT (8731).
set -euo pipefail

SIDECAR_DIR="$(cd "$(dirname "$0")" && pwd)"
CMD="${1:-all}"

find_upstream_entrypoint() {
  if [ -n "${HALOGEN_ORIGINAL_ENTRYPOINT:-}" ]; then
    printf '%s\n' "$HALOGEN_ORIGINAL_ENTRYPOINT"
    return
  fi
  local p
  for p in \
    /entrypoint.sh \
    /deploy/entrypoint.sh \
    /halogen/deploy/entrypoint.sh \
    /halogen/entrypoint.sh \
    /opt/halogen/entrypoint.sh \
    /usr/local/bin/entrypoint.sh
  do
    if [ -f "$p" ]; then
      printf '%s\n' "$p"
      return
    fi
  done
  return 1
}

# Standalone mode: this container is only the proxy (compose third service).
if [ -n "${HALOGEN_SIDECAR_UPSTREAM:-}" ]; then
  LISTEN="${HALOGEN_SIDECAR_LISTEN:-0.0.0.0:${HALOGEN_SIDECAR_PORT:-8731}}"
  exec python3 "$SIDECAR_DIR/proxy.py" \
    --listen "$LISTEN" \
    --upstream "$HALOGEN_SIDECAR_UPSTREAM"
fi

if [ "$CMD" = "engine" ] || [ "$CMD" = "bench" ] || [ "$CMD" = "sweep" ]; then
  EP="$(find_upstream_entrypoint)" || {
    echo "halogen-sidecar: cannot find original entrypoint; set HALOGEN_ORIGINAL_ENTRYPOINT" >&2
    exit 2
  }
  exec bash "$EP" "$@"
fi

EP="$(find_upstream_entrypoint)" || {
  echo "halogen-sidecar: cannot find original entrypoint; set HALOGEN_ORIGINAL_ENTRYPOINT" >&2
  exit 2
}

# Public port stays 8731 so existing -p 8731:8731 and llama-swap configs
# keep working. Halogen's own API moves to the loopback port below.
PUBLIC_PORT="${HALOGEN_SIDECAR_PORT:-${HALOGEN_API_PORT:-8731}}"
INTERNAL_PORT="${HALOGEN_API_INTERNAL_PORT:-18731}"
export HALOGEN_API_PORT="$INTERNAL_PORT"

echo "halogen-sidecar: Halogen API on :$INTERNAL_PORT, sidecar on :$PUBLIC_PORT"
echo "halogen-sidecar: llama-swap will see 503 until the engine finishes cold-load and serve_api.py binds"

UPSTREAM_LOG="${HALOGEN_UPSTREAM_LOG:-/tmp/halogen-upstream.log}"
: > "$UPSTREAM_LOG"
export HALOGEN_UPSTREAM_LOG="$UPSTREAM_LOG"

# serve_api.py prints the real prompt/cached/prefill/decode line to stdout.
# Tee it so the sidecar can parse those numbers instead of estimating.
bash "$EP" "$@" > >(tee -a "$UPSTREAM_LOG") 2>&1 &
UP_PID=$!

python3 "$SIDECAR_DIR/proxy.py" \
  --listen "0.0.0.0:${PUBLIC_PORT}" \
  --upstream "127.0.0.1:${INTERNAL_PORT}" &
SIDE_PID=$!

term() {
  kill -TERM "$UP_PID" "$SIDE_PID" 2>/dev/null || true
}
trap term TERM INT

# Exit if either dies (engine OOM, API crash, sidecar crash).
while kill -0 "$UP_PID" 2>/dev/null && kill -0 "$SIDE_PID" 2>/dev/null; do
  sleep 1
done

echo "halogen-sidecar: a component exited; shutting down" >&2
term
wait "$UP_PID" 2>/dev/null || true
wait "$SIDE_PID" 2>/dev/null || true
exit 1
