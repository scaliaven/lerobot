#!/usr/bin/env bash
# Separate environments keep legacy OpenVLA dependencies out of LeRobot's .venv.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: bash examples/remote_inference_libero/run.sh [--dry-run] COMMAND
  setup-server  Install OpenVLA in an isolated Python 3.10 environment (Linux GPU)
  server        Run real OpenVLA inference on the GPU (4-bit by default)
  setup-client  Install an isolated CPU simulation environment (native Linux only)
  client        Run LIBERO with remote inference (native Linux only)
  build-client  Build a CPU Linux simulation container (macOS or Linux)
  docker-client Run the simulation container; publish preview on localhost
  tunnel        Open SSH tunnel from laptop to GPU server
  health        Show server metadata without starting a simulator

Copy .env.example to .env beside this script, then set REMOTE_SSH.
--dry-run prints commands without installing dependencies or starting processes.
EOF
}

DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=true; shift; fi
COMMAND="${1:-help}"
case "$COMMAND" in
    help|-h|--help) usage; exit 0 ;;
    setup-server|server|setup-client|client|build-client|docker-client|tunnel|health) ;;
    *) usage >&2; exit 1 ;;
esac
[[ $# -eq 1 ]] || { usage >&2; exit 1; }
CONFIG="${LEROBOT_LIBERO_CONFIG:-$SCRIPT_DIR/.env}"
[[ -f "$CONFIG" ]] || { echo "Copy $SCRIPT_DIR/.env.example to $SCRIPT_DIR/.env first." >&2; exit 1; }
source "$CONFIG"
cd "$SCRIPT_DIR"

run() {
    printf 'Running:'; printf ' %q' "$@"; printf '\n'
    if [[ "$DRY_RUN" == false ]]; then "$@"; fi
}

require_linux() {
    if [[ "$(uname -s)" != Linux ]]; then
        echo "Use build-client / docker-client on macOS; native LIBERO requires Linux." >&2
        exit 1
    fi
}

SERVER_PORT="${SERVER_PORT:-8080}"
LOCAL_PORT="${LOCAL_PORT:-18080}"
LIBERO_SUITE="${LIBERO_SUITE:-libero_spatial}"
OPENVLA_MODEL="${OPENVLA_MODEL:-openvla/openvla-7b-finetuned-libero-spatial}"
PREVIEW_PORT="${PREVIEW_PORT:-8090}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$(date +%Y%m%d-%H%M%S)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-lerobot-libero-client}"
SERVER_PYTHON="$SCRIPT_DIR/.venv-openvla/bin/python"
CLIENT_PYTHON="$SCRIPT_DIR/.venv-sim/bin/python"
CLIENT_ARGS=(--suite "$LIBERO_SUITE" --task-id "${TASK_ID:-0}" --episodes "${EPISODES:-1}"
    --max-steps "${MAX_STEPS:-220}" --timeout "${REQUEST_TIMEOUT:-120}" --preview-port "$PREVIEW_PORT")

case "$COMMAND" in
    setup-server)
        require_linux
        if [[ ! -x "$SERVER_PYTHON" ]]; then run uv venv --python 3.10 "$SCRIPT_DIR/.venv-openvla"; fi
        run uv pip install --python "$SERVER_PYTHON" -r requirements-server.txt
        ;;
    server)
        require_linux
        run uv run --no-project --no-sync --python "$SERVER_PYTHON" python openvla_server.py \
            --port "$SERVER_PORT" --model "$OPENVLA_MODEL" --suite "$LIBERO_SUITE" \
            --quantization "${QUANTIZATION:-4bit}"
        ;;
    setup-client)
        require_linux
        if [[ ! -x "$CLIENT_PYTHON" ]]; then run uv venv --python 3.10 "$SCRIPT_DIR/.venv-sim"; fi
        run uv pip install --python "$CLIENT_PYTHON" --index-url https://download.pytorch.org/whl/cpu \
            torch==2.2.2 torchvision==0.17.2
        run uv pip install --python "$CLIENT_PYTHON" -r requirements-client.txt
        ;;
    client)
        require_linux
        run uv run --no-project --no-sync --python "$CLIENT_PYTHON" python libero_client.py \
            --endpoint "http://127.0.0.1:$LOCAL_PORT" --output-dir "$OUTPUT_DIR" "${CLIENT_ARGS[@]}"
        ;;
    build-client)
        # Pin the container architecture because LIBERO's renderer dependencies target Linux x86_64.
        run docker build --platform linux/amd64 -t "$DOCKER_IMAGE" "$SCRIPT_DIR"
        ;;
    docker-client)
        if [[ "$DRY_RUN" == false ]]; then mkdir -p -- "$OUTPUT_DIR"; fi
        if [[ "$OUTPUT_DIR" != /* ]]; then OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"; fi
        # Docker Desktop >=4.34: enable host networking in Settings > Resources > Network.
        # This reaches the host-loopback SSH tunnel without exposing it on the LAN.
        run docker run --rm --platform linux/amd64 --network host \
            -v "$OUTPUT_DIR:/outputs" -v lerobot-libero-cache:/root/.cache \
            "$DOCKER_IMAGE" --endpoint "http://127.0.0.1:$LOCAL_PORT" \
            --preview-host 127.0.0.1 --output-dir /outputs "${CLIENT_ARGS[@]}"
        ;;
    tunnel)
        [[ -n "${REMOTE_SSH:-}" ]] || { echo "Set REMOTE_SSH in $CONFIG first." >&2; exit 1; }
        run ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
            -o ServerAliveCountMax=3 -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$SERVER_PORT" -- "$REMOTE_SSH"
        ;;
    health)
        run curl --fail --show-error --max-time 10 "http://127.0.0.1:$LOCAL_PORT/health"
        ;;
esac
