#!/usr/bin/env bash
# LeRobot v0.6.1: Linux GPU server + SSH tunnel + local SO-101 client.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage: bash examples/remote_inference_ssh/run.sh [--dry-run] COMMAND

  setup-server  Install server and policy dependencies on the GPU computer
  setup-client  Install robot client dependencies on the robot computer
  server        Run the policy server on the GPU computer (localhost only)
  tunnel        Forward the server port over SSH from the robot computer
  client        Connect the local robot and start executing model actions

Copy .env.example to .env beside this script and edit it first.
Use LEROBOT_REMOTE_CONFIG to select another trusted Bash configuration file.
--dry-run prints commands without installing, connecting, or moving the robot.
EOF
}

DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then
    DRY_RUN=true
    shift
fi
COMMAND="${1:-help}"
case "$COMMAND" in
    help|-h|--help) usage; exit 0 ;;
    setup-server|setup-client|server|tunnel|client) ;;
    *) usage >&2; exit 1 ;;
esac
[[ $# -eq 1 ]] || { usage >&2; exit 1; }

CONFIG="${LEROBOT_REMOTE_CONFIG:-$SCRIPT_DIR/.env}"
[[ -f "$CONFIG" ]] || { echo "Copy $SCRIPT_DIR/.env.example to $SCRIPT_DIR/.env and edit it first." >&2; exit 1; }
# This is a user-owned Bash file; source only configurations you trust.
source "$CONFIG"
cd "$REPO_ROOT"

require_value() {
    [[ -n "${!1:-}" ]] || { echo "Set $1 in $CONFIG first." >&2; exit 1; }
}

run() {
    printf 'Running:'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" == false ]]; then
        command -v "$1" >/dev/null || { echo "Missing executable: $1. See README.md for prerequisites." >&2; exit 1; }
        "$@"
    fi
}

SERVER_PORT="${SERVER_PORT:-8080}"
LOCAL_PORT="${LOCAL_PORT:-18080}"
FPS="${FPS:-30}"

case "$COMMAND" in
    setup-server)
        require_value POLICY_TYPE
        EXTRAS=(--extra async)
        case "$POLICY_TYPE" in
            act|tdmpc|vqbet) ;;
            smolvla|diffusion|groot) EXTRAS+=(--extra "$POLICY_TYPE") ;;
            pi0|pi05) EXTRAS+=(--extra pi) ;;
            *) echo "Policy not supported by v0.6.1 async inference: $POLICY_TYPE" >&2; exit 1 ;;
        esac
        run uv sync --locked --inexact "${EXTRAS[@]}"
        run uv run --no-sync python -c 'import lerobot.async_inference.policy_server; print("Server imports OK")'
        ;;
    setup-client)
        run uv sync --locked --inexact --extra async --extra feetech
        run uv run --no-sync python -c 'import lerobot.async_inference.robot_client; print("Client imports OK")'
        ;;
    server)
        run uv run --no-sync python -m lerobot.async_inference.policy_server \
            --host=127.0.0.1 "--port=$SERVER_PORT" "--fps=$FPS"
        ;;
    tunnel)
        require_value REMOTE_SSH
        run ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
            -o ServerAliveCountMax=3 -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$SERVER_PORT" \
            -- "$REMOTE_SSH"
        ;;
    client)
        for NAME in POLICY_TYPE MODEL ROBOT_PORT ROBOT_ID CAMERAS ACTIONS_PER_CHUNK TASK; do
            require_value "$NAME"
        done
        run uv run --no-sync python -m lerobot.async_inference.robot_client \
            "--server_address=127.0.0.1:$LOCAL_PORT" \
            --robot.type=so101_follower "--robot.port=$ROBOT_PORT" \
            "--robot.id=$ROBOT_ID" "--robot.cameras=$CAMERAS" \
            "--policy_type=$POLICY_TYPE" "--pretrained_name_or_path=$MODEL" \
            "--policy_device=${POLICY_DEVICE:-cuda}" --client_device=cpu \
            "--task=$TASK" "--fps=$FPS" "--actions_per_chunk=$ACTIONS_PER_CHUNK" \
            "--chunk_size_threshold=${CHUNK_SIZE_THRESHOLD:-0.5}" \
            --debug_visualize_queue_size=False
        ;;
esac
