#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
uv_executable="$(command -v uv || true)"
environment_ready=0
if [[ -n "$uv_executable" && -d "$project_root/.venv" && -x "$project_root/.venv/bin/robot-camera-teleop" ]]; then
    if (
        cd -- "$project_root"
        "$uv_executable" run --locked --no-sync python -c \
            "import cv2, mediapipe, mujoco, websocket, robot_human_interface; assert callable(websocket.create_connection)"
    ) >/dev/null 2>&1; then
        environment_ready=1
    fi
fi
if (( ! environment_ready )); then
    bash "$project_root/scripts/setup_ubuntu.sh"
    uv_executable="$(command -v uv || true)"
    if [[ -z "$uv_executable" ]]; then
        uv_executable="$HOME/.local/bin/uv"
    fi
fi

cd -- "$project_root"
exec "$uv_executable" run --locked --no-sync robot-camera-teleop "$@"
