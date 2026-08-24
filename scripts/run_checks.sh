#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
uv_executable="$(command -v uv || true)"
if [[ -z "$uv_executable" || ! -d "$project_root/.venv" ]]; then
    bash "$project_root/scripts/setup_ubuntu.sh"
    uv_executable="$(command -v uv || true)"
    if [[ -z "$uv_executable" ]]; then
        uv_executable="$HOME/.local/bin/uv"
    fi
fi

cd -- "$project_root"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

"$uv_executable" sync --locked --group dev
"$uv_executable" run --locked --no-sync python -c \
    "from PyQt6.QtWidgets import QApplication; app = QApplication([]); assert app is not None; app.quit(); print('PyQt6 offscreen OK')"
"$uv_executable" run --locked --no-sync pytest
"$uv_executable" run --locked --no-sync robot-camera-teleop \
    --source synthetic \
    --headless \
    --free-base \
    --max-frames 30

mujoco_smoke=("$uv_executable" run --locked --no-sync python -c \
    "from robot_human_interface.simulation import HumanoidSimulation; sim = HumanoidSimulation('fixed'); state = sim.step(2); assert state.is_finite; sim.close(); print('MuJoCo smoke OK')")
if command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a "${mujoco_smoke[@]}"
else
    "${mujoco_smoke[@]}"
fi

echo "All checks passed."
