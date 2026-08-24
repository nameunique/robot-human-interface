#!/usr/bin/env bash
set -euo pipefail

install_system_deps=1
install_dev=1
for argument in "$@"; do
    case "$argument" in
        --skip-system-deps) install_system_deps=0 ;;
        --no-dev) install_dev=0 ;;
        *)
            echo "Unknown argument: $argument" >&2
            echo "Usage: $0 [--skip-system-deps] [--no-dev]" >&2
            exit 2
            ;;
    esac
done

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
lock_file="$project_root/uv.lock"

if [[ ! -f "$lock_file" ]]; then
    echo "The universal dependency lock is missing: $lock_file" >&2
    echo "Run 'uv lock' on a maintainer machine and commit uv.lock before setup." >&2
    exit 1
fi

if (( install_system_deps )); then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "This setup script supports Ubuntu 24.04 (apt-get was not found)." >&2
        exit 1
    fi
    sudo_command=()
    if (( EUID != 0 )); then
        if ! command -v sudo >/dev/null 2>&1; then
            echo "sudo is required to install Ubuntu system packages." >&2
            exit 1
        fi
        sudo_command=(sudo)
    fi
    "${sudo_command[@]}" apt-get update
    "${sudo_command[@]}" apt-get install -y \
        avahi-daemon \
        ca-certificates \
        curl \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libgl1 \
        libgl1-mesa-dri \
        libglib2.0-0t64 \
        libglfw3 \
        libnss-mdns \
        libportaudio2 \
        libx11-xcb1 \
        libxcb1 \
        libxcb-cursor0 \
        libxcb-glx0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-shm0 \
        libxcb-sync1 \
        libxcb-util1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxi6 \
        libxkbcommon-x11-0 \
        libxrender1 \
        python3.12-venv \
        v4l-utils \
        xauth \
        xvfb
fi

uv_executable="$(command -v uv || true)"
if [[ -z "$uv_executable" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to install uv." >&2
        exit 1
    fi
    uv_installer="$(mktemp)"
    cleanup_installer() {
        rm -f -- "$uv_installer"
    }
    trap cleanup_installer EXIT
    curl --proto '=https' --tlsv1.2 -LsSf \
        https://astral.sh/uv/0.12.5/install.sh \
        -o "$uv_installer"
    sh "$uv_installer"
    uv_executable="$HOME/.local/bin/uv"
    if [[ ! -x "$uv_executable" ]]; then
        echo "uv installation completed but $uv_executable was not found." >&2
        exit 1
    fi
fi

minimum_uv_version="0.12.5"
uv_version_output="$("$uv_executable" --version 2>&1 || true)"
if [[ ! "$uv_version_output" =~ ^uv[[:space:]]+([0-9]+\.[0-9]+\.[0-9]+) ]]; then
    echo "Could not determine the installed uv version from: $uv_version_output" >&2
    exit 1
fi
uv_version="${BASH_REMATCH[1]}"
version_at_least() {
    local actual="$1"
    local minimum="$2"
    local IFS=.
    local -a actual_parts minimum_parts
    local index actual_part minimum_part
    read -r -a actual_parts <<< "$actual"
    read -r -a minimum_parts <<< "$minimum"
    for index in 0 1 2; do
        actual_part=$((10#${actual_parts[$index]}))
        minimum_part=$((10#${minimum_parts[$index]}))
        if (( actual_part > minimum_part )); then
            return 0
        fi
        if (( actual_part < minimum_part )); then
            return 1
        fi
    done
    return 0
}
if ! version_at_least "$uv_version" "$minimum_uv_version"; then
    echo "uv $uv_version is too old; this project requires uv $minimum_uv_version or newer." >&2
    echo "Remove or upgrade the old uv executable, then rerun this script." >&2
    exit 1
fi

sync_arguments=(sync --locked --python 3.12)
if (( install_dev )); then
    sync_arguments+=(--group dev)
else
    sync_arguments+=(--no-dev)
fi

cd -- "$project_root"
"$uv_executable" "${sync_arguments[@]}"
"$uv_executable" run --locked --no-sync python -c \
    "import PyQt6, mediapipe, mujoco, websocket, robot_human_interface; assert callable(websocket.create_connection); print(f'Environment ready: MuJoCo {mujoco.__version__}, MediaPipe {mediapipe.__version__}')"

echo "Ubuntu setup complete."
echo "Run the operator GUI: bash \"$project_root/scripts/run_gui.sh\""
echo "Run the legacy CLI: bash \"$project_root/scripts/run_camera_teleop.sh\" --source synthetic"
