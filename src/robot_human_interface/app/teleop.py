"""End-to-end camera/synthetic pose teleoperation for the MuJoCo humanoid."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WINDOW_NAME = "Robot human interface - camera skeleton"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TeleopStats:
    """Small immutable run summary used by smoke tests and experiment logs."""

    source: str
    base_mode: str
    frames: int
    skeleton_frames: int
    stale_commands: int
    command_span_rad: float
    final_base_height_m: float


def _load_camera_yaml_defaults() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load editable camera/pose defaults while keeping CLI overrides explicit."""

    import yaml

    path = PROJECT_ROOT / "config" / "camera.yaml"
    if not path.is_file():
        return {}, {}
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, dict):
        raise ValueError(f"camera configuration root must be a mapping: {path}")
    camera = document.get("camera", {}) or {}
    pose = document.get("pose", {}) or {}
    if not isinstance(camera, dict) or not isinstance(pose, dict):
        raise ValueError(f"camera and pose sections must be mappings: {path}")
    return camera, pose


def build_parser() -> argparse.ArgumentParser:
    camera_defaults, pose_defaults = _load_camera_yaml_defaults()
    model_default = Path(
        pose_defaults.get(
            "model_asset_path",
            PROJECT_ROOT / "assets" / "models" / "pose_landmarker_full.task",
        )
    )
    if not model_default.is_absolute():
        model_default = PROJECT_ROOT / model_default
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("camera", "synthetic", "replay"),
        default="camera",
        help="Frame/pose source. Synthetic needs no camera or MediaPipe inference.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open the MuJoCo viewer or OpenCV skeleton window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames; zero means run until source/window exit.",
    )
    base_group = parser.add_mutually_exclusive_group()
    base_group.add_argument(
        "--free-base",
        action="store_true",
        help="Allow the unbalanced proxy to fall (experimental).",
    )
    base_group.add_argument(
        "--fixed-base",
        action="store_false",
        dest="free_base",
        help="Weld the torso to world (default).",
    )
    parser.set_defaults(free_base=False)
    parser.add_argument("--camera-index", type=int, default=int(camera_defaults.get("index", 0)))
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "dshow", "msmf", "v4l2", "avfoundation", "gstreamer"),
        default=str(camera_defaults.get("backend", "auto")),
    )
    parser.add_argument("--camera-width", type=int, default=int(camera_defaults.get("width", 1280)))
    parser.add_argument("--camera-height", type=int, default=int(camera_defaults.get("height", 720)))
    parser.add_argument("--camera-fps", type=float, default=float(camera_defaults.get("fps", 30.0)))
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument(
        "--mirror-input",
        action="store_true",
        dest="mirror_input",
        help="Mirror pixels before pose inference; anatomical labels are restored afterward.",
    )
    mirror_group.add_argument(
        "--no-mirror-input",
        action="store_false",
        dest="mirror_input",
    )
    parser.set_defaults(mirror_input=bool(camera_defaults.get("mirror", False)))
    parser.add_argument("--replay-path", type=Path)
    parser.add_argument("--loop-replay", action="store_true")
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=model_default,
    )
    parser.add_argument(
        "--min-pose-detection-confidence",
        type=float,
        default=float(pose_defaults.get("min_pose_detection_confidence", 0.5)),
    )
    parser.add_argument(
        "--min-pose-presence-confidence",
        type=float,
        default=float(pose_defaults.get("min_pose_presence_confidence", 0.5)),
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=float(pose_defaults.get("min_tracking_confidence", 0.5)),
    )
    parser.add_argument(
        "--physics-steps-per-frame",
        type=int,
        default=16,
        help="MuJoCo steps per input frame; 16 x 2 ms is approximately 30 Hz.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.physics_steps_per_frame <= 0:
        raise ValueError("--physics-steps-per-frame must be positive")
    if args.camera_index < 0:
        raise ValueError("--camera-index must be non-negative")
    if args.camera_width <= 0 or args.camera_height <= 0 or args.camera_fps <= 0.0:
        raise ValueError("camera width, height, and fps must be positive")
    for name in (
        "min_pose_detection_confidence",
        "min_pose_presence_confidence",
        "min_tracking_confidence",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be within [0, 1]")
    if args.source == "replay" and args.replay_path is None:
        raise ValueError("--replay-path is required when --source replay is selected")


def _make_perception(args: argparse.Namespace) -> tuple[Any, Any]:
    """Build a CameraSource and PoseEstimator without importing GUI code."""

    if args.source == "synthetic":
        from robot_human_interface.camera import SyntheticCameraConfig, SyntheticCameraSource
        from robot_human_interface.pose import SyntheticPoseEstimator

        maximum = args.max_frames if args.max_frames > 0 else None
        source = SyntheticCameraSource(
            SyntheticCameraConfig(
                width=min(args.camera_width, 960),
                height=min(args.camera_height, 720),
                fps=args.camera_fps,
                max_frames=maximum,
                realtime=not args.headless,
            )
        )
        return source, SyntheticPoseEstimator()

    from robot_human_interface.camera import (
        OpenCVCameraConfig,
        OpenCVCameraSource,
        OpenCVVideoSource,
    )
    from robot_human_interface.pose import MediaPipePoseConfig, MediaPipePoseLandmarker
    from robot_human_interface.skeleton import SkeletonEMAFilter, SkeletonFilterConfig

    if args.source == "camera":
        source = OpenCVCameraSource(
            OpenCVCameraConfig(
                index=args.camera_index,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                mirror=args.mirror_input,
                backend=args.camera_backend,
            )
        )
    else:
        assert args.replay_path is not None
        source = OpenCVVideoSource(
            args.replay_path.expanduser().resolve(),
            mirror=args.mirror_input,
            loop=args.loop_replay,
        )

    pose = MediaPipePoseLandmarker(
        MediaPipePoseConfig(
            model_asset_path=args.pose_model.expanduser().resolve(),
            min_pose_detection_confidence=args.min_pose_detection_confidence,
            min_pose_presence_confidence=args.min_pose_presence_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        ),
        landmark_filter=SkeletonEMAFilter(
            SkeletonFilterConfig(
                time_constant_s=0.08,
                confidence_threshold=0.5,
                max_gap_s=0.25,
            )
        ),
    )
    return source, pose


def _draw_status(
    image: np.ndarray,
    *,
    paused: bool,
    stale: bool,
    calibrating: bool,
    calibration_progress: float,
    base_mode: str,
) -> np.ndarray:
    import cv2

    status = "PAUSED" if paused else ("POSE LOST / SAFE RETURN" if stale else "TRACKING")
    color = (60, 220, 255) if paused else ((70, 70, 255) if stale else (70, 230, 90))
    cv2.putText(image, status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        image,
        f"base={base_mode} | C calibrate | R reset | Space pause | Esc exit",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
    )
    if calibrating:
        cv2.putText(
            image,
            f"calibration {calibration_progress * 100:3.0f}% - hold neutral pose",
            (16, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 220, 255),
            2,
        )
    return image


def _handle_key(
    key: int,
    *,
    simulation: Any,
    retargeter: Any,
    paused: bool,
) -> tuple[bool, bool]:
    """Return ``(keep_running, paused)`` after one OpenCV key code."""

    if key in (27,):
        return False, paused
    if key in (ord(" "),):
        paused = not paused
        LOGGER.info("teleoperation %s", "paused" if paused else "resumed")
    elif key in (ord("c"), ord("C")):
        retargeter.start_calibration(15)
        LOGGER.info("neutral-pose calibration started (15 confident frames)")
    elif key in (ord("r"), ord("R")):
        retargeter.reset()
        simulation.reset()
        LOGGER.info("simulation and retargeter reset")
    return True, paused


def run_teleop(args: argparse.Namespace) -> TeleopStats:
    """Run one complete perception/retarget/simulation loop."""

    _validate_args(args)
    from robot_human_interface.pose import draw_pose_overlay
    from robot_human_interface.retargeting import GeometricRetargeter
    from robot_human_interface.simulation import HumanoidSimulation

    source, pose = _make_perception(args)
    retargeter = GeometricRetargeter.from_yaml(
        joints_path=PROJECT_ROOT / "config" / "joints.yaml",
        retargeting_path=PROJECT_ROOT / "config" / "retargeting.yaml",
    )
    base_mode = "free" if args.free_base else "fixed"
    simulation = HumanoidSimulation(base_mode)
    display_enabled = not args.headless
    viewer_started = False
    paused = False
    frames = 0
    skeleton_frames = 0
    stale_commands = 0
    command_min: np.ndarray | None = None
    command_max: np.ndarray | None = None
    final_base_height = float(simulation.get_state().base_position_m[2])

    try:
        if display_enabled:
            simulation.launch_viewer()
            viewer_started = True

        while args.max_frames <= 0 or frames < args.max_frames:
            frame = source.read()
            if frame is None:
                break
            skeleton = pose.estimate(frame)
            if skeleton is not None:
                skeleton_frames += 1

            command = retargeter.retarget(skeleton, timestamp_s=frame.timestamp_s)
            stale_commands += int(command.stale)
            if command_min is None:
                command_min = command.positions_rad.copy()
                command_max = command.positions_rad.copy()
            else:
                command_min = np.minimum(command_min, command.positions_rad)
                command_max = np.maximum(command_max, command.positions_rad)

            if not paused:
                simulation.apply_joint_command(command)
                state = simulation.step(args.physics_steps_per_frame)
                final_base_height = float(state.base_position_m[2])
            else:
                simulation.sync_viewer()
            frames += 1

            if display_enabled:
                import cv2

                overlay = draw_pose_overlay(frame.image_bgr, skeleton, confidence_threshold=0.5)
                _draw_status(
                    overlay,
                    paused=paused,
                    stale=command.stale,
                    calibrating=retargeter.is_calibrating,
                    calibration_progress=retargeter.calibration_progress,
                    base_mode=base_mode,
                )
                cv2.imshow(WINDOW_NAME, overlay)
                keep_running, paused = _handle_key(
                    cv2.waitKey(1) & 0xFF,
                    simulation=simulation,
                    retargeter=retargeter,
                    paused=paused,
                )
                if not keep_running:
                    break
                if viewer_started and not simulation.viewer_is_running:
                    LOGGER.info("MuJoCo viewer was closed")
                    break
    finally:
        if display_enabled:
            try:
                import cv2

                cv2.destroyWindow(WINDOW_NAME)
                cv2.waitKey(1)
            except Exception:
                LOGGER.debug("OpenCV window was already closed", exc_info=True)
        pose.close()
        source.close()
        simulation.close()

    command_span = 0.0
    if command_min is not None and command_max is not None:
        command_span = float(np.max(command_max - command_min))
    return TeleopStats(
        source=args.source,
        base_mode=base_mode,
        frames=frames,
        skeleton_frames=skeleton_frames,
        stale_commands=stale_commands,
        command_span_rad=command_span,
        final_base_height_m=final_base_height,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.free_base:
        LOGGER.warning("FREE-BASE mode has no balance controller; the robot may fall.")
    try:
        stats = run_teleop(args)
    except KeyboardInterrupt:
        LOGGER.info("interrupted by user")
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2

    print(
        "teleop complete: "
        f"source={stats.source} frames={stats.frames} "
        f"skeleton_frames={stats.skeleton_frames} stale={stats.stale_commands} "
        f"base={stats.base_mode} command_span_deg={np.degrees(stats.command_span_rad):.3f} "
        f"base_z_m={stats.final_base_height_m:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
