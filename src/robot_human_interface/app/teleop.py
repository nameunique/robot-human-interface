"""End-to-end camera/synthetic pose teleoperation for the MuJoCo humanoid."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WINDOW_NAME = "Robot human interface - camera skeleton"
BUNDLED_VIDEO_PATHS = {
    "jumping-jacks": PROJECT_ROOT / "assets" / "videos" / "jumping_jacks_demo.mp4",
    "slow-balance": PROJECT_ROOT / "assets" / "videos" / "slow_balance_demo.mp4",
}
DEFAULT_BUNDLED_VIDEO = "slow-balance"
# Legacy constant name retained for callers that expect one default demo path.
DEMO_VIDEO_PATH = BUNDLED_VIDEO_PATHS[DEFAULT_BUNDLED_VIDEO]
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
    safe_command_span_rad: float
    final_base_height_m: float
    simulation_time_s: float
    media_time_s: float
    maximum_tilt_rad: float
    fell: bool
    robot_output_enabled: bool
    robot_send_attempts: int
    robot_commands_sent: int
    robot_last_error: str | None
    support_transitions: int
    right_swing_completed: bool
    left_swing_completed: bool
    maximum_right_foot_clearance_m: float
    maximum_left_foot_clearance_m: float


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
        choices=("camera", "mp4", "synthetic", "replay"),
        default="camera",
        help=(
            "Frame/pose source. 'mp4' uses the bundled demo video, 'replay' "
            "uses --video-path, and synthetic needs no camera or MediaPipe inference."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not open the MuJoCo viewer or OpenCV skeleton window.",
    )
    parser.add_argument(
        "--demo-video",
        choices=tuple(BUNDLED_VIDEO_PATHS),
        default=DEFAULT_BUNDLED_VIDEO,
        help="Bundled video used with --source mp4 (default: slow-balance).",
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
        help="Use the grounded vertical stabilizer so the feet carry the robot (default).",
    )
    parser.set_defaults(free_base=False)
    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument(
        "--balance-controller",
        action="store_true",
        dest="balance_controller",
        help="Run the deployable motor-angle balance layer in free-base mode (default).",
    )
    balance_group.add_argument(
        "--no-balance-controller",
        action="store_false",
        dest="balance_controller",
        help="Ablation: send retargeted angles directly, so a free-base robot may fall.",
    )
    parser.set_defaults(balance_controller=True)
    parser.add_argument(
        "--viewer-mode",
        choices=("visual", "joints"),
        default="visual",
        help="Initial MuJoCo rendering: imported robot meshes or joint skeleton.",
    )
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
    parser.add_argument("--replay-path", "--video-path", dest="replay_path", type=Path)
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
        default=0,
        help=(
            "Override MuJoCo steps per input frame; zero (default) follows source "
            "timestamps with a fixed-timestep accumulator."
        ),
    )
    parser.add_argument(
        "--robot-websocket-url",
        default="",
        help=(
            "Optional legacy robot/Unity ws:// or wss:// endpoint. The output is "
            "disabled unless this option is supplied."
        ),
    )
    parser.add_argument(
        "--robot-websocket-timeout-s",
        type=float,
        default=0.5,
        help="Legacy WebSocket connect/send timeout (default: 0.5 s).",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.physics_steps_per_frame < 0:
        raise ValueError("--physics-steps-per-frame must be non-negative")
    if args.camera_index < 0:
        raise ValueError("--camera-index must be non-negative")
    if args.camera_width <= 0 or args.camera_height <= 0 or args.camera_fps <= 0.0:
        raise ValueError("camera width, height, and fps must be positive")
    if not np.isfinite(args.robot_websocket_timeout_s) or args.robot_websocket_timeout_s <= 0.0:
        raise ValueError("--robot-websocket-timeout-s must be finite and positive")
    for name in (
        "min_pose_detection_confidence",
        "min_pose_presence_confidence",
        "min_tracking_confidence",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be within [0, 1]")
    if args.source == "replay" and args.replay_path is None:
        raise ValueError("--video-path is required when --source replay is selected")
    url = str(args.robot_websocket_url).strip()
    if url:
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        if parts.scheme not in {"ws", "wss"} or not parts.netloc:
            raise ValueError("--robot-websocket-url must be an absolute ws:// or wss:// URL")
        if not args.free_base or not args.balance_controller:
            raise ValueError(
                "robot WebSocket output requires --free-base and --balance-controller"
            )


def _make_robot_publisher(args: argparse.Namespace) -> Any | None:
    """Create a disabled-by-default, failure-isolated legacy output boundary."""

    url = str(args.robot_websocket_url).strip()
    if not url:
        return None
    try:
        import websocket  # type: ignore[import-not-found]
    except ImportError:
        websocket = None
    if websocket is None or not callable(getattr(websocket, "create_connection", None)):
        raise RuntimeError(
            "robot WebSocket output requires websocket-client; install it with "
            "scripts/setup_windows.ps1 or python -m pip install websocket-client"
        )
    from robot_human_interface.protocol import (
        LatestCommandPublisher,
        LegacyWebSocketEncoder,
        WebSocketTransport,
    )

    publisher = LatestCommandPublisher(
        WebSocketTransport(url, timeout_s=float(args.robot_websocket_timeout_s)),
        LegacyWebSocketEncoder(joint_config_path=PROJECT_ROOT / "config" / "joints.yaml"),
        rate_hz=10.0,
        repeat_latest=True,
    )
    return publisher


def _video_path(args: argparse.Namespace) -> Path:
    """Return the explicit replay path or the repository's licensed demo MP4."""

    path = BUNDLED_VIDEO_PATHS[args.demo_video] if args.source == "mp4" else args.replay_path
    if path is None:
        raise ValueError("--video-path is required when --source replay is selected")
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"video file does not exist: {resolved}")
    return resolved


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
                realtime=not args.headless or bool(args.robot_websocket_url),
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
        source = OpenCVVideoSource(
            _video_path(args),
            mirror=args.mirror_input,
            loop=args.loop_replay,
            realtime=not args.headless or bool(args.robot_websocket_url),
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
    view_mode: str,
    balance_enabled: bool,
    support_intent: str,
    support_phase: str,
    source_label: str,
    frame_index: int,
) -> np.ndarray:
    import cv2

    status = "PAUSED" if paused else ("POSE LOST / SAFE RETURN" if stale else "TRACKING")
    color = (60, 220, 255) if paused else ((70, 70, 255) if stale else (70, 230, 90))
    cv2.putText(image, status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        image,
        (
            f"source={source_label} | frame={frame_index} | base={base_mode} | "
            f"balance={'motor' if balance_enabled else 'off'} | "
            f"support={support_intent}/{support_phase} | view={view_mode}"
        ),
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
    )
    cv2.putText(
        image,
        "V visual/joints | C calibrate | R reset | Space pause | Esc exit",
        (16, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
    )
    if calibrating:
        cv2.putText(
            image,
            f"calibration {calibration_progress * 100:3.0f}% - hold neutral pose",
            (16, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 220, 255),
            2,
        )
    return image


def _create_display_window(cv2: Any, image: np.ndarray) -> None:
    """Create a resizable preview that also fits portrait phone videos."""

    flags = int(cv2.WINDOW_NORMAL) | int(getattr(cv2, "WINDOW_KEEPRATIO", 0))
    cv2.namedWindow(WINDOW_NAME, flags)
    height, width = image.shape[:2]
    scale = min(1.0, 1100.0 / width, 850.0 / height)
    cv2.resizeWindow(
        WINDOW_NAME,
        max(320, int(round(width * scale))),
        max(320, int(round(height * scale))),
    )


def _display_window_is_open(cv2: Any) -> bool:
    try:
        return float(cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)) >= 1.0
    except cv2.error:
        return False


def _handle_key(
    key: int,
    *,
    simulation: Any,
    retargeter: Any,
    paused: bool,
    calibrators: Sequence[Any] = (),
    calibration_resetters: Sequence[Any] = (),
    resetters: Sequence[Any] = (),
    allow_pause: bool = True,
) -> tuple[bool, bool]:
    """Return ``(keep_running, paused)`` after one OpenCV key code."""

    if key in (27,):
        return False, paused
    if key in (ord(" "),):
        if allow_pause:
            paused = not paused
            LOGGER.info("teleoperation %s", "paused" if paused else "resumed")
        else:
            LOGGER.warning(
                "pause is disabled while robot WebSocket output is active; "
                "use the hardware E-stop/watchdog for a safe stop"
            )
    elif key in (ord("c"), ord("C")):
        # Keep the retargeter and support-intent estimator on the same neutral
        # window.  A shorter retargeter window could start moving the robot
        # while the support estimator was still learning its floor/up axis.
        retargeter.start_calibration(30)
        for resetter in calibration_resetters:
            resetter.reset()
        for calibrator in calibrators:
            calibrator.start_calibration()
        LOGGER.info("neutral-pose calibration started (30 confident frames)")
    elif key in (ord("r"), ord("R")):
        retargeter.reset()
        simulation.reset()
        for resetter in resetters:
            resetter.reset()
        LOGGER.info("simulation and retargeter reset")
    elif key in (ord("v"), ord("V")):
        mode = simulation.toggle_view_mode()
        LOGGER.info("MuJoCo viewer mode: %s", mode)
    return True, paused


def run_teleop(args: argparse.Namespace) -> TeleopStats:
    """Run one complete perception/retarget/simulation loop."""

    _validate_args(args)
    from robot_human_interface.pose import draw_pose_overlay
    from robot_human_interface.control import (
        HumanSupportIntentEstimator,
        StandingBalanceController,
        SupportIntent,
        SupportIntentLatch,
        SupportPhase,
        SupportStateMachine,
        load_human_support_intent_config,
        load_standing_balance_config,
        load_support_control_config,
    )
    from robot_human_interface.retargeting import GeometricRetargeter
    from robot_human_interface.simulation import HumanoidSimulation

    robot_publisher = _make_robot_publisher(args)
    source, pose = _make_perception(args)
    retargeter = GeometricRetargeter.from_yaml(
        joints_path=PROJECT_ROOT / "config" / "joints.yaml",
        retargeting_path=PROJECT_ROOT / "config" / "retargeting.yaml",
    )
    base_mode = "free" if args.free_base else "fixed"
    simulation = HumanoidSimulation(base_mode)
    balance_enabled = bool(args.free_base and args.balance_controller)
    balance_config = load_standing_balance_config(PROJECT_ROOT / "config" / "balance.yaml")
    balance_controller = StandingBalanceController.from_simulation(
        simulation, replace(balance_config, enabled=balance_enabled)
    )
    support_machine = SupportStateMachine.from_simulation(
        simulation,
        load_support_control_config(PROJECT_ROOT / "config" / "balance.yaml"),
    )
    support_latch = SupportIntentLatch()
    support_intent_estimator = HumanSupportIntentEstimator(
        load_human_support_intent_config(PROJECT_ROOT / "config" / "balance.yaml")
    )
    support_enabled = balance_enabled
    robot_publisher_started = False
    display_enabled = not args.headless
    viewer_started = False
    display_window_created = False
    paused = False
    frames = 0
    skeleton_frames = 0
    stale_commands = 0
    command_min: np.ndarray | None = None
    command_max: np.ndarray | None = None
    safe_command_min: np.ndarray | None = None
    safe_command_max: np.ndarray | None = None
    final_base_height = float(simulation.get_state().base_position_m[2])
    final_simulation_time = float(simulation.get_state().simulation_time_s)
    first_frame_timestamp: float | None = None
    previous_frame_timestamp: float | None = None
    final_frame_timestamp: float | None = None
    physics_accumulator_s = 0.0
    maximum_tilt_rad = 0.0
    fell = False
    support_transitions = 0
    previous_support_phase = support_machine.phase
    right_swing_completed = False
    left_swing_completed = False
    maximum_right_foot_clearance_m = 0.0
    maximum_left_foot_clearance_m = 0.0
    observed_support_intent = SupportIntent.DOUBLE_SUPPORT
    last_pose_overlay: np.ndarray | None = None
    last_command_stale = True
    source_label = args.source
    if args.source in {"mp4", "replay"}:
        source_label = _video_path(args).name

    try:
        if display_enabled:
            simulation.launch_viewer(args.viewer_mode)
            viewer_started = True

        while args.max_frames <= 0 or frames < args.max_frames:
            if display_enabled and paused:
                import cv2

                simulation.sync_viewer()
                if last_pose_overlay is not None:
                    paused_overlay = _draw_status(
                        last_pose_overlay.copy(),
                        paused=True,
                        stale=last_command_stale,
                        calibrating=(
                            retargeter.is_calibrating
                            or support_intent_estimator.is_calibrating
                        ),
                        calibration_progress=min(
                            retargeter.calibration_progress,
                            support_intent_estimator.calibration_progress,
                        ),
                        base_mode=base_mode,
                        view_mode=simulation.viewer_mode,
                        balance_enabled=balance_enabled,
                        support_intent=support_intent_estimator.intent.value,
                        support_phase=support_machine.phase.value,
                        source_label=source_label,
                        frame_index=frames,
                    )
                    cv2.imshow(WINDOW_NAME, paused_overlay)
                was_paused = paused
                keep_running, paused = _handle_key(
                    cv2.waitKey(20) & 0xFF,
                    simulation=simulation,
                    retargeter=retargeter,
                    paused=paused,
                    calibrators=(support_intent_estimator,),
                    calibration_resetters=(support_latch,),
                    resetters=(
                        balance_controller,
                        support_machine,
                        support_latch,
                        support_intent_estimator,
                    ),
                    allow_pause=robot_publisher is None,
                )
                if was_paused and not paused:
                    # Live-camera timestamps use wall clock.  Without this
                    # reset, resuming after a long pause would execute the
                    # entire paused interval as a burst of 2 ms servo steps
                    # against one newly received camera command.
                    previous_frame_timestamp = None
                    physics_accumulator_s = 0.0
                if not keep_running or not _display_window_is_open(cv2):
                    break
                if viewer_started and not simulation.viewer_is_running:
                    LOGGER.info("MuJoCo viewer was closed")
                    break
                continue

            frame = source.read()
            if frame is None:
                break
            skeleton = pose.estimate(frame)
            if skeleton is not None:
                skeleton_frames += 1
            support_estimate = support_intent_estimator.update(
                skeleton,
                timestamp_s=frame.timestamp_s,
            )
            observed_support_intent = support_estimate.intent

            command = retargeter.retarget(skeleton, timestamp_s=frame.timestamp_s)
            if first_frame_timestamp is None:
                first_frame_timestamp = frame.timestamp_s
            if previous_frame_timestamp is None:
                frame_delta_s = 0.0
            else:
                frame_delta_s = max(0.0, frame.timestamp_s - previous_frame_timestamp)
            previous_frame_timestamp = frame.timestamp_s
            final_frame_timestamp = frame.timestamp_s
            stale_commands += int(command.stale)
            if command_min is None:
                command_min = command.positions_rad.copy()
                command_max = command.positions_rad.copy()
            else:
                command_min = np.minimum(command_min, command.positions_rad)
                command_max = np.maximum(command_max, command.positions_rad)

            if not paused:
                if args.physics_steps_per_frame > 0:
                    physics_steps = args.physics_steps_per_frame
                else:
                    physics_accumulator_s += frame_delta_s
                    timestep_s = float(simulation.model.opt.timestep)
                    physics_steps = int((physics_accumulator_s + 1e-12) // timestep_s)
                    physics_accumulator_s -= physics_steps * timestep_s

                state = simulation.get_state()
                for _ in range(physics_steps):
                    standing_command = balance_controller.update(
                        command,
                        state,
                        dt_s=float(simulation.model.opt.timestep),
                    )
                    if support_enabled:
                        prior_support = support_machine.last_diagnostics
                        aborting = bool(
                            prior_support is not None
                            and prior_support.abort_reason is not None
                            and support_machine.phase
                            in {SupportPhase.LOWER_SWING, SupportPhase.CENTER_WEIGHT}
                        )
                        requested_support = support_latch.update(
                            observed_support_intent,
                            support_machine.phase,
                            stale=command.stale or support_estimate.stale,
                            aborted=aborting,
                            # Media time is monotonic for camera, bundled MP4,
                            # replay loops and synthetic sources.  It gives the
                            # one-slot opposite-leg queue a deterministic lease
                            # independent of 500 Hz physics substeps.
                            timestamp_s=frame.timestamp_s,
                        )
                        safe_command = support_machine.update(
                            standing_command,
                            state,
                            dt_s=float(simulation.model.opt.timestep),
                            intent=requested_support,
                        )
                    else:
                        safe_command = standing_command
                    if safe_command_min is None:
                        safe_command_min = safe_command.positions_rad.copy()
                        safe_command_max = safe_command.positions_rad.copy()
                    else:
                        safe_command_min = np.minimum(
                            safe_command_min, safe_command.positions_rad
                        )
                        safe_command_max = np.maximum(
                            safe_command_max, safe_command.positions_rad
                        )
                    simulation.apply_joint_command(safe_command)
                    if robot_publisher is not None:
                        robot_publisher.submit(safe_command)
                        if not robot_publisher_started:
                            robot_publisher.start()
                            robot_publisher_started = True
                    state = simulation.step()
                    maximum_right_foot_clearance_m = max(
                        maximum_right_foot_clearance_m,
                        float(state.right_foot_position_m[2] - state.left_foot_position_m[2]),
                    )
                    maximum_left_foot_clearance_m = max(
                        maximum_left_foot_clearance_m,
                        float(state.left_foot_position_m[2] - state.right_foot_position_m[2]),
                    )
                    if support_enabled and support_machine.phase is not previous_support_phase:
                        support_transitions += 1
                        previous_support_phase = support_machine.phase
                        LOGGER.info(
                            "support phase=%s active=%s observed=%s t=%.3f "
                            "right_force=%.2f left_force=%.2f",
                            support_machine.phase.value,
                            support_machine.active_intent.value,
                            observed_support_intent.value,
                            state.simulation_time_s,
                            state.right_foot_normal_force_n,
                            state.left_foot_normal_force_n,
                        )
                    if support_machine.phase is SupportPhase.HOLD_SWING:
                        right_swing_completed |= (
                            support_machine.active_intent is SupportIntent.RIGHT_SWING
                        )
                        left_swing_completed |= (
                            support_machine.active_intent is SupportIntent.LEFT_SWING
                        )
                    quaternion = state.base_orientation_wxyz
                    upright = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
                    tilt_rad = float(np.arccos(np.clip(upright, -1.0, 1.0)))
                    maximum_tilt_rad = max(maximum_tilt_rad, tilt_rad)
                    fall_detected = bool(
                        state.base_position_m[2] < 0.55
                        or tilt_rad > np.radians(45.0)
                    )
                    if fall_detected and not fell:
                        LOGGER.warning(
                            "fall threshold crossed at t=%.3f: base_z=%.3f tilt_deg=%.2f",
                            state.simulation_time_s,
                            state.base_position_m[2],
                            np.degrees(tilt_rad),
                        )
                    fell |= fall_detected
                final_base_height = float(state.base_position_m[2])
                final_simulation_time = float(state.simulation_time_s)
            else:
                simulation.sync_viewer()
            frames += 1

            if display_enabled:
                import cv2

                overlay = draw_pose_overlay(frame.image_bgr, skeleton, confidence_threshold=0.5)
                last_pose_overlay = overlay.copy()
                last_command_stale = command.stale
                if not display_window_created:
                    _create_display_window(cv2, overlay)
                    display_window_created = True
                _draw_status(
                    overlay,
                    paused=paused,
                    stale=command.stale,
                    calibrating=(
                        retargeter.is_calibrating
                        or support_intent_estimator.is_calibrating
                    ),
                    calibration_progress=min(
                        retargeter.calibration_progress,
                        support_intent_estimator.calibration_progress,
                    ),
                    base_mode=base_mode,
                    view_mode=simulation.viewer_mode,
                    balance_enabled=balance_enabled,
                    support_intent=support_intent_estimator.intent.value,
                    support_phase=support_machine.phase.value,
                    source_label=source_label,
                    frame_index=frames,
                )
                cv2.imshow(WINDOW_NAME, overlay)
                keep_running, paused = _handle_key(
                    cv2.waitKey(1) & 0xFF,
                    simulation=simulation,
                    retargeter=retargeter,
                    paused=paused,
                    calibrators=(support_intent_estimator,),
                    calibration_resetters=(support_latch,),
                    resetters=(
                        balance_controller,
                        support_machine,
                        support_latch,
                        support_intent_estimator,
                    ),
                    allow_pause=robot_publisher is None,
                )
                if not keep_running:
                    break
                if not _display_window_is_open(cv2):
                    LOGGER.info("skeleton preview window was closed")
                    break
                if viewer_started and not simulation.viewer_is_running:
                    LOGGER.info("MuJoCo viewer was closed")
                    break
        if args.source in {"mp4", "replay"} and frames == 0:
            raise RuntimeError(f"video did not yield a decodable frame: {_video_path(args)}")
    finally:
        if robot_publisher is not None:
            robot_publisher.stop()
        if display_enabled and display_window_created:
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
    safe_command_span = 0.0
    if safe_command_min is not None and safe_command_max is not None:
        safe_command_span = float(np.max(safe_command_max - safe_command_min))
    media_time = 0.0
    if first_frame_timestamp is not None and final_frame_timestamp is not None:
        media_time = max(0.0, final_frame_timestamp - first_frame_timestamp)
    robot_last_error = None
    robot_send_attempts = 0
    robot_commands_sent = 0
    if robot_publisher is not None:
        robot_send_attempts = int(robot_publisher.attempt_count)
        robot_commands_sent = int(robot_publisher.sent_count)
        if robot_publisher.last_error is not None:
            robot_last_error = str(robot_publisher.last_error)
    return TeleopStats(
        source=args.source,
        base_mode=base_mode,
        frames=frames,
        skeleton_frames=skeleton_frames,
        stale_commands=stale_commands,
        command_span_rad=command_span,
        safe_command_span_rad=safe_command_span,
        final_base_height_m=final_base_height,
        simulation_time_s=final_simulation_time,
        media_time_s=media_time,
        maximum_tilt_rad=maximum_tilt_rad,
        fell=fell,
        robot_output_enabled=robot_publisher is not None,
        robot_send_attempts=robot_send_attempts,
        robot_commands_sent=robot_commands_sent,
        robot_last_error=robot_last_error,
        support_transitions=support_transitions,
        right_swing_completed=right_swing_completed,
        left_swing_completed=left_swing_completed,
        maximum_right_foot_clearance_m=maximum_right_foot_clearance_m,
        maximum_left_foot_clearance_m=maximum_left_foot_clearance_m,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.free_base and not args.balance_controller:
        LOGGER.warning("FREE-BASE balance controller is disabled; the robot may fall.")
    if args.robot_websocket_url:
        LOGGER.warning(
            "Legacy robot output is enabled. Verify the physical E-stop, joint signs, "
            "limits, and an onboard watchdog before allowing motor power."
        )
    try:
        stats = run_teleop(args)
    except KeyboardInterrupt:
        LOGGER.info("interrupted by user")
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2

    if stats.robot_last_error is not None:
        LOGGER.error("legacy robot output ended with an error: %s", stats.robot_last_error)

    print(
        "teleop complete: "
        f"source={stats.source} frames={stats.frames} "
        f"skeleton_frames={stats.skeleton_frames} stale={stats.stale_commands} "
        f"base={stats.base_mode} command_span_deg={np.degrees(stats.command_span_rad):.3f} "
        f"safe_span_deg={np.degrees(stats.safe_command_span_rad):.3f} "
        f"base_z_m={stats.final_base_height_m:.4f} "
        f"max_tilt_deg={np.degrees(stats.maximum_tilt_rad):.3f} "
        f"sim_s={stats.simulation_time_s:.3f} media_s={stats.media_time_s:.3f} "
        f"fell={int(stats.fell)} robot_output={int(stats.robot_output_enabled)} "
        f"robot_sent={stats.robot_commands_sent}/{stats.robot_send_attempts} "
        f"support_transitions={stats.support_transitions} "
        f"right_swing={int(stats.right_swing_completed)} "
        f"left_swing={int(stats.left_swing_completed)} "
        f"right_clearance_m={stats.maximum_right_foot_clearance_m:.4f} "
        f"left_clearance_m={stats.maximum_left_foot_clearance_m:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
