"""End-to-end camera/synthetic pose teleoperation for the MuJoCo humanoid."""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass, field, replace
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

# A settling pass is intentionally stricter than the fall detector.  These
# limits describe a genuinely quiet, trackable double-support state rather
# than merely an upright robot.  They are also exported in the acceptance
# report, so a completed settling interval cannot hide motion behind a boolean.
SETTLING_MAX_BASE_LINEAR_SPEED_M_S = 0.06
SETTLING_MAX_JOINT_SPEED_RAD_S = 0.40
SETTLING_MAX_JOINT_TRACKING_ERROR_RAD = 0.12
SETTLING_MAX_LOADED_FOOT_SLIP_SPEED_M_S = 0.03
SETTLING_MAX_CAPTURE_POINT_ERROR_M = 0.07
FOOT_CONTACT_DETECTION_THRESHOLD_N = 1e-3
FOOT_LOAD_TELEMETRY_THRESHOLD_N = 4.0


def _stale_support_force_return_reason(
    *,
    support_estimate_stale: bool,
    reference_stale: bool,
    prior_phase: object | None,
) -> str | None:
    """Abort only when stale perception invalidates an active lift request.

    The support latch still receives every stale observation and requests
    double support immediately.  A requested return is not yet a physical
    return while the FSM remains in SHIFT/VERIFY/LIFT/HOLD, so stale perception
    must still bypass any HOLD dwell and force the lower-and-center recovery.
    Suppression is safe only after the FSM has actually entered that recovery.
    An active or unknown phase remains fail-closed.
    """

    if not support_estimate_stale or reference_stale:
        return None
    phase_value = getattr(prior_phase, "value", prior_phase)
    if phase_value in {
        "lower_swing",
        "verify_touchdown",
        "center_weight",
        "double_support",
    }:
        return None
    return "stale_support_intent"


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
    minimum_base_height_m: float
    settling_requested_s: float
    settling_elapsed_s: float
    settling_stable_s: float
    settling_completed: bool
    settling_minimum_base_height_m: float
    settling_maximum_tilt_rad: float
    maximum_non_foot_ground_contacts: int
    maximum_loaded_foot_slip_speed_m_s: float
    right_foot_slip_distance_m: float
    left_foot_slip_distance_m: float
    maximum_swing_foot_impact_speed_m_s: float
    maximum_swing_foot_impact_force_n: float
    maximum_swing_foot_contact_impulse_n_s: float
    # Maxima over the final uninterrupted quiet interval.  Defaults preserve
    # compatibility with external callers that construct the older summary.
    maximum_swing_foot_precontact_vertical_speed_m_s: float = 0.0
    settling_maximum_base_linear_speed_m_s: float = 0.0
    settling_maximum_joint_speed_rad_s: float = 0.0
    settling_maximum_joint_tracking_error_rad: float = 0.0
    settling_maximum_loaded_foot_slip_speed_m_s: float = 0.0
    settling_maximum_capture_point_error_m: float = 0.0
    swing_foot_contact_episodes: int = 0
    settling_observation_count: int = 0
    support_abort_count: int = 0
    support_abort_reasons: tuple[str, ...] = ()
    calibration_mode: str = "automatic_window"
    calibration_source_path: str | None = None
    calibration_source_sha256: str | None = None
    calibration_frame_index: int | None = None


def _phase_aware_loaded_feet(
    phase: object | None,
    active_intent: object | None,
) -> tuple[bool, bool]:
    """Return which feet count as load-bearing for slip telemetry.

    A swing sole can briefly collide with the ground while moving quickly.  A
    one-tick impact is useful impact telemetry, but it is not stance-foot slip.
    Both feet remain stance feet through SHIFT_WEIGHT and VERIFY_STANCE. Count
    only the verified stance foot after lift begins. ``touchdown_ready`` in
    support diagnostics is only an instantaneous force predicate, so the swing
    sole remains impact telemetry throughout VERIFY_TOUCHDOWN. The support FSM
    enters CENTER_WEIGHT only after its timed touchdown confirmation; that
    phase, the pre-lift phases, and double support admit both feet as
    load-bearing.
    """

    phase_value = getattr(phase, "value", phase)
    intent_value = getattr(active_intent, "value", active_intent)
    if phase_value in {
        "double_support",
        "shift_weight",
        "verify_stance",
        "center_weight",
    }:
        # The selected swing sole remains planted while weight is shifted and
        # the stance load is verified.  Any loaded motion of either sole in
        # those phases is real stance slip, not a swing-foot collision.
        return True, True
    if intent_value == "right_swing":
        return False, True
    if intent_value == "left_swing":
        return True, False
    return True, True


@dataclass(slots=True)
class _SwingContactEpisode:
    """Per-foot state needed to measure a complete semantic landing episode."""

    airborne_sample_available: bool = False
    last_airborne_vertical_velocity_m_s: float = 0.0
    contact_active: bool = False
    contact_impulse_n_s: float = 0.0


@dataclass(slots=True)
class _SupportAbortTelemetry:
    """Count rising/changed support fault events without counting every tick."""

    count: int = 0
    reasons: list[str] = field(default_factory=list)
    previous_reason: str | None = None

    def update(self, reason: object | None) -> None:
        normalized = None if reason is None else str(reason)
        if normalized is not None and normalized != self.previous_reason:
            self.count += 1
            self.reasons.append(normalized)
        self.previous_reason = normalized


@dataclass(slots=True)
class _FootContactTelemetry:
    """Phase-aware stance-slip and swing-landing accumulator.

    A landing is admitted only after that swing foot was observed unloaded.
    The pre-contact vertical velocity therefore comes from the final physics
    sample before the first force-bearing sample.  Normal force is integrated
    for every tick of the ensuing semantic swing-contact episode, including
    zero-force contact chatter, until the support FSM promotes the foot to
    stance.  It is not approximated by a single maximum-force tick.
    """

    maximum_loaded_foot_slip_speed_m_s: float = 0.0
    right_foot_slip_distance_m: float = 0.0
    left_foot_slip_distance_m: float = 0.0
    maximum_swing_foot_impact_speed_m_s: float = 0.0
    maximum_swing_foot_precontact_vertical_speed_m_s: float = 0.0
    maximum_swing_foot_impact_force_n: float = 0.0
    maximum_swing_foot_contact_impulse_n_s: float = 0.0
    swing_foot_contact_episodes: int = 0
    _right: _SwingContactEpisode = field(
        default_factory=_SwingContactEpisode, init=False, repr=False
    )
    _left: _SwingContactEpisode = field(
        default_factory=_SwingContactEpisode, init=False, repr=False
    )

    def update(
        self,
        state: object,
        *,
        phase: object | None,
        active_intent: object | None,
        dt_s: float,
    ) -> None:
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        phase_value = getattr(phase, "value", phase)
        landing_admissible = phase_value in {
            "lift_swing",
            "hold_swing",
            "lower_swing",
            "verify_touchdown",
        }
        track_right, track_left = _phase_aware_loaded_feet(phase, active_intent)
        samples = (
            (
                float(state.right_foot_normal_force_n),
                np.asarray(state.right_foot_linear_velocity_m_s, dtype=np.float64),
                track_right,
                self._right,
            ),
            (
                float(state.left_foot_normal_force_n),
                np.asarray(state.left_foot_linear_velocity_m_s, dtype=np.float64),
                track_left,
                self._left,
            ),
        )
        for index, (force_n, velocity, is_stance, episode) in enumerate(samples):
            speed_m_s = float(np.linalg.norm(velocity[:2]))
            contacting = force_n >= FOOT_CONTACT_DETECTION_THRESHOLD_N
            loaded = force_n >= FOOT_LOAD_TELEMETRY_THRESHOLD_N
            if is_stance:
                if episode.contact_active:
                    self._finish_episode(episode)
                episode.airborne_sample_available = False
                if loaded:
                    self.maximum_loaded_foot_slip_speed_m_s = max(
                        self.maximum_loaded_foot_slip_speed_m_s, speed_m_s
                    )
                    if index == 0:
                        self.right_foot_slip_distance_m += speed_m_s * dt_s
                    else:
                        self.left_foot_slip_distance_m += speed_m_s * dt_s
                continue

            if not contacting:
                if episode.contact_active:
                    # Keep a semantic touchdown episode open across contact
                    # solver chatter/bounce.  It ends when the support FSM
                    # promotes this foot to stance, not every time normal
                    # force briefly crosses the measurement threshold.  The
                    # zero-force sample is nevertheless the new pre-contact
                    # velocity if the sole strikes again; otherwise a tiny
                    # early chatter contact could hide a later hard landing.
                    episode.airborne_sample_available = True
                    episode.last_airborne_vertical_velocity_m_s = float(
                        velocity[2]
                    )
                    continue
                episode.airborne_sample_available = True
                episode.last_airborne_vertical_velocity_m_s = float(velocity[2])
                continue

            # Contact during any airborne swing phase is part of the landing
            # episode once an unloaded sample has been observed.  In
            # particular, an unintended LIFT/HOLD collision must retain its
            # peak and impulse instead of disappearing before LOWER starts.
            # Requiring the unloaded sample still excludes the initially
            # loaded foot in SHIFT_WEIGHT from being mistaken for a touchdown.
            if not episode.contact_active:
                if (
                    not landing_admissible
                    or not episode.airborne_sample_available
                ):
                    continue
                episode.contact_active = True
                episode.contact_impulse_n_s = 0.0
                self.swing_foot_contact_episodes += 1
                downward_speed = max(
                    0.0, -episode.last_airborne_vertical_velocity_m_s
                )
                self.maximum_swing_foot_precontact_vertical_speed_m_s = max(
                    self.maximum_swing_foot_precontact_vertical_speed_m_s,
                    downward_speed,
                )
                episode.airborne_sample_available = False
            elif episode.airborne_sample_available:
                # Re-contact within the same semantic landing episode.  Keep
                # the integrated impulse, but conservatively update the
                # pre-contact speed from the final zero-force bounce sample.
                downward_speed = max(
                    0.0, -episode.last_airborne_vertical_velocity_m_s
                )
                self.maximum_swing_foot_precontact_vertical_speed_m_s = max(
                    self.maximum_swing_foot_precontact_vertical_speed_m_s,
                    downward_speed,
                )
                episode.airborne_sample_available = False
            self.maximum_swing_foot_impact_speed_m_s = max(
                self.maximum_swing_foot_impact_speed_m_s,
                speed_m_s,
            )
            episode.contact_impulse_n_s += force_n * dt_s
            self.maximum_swing_foot_impact_force_n = max(
                self.maximum_swing_foot_impact_force_n, force_n
            )
            self.maximum_swing_foot_contact_impulse_n_s = max(
                self.maximum_swing_foot_contact_impulse_n_s,
                episode.contact_impulse_n_s,
            )

    def _finish_episode(self, episode: _SwingContactEpisode) -> None:
        self.maximum_swing_foot_contact_impulse_n_s = max(
            self.maximum_swing_foot_contact_impulse_n_s,
            episode.contact_impulse_n_s,
        )
        episode.contact_active = False
        episode.contact_impulse_n_s = 0.0


def _loaded_foot_slip_speed_m_s(state: object, *, load_threshold_n: float) -> float:
    """Maximum horizontal speed of feet carrying at least ``load_threshold_n``."""

    speeds = [
        float(np.linalg.norm(state.right_foot_linear_velocity_m_s[:2]))
        if float(state.right_foot_normal_force_n) >= load_threshold_n
        else 0.0,
        float(np.linalg.norm(state.left_foot_linear_velocity_m_s[:2]))
        if float(state.left_foot_normal_force_n) >= load_threshold_n
        else 0.0,
    ]
    return max(speeds)


def _capture_point_error_m(balance_controller: object) -> float:
    """Controller-calibrated sagittal capture-point error.

    The physical robot has a non-zero nominal CoM-to-foot-center offset.  The
    standing controller learns that neutral offset, so its diagnostic is the
    authoritative stability signal; raw absolute geometry would reject a
    perfectly motionless nominal pose.
    """

    diagnostics = getattr(balance_controller, "last_diagnostics", None)
    value = getattr(diagnostics, "capture_point_error_x_m", None)
    if value is None or not np.isfinite(value):
        return float("inf")
    return abs(float(value))


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
        help=(
            "Use the unconstrained torso and motor-angle balance controller "
            "(default)."
        ),
    )
    base_group.add_argument(
        "--fixed-base",
        action="store_false",
        dest="free_base",
        help=(
            "Explicitly use the grounded vertical stabilizer for kinematic/visual "
            "inspection instead of free-base balance."
        ),
    )
    parser.set_defaults(free_base=True)
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
        "--retargeting",
        choices=("ik", "geometric"),
        default="ik",
        help=(
            "Human-to-robot mapping. 'ik' minimizes whole-body pose error under "
            "the robot joint limits; 'geometric' keeps the original scalar baseline."
        ),
    )
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
        "--calibration-video",
        type=Path,
        help=(
            "Explicit controlled-replay neutral source. Must be paired with "
            "--calibration-frame and is never enabled automatically."
        ),
    )
    parser.add_argument(
        "--calibration-frame",
        type=int,
        help=(
            "Zero-based frame in --calibration-video to validate and install "
            "as the replay neutral reference."
        ),
    )
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
        "--settle-seconds",
        type=float,
        default=0.0,
        help=(
            "After a finite source ends, command double support and require this "
            "many quiet seconds on the same free-base simulation (default: 0)."
        ),
    )
    parser.add_argument(
        "--settle-timeout-s",
        type=float,
        default=20.0,
        help=(
            "Maximum same-simulation return-and-settle time when --settle-seconds "
            "is enabled (default: 20 s)."
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
    if not np.isfinite(args.settle_seconds) or args.settle_seconds < 0.0:
        raise ValueError("--settle-seconds must be finite and non-negative")
    if not np.isfinite(args.settle_timeout_s) or args.settle_timeout_s <= 0.0:
        raise ValueError("--settle-timeout-s must be finite and positive")
    if args.settle_seconds > args.settle_timeout_s:
        raise ValueError("--settle-seconds cannot exceed --settle-timeout-s")
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
    calibration_requested = (
        args.calibration_video is not None or args.calibration_frame is not None
    )
    if (args.calibration_video is None) != (args.calibration_frame is None):
        raise ValueError(
            "--calibration-video and --calibration-frame must be supplied together"
        )
    if calibration_requested:
        if args.source not in {"mp4", "replay"}:
            raise ValueError(
                "explicit calibration video is allowed only with --source mp4/replay"
            )
        if args.calibration_frame < 0:
            raise ValueError("--calibration-frame must be non-negative")
    url = str(args.robot_websocket_url).strip()
    if url:
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        if parts.scheme not in {"ws", "wss"} or not parts.netloc:
            raise ValueError("--robot-websocket-url must be an absolute ws:// or wss:// URL")
        if not args.free_base or not args.balance_controller:
            raise ValueError(
                "robot WebSocket output requires free-base mode and "
                "--balance-controller; --fixed-base is incompatible"
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


def _calibration_video_path(args: argparse.Namespace) -> Path | None:
    """Resolve an explicitly requested controlled-replay calibration asset."""

    if args.calibration_video is None:
        return None
    resolved = args.calibration_video.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"calibration video file does not exist: {resolved}"
        )
    return resolved


def _apply_explicit_replay_calibration(
    args: argparse.Namespace,
    *,
    pose: Any,
    retargeter: Any,
    support_intent_estimator: Any,
) -> tuple[str, str, int] | None:
    """Validate and install one user-selected replay frame before statistics.

    Every frame through the selected index is inferred so MediaPipe tracking
    and the configured EMA have the same deterministic history as normal
    replay. Only the selected frame is admitted as a reference. Automatic
    live calibration is untouched, and no main-source frame or simulation
    step is consumed here.
    """

    path = _calibration_video_path(args)
    if path is None:
        return None
    frame_index = int(args.calibration_frame)
    from robot_human_interface.camera import OpenCVVideoSource

    calibration_source = OpenCVVideoSource(
        path,
        mirror=args.mirror_input,
        loop=False,
        realtime=False,
    )
    selected_skeleton = None
    try:
        for index in range(frame_index + 1):
            frame = calibration_source.read()
            if frame is None:
                raise ValueError(
                    f"--calibration-frame {frame_index} is outside video {path}"
                )
            skeleton = pose.estimate(frame)
            if index == frame_index:
                selected_skeleton = skeleton
    finally:
        calibration_source.close()
        # Keep the same estimator/configuration but clear calibration-video
        # tracking/filter timestamps before the operational replay starts.
        pose.close()
    if selected_skeleton is None:
        raise ValueError(
            f"no confident skeleton at --calibration-frame {frame_index}: {path}"
        )
    if not retargeter.calibrate(selected_skeleton):
        raise ValueError(
            "selected calibration frame failed retargeting visibility, "
            "double-support, arms-down, or extended-leg neutral gates: "
            f"{path} frame {frame_index}"
        )
    if not support_intent_estimator.calibrate(selected_skeleton):
        raise ValueError(
            "selected calibration frame failed support-intent visibility, "
            "double-support, arms-down, or extended-leg neutral gates: "
            f"{path} frame {frame_index}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    LOGGER.info(
        "controlled replay calibration installed: source=%s frame=%d sha256=%s",
        path,
        frame_index,
        digest,
    )
    return str(path), digest, frame_index


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
    retargeting_mode: str,
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
            f"source={source_label} | frame={frame_index} | retarget={retargeting_mode} | "
            f"base={base_mode} | "
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
    from robot_human_interface.retargeting import GeometricRetargeter, MujocoIKRetargeter
    from robot_human_interface.simulation import HumanoidSimulation
    from robot_human_interface.skeleton import RobotJointCommand

    robot_publisher = _make_robot_publisher(args)
    source, pose = _make_perception(args)
    retargeter_type = (
        MujocoIKRetargeter if args.retargeting == "ik" else GeometricRetargeter
    )
    retargeter = retargeter_type.from_yaml(
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
    support_config = load_support_control_config(PROJECT_ROOT / "config" / "balance.yaml")
    support_machine = SupportStateMachine.from_simulation(simulation, support_config)
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
    initial_state = simulation.get_state()
    final_base_height = float(initial_state.base_position_m[2])
    minimum_base_height = final_base_height
    final_simulation_time = float(initial_state.simulation_time_s)
    first_frame_timestamp: float | None = None
    previous_frame_timestamp: float | None = None
    final_frame_timestamp: float | None = None
    physics_accumulator_s = 0.0
    maximum_tilt_rad = 0.0
    fell = False
    support_transitions = 0
    support_abort_telemetry = _SupportAbortTelemetry()
    previous_support_phase = support_machine.phase
    right_swing_completed = False
    left_swing_completed = False
    maximum_right_foot_clearance_m = 0.0
    maximum_left_foot_clearance_m = 0.0
    settling_elapsed_s = 0.0
    settling_stable_s = 0.0
    settling_completed = args.settle_seconds <= 0.0
    settling_minimum_base_height = final_base_height
    settling_maximum_tilt_rad = 0.0
    settling_maximum_base_linear_speed_m_s = 0.0
    settling_maximum_joint_speed_rad_s = 0.0
    settling_maximum_joint_tracking_error_rad = 0.0
    settling_maximum_loaded_foot_slip_speed_m_s = 0.0
    settling_maximum_capture_point_error_m = 0.0
    settling_observation_count = 0
    settling_last_rejection_reasons: tuple[str, ...] = ()
    maximum_non_foot_ground_contacts = 0
    foot_contact_telemetry = _FootContactTelemetry()
    finite_input_completed = False
    observed_support_intent = SupportIntent.DOUBLE_SUPPORT
    last_pose_overlay: np.ndarray | None = None
    last_command_stale = True
    calibration_mode = "automatic_window"
    calibration_source_path: str | None = None
    calibration_source_sha256: str | None = None
    calibration_frame_index: int | None = None
    source_label = args.source
    if args.source in {"mp4", "replay"}:
        source_label = _video_path(args).name

    def record_foot_contact_motion(state: object, *, dt_s: float) -> None:
        """Accumulate phase-aware stance slip and separate swing impacts."""

        diagnostics = support_machine.last_diagnostics if support_enabled else None
        foot_contact_telemetry.update(
            state,
            phase=None if diagnostics is None else diagnostics.phase,
            active_intent=None if diagnostics is None else diagnostics.active_intent,
            dt_s=dt_s,
        )

    try:
        explicit_calibration = _apply_explicit_replay_calibration(
            args,
            pose=pose,
            retargeter=retargeter,
            support_intent_estimator=support_intent_estimator,
        )
        if explicit_calibration is not None:
            (
                calibration_source_path,
                calibration_source_sha256,
                calibration_frame_index,
            ) = explicit_calibration
            calibration_mode = "explicit_replay_frame"
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
                        retargeting_mode=args.retargeting,
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
                finite_input_completed = True
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
                timestep_s = float(simulation.model.opt.timestep)
                if args.physics_steps_per_frame > 0:
                    physics_steps = args.physics_steps_per_frame
                else:
                    physics_accumulator_s += frame_delta_s
                    physics_steps = int((physics_accumulator_s + 1e-12) // timestep_s)
                    physics_accumulator_s -= physics_steps * timestep_s

                state = simulation.get_state()
                for _ in range(physics_steps):
                    standing_command = balance_controller.update(
                        command,
                        state,
                        dt_s=float(simulation.model.opt.timestep),
                        squat_active=support_estimate.squat_active,
                        squat_observation_fresh=(
                            support_estimate.squat_observation_fresh
                        ),
                        squat_depth_ratio=support_estimate.squat_depth_ratio,
                        allow_squat=(
                            support_machine.phase is SupportPhase.DOUBLE_SUPPORT
                        ),
                    )
                    if support_enabled:
                        prior_support = support_machine.last_diagnostics
                        aborting = bool(
                            prior_support is not None
                            and prior_support.abort_reason is not None
                            and support_machine.phase
                            in {SupportPhase.LOWER_SWING, SupportPhase.CENTER_WEIGHT}
                        )
                        balance_diagnostics = balance_controller.last_diagnostics
                        squat_interlocked = bool(
                            balance_diagnostics is not None
                            and not balance_diagnostics.squat_ready_for_support
                        )
                        support_observation = (
                            SupportIntent.DOUBLE_SUPPORT
                            if squat_interlocked
                            or support_estimate.squat_active
                            else observed_support_intent
                        )
                        requested_support = support_latch.update(
                            support_observation,
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
                            force_return_reason=_stale_support_force_return_reason(
                                support_estimate_stale=support_estimate.stale,
                                reference_stale=command.stale,
                                prior_phase=support_machine.phase,
                            ),
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
                    minimum_base_height = min(
                        minimum_base_height, float(state.base_position_m[2])
                    )
                    maximum_non_foot_ground_contacts = max(
                        maximum_non_foot_ground_contacts,
                        state.non_foot_ground_contact_count,
                    )
                    record_foot_contact_motion(state, dt_s=timestep_s)
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
                    diagnostics = support_machine.last_diagnostics
                    abort_reason = (
                        None if diagnostics is None else diagnostics.abort_reason
                    )
                    support_abort_telemetry.update(abort_reason)
                    quaternion = state.base_orientation_wxyz
                    upright = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
                    tilt_rad = float(np.arccos(np.clip(upright, -1.0, 1.0)))
                    maximum_tilt_rad = max(maximum_tilt_rad, tilt_rad)
                    fall_detected = bool(
                        state.base_position_m[2] < 0.55
                        or tilt_rad > np.radians(45.0)
                        or state.non_foot_ground_contact_count > 0
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
                    retargeting_mode=args.retargeting,
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
        finite_input_completed |= bool(
            args.max_frames > 0 and frames >= args.max_frames
        )
        if args.settle_seconds > 0.0 and finite_input_completed and not fell:
            LOGGER.info(
                "input complete; returning to double support and requiring %.3f s "
                "of same-simulation settling",
                args.settle_seconds,
            )
            support_latch.reset()
            timestep_s = float(simulation.model.opt.timestep)
            stable_tilt_limit_rad = min(
                support_config.start_max_tilt_rad, np.radians(12.0)
            )
            while settling_elapsed_s + 1e-12 < args.settle_timeout_s:
                state = simulation.get_state()
                neutral_reference = RobotJointCommand.humanoid(
                    (final_frame_timestamp or 0.0) + settling_elapsed_s,
                    simulation.home_positions_rad,
                    1.0,
                )
                standing_command = balance_controller.update(
                    neutral_reference,
                    state,
                    dt_s=timestep_s,
                )
                if support_enabled:
                    safe_command = support_machine.update(
                        standing_command,
                        state,
                        dt_s=timestep_s,
                        intent=SupportIntent.DOUBLE_SUPPORT,
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
                settling_elapsed_s += timestep_s
                settling_observation_count += 1
                height_m = float(state.base_position_m[2])
                settling_minimum_base_height = min(
                    settling_minimum_base_height, height_m
                )
                minimum_base_height = min(minimum_base_height, height_m)
                maximum_non_foot_ground_contacts = max(
                    maximum_non_foot_ground_contacts,
                    state.non_foot_ground_contact_count,
                )
                record_foot_contact_motion(state, dt_s=timestep_s)
                quaternion = state.base_orientation_wxyz
                upright = 1.0 - 2.0 * (
                    quaternion[1] ** 2 + quaternion[2] ** 2
                )
                tilt_rad = float(np.arccos(np.clip(upright, -1.0, 1.0)))
                settling_maximum_tilt_rad = max(
                    settling_maximum_tilt_rad, tilt_rad
                )
                maximum_tilt_rad = max(maximum_tilt_rad, tilt_rad)
                fall_detected = bool(
                    height_m < 0.55
                    or tilt_rad > np.radians(45.0)
                    or state.non_foot_ground_contact_count > 0
                )
                fell |= fall_detected
                if support_enabled and support_machine.phase is not previous_support_phase:
                    support_transitions += 1
                    previous_support_phase = support_machine.phase
                diagnostics = support_machine.last_diagnostics
                abort_reason = (
                    None if diagnostics is None else diagnostics.abort_reason
                )
                support_abort_telemetry.update(abort_reason)
                angular_speed = float(
                    np.linalg.norm(state.base_angular_velocity_rad_s)
                )
                base_linear_speed_m_s = float(
                    np.linalg.norm(state.base_linear_velocity_m_s)
                )
                joint_speed_rad_s = float(
                    np.max(np.abs(state.joint_velocities_rad_s))
                )
                joint_tracking_error_rad = float(
                    np.max(
                        np.abs(
                            state.joint_positions_rad
                            - safe_command.positions_rad
                        )
                    )
                )
                loaded_foot_slip_speed_m_s = _loaded_foot_slip_speed_m_s(
                    state,
                    load_threshold_n=support_config.min_touchdown_force_n,
                )
                capture_point_error_m = _capture_point_error_m(
                    balance_controller
                )
                both_feet_loaded = bool(
                    state.right_foot_normal_force_n
                    >= support_config.min_touchdown_force_n
                    and state.left_foot_normal_force_n
                    >= support_config.min_touchdown_force_n
                    and (
                        state.right_foot_normal_force_n
                        + state.left_foot_normal_force_n
                    )
                    >= support_config.min_touchdown_total_force_n
                )
                double_support = bool(
                    not support_enabled
                    or support_machine.phase is SupportPhase.DOUBLE_SUPPORT
                )
                quiet_checks = {
                    "double_support": double_support,
                    "both_feet_loaded": both_feet_loaded,
                    "height": height_m >= 0.70,
                    "tilt": tilt_rad <= stable_tilt_limit_rad,
                    "angular_speed": angular_speed
                    <= support_config.start_max_angular_speed_rad_s,
                    "base_linear_speed": base_linear_speed_m_s
                    <= SETTLING_MAX_BASE_LINEAR_SPEED_M_S,
                    "joint_speed": joint_speed_rad_s
                    <= SETTLING_MAX_JOINT_SPEED_RAD_S,
                    "joint_tracking_error": joint_tracking_error_rad
                    <= SETTLING_MAX_JOINT_TRACKING_ERROR_RAD,
                    "loaded_foot_slip": loaded_foot_slip_speed_m_s
                    <= SETTLING_MAX_LOADED_FOOT_SLIP_SPEED_M_S,
                    "capture_point_error": capture_point_error_m
                    <= SETTLING_MAX_CAPTURE_POINT_ERROR_M,
                }
                quiet = all(quiet_checks.values())
                if not quiet:
                    settling_last_rejection_reasons = tuple(
                        name for name, passed in quiet_checks.items() if not passed
                    )
                if quiet:
                    if settling_stable_s <= 1e-12:
                        settling_maximum_base_linear_speed_m_s = (
                            base_linear_speed_m_s
                        )
                        settling_maximum_joint_speed_rad_s = joint_speed_rad_s
                        settling_maximum_joint_tracking_error_rad = (
                            joint_tracking_error_rad
                        )
                        settling_maximum_loaded_foot_slip_speed_m_s = (
                            loaded_foot_slip_speed_m_s
                        )
                        settling_maximum_capture_point_error_m = (
                            capture_point_error_m
                        )
                    else:
                        settling_maximum_base_linear_speed_m_s = max(
                            settling_maximum_base_linear_speed_m_s,
                            base_linear_speed_m_s,
                        )
                        settling_maximum_joint_speed_rad_s = max(
                            settling_maximum_joint_speed_rad_s,
                            joint_speed_rad_s,
                        )
                        settling_maximum_joint_tracking_error_rad = max(
                            settling_maximum_joint_tracking_error_rad,
                            joint_tracking_error_rad,
                        )
                        settling_maximum_loaded_foot_slip_speed_m_s = max(
                            settling_maximum_loaded_foot_slip_speed_m_s,
                            loaded_foot_slip_speed_m_s,
                        )
                        settling_maximum_capture_point_error_m = max(
                            settling_maximum_capture_point_error_m,
                            capture_point_error_m,
                        )
                    settling_stable_s += timestep_s
                else:
                    settling_stable_s = 0.0
                    settling_maximum_base_linear_speed_m_s = 0.0
                    settling_maximum_joint_speed_rad_s = 0.0
                    settling_maximum_joint_tracking_error_rad = 0.0
                    settling_maximum_loaded_foot_slip_speed_m_s = 0.0
                    settling_maximum_capture_point_error_m = 0.0
                final_base_height = height_m
                final_simulation_time = float(state.simulation_time_s)
                if settling_stable_s + 1e-12 >= args.settle_seconds:
                    settling_completed = True
                    break
                if fall_detected:
                    LOGGER.warning(
                        "fall threshold crossed during settling at t=%.3f: "
                        "base_z=%.3f tilt_deg=%.2f",
                        state.simulation_time_s,
                        height_m,
                        np.degrees(tilt_rad),
                    )
                    break
            if not settling_completed:
                LOGGER.warning(
                    "same-simulation settling did not complete: stable=%.3f s "
                    "elapsed=%.3f s phase=%s rejected_by=%s",
                    settling_stable_s,
                    settling_elapsed_s,
                    support_machine.phase.value,
                    ",".join(settling_last_rejection_reasons) or "unknown",
                )
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
        minimum_base_height_m=minimum_base_height,
        settling_requested_s=float(args.settle_seconds),
        settling_elapsed_s=settling_elapsed_s,
        settling_stable_s=settling_stable_s,
        settling_completed=settling_completed,
        settling_minimum_base_height_m=settling_minimum_base_height,
        settling_maximum_tilt_rad=settling_maximum_tilt_rad,
        maximum_non_foot_ground_contacts=maximum_non_foot_ground_contacts,
        maximum_loaded_foot_slip_speed_m_s=(
            foot_contact_telemetry.maximum_loaded_foot_slip_speed_m_s
        ),
        right_foot_slip_distance_m=(
            foot_contact_telemetry.right_foot_slip_distance_m
        ),
        left_foot_slip_distance_m=(
            foot_contact_telemetry.left_foot_slip_distance_m
        ),
        maximum_swing_foot_impact_speed_m_s=(
            foot_contact_telemetry.maximum_swing_foot_impact_speed_m_s
        ),
        maximum_swing_foot_impact_force_n=(
            foot_contact_telemetry.maximum_swing_foot_impact_force_n
        ),
        maximum_swing_foot_contact_impulse_n_s=(
            foot_contact_telemetry.maximum_swing_foot_contact_impulse_n_s
        ),
        settling_maximum_base_linear_speed_m_s=(
            settling_maximum_base_linear_speed_m_s
        ),
        maximum_swing_foot_precontact_vertical_speed_m_s=(
            foot_contact_telemetry.maximum_swing_foot_precontact_vertical_speed_m_s
        ),
        settling_maximum_joint_speed_rad_s=settling_maximum_joint_speed_rad_s,
        settling_maximum_joint_tracking_error_rad=(
            settling_maximum_joint_tracking_error_rad
        ),
        settling_maximum_loaded_foot_slip_speed_m_s=(
            settling_maximum_loaded_foot_slip_speed_m_s
        ),
        settling_maximum_capture_point_error_m=(
            settling_maximum_capture_point_error_m
        ),
        swing_foot_contact_episodes=(
            foot_contact_telemetry.swing_foot_contact_episodes
        ),
        settling_observation_count=settling_observation_count,
        support_abort_count=support_abort_telemetry.count,
        support_abort_reasons=tuple(support_abort_telemetry.reasons),
        calibration_mode=calibration_mode,
        calibration_source_path=calibration_source_path,
        calibration_source_sha256=calibration_source_sha256,
        calibration_frame_index=calibration_frame_index,
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
        f"left_clearance_m={stats.maximum_left_foot_clearance_m:.4f} "
        f"min_base_z_m={stats.minimum_base_height_m:.4f} "
        f"settled={int(stats.settling_completed)} "
        f"settle_stable_s={stats.settling_stable_s:.3f}/"
        f"{stats.settling_requested_s:.3f} "
        f"nonfoot_contacts={stats.maximum_non_foot_ground_contacts} "
        f"max_loaded_slip_m_s={stats.maximum_loaded_foot_slip_speed_m_s:.4f} "
        f"max_swing_precontact_vertical_m_s="
        f"{stats.maximum_swing_foot_precontact_vertical_speed_m_s:.4f} "
        f"max_swing_impact_Ns={stats.maximum_swing_foot_contact_impulse_n_s:.4f}"
    )
    if stats.fell:
        LOGGER.error("free-base safety acceptance failed: a fall invariant was crossed")
        return 3
    if stats.support_abort_count > 0:
        LOGGER.error(
            "free-base safety acceptance failed: support aborts=%d reasons=%s",
            stats.support_abort_count,
            ",".join(stats.support_abort_reasons),
        )
        return 3
    if stats.settling_requested_s > 0.0 and not stats.settling_completed:
        LOGGER.error("free-base safety acceptance failed: settling did not complete")
        return 3
    if stats.robot_output_enabled and (
        stats.robot_last_error is not None
        or stats.robot_commands_sent == 0
        or stats.robot_commands_sent != stats.robot_send_attempts
    ):
        if stats.robot_commands_sent == 0:
            LOGGER.error(
                "legacy robot output was requested, but no command was delivered"
            )
        elif stats.robot_commands_sent != stats.robot_send_attempts:
            LOGGER.error(
                "legacy robot output lost %d of %d attempted commands",
                stats.robot_send_attempts - stats.robot_commands_sent,
                stats.robot_send_attempts,
            )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
