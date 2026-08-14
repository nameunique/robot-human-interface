"""Run reproducible end-to-end free-base stability acceptance replays.

The default suite sends every bundled research replay through the production
``run_teleop`` path with MediaPipe, constrained IK, the balance/support
controllers, and the free-base MuJoCo model.  The resulting JSON is intended as
machine-readable evidence: failures are written to the report and make the
process return a non-zero exit code.

After each finite video, ``run_teleop`` returns the robot to double support and
requires five quiet seconds on the same ``HumanoidSimulation`` by default.  The
settling interval has its own completion, height, tilt, and duration gates.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import subprocess
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np

from robot_human_interface.app.teleop import (
    BUNDLED_VIDEO_PATHS,
    SETTLING_MAX_BASE_LINEAR_SPEED_M_S,
    SETTLING_MAX_CAPTURE_POINT_ERROR_M,
    SETTLING_MAX_JOINT_SPEED_RAD_S,
    SETTLING_MAX_JOINT_TRACKING_ERROR_RAD,
    SETTLING_MAX_LOADED_FOOT_SLIP_SPEED_M_S,
    TeleopStats,
    build_parser as build_teleop_parser,
    run_teleop,
)
from robot_human_interface.simulation import HumanoidSimulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "freebase-stability.json"
RUNTIME_INPUTS = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "config" / "balance.yaml",
    PROJECT_ROOT / "config" / "camera.yaml",
    PROJECT_ROOT / "config" / "joints.yaml",
    PROJECT_ROOT / "config" / "retargeting.yaml",
    PROJECT_ROOT / "models" / "humanoid" / "robot.xml",
    PROJECT_ROOT / "models" / "humanoid" / "scene_free.xml",
    PROJECT_ROOT / "assets" / "models" / "pose_landmarker_full.task",
    PROJECT_ROOT / "src" / "robot_human_interface" / "app" / "teleop.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "camera" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "camera" / "sources.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "control" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "control" / "human_intent.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "control" / "standing.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "control" / "support.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "pose" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "pose" / "calibration.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "pose" / "mediapipe_tasks.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "retargeting" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "retargeting" / "fidelity.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "retargeting" / "geometry.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "retargeting" / "mujoco_ik.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "retargeting" / "retargeter.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "simulation" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "simulation" / "humanoid.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "simulation" / "types.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "filtering.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "transforms.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "types.py",
)

ARM_INDICES = np.arange(0, 6, dtype=np.int64)
LEG_INDICES = np.arange(6, 18, dtype=np.int64)
HEAD_INDICES = np.arange(18, 20, dtype=np.int64)
MOTION_GROUPS = {
    "arms": ARM_INDICES,
    "legs": LEG_INDICES,
    "head": HEAD_INDICES,
}


@dataclass(frozen=True, slots=True)
class MotionExpectation:
    """Observable motion required from one semantic replay."""

    minimum_raw_command_span_deg: float = 10.0
    minimum_safe_command_span_deg: float = 5.0
    minimum_safe_arm_span_deg: float = 0.0
    minimum_safe_leg_span_deg: float = 0.0
    minimum_safe_head_span_deg: float = 0.0
    minimum_actual_arm_span_deg: float = 0.0
    minimum_actual_leg_span_deg: float = 0.0
    minimum_actual_head_span_deg: float = 0.0
    right_swing_completed: bool | None = None
    left_swing_completed: bool | None = None
    minimum_right_clearance_m: float = 0.0
    minimum_left_clearance_m: float = 0.0
    minimum_support_transitions: int = 0


@dataclass(frozen=True, slots=True)
class ClipSpec:
    """One named replay and its acceptance contract."""

    name: str
    source: str
    path: Path
    demo_video: str | None
    expected_frames: int
    expectation: MotionExpectation
    calibration_video: Path | None = None
    calibration_frame: int | None = None

    def __post_init__(self) -> None:
        if (self.calibration_video is None) != (self.calibration_frame is None):
            raise ValueError("calibration_video and calibration_frame must be paired")
        if self.calibration_frame is not None and self.calibration_frame < 0:
            raise ValueError("calibration_frame must be non-negative")


@dataclass(frozen=True, slots=True)
class StabilityThresholds:
    """Suite-wide safety and synchronization gates."""

    minimum_base_height_m: float = 0.80
    minimum_final_base_height_m: float = 0.88
    maximum_tilt_deg: float = 20.0
    maximum_media_sync_error_s: float = 0.003
    maximum_stale_fraction: float = 0.10
    minimum_skeleton_fraction: float = 0.85
    maximum_loaded_foot_slip_speed_m_s: float = 0.15
    maximum_foot_slip_distance_m: float = 0.075
    maximum_swing_precontact_vertical_speed_m_s: float = 0.50
    maximum_swing_impact_force_n: float = 60.0
    maximum_swing_contact_impulse_n_s: float = 9.0
    maximum_swing_peak_force_bodyweights: float = 2.25
    maximum_swing_contact_impulse_weight_s: float = 0.30
    settling_maximum_base_linear_speed_m_s: float = (
        SETTLING_MAX_BASE_LINEAR_SPEED_M_S
    )
    settling_maximum_joint_speed_rad_s: float = SETTLING_MAX_JOINT_SPEED_RAD_S
    settling_maximum_joint_tracking_error_rad: float = (
        SETTLING_MAX_JOINT_TRACKING_ERROR_RAD
    )
    settling_maximum_loaded_foot_slip_speed_m_s: float = (
        SETTLING_MAX_LOADED_FOOT_SLIP_SPEED_M_S
    )
    settling_maximum_capture_point_error_m: float = (
        SETTLING_MAX_CAPTURE_POINT_ERROR_M
    )


CANONICAL_SETTLING_SECONDS = 5.0
_MINIMUM_THRESHOLD_FIELDS = (
    "minimum_base_height_m",
    "minimum_final_base_height_m",
    "minimum_skeleton_fraction",
)
_MAXIMUM_THRESHOLD_FIELDS = (
    "maximum_tilt_deg",
    "maximum_media_sync_error_s",
    "maximum_stale_fraction",
    "maximum_loaded_foot_slip_speed_m_s",
    "maximum_foot_slip_distance_m",
    "maximum_swing_precontact_vertical_speed_m_s",
    "maximum_swing_impact_force_n",
    "maximum_swing_contact_impulse_n_s",
    "maximum_swing_peak_force_bodyweights",
    "maximum_swing_contact_impulse_weight_s",
    "settling_maximum_base_linear_speed_m_s",
    "settling_maximum_joint_speed_rad_s",
    "settling_maximum_joint_tracking_error_rad",
    "settling_maximum_loaded_foot_slip_speed_m_s",
    "settling_maximum_capture_point_error_m",
)


@dataclass(slots=True)
class _SimulationRecorder:
    safe_samples_rad: list[np.ndarray] = field(default_factory=list)
    actual_samples_rad: list[np.ndarray] = field(default_factory=list)
    timestep_s: float | None = None
    equality_constraint_count: int | None = None
    base_joint_type: str | None = None
    robot_weight_n: float | None = None
    simulation_instances: list[object] = field(default_factory=list, repr=False)
    data_instances: list[object] = field(default_factory=list, repr=False)

    def record_instance(self, simulation: HumanoidSimulation) -> None:
        """Retain objects so identity evidence cannot be defeated by id reuse."""

        if not any(item is simulation for item in self.simulation_instances):
            self.simulation_instances.append(simulation)
        data = simulation.data
        if not any(item is data for item in self.data_instances):
            self.data_instances.append(data)

    @property
    def same_simulation(self) -> bool:
        return len(self.simulation_instances) == len(self.data_instances) == 1

    def record_model(self, simulation: HumanoidSimulation) -> None:
        self.record_instance(simulation)
        if self.equality_constraint_count is not None:
            return
        import mujoco

        self.equality_constraint_count = int(simulation.model.neq)
        self.robot_weight_n = float(np.sum(simulation.model.body_mass) * 9.81)
        joint_id = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free"
        )
        if joint_id < 0:
            self.base_joint_type = "missing"
            return
        joint_type = int(simulation.model.jnt_type[joint_id])
        self.base_joint_type = (
            "free"
            if joint_type == int(mujoco.mjtJoint.mjJNT_FREE)
            else mujoco.mjtJoint(joint_type).name.removeprefix("mjJNT_").lower()
        )

    def record_safe_command(self, positions_rad: np.ndarray) -> None:
        self.safe_samples_rad.append(
            np.asarray(positions_rad, dtype=np.float64).copy()
        )

    def record_state(self, state: object, *, timestep_s: float) -> None:
        self.timestep_s = timestep_s
        self.actual_samples_rad.append(
            np.asarray(state.joint_positions_rad, dtype=np.float64).copy()
        )

    @staticmethod
    def _motion_summary(
        samples: Sequence[np.ndarray], *, excluded_tail_steps: int
    ) -> dict[str, dict[str, float | int | None]]:
        summary: dict[str, dict[str, float | int | None]] = {}
        retained = (
            samples[:-excluded_tail_steps]
            if excluded_tail_steps > 0
            else samples
        )
        values = np.asarray(retained, dtype=np.float64)
        for name, indices in MOTION_GROUPS.items():
            if values.size == 0:
                summary[name] = {
                    "maximum_span_deg": None,
                    "active_joints_over_2deg": 0,
                }
                continue
            span_deg = np.degrees(
                np.max(values[:, indices], axis=0)
                - np.min(values[:, indices], axis=0)
            )
            summary[name] = {
                "maximum_span_deg": float(np.max(span_deg)),
                "active_joints_over_2deg": int(np.count_nonzero(span_deg > 2.0)),
            }
        return summary

    def motion(self, *, settling_elapsed_s: float) -> dict[str, object]:
        excluded_steps = 0
        if self.timestep_s is not None and self.timestep_s > 0.0:
            excluded_steps = int(round(settling_elapsed_s / self.timestep_s))
        safe_excluded = min(excluded_steps, len(self.safe_samples_rad))
        actual_excluded = min(excluded_steps, len(self.actual_samples_rad))
        return {
            "scope": "input video physics; same-simulation settling tail excluded",
            "settling_steps_excluded": min(safe_excluded, actual_excluded),
            "safe_command": self._motion_summary(
                self.safe_samples_rad, excluded_tail_steps=safe_excluded
            ),
            "actual_joint": self._motion_summary(
                self.actual_samples_rad, excluded_tail_steps=actual_excluded
            ),
        }


@contextmanager
def _record_simulation() -> Iterator[_SimulationRecorder]:
    """Observe the simulation owned by ``run_teleop`` without changing it."""

    recorder = _SimulationRecorder()
    original_apply = HumanoidSimulation.apply_joint_command
    original_step = HumanoidSimulation.step

    def recorded_apply(
        simulation: HumanoidSimulation, command: object, *args: object, **kwargs: object
    ) -> object:
        recorder.record_model(simulation)
        recorder.record_safe_command(
            np.asarray(command.positions_rad, dtype=np.float64)
        )
        return original_apply(simulation, command, *args, **kwargs)

    def recorded_step(
        simulation: HumanoidSimulation, *args: object, **kwargs: object
    ) -> object:
        recorder.record_model(simulation)
        state = original_step(simulation, *args, **kwargs)
        recorder.record_state(
            state, timestep_s=float(simulation.model.opt.timestep)
        )
        return state

    HumanoidSimulation.apply_joint_command = recorded_apply  # type: ignore[method-assign]
    HumanoidSimulation.step = recorded_step  # type: ignore[method-assign]
    try:
        yield recorder
    finally:
        HumanoidSimulation.apply_joint_command = original_apply  # type: ignore[method-assign]
        HumanoidSimulation.step = original_step  # type: ignore[method-assign]


def default_clips() -> tuple[ClipSpec, ...]:
    """Return the stable, deliberately ordered default acceptance matrix."""

    external = PROJECT_ROOT / "assets" / "videos" / "external"
    return (
        ClipSpec(
            "slow-balance",
            "mp4",
            BUNDLED_VIDEO_PATHS["slow-balance"],
            "slow-balance",
            1961,
            MotionExpectation(
                minimum_raw_command_span_deg=30.0,
                minimum_safe_command_span_deg=30.0,
                minimum_safe_arm_span_deg=30.0,
                minimum_safe_leg_span_deg=20.0,
                minimum_safe_head_span_deg=5.0,
                minimum_actual_arm_span_deg=20.0,
                minimum_actual_leg_span_deg=15.0,
                minimum_actual_head_span_deg=5.0,
                right_swing_completed=True,
                left_swing_completed=True,
                minimum_right_clearance_m=0.020,
                minimum_left_clearance_m=0.025,
                minimum_support_transitions=8,
            ),
        ),
        ClipSpec(
            "jumping-jacks",
            "mp4",
            BUNDLED_VIDEO_PATHS["jumping-jacks"],
            "jumping-jacks",
            194,
            MotionExpectation(
                minimum_raw_command_span_deg=20.0,
                minimum_safe_command_span_deg=10.0,
                minimum_safe_arm_span_deg=15.0,
                minimum_safe_leg_span_deg=5.0,
                minimum_actual_arm_span_deg=10.0,
                minimum_actual_leg_span_deg=3.0,
                right_swing_completed=False,
                left_swing_completed=False,
            ),
            calibration_video=BUNDLED_VIDEO_PATHS["jumping-jacks"],
            calibration_frame=2,
        ),
        ClipSpec(
            "arm-circles",
            "replay",
            external / "dvids_arm_circles.mp4",
            None,
            796,
            MotionExpectation(
                minimum_raw_command_span_deg=60.0,
                minimum_safe_command_span_deg=30.0,
                minimum_safe_arm_span_deg=30.0,
                minimum_actual_arm_span_deg=25.0,
                right_swing_completed=False,
                left_swing_completed=False,
            ),
        ),
        ClipSpec(
            "frontal-leg-swing",
            "replay",
            external / "dvids_frontal_leg_swing.mp4",
            None,
            836,
            MotionExpectation(
                minimum_raw_command_span_deg=30.0,
                minimum_safe_command_span_deg=15.0,
                minimum_safe_leg_span_deg=15.0,
                minimum_actual_leg_span_deg=10.0,
                right_swing_completed=True,
                left_swing_completed=False,
                minimum_right_clearance_m=0.020,
                minimum_support_transitions=4,
            ),
            calibration_video=external / "dvids_arm_circles.mp4",
            calibration_frame=29,
        ),
        ClipSpec(
            "stationary-squat",
            "replay",
            external / "dvids_stationary_squat.mp4",
            None,
            817,
            MotionExpectation(
                minimum_raw_command_span_deg=30.0,
                minimum_safe_command_span_deg=10.0,
                minimum_safe_leg_span_deg=5.0,
                minimum_actual_leg_span_deg=3.0,
                right_swing_completed=False,
                left_swing_completed=False,
            ),
        ),
        ClipSpec(
            "trunk-circles",
            "replay",
            external / "dvids_trunk_circles.mp4",
            None,
            867,
            MotionExpectation(
                minimum_raw_command_span_deg=30.0,
                minimum_safe_command_span_deg=10.0,
                minimum_safe_leg_span_deg=5.0,
                minimum_safe_head_span_deg=5.0,
                minimum_actual_leg_span_deg=3.0,
                minimum_actual_head_span_deg=5.0,
                right_swing_completed=False,
                left_swing_completed=False,
            ),
            calibration_video=external / "dvids_arm_circles.mp4",
            calibration_frame=29,
        ),
    )


def _project_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_input_hashes(clips: Sequence[ClipSpec]) -> dict[str, str]:
    paths = {
        *RUNTIME_INPUTS,
        *(clip.path.expanduser().resolve() for clip in clips),
        *(
            clip.calibration_video.expanduser().resolve()
            for clip in clips
            if clip.calibration_video is not None
        ),
    }
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"runtime acceptance inputs are missing: {missing}")
    return {
        _project_path(path): _sha256(path)
        for path in sorted(paths, key=lambda item: str(item).lower())
    }


def _teleop_arguments(clip: ClipSpec, settling_seconds: float) -> list[str]:
    settling_timeout_s = max(20.0, settling_seconds + 15.0)
    arguments = [
        "--source",
        clip.source,
        "--headless",
        "--free-base",
        "--balance-controller",
        "--retargeting",
        "ik",
        "--settle-seconds",
        f"{settling_seconds:g}",
        "--settle-timeout-s",
        f"{settling_timeout_s:g}",
    ]
    if clip.source == "mp4":
        if clip.demo_video is None:
            raise ValueError(f"bundled clip lacks demo_video: {clip.name}")
        arguments.extend(("--demo-video", clip.demo_video))
    elif clip.source == "replay":
        arguments.extend(("--video-path", str(clip.path)))
    else:
        raise ValueError(f"unsupported acceptance source {clip.source!r}: {clip.name}")
    if clip.calibration_video is not None:
        arguments.extend(
            (
                "--calibration-video",
                str(clip.calibration_video),
                "--calibration-frame",
                str(clip.calibration_frame),
            )
        )
    return arguments


def _settling_metrics(
    stats: TeleopStats, recorder: _SimulationRecorder
) -> dict[str, object]:
    return {
        "requested_duration_s": float(stats.settling_requested_s),
        "elapsed_s": float(stats.settling_elapsed_s),
        "stable_s": float(stats.settling_stable_s),
        "completed": bool(stats.settling_completed),
        "status": "passed" if stats.settling_completed else "failed",
        "same_simulation": recorder.same_simulation,
        "simulation_instance_count": len(recorder.simulation_instances),
        "data_instance_count": len(recorder.data_instances),
        "minimum_base_height_m": float(stats.settling_minimum_base_height_m),
        "maximum_tilt_deg": float(np.degrees(stats.settling_maximum_tilt_rad)),
        "maximum_base_linear_speed_m_s": float(
            stats.settling_maximum_base_linear_speed_m_s
        ),
        "maximum_joint_speed_rad_s": float(
            stats.settling_maximum_joint_speed_rad_s
        ),
        "maximum_joint_tracking_error_rad": float(
            stats.settling_maximum_joint_tracking_error_rad
        ),
        "maximum_loaded_foot_slip_speed_m_s": float(
            stats.settling_maximum_loaded_foot_slip_speed_m_s
        ),
        "maximum_capture_point_error_m": float(
            stats.settling_maximum_capture_point_error_m
        ),
        "observation_count": int(stats.settling_observation_count),
    }


def _stats_metrics(
    stats: TeleopStats, recorder: _SimulationRecorder
) -> dict[str, object]:
    frames = int(stats.frames)
    input_simulation_time_s = float(
        stats.simulation_time_s - stats.settling_elapsed_s
    )
    robot_weight_n = recorder.robot_weight_n
    normalized_peak_force = (
        float(stats.maximum_swing_foot_impact_force_n / robot_weight_n)
        if robot_weight_n is not None and robot_weight_n > 0.0
        else math.nan
    )
    normalized_contact_impulse = (
        float(stats.maximum_swing_foot_contact_impulse_n_s / robot_weight_n)
        if robot_weight_n is not None and robot_weight_n > 0.0
        else math.nan
    )
    return {
        "base_mode": stats.base_mode,
        "equality_constraint_count": recorder.equality_constraint_count,
        "base_joint_type": recorder.base_joint_type,
        "frames": frames,
        "skeleton_frames": int(stats.skeleton_frames),
        "skeleton_fraction": (
            float(stats.skeleton_frames / frames) if frames > 0 else 0.0
        ),
        "stale_commands": int(stats.stale_commands),
        "stale_fraction": float(stats.stale_commands / frames) if frames > 0 else 1.0,
        "minimum_base_height_m": float(stats.minimum_base_height_m),
        "final_base_height_m": float(stats.final_base_height_m),
        "maximum_tilt_deg": float(np.degrees(stats.maximum_tilt_rad)),
        "fell": bool(stats.fell),
        "support_transitions": int(stats.support_transitions),
        "support_abort_count": int(stats.support_abort_count),
        "support_abort_reasons": list(stats.support_abort_reasons),
        "right_swing_completed": bool(stats.right_swing_completed),
        "left_swing_completed": bool(stats.left_swing_completed),
        "maximum_right_foot_clearance_m": float(
            stats.maximum_right_foot_clearance_m
        ),
        "maximum_left_foot_clearance_m": float(
            stats.maximum_left_foot_clearance_m
        ),
        "simulation_time_s": float(stats.simulation_time_s),
        "input_simulation_time_s": input_simulation_time_s,
        "media_time_s": float(stats.media_time_s),
        "media_sync_error_s": float(
            abs(input_simulation_time_s - stats.media_time_s)
        ),
        "raw_command_span_deg": float(np.degrees(stats.command_span_rad)),
        "safe_command_span_deg": float(np.degrees(stats.safe_command_span_rad)),
        "maximum_non_foot_ground_contacts": int(
            stats.maximum_non_foot_ground_contacts
        ),
        "maximum_loaded_foot_slip_speed_m_s": float(
            stats.maximum_loaded_foot_slip_speed_m_s
        ),
        "right_foot_slip_distance_m": float(stats.right_foot_slip_distance_m),
        "left_foot_slip_distance_m": float(stats.left_foot_slip_distance_m),
        "maximum_swing_foot_impact_speed_m_s": float(
            stats.maximum_swing_foot_impact_speed_m_s
        ),
        "maximum_swing_foot_precontact_vertical_speed_m_s": float(
            stats.maximum_swing_foot_precontact_vertical_speed_m_s
        ),
        "maximum_swing_foot_impact_force_n": float(
            stats.maximum_swing_foot_impact_force_n
        ),
        "maximum_swing_foot_contact_impulse_n_s": float(
            stats.maximum_swing_foot_contact_impulse_n_s
        ),
        "robot_weight_n": robot_weight_n,
        "maximum_swing_foot_peak_force_bodyweights": normalized_peak_force,
        "maximum_swing_foot_contact_impulse_weight_s": (
            normalized_contact_impulse
        ),
        "swing_foot_contact_episodes": int(stats.swing_foot_contact_episodes),
        "motion": recorder.motion(settling_elapsed_s=stats.settling_elapsed_s),
    }


def _teleop_exit_code(stats: TeleopStats) -> int:
    """Mirror ``teleop.main`` safety status for the direct ``run_teleop`` call."""

    if stats.fell:
        return 3
    if int(getattr(stats, "support_abort_count", 0)) > 0:
        return 3
    if stats.settling_requested_s > 0.0 and not stats.settling_completed:
        return 3
    return 0


def evaluate_clip(clip: ClipSpec, settling_seconds: float) -> dict[str, object]:
    """Execute one real MediaPipe/IK/free-base replay through ``run_teleop``."""

    path = clip.path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"acceptance video does not exist: {path}")
    arguments = _teleop_arguments(clip, settling_seconds)
    parsed = build_teleop_parser().parse_args(arguments)
    with _record_simulation() as recorder:
        stats = run_teleop(parsed)
    return {
        "exit_code": _teleop_exit_code(stats),
        "video": _project_path(path),
        "video_size_bytes": path.stat().st_size,
        "video_sha256": _sha256(path),
        "teleop_arguments": arguments,
        "calibration": {
            "mode": stats.calibration_mode,
            "source": (
                None
                if stats.calibration_source_path is None
                else _project_path(Path(stats.calibration_source_path))
            ),
            "source_sha256": stats.calibration_source_sha256,
            "frame_index": stats.calibration_frame_index,
        },
        "metrics": _stats_metrics(stats, recorder),
        "settling": _settling_metrics(stats, recorder),
    }


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def _motion_span(metrics: Mapping[str, object], source: str, group: str) -> object:
    motion = metrics.get("motion")
    if not isinstance(motion, Mapping):
        return None
    source_metrics = motion.get(source)
    if not isinstance(source_metrics, Mapping):
        return None
    group_metrics = source_metrics.get(group)
    if not isinstance(group_metrics, Mapping):
        return None
    return group_metrics.get("maximum_span_deg")


def _gate(
    name: str, passed: bool, observed: object, required: object
) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def assess_clip(
    clip: ClipSpec,
    evaluation: Mapping[str, object],
    thresholds: StabilityThresholds,
    *,
    settling_seconds: float,
) -> dict[str, object]:
    """Apply explicit safety, timing, and semantic-motion acceptance gates."""

    metrics = evaluation.get("metrics")
    if not isinstance(metrics, Mapping):
        return {
            **dict(evaluation),
            "name": clip.name,
            "expectation": asdict(clip.expectation),
            "gates": [_gate("evaluation_completed", False, False, True)],
            "passed": False,
        }

    gates: list[dict[str, object]] = []

    def equals(name: str, expected: object) -> None:
        observed = metrics.get(name)
        gates.append(_gate(name, observed == expected, observed, expected))

    def at_least(name: str, minimum: float) -> None:
        observed = metrics.get(name)
        gates.append(
            _gate(
                name,
                _finite_number(observed) and float(observed) >= minimum,
                observed,
                {"minimum": minimum},
            )
        )

    def at_most(name: str, maximum: float) -> None:
        observed = metrics.get(name)
        gates.append(
            _gate(
                name,
                _finite_number(observed) and float(observed) <= maximum,
                observed,
                {"maximum": maximum},
            )
        )

    calibration = evaluation.get("calibration")
    expected_calibration_path = (
        None
        if clip.calibration_video is None
        else _project_path(clip.calibration_video)
    )
    expected_calibration_hash = (
        None
        if clip.calibration_video is None
        else _sha256(clip.calibration_video.expanduser().resolve())
    )
    expected_calibration_mode = (
        "automatic_window"
        if clip.calibration_video is None
        else "explicit_replay_frame"
    )
    gates.append(
        _gate(
            "calibration_evidence",
            isinstance(calibration, Mapping),
            isinstance(calibration, Mapping),
            True,
        )
    )
    for name, expected in (
        ("calibration_mode", expected_calibration_mode),
        ("calibration_source", expected_calibration_path),
        ("calibration_source_sha256", expected_calibration_hash),
        ("calibration_frame_index", clip.calibration_frame),
    ):
        field = name.removeprefix("calibration_")
        observed = calibration.get(field) if isinstance(calibration, Mapping) else None
        gates.append(_gate(name, observed == expected, observed, expected))

    expected_video_path = _project_path(clip.path)
    observed_video_path = evaluation.get("video")
    gates.append(
        _gate(
            "video_path",
            observed_video_path == expected_video_path,
            observed_video_path,
            expected_video_path,
        )
    )
    expected_video_size = clip.path.expanduser().resolve().stat().st_size
    observed_video_size = evaluation.get("video_size_bytes")
    gates.append(
        _gate(
            "video_size_bytes",
            observed_video_size == expected_video_size,
            observed_video_size,
            expected_video_size,
        )
    )
    expected_video_hash = _sha256(clip.path.expanduser().resolve())
    observed_video_hash = evaluation.get("video_sha256")
    gates.append(
        _gate(
            "video_sha256",
            observed_video_hash == expected_video_hash,
            observed_video_hash,
            expected_video_hash,
        )
    )

    equals("base_mode", "free")
    equals("equality_constraint_count", 0)
    equals("base_joint_type", "free")
    observed_exit_code = evaluation.get("exit_code")
    gates.append(
        _gate("exit_code", observed_exit_code == 0, observed_exit_code, 0)
    )
    equals("fell", False)
    observed_frames = metrics.get("frames")
    gates.append(
        _gate(
            "frames",
            observed_frames == clip.expected_frames,
            observed_frames,
            clip.expected_frames,
        )
    )
    at_least("skeleton_fraction", thresholds.minimum_skeleton_fraction)
    at_most("stale_fraction", thresholds.maximum_stale_fraction)
    at_least("minimum_base_height_m", thresholds.minimum_base_height_m)
    at_least("final_base_height_m", thresholds.minimum_final_base_height_m)
    at_most("maximum_tilt_deg", thresholds.maximum_tilt_deg)
    at_most("media_sync_error_s", thresholds.maximum_media_sync_error_s)
    equals("maximum_non_foot_ground_contacts", 0)
    equals("support_abort_count", 0)
    observed_abort_reasons = metrics.get("support_abort_reasons")
    gates.append(
        _gate(
            "support_abort_reasons",
            observed_abort_reasons == [],
            observed_abort_reasons,
            [],
        )
    )
    at_most(
        "maximum_loaded_foot_slip_speed_m_s",
        thresholds.maximum_loaded_foot_slip_speed_m_s,
    )
    at_most(
        "right_foot_slip_distance_m", thresholds.maximum_foot_slip_distance_m
    )
    at_most(
        "left_foot_slip_distance_m", thresholds.maximum_foot_slip_distance_m
    )
    at_most(
        "maximum_swing_foot_precontact_vertical_speed_m_s",
        thresholds.maximum_swing_precontact_vertical_speed_m_s,
    )
    at_most(
        "maximum_swing_foot_impact_force_n",
        thresholds.maximum_swing_impact_force_n,
    )
    at_most(
        "maximum_swing_foot_contact_impulse_n_s",
        thresholds.maximum_swing_contact_impulse_n_s,
    )
    at_most(
        "maximum_swing_foot_peak_force_bodyweights",
        thresholds.maximum_swing_peak_force_bodyweights,
    )
    at_most(
        "maximum_swing_foot_contact_impulse_weight_s",
        thresholds.maximum_swing_contact_impulse_weight_s,
    )

    finite_fields = (
        "minimum_base_height_m",
        "final_base_height_m",
        "maximum_tilt_deg",
        "maximum_right_foot_clearance_m",
        "maximum_left_foot_clearance_m",
        "simulation_time_s",
        "input_simulation_time_s",
        "media_time_s",
        "media_sync_error_s",
        "raw_command_span_deg",
        "safe_command_span_deg",
        "maximum_loaded_foot_slip_speed_m_s",
        "right_foot_slip_distance_m",
        "left_foot_slip_distance_m",
        "maximum_swing_foot_impact_speed_m_s",
        "maximum_swing_foot_precontact_vertical_speed_m_s",
        "maximum_swing_foot_impact_force_n",
        "maximum_swing_foot_contact_impulse_n_s",
        "robot_weight_n",
        "maximum_swing_foot_peak_force_bodyweights",
        "maximum_swing_foot_contact_impulse_weight_s",
    )
    invalid = [name for name in finite_fields if not _finite_number(metrics.get(name))]
    gates.append(_gate("finite_metrics", not invalid, invalid, []))

    settling = evaluation.get("settling")
    if not isinstance(settling, Mapping):
        gates.append(_gate("settling_metrics", False, None, "object"))
    else:
        requested = settling.get("requested_duration_s")
        elapsed = settling.get("elapsed_s")
        stable = settling.get("stable_s")
        minimum_height = settling.get("minimum_base_height_m")
        maximum_tilt = settling.get("maximum_tilt_deg")
        maximum_base_linear_speed = settling.get(
            "maximum_base_linear_speed_m_s"
        )
        maximum_joint_speed = settling.get("maximum_joint_speed_rad_s")
        maximum_tracking_error = settling.get(
            "maximum_joint_tracking_error_rad"
        )
        maximum_settling_slip = settling.get(
            "maximum_loaded_foot_slip_speed_m_s"
        )
        maximum_capture_point_error = settling.get(
            "maximum_capture_point_error_m"
        )
        observation_count = settling.get("observation_count")
        settling_finite = all(
            _finite_number(value)
            for value in (
                requested,
                elapsed,
                stable,
                minimum_height,
                maximum_tilt,
                maximum_base_linear_speed,
                maximum_joint_speed,
                maximum_tracking_error,
                maximum_settling_slip,
                maximum_capture_point_error,
            )
        )
        gates.extend(
            (
                _gate(
                    "settling_same_simulation",
                    settling.get("same_simulation") is True,
                    settling.get("same_simulation"),
                    True,
                ),
                _gate(
                    "settling_simulation_instance_count",
                    settling.get("simulation_instance_count") == 1,
                    settling.get("simulation_instance_count"),
                    1,
                ),
                _gate(
                    "settling_data_instance_count",
                    settling.get("data_instance_count") == 1,
                    settling.get("data_instance_count"),
                    1,
                ),
                _gate(
                    "settling_status",
                    settling.get("status") == "passed",
                    settling.get("status"),
                    "passed",
                ),
                _gate(
                    "settling_completed",
                    settling.get("completed") is True,
                    settling.get("completed"),
                    True,
                ),
                _gate(
                    "settling_observations_present",
                    isinstance(observation_count, int)
                    and not isinstance(observation_count, bool)
                    and observation_count > 0,
                    observation_count,
                    {"minimum": 1},
                ),
                _gate(
                    "settling_requested_duration_s",
                    _finite_number(requested)
                    and abs(float(requested) - settling_seconds) <= 1e-9,
                    requested,
                    settling_seconds,
                ),
                _gate(
                    "settling_stable_duration_s",
                    _finite_number(stable)
                    and float(stable) + 1e-9 >= settling_seconds,
                    stable,
                    {"minimum": settling_seconds},
                ),
                _gate(
                    "settling_elapsed_s",
                    _finite_number(elapsed)
                    and _finite_number(stable)
                    and float(elapsed) + 1e-9 >= float(stable),
                    elapsed,
                    {"minimum": stable},
                ),
                _gate(
                    "settling_minimum_base_height_m",
                    _finite_number(minimum_height)
                    and float(minimum_height) >= thresholds.minimum_base_height_m,
                    minimum_height,
                    {"minimum": thresholds.minimum_base_height_m},
                ),
                _gate(
                    "settling_maximum_tilt_deg",
                    _finite_number(maximum_tilt)
                    and float(maximum_tilt) <= thresholds.maximum_tilt_deg,
                    maximum_tilt,
                    {"maximum": thresholds.maximum_tilt_deg},
                ),
                _gate(
                    "settling_maximum_base_linear_speed_m_s",
                    _finite_number(maximum_base_linear_speed)
                    and float(maximum_base_linear_speed)
                    <= thresholds.settling_maximum_base_linear_speed_m_s,
                    maximum_base_linear_speed,
                    {
                        "maximum": (
                            thresholds.settling_maximum_base_linear_speed_m_s
                        )
                    },
                ),
                _gate(
                    "settling_maximum_joint_speed_rad_s",
                    _finite_number(maximum_joint_speed)
                    and float(maximum_joint_speed)
                    <= thresholds.settling_maximum_joint_speed_rad_s,
                    maximum_joint_speed,
                    {
                        "maximum": thresholds.settling_maximum_joint_speed_rad_s
                    },
                ),
                _gate(
                    "settling_maximum_joint_tracking_error_rad",
                    _finite_number(maximum_tracking_error)
                    and float(maximum_tracking_error)
                    <= thresholds.settling_maximum_joint_tracking_error_rad,
                    maximum_tracking_error,
                    {
                        "maximum": (
                            thresholds.settling_maximum_joint_tracking_error_rad
                        )
                    },
                ),
                _gate(
                    "settling_maximum_loaded_foot_slip_speed_m_s",
                    _finite_number(maximum_settling_slip)
                    and float(maximum_settling_slip)
                    <= thresholds.settling_maximum_loaded_foot_slip_speed_m_s,
                    maximum_settling_slip,
                    {
                        "maximum": (
                            thresholds.settling_maximum_loaded_foot_slip_speed_m_s
                        )
                    },
                ),
                _gate(
                    "settling_maximum_capture_point_error_m",
                    _finite_number(maximum_capture_point_error)
                    and float(maximum_capture_point_error)
                    <= thresholds.settling_maximum_capture_point_error_m,
                    maximum_capture_point_error,
                    {
                        "maximum": (
                            thresholds.settling_maximum_capture_point_error_m
                        )
                    },
                ),
                _gate("finite_settling_metrics", settling_finite, settling_finite, True),
            )
        )

    expectation = clip.expectation
    at_least("raw_command_span_deg", expectation.minimum_raw_command_span_deg)
    at_least("safe_command_span_deg", expectation.minimum_safe_command_span_deg)

    for group, minimum in (
        ("arms", expectation.minimum_safe_arm_span_deg),
        ("legs", expectation.minimum_safe_leg_span_deg),
        ("head", expectation.minimum_safe_head_span_deg),
    ):
        if minimum <= 0.0:
            continue
        observed = _motion_span(metrics, "safe_command", group)
        gates.append(
            _gate(
                f"safe_{group}_motion",
                _finite_number(observed) and float(observed) >= minimum,
                observed,
                {"minimum_span_deg": minimum},
            )
        )
    for group, minimum in (
        ("arms", expectation.minimum_actual_arm_span_deg),
        ("legs", expectation.minimum_actual_leg_span_deg),
        ("head", expectation.minimum_actual_head_span_deg),
    ):
        if minimum <= 0.0:
            continue
        observed = _motion_span(metrics, "actual_joint", group)
        gates.append(
            _gate(
                f"actual_{group}_motion",
                _finite_number(observed) and float(observed) >= minimum,
                observed,
                {"minimum_span_deg": minimum},
            )
        )

    if expectation.right_swing_completed is not None:
        equals("right_swing_completed", expectation.right_swing_completed)
    if expectation.left_swing_completed is not None:
        equals("left_swing_completed", expectation.left_swing_completed)
    expected_landings = sum(
        value is True
        for value in (
            expectation.right_swing_completed,
            expectation.left_swing_completed,
        )
    )
    if expected_landings > 0:
        at_least("swing_foot_contact_episodes", float(expected_landings))
        observed_force = metrics.get("maximum_swing_foot_impact_force_n")
        gates.append(
            _gate(
                "swing_landing_force_observed",
                _finite_number(observed_force) and float(observed_force) >= 4.0,
                observed_force,
                {"minimum": 4.0},
            )
        )
        observed_impulse = metrics.get(
            "maximum_swing_foot_contact_impulse_n_s"
        )
        gates.append(
            _gate(
                "swing_landing_impulse_observed",
                _finite_number(observed_impulse)
                and float(observed_impulse) >= 0.001,
                observed_impulse,
                {"minimum": 0.001},
            )
        )
    if expectation.minimum_right_clearance_m > 0.0:
        at_least(
            "maximum_right_foot_clearance_m",
            expectation.minimum_right_clearance_m,
        )
    if expectation.minimum_left_clearance_m > 0.0:
        at_least(
            "maximum_left_foot_clearance_m",
            expectation.minimum_left_clearance_m,
        )
    if expectation.minimum_support_transitions > 0:
        at_least(
            "support_transitions", float(expectation.minimum_support_transitions)
        )

    return {
        **dict(evaluation),
        "name": clip.name,
        "expectation": asdict(expectation),
        "gates": gates,
        "passed": all(bool(gate["passed"]) for gate in gates),
    }


ClipEvaluator = Callable[[ClipSpec, float], dict[str, object]]


def evaluate_suite(
    clips: Sequence[ClipSpec],
    *,
    thresholds: StabilityThresholds,
    settling_seconds: float,
    evaluator: ClipEvaluator = evaluate_clip,
) -> list[dict[str, object]]:
    """Evaluate all clips, retaining failures so the JSON remains diagnostic."""

    results: list[dict[str, object]] = []
    for clip in clips:
        try:
            evaluation = evaluator(clip, settling_seconds)
        except Exception as error:  # the report must survive one failed replay
            evaluation = {
                "exit_code": 1,
                "video": _project_path(clip.path),
                "metrics": None,
                "settling": {
                    "requested_duration_s": settling_seconds,
                    "status": "failed",
                    "same_simulation": False,
                    "completed": False,
                    "reason": "evaluation failed before settling metrics were available",
                },
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(
            assess_clip(
                clip,
                evaluation,
                thresholds,
                settling_seconds=settling_seconds,
            )
        )
    return results


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("mediapipe", "mujoco", "numpy", "opencv-contrib-python"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _git_output(*arguments: str, text: bool = True) -> str | bytes | None:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def _git_revision() -> str | None:
    output = _git_output("rev-parse", "HEAD")
    if not isinstance(output, str):
        return None
    return output.strip() or None


def _git_worktree_provenance() -> dict[str, object]:
    status_output = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    status = status_output.strip() if isinstance(status_output, str) else None
    diff_output = _git_output("diff", "--binary", "--no-ext-diff", "HEAD", text=False)
    diff_sha256 = (
        hashlib.sha256(diff_output).hexdigest()
        if isinstance(diff_output, bytes)
        else None
    )
    return {
        "dirty": None if status is None else bool(status),
        "status_porcelain": None if status is None else status.splitlines(),
        "tracked_diff_sha256": diff_sha256,
        "note": (
            "Runtime input SHA-256 values below identify tested file contents, "
            "including untracked evaluator code; git revision alone is insufficient "
            "when dirty is true."
        ),
    }


def _capture_run_provenance(clips: Sequence[ClipSpec]) -> dict[str, object]:
    """Freeze identifiers before importing video data or stepping physics."""

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "git_worktree": _git_worktree_provenance(),
        "runtime_input_sha256": _runtime_input_hashes(clips),
    }


def _sanitize_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _threshold_policy_violations(
    thresholds: StabilityThresholds,
) -> dict[str, dict[str, float | str]]:
    canonical = StabilityThresholds()
    violations: dict[str, dict[str, float | str]] = {}
    for name in _MINIMUM_THRESHOLD_FIELDS:
        observed = getattr(thresholds, name)
        required = getattr(canonical, name)
        if not _finite_number(observed) or float(observed) < required:
            violations[name] = {
                "observed": observed,
                "canonical": required,
                "required_relation": ">=",
            }
    for name in _MAXIMUM_THRESHOLD_FIELDS:
        observed = getattr(thresholds, name)
        required = getattr(canonical, name)
        if not _finite_number(observed) or float(observed) > required:
            violations[name] = {
                "observed": observed,
                "canonical": required,
                "required_relation": "<=",
            }
    return violations


def _clip_contract(clip: ClipSpec) -> dict[str, object]:
    return {
        "name": clip.name,
        "source": clip.source,
        "path": _project_path(clip.path),
        "demo_video": clip.demo_video,
        "expected_frames": clip.expected_frames,
        "expectation": asdict(clip.expectation),
        "calibration_video": (
            None
            if clip.calibration_video is None
            else _project_path(clip.calibration_video)
        ),
        "calibration_frame": clip.calibration_frame,
    }


def build_report(
    results: Sequence[Mapping[str, object]],
    clips: Sequence[ClipSpec],
    *,
    thresholds: StabilityThresholds,
    settling_seconds: float,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    selected_clips = tuple(clips)
    canonical_clips = default_clips()
    captured = dict(provenance or _capture_run_provenance(selected_clips))
    initial_hashes = captured.get("runtime_input_sha256")
    final_hashes = _runtime_input_hashes(selected_clips)
    runtime_inputs_unchanged = initial_hashes == final_hashes
    canonical_names = [clip.name for clip in canonical_clips]
    selected_names = [clip.name for clip in selected_clips]
    result_names = [result.get("name") for result in results]
    threshold_violations = _threshold_policy_violations(thresholds)
    suite_gates = [
        _gate(
            "exact_canonical_clip_matrix",
            selected_clips == canonical_clips,
            [_clip_contract(clip) for clip in selected_clips],
            [_clip_contract(clip) for clip in canonical_clips],
        ),
        _gate(
            "complete_clip_result_count",
            len(results) == len(canonical_clips),
            len(results),
            len(canonical_clips),
        ),
        _gate(
            "exact_canonical_result_matrix",
            result_names == canonical_names,
            result_names,
            canonical_names,
        ),
        _gate(
            "canonical_or_stricter_thresholds",
            not threshold_violations,
            threshold_violations,
            {
                "minimum_thresholds": ">= canonical defaults",
                "maximum_thresholds": "<= canonical defaults",
            },
        ),
        _gate(
            "minimum_settling_duration_s",
            _finite_number(settling_seconds)
            and settling_seconds >= CANONICAL_SETTLING_SECONDS,
            settling_seconds,
            {"minimum": CANONICAL_SETTLING_SECONDS},
        ),
        _gate(
            "runtime_inputs_unchanged_during_run",
            runtime_inputs_unchanged,
            runtime_inputs_unchanged,
            True,
        ),
    ]
    clips_passed = bool(results) and all(
        bool(result.get("passed")) for result in results
    )
    suite_passed = all(bool(gate["passed"]) for gate in suite_gates)
    return {
        "schema_version": 2,
        "description": (
            "End-to-end MediaPipe -> constrained IK -> balance/support -> free-base "
            "MuJoCo replay acceptance. No floating-base constraint is enabled."
        ),
        "command": "python tools/evaluate_freebase_stability.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_started_at_utc": captured.get("captured_at_utc"),
        "git_revision": captured.get("git_revision"),
        "git_worktree": captured.get("git_worktree"),
        "runtime_input_sha256": initial_hashes,
        "runtime_input_sha256_at_completion": final_hashes,
        "runtime_inputs_unchanged_during_run": runtime_inputs_unchanged,
        "report_kind": "canonical_acceptance" if suite_passed else "diagnostic",
        "configuration": {
            "base_mode": "free",
            "headless": True,
            "retargeting": "ik",
            "balance_controller": True,
            "settling_requested_s": settling_seconds,
            "clips": selected_names,
            "canonical_complete_matrix": selected_clips == canonical_clips,
            "thresholds": asdict(thresholds),
            "versions": _package_versions(),
        },
        "suite_gates": suite_gates,
        "overall_passed": clips_passed and suite_passed,
        "clips": list(results),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        action="append",
        choices=tuple(clip.name for clip in default_clips()),
        default=[],
        help="Run only this named default clip; repeat to select several.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-base-height-m", type=float, default=0.80)
    parser.add_argument("--minimum-final-base-height-m", type=float, default=0.88)
    parser.add_argument("--maximum-tilt-deg", type=float, default=20.0)
    parser.add_argument("--maximum-media-sync-error-s", type=float, default=0.003)
    parser.add_argument("--maximum-stale-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-skeleton-fraction", type=float, default=0.85)
    parser.add_argument(
        "--maximum-loaded-foot-slip-speed-m-s", type=float, default=0.15
    )
    parser.add_argument("--maximum-foot-slip-distance-m", type=float, default=0.075)
    parser.add_argument(
        "--maximum-swing-precontact-vertical-speed-m-s", type=float, default=0.50
    )
    parser.add_argument("--maximum-swing-impact-force-n", type=float, default=60.0)
    parser.add_argument(
        "--maximum-swing-contact-impulse-n-s", type=float, default=9.0
    )
    parser.add_argument(
        "--maximum-swing-peak-force-bodyweights", type=float, default=2.25
    )
    parser.add_argument(
        "--maximum-swing-contact-impulse-weight-s", type=float, default=0.30
    )
    parser.add_argument(
        "--settling-maximum-base-linear-speed-m-s",
        type=float,
        default=SETTLING_MAX_BASE_LINEAR_SPEED_M_S,
    )
    parser.add_argument(
        "--settling-maximum-joint-speed-rad-s",
        type=float,
        default=SETTLING_MAX_JOINT_SPEED_RAD_S,
    )
    parser.add_argument(
        "--settling-maximum-joint-tracking-error-rad",
        type=float,
        default=SETTLING_MAX_JOINT_TRACKING_ERROR_RAD,
    )
    parser.add_argument(
        "--settling-maximum-loaded-foot-slip-speed-m-s",
        type=float,
        default=SETTLING_MAX_LOADED_FOOT_SLIP_SPEED_M_S,
    )
    parser.add_argument(
        "--settling-maximum-capture-point-error-m",
        type=float,
        default=SETTLING_MAX_CAPTURE_POINT_ERROR_M,
    )
    parser.add_argument(
        "--settling-seconds",
        type=float,
        default=5.0,
        help=(
            "Required quiet interval after returning to double support on the same "
            "MuJoCo simulation (default: 5 s)."
        ),
    )
    return parser


def _validated_thresholds(args: argparse.Namespace) -> StabilityThresholds:
    values = (
        args.minimum_base_height_m,
        args.minimum_final_base_height_m,
        args.maximum_tilt_deg,
        args.maximum_media_sync_error_s,
        args.maximum_stale_fraction,
        args.minimum_skeleton_fraction,
        args.maximum_loaded_foot_slip_speed_m_s,
        args.maximum_foot_slip_distance_m,
        args.maximum_swing_precontact_vertical_speed_m_s,
        args.maximum_swing_impact_force_n,
        args.maximum_swing_contact_impulse_n_s,
        args.maximum_swing_peak_force_bodyweights,
        args.maximum_swing_contact_impulse_weight_s,
        args.settling_maximum_base_linear_speed_m_s,
        args.settling_maximum_joint_speed_rad_s,
        args.settling_maximum_joint_tracking_error_rad,
        args.settling_maximum_loaded_foot_slip_speed_m_s,
        args.settling_maximum_capture_point_error_m,
        args.settling_seconds,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all acceptance thresholds must be finite")
    if args.minimum_base_height_m <= 0.0 or args.minimum_final_base_height_m <= 0.0:
        raise ValueError("base-height thresholds must be positive")
    if args.maximum_tilt_deg <= 0.0 or args.maximum_tilt_deg >= 90.0:
        raise ValueError("--maximum-tilt-deg must be within (0, 90)")
    if args.maximum_media_sync_error_s < 0.0:
        raise ValueError("--maximum-media-sync-error-s must be non-negative")
    if not 0.0 <= args.maximum_stale_fraction <= 1.0:
        raise ValueError("--maximum-stale-fraction must be within [0, 1]")
    if not 0.0 <= args.minimum_skeleton_fraction <= 1.0:
        raise ValueError("--minimum-skeleton-fraction must be within [0, 1]")
    if args.maximum_loaded_foot_slip_speed_m_s < 0.0:
        raise ValueError("--maximum-loaded-foot-slip-speed-m-s must be non-negative")
    if args.maximum_foot_slip_distance_m < 0.0:
        raise ValueError("--maximum-foot-slip-distance-m must be non-negative")
    non_negative = (
        "maximum_swing_precontact_vertical_speed_m_s",
        "maximum_swing_impact_force_n",
        "maximum_swing_contact_impulse_n_s",
        "maximum_swing_peak_force_bodyweights",
        "maximum_swing_contact_impulse_weight_s",
        "settling_maximum_base_linear_speed_m_s",
        "settling_maximum_joint_speed_rad_s",
        "settling_maximum_joint_tracking_error_rad",
        "settling_maximum_loaded_foot_slip_speed_m_s",
        "settling_maximum_capture_point_error_m",
    )
    for name in non_negative:
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.settling_seconds < 0.0:
        raise ValueError("--settling-seconds must be non-negative")
    return StabilityThresholds(
        minimum_base_height_m=args.minimum_base_height_m,
        minimum_final_base_height_m=args.minimum_final_base_height_m,
        maximum_tilt_deg=args.maximum_tilt_deg,
        maximum_media_sync_error_s=args.maximum_media_sync_error_s,
        maximum_stale_fraction=args.maximum_stale_fraction,
        minimum_skeleton_fraction=args.minimum_skeleton_fraction,
        maximum_loaded_foot_slip_speed_m_s=(
            args.maximum_loaded_foot_slip_speed_m_s
        ),
        maximum_foot_slip_distance_m=args.maximum_foot_slip_distance_m,
        maximum_swing_precontact_vertical_speed_m_s=(
            args.maximum_swing_precontact_vertical_speed_m_s
        ),
        maximum_swing_impact_force_n=args.maximum_swing_impact_force_n,
        maximum_swing_contact_impulse_n_s=(
            args.maximum_swing_contact_impulse_n_s
        ),
        maximum_swing_peak_force_bodyweights=(
            args.maximum_swing_peak_force_bodyweights
        ),
        maximum_swing_contact_impulse_weight_s=(
            args.maximum_swing_contact_impulse_weight_s
        ),
        settling_maximum_base_linear_speed_m_s=(
            args.settling_maximum_base_linear_speed_m_s
        ),
        settling_maximum_joint_speed_rad_s=(
            args.settling_maximum_joint_speed_rad_s
        ),
        settling_maximum_joint_tracking_error_rad=(
            args.settling_maximum_joint_tracking_error_rad
        ),
        settling_maximum_loaded_foot_slip_speed_m_s=(
            args.settling_maximum_loaded_foot_slip_speed_m_s
        ),
        settling_maximum_capture_point_error_m=(
            args.settling_maximum_capture_point_error_m
        ),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    evaluator: ClipEvaluator = evaluate_clip,
) -> int:
    args = build_parser().parse_args(argv)
    thresholds = _validated_thresholds(args)
    available = {clip.name: clip for clip in default_clips()}
    clips = (
        [available[name] for name in args.clip]
        if args.clip
        else list(available.values())
    )
    provenance = _capture_run_provenance(clips)
    results = evaluate_suite(
        clips,
        thresholds=thresholds,
        settling_seconds=args.settling_seconds,
        evaluator=evaluator,
    )
    report = build_report(
        results,
        clips,
        thresholds=thresholds,
        settling_seconds=args.settling_seconds,
        provenance=provenance,
    )
    sanitized = _sanitize_json(report)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sanitized, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"free-base stability report: {output}")
    for result in results:
        metrics = result.get("metrics")
        if isinstance(metrics, Mapping):
            print(
                f"{result['name']}: passed={int(bool(result['passed']))} "
                f"fell={int(bool(metrics.get('fell')))} "
                f"min_z={metrics.get('minimum_base_height_m')} "
                f"tilt_deg={metrics.get('maximum_tilt_deg')}"
            )
        else:
            print(
                f"{result['name']}: passed=0 error={result.get('error', 'evaluation failed')}"
            )
    return 0 if bool(report["overall_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
