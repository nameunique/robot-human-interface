"""Evaluate final safe targets and measured free-base pose against the human.

Unlike ``evaluate_pose_fidelity.py``, this runner executes the production
MediaPipe -> IK -> standing balance -> support FSM -> MuJoCo path.  A scoped
instrumentation context observes (without replacing) production perception,
intent, support, command, and physics methods and associates their final values
with each input video frame. Settling is disabled, so no post-input samples can
be mistaken for tracking evidence.

The report deliberately separates:

* ``safe_command``: the target after all deployable safety projections;
* ``actual_qpos``: the physical pose reached by the free-base simulation.

Task-space errors use :class:`MujocoPoseFidelityEvaluator`; joint-angle error
is never compared directly across the human and robot morphologies.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray

from robot_human_interface.app.teleop import (
    BUNDLED_VIDEO_PATHS,
    TeleopStats,
    build_parser as build_teleop_parser,
    run_teleop,
)
from robot_human_interface.control import (
    HumanSupportIntentEstimator,
    SupportControlConfig,
    SupportIntent,
    SupportPhase,
    SupportStateMachine,
    load_support_control_config,
)
from robot_human_interface.pose import MediaPipePoseLandmarker
from robot_human_interface.retargeting import (
    ARM_DIRECTION_NAMES,
    DIRECTION_NAMES,
    END_EFFECTOR_DIRECTION_NAMES,
    HEAD_DIRECTION_NAMES,
    LEG_DIRECTION_NAMES,
    MujocoPoseFidelityEvaluator,
    angular_pose_fidelity,
    human_anatomical_directions,
)
from robot_human_interface.simulation import HumanoidSimulation
from robot_human_interface.skeleton import JOINT_NAMES, SkeletonFrame


FloatArray = NDArray[np.float64]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "safe-pose-fidelity.json"

GROUPS: dict[str, tuple[str, ...]] = {
    "arms": ARM_DIRECTION_NAMES,
    "legs": LEG_DIRECTION_NAMES,
    "end_effectors": END_EFFECTOR_DIRECTION_NAMES,
    "head": HEAD_DIRECTION_NAMES,
}

# These are broad, a-priori directional-task limits, not fitted percentiles.
# A p90 error near/past 90 degrees means that the robot is usually orthogonal
# to the requested direction and should never be called faithful.  Actual qpos
# receives ten extra degrees for finite-bandwidth tracking dynamics.
SAFE_P90_LIMITS_DEG: dict[str, float] = {
    "arms": 70.0,
    "legs": 60.0,
    "end_effectors": 70.0,
    "head": 50.0,
}
ACTUAL_P90_LIMITS_DEG: dict[str, float] = {
    "arms": 80.0,
    "legs": 70.0,
    "end_effectors": 80.0,
    "head": 60.0,
}


@dataclass(frozen=True, slots=True)
class FidelityGatePolicy:
    """Predeclared acceptance limits for coverage and semantic motion."""

    minimum_evaluated_source_fraction: float = 0.70
    minimum_group_valid_direction_fraction: float = 0.80
    minimum_group_direction_frames: int = 30
    minimum_trajectory_samples: int = 30
    minimum_trajectory_duration_s: float = 1.0
    minimum_human_amplitude_deg: float = 12.0
    minimum_safe_amplitude_ratio: float = 0.25
    minimum_actual_amplitude_ratio: float = 0.20
    minimum_safe_correlation: float = 0.50
    minimum_actual_correlation: float = 0.40
    maximum_actual_lag_s: float = 0.50
    lag_search_horizon_s: float = 1.50
    minimum_safe_leg_event_clearance_m: float = 0.020
    minimum_actual_leg_event_clearance_m: float = 0.020

    def __post_init__(self) -> None:
        fractions = (
            self.minimum_evaluated_source_fraction,
            self.minimum_group_valid_direction_fraction,
            self.minimum_safe_amplitude_ratio,
            self.minimum_actual_amplitude_ratio,
        )
        correlations = (
            self.minimum_safe_correlation,
            self.minimum_actual_correlation,
        )
        positive = (
            self.minimum_trajectory_duration_s,
            self.minimum_human_amplitude_deg,
            self.maximum_actual_lag_s,
            self.lag_search_horizon_s,
            self.minimum_safe_leg_event_clearance_m,
            self.minimum_actual_leg_event_clearance_m,
        )
        if not all(np.isfinite(fractions)) or any(
            not 0.0 <= value <= 1.0 for value in fractions
        ):
            raise ValueError("coverage and amplitude ratios must be within [0, 1]")
        if not all(np.isfinite(correlations)) or any(
            not -1.0 <= value <= 1.0 for value in correlations
        ):
            raise ValueError("correlation limits must be within [-1, 1]")
        if not all(np.isfinite(positive)) or any(value <= 0.0 for value in positive):
            raise ValueError("trajectory duration, amplitude, and lag must be positive")
        if self.minimum_trajectory_samples < 3:
            raise ValueError("minimum_trajectory_samples must be at least three")
        if self.minimum_group_direction_frames < 1:
            raise ValueError("minimum_group_direction_frames must be positive")
        if self.lag_search_horizon_s <= self.maximum_actual_lag_s:
            raise ValueError(
                "lag_search_horizon_s must exceed maximum_actual_lag_s so the "
                "lag gate can observe late tracking"
            )


_MINIMUM_POLICY_FIELDS = (
    "minimum_evaluated_source_fraction",
    "minimum_group_valid_direction_fraction",
    "minimum_group_direction_frames",
    "minimum_trajectory_samples",
    "minimum_trajectory_duration_s",
    "minimum_human_amplitude_deg",
    "minimum_safe_amplitude_ratio",
    "minimum_actual_amplitude_ratio",
    "minimum_safe_correlation",
    "minimum_actual_correlation",
    "lag_search_horizon_s",
    "minimum_safe_leg_event_clearance_m",
    "minimum_actual_leg_event_clearance_m",
)
_MAXIMUM_POLICY_FIELDS = ("maximum_actual_lag_s",)


@dataclass(frozen=True, slots=True)
class ClipSpec:
    name: str
    source: str
    path: Path
    demo_video: str | None
    semantic_channels: tuple[str, ...]
    task_label: str
    capability: str
    expected_frames: int
    unsupported_limitations: tuple[str, ...] = ()
    calibration_video: Path | None = None
    calibration_frame: int | None = None
    required_leg_event_sides: tuple[str, ...] = ()
    gated_p90_groups: tuple[str, ...] = tuple(GROUPS)

    def __post_init__(self) -> None:
        if self.source not in {"mp4", "replay"}:
            raise ValueError(f"unsupported clip source: {self.source!r}")
        if not self.semantic_channels:
            raise ValueError("every task acceptance contract requires a semantic channel")
        if not self.task_label.strip():
            raise ValueError("task_label must be non-empty")
        if not self.capability.strip():
            raise ValueError("capability must be non-empty")
        if (
            not isinstance(self.expected_frames, int)
            or isinstance(self.expected_frames, bool)
            or self.expected_frames <= 0
        ):
            raise ValueError("expected_frames must be a positive integer")
        if any(not limitation.strip() for limitation in self.unsupported_limitations):
            raise ValueError("unsupported limitations must be non-empty strings")
        unknown = sorted(set(self.semantic_channels) - set(DIRECTION_NAMES))
        if unknown:
            raise ValueError(f"unknown semantic channels: {unknown}")
        if (self.calibration_video is None) != (self.calibration_frame is None):
            raise ValueError("calibration_video and calibration_frame must be paired")
        if self.calibration_frame is not None and self.calibration_frame < 0:
            raise ValueError("calibration_frame must be non-negative")
        accepted_sides = {
            SupportIntent.RIGHT_SWING.value,
            SupportIntent.LEFT_SWING.value,
        }
        unknown_sides = sorted(set(self.required_leg_event_sides) - accepted_sides)
        if unknown_sides:
            raise ValueError(f"unknown unilateral leg-event sides: {unknown_sides}")
        if len(set(self.required_leg_event_sides)) != len(
            self.required_leg_event_sides
        ):
            raise ValueError("required_leg_event_sides must be unique")
        if self.required_leg_event_sides and not (
            set(self.semantic_channels) & set(LEG_DIRECTION_NAMES)
        ):
            raise ValueError(
                "unilateral leg-event clips must declare a descriptive leg channel"
            )
        unknown_groups = sorted(set(self.gated_p90_groups) - set(GROUPS))
        if unknown_groups:
            raise ValueError(f"unknown p90 fidelity groups: {unknown_groups}")
        if len(set(self.gated_p90_groups)) != len(self.gated_p90_groups):
            raise ValueError("gated_p90_groups must be unique")


def default_clips() -> tuple[ClipSpec, ...]:
    """Return the ordered evaluation matrix and independent semantic tasks."""

    external = PROJECT_ROOT / "assets" / "videos" / "external"
    return (
        ClipSpec(
            "slow-balance",
            "mp4",
            BUNDLED_VIDEO_PATHS["slow-balance"],
            "slow-balance",
            ("right_arm", "left_arm", "right_leg", "left_leg"),
            "alternating unilateral balance with arm motion",
            "contact-gated left/right leg lift plus bounded arm imitation",
            1961,
            (
                "monocular 3-D leg depth is descriptive; calibrated image-plane "
                "intent and physical foot clearance define lift acceptance",
            ),
            required_leg_event_sides=("right_swing", "left_swing"),
            gated_p90_groups=("arms", "head"),
        ),
        ClipSpec(
            "jumping-jacks",
            "mp4",
            BUNDLED_VIDEO_PATHS["jumping-jacks"],
            "jumping-jacks",
            ("right_arm", "left_arm", "right_leg", "left_leg"),
            "whole-body jumping-jack projection",
            "bounded full-body imitation while preserving bilateral ground contact",
            194,
            (
                "flight and ballistic jumping are unsupported; required leg channels "
                "measure the supported grounded projection and remain acceptance gates",
            ),
            calibration_video=BUNDLED_VIDEO_PATHS["jumping-jacks"],
            calibration_frame=2,
            gated_p90_groups=("arms", "legs"),
        ),
        ClipSpec(
            "arm-circles",
            "replay",
            external / "dvids_arm_circles.mp4",
            None,
            ("right_upper_arm", "left_upper_arm", "right_arm", "left_arm"),
            "bilateral arm circles",
            "bilateral shoulder/arm trajectory imitation",
            796,
            gated_p90_groups=("arms",),
        ),
        ClipSpec(
            "frontal-leg-swing",
            "replay",
            external / "dvids_frontal_leg_swing.mp4",
            None,
            ("right_thigh", "right_shin", "right_leg"),
            "unilateral frontal right-leg swing",
            "contact-gated right leg lift with physical foot clearance",
            836,
            (
                "monocular 3-D leg direction is descriptive; calibrated image-plane "
                "intent and physical foot clearance define task acceptance",
            ),
            calibration_video=external / "dvids_arm_circles.mp4",
            calibration_frame=29,
            required_leg_event_sides=("right_swing",),
            gated_p90_groups=(),
        ),
        ClipSpec(
            "stationary-squat",
            "replay",
            external / "dvids_stationary_squat.mp4",
            None,
            ("right_thigh", "left_thigh", "right_shin", "left_shin"),
            "stationary bilateral squat",
            "feet-planted bilateral sagittal leg imitation",
            817,
            gated_p90_groups=("legs",),
        ),
        ClipSpec(
            "trunk-circles",
            "replay",
            external / "dvids_trunk_circles.mp4",
            None,
            ("head",),
            "trunk-circle orientation proxy",
            "head orientation tracks the observable upper-body orientation proxy",
            867,
            (
                "true trunk articulation is unsupported because the robot has no waist DOF",
                "torso-relative human leg-direction rotation is descriptive and is not "
                "interpreted as requested leg articulation while both feet stay planted",
            ),
            calibration_video=external / "dvids_arm_circles.mp4",
            calibration_frame=29,
            gated_p90_groups=("head",),
        ),
    )


@dataclass(slots=True)
class _TraceSample:
    timestamp_s: float
    skeleton: SkeletonFrame | None = None
    safe_positions_rad: FloatArray | None = None
    safe_stale: bool = True
    actual_positions_rad: FloatArray | None = None
    safe_right_minus_left_foot_height_m: float | None = None
    actual_right_minus_left_foot_height_m: float | None = None
    human_support_intent: str | None = None
    human_support_lift_ratio: float | None = None
    human_support_calibrated: bool = False
    human_support_stale: bool = True
    support_phase: str | None = None
    support_active_intent: str | None = None
    support_requested_intent: str | None = None
    support_abort_reason: str | None = None


class _RobotFootHeightEvaluator:
    """Evaluate a motor target in private MuJoCo data, never the live state."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self._model = model
        self._data = mujoco.MjData(model)
        joint_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(joint_ids < 0):
            raise ValueError("MuJoCo model is missing a canonical motor joint")
        self._qpos_addresses = np.asarray(
            model.jnt_qposadr[joint_ids], dtype=np.int32
        )
        self._right_site_id = int(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, "right_foot_contact"
            )
        )
        self._left_site_id = int(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, "left_foot_contact"
            )
        )
        if self._right_site_id < 0 or self._left_site_id < 0:
            raise ValueError("MuJoCo model is missing foot contact sites")

    def right_minus_left_m(self, positions_rad: Sequence[float]) -> float:
        positions = np.asarray(positions_rad, dtype=np.float64)
        if positions.shape != (len(JOINT_NAMES),) or not np.isfinite(
            positions
        ).all():
            raise ValueError("positions_rad must be a finite canonical vector")
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[self._qpos_addresses] = positions
        mujoco.mj_forward(self._model, self._data)
        return float(
            self._data.site_xpos[self._right_site_id, 2]
            - self._data.site_xpos[self._left_site_id, 2]
        )


class _TeleopTraceRecorder:
    """Frame-align the last production safe target and measured qpos."""

    def __init__(self) -> None:
        self.samples: list[_TraceSample] = []
        self._current: _TraceSample | None = None
        self.evaluator: MujocoPoseFidelityEvaluator | None = None
        self._foot_height_evaluator: _RobotFootHeightEvaluator | None = None
        self.simulation_instances: set[int] = set()
        self.data_instances: set[int] = set()
        self.equality_constraint_counts: set[int] = set()
        self.base_joint_types: set[str] = set()

    def begin_frame(self, timestamp_s: float) -> None:
        self.finish_frame()
        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s) or timestamp_s < 0.0:
            raise ValueError("input frame timestamp must be finite and non-negative")
        self._current = _TraceSample(timestamp_s)

    def record_skeleton(self, skeleton: SkeletonFrame | None) -> None:
        if self._current is None:
            raise RuntimeError("pose result arrived without an input frame")
        self._current.skeleton = skeleton

    def _record_simulation(self, simulation: HumanoidSimulation) -> None:
        self.simulation_instances.add(id(simulation))
        self.data_instances.add(id(simulation.data))
        model = simulation.model
        self.equality_constraint_counts.add(int(model.neq))
        base_joint_id = int(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, "base_free"
            )
        )
        if base_joint_id < 0:
            self.base_joint_types.add("missing")
        else:
            joint_type = int(model.jnt_type[base_joint_id])
            self.base_joint_types.add(
                "free"
                if joint_type == int(mujoco.mjtJoint.mjJNT_FREE)
                else f"mujoco_joint_type_{joint_type}"
            )
        if self.evaluator is None:
            self.evaluator = MujocoPoseFidelityEvaluator(model)
            self._foot_height_evaluator = _RobotFootHeightEvaluator(model)

    def record_human_support(self, estimate: object) -> None:
        if self._current is None:
            return
        intent = getattr(estimate, "intent", None)
        self._current.human_support_intent = str(getattr(intent, "value", intent))
        ratio = float(getattr(estimate, "signed_height_ratio"))
        if not np.isfinite(ratio):
            raise ValueError("human support lift ratio must be finite")
        self._current.human_support_lift_ratio = ratio
        self._current.human_support_calibrated = bool(
            getattr(estimate, "calibrated")
        )
        self._current.human_support_stale = bool(getattr(estimate, "stale"))

    def record_support(self, machine: SupportStateMachine) -> None:
        if self._current is None:
            return
        diagnostics = machine.last_diagnostics
        if diagnostics is None:
            return
        self._current.support_phase = diagnostics.phase.value
        self._current.support_active_intent = diagnostics.active_intent.value
        self._current.support_requested_intent = diagnostics.requested_intent.value
        self._current.support_abort_reason = diagnostics.abort_reason

    def record_safe_command(
        self,
        simulation: HumanoidSimulation,
        command: object,
        applied_positions_rad: Sequence[float],
    ) -> None:
        self._record_simulation(simulation)
        if self._current is None:
            return
        positions = np.asarray(applied_positions_rad, dtype=np.float64)
        if positions.shape != (len(JOINT_NAMES),) or not np.isfinite(positions).all():
            raise ValueError("safe command must be a finite canonical 20-vector")
        self._current.safe_positions_rad = positions.copy()
        self._current.safe_stale = bool(getattr(command, "stale", False))

    def record_state(self, simulation: HumanoidSimulation, state: object) -> None:
        self._record_simulation(simulation)
        if self._current is None:
            return
        names = tuple(getattr(state, "joint_names"))
        if names != JOINT_NAMES:
            raise ValueError("actual state must use canonical motor order")
        positions = np.asarray(getattr(state, "joint_positions_rad"), dtype=np.float64)
        if positions.shape != (len(JOINT_NAMES),) or not np.isfinite(positions).all():
            raise ValueError("actual qpos must be a finite canonical 20-vector")
        self._current.actual_positions_rad = positions.copy()
        right_foot = np.asarray(
            getattr(state, "right_foot_position_m"), dtype=np.float64
        )
        left_foot = np.asarray(
            getattr(state, "left_foot_position_m"), dtype=np.float64
        )
        if (
            right_foot.shape != (3,)
            or left_foot.shape != (3,)
            or not np.isfinite(np.concatenate((right_foot, left_foot))).all()
        ):
            raise ValueError("actual foot positions must be finite 3-vectors")
        self._current.actual_right_minus_left_foot_height_m = float(
            right_foot[2] - left_foot[2]
        )

    def finish_frame(self) -> None:
        if self._current is not None:
            if (
                self._current.safe_positions_rad is not None
                and self._foot_height_evaluator is not None
            ):
                self._current.safe_right_minus_left_foot_height_m = (
                    self._foot_height_evaluator.right_minus_left_m(
                        self._current.safe_positions_rad
                    )
                )
            self.samples.append(self._current)
            self._current = None


@contextmanager
def _record_end_to_end_trace() -> Iterator[_TeleopTraceRecorder]:
    """Temporarily observe production methods and restore them unconditionally."""

    recorder = _TeleopTraceRecorder()
    original_estimate = MediaPipePoseLandmarker.estimate
    original_intent_update = HumanSupportIntentEstimator.update
    original_support_update = SupportStateMachine.update
    original_apply = HumanoidSimulation.apply_joint_command
    original_step = HumanoidSimulation.step

    def recorded_estimate(pose: MediaPipePoseLandmarker, frame: object) -> object:
        recorder.begin_frame(float(getattr(frame, "timestamp_s")))
        skeleton = original_estimate(pose, frame)  # type: ignore[arg-type]
        recorder.record_skeleton(skeleton)
        return skeleton

    def recorded_apply(
        simulation: HumanoidSimulation,
        command: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        applied = original_apply(simulation, command, *args, **kwargs)
        recorder.record_safe_command(simulation, command, applied)
        return applied

    def recorded_intent_update(
        estimator: HumanSupportIntentEstimator,
        frame: SkeletonFrame | None,
        *,
        timestamp_s: float | None = None,
    ) -> object:
        estimate = original_intent_update(
            estimator, frame, timestamp_s=timestamp_s
        )
        recorder.record_human_support(estimate)
        return estimate

    def recorded_support_update(
        machine: SupportStateMachine,
        reference: object,
        state: object,
        *,
        dt_s: float,
        intent: object | None = None,
        force_return_reason: str | None = None,
    ) -> object:
        command = original_support_update(
            machine,
            reference,  # type: ignore[arg-type]
            state,
            dt_s=dt_s,
            intent=intent,  # type: ignore[arg-type]
            force_return_reason=force_return_reason,
        )
        recorder.record_support(machine)
        return command

    def recorded_step(
        simulation: HumanoidSimulation,
        *args: object,
        **kwargs: object,
    ) -> object:
        state = original_step(simulation, *args, **kwargs)
        recorder.record_state(simulation, state)
        return state

    MediaPipePoseLandmarker.estimate = recorded_estimate  # type: ignore[method-assign]
    HumanSupportIntentEstimator.update = recorded_intent_update  # type: ignore[method-assign]
    SupportStateMachine.update = recorded_support_update  # type: ignore[method-assign]
    HumanoidSimulation.apply_joint_command = recorded_apply  # type: ignore[method-assign]
    HumanoidSimulation.step = recorded_step  # type: ignore[method-assign]
    try:
        yield recorder
    finally:
        recorder.finish_frame()
        MediaPipePoseLandmarker.estimate = original_estimate  # type: ignore[method-assign]
        HumanSupportIntentEstimator.update = original_intent_update  # type: ignore[method-assign]
        SupportStateMachine.update = original_support_update  # type: ignore[method-assign]
        HumanoidSimulation.apply_joint_command = original_apply  # type: ignore[method-assign]
        HumanoidSimulation.step = original_step  # type: ignore[method-assign]


def _summary_deg(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean_deg": None, "p50_deg": None, "p90_deg": None}
    return {
        "count": int(finite.size),
        "mean_deg": float(np.mean(finite)),
        "p50_deg": float(np.percentile(finite, 50.0)),
        "p90_deg": float(np.percentile(finite, 90.0)),
    }


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _correlation(first: FloatArray, second: FloatArray) -> float | None:
    if first.size < 3 or second.size != first.size:
        return None
    if float(np.std(first)) < 1e-9 or float(np.std(second)) < 1e-9:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else None


def _best_lagged_correlation(
    human: FloatArray,
    robot: FloatArray,
    timestamps_s: FloatArray,
    *,
    maximum_lag_s: float,
) -> tuple[float | None, float | None]:
    """Return best correlation and lag; positive lag means robot follows human."""

    if human.size < 3 or robot.shape != human.shape or timestamps_s.shape != human.shape:
        return None, None
    deltas = np.diff(timestamps_s)
    deltas = deltas[np.isfinite(deltas) & (deltas > 1e-6)]
    if deltas.size == 0:
        return None, None
    median_dt_s = float(np.median(deltas))
    maximum_steps = max(0, int(np.floor(maximum_lag_s / median_dt_s)))
    minimum_overlap = max(10, human.size // 2)
    candidates: list[tuple[float, int]] = []
    for lag_steps in range(-maximum_steps, maximum_steps + 1):
        if lag_steps > 0:
            first = human[:-lag_steps]
            second = robot[lag_steps:]
        elif lag_steps < 0:
            first = human[-lag_steps:]
            second = robot[:lag_steps]
        else:
            first = human
            second = robot
        if first.size < minimum_overlap:
            continue
        correlation = _correlation(first, second)
        if correlation is not None:
            candidates.append((correlation, lag_steps))
    if not candidates:
        return None, None
    # Prefer the shortest lag when correlations are numerically tied.
    best_correlation, best_steps = max(
        candidates, key=lambda item: (round(item[0], 12), -abs(item[1]))
    )
    return float(best_correlation), float(best_steps * median_dt_s)


def _direction_trajectory_metrics(
    timestamps_s: Sequence[float],
    human_vectors: Sequence[Sequence[float]],
    safe_vectors: Sequence[Sequence[float]],
    actual_vectors: Sequence[Sequence[float]],
    *,
    policy: FidelityGatePolicy,
) -> dict[str, object]:
    """Compare signed motion along the human trajectory's dominant tangent."""

    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    human = np.asarray(human_vectors, dtype=np.float64)
    safe = np.asarray(safe_vectors, dtype=np.float64)
    actual = np.asarray(actual_vectors, dtype=np.float64)
    expected_shape = (timestamps.size, 3)
    if human.shape != expected_shape or safe.shape != expected_shape or actual.shape != expected_shape:
        raise ValueError("trajectory vectors must all have shape (sample_count, 3)")
    finite = (
        np.isfinite(timestamps)
        & np.isfinite(human).all(axis=1)
        & np.isfinite(safe).all(axis=1)
        & np.isfinite(actual).all(axis=1)
    )
    timestamps = timestamps[finite]
    human = human[finite]
    safe = safe[finite]
    actual = actual[finite]
    sample_count = int(timestamps.size)
    duration_s = float(timestamps[-1] - timestamps[0]) if sample_count > 1 else 0.0
    result: dict[str, object] = {
        "sample_count": sample_count,
        "duration_s": duration_s,
        "reliable": False,
        "reason": None,
        "human_amplitude_deg": None,
        "safe_command": None,
        "actual_qpos": None,
    }
    if sample_count < policy.minimum_trajectory_samples:
        result["reason"] = "insufficient_samples"
        return result
    if duration_s < policy.minimum_trajectory_duration_s:
        result["reason"] = "insufficient_duration"
        return result

    center = np.mean(human, axis=0)
    center_norm = float(np.linalg.norm(center))
    if center_norm < 1e-6:
        result["reason"] = "ambiguous_mean_direction"
        return result
    center /= center_norm
    _, singular_values, right_vectors = np.linalg.svd(human - center, full_matrices=False)
    if singular_values.size == 0 or float(singular_values[0]) < 1e-8:
        result["reason"] = "no_directional_motion"
        return result
    tangent = right_vectors[0] - np.dot(right_vectors[0], center) * center
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-8:
        result["reason"] = "degenerate_motion_axis"
        return result
    tangent /= tangent_norm

    def signed_angles(vectors: FloatArray) -> FloatArray:
        radians = np.arctan2(vectors @ tangent, vectors @ center)
        return np.degrees(np.unwrap(radians))

    human_angles = signed_angles(human)
    safe_angles = signed_angles(safe)
    actual_angles = signed_angles(actual)

    def amplitude(values: FloatArray) -> float:
        return float(np.percentile(values, 95.0) - np.percentile(values, 5.0))

    human_amplitude = amplitude(human_angles)
    result["human_amplitude_deg"] = human_amplitude
    if human_amplitude < policy.minimum_human_amplitude_deg:
        result["reason"] = "insufficient_human_amplitude"
        return result

    def stage_metrics(values: FloatArray) -> dict[str, float | None]:
        stage_amplitude = amplitude(values)
        best_correlation, lag_s = _best_lagged_correlation(
            human_angles,
            values,
            timestamps,
            maximum_lag_s=policy.lag_search_horizon_s,
        )
        return {
            "amplitude_deg": stage_amplitude,
            "amplitude_ratio": float(stage_amplitude / human_amplitude),
            "zero_lag_correlation": _correlation(human_angles, values),
            "best_correlation": best_correlation,
            "best_lag_s": lag_s,
        }

    result["safe_command"] = stage_metrics(safe_angles)
    result["actual_qpos"] = stage_metrics(actual_angles)
    result["reliable"] = True
    return result


_SWING_SIDES = (
    SupportIntent.RIGHT_SWING.value,
    SupportIntent.LEFT_SWING.value,
)


def _swing_sign(side: str) -> float:
    if side == SupportIntent.RIGHT_SWING.value:
        return 1.0
    if side == SupportIntent.LEFT_SWING.value:
        return -1.0
    raise ValueError(f"not a unilateral swing side: {side!r}")


def _trace_frame_interval_s(samples: Sequence[_TraceSample]) -> float:
    timestamps = np.asarray([sample.timestamp_s for sample in samples], dtype=np.float64)
    deltas = np.diff(timestamps)
    deltas = deltas[np.isfinite(deltas) & (deltas > 1e-6)]
    return float(np.median(deltas)) if deltas.size else 0.0


def _intent_events(samples: Sequence[_TraceSample]) -> list[dict[str, object]]:
    """Segment debounced, calibrated production lift requests."""

    events: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for index, sample in enumerate(samples):
        side = (
            sample.human_support_intent
            if sample.human_support_calibrated and not sample.human_support_stale
            else None
        )
        if side not in _SWING_SIDES:
            if current is not None:
                events.append(current)
                current = None
            continue
        ratio = sample.human_support_lift_ratio
        magnitude = abs(float(ratio)) if ratio is not None else 0.0
        if current is None or current["side"] != side:
            if current is not None:
                events.append(current)
            occupancy = sample.support_active_intent
            current = {
                "side": side,
                "start_index": index,
                "end_index": index,
                "start_timestamp_s": float(sample.timestamp_s),
                "end_timestamp_s": float(sample.timestamp_s),
                "maximum_observed_lift_ratio": magnitude,
                "controller_side_at_request": occupancy,
                "queued_behind_opposite_cycle": bool(
                    occupancy in _SWING_SIDES and occupancy != side
                ),
            }
        else:
            current["end_index"] = index
            current["end_timestamp_s"] = float(sample.timestamp_s)
            current["maximum_observed_lift_ratio"] = max(
                float(current["maximum_observed_lift_ratio"]), magnitude
            )
    if current is not None:
        events.append(current)
    # A confidence/filter dip can briefly report DOUBLE while the already
    # accepted FSM cycle keeps executing the same side. That is one physical
    # request, not multiple requests for duplicate cycles. Coalesce only when
    # every bridge frame proves the same active FSM side; a real return to
    # double support or an opposite cycle remains a distinct event.
    coalesced: list[dict[str, object]] = []
    for event in events:
        event["intent_segment_count"] = 1
        event["coalesced_interruption_frames"] = 0
        if coalesced and coalesced[-1]["side"] == event["side"]:
            previous = coalesced[-1]
            previous_end = int(previous["end_index"])
            current_start = int(event["start_index"])
            bridge = samples[previous_end + 1 : current_start + 1]
            if bridge and all(
                sample.support_active_intent == event["side"] for sample in bridge
            ):
                previous["end_index"] = event["end_index"]
                previous["end_timestamp_s"] = event["end_timestamp_s"]
                previous["maximum_observed_lift_ratio"] = max(
                    float(previous["maximum_observed_lift_ratio"]),
                    float(event["maximum_observed_lift_ratio"]),
                )
                previous["intent_segment_count"] = int(
                    previous["intent_segment_count"]
                ) + 1
                previous["coalesced_interruption_frames"] = int(
                    previous["coalesced_interruption_frames"]
                ) + max(0, current_start - previous_end - 1)
                continue
        coalesced.append(event)
    events = coalesced
    for event in events:
        event["duration_s"] = float(event["end_timestamp_s"]) - float(
            event["start_timestamp_s"]
        )
    return events


def _support_cycles(samples: Sequence[_TraceSample]) -> list[dict[str, object]]:
    """Segment executed FSM cycles and retain samples for deadline checks."""

    cycles: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for index, sample in enumerate(samples):
        side = sample.support_active_intent
        if side not in _SWING_SIDES:
            if current is not None:
                current["terminal_phase"] = sample.support_phase
                current["terminal_active_intent"] = sample.support_active_intent
                cycles.append(current)
                current = None
            continue
        if current is None or current["side"] != side:
            if current is not None:
                current["terminal_phase"] = sample.support_phase
                current["terminal_active_intent"] = sample.support_active_intent
                cycles.append(current)
            current = {
                "side": side,
                "start_index": index,
                "end_index": index,
                "start_timestamp_s": float(sample.timestamp_s),
                "end_timestamp_s": float(sample.timestamp_s),
                "phases": set(),
                "phase_sequence": [],
                "abort_reasons": set(),
                "samples": [],
                "terminal_phase": None,
                "terminal_active_intent": None,
            }
        current["end_index"] = index
        current["end_timestamp_s"] = float(sample.timestamp_s)
        phases = current["phases"]
        assert isinstance(phases, set)
        if sample.support_phase is not None:
            phases.add(sample.support_phase)
            phase_sequence = current["phase_sequence"]
            assert isinstance(phase_sequence, list)
            if not phase_sequence or phase_sequence[-1] != sample.support_phase:
                phase_sequence.append(sample.support_phase)
        reasons = current["abort_reasons"]
        assert isinstance(reasons, set)
        if sample.support_abort_reason is not None:
            reasons.add(sample.support_abort_reason)
        cycle_samples = current["samples"]
        assert isinstance(cycle_samples, list)
        cycle_samples.append(sample)
    if current is not None:
        cycles.append(current)
    return cycles


def _cycle_return_completed(cycle: Mapping[str, object]) -> bool:
    """Require an ordered lowering, confirmed landing, centering, and DS return."""

    raw_sequence = cycle.get("phase_sequence")
    if not isinstance(raw_sequence, (list, tuple)):
        return False
    sequence = [str(value) for value in raw_sequence]
    phase_order = {
        SupportPhase.SHIFT_WEIGHT.value: 0,
        SupportPhase.VERIFY_STANCE.value: 1,
        SupportPhase.LIFT_SWING.value: 2,
        SupportPhase.HOLD_SWING.value: 3,
        SupportPhase.LOWER_SWING.value: 4,
        SupportPhase.VERIFY_TOUCHDOWN.value: 5,
        SupportPhase.CENTER_WEIGHT.value: 6,
    }
    ranks = [phase_order.get(phase) for phase in sequence]
    ordered = bool(
        ranks
        and all(rank is not None for rank in ranks)
        and all(
            int(first) <= int(second)
            for first, second in zip(ranks, ranks[1:], strict=False)
        )
    )
    required = (
        SupportPhase.HOLD_SWING.value,
        SupportPhase.LOWER_SWING.value,
        SupportPhase.CENTER_WEIGHT.value,
    )
    cursor = 0
    for phase in sequence:
        if cursor < len(required) and phase == required[cursor]:
            cursor += 1
    terminal_phase = cycle.get("terminal_phase")
    terminal_intent = cycle.get("terminal_active_intent")
    returned_to_double = bool(
        terminal_phase == SupportPhase.DOUBLE_SUPPORT.value
        and terminal_intent == SupportIntent.DOUBLE_SUPPORT.value
    )
    queued_handoff = bool(
        terminal_phase == SupportPhase.SHIFT_WEIGHT.value
        and terminal_intent in _SWING_SIDES
        and terminal_intent != cycle.get("side")
    )
    return bool(
        ordered
        and cursor == len(required)
        and (returned_to_double or queued_handoff)
    )


def _public_cycle(cycle: Mapping[str, object], index: int) -> dict[str, object]:
    phases = cycle.get("phases", set())
    phase_sequence = cycle.get("phase_sequence", [])
    abort_reasons = cycle.get("abort_reasons", set())
    return {
        "index": index,
        "side": cycle.get("side"),
        "start_timestamp_s": cycle.get("start_timestamp_s"),
        "end_timestamp_s": cycle.get("end_timestamp_s"),
        "duration_s": float(cycle["end_timestamp_s"])
        - float(cycle["start_timestamp_s"]),
        "phases": sorted(str(value) for value in phases),
        "phase_sequence": [str(value) for value in phase_sequence],
        "terminal_phase": cycle.get("terminal_phase"),
        "terminal_active_intent": cycle.get("terminal_active_intent"),
        "return_completed": _cycle_return_completed(cycle),
        "abort_reasons": sorted(str(value) for value in abort_reasons),
    }


def _analyze_leg_events(
    samples: Sequence[_TraceSample],
    *,
    required_sides: Sequence[str],
    support_config: SupportControlConfig,
    policy: FidelityGatePolicy,
) -> dict[str, object]:
    """Match observable lift requests to safe FSM cycles in FIFO order.

    The camera-side event is the production confidence/debounce estimate, not
    monocular world-depth.  A request may wait behind the current safe cycle;
    deadlines therefore come from the deployed FSM configuration.  A final
    request is censored only when the source ends before its complete bounded
    response window, never after a wrong side, abort, or completed low lift.
    """

    required = tuple(str(side) for side in required_sides)
    unknown = sorted(set(required) - set(_SWING_SIDES))
    if unknown:
        raise ValueError(f"unknown required swing sides: {unknown}")
    if not required:
        return {
            "applicable": False,
            "required_completed_sides": [],
            "events": [],
            "cycles": [],
            "passed": True,
        }

    events = _intent_events(samples)
    cycles = _support_cycles(samples)
    frame_interval_s = _trace_frame_interval_s(samples)
    idle_acceptance_deadline_s = support_config.stance_load_timeout_s
    queued_acceptance_deadline_s = (
        support_config.minimum_hold_duration_s
        + support_config.lower_duration_s
        + support_config.touchdown_timeout_s
        + support_config.center_duration_s
    )
    own_clearance_time_s = (
        support_config.shift_duration_s
        + support_config.stance_load_timeout_s
        + support_config.lift_duration_s
    )
    return_completion_time_s = (
        support_config.lower_duration_s
        + support_config.touchdown_timeout_s
        + support_config.center_duration_s
    )
    trace_end_s = float(samples[-1].timestamp_s) if samples else 0.0
    next_cycle_index = 0
    unexpected_cycle_indices: list[int] = []
    event_results: list[dict[str, object]] = []
    completed_side_counts = {side: 0 for side in _SWING_SIDES}

    for event_index, event in enumerate(events):
        event_start_s = float(event["start_timestamp_s"])
        queued = bool(event["queued_behind_opposite_cycle"])
        acceptance_deadline_s = (
            queued_acceptance_deadline_s
            if queued
            else idle_acceptance_deadline_s
        )
        clearance_deadline_s = acceptance_deadline_s + own_clearance_time_s

        while (
            next_cycle_index < len(cycles)
            and float(cycles[next_cycle_index]["end_timestamp_s"])
            < event_start_s - frame_interval_s
        ):
            unexpected_cycle_indices.append(next_cycle_index)
            next_cycle_index += 1

        candidate_index: int | None = None
        if next_cycle_index < len(cycles):
            candidate = cycles[next_cycle_index]
            if float(candidate["start_timestamp_s"]) <= (
                event_start_s + clearance_deadline_s
            ):
                candidate_index = next_cycle_index
                next_cycle_index += 1

        result: dict[str, object] = {
            **event,
            "index": event_index,
            "acceptance_deadline_s": float(acceptance_deadline_s),
            "clearance_deadline_s": float(clearance_deadline_s),
            "remaining_source_tail_s": max(0.0, trace_end_s - event_start_s),
            "matched_cycle_index": candidate_index,
            "acceptance_latency_s": None,
            "safe_peak_clearance_m": None,
            "actual_peak_clearance_m": None,
            "reached_lift": False,
            "reached_hold": False,
            "return_completed": False,
            "hold_start_timestamp_s": None,
            "return_trigger_timestamp_s": None,
            "return_completion_deadline_timestamp_s": None,
            "return_window_observed": False,
            "abort_reasons": [],
            "failure_reasons": [],
            "status": "failed",
            "passed": False,
            "censored": False,
        }
        failure_reasons: list[str] = []
        if candidate_index is None:
            if trace_end_s - event_start_s < clearance_deadline_s:
                result["status"] = "censored_eof"
                result["censored"] = True
            else:
                failure_reasons.append("missing_support_cycle")
        else:
            cycle = cycles[candidate_index]
            cycle_side = str(cycle["side"])
            result["matched_cycle_side"] = cycle_side
            acceptance_latency_s = max(
                0.0, float(cycle["start_timestamp_s"]) - event_start_s
            )
            result["acceptance_latency_s"] = acceptance_latency_s
            if cycle_side != event["side"]:
                failure_reasons.append("wrong_side_fifo")
            if acceptance_latency_s > acceptance_deadline_s + frame_interval_s:
                failure_reasons.append("acceptance_deadline_exceeded")

            cycle_samples = cycle.get("samples", [])
            assert isinstance(cycle_samples, list)
            deadline_timestamp_s = event_start_s + clearance_deadline_s
            in_window = [
                sample
                for sample in cycle_samples
                if sample.timestamp_s <= deadline_timestamp_s + frame_interval_s
            ]
            phases = {
                sample.support_phase
                for sample in in_window
                if sample.support_phase is not None
            }
            reached_lift = bool(
                phases
                & {
                    SupportPhase.LIFT_SWING.value,
                    SupportPhase.HOLD_SWING.value,
                    SupportPhase.LOWER_SWING.value,
                    SupportPhase.VERIFY_TOUCHDOWN.value,
                    SupportPhase.CENTER_WEIGHT.value,
                }
            )
            reached_hold = bool(
                phases
                & {
                    SupportPhase.HOLD_SWING.value,
                    SupportPhase.LOWER_SWING.value,
                    SupportPhase.VERIFY_TOUCHDOWN.value,
                    SupportPhase.CENTER_WEIGHT.value,
                }
            )
            result["reached_lift"] = reached_lift
            result["reached_hold"] = reached_hold
            return_completed = _cycle_return_completed(cycle)
            result["return_completed"] = return_completed
            hold_or_later_phases = {
                SupportPhase.HOLD_SWING.value,
                SupportPhase.LOWER_SWING.value,
                SupportPhase.VERIFY_TOUCHDOWN.value,
                SupportPhase.CENTER_WEIGHT.value,
            }
            hold_start_s = next(
                (
                    float(sample.timestamp_s)
                    for sample in cycle_samples
                    if sample.support_phase in hold_or_later_phases
                ),
                None,
            )
            result["hold_start_timestamp_s"] = hold_start_s
            event_end_index = int(event["end_index"])
            intent_release_s = (
                float(samples[event_end_index + 1].timestamp_s)
                if event_end_index + 1 < len(samples)
                else None
            )
            return_trigger_s = (
                max(
                    intent_release_s,
                    hold_start_s + support_config.minimum_hold_duration_s,
                )
                if intent_release_s is not None and hold_start_s is not None
                else None
            )
            return_deadline_s = (
                return_trigger_s + return_completion_time_s
                if return_trigger_s is not None
                else None
            )
            return_window_observed = bool(
                return_deadline_s is not None
                and trace_end_s + frame_interval_s >= return_deadline_s
            )
            result["return_trigger_timestamp_s"] = return_trigger_s
            result["return_completion_deadline_timestamp_s"] = return_deadline_s
            result["return_window_observed"] = return_window_observed
            sign = _swing_sign(str(event["side"]))
            clearance_phases = {
                SupportPhase.LIFT_SWING.value,
                SupportPhase.HOLD_SWING.value,
            }
            clearance_samples = [
                sample
                for sample in in_window
                if sample.support_phase in clearance_phases
            ]
            safe_clearances = [
                sign * float(sample.safe_right_minus_left_foot_height_m)
                for sample in clearance_samples
                if sample.safe_right_minus_left_foot_height_m is not None
            ]
            actual_clearances = [
                sign * float(sample.actual_right_minus_left_foot_height_m)
                for sample in clearance_samples
                if sample.actual_right_minus_left_foot_height_m is not None
            ]
            safe_peak = max(safe_clearances, default=None)
            actual_peak = max(actual_clearances, default=None)
            result["safe_peak_clearance_m"] = safe_peak
            result["actual_peak_clearance_m"] = actual_peak
            abort_reasons = sorted(
                {
                    str(sample.support_abort_reason)
                    for sample in cycle_samples
                    if sample.support_abort_reason is not None
                }
            )
            result["abort_reasons"] = abort_reasons
            if abort_reasons:
                failure_reasons.append("support_abort")
            if not reached_hold:
                failure_reasons.append("hold_not_reached")
            if not return_completed:
                failure_reasons.append("support_cycle_incomplete")
            if (
                safe_peak is None
                or safe_peak < policy.minimum_safe_leg_event_clearance_m
            ):
                failure_reasons.append("safe_clearance_below_threshold")
            if (
                actual_peak is None
                or actual_peak < policy.minimum_actual_leg_event_clearance_m
            ):
                failure_reasons.append("actual_clearance_below_threshold")

            full_clearance_window_observed = (
                trace_end_s + frame_interval_s >= deadline_timestamp_s
            )
            hard_failure = bool(
                cycle_side != event["side"]
                or abort_reasons
                or acceptance_latency_s > acceptance_deadline_s + frame_interval_s
                or (
                    full_clearance_window_observed
                    and "hold_not_reached" in failure_reasons
                )
                or (
                    (reached_hold or full_clearance_window_observed)
                    and any(
                        reason
                        in {
                            "safe_clearance_below_threshold",
                            "actual_clearance_below_threshold",
                        }
                        for reason in failure_reasons
                    )
                )
            )
            incomplete_return_censored = bool(
                reached_hold
                and not return_completed
                and not return_window_observed
            )
            preclearance_censored = bool(
                not reached_hold and not full_clearance_window_observed
            )
            if (
                failure_reasons
                and not hard_failure
                and (incomplete_return_censored or preclearance_censored)
            ):
                result["status"] = "censored_eof"
                result["censored"] = True
                failure_reasons.clear()
            elif not failure_reasons:
                result["status"] = "passed"
                result["passed"] = True
                completed_side_counts[str(event["side"])] += 1

        result["failure_reasons"] = failure_reasons
        event_results.append(result)

    unexpected_cycle_indices.extend(range(next_cycle_index, len(cycles)))
    unexpected_cycle_indices = sorted(set(unexpected_cycle_indices))
    failed_events = [
        result for result in event_results if result["status"] == "failed"
    ]
    required_sides_met = all(completed_side_counts[side] >= 1 for side in required)
    return {
        "applicable": True,
        "required_completed_sides": list(required),
        "minimum_completed_events_per_required_side": 1,
        "deadlines": {
            "idle_acceptance_s": float(idle_acceptance_deadline_s),
            "queued_acceptance_s": float(queued_acceptance_deadline_s),
            "own_clearance_s": float(own_clearance_time_s),
            "return_completion_s": float(return_completion_time_s),
            "derivation": {
                "idle_acceptance": "stance_load_timeout_s",
                "queued_acceptance": (
                    "minimum_hold_duration_s + lower_duration_s + "
                    "touchdown_timeout_s + center_duration_s"
                ),
                "own_clearance": (
                    "shift_duration_s + stance_load_timeout_s + lift_duration_s"
                ),
                "return_completion": (
                    "lower_duration_s + touchdown_timeout_s + center_duration_s"
                ),
            },
        },
        "clearance_thresholds_m": {
            "safe_command": policy.minimum_safe_leg_event_clearance_m,
            "actual_state": policy.minimum_actual_leg_event_clearance_m,
        },
        "events": event_results,
        "cycles": [
            _public_cycle(cycle, index) for index, cycle in enumerate(cycles)
        ],
        "completed_side_counts": completed_side_counts,
        "failed_event_count": len(failed_events),
        "censored_event_count": sum(
            result["status"] == "censored_eof" for result in event_results
        ),
        "unexpected_cycle_indices": unexpected_cycle_indices,
        "required_sides_met": required_sides_met,
        "passed": bool(
            required_sides_met
            and not failed_events
            and not unexpected_cycle_indices
        ),
    }


def analyze_trace(
    recorder: _TeleopTraceRecorder,
    *,
    policy: FidelityGatePolicy,
    clip: ClipSpec | None = None,
    support_config: SupportControlConfig | None = None,
) -> dict[str, object]:
    """Turn aligned production samples into coverage, error, and motion evidence."""

    evaluator = recorder.evaluator
    source_frames = len(recorder.samples)
    skeleton_frames = sum(sample.skeleton is not None for sample in recorder.samples)
    safe_frames = sum(sample.safe_positions_rad is not None for sample in recorder.samples)
    actual_frames = sum(sample.actual_positions_rad is not None for sample in recorder.samples)
    paired_frames = sum(
        sample.skeleton is not None
        and sample.safe_positions_rad is not None
        and sample.actual_positions_rad is not None
        for sample in recorder.samples
    )
    eligible = [
        sample
        for sample in recorder.samples
        if sample.skeleton is not None
        and sample.safe_positions_rad is not None
        and sample.actual_positions_rad is not None
        and not sample.safe_stale
    ]
    evaluated_frames = len(eligible)

    stage_group_errors: dict[str, dict[str, list[float]]] = {
        stage: {group: [] for group in GROUPS}
        for stage in ("safe_command", "actual_qpos")
    }
    stage_direction_errors: dict[str, dict[str, list[float]]] = {
        stage: {name: [] for name in DIRECTION_NAMES}
        for stage in ("safe_command", "actual_qpos")
    }
    group_any_counts = {group: 0 for group in GROUPS}
    group_full_counts = {group: 0 for group in GROUPS}
    group_valid_direction_totals = {group: 0 for group in GROUPS}
    direction_valid_counts = {name: 0 for name in DIRECTION_NAMES}
    trajectory_series: dict[str, dict[str, list[object]]] = {
        name: {"timestamps": [], "human": [], "safe": [], "actual": []}
        for name in DIRECTION_NAMES
    }

    if evaluator is not None:
        for sample in eligible:
            assert sample.skeleton is not None
            assert sample.safe_positions_rad is not None
            assert sample.actual_positions_rad is not None
            human = human_anatomical_directions(sample.skeleton)
            # Both stages use the evaluator's private FK data and cannot mutate
            # the live free-base simulation observed above.
            safe = evaluator.robot_directions(sample.safe_positions_rad)
            actual = evaluator.robot_directions(sample.actual_positions_rad)
            fidelities = {
                "safe_command": angular_pose_fidelity(human, safe),
                "actual_qpos": angular_pose_fidelity(human, actual),
            }
            for group, names in GROUPS.items():
                indices = [DIRECTION_NAMES.index(name) for name in names]
                valid_count = int(np.count_nonzero(human.valid[indices]))
                group_valid_direction_totals[group] += valid_count
                group_any_counts[group] += int(valid_count > 0)
                group_full_counts[group] += int(valid_count == len(names))
                for stage, fidelity in fidelities.items():
                    stage_group_errors[stage][group].append(
                        fidelity.mean_error_deg(names)
                    )
            for name in DIRECTION_NAMES:
                direction_valid_counts[name] += int(human.is_valid(name))
            for name in DIRECTION_NAMES:
                for stage, fidelity in fidelities.items():
                    stage_direction_errors[stage][name].append(
                        fidelity.error_deg(name)
                    )
                if human.is_valid(name) and safe.is_valid(name) and actual.is_valid(name):
                    series = trajectory_series[name]
                    series["timestamps"].append(float(sample.timestamp_s))
                    series["human"].append(human.vector(name))
                    series["safe"].append(safe.vector(name))
                    series["actual"].append(actual.vector(name))

    coverage_groups: dict[str, object] = {}
    for group, names in GROUPS.items():
        denominator = evaluated_frames * len(names)
        coverage_groups[group] = {
            "frames_with_any_direction": group_any_counts[group],
            "frames_with_all_directions": group_full_counts[group],
            "any_direction_frame_fraction": _fraction(
                group_any_counts[group], evaluated_frames
            ),
            "all_directions_frame_fraction": _fraction(
                group_full_counts[group], evaluated_frames
            ),
            "valid_direction_fraction": _fraction(
                group_valid_direction_totals[group], denominator
            ),
        }

    fidelity: dict[str, object] = {}
    for stage in ("safe_command", "actual_qpos"):
        fidelity[stage] = {
            "groups": {
                group: _summary_deg(stage_group_errors[stage][group])
                for group in GROUPS
            },
            "directions": {
                name: _summary_deg(stage_direction_errors[stage][name])
                for name in DIRECTION_NAMES
            },
        }

    trajectories = {
        name: _direction_trajectory_metrics(
            series["timestamps"],
            series["human"],
            series["safe"],
            series["actual"],
            policy=policy,
        )
        for name, series in trajectory_series.items()
    }
    result = {
        "coverage": {
            "source_frames": source_frames,
            "skeleton_frames": skeleton_frames,
            "safe_command_frames": safe_frames,
            "actual_qpos_frames": actual_frames,
            "paired_frames": paired_frames,
            "evaluated_nonstale_frames": evaluated_frames,
            "skeleton_source_fraction": _fraction(skeleton_frames, source_frames),
            "paired_source_fraction": _fraction(paired_frames, source_frames),
            "evaluated_source_fraction": _fraction(evaluated_frames, source_frames),
            "evaluated_skeleton_fraction": _fraction(evaluated_frames, skeleton_frames),
            "groups": coverage_groups,
            "directions": {
                name: {
                    "valid_frames": direction_valid_counts[name],
                    "valid_frame_fraction": _fraction(
                        direction_valid_counts[name], evaluated_frames
                    ),
                }
                for name in DIRECTION_NAMES
            },
        },
        "fidelity": fidelity,
        "trajectories": trajectories,
        "instrumentation": {
            "simulation_instance_count": len(recorder.simulation_instances),
            "data_instance_count": len(recorder.data_instances),
            "same_simulation": (
                len(recorder.simulation_instances) == 1
                and len(recorder.data_instances) == 1
            ),
            "equality_constraint_count": (
                next(iter(recorder.equality_constraint_counts))
                if len(recorder.equality_constraint_counts) == 1
                else None
            ),
            "base_joint_type": (
                next(iter(recorder.base_joint_types))
                if len(recorder.base_joint_types) == 1
                else None
            ),
        },
    }
    if clip is not None and clip.required_leg_event_sides:
        result["leg_events"] = _analyze_leg_events(
            recorder.samples,
            required_sides=clip.required_leg_event_sides,
            support_config=(
                load_support_control_config(PROJECT_ROOT / "config" / "balance.yaml")
                if support_config is None
                else support_config
            ),
            policy=policy,
        )
    return result


def _value(mapping: Mapping[str, object], *keys: str) -> object:
    current: object = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _gate(
    name: str,
    value: object,
    threshold: object,
    passed: bool,
    requirement: str,
) -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "requirement": requirement,
        "passed": bool(passed),
    }


def evaluate_semantic_gate(
    clip: ClipSpec,
    trajectories: Mapping[str, object],
    *,
    policy: FidelityGatePolicy,
) -> dict[str, object]:
    """Require every independent clip channel at both command and qpos stages."""

    event_replaced_channels = (
        tuple(
            name
            for name in clip.semantic_channels
            if name in LEG_DIRECTION_NAMES
        )
        if clip.required_leg_event_sides
        else ()
    )
    trajectory_channels = tuple(
        name
        for name in clip.semantic_channels
        if name not in event_replaced_channels
    )
    channels: dict[str, object] = {}
    for name in trajectory_channels:
        trajectory = trajectories.get(name)
        trajectory_map = trajectory if isinstance(trajectory, Mapping) else {}
        reliable = trajectory_map.get("reliable") is True
        stage_results: dict[str, object] = {}
        for stage, amplitude_limit, correlation_limit in (
            (
                "safe_command",
                policy.minimum_safe_amplitude_ratio,
                policy.minimum_safe_correlation,
            ),
            (
                "actual_qpos",
                policy.minimum_actual_amplitude_ratio,
                policy.minimum_actual_correlation,
            ),
        ):
            stage_metrics = trajectory_map.get(stage)
            stage_map = stage_metrics if isinstance(stage_metrics, Mapping) else {}
            amplitude_ratio = _finite_float(stage_map.get("amplitude_ratio"))
            correlation = _finite_float(stage_map.get("best_correlation"))
            lag_s = _finite_float(stage_map.get("best_lag_s"))
            checks = {
                "amplitude": bool(
                    reliable
                    and amplitude_ratio is not None
                    and amplitude_ratio >= amplitude_limit
                ),
                "correlation": bool(
                    reliable
                    and correlation is not None
                    and correlation >= correlation_limit
                ),
            }
            if stage == "actual_qpos":
                checks["lag"] = bool(
                    reliable
                    and lag_s is not None
                    and abs(lag_s) <= policy.maximum_actual_lag_s
                )
            stage_results[stage] = {
                "amplitude_ratio": amplitude_ratio,
                "minimum_amplitude_ratio": amplitude_limit,
                "best_correlation": correlation,
                "minimum_correlation": correlation_limit,
                "best_lag_s": lag_s,
                "maximum_absolute_lag_s": (
                    policy.maximum_actual_lag_s if stage == "actual_qpos" else None
                ),
                "checks": checks,
                "passed": all(checks.values()),
            }
        channel_passed = bool(
            reliable
            and all(
                isinstance(result, Mapping) and result.get("passed") is True
                for result in stage_results.values()
            )
        )
        channels[name] = {
            "reliable": reliable,
            "reason": trajectory_map.get("reason"),
            "human_amplitude_deg": trajectory_map.get("human_amplitude_deg"),
            "stages": stage_results,
            "passed": channel_passed,
        }

    passed_channels = sum(
        isinstance(channel, Mapping) and channel.get("passed") is True
        for channel in channels.values()
    )
    # All declared anatomical channels are required.  Consequently a clip can
    # never pass because one highly active joint hides static or inverted limbs.
    required_channels = len(trajectory_channels)
    return {
        "required_channels": list(trajectory_channels),
        "event_replaced_3d_leg_channels": list(event_replaced_channels),
        "minimum_channels_required": required_channels,
        "passed_channels": int(passed_channels),
        "channels": channels,
        "passed": passed_channels >= required_channels,
    }


def _acceptance_direction_names(clip: ClipSpec) -> tuple[str, ...]:
    """Directions whose visibility and samples can affect task acceptance."""

    replaced = set(LEG_DIRECTION_NAMES) if clip.required_leg_event_sides else set()
    return tuple(name for name in clip.semantic_channels if name not in replaced)


def _direction_coverage(
    coverage: Mapping[str, object],
    names: Sequence[str],
) -> tuple[int | None, float | None]:
    directions = coverage.get("directions")
    direction_map = directions if isinstance(directions, Mapping) else {}
    counts: list[int] = []
    fractions: list[float] = []
    for name in names:
        entry = direction_map.get(name)
        entry_map = entry if isinstance(entry, Mapping) else {}
        count = entry_map.get("valid_frames")
        fraction = _finite_float(entry_map.get("valid_frame_fraction"))
        if not isinstance(count, int) or isinstance(count, bool) or fraction is None:
            return None, None
        counts.append(count)
        fractions.append(fraction)
    if not counts:
        return 0, 1.0
    return min(counts), min(fractions)


def assess_clip(
    clip: ClipSpec,
    evaluation: Mapping[str, object],
    *,
    policy: FidelityGatePolicy,
) -> dict[str, object]:
    metrics = evaluation.get("metrics")
    metrics_map = metrics if isinstance(metrics, Mapping) else {}
    coverage = metrics_map.get("coverage")
    coverage_map = coverage if isinstance(coverage, Mapping) else {}
    instrumentation = metrics_map.get("instrumentation")
    instrumentation_map = (
        instrumentation if isinstance(instrumentation, Mapping) else {}
    )
    stats = evaluation.get("teleop_stats")
    stats_map = stats if isinstance(stats, Mapping) else {}
    gates: list[dict[str, object]] = []
    base_mode = stats_map.get("base_mode")
    gates.append(_gate("base_mode", base_mode, "free", base_mode == "free", "=="))
    observed_frames = stats_map.get("frames")
    gates.append(
        _gate(
            "frames",
            observed_frames,
            clip.expected_frames,
            isinstance(observed_frames, int)
            and not isinstance(observed_frames, bool)
            and observed_frames == clip.expected_frames,
            "==",
        )
    )
    fell = stats_map.get("fell")
    gates.append(_gate("fell", fell, False, fell is False, "=="))
    non_foot_contacts = stats_map.get("maximum_non_foot_ground_contacts")
    gates.append(
        _gate(
            "maximum_non_foot_ground_contacts",
            non_foot_contacts,
            0,
            non_foot_contacts == 0,
            "==",
        )
    )
    abort_count = stats_map.get("support_abort_count")
    gates.append(
        _gate(
            "support_abort_count",
            abort_count,
            0,
            abort_count == 0,
            "==",
        )
    )
    abort_reasons = stats_map.get("support_abort_reasons")
    reasons_empty = isinstance(abort_reasons, (list, tuple)) and not abort_reasons
    gates.append(
        _gate(
            "support_abort_reasons",
            abort_reasons,
            [],
            reasons_empty,
            "==",
        )
    )
    same_simulation = instrumentation_map.get("same_simulation")
    gates.append(
        _gate(
            "same_simulation",
            same_simulation,
            True,
            same_simulation is True,
            "==",
        )
    )
    equality_constraint_count = instrumentation_map.get(
        "equality_constraint_count"
    )
    gates.append(
        _gate(
            "equality_constraint_count",
            equality_constraint_count,
            0,
            equality_constraint_count == 0,
            "==",
        )
    )
    base_joint_type = instrumentation_map.get("base_joint_type")
    gates.append(
        _gate(
            "base_joint_type",
            base_joint_type,
            "free",
            base_joint_type == "free",
            "==",
        )
    )
    expected_video = _project_path(clip.path.resolve())
    observed_video = evaluation.get("video")
    gates.append(
        _gate(
            "video_path",
            observed_video,
            expected_video,
            observed_video == expected_video,
            "==",
        )
    )
    expected_video_sha256 = _sha256(clip.path.resolve())
    observed_video_sha256 = evaluation.get("video_sha256")
    gates.append(
        _gate(
            "video_sha256",
            observed_video_sha256,
            expected_video_sha256,
            observed_video_sha256 == expected_video_sha256,
            "==",
        )
    )
    calibration = stats_map.get("calibration")
    calibration_map = calibration if isinstance(calibration, Mapping) else {}
    explicit_calibration = clip.calibration_video is not None
    expected_calibration_mode = (
        "explicit_replay_frame" if explicit_calibration else "automatic_window"
    )
    expected_calibration_source = (
        _project_path(clip.calibration_video.resolve())
        if clip.calibration_video is not None
        else None
    )
    expected_calibration_sha256 = (
        _sha256(clip.calibration_video.resolve())
        if clip.calibration_video is not None
        else None
    )
    for name, observed, expected in (
        ("mode", calibration_map.get("mode"), expected_calibration_mode),
        ("source", calibration_map.get("source"), expected_calibration_source),
        (
            "source_sha256",
            calibration_map.get("source_sha256"),
            expected_calibration_sha256,
        ),
        ("frame_index", calibration_map.get("frame_index"), clip.calibration_frame),
    ):
        gates.append(
            _gate(
                f"calibration_{name}",
                observed,
                expected,
                observed == expected,
                "==",
            )
        )
    source_fraction = _finite_float(coverage_map.get("evaluated_source_fraction"))
    source_frames = coverage_map.get("source_frames")
    gates.append(
        _gate(
            "source_frames",
            source_frames,
            clip.expected_frames,
            isinstance(source_frames, int)
            and not isinstance(source_frames, bool)
            and source_frames == clip.expected_frames,
            "==",
        )
    )
    gates.append(
        _gate(
            "evaluated_source_fraction",
            source_fraction,
            policy.minimum_evaluated_source_fraction,
            source_fraction is not None
            and source_fraction >= policy.minimum_evaluated_source_fraction,
            ">=",
        )
    )
    global_gate_count = len(gates)
    acceptance_directions = _acceptance_direction_names(clip)
    for name in acceptance_directions:
        frame_count, valid_fraction = _direction_coverage(coverage_map, (name,))
        gates.append(
            _gate(
                f"coverage_{name}_valid_frames",
                frame_count,
                policy.minimum_group_direction_frames,
                frame_count is not None
                and frame_count >= policy.minimum_group_direction_frames,
                ">=",
            )
        )
        gates.append(
            _gate(
                f"coverage_{name}_valid_frame_fraction",
                valid_fraction,
                policy.minimum_group_valid_direction_fraction,
                valid_fraction is not None
                and valid_fraction
                >= policy.minimum_group_valid_direction_fraction,
                ">=",
            )
        )

    fidelity = metrics_map.get("fidelity")
    fidelity_map = fidelity if isinstance(fidelity, Mapping) else {}
    for stage, limits in (
        ("safe_command", SAFE_P90_LIMITS_DEG),
        ("actual_qpos", ACTUAL_P90_LIMITS_DEG),
    ):
        for group, limit in limits.items():
            summary = _value(fidelity_map, stage, "groups", group)
            summary_map = summary if isinstance(summary, Mapping) else {}
            if group not in clip.gated_p90_groups:
                # Every group remains fully reported.  Only groups declared by
                # the a-priori task contract may affect acceptance; unrelated
                # or unsupported anatomy remains descriptive evidence.
                continue
            count = summary_map.get("count")
            gates.append(
                _gate(
                    f"{stage}_{group}_count",
                    count,
                    policy.minimum_group_direction_frames,
                    isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= policy.minimum_group_direction_frames,
                    ">=",
                )
            )
            p90 = _finite_float(
                _value(fidelity_map, stage, "groups", group, "p90_deg")
            )
            gates.append(
                _gate(
                    f"{stage}_{group}_p90_deg",
                    p90,
                    limit,
                    p90 is not None and p90 <= limit,
                    "<=",
                )
            )

    trajectories = metrics_map.get("trajectories")
    trajectory_map = trajectories if isinstance(trajectories, Mapping) else {}
    semantic = evaluate_semantic_gate(clip, trajectory_map, policy=policy)
    leg_events = metrics_map.get("leg_events")
    leg_event_map = leg_events if isinstance(leg_events, Mapping) else {}
    leg_event_passed = (
        not clip.required_leg_event_sides
        or leg_event_map.get("passed") is True
    )
    if clip.required_leg_event_sides:
        gates.append(
            _gate(
                "unilateral_leg_event_semantics",
                leg_event_map.get("passed"),
                True,
                leg_event_passed,
                "==",
            )
        )
    descriptive_warnings: list[dict[str, object]] = []
    coverage_groups = coverage_map.get("groups")
    coverage_group_map = (
        coverage_groups if isinstance(coverage_groups, Mapping) else {}
    )
    acceptance_group_names = set(clip.gated_p90_groups)
    for group in GROUPS:
        group_coverage = coverage_group_map.get(group)
        group_map = group_coverage if isinstance(group_coverage, Mapping) else {}
        valid_fraction = _finite_float(group_map.get("valid_direction_fraction"))
        frame_count = group_map.get("frames_with_any_direction")
        if (
            group not in acceptance_group_names
            and (
                valid_fraction is None
                or valid_fraction < policy.minimum_group_valid_direction_fraction
                or not isinstance(frame_count, int)
                or isinstance(frame_count, bool)
                or frame_count < policy.minimum_group_direction_frames
            )
        ):
            descriptive_warnings.append(
                {
                    "name": "insufficient_descriptive_coverage",
                    "group": group,
                    "frames_with_any_direction": frame_count,
                    "valid_direction_fraction": valid_fraction,
                    "acceptance_effect": "none",
                }
            )
    global_gates = gates[:global_gate_count]
    task_gates = gates[global_gate_count:]
    global_passed = all(gate["passed"] for gate in global_gates)
    task_passed = bool(
        all(gate["passed"] for gate in task_gates)
        and semantic["passed"] is True
        and leg_event_passed
    )
    task_contract = {
        "task_label": clip.task_label,
        "capability": clip.capability,
        "required_semantic_channels": list(clip.semantic_channels),
        "acceptance_direction_channels": list(acceptance_directions),
        "required_leg_event_sides": list(clip.required_leg_event_sides),
        "acceptance_p90_groups": list(clip.gated_p90_groups),
        "unsupported_limitations": list(clip.unsupported_limitations),
    }
    return {
        **dict(evaluation),
        "gates": gates,
        "global_acceptance": {
            "gates": global_gates,
            "passed": global_passed,
        },
        "task_contract": task_contract,
        "task_acceptance": {
            "gates": task_gates,
            "trajectory_semantics": semantic,
            "leg_event_semantics": dict(leg_event_map),
            "passed": task_passed,
        },
        # Compatibility alias retained while schema v2 consumers migrate.
        "semantic_gate": semantic,
        "leg_event_gate": dict(leg_event_map),
        "gated_p90_groups": list(clip.gated_p90_groups),
        "descriptive_only_3d_groups": [
            group for group in GROUPS if group not in clip.gated_p90_groups
        ],
        "whole_body_fidelity": {
            "role": "descriptive_full_body_context",
            "acceptance_effect": (
                "none except acceptance channels/groups predeclared in task_contract"
            ),
            "fidelity": dict(fidelity_map),
            "coverage": dict(coverage_map),
            "unsupported_limitations": list(clip.unsupported_limitations),
            "warnings": descriptive_warnings,
        },
        "passed": bool(global_passed and task_passed),
    }


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


def _teleop_arguments(clip: ClipSpec, *, max_frames: int) -> list[str]:
    arguments = [
        "--source",
        clip.source,
        "--headless",
        "--free-base",
        "--balance-controller",
        "--retargeting",
        "ik",
        "--settle-seconds",
        "0",
    ]
    if max_frames > 0:
        arguments.extend(("--max-frames", str(max_frames)))
    if clip.source == "mp4":
        if clip.demo_video is None:
            raise ValueError(f"bundled clip lacks demo_video: {clip.name}")
        arguments.extend(("--demo-video", clip.demo_video))
    else:
        arguments.extend(("--video-path", str(clip.path)))
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


def _teleop_summary(stats: TeleopStats) -> dict[str, object]:
    return {
        "base_mode": stats.base_mode,
        "frames": int(stats.frames),
        "skeleton_frames": int(stats.skeleton_frames),
        "stale_commands": int(stats.stale_commands),
        "fell": bool(stats.fell),
        "minimum_base_height_m": float(stats.minimum_base_height_m),
        "maximum_tilt_deg": float(np.degrees(stats.maximum_tilt_rad)),
        "support_transitions": int(stats.support_transitions),
        "maximum_non_foot_ground_contacts": int(
            stats.maximum_non_foot_ground_contacts
        ),
        "support_abort_count": int(stats.support_abort_count),
        "support_abort_reasons": list(stats.support_abort_reasons),
        "settling_requested_s": float(stats.settling_requested_s),
        "settling_elapsed_s": float(stats.settling_elapsed_s),
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
    }


def evaluate_clip(
    clip: ClipSpec,
    *,
    policy: FidelityGatePolicy,
    max_frames: int = 0,
    runner: Callable[[argparse.Namespace], TeleopStats] | None = None,
) -> dict[str, object]:
    """Run one real production replay, then assess its aligned frame trace."""

    path = clip.path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fidelity video does not exist: {path}")
    if max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    arguments = _teleop_arguments(clip, max_frames=max_frames)
    parsed = build_teleop_parser().parse_args(arguments)
    if parsed.settle_seconds != 0.0:
        raise AssertionError("safe-pose fidelity must never include settling")
    execute = run_teleop if runner is None else runner
    with _record_end_to_end_trace() as recorder:
        stats = execute(parsed)
    if recorder.evaluator is None:
        raise RuntimeError("production run emitted no instrumented MuJoCo samples")
    if len(recorder.samples) < stats.frames:
        raise RuntimeError(
            "instrumentation emitted fewer samples than the operational replay"
        )
    # Explicit calibration uses the same real MediaPipe method and is therefore
    # visible to instrumentation. It is not part of the main source and has no
    # safe/actual command pair; trim it using TeleopStats' authoritative frame
    # count before computing source coverage or trajectories.
    recorder.samples = recorder.samples[-stats.frames :] if stats.frames else []
    raw = {
        "clip": clip.name,
        "video": _project_path(path),
        "video_size_bytes": path.stat().st_size,
        "video_sha256": _sha256(path),
        "teleop_arguments": arguments,
        "teleop_stats": _teleop_summary(stats),
        "metrics": analyze_trace(
            recorder,
            policy=policy,
            clip=clip,
            support_config=load_support_control_config(
                PROJECT_ROOT / "config" / "balance.yaml"
            ),
        ),
    }
    return assess_clip(clip, raw, policy=policy)


def evaluate_suite(
    clips: Sequence[ClipSpec],
    *,
    policy: FidelityGatePolicy,
    max_frames: int = 0,
    evaluator: Callable[..., dict[str, object]] = evaluate_clip,
) -> list[dict[str, object]]:
    return [
        evaluator(clip, policy=policy, max_frames=max_frames)
        for clip in clips
    ]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _runtime_input_paths(clips: Sequence[ClipSpec]) -> set[Path]:
    return {
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "robot_human_interface" / "app" / "teleop.py",
        PROJECT_ROOT / "src" / "robot_human_interface" / "camera" / "__init__.py",
        PROJECT_ROOT / "src" / "robot_human_interface" / "camera" / "sources.py",
        PROJECT_ROOT / "src" / "robot_human_interface" / "control" / "__init__.py",
        PROJECT_ROOT / "src" / "robot_human_interface" / "control" / "standing.py",
        PROJECT_ROOT / "src" / "robot_human_interface" / "control" / "support.py",
        PROJECT_ROOT / "src" / "robot_human_interface" / "control" / "human_intent.py",
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
        PROJECT_ROOT / "config" / "balance.yaml",
        PROJECT_ROOT / "config" / "camera.yaml",
        PROJECT_ROOT / "config" / "joints.yaml",
        PROJECT_ROOT / "config" / "retargeting.yaml",
        PROJECT_ROOT / "models" / "humanoid" / "scene_free.xml",
        PROJECT_ROOT / "models" / "humanoid" / "robot.xml",
        PROJECT_ROOT / "assets" / "models" / "pose_landmarker_full.task",
        *(clip.path.resolve() for clip in clips),
        *(
            clip.calibration_video.resolve()
            for clip in clips
            if clip.calibration_video is not None
        ),
    }


def _runtime_hashes(clips: Sequence[ClipSpec]) -> dict[str, str]:
    paths = _runtime_input_paths(clips)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"runtime fidelity inputs are missing: {missing}")
    return {
        _project_path(path): _sha256(path)
        for path in sorted(paths, key=lambda item: str(item).lower())
    }


def _policy_violations(
    policy: FidelityGatePolicy,
) -> dict[str, dict[str, float | int | str]]:
    canonical = FidelityGatePolicy()
    violations: dict[str, dict[str, float | int | str]] = {}
    for name in _MINIMUM_POLICY_FIELDS:
        observed = getattr(policy, name)
        required = getattr(canonical, name)
        if observed < required:
            violations[name] = {
                "observed": observed,
                "canonical": required,
                "required_relation": ">=",
            }
    for name in _MAXIMUM_POLICY_FIELDS:
        observed = getattr(policy, name)
        required = getattr(canonical, name)
        if observed > required:
            violations[name] = {
                "observed": observed,
                "canonical": required,
                "required_relation": "<=",
            }
    return violations


def _runtime_snapshot_violations(
    snapshot: Mapping[str, str], clips: Sequence[ClipSpec]
) -> dict[str, object]:
    expected_keys = {
        _project_path(path) for path in _runtime_input_paths(clips)
    }
    observed_keys = {str(key) for key in snapshot}
    invalid_hashes = sorted(
        str(key)
        for key, value in snapshot.items()
        if not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    )
    return {
        "missing_keys": sorted(expected_keys - observed_keys),
        "unexpected_keys": sorted(observed_keys - expected_keys),
        "invalid_sha256_keys": invalid_hashes,
    }


def build_report(
    results: Sequence[Mapping[str, object]],
    clips: Sequence[ClipSpec],
    *,
    policy: FidelityGatePolicy,
    max_frames: int,
    runtime_hashes_at_start: Mapping[str, str],
    runtime_hashes_at_completion: Mapping[str, str],
) -> dict[str, object]:
    canonical_clips = default_clips()
    exact_canonical_matrix = tuple(clips) == canonical_clips
    complete_result_count = len(results) == len(canonical_clips)
    result_names = [result.get("clip") for result in results]
    canonical_names = [clip.name for clip in canonical_clips]
    policy_violations = _policy_violations(policy)
    start_snapshot_violations = _runtime_snapshot_violations(
        runtime_hashes_at_start, clips
    )
    completion_snapshot_violations = _runtime_snapshot_violations(
        runtime_hashes_at_completion, clips
    )
    start_snapshot_complete = not any(start_snapshot_violations.values())
    completion_snapshot_complete = not any(completion_snapshot_violations.values())
    suite_gates = [
        _gate(
            "exact_canonical_clip_matrix",
            [clip.name for clip in clips],
            canonical_names,
            exact_canonical_matrix,
            "==",
        ),
        _gate(
            "complete_clip_result_count",
            len(results),
            len(canonical_clips),
            complete_result_count,
            "==",
        ),
        _gate(
            "exact_canonical_result_matrix",
            result_names,
            canonical_names,
            result_names == canonical_names,
            "==",
        ),
        _gate(
            "full_length_replays",
            max_frames,
            0,
            max_frames == 0,
            "==",
        ),
        _gate(
            "canonical_or_stricter_policy",
            policy_violations,
            {},
            not policy_violations,
            "==",
        ),
        _gate(
            "runtime_snapshot_at_start_complete",
            start_snapshot_violations,
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "invalid_sha256_keys": [],
            },
            start_snapshot_complete,
            "==",
        ),
        _gate(
            "runtime_snapshot_at_completion_complete",
            completion_snapshot_violations,
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "invalid_sha256_keys": [],
            },
            completion_snapshot_complete,
            "==",
        ),
        _gate(
            "runtime_inputs_unchanged",
            runtime_hashes_at_completion,
            runtime_hashes_at_start,
            dict(runtime_hashes_at_completion) == dict(runtime_hashes_at_start),
            "==",
        ),
    ]
    return {
        "schema_version": 2,
        "description": (
            "End-to-end human task-space fidelity of the final safety-projected "
            "command and measured free-base MuJoCo qpos; raw IK is not reported."
        ),
        "measurement_contract": {
            "safe_command": (
                "last motor target applied after standing balance and support FSM "
                "during each input frame"
            ),
            "actual_qpos": (
                "measured canonical joint qpos after the last MuJoCo step driven "
                "by that same input frame"
            ),
            "frame_alignment": (
                "MediaPipePoseLandmarker.estimate starts a frame; the next estimate "
                "closes it, retaining the final apply/step pair"
            ),
            "settling_included": False,
            "base_mode": "free",
            "unilateral_leg_semantics": (
                "production calibrated/debounced lift intent matched FIFO to the "
                "support FSM side and measured safe/actual foot clearance"
            ),
            "task_acceptance": (
                "only the semantic directions, event sides, and p90 groups "
                "predeclared by each canonical task contract affect task passage"
            ),
            "whole_body_fidelity": (
                "all anatomical groups remain reported; non-task and unsupported "
                "channels are descriptive limitations/warnings and never silently "
                "become acceptance gates"
            ),
        },
        "threshold_policy": {
            **asdict(policy),
            "safe_p90_limits_deg": SAFE_P90_LIMITS_DEG,
            "actual_p90_limits_deg": ACTUAL_P90_LIMITS_DEG,
            "rationale": (
                "Directional p90 limits stay materially below orthogonal tracking; "
                "semantic motion must reproduce at least 20-25% of human amplitude, "
                "show positive moderate correlation, and follow within 0.5 s. The "
                "correlation search extends beyond that gate so late tracking can "
                "fail rather than win at the search boundary. Calibrated unilateral "
                "leg clips instead use FIFO intent/FSM/clearance events because "
                "monocular depth is not an independent leg-direction oracle. Every "
                "declared task must pass, while coverage and sample-count gates apply "
                "only to its predeclared acceptance channels/groups. Limits and task "
                "contracts were declared independently of a generated result artifact."
            ),
        },
        "configuration": {
            "max_frames": max_frames,
            "clips": [clip.name for clip in clips],
            "task_contracts": [
                {
                    "clip": clip.name,
                    "expected_frames": clip.expected_frames,
                    "task_label": clip.task_label,
                    "capability": clip.capability,
                    "semantic_channels": list(clip.semantic_channels),
                    "required_leg_event_sides": list(clip.required_leg_event_sides),
                    "acceptance_p90_groups": list(clip.gated_p90_groups),
                    "unsupported_limitations": list(clip.unsupported_limitations),
                }
                for clip in clips
            ],
            "canonical_complete_matrix": bool(
                exact_canonical_matrix and complete_result_count and max_frames == 0
            ),
            "versions": {
                "mediapipe": _package_version("mediapipe"),
                "mujoco": _package_version("mujoco"),
                "numpy": _package_version("numpy"),
            },
            "runtime_input_sha256": dict(runtime_hashes_at_start),
            "runtime_input_sha256_at_completion": dict(
                runtime_hashes_at_completion
            ),
            "runtime_inputs_unchanged_during_run": bool(
                dict(runtime_hashes_at_start)
                == dict(runtime_hashes_at_completion)
            ),
        },
        "suite_gates": suite_gates,
        "clips": [dict(result) for result in results],
        "overall_passed": bool(results)
        and all(gate["passed"] for gate in suite_gates)
        and all(result.get("passed") is True for result in results),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        action="append",
        choices=tuple(clip.name for clip in default_clips()),
        default=[],
        help="Named clip to evaluate; repeat as needed (default: all six).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Debug-only input-frame limit; zero evaluates each complete clip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path (default: artifacts/safe-pose-fidelity.json).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    available = {clip.name: clip for clip in default_clips()}
    clips = (
        tuple(available[name] for name in args.clip)
        if args.clip
        else tuple(available.values())
    )
    missing = [clip.path for clip in clips if not clip.path.is_file()]
    if missing:
        raise FileNotFoundError(f"selected fidelity videos do not exist: {missing}")
    policy = FidelityGatePolicy()
    runtime_hashes_at_start = _runtime_hashes(clips)
    results = evaluate_suite(
        clips,
        policy=policy,
        max_frames=args.max_frames,
    )
    runtime_hashes_at_completion = _runtime_hashes(clips)
    report = build_report(
        results,
        clips,
        policy=policy,
        max_frames=args.max_frames,
        runtime_hashes_at_start=runtime_hashes_at_start,
        runtime_hashes_at_completion=runtime_hashes_at_completion,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"safe-pose-fidelity report: {output}")
    for result in results:
        leg_gate = result.get("leg_event_gate")
        leg_suffix = ""
        if isinstance(leg_gate, Mapping) and leg_gate.get("applicable") is True:
            leg_suffix = (
                f" leg_events={'PASS' if leg_gate.get('passed') is True else 'FAIL'}"
                f" completed={leg_gate.get('completed_side_counts')}"
                f" failed={leg_gate.get('failed_event_count')}"
                f" censored={leg_gate.get('censored_event_count')}"
            )
        print(
            f"{result['clip']}: {'PASS' if result['passed'] else 'FAIL'} "
            f"evaluated={result['metrics']['coverage']['evaluated_nonstale_frames']} "
            f"semantic={result['semantic_gate']['passed_channels']}/"
            f"{result['semantic_gate']['minimum_channels_required']}"
            f"{leg_suffix}"
        )
    return 0 if report["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
