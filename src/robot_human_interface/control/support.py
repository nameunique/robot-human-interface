"""Contact-gated support transitions expressed only as motor-angle targets.

The state machine is deliberately independent of MuJoCo.  It consumes a
canonical 20-angle command plus the two measured sole loads and overlays a
verified, smoothly-ramped support motion.  Consequently the same code can sit
after :class:`StandingBalanceController` in simulation and after an equivalent
IMU/encoder balance layer on the real robot.

No phase writes generalized coordinates, applies external forces, or changes
constraints.  A requested leg lift is allowed only after the opposite foot has
carried a confirmed share of the measured load.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import pi
from pathlib import Path
from time import monotonic
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import NDArray

from robot_human_interface.skeleton import JOINT_NAMES, RobotJointCommand


FloatArray = NDArray[np.float64]
UPPER_BODY_INDICES = np.asarray((0, 1, 2, 3, 4, 5, 18, 19), dtype=np.int64)
LOWER_BODY_INDICES = np.arange(6, 18, dtype=np.int64)
RIGHT_LEG_INDICES = np.asarray((6, 8, 10, 12, 14, 16), dtype=np.int64)
LEFT_LEG_INDICES = np.asarray((7, 9, 11, 13, 15, 17), dtype=np.int64)


class SupportIntent(str, Enum):
    """Operator-level support request, independent of controller phase."""

    DOUBLE_SUPPORT = "double_support"
    RIGHT_SWING = "right_swing"
    LEFT_SWING = "left_swing"


class SupportPhase(str, Enum):
    """Internally enforced transition sequence for a support request."""

    DOUBLE_SUPPORT = "double_support"
    SHIFT_WEIGHT = "shift_weight"
    VERIFY_STANCE = "verify_stance"
    LIFT_SWING = "lift_swing"
    HOLD_SWING = "hold_swing"
    LOWER_SWING = "lower_swing"
    VERIFY_TOUCHDOWN = "verify_touchdown"
    CENTER_WEIGHT = "center_weight"


@dataclass(frozen=True, slots=True)
class SupportControlConfig:
    """Timing and force gates for the support transition state machine."""

    shift_duration_s: float = 2.0
    load_confirm_duration_s: float = 0.15
    stance_load_timeout_s: float = 1.5
    lift_duration_s: float = 1.5
    minimum_hold_duration_s: float = 0.25
    lower_duration_s: float = 3.0
    touchdown_confirm_duration_s: float = 0.15
    touchdown_preload_duration_s: float = 8.0
    touchdown_timeout_s: float = 8.0
    center_duration_s: float = 3.0
    support_loss_grace_s: float = 0.08
    min_stance_force_n: float = 12.0
    min_stance_load_fraction: float = 0.65
    max_swing_load_fraction: float = 0.35
    min_touchdown_force_n: float = 4.0
    min_touchdown_total_force_n: float = 20.0
    early_touchdown_max_swing_progress: float = 0.45
    start_max_tilt_rad: float = 12.0 * pi / 180.0
    active_max_tilt_rad: float = 18.0 * pi / 180.0
    start_max_angular_speed_rad_s: float = 1.0
    active_max_angular_speed_rad_s: float = 3.0
    swing_reference_blend: float = 0.25
    max_swing_reference_delta_rad: float = 12.0 * pi / 180.0
    single_support_upper_body_scale: float = 0.15
    upper_body_rate_limit_rad_s: float = 2.5
    lower_body_rate_limit_rad_s: float = 1.2
    cancel_on_stale_reference: bool = True

    def __post_init__(self) -> None:
        durations = (
            self.shift_duration_s,
            self.load_confirm_duration_s,
            self.stance_load_timeout_s,
            self.lift_duration_s,
            self.lower_duration_s,
            self.touchdown_confirm_duration_s,
            self.touchdown_preload_duration_s,
            self.touchdown_timeout_s,
            self.center_duration_s,
            self.support_loss_grace_s,
        )
        if not np.isfinite(durations).all() or any(value <= 0.0 for value in durations):
            raise ValueError("support phase durations must be finite and positive")
        if (
            not np.isfinite(self.minimum_hold_duration_s)
            or self.minimum_hold_duration_s < 0.0
        ):
            raise ValueError("minimum_hold_duration_s must be finite and non-negative")
        if self.stance_load_timeout_s < self.load_confirm_duration_s:
            raise ValueError("stance_load_timeout_s must cover load_confirm_duration_s")
        if not np.isfinite(self.min_stance_force_n) or self.min_stance_force_n <= 0.0:
            raise ValueError("min_stance_force_n must be finite and positive")
        if not np.isfinite(self.min_touchdown_force_n) or self.min_touchdown_force_n <= 0.0:
            raise ValueError("min_touchdown_force_n must be finite and positive")
        if (
            not np.isfinite(self.min_touchdown_total_force_n)
            or self.min_touchdown_total_force_n <= 0.0
        ):
            raise ValueError("min_touchdown_total_force_n must be finite and positive")
        if not 0.0 < self.early_touchdown_max_swing_progress < 1.0:
            raise ValueError(
                "early_touchdown_max_swing_progress must be within (0, 1)"
            )
        stability_limits = (
            self.start_max_tilt_rad,
            self.active_max_tilt_rad,
            self.start_max_angular_speed_rad_s,
            self.active_max_angular_speed_rad_s,
            self.max_swing_reference_delta_rad,
        )
        if not np.isfinite(stability_limits).all() or any(
            value <= 0.0 for value in stability_limits
        ):
            raise ValueError("support stability limits must be finite and positive")
        if self.start_max_tilt_rad >= self.active_max_tilt_rad:
            raise ValueError("start_max_tilt_rad must be below active_max_tilt_rad")
        if self.active_max_tilt_rad >= pi:
            raise ValueError("active_max_tilt_rad must be below pi")
        if (
            self.start_max_angular_speed_rad_s
            >= self.active_max_angular_speed_rad_s
        ):
            raise ValueError(
                "start_max_angular_speed_rad_s must be below "
                "active_max_angular_speed_rad_s"
            )
        if (
            not np.isfinite(self.swing_reference_blend)
            or not 0.0 <= self.swing_reference_blend <= 1.0
        ):
            raise ValueError("swing_reference_blend must be within [0, 1]")
        if (
            not np.isfinite(self.single_support_upper_body_scale)
            or not 0.0 <= self.single_support_upper_body_scale <= 1.0
        ):
            raise ValueError("single_support_upper_body_scale must be within [0, 1]")
        if (
            not np.isfinite(self.upper_body_rate_limit_rad_s)
            or self.upper_body_rate_limit_rad_s <= 0.0
            or not np.isfinite(self.lower_body_rate_limit_rad_s)
            or self.lower_body_rate_limit_rad_s <= 0.0
        ):
            raise ValueError("support motor-target rate limits must be finite and positive")
        if not 0.5 < self.min_stance_load_fraction <= 1.0:
            raise ValueError("min_stance_load_fraction must be within (0.5, 1]")
        if not 0.0 <= self.max_swing_load_fraction < 0.5:
            raise ValueError("max_swing_load_fraction must be within [0, 0.5)")


@dataclass(frozen=True, slots=True)
class SupportDiagnostics:
    """Observable phase and load evidence for logs, UI, and experiments."""

    requested_intent: SupportIntent
    active_intent: SupportIntent
    phase: SupportPhase
    shift_progress: float
    swing_progress: float
    stance_force_n: float
    swing_force_n: float
    stance_load_fraction: float
    support_ready: bool
    touchdown_ready: bool
    base_tilt_rad: float
    base_angular_speed_rad_s: float
    start_stable: bool
    active_stable: bool
    blocked_intent: SupportIntent | None
    abort_reason: str | None
    applied_offset_rad: FloatArray

    def __post_init__(self) -> None:
        offset = np.asarray(self.applied_offset_rad, dtype=np.float64)
        if offset.shape != (len(JOINT_NAMES),) or not np.isfinite(offset).all():
            raise ValueError("applied_offset_rad must be a finite canonical vector")
        if (
            not np.isfinite(self.base_tilt_rad)
            or not 0.0 <= self.base_tilt_rad <= pi
        ):
            raise ValueError("base_tilt_rad must be finite and within [0, pi]")
        if (
            not np.isfinite(self.base_angular_speed_rad_s)
            or self.base_angular_speed_rad_s < 0.0
        ):
            raise ValueError(
                "base_angular_speed_rad_s must be finite and non-negative"
            )
        offset = offset.copy()
        offset.setflags(write=False)
        object.__setattr__(self, "applied_offset_rad", offset)


def _degrees_profile(indices: Sequence[int], values_deg: Sequence[float]) -> FloatArray:
    result = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    result[np.asarray(indices, dtype=np.int64)] = np.radians(values_deg)
    result.setflags(write=False)
    return result


# Verified in the current free-base MuJoCo proxy.  These are actuator-index
# offsets in the canonical Unity/WebSocket order, not modifications to qpos.
_RIGHT_SWING_SHIFT = _degrees_profile((8, 9, 16, 17), (25.0, -25.0, -25.0, 25.0))
_RIGHT_SWING_LIFT = _degrees_profile((10, 12, 14), (20.0, 20.0, -20.0))
_LEFT_SWING_SHIFT = _degrees_profile((8, 9, 16, 17), (-28.0, 28.0, 28.0, -28.0))
_LEFT_SWING_LIFT = _degrees_profile((11, 13, 15), (20.0, 30.0, -10.0))


def support_offsets(intent: SupportIntent | str) -> tuple[FloatArray, FloatArray]:
    """Return copies of the verified ``(weight_shift, swing_lift)`` offsets."""

    accepted = SupportIntent(intent)
    if accepted is SupportIntent.RIGHT_SWING:
        return _RIGHT_SWING_SHIFT.copy(), _RIGHT_SWING_LIFT.copy()
    if accepted is SupportIntent.LEFT_SWING:
        return _LEFT_SWING_SHIFT.copy(), _LEFT_SWING_LIFT.copy()
    return np.zeros(len(JOINT_NAMES)), np.zeros(len(JOINT_NAMES))


def load_support_control_config(path: str | Path | None) -> SupportControlConfig:
    """Load the optional ``support_control`` section of ``balance.yaml``."""

    if path is None or not Path(path).is_file():
        return SupportControlConfig()
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, Mapping):
        raise ValueError("balance YAML root must be a mapping")
    settings = document.get("support_control", {})
    if not isinstance(settings, Mapping):
        raise ValueError("support_control settings must be a mapping")
    allowed = set(SupportControlConfig.__dataclass_fields__)
    return SupportControlConfig(**{key: value for key, value in settings.items() if key in allowed})


def _smoothstep(progress: float) -> float:
    value = float(np.clip(progress, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class SupportStateMachine:
    """Safely sequence one-leg targets after a balanced standing command.

    Typical use::

        standing_command = standing.update(human_reference, state, dt_s=dt)
        support.set_intent(SupportIntent.RIGHT_SWING)
        motor_command = support.update(standing_command, state, dt_s=dt)

    ``update`` deliberately has the same command/state/dt shape as
    ``StandingBalanceController.update``.  It returns only canonical bounded
    motor positions and never mutates the state object or simulator.
    """

    def __init__(
        self,
        lower_limits_rad: Sequence[float],
        upper_limits_rad: Sequence[float],
        config: SupportControlConfig | None = None,
        *,
        home_positions_rad: Sequence[float] | None = None,
    ) -> None:
        self.config = config or SupportControlConfig()
        self._lower = self._vector(lower_limits_rad, "lower_limits_rad")
        self._upper = self._vector(upper_limits_rad, "upper_limits_rad")
        if np.any(self._lower >= self._upper):
            raise ValueError("every lower motor limit must be below its upper limit")
        self._home = (
            None
            if home_positions_rad is None
            else self._vector(home_positions_rad, "home_positions_rad")
        )
        if self._home is not None and (
            np.any(self._home < self._lower) or np.any(self._home > self._upper)
        ):
            raise ValueError("home motor angles must lie inside configured limits")
        self._rate_limits = np.full(
            len(JOINT_NAMES), self.config.lower_body_rate_limit_rad_s
        )
        self._rate_limits[UPPER_BODY_INDICES] = (
            self.config.upper_body_rate_limit_rad_s
        )
        self.reset()

    @staticmethod
    def _vector(values: Sequence[float], name: str) -> FloatArray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (len(JOINT_NAMES),) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite {len(JOINT_NAMES)}-vector")
        return result.copy()

    @classmethod
    def from_simulation(
        cls,
        simulation: object,
        config: SupportControlConfig | None = None,
    ) -> "SupportStateMachine":
        return cls(
            getattr(simulation, "lower_limits_rad"),
            getattr(simulation, "upper_limits_rad"),
            config,
            home_positions_rad=getattr(simulation, "home_positions_rad"),
        )

    def reset(self) -> None:
        self._requested_intent = SupportIntent.DOUBLE_SUPPORT
        self._active_intent = SupportIntent.DOUBLE_SUPPORT
        self._phase = SupportPhase.DOUBLE_SUPPORT
        self._phase_elapsed_s = 0.0
        self._load_confirm_elapsed_s = 0.0
        self._touchdown_confirm_elapsed_s = 0.0
        self._support_unsafe_elapsed_s = 0.0
        self._shift_progress = 0.0
        self._swing_progress = 0.0
        self._center_start_shift_progress = 1.0
        self._blocked_intent: SupportIntent | None = None
        self._abort_reason: str | None = None
        self._last_diagnostics: SupportDiagnostics | None = None
        self._cycle_reference_rad: FloatArray | None = None
        self._admitted_pose_rad: FloatArray | None = None
        self._last_output_rad = None if self._home is None else self._home.copy()

    @property
    def intent(self) -> SupportIntent:
        return self._requested_intent

    @property
    def active_intent(self) -> SupportIntent:
        return self._active_intent

    @property
    def phase(self) -> SupportPhase:
        return self._phase

    @property
    def last_diagnostics(self) -> SupportDiagnostics | None:
        return self._last_diagnostics

    def set_intent(self, intent: SupportIntent | str) -> SupportIntent:
        """Request a support mode; transitions occur only inside ``update``."""

        accepted = SupportIntent(intent)
        self._requested_intent = accepted
        if accepted is SupportIntent.DOUBLE_SUPPORT:
            # Explicitly returning to two feet acknowledges any aborted lift.
            self._blocked_intent = None
        return accepted

    def _transition(self, phase: SupportPhase) -> None:
        self._phase = phase
        self._phase_elapsed_s = 0.0
        if phase is SupportPhase.DOUBLE_SUPPORT:
            self._cycle_reference_rad = None
            self._admitted_pose_rad = None
        elif phase is SupportPhase.VERIFY_STANCE:
            self._load_confirm_elapsed_s = 0.0
        elif phase is SupportPhase.VERIFY_TOUCHDOWN:
            self._touchdown_confirm_elapsed_s = 0.0
        elif phase is SupportPhase.LOWER_SWING:
            self._touchdown_confirm_elapsed_s = 0.0
        elif phase is SupportPhase.CENTER_WEIGHT:
            self._center_start_shift_progress = max(self._shift_progress, 1e-9)

    @staticmethod
    def _validated_force(state: object, name: str) -> float:
        value = float(getattr(state, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def _loads(self, state: object) -> tuple[float, float, float, bool]:
        right = self._validated_force(state, "right_foot_normal_force_n")
        left = self._validated_force(state, "left_foot_normal_force_n")
        if self._active_intent is SupportIntent.RIGHT_SWING:
            stance, swing = left, right
        elif self._active_intent is SupportIntent.LEFT_SWING:
            stance, swing = right, left
        else:
            return 0.0, 0.0, 0.0, False
        total = stance + swing
        stance_fraction = stance / total if total > 1e-9 else 0.0
        swing_fraction = swing / total if total > 1e-9 else 1.0
        ready = bool(
            stance >= self.config.min_stance_force_n
            and stance_fraction >= self.config.min_stance_load_fraction
            and swing_fraction <= self.config.max_swing_load_fraction
        )
        return stance, swing, stance_fraction, ready

    @staticmethod
    def _base_motion(state: object) -> tuple[float, float]:
        quaternion = np.asarray(
            getattr(state, "base_orientation_wxyz"), dtype=np.float64
        )
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("base_orientation_wxyz must be a finite 4-vector")
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm < 1e-12:
            raise ValueError("base_orientation_wxyz must have non-zero norm")
        _, x, y, _ = quaternion / quaternion_norm
        upright = 1.0 - 2.0 * (x * x + y * y)
        tilt_rad = float(np.arccos(np.clip(upright, -1.0, 1.0)))

        angular_velocity = np.asarray(
            getattr(state, "base_angular_velocity_rad_s"), dtype=np.float64
        )
        if angular_velocity.shape != (3,) or not np.isfinite(angular_velocity).all():
            raise ValueError(
                "base_angular_velocity_rad_s must be a finite 3-vector"
            )
        angular_speed_rad_s = float(np.linalg.norm(angular_velocity))
        return tilt_rad, angular_speed_rad_s

    def _stability_reason(
        self,
        tilt_rad: float,
        angular_speed_rad_s: float,
        *,
        active: bool,
    ) -> str | None:
        prefix = "active" if active else "start"
        tilt_limit = (
            self.config.active_max_tilt_rad
            if active
            else self.config.start_max_tilt_rad
        )
        angular_speed_limit = (
            self.config.active_max_angular_speed_rad_s
            if active
            else self.config.start_max_angular_speed_rad_s
        )
        if tilt_rad > tilt_limit:
            return f"{prefix}_tilt_limit"
        if angular_speed_rad_s > angular_speed_limit:
            return f"{prefix}_angular_speed_limit"
        return None

    def _abort(self, reason: str) -> None:
        self._abort_reason = reason
        self._blocked_intent = self._active_intent
        if self._swing_progress > 0.0:
            self._transition(SupportPhase.LOWER_SWING)
        else:
            self._transition(SupportPhase.CENTER_WEIGHT)

    def _advance(
        self,
        dt_s: float,
        support_ready: bool,
        touchdown_ready: bool,
        start_stability_reason: str | None,
        active_stability_reason: str | None,
        *,
        force_return: bool = False,
    ) -> None:
        self._phase_elapsed_s += dt_s

        if self._phase is SupportPhase.DOUBLE_SUPPORT:
            self._shift_progress = 0.0
            self._swing_progress = 0.0
            self._active_intent = SupportIntent.DOUBLE_SUPPORT
            if (
                self._requested_intent is not SupportIntent.DOUBLE_SUPPORT
                and self._requested_intent is not self._blocked_intent
            ):
                if start_stability_reason is None:
                    self._active_intent = self._requested_intent
                    self._abort_reason = None
                    self._transition(SupportPhase.SHIFT_WEIGHT)
                else:
                    self._abort_reason = start_stability_reason
            return

        requested_changed = self._requested_intent is not self._active_intent
        hold_dwell_pending = bool(
            self._phase is SupportPhase.HOLD_SWING
            and self._phase_elapsed_s < self.config.minimum_hold_duration_s
            and not force_return
            and active_stability_reason is None
        )
        if requested_changed and not hold_dwell_pending and self._phase not in {
            SupportPhase.LOWER_SWING,
            SupportPhase.VERIFY_TOUCHDOWN,
            SupportPhase.CENTER_WEIGHT,
        }:
            if self._swing_progress > 0.0:
                self._transition(SupportPhase.LOWER_SWING)
            else:
                self._transition(SupportPhase.CENTER_WEIGHT)

        if (
            active_stability_reason is not None
            and self._phase
            in {
                SupportPhase.SHIFT_WEIGHT,
                SupportPhase.VERIFY_STANCE,
                SupportPhase.LIFT_SWING,
                SupportPhase.HOLD_SWING,
            }
        ):
            self._abort(active_stability_reason)
            return

        if self._phase is SupportPhase.SHIFT_WEIGHT:
            self._shift_progress = min(
                1.0, self._shift_progress + dt_s / self.config.shift_duration_s
            )
            if self._shift_progress >= 1.0:
                self._transition(SupportPhase.VERIFY_STANCE)
        elif self._phase is SupportPhase.VERIFY_STANCE:
            if support_ready:
                self._load_confirm_elapsed_s += dt_s
                if self._load_confirm_elapsed_s >= self.config.load_confirm_duration_s:
                    self._support_unsafe_elapsed_s = 0.0
                    self._transition(SupportPhase.LIFT_SWING)
            else:
                self._load_confirm_elapsed_s = 0.0
            if self._phase_elapsed_s >= self.config.stance_load_timeout_s:
                self._abort("stance_load_timeout")
        elif self._phase in {SupportPhase.LIFT_SWING, SupportPhase.HOLD_SWING}:
            if support_ready:
                self._support_unsafe_elapsed_s = 0.0
            else:
                self._support_unsafe_elapsed_s += dt_s
                if self._support_unsafe_elapsed_s >= self.config.support_loss_grace_s:
                    self._abort("stance_load_lost")
                    return
            if self._phase is SupportPhase.LIFT_SWING:
                self._swing_progress = min(
                    1.0, self._swing_progress + dt_s / self.config.lift_duration_s
                )
                if self._swing_progress >= 1.0:
                    self._transition(SupportPhase.HOLD_SWING)
        elif self._phase is SupportPhase.LOWER_SWING:
            # A real sole contact is stronger evidence than a nominal profile
            # endpoint.  Confirm it while lowering and begin a simultaneous
            # center-and-lower recovery instead of driving through the ground
            # pose, losing contact, and only then entering VERIFY_TOUCHDOWN.
            if (
                self._swing_progress
                <= self.config.early_touchdown_max_swing_progress
                and touchdown_ready
            ):
                self._touchdown_confirm_elapsed_s += dt_s
                if (
                    self._touchdown_confirm_elapsed_s
                    >= self.config.touchdown_confirm_duration_s
                ):
                    # This contact has already passed the same timed force
                    # confirmation used by VERIFY_TOUCHDOWN.  Preserve it by
                    # centering immediately while CENTER_WEIGHT continues to
                    # lower the residual swing profile; demanding a second
                    # confirmation after changing composition can unload the
                    # sole again.
                    self._transition(SupportPhase.CENTER_WEIGHT)
                    return
            else:
                self._touchdown_confirm_elapsed_s = 0.0
            self._swing_progress = max(
                0.0, self._swing_progress - dt_s / self.config.lower_duration_s
            )
            if self._swing_progress <= 0.0:
                self._transition(SupportPhase.VERIFY_TOUCHDOWN)
        elif self._phase is SupportPhase.VERIFY_TOUCHDOWN:
            # Preload the returning sole by reducing the lateral shift much
            # more slowly than the normal center phase.  Do not fully remove
            # the known-good stance until both feet physically carry load.
            self._shift_progress = max(
                0.0,
                self._shift_progress
                - dt_s / self.config.touchdown_preload_duration_s,
            )
            if touchdown_ready:
                self._touchdown_confirm_elapsed_s += dt_s
                if (
                    self._touchdown_confirm_elapsed_s
                    >= self.config.touchdown_confirm_duration_s
                ):
                    self._transition(SupportPhase.CENTER_WEIGHT)
            else:
                self._touchdown_confirm_elapsed_s = 0.0
            if self._phase_elapsed_s >= self.config.touchdown_timeout_s:
                self._abort_reason = "touchdown_timeout"
        elif self._phase is SupportPhase.CENTER_WEIGHT:
            self._shift_progress = max(
                0.0, self._shift_progress - dt_s / self.config.center_duration_s
            )
            self._swing_progress = max(
                0.0, self._swing_progress - dt_s / self.config.lower_duration_s
            )
            if self._shift_progress <= 0.0 and self._swing_progress <= 0.0:
                self._active_intent = SupportIntent.DOUBLE_SUPPORT
                self._transition(SupportPhase.DOUBLE_SUPPORT)

    def update(
        self,
        reference: RobotJointCommand,
        state: object,
        *,
        dt_s: float,
        intent: SupportIntent | str | None = None,
    ) -> RobotJointCommand:
        """Overlay a contact-gated support motion on a balanced motor command."""

        dt_s = float(dt_s)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if tuple(reference.joint_names) != JOINT_NAMES:
            raise ValueError("reference must use canonical motor order")
        positions = self._vector(reference.positions_rad, "reference.positions_rad")
        pose_reference = self._vector(
            getattr(reference, "pose_reference_positions_rad", positions),
            "reference.pose_reference_positions_rad",
        )
        if self._home is None:
            # Lightweight/controller-only callers need not provide a model;
            # their first canonical balanced command becomes the neutral seed.
            self._home = positions.copy()
        if intent is not None:
            self.set_intent(intent)
        if reference.stale and self.config.cancel_on_stale_reference:
            self.set_intent(SupportIntent.DOUBLE_SUPPORT)

        base_tilt_rad, base_angular_speed_rad_s = self._base_motion(state)
        start_stability_reason = self._stability_reason(
            base_tilt_rad,
            base_angular_speed_rad_s,
            active=False,
        )
        active_stability_reason = self._stability_reason(
            base_tilt_rad,
            base_angular_speed_rad_s,
            active=True,
        )
        stance_force, swing_force, stance_fraction, support_ready = self._loads(state)
        touchdown_ready = bool(
            stance_force >= self.config.min_touchdown_force_n
            and swing_force >= self.config.min_touchdown_force_n
            and stance_force + swing_force
            >= self.config.min_touchdown_total_force_n
        )
        self._advance(
            dt_s,
            support_ready,
            touchdown_ready,
            start_stability_reason,
            active_stability_reason,
            force_return=reference.stale,
        )
        if self._phase is SupportPhase.SHIFT_WEIGHT and self._cycle_reference_rad is None:
            # Capture the already safe, continuous double-support pose.  Never
            # reset it to home at phase admission: the final slew limiter is a
            # guard, not a substitute for a continuous desired trajectory.
            self._cycle_reference_rad = positions.copy()
        # Re-evaluate side-specific diagnostics if DOUBLE_SUPPORT just accepted
        # a request during this update.
        stance_force, swing_force, stance_fraction, support_ready = self._loads(state)
        touchdown_ready = bool(
            stance_force >= self.config.min_touchdown_force_n
            and swing_force >= self.config.min_touchdown_force_n
            and stance_force + swing_force
            >= self.config.min_touchdown_total_force_n
        )

        composed_positions = positions.copy()
        lift_profile_scale = 1.0
        if (
            self._cycle_reference_rad is not None
            and self._active_intent is not SupportIntent.DOUBLE_SUPPORT
        ):
            support_upper = self._home[UPPER_BODY_INDICES] + (
                self.config.single_support_upper_body_scale
                * (positions[UPPER_BODY_INDICES] - self._home[UPPER_BODY_INDICES])
            )
            if self._phase in {SupportPhase.SHIFT_WEIGHT, SupportPhase.VERIFY_STANCE}:
                upper_safety_weight = _smoothstep(self._shift_progress)
            elif self._phase is SupportPhase.CENTER_WEIGHT:
                upper_safety_weight = _smoothstep(
                    self._shift_progress / self._center_start_shift_progress
                )
            else:
                upper_safety_weight = 1.0
            composed_positions[UPPER_BODY_INDICES] = (
                (1.0 - upper_safety_weight) * positions[UPPER_BODY_INDICES]
                + upper_safety_weight * support_upper
            )
            # Keep the standing layer's continuous, already projected lower
            # pose captured at admission as the support-profile base.
            # Resetting these joints to ``home`` at SHIFT creates a knee snap;
            # following every later camera-frame perturbation would instead
            # move the stance geometry during the one-foot support cycle.
            cycle_lower = self._cycle_reference_rad[LOWER_BODY_INDICES]
            if self._phase is SupportPhase.CENTER_WEIGHT:
                cycle_weight = _smoothstep(
                    self._shift_progress / self._center_start_shift_progress
                )
                composed_positions[LOWER_BODY_INDICES] = (
                    cycle_weight * cycle_lower
                    + (1.0 - cycle_weight) * positions[LOWER_BODY_INDICES]
                )
            else:
                composed_positions[LOWER_BODY_INDICES] = cycle_lower
            if self._active_intent is SupportIntent.RIGHT_SWING:
                swing_indices = RIGHT_LEG_INDICES
            else:
                swing_indices = LEFT_LEG_INDICES
            pose_admission_active = self._phase in {
                SupportPhase.LIFT_SWING,
                SupportPhase.HOLD_SWING,
                SupportPhase.LOWER_SWING,
            } or (
                self._phase is SupportPhase.VERIFY_STANCE and support_ready
            )
            if pose_admission_active:
                if self._admitted_pose_rad is None:
                    # Continue from the standing layer's already rate-limited
                    # pose.  Pre-admit while VERIFY_STANCE is physically
                    # confirming the intentional unload, matching the
                    # continuous servo path without exposing the candidate in
                    # SHIFT or incidental DOUBLE_SUPPORT unloading.
                    self._admitted_pose_rad = positions.copy()
                maximum_pose_change = self._rate_limits[swing_indices] * dt_s
                # Stop chasing a visual swing pose as soon as the returning
                # sole carries touchdown-level load.  During LOWER this hands
                # the leg back to the balanced standing projection before the
                # support profile reaches zero, instead of fighting contact.
                candidate_target = (
                    pose_reference
                    if support_ready
                    and swing_force < self.config.min_touchdown_force_n
                    else positions
                )
                self._admitted_pose_rad[swing_indices] += np.clip(
                    candidate_target[swing_indices]
                    - self._admitted_pose_rad[swing_indices],
                    -maximum_pose_change,
                    maximum_pose_change,
                )
                admitted_pose = self._admitted_pose_rad
            else:
                admitted_pose = composed_positions
            reference_delta = np.clip(
                admitted_pose[swing_indices] - composed_positions[swing_indices],
                -self.config.max_swing_reference_delta_rad,
                self.config.max_swing_reference_delta_rad,
            )
            if self._phase in {
                SupportPhase.LIFT_SWING,
                SupportPhase.HOLD_SWING,
                SupportPhase.LOWER_SWING,
            }:
                pose_activity = float(
                    np.clip(
                        np.max(np.abs(reference_delta))
                        / self.config.max_swing_reference_delta_rad,
                        0.0,
                        1.0,
                    )
                )
                active_blend = self.config.swing_reference_blend * pose_activity
                lift_profile_scale = 1.0 - active_blend
                reference_blend = active_blend * _smoothstep(self._swing_progress)
            else:
                reference_blend = 0.0
            composed_positions[swing_indices] = (
                composed_positions[swing_indices] + reference_blend * reference_delta
            )

        shift_offset, lift_offset = support_offsets(self._active_intent)
        applied_offset = (
            _smoothstep(self._shift_progress) * shift_offset
            + _smoothstep(self._swing_progress) * lift_profile_scale * lift_offset
        )
        desired_safe = np.clip(
            composed_positions + applied_offset, self._lower, self._upper
        )
        if self._last_output_rad is None:
            safe = desired_safe
        else:
            maximum_change = self._rate_limits * dt_s
            safe = self._last_output_rad + np.clip(
                desired_safe - self._last_output_rad,
                -maximum_change,
                maximum_change,
            )
            safe = np.clip(safe, self._lower, self._upper)
        self._last_output_rad = safe.copy()
        applied_offset = safe - positions
        self._last_diagnostics = SupportDiagnostics(
            requested_intent=self._requested_intent,
            active_intent=self._active_intent,
            phase=self._phase,
            shift_progress=float(self._shift_progress),
            swing_progress=float(self._swing_progress),
            stance_force_n=stance_force,
            swing_force_n=swing_force,
            stance_load_fraction=stance_fraction,
            support_ready=support_ready,
            touchdown_ready=touchdown_ready,
            base_tilt_rad=base_tilt_rad,
            base_angular_speed_rad_s=base_angular_speed_rad_s,
            start_stable=start_stability_reason is None,
            active_stable=active_stability_reason is None,
            blocked_intent=self._blocked_intent,
            abort_reason=self._abort_reason,
            applied_offset_rad=applied_offset,
        )
        return RobotJointCommand.humanoid(
            reference.timestamp_s,
            safe,
            reference.confidence,
            stale=reference.stale,
        )


class SupportIntentLatch:
    """Finish a safe lift sequence even when a camera gesture is brief.

    Camera intent is a slow, noisy request rather than a servo command.  Once a
    swing is accepted, keep it requested until the controller has actually
    reached ``HOLD_SWING``.  Stale input or a controller abort still releases
    immediately to the state machine's safe lower-and-center path.

    One bounded pending slot preserves an opposite-leg request observed while
    the current cycle is safely lowering, confirming touchdown, or centering.
    It is consumed only after the state machine reports ``DOUBLE_SUPPORT``;
    there is never a direct side-to-side switch.
    """

    _PENDING_PHASES = frozenset(
        {
            SupportPhase.LOWER_SWING,
            SupportPhase.VERIFY_TOUCHDOWN,
            SupportPhase.CENTER_WEIGHT,
        }
    )

    def __init__(
        self,
        pending_max_age_s: float = 6.0,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        pending_max_age_s = float(pending_max_age_s)
        if not np.isfinite(pending_max_age_s) or pending_max_age_s <= 0.0:
            raise ValueError("pending_max_age_s must be finite and positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.pending_max_age_s = pending_max_age_s
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self._intent = SupportIntent.DOUBLE_SUPPORT
        self._blocked_intent: SupportIntent | None = None
        self._cycle_intent: SupportIntent | None = None
        self._pending_intent: SupportIntent | None = None
        self._pending_since_s: float | None = None
        self._expired_pending_intent: SupportIntent | None = None
        self._last_timestamp_s: float | None = None

    @property
    def intent(self) -> SupportIntent:
        return self._intent

    @property
    def blocked_intent(self) -> SupportIntent | None:
        return self._blocked_intent

    @property
    def pending_intent(self) -> SupportIntent | None:
        return self._pending_intent

    @property
    def pending_since_s(self) -> float | None:
        return self._pending_since_s

    def _timestamp(self, timestamp_s: float | None) -> float:
        now_s = float(self._clock() if timestamp_s is None else timestamp_s)
        if not np.isfinite(now_s) or now_s < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if self._last_timestamp_s is not None and now_s < self._last_timestamp_s:
            raise ValueError("support-latch timestamps must be monotonic")
        self._last_timestamp_s = now_s
        return now_s

    def _clear_pending(self) -> None:
        self._pending_intent = None
        self._pending_since_s = None

    def _expire_pending(self, now_s: float) -> None:
        if (
            self._pending_intent is not None
            and self._pending_since_s is not None
            and now_s - self._pending_since_s > self.pending_max_age_s
        ):
            # Do not continuously recreate an expired request from the same
            # held observation.  A return through another observed state makes
            # a later gesture a fresh request again.
            self._expired_pending_intent = self._pending_intent
            self._clear_pending()

    def update(
        self,
        observed: SupportIntent | str,
        phase: SupportPhase | str,
        *,
        stale: bool = False,
        aborted: bool = False,
        timestamp_s: float | None = None,
    ) -> SupportIntent:
        observed_intent = SupportIntent(observed)
        active_phase = SupportPhase(phase)
        now_s = self._timestamp(timestamp_s)
        self._expire_pending(now_s)
        if (
            self._expired_pending_intent is not None
            and observed_intent is not self._expired_pending_intent
        ):
            self._expired_pending_intent = None

        if stale or aborted:
            self._clear_pending()
            self._expired_pending_intent = None
            blocked = (
                self._intent
                if self._intent is not SupportIntent.DOUBLE_SUPPORT
                else observed_intent
            )
            if blocked is not SupportIntent.DOUBLE_SUPPORT:
                self._blocked_intent = blocked
            self._intent = SupportIntent.DOUBLE_SUPPORT
        elif self._blocked_intent is not None:
            self._clear_pending()
            # Require a real camera-side two-foot observation before retrying
            # the failed/stale gesture.  A continuously raised leg is not a
            # fresh command edge and must not create an abort/retry loop.
            if observed_intent is SupportIntent.DOUBLE_SUPPORT:
                self._blocked_intent = None
                self._cycle_intent = None
            self._intent = SupportIntent.DOUBLE_SUPPORT
        else:
            if (
                active_phase in self._PENDING_PHASES
                and self._cycle_intent is not None
                and observed_intent
                not in {SupportIntent.DOUBLE_SUPPORT, self._cycle_intent}
                and observed_intent is not self._expired_pending_intent
                and self._pending_intent is None
            ):
                self._pending_intent = observed_intent
                self._pending_since_s = now_s

            if (
                active_phase is SupportPhase.DOUBLE_SUPPORT
                and self._pending_intent is not None
            ):
                pending = self._pending_intent
                self._clear_pending()
                self._expired_pending_intent = None
                self._intent = pending
                self._cycle_intent = pending
            elif self._intent is SupportIntent.DOUBLE_SUPPORT:
                if (
                    active_phase is SupportPhase.DOUBLE_SUPPORT
                    and observed_intent is not SupportIntent.DOUBLE_SUPPORT
                ):
                    self._intent = observed_intent
                    self._cycle_intent = observed_intent
                elif active_phase is SupportPhase.DOUBLE_SUPPORT:
                    self._cycle_intent = None
            elif (
                active_phase is SupportPhase.DOUBLE_SUPPORT
                and observed_intent is SupportIntent.DOUBLE_SUPPORT
            ):
                # The FSM may deliberately remain in DOUBLE_SUPPORT while an
                # admission gate is unsafe.  If the camera gesture ends during
                # that wait, do not keep an invisible request latched forever.
                self._intent = SupportIntent.DOUBLE_SUPPORT
                self._cycle_intent = None
            elif (
                active_phase is SupportPhase.HOLD_SWING
                and observed_intent is not self._intent
            ):
                self._intent = SupportIntent.DOUBLE_SUPPORT
        return self._intent
