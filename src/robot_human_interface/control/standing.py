"""Double-support balance baseline whose only action is 20 motor angles.

This is intentionally a small, inspectable classical controller.  It keeps the
lower body close to a verified standing pose, copies the upper-body reference,
and adds bounded ankle-position residuals from deployable IMU observations.
It does not modify a simulator base pose, apply external forces, or use a weld.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml
from numpy.typing import NDArray

from robot_human_interface.skeleton import JOINT_NAMES, RobotJointCommand


FloatArray = NDArray[np.float64]
UPPER_BODY_INDICES = np.asarray((0, 1, 2, 3, 4, 5, 18, 19), dtype=np.int64)
LOWER_BODY_INDICES = np.arange(6, 18, dtype=np.int64)
HIP_YAW_INDICES = np.asarray((6, 7), dtype=np.int64)
HIP_ROLL_INDICES = np.asarray((8, 9), dtype=np.int64)
HIP_PITCH_INDICES = np.asarray((10, 11), dtype=np.int64)
KNEE_INDICES = np.asarray((12, 13), dtype=np.int64)
ANKLE_PITCH_INDICES = np.asarray((14, 15), dtype=np.int64)
ANKLE_ROLL_INDICES = np.asarray((16, 17), dtype=np.int64)
SHOULDER_PITCH_INDICES = np.asarray((0, 1), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class StandingBalanceConfig:
    """Tunable double-support projection and residual feedback gains."""

    enabled: bool = True
    lower_body_imitation_scale: float = 0.3
    transverse_lower_body_imitation_scale: float = 0.0
    swing_leg_imitation_scale: float = 0.65
    max_hip_yaw_deviation_rad: float = 8.0 * pi / 180.0
    max_hip_roll_deviation_rad: float = 4.0 * pi / 180.0
    max_hip_pitch_deviation_rad: float = 12.0 * pi / 180.0
    max_knee_deviation_rad: float = 18.0 * pi / 180.0
    max_ankle_pitch_deviation_rad: float = 6.0 * pi / 180.0
    max_ankle_roll_deviation_rad: float = 4.0 * pi / 180.0
    unsupported_hip_fade_start_rad: float = 15.0 * pi / 180.0
    unsupported_hip_fade_full_rad: float = 30.0 * pi / 180.0
    unsupported_knee_fade_start_rad: float = 12.0 * pi / 180.0
    unsupported_knee_fade_full_rad: float = 25.0 * pi / 180.0
    unsupported_pose_tracking_scale: float = 0.0
    ankle_pitch_bias_rad: float = -0.04
    arm_to_ankle_gain: float = -0.055
    pitch_feedback_gain: float = 0.08
    pitch_rate_feedback_s: float = 0.005
    upper_body_rate_limit_rad_s: float = 2.5
    lower_body_rate_limit_rad_s: float = 1.2
    max_shoulder_deviation_rad: float = 70.0 * pi / 180.0
    tracking_fade_start_rad: float = 8.0 * pi / 180.0
    recovery_tilt_rad: float = 18.0 * pi / 180.0
    capture_velocity_filter_time_constant_s: float = 0.08
    capture_tracking_margin_start_m: float = 0.035
    capture_tracking_margin_full_m: float = 0.075
    capture_recovery_gain_rad_per_m: float = 1.0
    capture_recovery_full_gain_rad_per_m: float = 2.0
    capture_recovery_max_rad: float = 18.0 * pi / 180.0
    capture_full_gain_start_foot_force_n: float = 4.0
    capture_full_gain_min_foot_force_n: float = 10.0
    capture_minimum_com_height_m: float = 0.20
    capture_minimum_total_support_force_n: float = 1.0
    capture_support_point_filter_time_constant_s: float = 0.04
    max_inverse_crouch_amplitude_rad: float = 6.0 * pi / 180.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.lower_body_imitation_scale <= 1.0:
            raise ValueError("lower_body_imitation_scale must be within [0, 1]")
        if not 0.0 <= self.transverse_lower_body_imitation_scale <= 1.0:
            raise ValueError(
                "transverse_lower_body_imitation_scale must be within [0, 1]"
            )
        if not 0.0 <= self.swing_leg_imitation_scale <= 1.0:
            raise ValueError("swing_leg_imitation_scale must be within [0, 1]")
        if not 0.0 <= self.unsupported_pose_tracking_scale <= 1.0:
            raise ValueError("unsupported_pose_tracking_scale must be within [0, 1]")
        finite = (
            self.ankle_pitch_bias_rad,
            self.max_hip_yaw_deviation_rad,
            self.max_hip_roll_deviation_rad,
            self.max_hip_pitch_deviation_rad,
            self.max_knee_deviation_rad,
            self.max_ankle_pitch_deviation_rad,
            self.max_ankle_roll_deviation_rad,
            self.unsupported_hip_fade_start_rad,
            self.unsupported_hip_fade_full_rad,
            self.unsupported_knee_fade_start_rad,
            self.unsupported_knee_fade_full_rad,
            self.arm_to_ankle_gain,
            self.pitch_feedback_gain,
            self.pitch_rate_feedback_s,
            self.upper_body_rate_limit_rad_s,
            self.lower_body_rate_limit_rad_s,
            self.max_shoulder_deviation_rad,
            self.tracking_fade_start_rad,
            self.recovery_tilt_rad,
            self.capture_velocity_filter_time_constant_s,
            self.capture_tracking_margin_start_m,
            self.capture_tracking_margin_full_m,
            self.capture_recovery_gain_rad_per_m,
            self.capture_recovery_full_gain_rad_per_m,
            self.capture_recovery_max_rad,
            self.capture_full_gain_start_foot_force_n,
            self.capture_full_gain_min_foot_force_n,
            self.capture_minimum_com_height_m,
            self.capture_minimum_total_support_force_n,
            self.capture_support_point_filter_time_constant_s,
            self.max_inverse_crouch_amplitude_rad,
        )
        if not np.isfinite(finite).all():
            raise ValueError("balance-controller parameters must be finite")
        if self.upper_body_rate_limit_rad_s <= 0.0 or self.lower_body_rate_limit_rad_s <= 0.0:
            raise ValueError("motor target rate limits must be positive")
        if any(
            value <= 0.0
            for value in (
                self.max_hip_yaw_deviation_rad,
                self.max_hip_roll_deviation_rad,
                self.max_hip_pitch_deviation_rad,
                self.max_knee_deviation_rad,
                self.max_ankle_pitch_deviation_rad,
                self.max_ankle_roll_deviation_rad,
            )
        ):
            raise ValueError("lower-body imitation bounds must be positive")
        if (
            self.unsupported_hip_fade_start_rad
            >= self.unsupported_hip_fade_full_rad
            or self.unsupported_knee_fade_start_rad
            >= self.unsupported_knee_fade_full_rad
        ):
            raise ValueError("unsupported-pose fade start must be below fade full")
        if self.max_shoulder_deviation_rad <= 0.0:
            raise ValueError("max_shoulder_deviation_rad must be positive")
        if self.tracking_fade_start_rad < 0.0:
            raise ValueError("tracking_fade_start_rad must be non-negative")
        if self.recovery_tilt_rad <= self.tracking_fade_start_rad:
            raise ValueError("recovery_tilt_rad must exceed tracking_fade_start_rad")
        if self.capture_velocity_filter_time_constant_s < 0.0:
            raise ValueError(
                "capture_velocity_filter_time_constant_s must be non-negative"
            )
        if self.capture_support_point_filter_time_constant_s < 0.0:
            raise ValueError(
                "capture_support_point_filter_time_constant_s must be non-negative"
            )
        if (
            self.capture_tracking_margin_start_m < 0.0
            or self.capture_tracking_margin_full_m
            <= self.capture_tracking_margin_start_m
        ):
            raise ValueError("capture tracking margins must be positive and ordered")
        if (
            self.capture_recovery_gain_rad_per_m < 0.0
            or self.capture_recovery_full_gain_rad_per_m
            < self.capture_recovery_gain_rad_per_m
            or self.capture_recovery_max_rad <= 0.0
            or self.capture_full_gain_start_foot_force_n < 0.0
            or self.capture_minimum_com_height_m <= 0.0
            or self.capture_minimum_total_support_force_n <= 0.0
            or self.max_inverse_crouch_amplitude_rad <= 0.0
        ):
            raise ValueError("capture and inverse-crouch limits must be positive")
        if (
            self.capture_full_gain_min_foot_force_n
            <= self.capture_full_gain_start_foot_force_n
        ):
            raise ValueError("capture full-gain foot-force thresholds must be ordered")


@dataclass(frozen=True, slots=True)
class BalanceDiagnostics:
    """Signals needed to distinguish imitation from stability corrections.

    ``capture_recovery_rad`` is the raw bounded recovery request.  The
    ``BalancedJointCommand`` separately exports the portion actually deployed
    after motor-target slew and joint-limit projection.
    """

    roll_rad: float
    pitch_rad: float
    tilt_rad: float
    tracking_weight: float
    ankle_pitch_residual_rad: float
    com_offset_x_m: float
    com_velocity_x_m_s: float
    capture_point_error_x_m: float
    capture_tracking_weight: float
    capture_recovery_rad: float
    reference_positions_rad: FloatArray
    safe_positions_rad: FloatArray

    @property
    def residual_positions_rad(self) -> FloatArray:
        return self.safe_positions_rad - self.reference_positions_rad


@dataclass(frozen=True, slots=True)
class BalancedJointCommand(RobotJointCommand):
    """Safe standing target plus a bounded, not-yet-authorized pose candidate.

    ``positions_rad`` is safe for double support.  The support state machine
    may consume ``pose_reference_positions_rad`` only for its explicitly
    selected swing leg after the contact-gated phase reaches LIFT/HOLD.
    """

    pose_reference_positions_rad: FloatArray = field(kw_only=True)
    capture_recovery_positions_rad: FloatArray | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        RobotJointCommand.__post_init__(self)
        pose_reference = np.asarray(
            self.pose_reference_positions_rad, dtype=np.float64
        )
        if pose_reference.shape != (len(JOINT_NAMES),) or not np.isfinite(
            pose_reference
        ).all():
            raise ValueError(
                "pose_reference_positions_rad must be a finite canonical vector"
            )
        pose_reference = pose_reference.copy()
        pose_reference.setflags(write=False)
        object.__setattr__(
            self, "pose_reference_positions_rad", pose_reference
        )
        capture_recovery = np.asarray(
            np.zeros(len(JOINT_NAMES))
            if self.capture_recovery_positions_rad is None
            else self.capture_recovery_positions_rad,
            dtype=np.float64,
        )
        if capture_recovery.shape != (len(JOINT_NAMES),) or not np.isfinite(
            capture_recovery
        ).all():
            raise ValueError(
                "capture_recovery_positions_rad must be a finite canonical vector"
            )
        capture_recovery = capture_recovery.copy()
        capture_recovery.setflags(write=False)
        object.__setattr__(
            self, "capture_recovery_positions_rad", capture_recovery
        )


def load_standing_balance_config(path: str | Path | None) -> StandingBalanceConfig:
    if path is None or not Path(path).is_file():
        return StandingBalanceConfig()
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, Mapping):
        raise ValueError("balance YAML root must be a mapping")
    settings = document.get("standing_balance", document)
    if not isinstance(settings, Mapping):
        raise ValueError("standing_balance settings must be a mapping")
    allowed = set(StandingBalanceConfig.__dataclass_fields__)
    return StandingBalanceConfig(
        **{key: value for key, value in settings.items() if key in allowed}
    )


def _projected_gravity_angles(quaternion_wxyz: Sequence[float]) -> tuple[float, float, float]:
    """Return body-frame roll, pitch and total tilt from an IMU quaternion."""

    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("base orientation must be a finite wxyz quaternion")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("base orientation quaternion must have non-zero norm")
    w, x, y, z = q / norm
    rotation = np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    gravity_body = rotation.T @ np.array((0.0, 0.0, -1.0))
    roll = float(np.arctan2(-gravity_body[1], -gravity_body[2]))
    pitch = float(np.arctan2(gravity_body[0], -gravity_body[2]))
    tilt = float(np.arccos(np.clip(-gravity_body[2], -1.0, 1.0)))
    return roll, pitch, tilt


class StandingBalanceController:
    """Project a human reference into a safe double-support motor command."""

    def __init__(
        self,
        home_positions_rad: Sequence[float],
        lower_limits_rad: Sequence[float],
        upper_limits_rad: Sequence[float],
        config: StandingBalanceConfig | None = None,
    ) -> None:
        self.config = config or StandingBalanceConfig()
        self._home = self._vector(home_positions_rad, "home_positions_rad")
        self._lower = self._vector(lower_limits_rad, "lower_limits_rad")
        self._upper = self._vector(upper_limits_rad, "upper_limits_rad")
        if np.any(self._lower >= self._upper):
            raise ValueError("every lower motor limit must be below its upper limit")
        if np.any(self._home < self._lower) or np.any(self._home > self._upper):
            raise ValueError("home motor angles must lie inside configured limits")
        self._rate_limits = np.full(len(JOINT_NAMES), self.config.lower_body_rate_limit_rad_s)
        self._rate_limits[UPPER_BODY_INDICES] = self.config.upper_body_rate_limit_rad_s
        self._lower_body_deviation_limits = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        for indices, limit in (
            (HIP_YAW_INDICES, self.config.max_hip_yaw_deviation_rad),
            (HIP_ROLL_INDICES, self.config.max_hip_roll_deviation_rad),
            (HIP_PITCH_INDICES, self.config.max_hip_pitch_deviation_rad),
            (KNEE_INDICES, self.config.max_knee_deviation_rad),
            (ANKLE_PITCH_INDICES, self.config.max_ankle_pitch_deviation_rad),
            (ANKLE_ROLL_INDICES, self.config.max_ankle_roll_deviation_rad),
        ):
            self._lower_body_deviation_limits[indices] = limit
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
        config: StandingBalanceConfig | None = None,
    ) -> "StandingBalanceController":
        return cls(
            getattr(simulation, "home_positions_rad"),
            getattr(simulation, "lower_limits_rad"),
            getattr(simulation, "upper_limits_rad"),
            config,
        )

    def reset(self) -> None:
        self._last_output = self._home.copy()
        self._last_output_without_capture = self._home.copy()
        self._last_diagnostics: BalanceDiagnostics | None = None
        self._neutral_com_offset_x_m: float | None = None
        self._previous_com_offset_x_m: float | None = None
        self._filtered_com_velocity_x_m_s = 0.0
        self._filtered_support_point_m: FloatArray | None = None

    @property
    def last_diagnostics(self) -> BalanceDiagnostics | None:
        return self._last_diagnostics

    def _capture_observation(
        self,
        state: object,
        dt_s: float,
    ) -> tuple[float, float, float, float, float]:
        """Return CoM/CP signals, tracking weight, and bilateral-load weight.

        Lightweight controller callers predating the free-base governor do
        not necessarily expose CoM/foot positions.  In that case the governor
        is deliberately neutral while the existing IMU feedback still runs.
        """

        try:
            center_of_mass = np.asarray(
                getattr(state, "center_of_mass_position_m"), dtype=np.float64
            )
            right_foot = np.asarray(
                getattr(state, "right_foot_position_m"), dtype=np.float64
            )
            left_foot = np.asarray(
                getattr(state, "left_foot_position_m"), dtype=np.float64
            )
        except (AttributeError, TypeError, ValueError):
            return 0.0, 0.0, 0.0, 1.0, 0.0
        if (
            center_of_mass.shape != (3,)
            or right_foot.shape != (3,)
            or left_foot.shape != (3,)
            or not np.isfinite(
                np.concatenate((center_of_mass, right_foot, left_foot))
            ).all()
        ):
            return 0.0, 0.0, 0.0, 1.0, 0.0

        support_point = 0.5 * (right_foot + left_foot)
        try:
            foot_forces = np.asarray(
                (
                    getattr(state, "right_foot_normal_force_n"),
                    getattr(state, "left_foot_normal_force_n"),
                ),
                dtype=np.float64,
            )
        except (AttributeError, TypeError, ValueError):
            foot_forces = np.zeros(2, dtype=np.float64)
        if np.isfinite(foot_forces).all():
            foot_forces = np.maximum(foot_forces, 0.0)
            total_support_force_n = float(np.sum(foot_forces))
            if total_support_force_n >= self.config.capture_minimum_total_support_force_n:
                support_point = (
                    foot_forces[0] * right_foot + foot_forces[1] * left_foot
                ) / total_support_force_n
        bilateral_support_weight = 0.0
        if np.isfinite(foot_forces).all():
            load_progress = float(
                np.clip(
                    (
                        np.min(foot_forces)
                        - self.config.capture_full_gain_start_foot_force_n
                    )
                    / (
                        self.config.capture_full_gain_min_foot_force_n
                        - self.config.capture_full_gain_start_foot_force_n
                    ),
                    0.0,
                    1.0,
                )
            )
            bilateral_support_weight = load_progress * load_progress * (
                3.0 - 2.0 * load_progress
            )
        if self._filtered_support_point_m is None:
            self._filtered_support_point_m = support_point.copy()
        else:
            support_time_constant = (
                self.config.capture_support_point_filter_time_constant_s
            )
            support_alpha = (
                1.0
                if support_time_constant == 0.0
                else dt_s / (support_time_constant + dt_s)
            )
            self._filtered_support_point_m += support_alpha * (
                support_point - self._filtered_support_point_m
            )
        support_point = self._filtered_support_point_m
        com_offset_x_m = float(center_of_mass[0] - support_point[0])
        if self._neutral_com_offset_x_m is None:
            self._neutral_com_offset_x_m = com_offset_x_m
        if self._previous_com_offset_x_m is None:
            base_linear_velocity = np.asarray(
                getattr(state, "base_linear_velocity_m_s", np.zeros(3)),
                dtype=np.float64,
            )
            raw_velocity_x_m_s = (
                float(base_linear_velocity[0])
                if base_linear_velocity.shape == (3,)
                and np.isfinite(base_linear_velocity).all()
                else 0.0
            )
        else:
            raw_velocity_x_m_s = (
                com_offset_x_m - self._previous_com_offset_x_m
            ) / dt_s
        self._previous_com_offset_x_m = com_offset_x_m
        time_constant = self.config.capture_velocity_filter_time_constant_s
        velocity_alpha = 1.0 if time_constant == 0.0 else dt_s / (
            time_constant + dt_s
        )
        self._filtered_com_velocity_x_m_s += velocity_alpha * (
            raw_velocity_x_m_s - self._filtered_com_velocity_x_m_s
        )

        com_height_m = max(
            float(center_of_mass[2] - support_point[2]),
            self.config.capture_minimum_com_height_m,
        )
        natural_frequency_rad_s = float(np.sqrt(9.81 / com_height_m))
        capture_point_error_x_m = (
            com_offset_x_m
            - self._neutral_com_offset_x_m
            + self._filtered_com_velocity_x_m_s / natural_frequency_rad_s
        )
        margin_span = (
            self.config.capture_tracking_margin_full_m
            - self.config.capture_tracking_margin_start_m
        )
        progress = float(
            np.clip(
                (
                    abs(capture_point_error_x_m)
                    - self.config.capture_tracking_margin_start_m
                )
                / margin_span,
                0.0,
                1.0,
            )
        )
        capture_tracking_weight = 1.0 - progress * progress * (
            3.0 - 2.0 * progress
        )
        return (
            com_offset_x_m,
            self._filtered_com_velocity_x_m_s,
            capture_point_error_x_m,
            capture_tracking_weight,
            bilateral_support_weight,
        )

    def update(
        self,
        reference: RobotJointCommand,
        state: object,
        *,
        dt_s: float,
    ) -> RobotJointCommand:
        """Return one bounded position target from q/dq/IMU feedback."""

        dt_s = float(dt_s)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if tuple(reference.joint_names) != JOINT_NAMES:
            raise ValueError("reference must use canonical motor order")
        human = self._vector(reference.positions_rad, "reference.positions_rad")
        orientation = getattr(state, "base_orientation_wxyz")
        angular_velocity = np.asarray(
            getattr(state, "base_angular_velocity_rad_s"), dtype=np.float64
        )
        if angular_velocity.shape != (3,) or not np.isfinite(angular_velocity).all():
            raise ValueError("base angular velocity must be a finite 3-vector")

        roll, pitch, tilt = _projected_gravity_angles(orientation)
        span = self.config.recovery_tilt_rad - self.config.tracking_fade_start_rad
        tracking_weight = float(
            np.clip((self.config.recovery_tilt_rad - tilt) / span, 0.0, 1.0)
        )
        (
            com_offset_x_m,
            com_velocity_x_m_s,
            capture_point_error_x_m,
            capture_tracking_weight,
            bilateral_support_weight,
        ) = self._capture_observation(state, dt_s)
        tracking_weight *= capture_tracking_weight
        if reference.stale:
            # A stale camera pose is not an authorization to hold an arbitrary
            # imitation target.  Slew back toward neutral while retaining the
            # feedback-only ankle/capture recovery needed by the free base.
            tracking_weight = 0.0

        raw_lower_delta = human - self._home
        bilateral_hip_bend = max(
            0.0, min(raw_lower_delta[10], raw_lower_delta[11])
        )
        bilateral_knee_bend = max(
            0.0, min(raw_lower_delta[12], raw_lower_delta[13])
        )
        hip_progress = float(
            np.clip(
                (bilateral_hip_bend - self.config.unsupported_hip_fade_start_rad)
                / (
                    self.config.unsupported_hip_fade_full_rad
                    - self.config.unsupported_hip_fade_start_rad
                ),
                0.0,
                1.0,
            )
        )
        knee_progress = float(
            np.clip(
                (bilateral_knee_bend - self.config.unsupported_knee_fade_start_rad)
                / (
                    self.config.unsupported_knee_fade_full_rad
                    - self.config.unsupported_knee_fade_start_rad
                ),
                0.0,
                1.0,
            )
        )
        unsupported_progress = min(
            hip_progress * hip_progress * (3.0 - 2.0 * hip_progress),
            knee_progress * knee_progress * (3.0 - 2.0 * knee_progress),
        )
        tracking_weight *= 1.0 - unsupported_progress * (
            1.0 - self.config.unsupported_pose_tracking_scale
        )

        desired = self._home.copy()
        upper_delta = human[UPPER_BODY_INDICES] - self._home[UPPER_BODY_INDICES]
        desired[UPPER_BODY_INDICES] += tracking_weight * upper_delta
        shoulder_delta = np.clip(
            human[SHOULDER_PITCH_INDICES] - self._home[SHOULDER_PITCH_INDICES],
            -self.config.max_shoulder_deviation_rad,
            self.config.max_shoulder_deviation_rad,
        )
        desired[SHOULDER_PITCH_INDICES] = (
            self._home[SHOULDER_PITCH_INDICES] + tracking_weight * shoulder_delta
        )
        lower_delta = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        transverse_indices = np.concatenate(
            (HIP_YAW_INDICES, HIP_ROLL_INDICES, ANKLE_ROLL_INDICES)
        )
        lower_delta[transverse_indices] = (
            self.config.transverse_lower_body_imitation_scale
            * raw_lower_delta[transverse_indices]
        )

        # In double support, project the six independent sagittal targets onto
        # one symmetric, sole-flat crouch coordinate.  The joint-axis signs are
        # +Y hip, -Y knee, +Y ankle, hence 0.7 - 1.0 + 0.3 = 0 keeps the sole
        # pitch approximately unchanged.  Arbitrary independent targets move
        # the support polygon and topple this provisional free-base proxy.
        crouch_basis = np.asarray((0.7, 1.0, 0.3), dtype=np.float64)
        right_sagittal = np.asarray((10, 12, 14), dtype=np.int64)
        left_sagittal = np.asarray((11, 13, 15), dtype=np.int64)
        symmetric_sagittal = 0.5 * (
            raw_lower_delta[right_sagittal] + raw_lower_delta[left_sagittal]
        )
        crouch_amplitude = float(
            np.dot(symmetric_sagittal, crouch_basis)
            / np.dot(crouch_basis, crouch_basis)
        )
        scaled_crouch_amplitude = max(
            self.config.lower_body_imitation_scale * crouch_amplitude,
            -self.config.max_inverse_crouch_amplitude_rad,
        )
        crouch_delta = scaled_crouch_amplitude * crouch_basis
        lower_delta[right_sagittal] = crouch_delta
        lower_delta[left_sagittal] = crouch_delta

        lower_delta[LOWER_BODY_INDICES] = np.clip(
            lower_delta[LOWER_BODY_INDICES],
            -self._lower_body_deviation_limits[LOWER_BODY_INDICES],
            self._lower_body_deviation_limits[LOWER_BODY_INDICES],
        )
        desired[LOWER_BODY_INDICES] += (
            tracking_weight * lower_delta[LOWER_BODY_INDICES]
        )
        pose_reference = self._home.copy()
        pose_delta = np.clip(
            self.config.swing_leg_imitation_scale
            * raw_lower_delta[LOWER_BODY_INDICES],
            -self._lower_body_deviation_limits[LOWER_BODY_INDICES],
            self._lower_body_deviation_limits[LOWER_BODY_INDICES],
        )
        pose_reference[LOWER_BODY_INDICES] += tracking_weight * pose_delta

        # Follow the rate-limited motor trajectory, rather than the newly
        # received human target.  This keeps ankle and arm feed-forward motion
        # synchronized across abrupt camera cuts while still leading the
        # mechanically lagging arm by one servo tick.
        arm_elevation = max(
            0.0,
            float(
                np.mean(
                    self._last_output[SHOULDER_PITCH_INDICES]
                    - self._home[SHOULDER_PITCH_INDICES]
                )
            ),
        )
        ankle_residual = (
            self.config.ankle_pitch_bias_rad
            + self.config.arm_to_ankle_gain * arm_elevation
            + self.config.pitch_feedback_gain * pitch
            + self.config.pitch_rate_feedback_s * float(angular_velocity[1])
        )
        desired[ANKLE_PITCH_INDICES] += ankle_residual
        recovery_gain = (
            self.config.capture_recovery_gain_rad_per_m
            + bilateral_support_weight
            * (1.0 - capture_tracking_weight)
            * (
                self.config.capture_recovery_full_gain_rad_per_m
                - self.config.capture_recovery_gain_rad_per_m
            )
        )
        capture_recovery = float(
            np.clip(
                recovery_gain * capture_point_error_x_m,
                -self.config.capture_recovery_max_rad,
                self.config.capture_recovery_max_rad,
            )
        )
        # Shift the pressure response opposite the divergent capture-point
        # motion while keeping both legs symmetric.  The blended coefficients
        # obey hip - knee + ankle = 0 for this model's signed axes, preserving
        # sole pitch.  This is a motor-angle recovery, not a base-pose edit.
        raw_capture_recovery_positions = np.zeros(
            len(JOINT_NAMES), dtype=np.float64
        )
        raw_capture_recovery_positions[HIP_PITCH_INDICES] -= (
            0.5 * capture_recovery
        )
        raw_capture_recovery_positions[KNEE_INDICES] += 0.5 * capture_recovery
        raw_capture_recovery_positions[ANKLE_PITCH_INDICES] += capture_recovery
        desired_without_capture = desired.copy()
        desired += raw_capture_recovery_positions
        # The support FSM may replace the safe double-support leg projection
        # with this bounded pose candidate only after it has established an
        # intentional swing phase.  Keep the same live ankle stabilization in
        # both paths: omitting it from the candidate changes sole pitch at the
        # LIFT gate and can prevent a controlled touchdown.
        pose_reference[ANKLE_PITCH_INDICES] += ankle_residual
        pose_reference = np.clip(pose_reference, self._lower, self._upper)
        desired_without_capture = np.clip(
            desired_without_capture, self._lower, self._upper
        )
        desired = np.clip(desired, self._lower, self._upper)

        if self.config.enabled:
            maximum_change = self._rate_limits * dt_s
            safe = self._last_output + np.clip(
                desired - self._last_output, -maximum_change, maximum_change
            )
            safe_without_capture = self._last_output_without_capture + np.clip(
                desired_without_capture - self._last_output_without_capture,
                -maximum_change,
                maximum_change,
            )
        else:
            safe = np.clip(human, self._lower, self._upper)
            safe_without_capture = safe.copy()
            ankle_residual = 0.0
            tracking_weight = 1.0
        safe = np.clip(safe, self._lower, self._upper)
        safe_without_capture = np.clip(
            safe_without_capture, self._lower, self._upper
        )
        deployed_capture_recovery_positions = safe - safe_without_capture
        self._last_output = safe.copy()
        self._last_output_without_capture = safe_without_capture.copy()
        self._last_diagnostics = BalanceDiagnostics(
            roll_rad=roll,
            pitch_rad=pitch,
            tilt_rad=tilt,
            tracking_weight=tracking_weight,
            ankle_pitch_residual_rad=float(ankle_residual),
            com_offset_x_m=com_offset_x_m,
            com_velocity_x_m_s=com_velocity_x_m_s,
            capture_point_error_x_m=capture_point_error_x_m,
            capture_tracking_weight=capture_tracking_weight,
            capture_recovery_rad=capture_recovery,
            reference_positions_rad=human.copy(),
            safe_positions_rad=safe.copy(),
        )
        return BalancedJointCommand(
            reference.timestamp_s,
            JOINT_NAMES,
            safe,
            reference.confidence,
            reference.stale,
            pose_reference_positions_rad=pose_reference,
            capture_recovery_positions_rad=deployed_capture_recovery_positions,
        )
