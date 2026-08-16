"""Convert a camera skeleton into a debounced one-leg support request.

The MediaPipe body-up vector follows the torso and therefore tilts when a
person leans.  Using that instantaneous vector to compare the feet can invert
the apparent lifted leg.  This estimator learns one fixed camera-space up axis
and the neutral left/right foot offset during the same initial neutral hold as
the retargeter, then measures foot-height difference in leg-length units.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import yaml

from robot_human_interface.pose.calibration import (
    NeutralCalibrationError,
    NeutralCalibrationGate,
)
from robot_human_interface.retargeting.geometry import body_basis
from robot_human_interface.skeleton import PoseLandmark as L, SkeletonFrame

from .support import SupportIntent


_CORE = (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP)
_LEGS = (L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE)
_MEASUREMENT = _CORE + _LEGS
_BALANCE_SECTIONS = {
    "standing_balance",
    "human_support_intent",
    "support_control",
}


@dataclass(frozen=True, slots=True)
class HumanSupportIntentConfig:
    """Thresholds for calibrated one-leg and bilateral-squat intent."""

    calibration_frames: int = 30
    calibration_max_observations: int = 150
    calibration_max_pose_spread_ratio: float = 0.06
    calibration_max_ankle_offset_ratio: float = 0.08
    calibration_max_ankle_spread_ratio: float = 0.035
    calibration_max_arm_deviation_rad: float = pi / 3.0
    calibration_max_upper_arm_deviation_rad: float = pi / 3.0
    calibration_max_elbow_flexion_rad: float = pi / 3.0
    calibration_max_knee_flexion_rad: float = pi / 4.0
    confidence_threshold: float = 0.5
    activate_height_ratio: float = 0.15
    release_height_ratio: float = 0.08
    activation_hold_s: float = 0.30
    release_hold_s: float = 0.20
    filter_time_constant_s: float = 0.10
    max_gap_s: float = 0.25
    activation_confidence: float = 0.65
    maintain_confidence: float = 0.50
    squat_enter_pelvis_descent_ratio: float = 0.10
    squat_exit_pelvis_descent_ratio: float = 0.06
    squat_full_pelvis_descent_ratio: float = 0.50
    squat_enter_hip_flexion_rad: float = 20.0 * pi / 180.0
    squat_exit_hip_flexion_rad: float = 12.0 * pi / 180.0
    squat_enter_knee_flexion_rad: float = 25.0 * pi / 180.0
    squat_exit_knee_flexion_rad: float = 15.0 * pi / 180.0
    squat_activate_max_ankle_asymmetry_ratio: float = 0.10
    squat_maintain_max_ankle_asymmetry_ratio: float = 0.14
    squat_activate_max_stance_width_change_ratio: float = 0.10
    squat_maintain_max_stance_width_change_ratio: float = 0.15
    squat_activation_hold_s: float = 0.05
    squat_release_hold_s: float = 0.15
    squat_filter_time_constant_s: float = 0.05

    def __post_init__(self) -> None:
        _require_real_config_fields(
            self,
            type(self).__dataclass_fields__,
            section="human_support_intent",
        )
        if (
            not np.isfinite(float(self.calibration_frames))
            or int(self.calibration_frames) != self.calibration_frames
            or self.calibration_frames <= 0
        ):
            raise ValueError("calibration_frames must be a positive integer")
        object.__setattr__(self, "calibration_frames", int(self.calibration_frames))
        if (
            not np.isfinite(float(self.calibration_max_observations))
            or int(self.calibration_max_observations)
            != self.calibration_max_observations
            or self.calibration_max_observations < self.calibration_frames
        ):
            raise ValueError(
                "calibration_max_observations must be an integer covering "
                "calibration_frames"
            )
        object.__setattr__(
            self,
            "calibration_max_observations",
            int(self.calibration_max_observations),
        )
        calibration_limits = (
            self.calibration_max_pose_spread_ratio,
            self.calibration_max_ankle_offset_ratio,
            self.calibration_max_ankle_spread_ratio,
            self.calibration_max_arm_deviation_rad,
            self.calibration_max_upper_arm_deviation_rad,
            self.calibration_max_elbow_flexion_rad,
            self.calibration_max_knee_flexion_rad,
        )
        if not np.isfinite(calibration_limits).all() or any(
            value <= 0.0 for value in calibration_limits
        ):
            raise ValueError("calibration stillness/neutral limits must be finite and positive")
        if (
            self.calibration_max_arm_deviation_rad >= pi
            or self.calibration_max_upper_arm_deviation_rad >= pi
            or self.calibration_max_elbow_flexion_rad >= pi
            or self.calibration_max_knee_flexion_rad >= pi
        ):
            raise ValueError("calibration posture angle limits must be below pi")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        values = (
            self.activate_height_ratio,
            self.release_height_ratio,
            self.activation_hold_s,
            self.release_hold_s,
            self.filter_time_constant_s,
            self.max_gap_s,
            self.activation_confidence,
            self.maintain_confidence,
            self.squat_enter_pelvis_descent_ratio,
            self.squat_exit_pelvis_descent_ratio,
            self.squat_full_pelvis_descent_ratio,
            self.squat_enter_hip_flexion_rad,
            self.squat_exit_hip_flexion_rad,
            self.squat_enter_knee_flexion_rad,
            self.squat_exit_knee_flexion_rad,
            self.squat_activate_max_ankle_asymmetry_ratio,
            self.squat_maintain_max_ankle_asymmetry_ratio,
            self.squat_activate_max_stance_width_change_ratio,
            self.squat_maintain_max_stance_width_change_ratio,
            self.squat_activation_hold_s,
            self.squat_release_hold_s,
            self.squat_filter_time_constant_s,
        )
        if not np.isfinite(values).all() or any(value <= 0.0 for value in values):
            raise ValueError("intent thresholds and time constants must be finite and positive")
        if self.release_height_ratio >= self.activate_height_ratio:
            raise ValueError("release_height_ratio must be below activate_height_ratio")
        if not 0.0 <= self.maintain_confidence <= self.activation_confidence <= 1.0:
            raise ValueError(
                "confidence gates must satisfy 0 <= maintain <= activation <= 1"
            )
        if not (
            0.0
            < self.squat_exit_pelvis_descent_ratio
            < self.squat_enter_pelvis_descent_ratio
            < self.squat_full_pelvis_descent_ratio
            < 1.0
        ):
            raise ValueError(
                "squat pelvis-descent thresholds must be positive and ordered"
            )
        if not (
            0.0
            < self.squat_exit_hip_flexion_rad
            < self.squat_enter_hip_flexion_rad
            < pi
        ):
            raise ValueError("squat hip-flexion thresholds must be ordered below pi")
        if not (
            0.0
            < self.squat_exit_knee_flexion_rad
            < self.squat_enter_knee_flexion_rad
            < pi
        ):
            raise ValueError("squat knee-flexion thresholds must be ordered below pi")
        if not (
            self.squat_activate_max_ankle_asymmetry_ratio
            < self.squat_maintain_max_ankle_asymmetry_ratio
            < 1.0
        ):
            raise ValueError("squat ankle-asymmetry limits must be ordered below one")
        if not (
            self.squat_activate_max_stance_width_change_ratio
            < self.squat_maintain_max_stance_width_change_ratio
            < 1.0
        ):
            raise ValueError("squat stance-width limits must be ordered below one")


def _require_real_config_fields(
    config: object,
    field_names: Iterable[str],
    *,
    section: str,
) -> None:
    for name in field_names:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"{section}.{name} must be a real number "
                "(booleans are not accepted)"
            )


@dataclass(frozen=True, slots=True)
class HumanSupportEstimate:
    """Observable camera-side support and bilateral-squat estimate."""

    intent: SupportIntent
    signed_height_ratio: float
    confidence: float
    calibrated: bool
    stale: bool
    squat_active: bool = False
    squat_observation_fresh: bool = False
    squat_depth_ratio: float = 0.0
    squat_pelvis_descent_ratio: float = 0.0
    squat_bilateral_hip_flexion_rad: float = 0.0
    squat_bilateral_knee_flexion_rad: float = 0.0
    squat_confidence: float = 0.0
    squat_ankle_asymmetry_ratio: float = 0.0
    squat_stance_width_change_ratio: float = 0.0

    @property
    def right_lift_ratio(self) -> float:
        return max(0.0, self.signed_height_ratio)

    @property
    def left_lift_ratio(self) -> float:
        return max(0.0, -self.signed_height_ratio)


@dataclass(frozen=True, slots=True)
class _SquatGeometry:
    """One fully observable image-plane squat measurement."""

    up_2d: np.ndarray
    leg_scale_2d: float
    pelvis_height_2d: float
    stance_width_2d: float
    ankle_asymmetry_2d: float
    hip_flexion_rad: np.ndarray
    knee_flexion_rad: np.ndarray
    confidence: float


def _angle_between(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not np.isfinite(denominator) or denominator < 1e-8:
        return None
    cosine = float(np.dot(first, second) / denominator)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


class HumanSupportIntentEstimator:
    """Estimate unilateral support and bilateral squat from one calibration."""

    def __init__(self, config: HumanSupportIntentConfig | None = None) -> None:
        self.config = config or HumanSupportIntentConfig()
        self.reset()

    def reset(self) -> None:
        self._up_samples: list[np.ndarray] = []
        self._difference_samples: list[np.ndarray] = []
        self._scale_samples: list[float] = []
        self._fixed_up: np.ndarray | None = None
        self._fixed_scale: float | None = None
        self._baseline_ratio = 0.0
        self._filtered_ratio = 0.0
        self._squat_geometry_samples: list[_SquatGeometry] = []
        self._squat_frame_samples: list[SkeletonFrame] = []
        self._fixed_squat_up_2d: np.ndarray | None = None
        self._fixed_squat_leg_scale_2d: float | None = None
        self._neutral_pelvis_height_2d: float | None = None
        self._neutral_stance_width_2d: float | None = None
        self._neutral_hip_flexion_rad: np.ndarray | None = None
        self._neutral_knee_flexion_rad: np.ndarray | None = None
        self._filtered_squat_pelvis_descent_ratio = 0.0
        self._filtered_squat_min_hip_flexion_rad = 0.0
        self._filtered_squat_min_knee_flexion_rad = 0.0
        self._filtered_squat_ankle_asymmetry_ratio = 0.0
        self._filtered_squat_stance_width_change_ratio = 0.0
        self._squat_active = False
        self._squat_candidate = False
        self._squat_candidate_since_s: float | None = None
        self._intent = SupportIntent.DOUBLE_SUPPORT
        self._candidate = SupportIntent.DOUBLE_SUPPORT
        self._candidate_since_s: float | None = None
        self._last_timestamp_s: float | None = None
        self._last_valid_timestamp_s: float | None = None
        self._last_squat_valid_timestamp_s: float | None = None
        self._last_confidence = 0.0
        self._last_squat_confidence = 0.0
        self._calibration_gate: NeutralCalibrationGate | None = (
            self._new_calibration_gate(self.config.calibration_frames)
        )

    def _new_calibration_gate(
        self,
        sample_count: int,
        *,
        max_observations: int | None = None,
    ) -> NeutralCalibrationGate:
        return NeutralCalibrationGate(
            sample_count=sample_count,
            max_observations=(
                self.config.calibration_max_observations
                if max_observations is None
                else max_observations
            ),
            landmark_indices=_MEASUREMENT,
            confidence_threshold=self.config.confidence_threshold,
            max_pose_spread_ratio=self.config.calibration_max_pose_spread_ratio,
            require_double_support=True,
            max_ankle_offset_ratio=self.config.calibration_max_ankle_offset_ratio,
            max_ankle_spread_ratio=self.config.calibration_max_ankle_spread_ratio,
            max_arm_deviation_rad=self.config.calibration_max_arm_deviation_rad,
            max_upper_arm_deviation_rad=(
                self.config.calibration_max_upper_arm_deviation_rad
            ),
            max_elbow_flexion_rad=self.config.calibration_max_elbow_flexion_rad,
            max_knee_flexion_rad=self.config.calibration_max_knee_flexion_rad,
            require_extended_legs=True,
            label="human support-intent neutral-pose",
        )

    def start_calibration(self) -> None:
        self.reset()

    def calibrate(self, frame: SkeletonFrame) -> bool:
        """Install one explicitly selected, posture-validated neutral frame."""

        measurement = self._measurement(frame)
        if measurement is None or measurement[3] < self.config.activation_confidence:
            return False
        gate = self._new_calibration_gate(1, max_observations=1)
        if not gate.accepts_explicit(frame):
            return False
        squat_geometry = self._squat_geometry(frame)
        if (
            squat_geometry is None
            or squat_geometry.confidence < self.config.activation_confidence
        ):
            return False
        up, difference, scale, confidence = measurement
        self.reset()
        self._up_samples = [up.copy()]
        self._difference_samples = [difference.copy()]
        self._scale_samples = [scale]
        self._squat_geometry_samples = [squat_geometry]
        self._squat_frame_samples = [frame]
        # The selected frame can come from a separate replay whose media clock
        # is unrelated to the main source. Install geometry only; let the
        # first operational frame establish the debounce/filter timeline.
        self._last_timestamp_s = None
        self._last_valid_timestamp_s = None
        self._last_confidence = confidence
        self._finish_calibration_if_ready(force=True)
        return not self.is_calibrating

    @property
    def is_calibrating(self) -> bool:
        return self._fixed_up is None

    @property
    def calibration_progress(self) -> float:
        if not self.is_calibrating:
            return 1.0
        assert self._calibration_gate is not None
        return self._calibration_gate.progress

    @property
    def intent(self) -> SupportIntent:
        return self._intent

    def _measurement(
        self, frame: SkeletonFrame
    ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        valid = frame.valid_mask(self.config.confidence_threshold)
        indices = np.asarray([int(index) for index in _MEASUREMENT], dtype=np.int64)
        if not bool(np.all(valid[indices])):
            return None
        points = frame.landmarks_3d
        left_thigh = np.linalg.norm(
            points[int(L.LEFT_HIP)] - points[int(L.LEFT_KNEE)]
        )
        left_shin = np.linalg.norm(
            points[int(L.LEFT_KNEE)] - points[int(L.LEFT_ANKLE)]
        )
        right_thigh = np.linalg.norm(
            points[int(L.RIGHT_HIP)] - points[int(L.RIGHT_KNEE)]
        )
        right_shin = np.linalg.norm(
            points[int(L.RIGHT_KNEE)] - points[int(L.RIGHT_ANKLE)]
        )
        scale = 0.5 * float(left_thigh + left_shin + right_thigh + right_shin)
        if not np.isfinite(scale) or scale < 1e-4:
            return None
        up = body_basis(points).vertical_up
        if not np.isfinite(up).all():
            return None
        confidence_values = frame.confidence()[indices]
        confidence = min(
            float(np.quantile(confidence_values, 0.10)),
            float(frame.confidence()[int(L.LEFT_ANKLE)]),
            float(frame.confidence()[int(L.RIGHT_ANKLE)]),
        )
        difference = points[int(L.RIGHT_ANKLE)] - points[int(L.LEFT_ANKLE)]
        return up, difference, scale, confidence

    def _squat_geometry(
        self,
        frame: SkeletonFrame,
        *,
        fixed_up_2d: np.ndarray | None = None,
    ) -> _SquatGeometry | None:
        """Measure bilateral flexion and pelvis height in the camera plane."""

        valid = frame.valid_mask(self.config.confidence_threshold)
        indices = np.asarray([int(index) for index in _MEASUREMENT], dtype=np.int64)
        if not bool(np.all(valid[indices])):
            return None
        points = frame.landmarks_2d.copy()
        if frame.image_size is not None:
            width, height = frame.image_size
            points[:, 0] *= float(width) / float(height)
        left_shoulder = points[int(L.LEFT_SHOULDER)]
        right_shoulder = points[int(L.RIGHT_SHOULDER)]
        left_hip = points[int(L.LEFT_HIP)]
        right_hip = points[int(L.RIGHT_HIP)]
        left_knee = points[int(L.LEFT_KNEE)]
        right_knee = points[int(L.RIGHT_KNEE)]
        left_ankle = points[int(L.LEFT_ANKLE)]
        right_ankle = points[int(L.RIGHT_ANKLE)]
        vectors = np.concatenate(
            (
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip,
                left_knee,
                right_knee,
                left_ankle,
                right_ankle,
            )
        )
        if not np.isfinite(vectors).all():
            return None
        shoulder_mid = 0.5 * (left_shoulder + right_shoulder)
        hip_mid = 0.5 * (left_hip + right_hip)
        ankle_mid = 0.5 * (left_ankle + right_ankle)
        observed_up_2d = shoulder_mid - hip_mid
        up_norm = float(np.linalg.norm(observed_up_2d))
        if up_norm < 1e-6:
            return None
        observed_up_2d = observed_up_2d / up_norm
        up_2d = observed_up_2d if fixed_up_2d is None else fixed_up_2d
        leg_scale = 0.5 * float(
            np.linalg.norm(left_hip - left_knee)
            + np.linalg.norm(left_knee - left_ankle)
            + np.linalg.norm(right_hip - right_knee)
            + np.linalg.norm(right_knee - right_ankle)
        )
        if not np.isfinite(leg_scale) or leg_scale < 1e-5:
            return None
        pelvis_height = float(np.dot(ankle_mid - hip_mid, -up_2d))
        horizontal = np.asarray((-up_2d[1], up_2d[0]), dtype=np.float64)
        stance_width = abs(float(np.dot(right_ankle - left_ankle, horizontal)))
        ankle_asymmetry = abs(float(np.dot(right_ankle - left_ankle, up_2d)))
        hip_flexions: list[float] = []
        knee_flexions: list[float] = []
        torso_down = hip_mid - shoulder_mid
        for hip, knee, ankle in (
            (left_hip, left_knee, left_ankle),
            (right_hip, right_knee, right_ankle),
        ):
            hip_internal = _angle_between(knee - hip, torso_down)
            knee_internal = _angle_between(hip - knee, ankle - knee)
            if hip_internal is None or knee_internal is None:
                return None
            hip_flexions.append(max(0.0, hip_internal))
            knee_flexions.append(max(0.0, pi - knee_internal))
        confidence = float(np.min(frame.confidence()[indices]))
        return _SquatGeometry(
            up_2d.copy(),
            leg_scale,
            pelvis_height,
            stance_width,
            ankle_asymmetry,
            np.asarray(hip_flexions, dtype=np.float64),
            np.asarray(knee_flexions, dtype=np.float64),
            confidence,
        )

    def _finish_calibration_if_ready(self, *, force: bool = False) -> None:
        if not force and len(self._up_samples) < self.config.calibration_frames:
            return
        fixed_up = np.median(np.asarray(self._up_samples), axis=0)
        # MediaPipe Z is learned monocular depth.  Retaining the torso's depth
        # component makes a forward lean look like one foot changed height.
        # Keep the calibrated image-plane roll, but anchor vertical to the
        # camera image plane; a later RGB-D/IMU fusion can supply true gravity.
        fixed_up[2] = 0.0
        norm = float(np.linalg.norm(fixed_up))
        if not np.isfinite(norm) or norm < 1e-6:
            raise NeutralCalibrationError(
                "human support-intent neutral-pose calibration produced an "
                "invalid vertical reference; explicitly restart calibration"
            )
        self._fixed_up = fixed_up / norm
        self._fixed_scale = float(np.median(self._scale_samples))
        if not np.isfinite(self._fixed_scale) or self._fixed_scale < 1e-4:
            raise NeutralCalibrationError(
                "human support-intent neutral-pose calibration produced an "
                "invalid leg scale; explicitly restart calibration"
            )
        ratios = [
            float(np.dot(difference, self._fixed_up) / self._fixed_scale)
            for difference in self._difference_samples
        ]
        self._baseline_ratio = float(np.median(ratios))
        self._filtered_ratio = 0.0
        if not self._squat_geometry_samples or not self._squat_frame_samples:
            raise NeutralCalibrationError(
                "human squat-intent calibration has no admitted geometry"
            )
        fixed_squat_up = np.median(
            np.asarray([item.up_2d for item in self._squat_geometry_samples]),
            axis=0,
        )
        squat_up_norm = float(np.linalg.norm(fixed_squat_up))
        if not np.isfinite(squat_up_norm) or squat_up_norm < 1e-6:
            raise NeutralCalibrationError(
                "human squat-intent calibration produced an invalid image up axis"
            )
        self._fixed_squat_up_2d = fixed_squat_up / squat_up_norm
        fixed_geometry = [
            self._squat_geometry(sample, fixed_up_2d=self._fixed_squat_up_2d)
            for sample in self._squat_frame_samples
        ]
        if any(item is None for item in fixed_geometry):
            raise NeutralCalibrationError(
                "human squat-intent admitted geometry became unobservable"
            )
        squat_geometry = [item for item in fixed_geometry if item is not None]
        self._fixed_squat_leg_scale_2d = float(
            np.median([item.leg_scale_2d for item in squat_geometry])
        )
        if (
            not np.isfinite(self._fixed_squat_leg_scale_2d)
            or self._fixed_squat_leg_scale_2d < 1e-5
        ):
            raise NeutralCalibrationError(
                "human squat-intent calibration produced an invalid image leg scale"
            )
        self._neutral_pelvis_height_2d = float(
            np.median([item.pelvis_height_2d for item in squat_geometry])
            / self._fixed_squat_leg_scale_2d
        )
        self._neutral_stance_width_2d = float(
            np.median([item.stance_width_2d for item in squat_geometry])
            / self._fixed_squat_leg_scale_2d
        )
        self._neutral_hip_flexion_rad = np.median(
            np.asarray([item.hip_flexion_rad for item in squat_geometry]),
            axis=0,
        )
        self._neutral_knee_flexion_rad = np.median(
            np.asarray([item.knee_flexion_rad for item in squat_geometry]),
            axis=0,
        )
        if not np.isfinite(
            np.concatenate(
                (
                    np.asarray(
                        (
                            self._neutral_pelvis_height_2d,
                            self._neutral_stance_width_2d,
                        )
                    ),
                    self._neutral_hip_flexion_rad,
                    self._neutral_knee_flexion_rad,
                )
            )
        ).all():
            raise NeutralCalibrationError(
                "human squat-intent calibration produced non-finite references"
            )
        self._calibration_gate = None

    def _raw_candidate(self, ratio: float) -> SupportIntent:
        # A side change must pass through a confirmed two-foot state.  This is
        # a safety contract even if a noisy measurement jumps across zero.
        if self._intent is SupportIntent.RIGHT_SWING:
            return (
                SupportIntent.RIGHT_SWING
                if ratio > self.config.release_height_ratio
                else SupportIntent.DOUBLE_SUPPORT
            )
        if self._intent is SupportIntent.LEFT_SWING:
            return (
                SupportIntent.LEFT_SWING
                if ratio < -self.config.release_height_ratio
                else SupportIntent.DOUBLE_SUPPORT
            )
        if ratio >= self.config.activate_height_ratio:
            return SupportIntent.RIGHT_SWING
        if ratio <= -self.config.activate_height_ratio:
            return SupportIntent.LEFT_SWING
        return SupportIntent.DOUBLE_SUPPORT

    def _debounce(self, candidate: SupportIntent, timestamp_s: float) -> None:
        if candidate is self._intent:
            self._candidate = candidate
            self._candidate_since_s = None
            return
        if candidate is not self._candidate:
            self._candidate = candidate
            self._candidate_since_s = timestamp_s
            return
        if self._candidate_since_s is None:
            self._candidate_since_s = timestamp_s
            return
        required = (
            self.config.release_hold_s
            if candidate is SupportIntent.DOUBLE_SUPPORT
            else self.config.activation_hold_s
        )
        if timestamp_s - self._candidate_since_s >= required:
            self._intent = candidate
            self._candidate_since_s = None

    def _reset_squat_request(self, *, clear_filters: bool) -> None:
        self._squat_active = False
        self._squat_candidate = False
        self._squat_candidate_since_s = None
        if clear_filters:
            self._filtered_squat_pelvis_descent_ratio = 0.0
            self._filtered_squat_min_hip_flexion_rad = 0.0
            self._filtered_squat_min_knee_flexion_rad = 0.0
            self._filtered_squat_ankle_asymmetry_ratio = 0.0
            self._filtered_squat_stance_width_change_ratio = 0.0

    def _debounce_squat(self, candidate: bool, timestamp_s: float) -> None:
        if candidate is self._squat_active:
            self._squat_candidate = candidate
            self._squat_candidate_since_s = None
            return
        if candidate is not self._squat_candidate:
            self._squat_candidate = candidate
            self._squat_candidate_since_s = timestamp_s
            return
        if self._squat_candidate_since_s is None:
            self._squat_candidate_since_s = timestamp_s
            return
        required = (
            self.config.squat_release_hold_s
            if not candidate
            else self.config.squat_activation_hold_s
        )
        if timestamp_s - self._squat_candidate_since_s >= required:
            self._squat_active = candidate
            self._squat_candidate_since_s = None

    def _update_squat_measurement(
        self,
        geometry: _SquatGeometry,
        *,
        timestamp_s: float,
        previous_timestamp_s: float | None,
    ) -> None:
        assert self._fixed_squat_leg_scale_2d is not None
        assert self._neutral_pelvis_height_2d is not None
        assert self._neutral_stance_width_2d is not None
        assert self._neutral_hip_flexion_rad is not None
        assert self._neutral_knee_flexion_rad is not None
        scale = self._fixed_squat_leg_scale_2d
        pelvis_descent = max(
            0.0,
            self._neutral_pelvis_height_2d - geometry.pelvis_height_2d / scale,
        )
        hip_flexion = float(
            np.min(np.maximum(geometry.hip_flexion_rad - self._neutral_hip_flexion_rad, 0.0))
        )
        knee_flexion = float(
            np.min(
                np.maximum(
                    geometry.knee_flexion_rad - self._neutral_knee_flexion_rad,
                    0.0,
                )
            )
        )
        ankle_asymmetry = geometry.ankle_asymmetry_2d / scale
        stance_width_change = abs(
            geometry.stance_width_2d / scale - self._neutral_stance_width_2d
        )
        dt_s = (
            0.0
            if previous_timestamp_s is None
            else timestamp_s - previous_timestamp_s
        )
        alpha = (
            1.0
            if dt_s <= 0.0
            else 1.0
            - exp(-dt_s / self.config.squat_filter_time_constant_s)
        )
        for name, value in (
            ("_filtered_squat_pelvis_descent_ratio", pelvis_descent),
            ("_filtered_squat_min_hip_flexion_rad", hip_flexion),
            ("_filtered_squat_min_knee_flexion_rad", knee_flexion),
            ("_filtered_squat_ankle_asymmetry_ratio", ankle_asymmetry),
            ("_filtered_squat_stance_width_change_ratio", stance_width_change),
        ):
            previous = float(getattr(self, name))
            setattr(self, name, previous + alpha * (value - previous))
        self._last_squat_confidence = geometry.confidence
        self._last_squat_valid_timestamp_s = timestamp_s

        if self._squat_active:
            candidate = bool(
                self._filtered_squat_pelvis_descent_ratio
                >= self.config.squat_exit_pelvis_descent_ratio
                and self._filtered_squat_min_hip_flexion_rad
                >= self.config.squat_exit_hip_flexion_rad
                and self._filtered_squat_min_knee_flexion_rad
                >= self.config.squat_exit_knee_flexion_rad
                and self._filtered_squat_ankle_asymmetry_ratio
                <= self.config.squat_maintain_max_ankle_asymmetry_ratio
                and self._filtered_squat_stance_width_change_ratio
                <= self.config.squat_maintain_max_stance_width_change_ratio
                and geometry.confidence >= self.config.maintain_confidence
            )
        else:
            candidate = bool(
                self._intent is SupportIntent.DOUBLE_SUPPORT
                and self._filtered_squat_pelvis_descent_ratio
                >= self.config.squat_enter_pelvis_descent_ratio
                and self._filtered_squat_min_hip_flexion_rad
                >= self.config.squat_enter_hip_flexion_rad
                and self._filtered_squat_min_knee_flexion_rad
                >= self.config.squat_enter_knee_flexion_rad
                and self._filtered_squat_ankle_asymmetry_ratio
                <= self.config.squat_activate_max_ankle_asymmetry_ratio
                and self._filtered_squat_stance_width_change_ratio
                <= self.config.squat_activate_max_stance_width_change_ratio
                and geometry.confidence >= self.config.activation_confidence
            )
        self._debounce_squat(candidate, timestamp_s)
        if self._squat_active:
            # Squat and a unilateral support request are mutually exclusive.
            self._intent = SupportIntent.DOUBLE_SUPPORT
            self._candidate = SupportIntent.DOUBLE_SUPPORT
            self._candidate_since_s = None

    def _estimate(
        self,
        *,
        calibrated: bool,
        stale: bool,
        squat_observation_fresh: bool,
    ) -> HumanSupportEstimate:
        depth = float(
            np.clip(
                (
                    self._filtered_squat_pelvis_descent_ratio
                    - self.config.squat_exit_pelvis_descent_ratio
                )
                / (
                    self.config.squat_full_pelvis_descent_ratio
                    - self.config.squat_exit_pelvis_descent_ratio
                ),
                0.0,
                1.0,
            )
        )
        return HumanSupportEstimate(
            self._intent,
            self._filtered_ratio,
            self._last_confidence,
            calibrated=calibrated,
            stale=stale,
            squat_active=bool(self._squat_active and calibrated and not stale),
            squat_observation_fresh=bool(
                squat_observation_fresh and calibrated and not stale
            ),
            squat_depth_ratio=(
                depth if self._squat_active and calibrated and not stale else 0.0
            ),
            squat_pelvis_descent_ratio=(
                self._filtered_squat_pelvis_descent_ratio
            ),
            squat_bilateral_hip_flexion_rad=(
                self._filtered_squat_min_hip_flexion_rad
            ),
            squat_bilateral_knee_flexion_rad=(
                self._filtered_squat_min_knee_flexion_rad
            ),
            squat_confidence=self._last_squat_confidence,
            squat_ankle_asymmetry_ratio=(
                self._filtered_squat_ankle_asymmetry_ratio
            ),
            squat_stance_width_change_ratio=(
                self._filtered_squat_stance_width_change_ratio
            ),
        )

    def update(
        self,
        frame: SkeletonFrame | None,
        *,
        timestamp_s: float | None = None,
    ) -> HumanSupportEstimate:
        """Consume one pose sample and return a debounced support request."""

        if frame is not None:
            timestamp = float(frame.timestamp_s)
        elif timestamp_s is not None:
            timestamp = float(timestamp_s)
        else:
            raise ValueError("timestamp_s is required when frame is None")
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if self._last_timestamp_s is not None and timestamp < self._last_timestamp_s:
            raise ValueError("support-intent timestamps must be monotonic")
        previous_timestamp = self._last_timestamp_s
        self._last_timestamp_s = timestamp

        measurement = None if frame is None else self._measurement(frame)
        if measurement is not None:
            self._last_confidence = measurement[3]
            required_confidence = (
                self.config.activation_confidence
                if self.is_calibrating
                or (
                    self._intent is SupportIntent.DOUBLE_SUPPORT
                    and not self._squat_active
                )
                else self.config.maintain_confidence
            )
            if measurement[3] < required_confidence:
                measurement = None
            elif self.is_calibrating:
                # Support intent deliberately uses a robust lower quantile,
                # but squat geometry needs every shoulder/hip/knee/ankle.
                # Do not admit a neutral reference with one weak landmark:
                # it would permanently bias pelvis height or joint flexion.
                calibration_geometry = self._squat_geometry(frame)
                if (
                    calibration_geometry is None
                    or calibration_geometry.confidence
                    < self.config.activation_confidence
                ):
                    measurement = None
        if measurement is None:
            stale = bool(
                self._last_valid_timestamp_s is None
                or timestamp - self._last_valid_timestamp_s > self.config.max_gap_s
            )
            if stale:
                self._intent = SupportIntent.DOUBLE_SUPPORT
                self._candidate = SupportIntent.DOUBLE_SUPPORT
                self._candidate_since_s = None
                self._filtered_ratio = 0.0
                self._reset_squat_request(clear_filters=True)
            elif not self._squat_active:
                # Missing data can never accumulate an entry dwell.
                self._squat_candidate = False
                self._squat_candidate_since_s = None
            return self._estimate(
                calibrated=not self.is_calibrating,
                stale=stale,
                squat_observation_fresh=False,
            )

        up, difference, scale, confidence = measurement
        self._last_valid_timestamp_s = timestamp
        self._last_confidence = confidence
        if self.is_calibrating:
            assert self._calibration_gate is not None
            accepted = self._calibration_gate.observe(frame)
            if accepted is not None:
                accepted_measurements = [
                    self._measurement(sample) for sample in accepted
                ]
                if any(item is None for item in accepted_measurements):
                    raise NeutralCalibrationError(
                        "accepted support-intent neutral window became "
                        "unobservable; explicitly restart calibration"
                    )
                measurements = [
                    item for item in accepted_measurements if item is not None
                ]
                self._up_samples = [item[0].copy() for item in measurements]
                self._difference_samples = [
                    item[1].copy() for item in measurements
                ]
                self._scale_samples = [item[2] for item in measurements]
                accepted_squat_geometry = [
                    self._squat_geometry(sample) for sample in accepted
                ]
                if any(item is None for item in accepted_squat_geometry):
                    raise NeutralCalibrationError(
                        "accepted squat-intent neutral window became "
                        "unobservable; explicitly restart calibration"
                    )
                self._squat_geometry_samples = [
                    item for item in accepted_squat_geometry if item is not None
                ]
                self._squat_frame_samples = list(accepted)
                self._finish_calibration_if_ready()
            return self._estimate(
                calibrated=not self.is_calibrating,
                stale=False,
                squat_observation_fresh=False,
            )

        assert self._fixed_up is not None and self._fixed_scale is not None
        ratio = float(
            np.dot(difference, self._fixed_up) / self._fixed_scale
            - self._baseline_ratio
        )
        dt_s = 0.0 if previous_timestamp is None else timestamp - previous_timestamp
        alpha = 1.0 if dt_s <= 0.0 else 1.0 - exp(-dt_s / self.config.filter_time_constant_s)
        self._filtered_ratio += alpha * (ratio - self._filtered_ratio)
        self._debounce(self._raw_candidate(self._filtered_ratio), timestamp)
        assert self._fixed_squat_up_2d is not None
        squat_geometry = self._squat_geometry(
            frame,
            fixed_up_2d=self._fixed_squat_up_2d,
        )
        if squat_geometry is None:
            if not self._squat_active:
                self._squat_candidate = False
                self._squat_candidate_since_s = None
        else:
            self._update_squat_measurement(
                squat_geometry,
                timestamp_s=timestamp,
                previous_timestamp_s=previous_timestamp,
            )
        squat_stale = bool(
            self._last_squat_valid_timestamp_s is None
            or timestamp - self._last_squat_valid_timestamp_s > self.config.max_gap_s
        )
        if squat_stale:
            self._reset_squat_request(clear_filters=True)
        return self._estimate(
            calibrated=True,
            stale=False,
            squat_observation_fresh=squat_geometry is not None,
        )


def load_human_support_intent_config(
    path: str | Path | None,
) -> HumanSupportIntentConfig:
    """Load the optional ``human_support_intent`` block from balance YAML."""

    if path is None or not Path(path).is_file():
        return HumanSupportIntentConfig()
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    if not isinstance(document, Mapping):
        raise ValueError("balance YAML root must be a mapping")
    unknown_sections = set(document) - _BALANCE_SECTIONS
    if unknown_sections:
        raise ValueError(
            "balance YAML contains unknown section(s): "
            f"{sorted(unknown_sections)}"
        )
    settings = document.get("human_support_intent", {})
    if not isinstance(settings, Mapping):
        raise ValueError("human_support_intent settings must be a mapping")
    allowed = set(HumanSupportIntentConfig.__dataclass_fields__)
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(
            "human_support_intent contains unknown key(s): " f"{sorted(unknown)}"
        )
    return HumanSupportIntentConfig(**dict(settings))
