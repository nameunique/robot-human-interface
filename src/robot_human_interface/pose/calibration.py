"""Robust neutral-pose admission for automatic camera calibration.

Automatic calibration is a safety boundary: a moving pose must not become the
reference merely because it happened to occupy the first N confident frames.
This module keeps a sliding window and admits it only when its normalized body
shape is still and, for whole-body use, both ankles are mutually level.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Sequence

import numpy as np

from robot_human_interface.skeleton import (
    PoseLandmark as L,
    SkeletonFrame,
    canonicalize_mirrored_skeleton,
)


class NeutralCalibrationError(RuntimeError):
    """Raised when automatic calibration cannot find a safe neutral window."""


@dataclass(frozen=True, slots=True)
class NeutralWindowAssessment:
    """Observable robust statistics for one candidate calibration window."""

    pose_spread_ratio: float
    ankle_offset_ratio: float
    ankle_spread_ratio: float
    arm_deviation_rad: float
    upper_arm_deviation_rad: float
    elbow_flexion_rad: float
    knee_flexion_rad: float
    acceptable: bool


class NeutralCalibrationGate:
    """Admit a sliding window only after stillness and neutrality checks.

    ``observe`` is called only for frames which the owning estimator already
    considers confident. A failed full window slides by one frame. Once
    ``max_observations`` is exhausted, a deliberate ``reset`` (normally via
    the UI's recalibration action) is required before observations resume.
    """

    def __init__(
        self,
        *,
        sample_count: int,
        max_observations: int,
        landmark_indices: Sequence[int | L],
        confidence_threshold: float,
        max_pose_spread_ratio: float,
        require_double_support: bool,
        max_ankle_offset_ratio: float,
        max_ankle_spread_ratio: float,
        max_arm_deviation_rad: float,
        max_upper_arm_deviation_rad: float,
        max_elbow_flexion_rad: float,
        max_knee_flexion_rad: float,
        require_extended_legs: bool,
        mirrored_input: bool = False,
        label: str = "neutral-pose",
    ) -> None:
        self._landmark_indices = np.asarray(
            [int(index) for index in landmark_indices], dtype=np.int64
        )
        if self._landmark_indices.size == 0:
            raise ValueError("neutral calibration needs at least one landmark")
        self.confidence_threshold = float(confidence_threshold)
        self.max_pose_spread_ratio = float(max_pose_spread_ratio)
        self.require_double_support = bool(require_double_support)
        self.max_ankle_offset_ratio = float(max_ankle_offset_ratio)
        self.max_ankle_spread_ratio = float(max_ankle_spread_ratio)
        self.max_arm_deviation_rad = float(max_arm_deviation_rad)
        self.max_upper_arm_deviation_rad = float(max_upper_arm_deviation_rad)
        self.max_elbow_flexion_rad = float(max_elbow_flexion_rad)
        self.max_knee_flexion_rad = float(max_knee_flexion_rad)
        self.require_extended_legs = bool(require_extended_legs)
        self.mirrored_input = bool(mirrored_input)
        self.label = str(label)
        self.reset(sample_count=sample_count, max_observations=max_observations)

    def reset(self, *, sample_count: int, max_observations: int) -> None:
        if isinstance(sample_count, bool) or int(sample_count) != sample_count:
            raise ValueError("calibration sample_count must be an integer")
        if isinstance(max_observations, bool) or int(max_observations) != max_observations:
            raise ValueError("calibration max_observations must be an integer")
        if sample_count <= 0:
            raise ValueError("calibration sample_count must be positive")
        if max_observations < sample_count:
            raise ValueError("calibration max_observations must cover sample_count")
        self.sample_count = int(sample_count)
        self.max_observations = int(max_observations)
        self._observations = 0
        self._frames: list[SkeletonFrame] = []
        self._source_frames: list[SkeletonFrame] = []
        self._failed = False
        self._last_assessment: NeutralWindowAssessment | None = None

    @property
    def observations(self) -> int:
        return self._observations

    @property
    def progress(self) -> float:
        # 100% means accepted to callers. A rejected full window therefore
        # remains visibly in progress instead of claiming completion.
        return min(0.99, len(self._frames) / self.sample_count)

    @property
    def last_assessment(self) -> NeutralWindowAssessment | None:
        return self._last_assessment

    def _canonical_confident(self, frame: SkeletonFrame) -> SkeletonFrame | None:
        canonical = (
            canonicalize_mirrored_skeleton(frame)
            if self.mirrored_input
            else frame
        )
        required = [
            *self._landmark_indices,
            int(L.LEFT_SHOULDER),
            int(L.RIGHT_SHOULDER),
            int(L.LEFT_HIP),
            int(L.RIGHT_HIP),
            int(L.LEFT_WRIST),
            int(L.RIGHT_WRIST),
            int(L.LEFT_ELBOW),
            int(L.RIGHT_ELBOW),
        ]
        if self.require_double_support or self.require_extended_legs:
            required.extend(
                (
                    int(L.LEFT_KNEE),
                    int(L.RIGHT_KNEE),
                    int(L.LEFT_ANKLE),
                    int(L.RIGHT_ANKLE),
                )
            )
        required_indices = np.unique(np.asarray(required, dtype=np.int64))
        # ``max_observations`` deliberately counts only complete, confident
        # gate inputs. Camera occlusion keeps the robot at home but cannot
        # consume the operator's stillness timeout.
        if not bool(
            np.all(
                canonical.valid_mask(self.confidence_threshold)[required_indices]
            )
        ):
            return None
        return canonical

    def accepts_explicit(self, frame: SkeletonFrame) -> bool:
        """Check one deliberately selected frame without a stillness window.

        Single-frame dispersion is necessarily zero, but visibility,
        double-support and anatomical posture criteria remain mandatory.
        This method does not consume or reset automatic-calibration state.
        """

        canonical = self._canonical_confident(frame)
        if canonical is None:
            return False
        assessment = assess_neutral_window(
            (canonical,),
            landmark_indices=self._landmark_indices,
            confidence_threshold=self.confidence_threshold,
            max_pose_spread_ratio=self.max_pose_spread_ratio,
            require_double_support=self.require_double_support,
            max_ankle_offset_ratio=self.max_ankle_offset_ratio,
            max_ankle_spread_ratio=self.max_ankle_spread_ratio,
            max_arm_deviation_rad=self.max_arm_deviation_rad,
            max_upper_arm_deviation_rad=self.max_upper_arm_deviation_rad,
            max_elbow_flexion_rad=self.max_elbow_flexion_rad,
            max_knee_flexion_rad=self.max_knee_flexion_rad,
            require_extended_legs=self.require_extended_legs,
        )
        self._last_assessment = assessment
        return assessment.acceptable

    def observe(self, frame: SkeletonFrame) -> tuple[SkeletonFrame, ...] | None:
        if self._failed:
            raise NeutralCalibrationError(self._failure_message())
        canonical = self._canonical_confident(frame)
        if canonical is None:
            return None
        self._observations += 1
        self._frames.append(canonical)
        self._source_frames.append(frame)
        if len(self._frames) > self.sample_count:
            del self._frames[0]
            del self._source_frames[0]
        if len(self._frames) < self.sample_count:
            return None

        self._last_assessment = assess_neutral_window(
            self._frames,
            landmark_indices=self._landmark_indices,
            confidence_threshold=self.confidence_threshold,
            max_pose_spread_ratio=self.max_pose_spread_ratio,
            require_double_support=self.require_double_support,
            max_ankle_offset_ratio=self.max_ankle_offset_ratio,
            max_ankle_spread_ratio=self.max_ankle_spread_ratio,
            max_arm_deviation_rad=self.max_arm_deviation_rad,
            max_upper_arm_deviation_rad=self.max_upper_arm_deviation_rad,
            max_elbow_flexion_rad=self.max_elbow_flexion_rad,
            max_knee_flexion_rad=self.max_knee_flexion_rad,
            require_extended_legs=self.require_extended_legs,
        )
        if self._last_assessment.acceptable:
            # Owners receive the exact source-domain frames admitted by the
            # gate. Mirrored inputs must not be canonicalized a second time by
            # their existing observation functions.
            return tuple(self._source_frames)
        if self._observations >= self.max_observations:
            self._failed = True
            raise NeutralCalibrationError(self._failure_message())
        return None

    def _failure_message(self) -> str:
        details = "no complete candidate window was available"
        if self._last_assessment is not None:
            assessment = self._last_assessment
            details = (
                f"pose spread={assessment.pose_spread_ratio:.4f} "
                f"(limit {self.max_pose_spread_ratio:.4f}), "
                f"ankle offset={assessment.ankle_offset_ratio:.4f} "
                f"(limit {self.max_ankle_offset_ratio:.4f}), "
                f"ankle spread={assessment.ankle_spread_ratio:.4f} "
                f"(limit {self.max_ankle_spread_ratio:.4f}), "
                f"arm deviation={assessment.arm_deviation_rad:.4f} rad "
                f"(limit {self.max_arm_deviation_rad:.4f}), "
                f"upper-arm deviation={assessment.upper_arm_deviation_rad:.4f} rad "
                f"(limit {self.max_upper_arm_deviation_rad:.4f}), "
                f"elbow flexion={assessment.elbow_flexion_rad:.4f} rad "
                f"(limit {self.max_elbow_flexion_rad:.4f}), "
                f"knee flexion={assessment.knee_flexion_rad:.4f} rad "
                f"(limit {self.max_knee_flexion_rad:.4f})"
            )
        return (
            f"{self.label} calibration failed after {self._observations} confident "
            f"observations: {details}. Hold a still, two-foot neutral pose and "
            "explicitly restart calibration."
        )


def _normalized_pose(
    frame: SkeletonFrame,
    landmark_indices: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray | None:
    required = np.unique(
        np.concatenate(
            (
                landmark_indices,
                np.asarray(
                    [
                        int(L.LEFT_SHOULDER),
                        int(L.RIGHT_SHOULDER),
                        int(L.LEFT_HIP),
                        int(L.RIGHT_HIP),
                    ],
                    dtype=np.int64,
                ),
            )
        )
    )
    if not bool(np.all(frame.valid_mask(confidence_threshold)[required])):
        return None
    points = frame.landmarks_3d
    hip_center = 0.5 * (
        points[int(L.LEFT_HIP)] + points[int(L.RIGHT_HIP)]
    )
    torso_scale = 0.5 * (
        np.linalg.norm(points[int(L.LEFT_SHOULDER)] - points[int(L.LEFT_HIP)])
        + np.linalg.norm(points[int(L.RIGHT_SHOULDER)] - points[int(L.RIGHT_HIP)])
    )
    if not np.isfinite(torso_scale) or torso_scale < 1e-5:
        return None
    normalized = (points[landmark_indices] - hip_center) / float(torso_scale)
    return normalized if np.isfinite(normalized).all() else None


def _ankle_height_ratio(
    frame: SkeletonFrame,
    confidence_threshold: float,
) -> float | None:
    indices = np.asarray(
        [
            int(L.LEFT_SHOULDER),
            int(L.RIGHT_SHOULDER),
            int(L.LEFT_HIP),
            int(L.RIGHT_HIP),
            int(L.LEFT_KNEE),
            int(L.RIGHT_KNEE),
            int(L.LEFT_ANKLE),
            int(L.RIGHT_ANKLE),
        ],
        dtype=np.int64,
    )
    if not bool(np.all(frame.valid_mask(confidence_threshold)[indices])):
        return None
    points = frame.landmarks_2d
    shoulder_center = 0.5 * (
        points[int(L.LEFT_SHOULDER)] + points[int(L.RIGHT_SHOULDER)]
    )
    hip_center = 0.5 * (
        points[int(L.LEFT_HIP)] + points[int(L.RIGHT_HIP)]
    )
    up = shoulder_center - hip_center
    up_norm = float(np.linalg.norm(up))
    if not np.isfinite(up_norm) or up_norm < 1e-6:
        return None
    up /= up_norm
    left_leg = (
        np.linalg.norm(points[int(L.LEFT_KNEE)] - points[int(L.LEFT_HIP)])
        + np.linalg.norm(points[int(L.LEFT_ANKLE)] - points[int(L.LEFT_KNEE)])
    )
    right_leg = (
        np.linalg.norm(points[int(L.RIGHT_KNEE)] - points[int(L.RIGHT_HIP)])
        + np.linalg.norm(points[int(L.RIGHT_ANKLE)] - points[int(L.RIGHT_KNEE)])
    )
    scale = 0.5 * float(left_leg + right_leg)
    if not np.isfinite(scale) or scale < 1e-6:
        return None
    ratio = float(
        np.dot(
            points[int(L.RIGHT_ANKLE)] - points[int(L.LEFT_ANKLE)],
            up,
        )
        / scale
    )
    return ratio if np.isfinite(ratio) else None


def _angle(first: np.ndarray, second: np.ndarray) -> float | None:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if (
        not np.isfinite((first_norm, second_norm)).all()
        or first_norm < 1e-6
        or second_norm < 1e-6
    ):
        return None
    cosine = float(
        np.clip(np.dot(first, second) / (first_norm * second_norm), -1.0, 1.0)
    )
    return float(np.arccos(cosine))


def _posture_angles(
    frame: SkeletonFrame, *, require_extended_legs: bool
) -> tuple[float, float, float, float] | None:
    """Return worst arm-down deviation and knee flexion for one frame.

    All comparisons are between anatomical vectors in the same camera frame,
    so a rigid camera/person tilt does not change either angle.
    """

    points = frame.landmarks_3d
    shoulder_center = 0.5 * (
        points[int(L.LEFT_SHOULDER)] + points[int(L.RIGHT_SHOULDER)]
    )
    hip_center = 0.5 * (
        points[int(L.LEFT_HIP)] + points[int(L.RIGHT_HIP)]
    )
    torso_down = hip_center - shoulder_center
    arm_deviations: list[float] = []
    upper_arm_deviations: list[float] = []
    elbow_flexions: list[float] = []
    knee_flexions: list[float] = []
    for left in (False, True):
        shoulder = int(L.LEFT_SHOULDER if left else L.RIGHT_SHOULDER)
        elbow = int(L.LEFT_ELBOW if left else L.RIGHT_ELBOW)
        wrist = int(L.LEFT_WRIST if left else L.RIGHT_WRIST)
        hip = int(L.LEFT_HIP if left else L.RIGHT_HIP)
        knee = int(L.LEFT_KNEE if left else L.RIGHT_KNEE)
        ankle = int(L.LEFT_ANKLE if left else L.RIGHT_ANKLE)
        arm_angle = _angle(points[wrist] - points[shoulder], torso_down)
        upper_arm_angle = _angle(points[elbow] - points[shoulder], torso_down)
        elbow_angle = _angle(
            points[shoulder] - points[elbow], points[wrist] - points[elbow]
        )
        knee_angle = (
            _angle(points[hip] - points[knee], points[ankle] - points[knee])
            if require_extended_legs
            else pi
        )
        if (
            arm_angle is None
            or upper_arm_angle is None
            or elbow_angle is None
            or knee_angle is None
        ):
            return None
        arm_deviations.append(arm_angle)
        upper_arm_deviations.append(upper_arm_angle)
        elbow_flexions.append(max(0.0, pi - elbow_angle))
        knee_flexions.append(max(0.0, pi - knee_angle))
    return (
        max(arm_deviations),
        max(upper_arm_deviations),
        max(elbow_flexions),
        max(knee_flexions),
    )


def assess_neutral_window(
    frames: Sequence[SkeletonFrame],
    *,
    landmark_indices: Sequence[int | L] | np.ndarray,
    confidence_threshold: float,
    max_pose_spread_ratio: float,
    require_double_support: bool,
    max_ankle_offset_ratio: float,
    max_ankle_spread_ratio: float,
    max_arm_deviation_rad: float,
    max_upper_arm_deviation_rad: float,
    max_elbow_flexion_rad: float,
    max_knee_flexion_rad: float,
    require_extended_legs: bool,
) -> NeutralWindowAssessment:
    """Measure robust dispersion about the median normalized body shape."""

    indices = np.asarray([int(index) for index in landmark_indices], dtype=np.int64)
    poses = [
        _normalized_pose(frame, indices, confidence_threshold) for frame in frames
    ]
    if not poses or any(pose is None for pose in poses):
        return NeutralWindowAssessment(
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            False,
        )
    stack = np.stack(poses)  # type: ignore[arg-type]
    median_pose = np.median(stack, axis=0)
    deviations = np.linalg.norm(stack - median_pose, axis=2)
    pose_spread = float(np.quantile(deviations, 0.95))

    ankle_offset = 0.0
    ankle_spread = 0.0
    if require_double_support:
        ratios = [
            _ankle_height_ratio(frame, confidence_threshold) for frame in frames
        ]
        if any(ratio is None for ratio in ratios):
            return NeutralWindowAssessment(
                pose_spread,
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                float("inf"),
                False,
            )
        ratio_values = np.asarray(ratios, dtype=np.float64)
        median_ratio = float(np.median(ratio_values))
        ankle_offset = abs(median_ratio)
        ankle_spread = float(
            np.quantile(np.abs(ratio_values - median_ratio), 0.95)
        )

    posture = [
        _posture_angles(frame, require_extended_legs=require_extended_legs)
        for frame in frames
    ]
    if any(value is None for value in posture):
        return NeutralWindowAssessment(
            pose_spread,
            ankle_offset,
            ankle_spread,
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            False,
        )
    posture_values = np.asarray(posture, dtype=np.float64)
    arm_deviation = float(np.quantile(posture_values[:, 0], 0.95))
    upper_arm_deviation = float(np.quantile(posture_values[:, 1], 0.95))
    elbow_flexion = float(np.quantile(posture_values[:, 2], 0.95))
    knee_flexion = float(np.quantile(posture_values[:, 3], 0.95))

    finite = np.isfinite(
        (
            pose_spread,
            ankle_offset,
            ankle_spread,
            arm_deviation,
            upper_arm_deviation,
            elbow_flexion,
            knee_flexion,
        )
    ).all()
    acceptable = bool(
        finite
        and pose_spread <= max_pose_spread_ratio
        and (
            not require_double_support
            or (
                ankle_offset <= max_ankle_offset_ratio
                and ankle_spread <= max_ankle_spread_ratio
            )
        )
        and arm_deviation <= max_arm_deviation_rad
        and upper_arm_deviation <= max_upper_arm_deviation_rad
        and elbow_flexion <= max_elbow_flexion_rad
        and (not require_extended_legs or knee_flexion <= max_knee_flexion_rad)
    )
    return NeutralWindowAssessment(
        pose_spread,
        ankle_offset,
        ankle_spread,
        arm_deviation,
        upper_arm_deviation,
        elbow_flexion,
        knee_flexion,
        acceptable,
    )
