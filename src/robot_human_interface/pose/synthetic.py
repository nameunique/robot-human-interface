"""Deterministic skeleton generator for headless end-to-end tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import numpy as np

from robot_human_interface.skeleton import LANDMARK_COUNT, CameraFrame, PoseLandmark as L, SkeletonFrame


@dataclass(frozen=True, slots=True)
class SyntheticPoseConfig:
    arm_amplitude_rad: float = 0.75
    frequency_hz: float = 0.25
    confidence: float = 0.99

    def __post_init__(self) -> None:
        if not 0.0 <= self.arm_amplitude_rad <= pi:
            raise ValueError("arm_amplitude_rad must be within [0, pi]")
        if self.frequency_hz < 0.0:
            raise ValueError("frequency_hz must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")


def make_synthetic_skeleton(
    timestamp_s: float,
    *,
    sequence: int = 0,
    image_size: tuple[int, int] = (640, 480),
    phase_rad: float = 0.0,
    confidence: float = 0.99,
) -> SkeletonFrame:
    """Create a camera-facing pose in MediaPipe world-landmark coordinates.

    The axes intentionally match an unmirrored camera frame: positive X points
    to image-right, positive Y points down, and positive Z points away from the
    camera.  Consequently, anatomical right is negative X for this
    camera-facing person and anatomical forward is negative Z.

    ``phase_rad`` rotates the right arm from down (zero) toward anatomical
    forward (``pi / 2``), while keeping it straight.
    """

    points = np.zeros((LANDMARK_COUNT, 3), dtype=np.float64)
    # Face and torso.  The person's anatomical left appears on image-right.
    points[int(L.NOSE)] = (0.0, -0.73, -0.10)
    points[int(L.LEFT_EYE_INNER)] = (0.025, -0.75, -0.075)
    points[int(L.LEFT_EYE)] = (0.045, -0.75, -0.07)
    points[int(L.LEFT_EYE_OUTER)] = (0.065, -0.75, -0.06)
    points[int(L.RIGHT_EYE_INNER)] = (-0.025, -0.75, -0.075)
    points[int(L.RIGHT_EYE)] = (-0.045, -0.75, -0.07)
    points[int(L.RIGHT_EYE_OUTER)] = (-0.065, -0.75, -0.06)
    points[int(L.LEFT_EAR)] = (0.09, -0.72, 0.0)
    points[int(L.RIGHT_EAR)] = (-0.09, -0.72, 0.0)
    points[int(L.MOUTH_LEFT)] = (0.03, -0.68, -0.075)
    points[int(L.MOUTH_RIGHT)] = (-0.03, -0.68, -0.075)
    points[int(L.LEFT_SHOULDER)] = (0.22, -0.52, 0.0)
    points[int(L.RIGHT_SHOULDER)] = (-0.22, -0.52, 0.0)
    points[int(L.LEFT_HIP)] = (0.12, 0.0, 0.0)
    points[int(L.RIGHT_HIP)] = (-0.12, 0.0, 0.0)

    # Left arm remains neutral and straight.
    points[int(L.LEFT_ELBOW)] = (0.23, -0.27, 0.0)
    points[int(L.LEFT_WRIST)] = (0.23, -0.02, 0.0)
    points[int(L.LEFT_PINKY)] = (0.24, 0.04, -0.01)
    points[int(L.LEFT_INDEX)] = (0.22, 0.05, -0.02)
    points[int(L.LEFT_THUMB)] = (0.20, 0.02, -0.03)

    # Right arm rotates from camera-down toward body-forward (-Z).
    angle = float(phase_rad)
    direction = np.array((0.0, cos(angle), -sin(angle)))
    shoulder = points[int(L.RIGHT_SHOULDER)]
    elbow = shoulder + 0.25 * direction
    wrist = elbow + 0.25 * direction
    points[int(L.RIGHT_ELBOW)] = elbow
    points[int(L.RIGHT_WRIST)] = wrist
    points[int(L.RIGHT_PINKY)] = wrist + 0.06 * direction + (0.01, 0.0, 0.0)
    points[int(L.RIGHT_INDEX)] = wrist + 0.07 * direction + (-0.01, 0.0, 0.0)
    points[int(L.RIGHT_THUMB)] = wrist + 0.045 * direction + (-0.025, 0.0, 0.0)

    for left, x in ((True, 0.12), (False, -0.12)):
        knee = L.LEFT_KNEE if left else L.RIGHT_KNEE
        ankle = L.LEFT_ANKLE if left else L.RIGHT_ANKLE
        heel = L.LEFT_HEEL if left else L.RIGHT_HEEL
        toe = L.LEFT_FOOT_INDEX if left else L.RIGHT_FOOT_INDEX
        points[int(knee)] = (x, 0.40, 0.0)
        points[int(ankle)] = (x, 0.80, 0.0)
        points[int(heel)] = (x, 0.84, 0.06)
        points[int(toe)] = (x, 0.84, -0.18)

    width, height = image_size
    points_2d = np.empty((LANDMARK_COUNT, 2), dtype=np.float64)
    points_2d[:, 0] = 0.5 + 0.75 * points[:, 0]
    points_2d[:, 1] = 0.52 + 0.58 * points[:, 1]
    np.clip(points_2d, 0.0, 1.0, out=points_2d)
    scores = np.full(LANDMARK_COUNT, float(confidence))
    return SkeletonFrame(
        timestamp_s=timestamp_s,
        landmarks_2d=points_2d,
        landmarks_3d=points,
        visibility=scores,
        presence=scores,
        image_size=(width, height),
        sequence=sequence,
    )


class SyntheticPoseEstimator:
    """PoseEstimator-compatible sinusoidal skeleton source."""

    def __init__(self, config: SyntheticPoseConfig | None = None) -> None:
        self.config = config or SyntheticPoseConfig()
        self._origin_s: float | None = None
        self.closed = False

    def estimate(self, frame: CameraFrame) -> SkeletonFrame:
        if self.closed:
            raise RuntimeError("SyntheticPoseEstimator is closed")
        if self._origin_s is None:
            self._origin_s = frame.timestamp_s
        elapsed = frame.timestamp_s - self._origin_s
        phase = self.config.arm_amplitude_rad * (
            0.5 + 0.5 * sin(2.0 * pi * self.config.frequency_hz * elapsed)
        )
        return make_synthetic_skeleton(
            frame.timestamp_s,
            sequence=frame.sequence,
            image_size=(frame.width, frame.height),
            phase_rad=phase,
            confidence=self.config.confidence,
        )

    def close(self) -> None:
        self.closed = True
