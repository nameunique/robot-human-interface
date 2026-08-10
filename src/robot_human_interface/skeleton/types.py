"""Typed, simulator-independent data exchanged by the perception pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


LANDMARK_COUNT = 33


class PoseLandmark(IntEnum):
    """MediaPipe Pose's stable 33-landmark index convention."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


JOINT_NAMES: tuple[str, ...] = (
    "shoulder_rh",
    "shoulder_lh",
    "elbow_rh",
    "elbow_lh",
    "wrist_rh",
    "wrist_lh",
    "rotat_axis_rl",
    "rotat_axis_ll",
    "motors_thigh_rl",
    "motors_thigh_ll",
    "knee_rl",
    "knee_ll",
    "shin_rl",
    "shin_ll",
    "motors_feet_rl",
    "motors_feet_ll",
    "foot_rl",
    "foot_ll",
    "neck",
    "head",
)


FloatArray = NDArray[np.float64]
ImageArray = NDArray[np.uint8]


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _readonly_array(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
    dtype: np.dtype[np.float64] | type[np.float64] = np.float64,
) -> FloatArray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One BGR image and its capture timestamp on a monotonic clock."""

    image_bgr: ImageArray
    timestamp_s: float
    sequence: int = 0
    mirrored: bool = False

    def __post_init__(self) -> None:
        timestamp_s = _finite_nonnegative(self.timestamp_s, "timestamp_s")
        image = np.asarray(self.image_bgr)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_bgr must have shape (height, width, 3)")
        if image.dtype != np.uint8:
            raise ValueError("image_bgr must use uint8 pixels")
        if int(self.sequence) < 0:
            raise ValueError("sequence must be non-negative")
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "sequence", int(self.sequence))

    @property
    def width(self) -> int:
        return int(self.image_bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_bgr.shape[0])


@dataclass(frozen=True, slots=True)
class SkeletonFrame:
    """A single person in MediaPipe's normalized and hip-relative coordinates.

    ``landmarks_2d`` contains normalized image ``(x, y)``. ``landmarks_3d``
    contains metric-ish world ``(x, y, z)`` coordinates relative to the hips;
    monocular depth is not a physical measurement. Invalid coordinates may be
    NaN and are always excluded by :meth:`valid_mask`.
    """

    timestamp_s: float
    landmarks_2d: FloatArray
    landmarks_3d: FloatArray
    visibility: FloatArray
    presence: FloatArray | None = None
    image_size: tuple[int, int] | None = None
    sequence: int = 0

    def __post_init__(self) -> None:
        timestamp_s = _finite_nonnegative(self.timestamp_s, "timestamp_s")
        points_2d = _readonly_array(
            self.landmarks_2d,
            shape=(LANDMARK_COUNT, 2),
            name="landmarks_2d",
        )
        points_3d = _readonly_array(
            self.landmarks_3d,
            shape=(LANDMARK_COUNT, 3),
            name="landmarks_3d",
        )
        visibility = _readonly_array(
            self.visibility,
            shape=(LANDMARK_COUNT,),
            name="visibility",
        )
        if self.presence is None:
            presence = np.ones(LANDMARK_COUNT, dtype=np.float64)
            presence.setflags(write=False)
        else:
            presence = _readonly_array(
                self.presence,
                shape=(LANDMARK_COUNT,),
                name="presence",
            )
        if np.any((visibility < 0.0) | (visibility > 1.0)):
            raise ValueError("visibility must be within [0, 1]")
        if np.any((presence < 0.0) | (presence > 1.0)):
            raise ValueError("presence must be within [0, 1]")
        if self.image_size is not None:
            width, height = (int(v) for v in self.image_size)
            if width <= 0 or height <= 0:
                raise ValueError("image_size must contain positive width and height")
            object.__setattr__(self, "image_size", (width, height))
        if int(self.sequence) < 0:
            raise ValueError("sequence must be non-negative")
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "landmarks_2d", points_2d)
        object.__setattr__(self, "landmarks_3d", points_3d)
        object.__setattr__(self, "visibility", visibility)
        object.__setattr__(self, "presence", presence)

    def confidence(self) -> FloatArray:
        """Per-landmark confidence combining MediaPipe visibility/presence."""

        result = np.minimum(self.visibility, self.presence)
        result.setflags(write=False)
        return result

    def valid_mask(self, threshold: float = 0.5) -> NDArray[np.bool_]:
        threshold = float(threshold)
        finite = np.isfinite(self.landmarks_2d).all(axis=1)
        finite &= np.isfinite(self.landmarks_3d).all(axis=1)
        return finite & (self.confidence() >= threshold)

    def coverage(self, indices: Iterable[int | PoseLandmark], threshold: float = 0.5) -> float:
        chosen = np.fromiter((int(index) for index in indices), dtype=np.int64)
        if chosen.size == 0:
            return 0.0
        return float(np.mean(self.valid_mask(threshold)[chosen]))

    def mean_confidence(self, indices: Iterable[int | PoseLandmark]) -> float:
        chosen = np.fromiter((int(index) for index in indices), dtype=np.int64)
        if chosen.size == 0:
            return 0.0
        values = self.confidence()[chosen]
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else 0.0


@dataclass(frozen=True, slots=True)
class RobotJointCommand:
    """Ordered joint targets in radians, suitable for sim or a later adapter."""

    timestamp_s: float
    joint_names: tuple[str, ...]
    positions_rad: FloatArray
    confidence: float
    stale: bool = False

    def __post_init__(self) -> None:
        timestamp_s = _finite_nonnegative(self.timestamp_s, "timestamp_s")
        names = tuple(str(name) for name in self.joint_names)
        positions = np.asarray(self.positions_rad, dtype=np.float64)
        if positions.shape != (len(names),):
            raise ValueError("positions_rad must have one value per joint name")
        if not np.isfinite(positions).all():
            raise ValueError("positions_rad must contain only finite values")
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        positions = positions.copy()
        positions.setflags(write=False)
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "positions_rad", positions)
        object.__setattr__(self, "confidence", confidence)

    @classmethod
    def humanoid(
        cls,
        timestamp_s: float,
        positions_rad: Sequence[float],
        confidence: float,
        *,
        stale: bool = False,
    ) -> "RobotJointCommand":
        return cls(timestamp_s, JOINT_NAMES, np.asarray(positions_rad), confidence, stale)
