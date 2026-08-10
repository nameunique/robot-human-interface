"""MediaPipe Tasks Pose Landmarker adapter.

Neither MediaPipe nor OpenCV is imported at module import time. This keeps the
core types and retargeting usable in headless tests and downstream services.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from robot_human_interface.skeleton import (
    LANDMARK_COUNT,
    CameraFrame,
    SkeletonEMAFilter,
    SkeletonFrame,
    canonicalize_mirrored_skeleton,
)


class PoseEstimatorError(RuntimeError):
    pass


class PoseDependencyError(PoseEstimatorError):
    pass


@runtime_checkable
class PoseEstimator(Protocol):
    def estimate(self, frame: CameraFrame) -> SkeletonFrame | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MediaPipePoseConfig:
    model_asset_path: Path
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    num_poses: int = 1
    output_segmentation_masks: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_asset_path", Path(self.model_asset_path))
        for name in (
            "min_pose_detection_confidence",
            "min_pose_presence_confidence",
            "min_tracking_confidence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.num_poses != 1:
            raise ValueError("the teleoperation adapter currently supports exactly one person")


def _landmark_array(landmarks: object, *, world: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequence = list(landmarks)  # type: ignore[arg-type]
    if len(sequence) != LANDMARK_COUNT:
        raise PoseEstimatorError(
            f"MediaPipe returned {len(sequence)} landmarks; expected {LANDMARK_COUNT}"
        )
    points = np.empty((LANDMARK_COUNT, 3), dtype=np.float64)
    visibility = np.ones(LANDMARK_COUNT, dtype=np.float64)
    presence = np.ones(LANDMARK_COUNT, dtype=np.float64)
    for index, landmark in enumerate(sequence):
        points[index] = (float(landmark.x), float(landmark.y), float(landmark.z))
        visibility[index] = float(getattr(landmark, "visibility", 1.0) or 0.0)
        presence[index] = float(getattr(landmark, "presence", 1.0) or 0.0)
    np.clip(visibility, 0.0, 1.0, out=visibility)
    np.clip(presence, 0.0, 1.0, out=presence)
    return points, visibility, presence


class MediaPipePoseLandmarker:
    """Synchronous Tasks API adapter using VIDEO running mode.

    VIDEO mode is intentional: a camera frame is processed synchronously and
    its monotonically increasing timestamp is preserved end-to-end. It avoids
    an extra MediaPipe callback thread while still enabling tracking between
    frames. The caller may run this object in its own perception thread.
    """

    def __init__(
        self,
        config: MediaPipePoseConfig,
        *,
        landmark_filter: SkeletonEMAFilter | None = None,
    ) -> None:
        self.config = config
        self.landmark_filter = landmark_filter
        self._mediapipe = None
        self._landmarker = None
        self._origin_s: float | None = None
        self._last_timestamp_ms = -1

    @property
    def is_open(self) -> bool:
        return self._landmarker is not None

    def open(self) -> "MediaPipePoseLandmarker":
        if self._landmarker is not None:
            return self
        if not self.config.model_asset_path.is_file():
            raise PoseEstimatorError(
                f"MediaPipe pose model does not exist: {self.config.model_asset_path}"
            )
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
        except ImportError as error:
            raise PoseDependencyError(
                "MediaPipe is required for pose estimation; install mediapipe"
            ) from error

        try:
            base_options = mp.tasks.BaseOptions(
                model_asset_path=str(self.config.model_asset_path)
            )
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=self.config.num_poses,
                min_pose_detection_confidence=self.config.min_pose_detection_confidence,
                min_pose_presence_confidence=self.config.min_pose_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                output_segmentation_masks=self.config.output_segmentation_masks,
            )
            landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        except Exception as error:
            raise PoseEstimatorError(f"failed to initialize MediaPipe Pose Landmarker: {error}") from error
        self._mediapipe = mp
        self._landmarker = landmarker
        return self

    def _timestamp_ms(self, timestamp_s: float) -> int:
        if self._origin_s is None:
            self._origin_s = timestamp_s
        candidate = int(round((timestamp_s - self._origin_s) * 1000.0))
        # Tasks VIDEO mode requires strictly increasing integer milliseconds.
        candidate = max(candidate, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = candidate
        return candidate

    def estimate(self, frame: CameraFrame) -> SkeletonFrame | None:
        if self._landmarker is None:
            self.open()
        assert self._landmarker is not None and self._mediapipe is not None
        rgb = np.ascontiguousarray(frame.image_bgr[:, :, ::-1])
        mp_image = self._mediapipe.Image(
            image_format=self._mediapipe.ImageFormat.SRGB,
            data=rgb,
        )
        try:
            result = self._landmarker.detect_for_video(
                mp_image,
                self._timestamp_ms(frame.timestamp_s),
            )
        except Exception as error:
            raise PoseEstimatorError(f"MediaPipe pose inference failed: {error}") from error
        if not result.pose_landmarks:
            return None

        normalized, visibility, presence = _landmark_array(
            result.pose_landmarks[0], world=False
        )
        world_collection = getattr(result, "pose_world_landmarks", None)
        if world_collection:
            world, world_visibility, world_presence = _landmark_array(
                world_collection[0], world=True
            )
            visibility = np.minimum(visibility, world_visibility)
            presence = np.minimum(presence, world_presence)
        else:
            # Keep a usable relative 3-D fallback when a model/runtime omits
            # world landmarks. It is explicitly not metric depth.
            world = normalized.copy()
            hip_center = 0.5 * (world[23] + world[24])
            world -= hip_center

        skeleton = SkeletonFrame(
            timestamp_s=frame.timestamp_s,
            landmarks_2d=normalized[:, :2],
            landmarks_3d=world,
            visibility=visibility,
            presence=presence,
            image_size=(frame.width, frame.height),
            sequence=frame.sequence,
        )
        # When input pixels were explicitly mirrored, MediaPipe observes the
        # reflected person. Restore sensor-space x and anatomical identities so
        # physical human right always drives robot *_rh joints.
        if frame.mirrored:
            skeleton = canonicalize_mirrored_skeleton(skeleton)
        return self.landmark_filter.update(skeleton) if self.landmark_filter else skeleton

    def close(self) -> None:
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            landmarker.close()
        self._mediapipe = None
        self._origin_s = None
        self._last_timestamp_ms = -1
        if self.landmark_filter is not None:
            self.landmark_filter.reset()

    def __enter__(self) -> "MediaPipePoseLandmarker":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()
