"""Pose overlay drawing; OpenCV is optional until this function is called."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from robot_human_interface.skeleton import PoseLandmark, SkeletonFrame


POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


def draw_pose_overlay(
    image_bgr: NDArray[np.uint8],
    skeleton: SkeletonFrame | None,
    *,
    confidence_threshold: float = 0.5,
    copy: bool = True,
) -> NDArray[np.uint8]:
    """Draw confidence-gated MediaPipe connections on a BGR image."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("draw_pose_overlay requires opencv-contrib-python") from error
    output = image_bgr.copy() if copy else image_bgr
    if skeleton is None:
        return output
    height, width = output.shape[:2]
    valid = skeleton.valid_mask(confidence_threshold)
    points = np.nan_to_num(skeleton.landmarks_2d, nan=0.0, posinf=0.0, neginf=0.0)
    pixels = np.rint(points * np.array([width - 1, height - 1])).astype(np.int32)
    for start, end in POSE_CONNECTIONS:
        if valid[start] and valid[end]:
            cv2.line(output, tuple(pixels[start]), tuple(pixels[end]), (80, 220, 80), 2, cv2.LINE_AA)
    for index in np.flatnonzero(valid):
        color = (30, 180, 255) if index in (PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST) else (255, 180, 30)
        cv2.circle(output, tuple(pixels[index]), 3, color, -1, cv2.LINE_AA)
    return output
