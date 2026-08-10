"""Coordinate/identity transforms shared by pose sources and retargeting."""

from __future__ import annotations

from .types import PoseLandmark as L, SkeletonFrame


LEFT_RIGHT_PAIRS: tuple[tuple[L, L], ...] = (
    (L.LEFT_EYE_INNER, L.RIGHT_EYE_INNER),
    (L.LEFT_EYE, L.RIGHT_EYE),
    (L.LEFT_EYE_OUTER, L.RIGHT_EYE_OUTER),
    (L.LEFT_EAR, L.RIGHT_EAR),
    (L.MOUTH_LEFT, L.MOUTH_RIGHT),
    (L.LEFT_SHOULDER, L.RIGHT_SHOULDER),
    (L.LEFT_ELBOW, L.RIGHT_ELBOW),
    (L.LEFT_WRIST, L.RIGHT_WRIST),
    (L.LEFT_PINKY, L.RIGHT_PINKY),
    (L.LEFT_INDEX, L.RIGHT_INDEX),
    (L.LEFT_THUMB, L.RIGHT_THUMB),
    (L.LEFT_HIP, L.RIGHT_HIP),
    (L.LEFT_KNEE, L.RIGHT_KNEE),
    (L.LEFT_ANKLE, L.RIGHT_ANKLE),
    (L.LEFT_HEEL, L.RIGHT_HEEL),
    (L.LEFT_FOOT_INDEX, L.RIGHT_FOOT_INDEX),
)


def canonicalize_mirrored_skeleton(frame: SkeletonFrame) -> SkeletonFrame:
    """Undo pixel reflection and the resulting anatomical left/right swap."""

    points_2d = frame.landmarks_2d.copy()
    points_3d = frame.landmarks_3d.copy()
    visibility = frame.visibility.copy()
    presence = frame.presence.copy()
    points_2d[:, 0] = 1.0 - points_2d[:, 0]
    points_3d[:, 0] *= -1.0
    for left, right in LEFT_RIGHT_PAIRS:
        li, ri = int(left), int(right)
        points_2d[[li, ri]] = points_2d[[ri, li]]
        points_3d[[li, ri]] = points_3d[[ri, li]]
        visibility[[li, ri]] = visibility[[ri, li]]
        presence[[li, ri]] = presence[[ri, li]]
    return SkeletonFrame(
        frame.timestamp_s,
        points_2d,
        points_3d,
        visibility,
        presence,
        frame.image_size,
        frame.sequence,
    )
