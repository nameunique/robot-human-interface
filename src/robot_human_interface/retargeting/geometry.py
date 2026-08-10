"""Pure geometric mapping from a 33-point human pose to 20 robot DOFs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from robot_human_interface.skeleton import (
    JOINT_NAMES,
    PoseLandmark as L,
    SkeletonFrame,
    canonicalize_mirrored_skeleton,
)


FloatArray = NDArray[np.float64]


# This is deliberately explicit: it makes the MVP's assumptions reviewable and
# gives the later learned policy a stable baseline to improve upon.
HUMAN_TO_ROBOT_MAPPING: dict[str, str] = {
    "shoulder_rh": "right upper-arm sagittal elevation",
    "shoulder_lh": "left upper-arm sagittal elevation",
    "elbow_rh": "right elbow flexion",
    "elbow_lh": "left elbow flexion",
    "wrist_rh": "right hand/forearm sagittal flexion",
    "wrist_lh": "left hand/forearm sagittal flexion",
    "rotat_axis_rl": "right foot heading (hip-yaw proxy)",
    "rotat_axis_ll": "left foot heading (hip-yaw proxy)",
    "motors_thigh_rl": "right hip abduction",
    "motors_thigh_ll": "left hip abduction",
    "knee_rl": "right hip sagittal flexion",
    "knee_ll": "left hip sagittal flexion",
    "shin_rl": "right knee flexion",
    "shin_ll": "left knee flexion",
    "motors_feet_rl": "right foot pitch (ankle-pitch proxy)",
    "motors_feet_ll": "left foot pitch (ankle-pitch proxy)",
    "foot_rl": "right lower-leg lateral tilt (ankle-roll proxy)",
    "foot_ll": "left lower-leg lateral tilt (ankle-roll proxy)",
    "neck": "head yaw relative to torso",
    "head": "head pitch relative to torso",
}


UPPER_BODY_REQUIRED: tuple[L, ...] = (
    L.LEFT_SHOULDER,
    L.RIGHT_SHOULDER,
    L.LEFT_ELBOW,
    L.RIGHT_ELBOW,
    L.LEFT_WRIST,
    L.RIGHT_WRIST,
    L.LEFT_HIP,
    L.RIGHT_HIP,
)

WHOLE_BODY_REQUIRED: tuple[L, ...] = UPPER_BODY_REQUIRED + (
    L.LEFT_KNEE,
    L.RIGHT_KNEE,
    L.LEFT_ANKLE,
    L.RIGHT_ANKLE,
    L.LEFT_HEEL,
    L.RIGHT_HEEL,
    L.LEFT_FOOT_INDEX,
    L.RIGHT_FOOT_INDEX,
)


def _unit(vector: FloatArray) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.full(3, np.nan)
    return vector / norm


def _angle(first: FloatArray, second: FloatArray) -> float:
    a, b = _unit(first), _unit(second)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _signed_angle(first: FloatArray, second: FloatArray, axis: FloatArray) -> float:
    a, b, n = _unit(first), _unit(second), _unit(axis)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(n).all():
        return float("nan")
    return float(np.arctan2(np.dot(n, np.cross(a, b)), np.dot(a, b)))


@dataclass(frozen=True, slots=True)
class BodyBasis:
    lateral_right: FloatArray
    vertical_up: FloatArray
    forward: FloatArray


def body_basis(points: FloatArray) -> BodyBasis:
    hips = 0.5 * (points[int(L.LEFT_HIP)] + points[int(L.RIGHT_HIP)])
    shoulders = 0.5 * (
        points[int(L.LEFT_SHOULDER)] + points[int(L.RIGHT_SHOULDER)]
    )
    up = _unit(shoulders - hips)
    lateral = points[int(L.RIGHT_HIP)] - points[int(L.LEFT_HIP)]
    lateral = _unit(lateral - up * np.dot(lateral, up))
    forward = _unit(np.cross(lateral, up))
    return BodyBasis(lateral, up, forward)


def _arm_angles(points: FloatArray, basis: BodyBasis, *, left: bool) -> tuple[float, float, float]:
    shoulder = points[int(L.LEFT_SHOULDER if left else L.RIGHT_SHOULDER)]
    elbow = points[int(L.LEFT_ELBOW if left else L.RIGHT_ELBOW)]
    wrist = points[int(L.LEFT_WRIST if left else L.RIGHT_WRIST)]
    index = points[int(L.LEFT_INDEX if left else L.RIGHT_INDEX)]
    pinky = points[int(L.LEFT_PINKY if left else L.RIGHT_PINKY)]
    upper = elbow - shoulder
    forearm = wrist - elbow
    hand = 0.5 * (index + pinky) - wrist
    shoulder_pitch = float(
        np.arctan2(np.dot(upper, basis.forward), np.dot(upper, -basis.vertical_up))
    )
    elbow_flex = _angle(upper, forearm)
    wrist_flex = _signed_angle(forearm, hand, basis.lateral_right)
    return shoulder_pitch, elbow_flex, wrist_flex


def _leg_angles(points: FloatArray, basis: BodyBasis, *, left: bool) -> tuple[float, ...]:
    side = -1.0 if left else 1.0
    hip = points[int(L.LEFT_HIP if left else L.RIGHT_HIP)]
    knee = points[int(L.LEFT_KNEE if left else L.RIGHT_KNEE)]
    ankle = points[int(L.LEFT_ANKLE if left else L.RIGHT_ANKLE)]
    heel = points[int(L.LEFT_HEEL if left else L.RIGHT_HEEL)]
    toe = points[int(L.LEFT_FOOT_INDEX if left else L.RIGHT_FOOT_INDEX)]
    thigh = knee - hip
    shin = ankle - knee
    foot = toe - heel
    hip_yaw = side * float(
        np.arctan2(np.dot(foot, basis.lateral_right), np.dot(foot, basis.forward))
    )
    hip_abduction = side * float(
        np.arctan2(np.dot(thigh, basis.lateral_right), np.dot(thigh, -basis.vertical_up))
    )
    hip_pitch = float(
        np.arctan2(np.dot(thigh, basis.forward), np.dot(thigh, -basis.vertical_up))
    )
    knee_flex = _angle(thigh, shin)
    ankle_pitch = float(
        np.arctan2(np.dot(foot, basis.vertical_up), np.dot(foot, basis.forward))
    )
    ankle_roll = -side * float(
        np.arctan2(np.dot(shin, basis.lateral_right), np.dot(shin, -basis.vertical_up))
    )
    return hip_yaw, hip_abduction, hip_pitch, knee_flex, ankle_pitch, ankle_roll


def _head_angles(points: FloatArray, basis: BodyBasis) -> tuple[float, float]:
    face_center = 0.5 * (points[int(L.LEFT_EAR)] + points[int(L.RIGHT_EAR)])
    face = points[int(L.NOSE)] - face_center
    yaw = float(np.arctan2(np.dot(face, basis.lateral_right), np.dot(face, basis.forward)))
    pitch = float(np.arctan2(np.dot(face, basis.vertical_up), np.dot(face, basis.forward)))
    return yaw, pitch


def compute_human_joint_angles(frame: SkeletonFrame, *, mirrored_input: bool = False) -> FloatArray:
    """Return 20 source angles, in robot joint order and in radians.

    This performs geometry only: no gains, calibration offsets, limits, temporal
    filtering, or stale fallback are applied here.
    """

    if mirrored_input:
        frame = canonicalize_mirrored_skeleton(frame)
    points = frame.landmarks_3d
    basis = body_basis(points)
    right_arm = _arm_angles(points, basis, left=False)
    left_arm = _arm_angles(points, basis, left=True)
    right_leg = _leg_angles(points, basis, left=False)
    left_leg = _leg_angles(points, basis, left=True)
    neck, head = _head_angles(points, basis)
    result = np.array(
        (
            right_arm[0], left_arm[0],
            right_arm[1], left_arm[1],
            right_arm[2], left_arm[2],
            right_leg[0], left_leg[0],
            right_leg[1], left_leg[1],
            right_leg[2], left_leg[2],
            right_leg[3], left_leg[3],
            right_leg[4], left_leg[4],
            right_leg[5], left_leg[5],
            neck, head,
        ),
        dtype=np.float64,
    )
    assert result.shape == (len(JOINT_NAMES),)
    return result


def joint_landmark_validity(
    frame: SkeletonFrame,
    confidence_threshold: float,
    *,
    mirrored_input: bool = False,
) -> NDArray[np.bool_]:
    """Return which output joints have every landmark they depend on.

    Global coverage decides whether a frame is usable at all; this finer mask
    prevents a low-confidence hand, face, or one leg from injecting a random
    angle into an otherwise valid command.
    """

    if mirrored_input:
        frame = canonicalize_mirrored_skeleton(frame)
    valid = frame.valid_mask(confidence_threshold)
    core = (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP)

    def all_valid(*indices: L) -> bool:
        return bool(np.all(valid[[int(index) for index in core + indices]]))

    dependencies = (
        (L.RIGHT_ELBOW,),
        (L.LEFT_ELBOW,),
        (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST),
        (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST),
        (L.RIGHT_ELBOW, L.RIGHT_WRIST, L.RIGHT_INDEX, L.RIGHT_PINKY),
        (L.LEFT_ELBOW, L.LEFT_WRIST, L.LEFT_INDEX, L.LEFT_PINKY),
        (L.RIGHT_HIP, L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX),
        (L.LEFT_HIP, L.LEFT_HEEL, L.LEFT_FOOT_INDEX),
        (L.RIGHT_HIP, L.RIGHT_KNEE),
        (L.LEFT_HIP, L.LEFT_KNEE),
        (L.RIGHT_HIP, L.RIGHT_KNEE),
        (L.LEFT_HIP, L.LEFT_KNEE),
        (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE),
        (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE),
        (L.RIGHT_ANKLE, L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX),
        (L.LEFT_ANKLE, L.LEFT_HEEL, L.LEFT_FOOT_INDEX),
        (L.RIGHT_KNEE, L.RIGHT_ANKLE),
        (L.LEFT_KNEE, L.LEFT_ANKLE),
        (L.NOSE, L.LEFT_EAR, L.RIGHT_EAR),
        (L.NOSE, L.LEFT_EAR, L.RIGHT_EAR),
    )
    return np.array([all_valid(*items) for items in dependencies], dtype=bool)
