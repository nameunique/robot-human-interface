from __future__ import annotations

from math import pi

import numpy as np

from robot_human_interface.pose import make_synthetic_skeleton
from robot_human_interface.retargeting.geometry import body_basis
from robot_human_interface.skeleton import PoseLandmark as L


def test_synthetic_pose_matches_unmirrored_mediapipe_camera_axes() -> None:
    frame = make_synthetic_skeleton(1.0, phase_rad=0.0)
    points = frame.landmarks_3d

    # A camera-facing person's anatomical left is image-right, while their
    # anatomical right is image-left in an unmirrored MediaPipe frame.
    assert points[int(L.LEFT_SHOULDER), 0] > points[int(L.RIGHT_SHOULDER), 0]
    assert points[int(L.LEFT_HIP), 0] > points[int(L.RIGHT_HIP), 0]

    # MediaPipe camera Y grows downward and the face projects toward -Z.
    assert points[int(L.NOSE), 1] < points[int(L.LEFT_HIP), 1]
    assert points[int(L.LEFT_ANKLE), 1] > points[int(L.LEFT_HIP), 1]
    ear_center = 0.5 * (
        points[int(L.LEFT_EAR)] + points[int(L.RIGHT_EAR)]
    )
    face_direction = points[int(L.NOSE)] - ear_center
    assert face_direction[2] < 0.0

    basis = body_basis(points)
    np.testing.assert_allclose(basis.lateral_right, (-1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(basis.vertical_up, (0.0, -1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(basis.forward, (0.0, 0.0, -1.0), atol=1e-12)
    assert np.dot(face_direction, basis.forward) > 0.0

    # The overlay uses the same camera X/Y signs as the world landmarks.
    left_shoulder = int(L.LEFT_SHOULDER)
    right_shoulder = int(L.RIGHT_SHOULDER)
    left_ankle = int(L.LEFT_ANKLE)
    np.testing.assert_allclose(
        frame.landmarks_2d[:, 0],
        np.clip(0.5 + 0.75 * points[:, 0], 0.0, 1.0),
    )
    np.testing.assert_allclose(
        frame.landmarks_2d[:, 1],
        np.clip(0.52 + 0.58 * points[:, 1], 0.0, 1.0),
    )
    assert frame.landmarks_2d[left_shoulder, 0] > frame.landmarks_2d[right_shoulder, 0]
    assert frame.landmarks_2d[left_ankle, 1] > frame.landmarks_2d[left_shoulder, 1]


def test_positive_right_arm_phase_moves_toward_anatomical_forward() -> None:
    neutral = make_synthetic_skeleton(1.0, phase_rad=0.0)
    forward = make_synthetic_skeleton(1.1, phase_rad=pi / 2.0)
    basis = body_basis(neutral.landmarks_3d)

    shoulder_index = int(L.RIGHT_SHOULDER)
    wrist_index = int(L.RIGHT_WRIST)
    neutral_arm = (
        neutral.landmarks_3d[wrist_index]
        - neutral.landmarks_3d[shoulder_index]
    )
    forward_arm = (
        forward.landmarks_3d[wrist_index]
        - forward.landmarks_3d[shoulder_index]
    )

    assert np.dot(neutral_arm, basis.forward) == 0.0
    assert np.dot(forward_arm, basis.forward) > 0.49
    np.testing.assert_allclose(
        forward_arm / np.linalg.norm(forward_arm),
        basis.forward,
        atol=1e-12,
    )
    # The semantic phase is unilateral: it must not move the left arm.
    for landmark in (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST):
        np.testing.assert_allclose(
            forward.landmarks_3d[int(landmark)],
            neutral.landmarks_3d[int(landmark)],
        )
