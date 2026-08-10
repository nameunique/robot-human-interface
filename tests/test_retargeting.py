from __future__ import annotations

from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np

from robot_human_interface.pose import make_synthetic_skeleton
from robot_human_interface.retargeting import (
    DEFAULT_JOINT_SPECS,
    GeometricRetargeter,
    RetargetingConfig,
    canonicalize_mirrored_skeleton,
    compute_human_joint_angles,
    joint_landmark_validity,
    load_joint_specs,
)
from robot_human_interface.skeleton import JOINT_NAMES, PoseLandmark as L, SkeletonFrame


def test_shared_joint_yaml_loads_exact_unity_home_pose() -> None:
    path = Path(__file__).parents[1] / "config" / "joints.yaml"
    specs = load_joint_specs(path)
    assert tuple(spec.name for spec in specs) == JOINT_NAMES
    expected_deg = [30, 30, 15, 15, 15, 15, 0, 0, 5, 5, 28, 28, 45, 45, 20, 20, -5, -5, 0, 0]
    np.testing.assert_allclose(np.rad2deg([spec.start_rad for spec in specs]), expected_deg)
    assert [spec.axis for spec in specs] == [spec.axis for spec in DEFAULT_JOINT_SPECS]


def test_mirror_canonicalization_is_involutive_and_preserves_anatomy() -> None:
    original = make_synthetic_skeleton(1.0, phase_rad=0.6)
    mirrored = canonicalize_mirrored_skeleton(original)
    restored = canonicalize_mirrored_skeleton(mirrored)
    np.testing.assert_allclose(restored.landmarks_2d, original.landmarks_2d)
    np.testing.assert_allclose(restored.landmarks_3d, original.landmarks_3d)
    np.testing.assert_allclose(
        compute_human_joint_angles(mirrored, mirrored_input=True),
        compute_human_joint_angles(original),
    )


def test_calibration_maps_neutral_human_pose_to_exact_robot_home() -> None:
    frame = make_synthetic_skeleton(1.0, phase_rad=0.0)
    retargeter = GeometricRetargeter(
        config=RetargetingConfig(mode="whole_body", smoothing_time_constant_s=0.0)
    )
    assert retargeter.calibrate(frame)
    command = retargeter.retarget(frame)
    expected = np.array([spec.start_rad for spec in DEFAULT_JOINT_SPECS])
    np.testing.assert_allclose(command.positions_rad, expected, atol=1e-12)
    assert not command.stale


def test_right_human_arm_drives_right_robot_shoulder_only() -> None:
    neutral = make_synthetic_skeleton(1.0, phase_rad=0.0)
    raised = make_synthetic_skeleton(1.1, phase_rad=0.7)
    retargeter = GeometricRetargeter(
        config=RetargetingConfig(mode="upper_body", smoothing_time_constant_s=0.0)
    )
    assert retargeter.calibrate(neutral)
    neutral_command = retargeter.retarget(neutral)
    raised_command = retargeter.retarget(raised)
    assert raised_command.positions_rad[0] > neutral_command.positions_rad[0] + 0.6
    np.testing.assert_allclose(raised_command.positions_rad[1], neutral_command.positions_rad[1])
    # Upper-body mode leaves all leg joints at their verified Unity home pose.
    np.testing.assert_allclose(raised_command.positions_rad[6:18], neutral_command.positions_rad[6:18])


def test_low_confidence_hand_cannot_inject_wrist_angle() -> None:
    frame = make_synthetic_skeleton(1.0, phase_rad=0.4)
    visibility = frame.visibility.copy()
    visibility[[int(L.RIGHT_INDEX), int(L.RIGHT_PINKY)]] = 0.1
    degraded = SkeletonFrame(
        1.1,
        frame.landmarks_2d,
        frame.landmarks_3d,
        visibility,
        frame.presence,
        frame.image_size,
        frame.sequence + 1,
    )
    validity = joint_landmark_validity(degraded, 0.55)
    assert validity[0]
    assert not validity[4]

    retargeter = GeometricRetargeter(
        config=RetargetingConfig(mode="upper_body", smoothing_time_constant_s=0.0)
    )
    first = retargeter.retarget(frame)
    second = retargeter.retarget(degraded)
    assert not second.stale
    assert second.positions_rad[4] == first.positions_rad[4]


def test_missing_skeleton_holds_then_returns_smoothly_to_neutral() -> None:
    neutral_frame = make_synthetic_skeleton(1.0, phase_rad=0.0)
    raised_frame = make_synthetic_skeleton(2.0, phase_rad=0.7)
    retargeter = GeometricRetargeter(
        config=RetargetingConfig(
            mode="upper_body",
            smoothing_time_constant_s=0.0,
            hold_seconds=0.2,
            return_seconds=0.8,
        )
    )
    retargeter.calibrate(neutral_frame)
    valid = retargeter.retarget(raised_frame)
    held = retargeter.retarget(None, timestamp_s=2.1)
    halfway = retargeter.retarget(None, timestamp_s=2.6)
    returned = retargeter.retarget(None, timestamp_s=3.2)
    neutral = retargeter.neutral_positions_rad
    assert held.stale and held.confidence == 0.0
    np.testing.assert_allclose(held.positions_rad, valid.positions_rad)
    np.testing.assert_allclose(halfway.positions_rad, 0.5 * (valid.positions_rad + neutral))
    np.testing.assert_allclose(returned.positions_rad, neutral)


def test_joint_targets_are_always_clamped_to_unity_limits() -> None:
    neutral = make_synthetic_skeleton(1.0, phase_rad=0.0)
    extreme = make_synthetic_skeleton(1.1, phase_rad=pi)
    config = RetargetingConfig(
        mode="whole_body",
        smoothing_time_constant_s=0.0,
        joint_scales={"shoulder_rh": 100.0},
    )
    retargeter = GeometricRetargeter(config=config)
    retargeter.calibrate(neutral)
    command = retargeter.retarget(extreme)
    lower = np.array([spec.lower_rad for spec in DEFAULT_JOINT_SPECS])
    upper = np.array([spec.upper_rad for spec in DEFAULT_JOINT_SPECS])
    assert np.all(command.positions_rad >= lower)
    assert np.all(command.positions_rad <= upper)
    assert command.positions_rad[0] == upper[0]
