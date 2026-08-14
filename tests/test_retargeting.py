from __future__ import annotations

from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_human_interface.pose import make_synthetic_skeleton
from robot_human_interface.retargeting import (
    DEFAULT_JOINT_SPECS,
    GeometricRetargeter,
    JointSpec,
    RetargetingConfig,
    canonicalize_mirrored_skeleton,
    compute_human_joint_angles,
    joint_landmark_validity,
    load_joint_specs,
)
from robot_human_interface.skeleton import JOINT_NAMES, PoseLandmark as L, SkeletonFrame
from robot_human_interface.retargeting.geometry import body_basis


def _mediapipe_camera_skeleton(
    timestamp_s: float,
    *,
    phase_rad: float = 0.0,
) -> SkeletonFrame:
    """Return the fixture already expressed in MediaPipe camera axes."""

    return make_synthetic_skeleton(timestamp_s, phase_rad=phase_rad)


def _with_right_arm_direction(
    frame: SkeletonFrame,
    direction: np.ndarray,
) -> SkeletonFrame:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    points = frame.landmarks_3d.copy()
    shoulder = points[int(L.RIGHT_SHOULDER)]
    elbow = shoulder + 0.25 * direction
    wrist = shoulder + 0.50 * direction
    points[int(L.RIGHT_ELBOW)] = elbow
    points[int(L.RIGHT_WRIST)] = wrist
    points[int(L.RIGHT_PINKY)] = wrist + 0.06 * direction + (0.01, 0.0, 0.0)
    points[int(L.RIGHT_INDEX)] = wrist + 0.07 * direction + (-0.01, 0.0, 0.0)
    points[int(L.RIGHT_THUMB)] = wrist + 0.045 * direction + (-0.025, 0.0, 0.0)
    return replace(frame, landmarks_3d=points)


def _with_face_direction(
    frame: SkeletonFrame,
    direction: np.ndarray,
) -> SkeletonFrame:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    points = frame.landmarks_3d.copy()
    ear_center = 0.5 * (
        points[int(L.LEFT_EAR)] + points[int(L.RIGHT_EAR)]
    )
    points[int(L.NOSE)] = ear_center + 0.10 * direction
    return replace(frame, landmarks_3d=points)


def _with_right_leg_forward(frame: SkeletonFrame, angle_rad: float) -> SkeletonFrame:
    points = frame.landmarks_3d.copy()
    direction = np.array((0.0, np.cos(angle_rad), -np.sin(angle_rad)))
    hip = points[int(L.RIGHT_HIP)]
    knee = hip + 0.40 * direction
    ankle = knee + 0.40 * direction
    points[int(L.RIGHT_KNEE)] = knee
    points[int(L.RIGHT_ANKLE)] = ankle
    points[int(L.RIGHT_HEEL)] = ankle + (0.0, 0.04, 0.06)
    points[int(L.RIGHT_FOOT_INDEX)] = ankle + (0.0, 0.04, -0.18)
    return replace(frame, landmarks_3d=points)


def _with_optional_calibration_channels_occluded(
    frame: SkeletonFrame,
) -> SkeletonFrame:
    visibility = frame.visibility.copy()
    presence = frame.presence.copy()
    optional = (
        L.NOSE,
        L.LEFT_EAR,
        L.RIGHT_EAR,
        L.LEFT_INDEX,
        L.LEFT_PINKY,
        L.RIGHT_INDEX,
        L.RIGHT_PINKY,
        L.LEFT_HEEL,
        L.RIGHT_HEEL,
        L.LEFT_FOOT_INDEX,
        L.RIGHT_FOOT_INDEX,
    )
    indices = [int(landmark) for landmark in optional]
    visibility[indices] = 0.0
    presence[indices] = 0.0
    return replace(frame, visibility=visibility, presence=presence)


def test_shared_joint_yaml_loads_exact_unity_home_pose() -> None:
    path = Path(__file__).parents[1] / "config" / "joints.yaml"
    specs = load_joint_specs(path)
    assert tuple(spec.name for spec in specs) == JOINT_NAMES
    expected_deg = [30, 30, 15, 15, 15, 15, 0, 0, 5, 5, 28, 28, 45, 45, 20, 20, -5, -5, 0, 0]
    np.testing.assert_allclose(np.rad2deg([spec.start_rad for spec in specs]), expected_deg)
    assert [spec.axis for spec in specs] == [spec.axis for spec in DEFAULT_JOINT_SPECS]
    with path.open("r", encoding="utf-8") as stream:
        coordinates = yaml.safe_load(stream)["coordinates"]
    assert coordinates["robot_front_mujoco"] == [-1.0, 0.0, 0.0]
    assert coordinates["robot_left_mujoco"] == [0.0, 1.0, 0.0]
    assert coordinates["robot_up_mujoco"] == [0.0, 0.0, 1.0]


def _minimal_joint_yaml() -> dict[str, object]:
    return {
        "schema_version": 1,
        "joints": [
            {
                "index": spec.index,
                "name": spec.name,
                "axis": spec.axis,
                "limit_rad": [spec.lower_rad, spec.upper_rad],
                "home_rad": spec.start_rad,
                "zero_offset_rad": spec.zero_offset_rad,
                "retarget_sign": spec.retarget_sign,
            }
            for spec in DEFAULT_JOINT_SPECS
        ],
    }


def _write_joint_yaml(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "joints.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_joint_yaml_rejects_unknown_root_and_record_keys(tmp_path: Path) -> None:
    unknown_root = _minimal_joint_yaml()
    unknown_root["schema_verzion"] = 1
    with pytest.raises(ValueError, match=r"top-level.*schema_verzion"):
        load_joint_specs(_write_joint_yaml(tmp_path, unknown_root))

    unknown_record = _minimal_joint_yaml()
    records = unknown_record["joints"]
    assert isinstance(records, list)
    records[0]["home_radians"] = 0.0
    with pytest.raises(ValueError, match=r"record 0.*home_radians"):
        load_joint_specs(_write_joint_yaml(tmp_path, unknown_record))


@pytest.mark.parametrize("bad_index", (True, 0.0, "0"))
def test_joint_yaml_index_is_an_exact_integer(
    tmp_path: Path,
    bad_index: object,
) -> None:
    data = _minimal_joint_yaml()
    records = data["joints"]
    assert isinstance(records, list)
    records[0]["index"] = bad_index

    with pytest.raises(ValueError, match=r"index.*integer.*without coercion"):
        load_joint_specs(_write_joint_yaml(tmp_path, data))


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("limit_rad", [False, 1.0]),
        ("home_rad", float("nan")),
        ("zero_offset_rad", float("inf")),
        ("retarget_sign", True),
        ("axis", [1.0, 0.0, float("-inf")]),
        ("mass_kg", False),
    ),
)
def test_joint_yaml_numeric_fields_are_finite_reals_not_booleans(
    tmp_path: Path,
    field_name: str,
    bad_value: object,
) -> None:
    data = _minimal_joint_yaml()
    records = data["joints"]
    assert isinstance(records, list)
    records[0][field_name] = bad_value

    with pytest.raises(ValueError, match=rf"{field_name}.*(?:finite|real number)"):
        load_joint_specs(_write_joint_yaml(tmp_path, data))


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("index", True),
        ("lower_rad", float("nan")),
        ("upper_rad", float("inf")),
        ("start_rad", False),
        ("zero_offset_rad", "0"),
        ("retarget_sign", True),
    ),
)
def test_joint_spec_constructor_enforces_strict_numeric_types(
    field_name: str,
    bad_value: object,
) -> None:
    values: dict[str, object] = {
        "index": 0,
        "name": "joint",
        "axis": "+X",
        "lower_rad": -1.0,
        "upper_rad": 1.0,
        "start_rad": 0.0,
        "zero_offset_rad": 0.0,
        "retarget_sign": 1.0,
    }
    values[field_name] = bad_value

    with pytest.raises(ValueError):
        JointSpec(**values)


def test_body_basis_front_matches_face_and_not_camera_back() -> None:
    frame = _mediapipe_camera_skeleton(1.0)
    basis = body_basis(frame.landmarks_3d)
    np.testing.assert_allclose(basis.lateral_right, (-1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(basis.vertical_up, (0.0, -1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(basis.forward, (0.0, 0.0, -1.0), atol=1e-12)
    ear_center = 0.5 * (
        frame.landmarks_3d[int(L.LEFT_EAR)]
        + frame.landmarks_3d[int(L.RIGHT_EAR)]
    )
    face = frame.landmarks_3d[int(L.NOSE)] - ear_center
    assert np.dot(face, basis.forward) > 0.0


def test_mirror_canonicalization_is_involutive_and_preserves_anatomy() -> None:
    original = _mediapipe_camera_skeleton(1.0, phase_rad=0.6)
    mirrored = canonicalize_mirrored_skeleton(original)
    # A mirrored MediaPipe observation exchanges anatomical labels as well as
    # reflecting camera X.  The deliberately asymmetric right arm makes a
    # left/right regression observable instead of merely checking symmetry.
    np.testing.assert_allclose(
        mirrored.landmarks_3d[int(L.LEFT_WRIST)],
        original.landmarks_3d[int(L.RIGHT_WRIST)] * (-1.0, 1.0, 1.0),
    )
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


def test_explicit_recalibration_clears_precalibration_stale_fallback() -> None:
    neutral = make_synthetic_skeleton(1.0, phase_rad=0.0)
    moved = make_synthetic_skeleton(2.0, phase_rad=0.9)
    retargeter = GeometricRetargeter(
        config=RetargetingConfig(
            mode="upper_body",
            smoothing_time_constant_s=0.0,
            hold_seconds=0.2,
        )
    )
    assert retargeter.calibrate(neutral)
    old_command = retargeter.retarget(moved)
    assert np.max(
        np.abs(old_command.positions_rad - retargeter.neutral_positions_rad)
    ) > np.radians(5.0)

    recalibration = replace(neutral, timestamp_s=2.05, sequence=2)
    assert retargeter.calibrate(recalibration)
    missing = retargeter.retarget(None, timestamp_s=2.1)

    assert missing.stale
    np.testing.assert_allclose(
        missing.positions_rad,
        retargeter.neutral_positions_rad,
        atol=1e-12,
    )


def test_right_human_arm_drives_right_robot_shoulder_only() -> None:
    neutral = _mediapipe_camera_skeleton(1.0, phase_rad=0.0)
    raised = _mediapipe_camera_skeleton(1.1, phase_rad=0.7)
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


def test_human_forward_raise_moves_robot_hand_toward_physical_front() -> None:
    import mujoco

    from robot_human_interface.simulation import HumanoidSimulation

    neutral = _mediapipe_camera_skeleton(1.0)
    raised = _mediapipe_camera_skeleton(1.1, phase_rad=0.7)
    retargeter = GeometricRetargeter(
        config=RetargetingConfig(
            mode="upper_body",
            auto_calibration_frames=0,
            smoothing_time_constant_s=0.0,
        )
    )
    assert retargeter.calibrate(neutral)
    neutral_command = retargeter.retarget(neutral)
    raised_command = retargeter.retarget(raised)

    with HumanoidSimulation("fixed") as simulation:
        right_wrist = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "wrist_rh",
        )
        left_wrist = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "wrist_lh",
        )
        simulation.reset(neutral_command.positions_rad)
        right_neutral = simulation.data.xpos[right_wrist].copy()
        left_neutral = simulation.data.xpos[left_wrist].copy()
        simulation.reset(raised_command.positions_rad)
        right_delta = simulation.data.xpos[right_wrist] - right_neutral
        left_delta = simulation.data.xpos[left_wrist] - left_neutral

    robot_front = np.array((-1.0, 0.0, 0.0))
    assert np.dot(right_delta, robot_front) > 0.05
    np.testing.assert_allclose(left_delta, 0.0, atol=1e-12)


def test_human_left_head_turn_moves_robot_nose_to_physical_left() -> None:
    import mujoco

    from robot_human_interface.simulation import HumanoidSimulation

    neutral = _mediapipe_camera_skeleton(1.0)
    # Camera +X is anatomical left for a person facing the camera; camera -Z
    # is the person's front.
    turned_left = _with_face_direction(neutral, np.array((0.5, 0.0, -1.0)))
    retargeter = GeometricRetargeter(
        config=RetargetingConfig(
            mode="upper_body",
            auto_calibration_frames=0,
            smoothing_time_constant_s=0.0,
        )
    )
    assert retargeter.calibrate(neutral)
    neutral_command = retargeter.retarget(neutral)
    turned_command = retargeter.retarget(turned_left)
    assert turned_command.positions_rad[18] > neutral_command.positions_rad[18]

    with HumanoidSimulation("fixed") as simulation:
        head = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "head",
        )
        robot_nose_local = np.array((-0.15, 0.0, 0.07))

        def robot_nose() -> np.ndarray:
            return (
                simulation.data.xpos[head]
                + simulation.data.xmat[head].reshape(3, 3) @ robot_nose_local
            )

        simulation.reset(neutral_command.positions_rad)
        nose_neutral = robot_nose().copy()
        simulation.reset(turned_command.positions_rad)
        nose_delta = robot_nose() - nose_neutral

    robot_left = np.array((0.0, 1.0, 0.0))
    assert np.dot(nose_delta, robot_left) > 0.03


def test_human_right_leg_forward_moves_only_robot_right_foot_forward() -> None:
    import mujoco

    from robot_human_interface.simulation import HumanoidSimulation

    neutral = _mediapipe_camera_skeleton(1.0)
    leg_forward = _with_right_leg_forward(neutral, 0.45)
    retargeter = GeometricRetargeter(
        config=RetargetingConfig(
            mode="whole_body",
            auto_calibration_frames=0,
            smoothing_time_constant_s=0.0,
        )
    )
    assert retargeter.calibrate(neutral)
    neutral_command = retargeter.retarget(neutral)
    moved_command = retargeter.retarget(leg_forward)
    assert moved_command.positions_rad[10] > neutral_command.positions_rad[10] + 0.4
    np.testing.assert_allclose(
        moved_command.positions_rad[11],
        neutral_command.positions_rad[11],
        atol=1e-12,
    )

    with HumanoidSimulation("fixed") as simulation:
        right_foot = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "right_foot_contact",
        )
        left_foot = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "left_foot_contact",
        )
        simulation.reset(neutral_command.positions_rad)
        right_neutral = simulation.data.site_xpos[right_foot].copy()
        left_neutral = simulation.data.site_xpos[left_foot].copy()
        simulation.reset(moved_command.positions_rad)
        right_delta = simulation.data.site_xpos[right_foot] - right_neutral
        left_delta = simulation.data.site_xpos[left_foot] - left_neutral

    robot_front = np.array((-1.0, 0.0, 0.0))
    assert np.dot(right_delta, robot_front) > 0.1
    np.testing.assert_allclose(left_delta, 0.0, atol=1e-12)


def test_side_raise_maps_to_shoulder_elevation_without_pi_sign_jump() -> None:
    neutral = _mediapipe_camera_skeleton(1.0)
    # Physical right is camera -X and physical front is camera -Z.  Tiny depth
    # noise on either side of a lateral raise must not reverse the motor.
    almost_side_front = _with_right_arm_direction(neutral, np.array((-1.0, 0.0, -0.02)))
    almost_side_back = _with_right_arm_direction(neutral, np.array((-1.0, 0.0, 0.02)))
    front_angle = compute_human_joint_angles(almost_side_front)[0]
    back_angle = compute_human_joint_angles(almost_side_back)[0]
    assert front_angle > 1.5
    assert back_angle > 1.5
    assert abs(front_angle - back_angle) < 0.01


def test_mirrored_and_unmirrored_input_produce_identical_motor_commands() -> None:
    neutral = _mediapipe_camera_skeleton(1.0)
    raised = _mediapipe_camera_skeleton(1.1, phase_rad=0.7)
    mirrored_neutral = canonicalize_mirrored_skeleton(neutral)
    mirrored_raised = canonicalize_mirrored_skeleton(raised)
    direct = GeometricRetargeter(
        config=RetargetingConfig(
            mode="upper_body",
            auto_calibration_frames=0,
            smoothing_time_constant_s=0.0,
        )
    )
    reflected = GeometricRetargeter(
        config=RetargetingConfig(
            mode="upper_body",
            auto_calibration_frames=0,
            smoothing_time_constant_s=0.0,
            mirrored_input=True,
        )
    )
    assert direct.calibrate(neutral)
    assert reflected.calibrate(mirrored_neutral)
    direct_command = direct.retarget(raised)
    reflected_command = reflected.retarget(mirrored_raised)
    np.testing.assert_allclose(
        reflected_command.positions_rad,
        direct_command.positions_rad,
        atol=1e-12,
    )


def test_auto_calibration_holds_home_for_full_reference_window() -> None:
    retargeter = GeometricRetargeter(
        config=RetargetingConfig(
            mode="upper_body",
            auto_calibration_frames=3,
            smoothing_time_constant_s=0.0,
        )
    )
    home = retargeter.neutral_positions_rad
    for index, phase in enumerate((0.0, 0.05, -0.05), start=1):
        command = retargeter.retarget(
            _mediapipe_camera_skeleton(float(index), phase_rad=phase)
        )
        np.testing.assert_allclose(command.positions_rad, home, atol=1e-12)
        assert not command.stale
    assert not retargeter.is_calibrating
    moved = retargeter.retarget(_mediapipe_camera_skeleton(4.0, phase_rad=0.7))
    assert moved.positions_rad[0] > home[0] + 0.6


def test_auto_calibration_does_not_invent_references_for_occluded_channels() -> None:
    retargeter = GeometricRetargeter(
        config=RetargetingConfig(
            mode="whole_body",
            auto_calibration_frames=3,
            smoothing_time_constant_s=0.0,
        )
    )
    home = retargeter.neutral_positions_rad
    for sequence in range(3):
        frame = make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
        command = retargeter.retarget(
            _with_optional_calibration_channels_occluded(frame)
        )
        np.testing.assert_allclose(command.positions_rad, home, atol=1e-12)

    assert not retargeter.is_calibrating
    unavailable = np.array((4, 5, 6, 7, 14, 15, 18, 19))
    assert not np.any(retargeter._calibration_reference_valid[unavailable])

    # Merely seeing those landmarks later must not interpret their absolute
    # angles relative to an implicit zero reference and jump the robot.
    visible_neutral = make_synthetic_skeleton(0.1, sequence=3)
    command = retargeter.retarget(visible_neutral)
    np.testing.assert_allclose(command.positions_rad, home, atol=1e-12)

    # A deliberate, accepted recalibration is the only operation that can
    # admit the previously unavailable head, hand, and foot channels.
    assert retargeter.calibrate(visible_neutral)
    assert np.all(retargeter._calibration_reference_valid[unavailable])


def test_calibrated_angles_use_shortest_delta_across_pi_branch() -> None:
    retargeter = GeometricRetargeter(
        config=RetargetingConfig(auto_calibration_frames=0)
    )
    raw = np.zeros(len(JOINT_NAMES))
    retargeter._calibration_reference[:] = raw
    retargeter._calibration_reference[18] = np.deg2rad(179.0)
    raw[18] = np.deg2rad(-179.0)
    desired = retargeter._desired(raw)
    np.testing.assert_allclose(
        np.rad2deg(desired[18] - retargeter.neutral_positions_rad[18]),
        2.0,
        atol=1e-12,
    )


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
