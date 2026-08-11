from __future__ import annotations

from dataclasses import replace
from math import pi

import mujoco
import numpy as np

from robot_human_interface.pose import make_synthetic_skeleton
from robot_human_interface.retargeting import (
    GeometricRetargeter,
    LEG_DIRECTION_NAMES,
    MujocoIKRetargeter,
    MujocoPoseFidelityEvaluator,
    RetargetingConfig,
    canonicalize_mirrored_skeleton,
)
from robot_human_interface.simulation import HumanoidSimulation
from robot_human_interface.skeleton import JOINT_NAMES, PoseLandmark as L, SkeletonFrame


def _retargeter(*, calibration_frames: int = 0) -> MujocoIKRetargeter:
    return MujocoIKRetargeter(
        config=RetargetingConfig(
            auto_calibration_frames=calibration_frames,
            smoothing_time_constant_s=0.0,
        )
    )


def _site_position(simulation: HumanoidSimulation, name: str) -> np.ndarray:
    identifier = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_SITE,
        name,
    )
    assert identifier >= 0
    return simulation.data.site_xpos[identifier].copy()


def _with_face_direction(frame: SkeletonFrame, direction: np.ndarray) -> SkeletonFrame:
    points = frame.landmarks_3d.copy()
    ear_center = 0.5 * (
        points[int(L.LEFT_EAR)] + points[int(L.RIGHT_EAR)]
    )
    points[int(L.NOSE)] = ear_center + direction / np.linalg.norm(direction) * 0.1
    return replace(frame, landmarks_3d=points)


def _with_right_leg_forward(frame: SkeletonFrame) -> SkeletonFrame:
    points = frame.landmarks_3d.copy()
    hip = points[int(L.RIGHT_HIP)]
    thigh = np.array((0.0, 0.36, -0.18))
    knee = hip + thigh
    ankle = knee + thigh
    points[int(L.RIGHT_KNEE)] = knee
    points[int(L.RIGHT_ANKLE)] = ankle
    points[int(L.RIGHT_HEEL)] = ankle + (0.0, 0.04, 0.06)
    points[int(L.RIGHT_FOOT_INDEX)] = ankle + (0.0, 0.04, -0.18)
    return replace(frame, landmarks_3d=points)


def _with_right_leg_lateral(frame: SkeletonFrame) -> SkeletonFrame:
    points = frame.landmarks_3d.copy()
    hip = points[int(L.RIGHT_HIP)]
    segment = np.asarray((-0.20, 0.35, 0.0))
    knee = hip + segment
    ankle = knee + segment
    points[int(L.RIGHT_KNEE)] = knee
    points[int(L.RIGHT_ANKLE)] = ankle
    points[int(L.RIGHT_HEEL)] = ankle + (0.0, 0.04, 0.06)
    points[int(L.RIGHT_FOOT_INDEX)] = ankle + (0.0, 0.04, -0.18)
    return replace(frame, landmarks_3d=points)


def _with_squat(frame: SkeletonFrame) -> SkeletonFrame:
    points = frame.landmarks_3d.copy()
    for left in (False, True):
        hip_index = L.LEFT_HIP if left else L.RIGHT_HIP
        knee_index = L.LEFT_KNEE if left else L.RIGHT_KNEE
        ankle_index = L.LEFT_ANKLE if left else L.RIGHT_ANKLE
        heel_index = L.LEFT_HEEL if left else L.RIGHT_HEEL
        toe_index = L.LEFT_FOOT_INDEX if left else L.RIGHT_FOOT_INDEX
        hip = points[int(hip_index)]
        side = 1.0 if left else -1.0
        knee = hip + np.asarray((side * 0.12, 0.25, -0.18))
        ankle = hip + np.asarray((0.0, 0.50, 0.0))
        points[int(knee_index)] = knee
        points[int(ankle_index)] = ankle
        points[int(heel_index)] = ankle + (0.0, 0.04, 0.06)
        points[int(toe_index)] = ankle + (0.0, 0.04, -0.18)
    return replace(frame, landmarks_3d=points)


def _with_slow_balance_checkpoint(
    frame: SkeletonFrame, *, left_swing: bool, timestamp_s: float
) -> SkeletonFrame:
    """Make the two depth-heavy one-leg poses used by the replay regression.

    Only the torso and leg landmarks needed by whole-body IK are replaced.
    The values are compact deterministic reductions of slow_balance_demo at
    43.2 s (left swing) and 50.5 s (right swing), so the test needs no video or
    MediaPipe runtime.
    """

    left = (
        (-0.142030, -0.232952, -0.428860),
        (0.091663, -0.042584, -0.044291),
        (0.333857, 0.147404, 0.180194),
        (0.511065, 0.025728, 0.496847),
        (0.547329, 0.020135, 0.522340),
        (0.627172, 0.132330, 0.483445),
    ) if left_swing else (
        (-0.133346, -0.214146, -0.408858),
        (0.087264, -0.018778, -0.057785),
        (-0.084475, 0.352952, -0.122122),
        (-0.158649, 0.610415, -0.115625),
        (-0.164889, 0.644041, -0.120568),
        (-0.271142, 0.678521, -0.179905),
    )
    right = (
        (-0.449062, -0.230541, -0.280850),
        (-0.092676, 0.042120, 0.044964),
        (-0.093806, 0.397847, 0.054447),
        (-0.083296, 0.732277, 0.114742),
        (-0.081068, 0.780867, 0.119604),
        (-0.147480, 0.851790, 0.074692),
    ) if left_swing else (
        (-0.441446, -0.228875, -0.243577),
        (-0.088176, 0.019000, 0.058836),
        (-0.062144, 0.288024, 0.123783),
        (0.141698, 0.408279, 0.304074),
        (0.165321, 0.435030, 0.318294),
        (0.118362, 0.522646, 0.322024),
    )
    left_2d = (
        (0.687306, 0.379463),
        (0.738137, 0.467334),
        (0.782752, 0.502773),
        (0.821724, 0.497066),
        (0.826943, 0.481074),
        (0.840851, 0.551473),
    ) if left_swing else (
        (0.643649, 0.373731),
        (0.695486, 0.472276),
        (0.662796, 0.629720),
        (0.652560, 0.794931),
        (0.649060, 0.791131),
        (0.613430, 0.832553),
    )
    right_2d = (
        (0.615483, 0.392782),
        (0.699898, 0.501851),
        (0.691306, 0.633145),
        (0.700247, 0.773997),
        (0.709583, 0.797132),
        (0.678635, 0.818843),
    ) if left_swing else (
        (0.568076, 0.381982),
        (0.661358, 0.491346),
        (0.656482, 0.613058),
        (0.709635, 0.647049),
        (0.723190, 0.651487),
        (0.701410, 0.701919),
    )
    indices = (
        (L.LEFT_SHOULDER, L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_HEEL, L.LEFT_FOOT_INDEX),
        (L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX),
    )
    points = frame.landmarks_3d.copy()
    points_2d = frame.landmarks_2d.copy()
    for side_indices, values, values_2d in zip(
        indices, (left, right), (left_2d, right_2d), strict=True
    ):
        points[[int(index) for index in side_indices]] = values
        points_2d[[int(index) for index in side_indices]] = values_2d
    return replace(
        frame,
        timestamp_s=timestamp_s,
        landmarks_2d=points_2d,
        landmarks_3d=points,
    )


def _rigid_camera_x_rotation(
    frame: SkeletonFrame, angle_rad: float, *, timestamp_s: float
) -> SkeletonFrame:
    points = frame.landmarks_3d.copy()
    pivot = 0.5 * (
        points[int(L.LEFT_HIP)] + points[int(L.RIGHT_HIP)]
    )
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    rotation = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))
    )
    points = (points - pivot) @ rotation.T + pivot
    return replace(frame, timestamp_s=timestamp_s, landmarks_3d=points)


def test_ik_uses_canonical_motor_order_and_model_bounds() -> None:
    retargeter = MujocoIKRetargeter.from_yaml()
    assert tuple(item.name for item in retargeter.joint_specs) == JOINT_NAMES
    assert retargeter.neutral_positions_rad.shape == (20,)
    assert {
        "right_elbow_ik", "right_wrist_ik", "right_hand_ik",
        "left_elbow_ik", "left_wrist_ik", "left_hand_ik",
        "right_knee_ik", "right_ankle_ik", "right_toe_ik",
        "left_knee_ik", "left_ankle_ik", "left_toe_ik",
        "head_nose_ik",
    }.issubset(retargeter._site_ids)


def test_full_turn_joint_cannot_jump_across_the_pi_branch() -> None:
    retargeter = _retargeter()
    previous = retargeter.neutral_positions_rad
    proposed = previous.copy()
    previous[18] = np.radians(179.0)
    proposed[18] = np.radians(-179.0)

    accepted = retargeter._nearest_bounded_equivalent(proposed, previous)

    assert accepted[18] == retargeter._upper[18]
    assert abs(accepted[18] - previous[18]) <= np.radians(1.01)


def test_calibrated_neutral_pose_is_exact_robot_home() -> None:
    frame = make_synthetic_skeleton(1.0)
    retargeter = _retargeter()
    assert retargeter.calibrate(frame)
    command = retargeter.retarget(frame)
    np.testing.assert_allclose(
        command.positions_rad,
        retargeter.neutral_positions_rad,
        atol=1e-8,
    )
    assert retargeter.last_diagnostics is not None
    assert retargeter.last_diagnostics.marker_count >= 13


def test_ik_forward_arm_raise_moves_matching_hand_toward_robot_front() -> None:
    neutral = make_synthetic_skeleton(1.0)
    raised = make_synthetic_skeleton(1.1, phase_rad=pi / 2.0)
    retargeter = _retargeter()
    assert retargeter.calibrate(neutral)
    command = retargeter.retarget(raised)

    with HumanoidSimulation("fixed") as simulation:
        simulation.reset(retargeter.neutral_positions_rad)
        right_home = _site_position(simulation, "right_hand_ik")
        left_home = _site_position(simulation, "left_hand_ik")
        simulation.reset(command.positions_rad)
        right_delta = _site_position(simulation, "right_hand_ik") - right_home
        left_delta = _site_position(simulation, "left_hand_ik") - left_home

    assert np.dot(right_delta, (-1.0, 0.0, 0.0)) > 0.08
    assert np.linalg.norm(left_delta) < 1e-5


def test_ik_head_turn_moves_robot_nose_to_anatomical_left() -> None:
    neutral = make_synthetic_skeleton(1.0)
    turned = _with_face_direction(neutral, np.array((0.5, 0.0, -1.0)))
    retargeter = _retargeter()
    assert retargeter.calibrate(neutral)
    command = retargeter.retarget(turned)

    with HumanoidSimulation("fixed") as simulation:
        simulation.reset(retargeter.neutral_positions_rad)
        home = _site_position(simulation, "head_nose_ik")
        simulation.reset(command.positions_rad)
        delta = _site_position(simulation, "head_nose_ik") - home

    assert np.dot(delta, (0.0, 1.0, 0.0)) > 0.04


def test_task_missing_from_neutral_window_cannot_lazy_calibrate_while_moving() -> None:
    neutral = make_synthetic_skeleton(1.0)
    visibility = neutral.visibility.copy()
    presence = neutral.presence.copy()
    hidden = np.asarray(
        (
            int(L.NOSE),
            int(L.LEFT_EAR),
            int(L.RIGHT_EAR),
            int(L.RIGHT_INDEX),
            int(L.RIGHT_PINKY),
        )
    )
    visibility[hidden] = 0.0
    presence[hidden] = 0.0
    incomplete_neutral = replace(
        neutral,
        visibility=visibility,
        presence=presence,
    )
    retargeter = _retargeter()
    assert retargeter.calibrate(incomplete_neutral)
    calibrated_tasks = set(retargeter._alignments)
    assert "face" not in calibrated_tasks
    assert "right_hand" not in calibrated_tasks

    first_visible_pose = _with_face_direction(
        make_synthetic_skeleton(2.0, phase_rad=0.9),
        np.asarray((0.5, 0.0, -1.0)),
    )
    command = retargeter.retarget(first_visible_pose)

    assert set(retargeter._alignments) == calibrated_tasks
    np.testing.assert_allclose(
        command.positions_rad[18:20],
        retargeter.neutral_positions_rad[18:20],
        atol=1e-9,
    )

    # A deliberate valid recalibration is the only operation that admits the
    # formerly hidden task; the later head turn then produces a real command.
    assert retargeter.calibrate(neutral)
    assert "face" in retargeter._alignments
    turned = retargeter.retarget(first_visible_pose)
    assert np.linalg.norm(
        turned.positions_rad[18:20]
        - retargeter.neutral_positions_rad[18:20]
    ) > np.radians(5.0)


def test_ik_forward_leg_moves_matching_ankle_toward_robot_front() -> None:
    neutral = make_synthetic_skeleton(1.0)
    moved = _with_right_leg_forward(neutral)
    retargeter = _retargeter()
    assert retargeter.calibrate(neutral)
    command = retargeter.retarget(moved)

    with HumanoidSimulation("fixed") as simulation:
        simulation.reset(retargeter.neutral_positions_rad)
        right_home = _site_position(simulation, "right_ankle_ik")
        left_home = _site_position(simulation, "left_ankle_ik")
        simulation.reset(command.positions_rad)
        right_delta = _site_position(simulation, "right_ankle_ik") - right_home
        left_delta = _site_position(simulation, "left_ankle_ik") - left_home

    assert np.dot(right_delta, (-1.0, 0.0, 0.0)) > 0.20
    assert np.linalg.norm(left_delta) < 1e-5


def test_ik_lateral_leg_improves_full_chain_without_rolling_sole() -> None:
    neutral = make_synthetic_skeleton(1.0)
    moved = _with_right_leg_lateral(neutral)
    geometric = GeometricRetargeter(config=_retargeter().config)
    ik = _retargeter()
    assert geometric.calibrate(neutral)
    assert ik.calibrate(neutral)
    evaluator = MujocoPoseFidelityEvaluator(ik.model)
    tasks = ("right_thigh", "right_shin", "right_foot", "right_leg")
    geometric_error = evaluator.evaluate(
        moved, geometric.retarget(moved).positions_rad
    )
    ik_error = evaluator.evaluate(moved, ik.retarget(moved).positions_rad)
    assert ik_error.mean_error_deg(tasks) < geometric_error.mean_error_deg(tasks) * 0.92
    assert ik_error.error_deg("right_foot") < 5.0


def test_ik_squat_improves_bilateral_leg_fk() -> None:
    neutral = make_synthetic_skeleton(1.0)
    squat = _with_squat(neutral)
    geometric = GeometricRetargeter(config=_retargeter().config)
    ik = _retargeter()
    assert geometric.calibrate(neutral)
    assert ik.calibrate(neutral)
    evaluator = MujocoPoseFidelityEvaluator(ik.model)
    geometric_error = evaluator.evaluate(
        squat, geometric.retarget(squat).positions_rad
    )
    ik_error = evaluator.evaluate(squat, ik.retarget(squat).positions_rad)
    assert (
        ik_error.mean_error_deg(LEG_DIRECTION_NAMES)
        < geometric_error.mean_error_deg(LEG_DIRECTION_NAMES) * 0.60
    )


def test_depth_heavy_balance_pose_keeps_swing_on_anatomical_side() -> None:
    neutral = make_synthetic_skeleton(0.0)
    retargeter = _retargeter(calibration_frames=3)
    for sequence, timestamp_s in enumerate((0.0, 0.03, 0.06)):
        command = retargeter.retarget(
            replace(neutral, timestamp_s=timestamp_s, sequence=sequence)
        )
        np.testing.assert_allclose(
            command.positions_rad, retargeter.neutral_positions_rad
        )

    left_frame = _with_slow_balance_checkpoint(
        neutral, left_swing=True, timestamp_s=1.0
    )
    right_frame = _with_slow_balance_checkpoint(
        neutral, left_swing=False, timestamp_s=2.0
    )
    for frame, expected_side in (
        (left_frame, "left"),
        (right_frame, "right"),
    ):
        directions, confidences = retargeter._observations(frame)
        image_lifts = retargeter._image_leg_lifts(frame)
        assert image_lifts is not None
        other_side = "right" if expected_side == "left" else "left"
        assert image_lifts[expected_side] > 0.15
        assert image_lifts[other_side] == 0.0
        targets = {
            name: target
            for name, target, _ in retargeter._targets(
                directions, confidences, image_lifts
            )
        }
        expected_delta = (
            targets[f"{expected_side}_ankle_ik"][2]
            - retargeter._home_sites[f"{expected_side}_ankle_ik"][2]
        )
        other_delta = (
            targets[f"{other_side}_ankle_ik"][2]
            - retargeter._home_sites[f"{other_side}_ankle_ik"][2]
        )
        assert expected_delta > 0.15
        assert other_delta == 0.0

    left_command = retargeter.retarget(left_frame)
    right_command = retargeter.retarget(right_frame)
    right_indices = np.asarray((6, 8, 10, 12, 14, 16))
    left_indices = np.asarray((7, 9, 11, 13, 15, 17))
    home = retargeter.neutral_positions_rad
    assert np.linalg.norm(
        left_command.positions_rad[left_indices] - home[left_indices]
    ) > np.linalg.norm(
        left_command.positions_rad[right_indices] - home[right_indices]
    )
    assert np.linalg.norm(
        right_command.positions_rad[right_indices] - home[right_indices]
    ) > np.linalg.norm(
        right_command.positions_rad[left_indices] - home[left_indices]
    )

    with HumanoidSimulation("fixed") as simulation:
        simulation.reset(home)
        home_right_z = _site_position(simulation, "right_ankle_ik")[2]
        home_left_z = _site_position(simulation, "left_ankle_ik")[2]
        clearances: list[tuple[float, float]] = []
        for command in (left_command, right_command):
            simulation.reset(command.positions_rad)
            clearances.append(
                (
                    _site_position(simulation, "right_ankle_ik")[2] - home_right_z,
                    _site_position(simulation, "left_ankle_ik")[2] - home_left_z,
                )
            )

    left_pose_right_m, left_pose_left_m = clearances[0]
    right_pose_right_m, right_pose_left_m = clearances[1]
    assert left_pose_left_m > left_pose_right_m + 0.20
    assert right_pose_right_m > right_pose_left_m + 0.08


def test_rigid_torso_rotation_remains_neutral_after_calibration() -> None:
    neutral = make_synthetic_skeleton(0.0)
    retargeter = _retargeter()
    assert retargeter.calibrate(neutral)
    rotated = _rigid_camera_x_rotation(
        neutral, np.radians(30.0), timestamp_s=1.0
    )

    command = retargeter.retarget(rotated)

    np.testing.assert_allclose(
        command.positions_rad,
        retargeter.neutral_positions_rad,
        atol=np.radians(0.25),
    )


def test_ik_auto_calibration_holds_home_for_reference_window() -> None:
    retargeter = _retargeter(calibration_frames=3)
    for sequence, phase in enumerate((0.0, 0.02, -0.02), start=1):
        command = retargeter.retarget(
            make_synthetic_skeleton(float(sequence), phase_rad=phase)
        )
        np.testing.assert_allclose(command.positions_rad, retargeter.neutral_positions_rad)
    assert not retargeter.is_calibrating
    moved = retargeter.retarget(make_synthetic_skeleton(4.0, phase_rad=1.0))
    assert np.linalg.norm(moved.positions_rad - retargeter.neutral_positions_rad) > 0.5


def test_ik_mirror_canonicalization_preserves_motor_command() -> None:
    neutral = make_synthetic_skeleton(1.0)
    moved = make_synthetic_skeleton(1.1, phase_rad=0.9)
    direct = _retargeter()
    reflected = MujocoIKRetargeter(
        config=RetargetingConfig(
            auto_calibration_frames=0,
            smoothing_time_constant_s=0.0,
            mirrored_input=True,
        )
    )
    assert direct.calibrate(neutral)
    assert reflected.calibrate(canonicalize_mirrored_skeleton(neutral))
    expected = direct.retarget(moved)
    actual = reflected.retarget(canonicalize_mirrored_skeleton(moved))
    np.testing.assert_allclose(actual.positions_rad, expected.positions_rad, atol=1e-7)
