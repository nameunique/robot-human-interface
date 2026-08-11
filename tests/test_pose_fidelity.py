from __future__ import annotations

from dataclasses import replace
from math import pi

import mujoco
import numpy as np

from robot_human_interface.pose import make_synthetic_skeleton
from robot_human_interface.retargeting import (
    ARM_DIRECTION_NAMES,
    DIRECTION_NAMES,
    END_EFFECTOR_DIRECTION_NAMES,
    AnatomicalDirections,
    GeometricRetargeter,
    MujocoIKRetargeter,
    MujocoPoseFidelityEvaluator,
    RetargetingConfig,
    angular_pose_fidelity,
    human_anatomical_directions,
    robot_anatomical_directions,
)
from robot_human_interface.simulation import HumanoidSimulation
from robot_human_interface.skeleton import PoseLandmark as L, SkeletonFrame


def _direct_config() -> RetargetingConfig:
    return RetargetingConfig(
        mode="whole_body",
        auto_calibration_frames=0,
        smoothing_time_constant_s=0.0,
    )


def _with_right_arm_direction(
    frame: SkeletonFrame,
    direction: np.ndarray,
) -> SkeletonFrame:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    points = frame.landmarks_3d.copy()
    shoulder = points[int(L.RIGHT_SHOULDER)]
    elbow = shoulder + 0.25 * direction
    wrist = elbow + 0.25 * direction
    points[int(L.RIGHT_ELBOW)] = elbow
    points[int(L.RIGHT_WRIST)] = wrist
    points[int(L.RIGHT_PINKY)] = wrist + 0.06 * direction + (0.01, 0.0, 0.0)
    points[int(L.RIGHT_INDEX)] = wrist + 0.07 * direction + (-0.01, 0.0, 0.0)
    points[int(L.RIGHT_THUMB)] = wrist + 0.045 * direction + (-0.025, 0.0, 0.0)
    return replace(frame, landmarks_3d=points)


def _with_face_direction(frame: SkeletonFrame, direction: np.ndarray) -> SkeletonFrame:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    points = frame.landmarks_3d.copy()
    ear_center = 0.5 * (
        points[int(L.LEFT_EAR)] + points[int(L.RIGHT_EAR)]
    )
    points[int(L.NOSE)] = ear_center + 0.10 * direction
    return replace(frame, landmarks_3d=points)


def _with_articulated_right_leg(frame: SkeletonFrame) -> SkeletonFrame:
    points = frame.landmarks_3d.copy()
    hip = points[int(L.RIGHT_HIP)]
    knee = hip + np.asarray((-0.12, 0.20, -0.32))
    ankle = knee + np.asarray((0.12, 0.34, 0.18))
    points[int(L.RIGHT_KNEE)] = knee
    points[int(L.RIGHT_ANKLE)] = ankle
    points[int(L.RIGHT_HEEL)] = ankle + (0.0, 0.04, 0.06)
    points[int(L.RIGHT_FOOT_INDEX)] = ankle + (0.0, 0.04, -0.18)
    return replace(frame, landmarks_3d=points)


def test_human_directions_use_forward_right_up_anatomical_basis() -> None:
    neutral = make_synthetic_skeleton(1.0, phase_rad=0.0)
    raised_forward = make_synthetic_skeleton(1.1, phase_rad=pi / 2.0)

    neutral_directions = human_anatomical_directions(neutral)
    raised_directions = human_anatomical_directions(raised_forward)

    np.testing.assert_allclose(
        neutral_directions.vector("right_arm"),
        (0.0, 0.0, -1.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        raised_directions.vector("right_arm"),
        (1.0, 0.0, 0.0),
        atol=1e-12,
    )
    # The synthetic face is primarily anatomical-forward, with a small upward
    # component inherited from its nose/ear landmarks.
    assert raised_directions.vector("head")[0] > 0.99
    assert raised_directions.vector("head")[2] > 0.0


def test_low_confidence_limb_is_excluded_without_hiding_other_tasks() -> None:
    frame = make_synthetic_skeleton(1.0)
    visibility = frame.visibility.copy()
    visibility[int(L.RIGHT_WRIST)] = 0.1
    degraded = replace(frame, visibility=visibility)

    directions = human_anatomical_directions(degraded, confidence_threshold=0.55)

    assert not directions.is_valid("right_forearm")
    assert not directions.is_valid("right_arm")
    assert directions.is_valid("right_upper_arm")
    assert directions.is_valid("left_arm")
    assert directions.is_valid("head")


def test_robot_directions_are_invariant_to_free_base_pose() -> None:
    with HumanoidSimulation("free") as simulation:
        simulation.reset(simulation.home_positions_rad)
        expected = robot_anatomical_directions(simulation.model, simulation.data)

        base_joint = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free"
        )
        base_qpos = int(simulation.model.jnt_qposadr[base_joint])
        simulation.data.qpos[base_qpos : base_qpos + 3] = (0.4, -0.3, 1.2)
        angle = 0.83
        simulation.data.qpos[base_qpos + 3 : base_qpos + 7] = (
            np.cos(angle / 2.0),
            0.0,
            0.0,
            np.sin(angle / 2.0),
        )
        mujoco.mj_forward(simulation.model, simulation.data)
        transformed = robot_anatomical_directions(simulation.model, simulation.data)

    np.testing.assert_allclose(transformed.vectors, expected.vectors, atol=1e-12)


def test_robot_foot_metric_uses_longitudinal_sole_direction() -> None:
    with HumanoidSimulation("fixed") as simulation:
        evaluator = MujocoPoseFidelityEvaluator(simulation.model)
        directions = evaluator.robot_directions(simulation.home_positions_rad)

    # Both physical toes point toward robot -X, reported as anatomical +forward.
    assert directions.vector("right_foot")[0] > 0.99
    assert directions.vector("left_foot")[0] > 0.99


def test_angular_fidelity_reports_named_errors_and_group_means() -> None:
    vectors = np.tile(np.asarray((1.0, 0.0, 0.0)), (len(DIRECTION_NAMES), 1))
    reference = AnatomicalDirections(vectors, np.ones(len(DIRECTION_NAMES), dtype=bool))
    candidate_vectors = vectors.copy()
    candidate_vectors[0] = (0.0, 1.0, 0.0)
    candidate = AnatomicalDirections(
        candidate_vectors, np.ones(len(DIRECTION_NAMES), dtype=bool)
    )

    fidelity = angular_pose_fidelity(reference, candidate)

    assert fidelity.error_deg("right_upper_arm") == 90.0
    assert fidelity.error_deg("left_upper_arm") == 0.0
    assert fidelity.arm_mean_error_deg == 11.25
    assert fidelity.end_effector_mean_error_deg == 0.0


def test_ik_improves_side_raise_fk_direction_over_scalar_geometric_mapping() -> None:
    """Acceptance gate for the failure visible in the bundled replay.

    A lateral raise is the adversarial case for the old scalar mapper: it maps
    front and side elevation to the same shoulder angle.  The IK must use the
    remaining arm chain to produce a materially closer wrist direction.
    """

    neutral = make_synthetic_skeleton(1.0)
    # Camera -X is anatomical right for the camera-facing fixture.
    side_raise = _with_right_arm_direction(neutral, np.asarray((-1.0, 0.0, 0.0)))
    geometric = GeometricRetargeter(config=_direct_config())
    ik = MujocoIKRetargeter(config=_direct_config())
    assert geometric.calibrate(neutral)
    assert ik.calibrate(neutral)
    geometric_command = geometric.retarget(side_raise)
    ik_command = ik.retarget(side_raise)

    with HumanoidSimulation("fixed") as simulation:
        evaluator = MujocoPoseFidelityEvaluator(simulation.model)
        geometric_error = evaluator.evaluate(side_raise, geometric_command.positions_rad)
        ik_error = evaluator.evaluate(side_raise, ik_command.positions_rad)

    geometric_right_arm = geometric_error.error_deg("right_arm")
    ik_right_arm = ik_error.error_deg("right_arm")
    assert geometric_right_arm > 45.0
    assert ik_right_arm < geometric_right_arm * 0.75
    assert ik_right_arm < 35.0


def test_ik_improves_articulated_leg_fk_over_scalar_geometric_mapping() -> None:
    neutral = make_synthetic_skeleton(1.0)
    articulated = _with_articulated_right_leg(neutral)
    geometric = GeometricRetargeter(config=_direct_config())
    ik = MujocoIKRetargeter(config=_direct_config())
    assert geometric.calibrate(neutral)
    assert ik.calibrate(neutral)

    with HumanoidSimulation("fixed") as simulation:
        evaluator = MujocoPoseFidelityEvaluator(simulation.model)
        geometric_error = evaluator.evaluate(
            articulated, geometric.retarget(articulated).positions_rad
        )
        ik_error = evaluator.evaluate(articulated, ik.retarget(articulated).positions_rad)

    right_leg_tasks = ("right_thigh", "right_shin", "right_foot", "right_leg")
    geometric_mean = geometric_error.mean_error_deg(right_leg_tasks)
    ik_mean = ik_error.mean_error_deg(right_leg_tasks)
    assert ik_mean < geometric_mean * 0.85
    assert ik_error.error_deg("right_leg") < 25.0


def test_ik_head_fk_has_absolute_accuracy_and_does_not_regress_geometric() -> None:
    neutral = make_synthetic_skeleton(1.0)
    turned = _with_face_direction(neutral, np.asarray((0.60, -0.25, -1.0)))
    geometric = GeometricRetargeter(config=_direct_config())
    ik = MujocoIKRetargeter(config=_direct_config())
    assert geometric.calibrate(neutral)
    assert ik.calibrate(neutral)

    with HumanoidSimulation("fixed") as simulation:
        evaluator = MujocoPoseFidelityEvaluator(simulation.model)
        geometric_error = evaluator.evaluate(
            turned, geometric.retarget(turned).positions_rad
        ).error_deg("head")
        ik_error = evaluator.evaluate(
            turned, ik.retarget(turned).positions_rad
        ).error_deg("head")

    assert ik_error <= geometric_error + 2.0
    assert ik_error < 10.0


def test_kinematic_evaluator_does_not_change_live_simulation_state() -> None:
    frame = make_synthetic_skeleton(1.0, phase_rad=0.7)
    with HumanoidSimulation("free") as simulation:
        simulation.step(3)
        qpos_before = simulation.data.qpos.copy()
        qvel_before = simulation.data.qvel.copy()
        time_before = float(simulation.data.time)
        evaluator = MujocoPoseFidelityEvaluator(simulation.model)
        result = evaluator.evaluate(frame, simulation.home_positions_rad)
        np.testing.assert_array_equal(simulation.data.qpos, qpos_before)
        np.testing.assert_array_equal(simulation.data.qvel, qvel_before)
        assert simulation.data.time == time_before

    assert np.isfinite(result.mean_error_deg(ARM_DIRECTION_NAMES))
    assert np.isfinite(result.mean_error_deg(END_EFFECTOR_DIRECTION_NAMES))
