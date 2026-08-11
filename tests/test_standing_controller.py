from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_human_interface.control import (
    StandingBalanceConfig,
    StandingBalanceController,
    load_standing_balance_config,
)
from robot_human_interface.simulation import HumanoidSimulation
from robot_human_interface.skeleton import RobotJointCommand


def _state(
    *,
    pitch_rad: float = 0.0,
    pitch_rate_rad_s: float = 0.0,
    right_force_n: float | None = None,
    left_force_n: float | None = None,
) -> SimpleNamespace:
    state = SimpleNamespace(
        base_orientation_wxyz=np.array(
            (np.cos(pitch_rad / 2.0), 0.0, np.sin(pitch_rad / 2.0), 0.0)
        ),
        base_angular_velocity_rad_s=np.array((0.0, pitch_rate_rad_s, 0.0)),
    )
    if right_force_n is not None:
        state.right_foot_normal_force_n = right_force_n
    if left_force_n is not None:
        state.left_foot_normal_force_n = left_force_n
    return state


def test_controller_outputs_only_bounded_canonical_motor_angles() -> None:
    with HumanoidSimulation("free") as simulation:
        controller = StandingBalanceController.from_simulation(
            simulation,
            StandingBalanceConfig(
                upper_body_rate_limit_rad_s=1000.0,
                lower_body_rate_limit_rad_s=1000.0,
            ),
        )
        target = simulation.home_positions_rad.copy()
        target[0] += 0.6
        target[8] += 0.4
        reference = RobotJointCommand.humanoid(1.0, target, 0.9)
        command = controller.update(reference, _state(), dt_s=0.01)

        assert command.joint_names == simulation.joint_names
        assert command.positions_rad.shape == (20,)
        assert np.all(command.positions_rad >= simulation.lower_limits_rad)
        assert np.all(command.positions_rad <= simulation.upper_limits_rad)
        assert command.positions_rad[0] == target[0]
        # A planted hip-roll command would shear the two sole contacts, so
        # transverse imitation waits for a physically unloaded swing leg.
        assert command.positions_rad[8] == simulation.home_positions_rad[8]
        assert command.positions_rad[14] < simulation.home_positions_rad[14]


def test_pitch_feedback_changes_only_motor_targets() -> None:
    with HumanoidSimulation("free") as simulation:
        config = StandingBalanceConfig(
            upper_body_rate_limit_rad_s=1000.0,
            lower_body_rate_limit_rad_s=1000.0,
        )
        neutral = RobotJointCommand.humanoid(1.0, simulation.home_positions_rad, 1.0)
        backward = StandingBalanceController.from_simulation(simulation, config)
        forward = StandingBalanceController.from_simulation(simulation, config)
        backward_command = backward.update(neutral, _state(pitch_rad=-0.1), dt_s=0.01)
        forward_command = forward.update(neutral, _state(pitch_rad=0.1), dt_s=0.01)

        assert backward_command.positions_rad[14] < forward_command.positions_rad[14]
        assert backward_command.positions_rad[15] < forward_command.positions_rad[15]
        np.testing.assert_allclose(
            backward_command.positions_rad[:14], forward_command.positions_rad[:14]
        )
        np.testing.assert_allclose(
            backward_command.positions_rad[16:], forward_command.positions_rad[16:]
        )


def test_unbalanceable_overhead_reference_is_projected_to_feasible_shoulders() -> None:
    with HumanoidSimulation("free") as simulation:
        config = StandingBalanceConfig(
            upper_body_rate_limit_rad_s=1000.0,
            lower_body_rate_limit_rad_s=1000.0,
        )
        controller = StandingBalanceController.from_simulation(simulation, config)
        target = simulation.home_positions_rad.copy()
        target[:2] += np.radians(140.0)
        reference = RobotJointCommand.humanoid(1.0, target, 1.0)

        command = controller.update(reference, _state(), dt_s=0.01)

        np.testing.assert_allclose(
            command.positions_rad[:2],
            simulation.home_positions_rad[:2] + config.max_shoulder_deviation_rad,
        )
        assert controller.last_diagnostics is not None
        assert np.max(np.abs(controller.last_diagnostics.residual_positions_rad[:2])) > 1.0


def test_balance_yaml_is_the_runtime_source_of_controller_gains() -> None:
    config = load_standing_balance_config("config/balance.yaml")
    assert config.enabled
    assert config.lower_body_imitation_scale == 0.3
    assert config.transverse_lower_body_imitation_scale == 0.0
    assert config.swing_leg_imitation_scale == 0.65
    assert config.ankle_pitch_bias_rad == -0.04


def test_continuous_lower_body_reference_is_correlated_and_family_bounded() -> None:
    with HumanoidSimulation("free") as simulation:
        config = StandingBalanceConfig(
            ankle_pitch_bias_rad=0.0,
            upper_body_rate_limit_rad_s=1000.0,
            lower_body_rate_limit_rad_s=1000.0,
        )
        controller = StandingBalanceController.from_simulation(simulation, config)
        target = simulation.home_positions_rad.copy()
        selected = np.asarray((10, 11, 12, 13, 14, 15))
        target[selected] += np.asarray((0.14, 0.14, 0.20, 0.20, 0.06, 0.06))
        transverse = np.asarray((6, 7, 8, 9, 16, 17))
        target[transverse] += 0.30

        command = controller.update(
            RobotJointCommand.humanoid(1.0, target, 1.0),
            _state(),
            dt_s=0.01,
        )

        input_delta = target[selected] - simulation.home_positions_rad[selected]
        output_delta = command.positions_rad[selected] - simulation.home_positions_rad[selected]
        assert np.corrcoef(input_delta, output_delta)[0, 1] > 0.95
        assert np.max(np.abs(output_delta)) > np.radians(3.0)
        np.testing.assert_allclose(
            command.positions_rad[transverse],
            simulation.home_positions_rad[transverse],
            atol=1e-12,
        )
        np.testing.assert_array_less(
            np.abs(output_delta),
            np.asarray(
                (
                    config.max_hip_pitch_deviation_rad,
                    config.max_hip_pitch_deviation_rad,
                    config.max_knee_deviation_rad,
                    config.max_knee_deviation_rad,
                    config.max_ankle_pitch_deviation_rad,
                    config.max_ankle_pitch_deviation_rad,
                )
            )
            + 1e-12,
        )


def test_sole_unload_alone_cannot_apply_the_swing_pose_candidate() -> None:
    with HumanoidSimulation("free") as simulation:
        config = StandingBalanceConfig(
            upper_body_rate_limit_rad_s=1000.0,
            lower_body_rate_limit_rad_s=1000.0,
        )
        controller = StandingBalanceController.from_simulation(simulation, config)
        target = simulation.home_positions_rad.copy()
        target[10:12] += 0.20
        command = controller.update(
            RobotJointCommand.humanoid(1.0, target, 1.0),
            _state(right_force_n=1.0, left_force_n=27.0),
            dt_s=0.01,
        )

        right_delta = command.positions_rad[10] - simulation.home_positions_rad[10]
        left_delta = command.positions_rad[11] - simulation.home_positions_rad[11]
        assert right_delta == pytest.approx(left_delta)
        assert right_delta < config.swing_leg_imitation_scale * 0.20
        pose_candidate = command.pose_reference_positions_rad
        assert pose_candidate[10] - simulation.home_positions_rad[10] == pytest.approx(
            config.swing_leg_imitation_scale * 0.20
        )


def test_deep_bilateral_squat_fades_unsupported_whole_body_pose_to_home() -> None:
    with HumanoidSimulation("free") as simulation:
        config = StandingBalanceConfig(
            ankle_pitch_bias_rad=0.0,
            upper_body_rate_limit_rad_s=1000.0,
            lower_body_rate_limit_rad_s=1000.0,
        )
        controller = StandingBalanceController.from_simulation(simulation, config)
        target = simulation.home_positions_rad.copy()
        target[:2] += 1.0
        target[10:12] += np.radians(60.0)
        target[12:14] += np.radians(40.0)
        reference = RobotJointCommand.humanoid(1.0, target, 1.0)

        command = controller.update(reference, _state(), dt_s=0.01)

        np.testing.assert_allclose(
            command.positions_rad, simulation.home_positions_rad, atol=1e-12
        )
        np.testing.assert_allclose(
            command.pose_reference_positions_rad[6:18],
            simulation.home_positions_rad[6:18],
            atol=1e-12,
        )
        assert controller.last_diagnostics is not None
        assert controller.last_diagnostics.tracking_weight == 0.0


def test_free_base_stands_for_twenty_seconds_using_only_motor_targets() -> None:
    with HumanoidSimulation("free") as simulation:
        controller = StandingBalanceController.from_simulation(
            simulation, load_standing_balance_config("config/balance.yaml")
        )
        reference = RobotJointCommand.humanoid(0.0, simulation.home_positions_rad, 1.0)
        maximum_tilt_rad = 0.0
        initial_base_xy = simulation.get_state().base_position_m[:2].copy()

        for _ in range(10_000):
            state = simulation.get_state()
            command = controller.update(reference, state, dt_s=simulation.model.opt.timestep)
            simulation.apply_joint_command(command)
            state = simulation.step()
            diagnostics = controller.last_diagnostics
            assert diagnostics is not None
            maximum_tilt_rad = max(maximum_tilt_rad, diagnostics.tilt_rad)

        assert state.simulation_time_s == pytest.approx(20.0)
        assert state.base_position_m[2] > 0.85
        assert np.degrees(maximum_tilt_rad) < 10.0
        assert np.linalg.norm(state.base_position_m[:2] - initial_base_xy) < 0.1


def test_slow_arm_motion_is_copied_without_free_base_fall() -> None:
    with HumanoidSimulation("free") as simulation:
        controller = StandingBalanceController.from_simulation(
            simulation, load_standing_balance_config("config/balance.yaml")
        )
        dt_s = float(simulation.model.opt.timestep)
        maximum_tilt_rad = 0.0
        final_reference = simulation.home_positions_rad.copy()

        for step in range(10_000):
            elapsed = step * dt_s
            final_reference = simulation.home_positions_rad.copy()
            elevation = np.radians(70.0) * (
                0.5 + 0.5 * np.sin(2.0 * np.pi * 0.1 * elapsed)
            )
            final_reference[0:2] += elevation
            reference = RobotJointCommand.humanoid(elapsed, final_reference, 1.0)
            state = simulation.get_state()
            command = controller.update(reference, state, dt_s=dt_s)
            simulation.apply_joint_command(command)
            state = simulation.step()
            diagnostics = controller.last_diagnostics
            assert diagnostics is not None
            maximum_tilt_rad = max(maximum_tilt_rad, diagnostics.tilt_rad)

        assert state.base_position_m[2] > 0.85
        assert np.degrees(maximum_tilt_rad) < 12.0
        np.testing.assert_allclose(
            state.joint_positions_rad[:2], final_reference[:2], atol=np.radians(8.0)
        )


def test_bounded_continuous_leg_motion_keeps_free_base_upright() -> None:
    with HumanoidSimulation("free") as simulation:
        controller = StandingBalanceController.from_simulation(
            simulation, load_standing_balance_config("config/balance.yaml")
        )
        dt_s = float(simulation.model.opt.timestep)
        commanded_samples: list[np.ndarray] = []
        reference_samples: list[np.ndarray] = []
        maximum_tilt_rad = 0.0

        for step in range(10_000):
            elapsed = step * dt_s
            phase = np.sin(2.0 * np.pi * 0.08 * elapsed)
            target = simulation.home_positions_rad.copy()
            # A slow symmetric crouch plus small hip/ankle rotations exercises
            # every lower-body family without requesting single support.
            target[10:12] += np.radians(14.0) * phase
            target[12:14] += np.radians(20.0) * phase
            # MuJoCo axes are +Y hip, -Y knee, +Y ankle, so this relation
            # approximately preserves the sole pitch while crouching.
            target[14:16] += np.radians(6.0) * phase
            reference = RobotJointCommand.humanoid(elapsed, target, 1.0)
            state = simulation.get_state()
            command = controller.update(reference, state, dt_s=dt_s)
            simulation.apply_joint_command(command)
            state = simulation.step()
            diagnostics = controller.last_diagnostics
            assert diagnostics is not None
            maximum_tilt_rad = max(maximum_tilt_rad, diagnostics.tilt_rad)
            if step % 20 == 0:
                reference_samples.append(target[6:18] - simulation.home_positions_rad[6:18])
                commanded_samples.append(
                    command.positions_rad[6:18] - simulation.home_positions_rad[6:18]
                )

        reference_trace = np.asarray(reference_samples).reshape(-1)
        command_trace = np.asarray(commanded_samples).reshape(-1)
        assert np.corrcoef(reference_trace, command_trace)[0, 1] > 0.7
        assert np.ptp(command_trace) > np.radians(10.0)
        assert state.base_position_m[2] > 0.82
        assert np.degrees(maximum_tilt_rad) < 18.0
