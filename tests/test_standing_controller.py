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


def _state(*, pitch_rad: float = 0.0, pitch_rate_rad_s: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        base_orientation_wxyz=np.array(
            (np.cos(pitch_rad / 2.0), 0.0, np.sin(pitch_rad / 2.0), 0.0)
        ),
        base_angular_velocity_rad_s=np.array((0.0, pitch_rate_rad_s, 0.0)),
    )


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
        # Unsafe direct leg imitation is projected back into double support.
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
    assert config.lower_body_imitation_scale == 0.0
    assert config.ankle_pitch_bias_rad == -0.04


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
