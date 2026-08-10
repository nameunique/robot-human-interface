from __future__ import annotations

import mujoco
import numpy as np
import pytest

from robot_human_interface.simulation import HumanoidSimulation, LatestJointCommandBuffer


def _self_contact_pairs(simulation: HumanoidSimulation) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for contact in simulation.data.contact:
        first = mujoco.mj_id2name(
            simulation.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(contact.geom1),
        )
        second = mujoco.mj_id2name(
            simulation.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(contact.geom2),
        )
        if "ground" not in {first, second}:
            pairs.append((first, second))
    return pairs


def test_simulation_reset_step_and_state_contract() -> None:
    with HumanoidSimulation("fixed") as simulation:
        state = simulation.get_state()
        assert len(simulation.joint_names) == 20
        assert state.joint_names == simulation.joint_names
        assert state.joint_positions_rad.shape == (20,)
        assert state.joint_velocities_rad_s.shape == (20,)
        assert state.base_position_m.shape == (3,)
        assert state.base_orientation_wxyz.shape == (4,)
        assert state.actuator_forces.shape == (20,)
        assert state.is_finite

        stepped = simulation.step(25)
        assert stepped.simulation_time_s == pytest.approx(25 * simulation.model.opt.timestep)
        assert stepped.is_finite


def test_fixed_home_has_no_proxy_self_contact_and_tracks_targets() -> None:
    with HumanoidSimulation("fixed") as simulation:
        assert _self_contact_pairs(simulation) == []

        settled = simulation.step(1000)
        assert _self_contact_pairs(simulation) == []
        max_tracking_error = float(
            np.max(np.abs(settled.joint_positions_rad - simulation.home_positions_rad))
        )
        assert max_tracking_error < np.deg2rad(2.0)


def test_targets_are_reordered_and_clamped() -> None:
    with HumanoidSimulation("fixed") as simulation:
        reverse_names = tuple(reversed(simulation.joint_names))
        mapping = {
            name: float(value)
            for name, value in zip(
                reverse_names,
                reversed(simulation.home_positions_rad),
                strict=True,
            )
        }
        accepted = simulation.set_joint_targets(mapping)
        np.testing.assert_allclose(accepted, simulation.home_positions_rad)

        clamped = simulation.set_joint_targets(np.full(20, 100.0))
        np.testing.assert_allclose(clamped, simulation.upper_limits_rad)
        with pytest.raises(ValueError, match="exceed"):
            simulation.set_joint_targets(np.full(20, 100.0), clamp=False)


def test_robot_joint_command_duck_type_is_supported() -> None:
    from robot_human_interface.skeleton import RobotJointCommand

    with HumanoidSimulation("fixed") as simulation:
        command = RobotJointCommand.humanoid(
            timestamp_s=1.0,
            positions_rad=simulation.home_positions_rad,
            confidence=0.9,
        )
        accepted = simulation.apply_joint_command(command)
        np.testing.assert_allclose(accepted, simulation.home_positions_rad)
        state = simulation.step()
        assert state.is_finite


def test_reset_normalizes_base_quaternion() -> None:
    with HumanoidSimulation("free") as simulation:
        state = simulation.reset(
            base_position_m=(0.0, 0.0, 1.0),
            base_orientation_wxyz=(2.0, 0.0, 0.0, 0.0),
        )
        np.testing.assert_allclose(state.base_position_m, (0.0, 0.0, 1.0), atol=1e-12)
        np.testing.assert_allclose(state.base_orientation_wxyz, (1.0, 0.0, 0.0, 0.0))


def test_latest_command_buffer_returns_copies() -> None:
    buffer = LatestJointCommandBuffer(np.zeros(20))
    sequence = buffer.update(np.arange(20, dtype=np.float64))
    snapshot, snapshot_sequence = buffer.snapshot()
    assert sequence == snapshot_sequence == 1
    snapshot[:] = -1.0
    second_snapshot, _ = buffer.snapshot()
    np.testing.assert_allclose(second_snapshot, np.arange(20, dtype=np.float64))


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_step_count_is_rejected(steps: object) -> None:
    with HumanoidSimulation("fixed") as simulation:
        with pytest.raises(ValueError, match="positive integer"):
            simulation.step(steps)  # type: ignore[arg-type]
