from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import mujoco
import mujoco.viewer
import numpy as np
import pytest

from robot_human_interface.simulation import HumanoidSimulation, LatestJointCommandBuffer


class _FakeViewer:
    def __init__(self) -> None:
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.running = True
        self.lock_count = 0
        self.sync_count = 0

    def is_running(self) -> bool:
        return self.running

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.lock_count += 1
        yield

    def sync(self) -> None:
        self.sync_count += 1

    def close(self) -> None:
        self.running = False


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
        # The legs now carry gravity through bilateral ground contacts instead
        # of hanging from a world weld, so a small compliant deflection is real.
        assert max_tracking_error < np.deg2rad(3.0)


def test_grounded_mode_is_supported_by_both_feet_and_robot_weight() -> None:
    with HumanoidSimulation("fixed") as simulation:
        initial_height = float(simulation.get_state().base_position_m[2])
        settled = simulation.step(200)
        ground_contacts: set[str] = set()
        normal_force = 0.0
        for contact_id, contact in enumerate(simulation.data.contact):
            first = mujoco.mj_id2name(
                simulation.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
            )
            second = mujoco.mj_id2name(
                simulation.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
            )
            if "ground" not in {first, second}:
                continue
            ground_contacts.update({first, second})
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(simulation.model, simulation.data, contact_id, force)
            normal_force += float(force[0])

        assert {"foot_rl_geom", "foot_ll_geom"} <= ground_contacts
        robot_mass = 2.933134
        assert normal_force == pytest.approx(robot_mass * 9.81, rel=0.03)
        assert settled.base_position_m[2] < initial_height


def test_free_mode_initially_transfers_weight_through_both_feet() -> None:
    with HumanoidSimulation("free") as simulation:
        simulation.step(100)
        contacted_feet: set[str] = set()
        normal_force = 0.0
        for contact_id, contact in enumerate(simulation.data.contact):
            names = {
                mujoco.mj_id2name(
                    simulation.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                ),
                mujoco.mj_id2name(
                    simulation.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                ),
            }
            if "ground" in names:
                contacted_feet.update(names)
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(simulation.model, simulation.data, contact_id, force)
                normal_force += float(force[0])

        assert {"foot_rl_geom", "foot_ll_geom"} <= contacted_feet
        assert normal_force == pytest.approx(2.933134 * 9.81, rel=0.05)


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


def test_view_modes_apply_deterministic_layers_under_viewer_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewer = _FakeViewer()
    callbacks: list[object] = []

    def fake_launch_passive(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        key_callback: object,
    ) -> _FakeViewer:
        assert model is not None
        assert data is not None
        callbacks.append(key_callback)
        return viewer

    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_launch_passive)

    with HumanoidSimulation("fixed") as simulation:
        assert simulation.launch_viewer("visual") is viewer
        assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE
        assert viewer.cam.fixedcamid == -1
        assert viewer.cam.distance > 0.0
        assert np.isfinite(viewer.cam.lookat).all()
        assert simulation.viewer_mode == "visual"
        np.testing.assert_array_equal(viewer.opt.geomgroup, (1, 1, 0, 0, 0, 0))
        np.testing.assert_array_equal(viewer.opt.sitegroup, (0, 0, 0, 0, 0, 0))
        assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] == 0
        assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] == 0

        assert simulation.set_view_mode("joints") == "joints"
        np.testing.assert_array_equal(viewer.opt.geomgroup, (1, 0, 0, 0, 0, 0))
        np.testing.assert_array_equal(viewer.opt.jointgroup, (1, 0, 0, 0, 0, 0))
        assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] == 1
        assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] == 1

        assert simulation.toggle_view_mode() == "visual"
        # Camera changes made by MuJoCo's mouse handler must survive syncs;
        # only rendering layers are controlled by the application.
        initial_azimuth = float(viewer.cam.azimuth)
        mujoco.mjv_moveCamera(
            simulation.model,
            mujoco.mjtMouse.mjMOUSE_ROTATE_H,
            0.1,
            0.0,
            viewer.cam,
        )
        moved_azimuth = float(viewer.cam.azimuth)
        assert moved_azimuth != initial_azimuth
        simulation.sync_viewer()
        assert viewer.cam.azimuth == moved_azimuth
        assert viewer.lock_count == 4
        assert viewer.sync_count == 4
        with pytest.raises(ValueError, match="visual.*joints"):
            simulation.set_view_mode("invalid")  # type: ignore[arg-type]

    assert len(callbacks) == 1
    assert not viewer.running


def test_viewer_v_key_defers_toggle_until_simulation_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewer = _FakeViewer()
    callback = None

    def fake_launch_passive(
        _model: mujoco.MjModel,
        _data: mujoco.MjData,
        *,
        key_callback: object,
    ) -> _FakeViewer:
        nonlocal callback
        callback = key_callback
        return viewer

    monkeypatch.setattr(mujoco.viewer, "launch_passive", fake_launch_passive)

    with HumanoidSimulation("fixed") as simulation:
        simulation.launch_viewer()
        assert callable(callback)
        callback(ord("V"))
        assert simulation.viewer_mode == "visual"
        assert viewer.lock_count == 2

        simulation.sync_viewer()
        assert simulation.viewer_mode == "joints"
        assert viewer.lock_count == 3
        assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] == 1
        assert viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] == 1


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
