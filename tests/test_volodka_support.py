from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_human_interface.control import (
    StandingBalanceController,
    SupportControlConfig,
    SupportIntent,
    SupportPhase,
    SupportStateMachine,
    load_standing_balance_config,
    load_support_control_config,
    support_offsets,
)
from robot_human_interface.simulation import HumanoidSimulation
from robot_human_interface.skeleton import RobotJointCommand


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "volodka"
JOINTS = ROOT / "config" / "joints_volodka.yaml"
BALANCE = ROOT / "config" / "balance_volodka.yaml"


def _simulation() -> HumanoidSimulation:
    return HumanoidSimulation(MODEL / "scene_free.xml", joint_config_path=JOINTS)


def _tilt_rad(quaternion_wxyz: np.ndarray) -> float:
    _, x, y, _ = quaternion_wxyz
    return float(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


def test_volodka_profile_is_explicit_and_legacy_default_is_unchanged() -> None:
    assert SupportControlConfig().motion_profile == "unity_legacy"
    assert load_support_control_config(BALANCE).motion_profile == "volodka_cad"
    assert load_support_control_config(BALANCE).swing_reference_blend == 0.0
    assert SupportControlConfig().swing_reference_blend == 0.25

    legacy_shift, legacy_lift = support_offsets(SupportIntent.RIGHT_SWING)
    explicit_shift, explicit_lift = support_offsets(
        SupportIntent.RIGHT_SWING, "unity_legacy"
    )
    np.testing.assert_array_equal(legacy_shift, explicit_shift)
    np.testing.assert_array_equal(legacy_lift, explicit_lift)

    _, lift = support_offsets(SupportIntent.RIGHT_SWING, "volodka_cad")
    np.testing.assert_allclose(np.degrees(lift[[10, 12, 14]]), (42.5, 85.0, 42.5))
    assert lift[10] - lift[12] + lift[14] == pytest.approx(0.0)

    for invalid in ("unknown", ["volodka_cad"]):
        with pytest.raises(ValueError, match="motion profile"):
            support_offsets(SupportIntent.RIGHT_SWING, invalid)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="motion_profile"):
            SupportControlConfig(motion_profile=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("intent", (SupportIntent.RIGHT_SWING, SupportIntent.LEFT_SWING))
def test_volodka_free_base_lifts_and_returns_each_leg(intent: SupportIntent) -> None:
    with _simulation() as simulation:
        standing = StandingBalanceController.from_simulation(
            simulation, load_standing_balance_config(BALANCE)
        )
        support = SupportStateMachine.from_simulation(
            simulation, load_support_control_config(BALANCE)
        )
        reference = RobotJointCommand.humanoid(
            0.0, simulation.home_positions_rad, 1.0
        )
        dt_s = float(simulation.model.opt.timestep)
        maximum_tilt = 0.0

        def physics_step():
            nonlocal maximum_tilt
            state = simulation.get_state()
            standing_command = standing.update(reference, state, dt_s=dt_s)
            motor_command = support.update(standing_command, state, dt_s=dt_s)
            simulation.apply_joint_command(motor_command)
            stepped = simulation.step()
            assert stepped.non_foot_ground_contact_count == 0
            maximum_tilt = max(maximum_tilt, _tilt_rad(stepped.base_orientation_wxyz))
            return stepped

        for _ in range(1_000):
            state = physics_step()

        support.set_intent(intent)
        for _ in range(3_000):
            state = physics_step()
            if support.phase is SupportPhase.HOLD_SWING:
                break
        assert support.phase is SupportPhase.HOLD_SWING

        for _ in range(500):
            state = physics_step()
        assert support.phase is SupportPhase.HOLD_SWING
        if intent is SupportIntent.RIGHT_SWING:
            swing_height = state.right_foot_position_m[2]
            stance_height = state.left_foot_position_m[2]
            swing_force = state.right_foot_normal_force_n
            stance_force = state.left_foot_normal_force_n
        else:
            swing_height = state.left_foot_position_m[2]
            stance_height = state.right_foot_position_m[2]
            swing_force = state.left_foot_normal_force_n
            stance_force = state.right_foot_normal_force_n
        assert swing_height - stance_height > 0.005
        assert swing_force < 1.0
        assert stance_force > 20.0
        assert np.degrees(maximum_tilt) < 18.0

        support.set_intent(SupportIntent.DOUBLE_SUPPORT)
        for _ in range(5_000):
            state = physics_step()
            if support.phase is SupportPhase.DOUBLE_SUPPORT:
                break
        assert support.phase is SupportPhase.DOUBLE_SUPPORT
        for _ in range(300):
            state = physics_step()

        assert state.base_position_m[2] > 0.19
        assert state.right_foot_normal_force_n > 5.0
        assert state.left_foot_normal_force_n > 5.0
        assert _tilt_rad(state.base_orientation_wxyz) < np.radians(10.0)
