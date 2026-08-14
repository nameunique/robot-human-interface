from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_human_interface.app.teleop import _stale_support_force_return_reason
from robot_human_interface.control import (
    StandingBalanceController,
    SupportControlConfig,
    SupportIntent,
    SupportIntentLatch,
    SupportPhase,
    SupportStateMachine,
    load_standing_balance_config,
    load_support_control_config,
    support_offsets,
)
from robot_human_interface.control.standing import BalancedJointCommand
from robot_human_interface.retargeting import DEFAULT_JOINT_SPECS
from robot_human_interface.simulation import HumanoidSimulation
from robot_human_interface.skeleton import RobotJointCommand


HOME = np.asarray([spec.start_rad for spec in DEFAULT_JOINT_SPECS])
LOWER = np.asarray([spec.lower_rad for spec in DEFAULT_JOINT_SPECS])
UPPER = np.asarray([spec.upper_rad for spec in DEFAULT_JOINT_SPECS])


def _loads(
    *,
    right_n: float,
    left_n: float,
    tilt_rad: float = 0.0,
    angular_speed_rad_s: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        right_foot_normal_force_n=right_n,
        left_foot_normal_force_n=left_n,
        base_orientation_wxyz=np.asarray(
            (np.cos(0.5 * tilt_rad), 0.0, np.sin(0.5 * tilt_rad), 0.0)
        ),
        base_angular_velocity_rad_s=np.asarray(
            (angular_speed_rad_s, 0.0, 0.0)
        ),
    )


def _reference(*, stale: bool = False) -> RobotJointCommand:
    return RobotJointCommand.humanoid(0.0, HOME, 1.0, stale=stale)


def _fast_config() -> SupportControlConfig:
    return SupportControlConfig(
        shift_duration_s=0.20,
        load_confirm_duration_s=0.05,
        stance_load_timeout_s=0.20,
        lift_duration_s=0.15,
        minimum_hold_duration_s=0.0,
        lower_duration_s=0.15,
        touchdown_confirm_duration_s=0.05,
        touchdown_preload_duration_s=0.20,
        touchdown_timeout_s=0.40,
        center_duration_s=0.20,
        support_loss_grace_s=0.03,
        min_stance_force_n=10.0,
        min_stance_load_fraction=0.65,
        max_swing_load_fraction=0.35,
        min_touchdown_force_n=2.0,
        min_touchdown_contact_force_n=1.0,
        min_touchdown_total_force_n=10.0,
        upper_body_rate_limit_rad_s=1000.0,
        lower_body_rate_limit_rad_s=1000.0,
    )


def _admit_from_double_support(
    machine: SupportStateMachine,
    intent: SupportIntent,
    *,
    reference: RobotJointCommand | BalancedJointCommand | None = None,
) -> RobotJointCommand:
    """Admit a one-leg cycle from measured, balanced two-foot contact."""

    machine.set_intent(intent)
    command = machine.update(
        _reference() if reference is None else reference,
        _loads(right_n=14.0, left_n=14.0),
        dt_s=0.01,
    )
    assert machine.phase is SupportPhase.SHIFT_WEIGHT
    return command


def _reach_touchdown_fault(
    machine: SupportStateMachine,
    *,
    failed_intent: SupportIntent = SupportIntent.RIGHT_SWING,
) -> tuple[SimpleNamespace, RobotJointCommand]:
    no_touchdown = (
        _loads(right_n=1.0, left_n=27.0)
        if failed_intent is SupportIntent.RIGHT_SWING
        else _loads(right_n=27.0, left_n=1.0)
    )
    _admit_from_double_support(machine, failed_intent)
    for _ in range(100):
        machine.update(_reference(), no_touchdown, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING

    machine.set_intent(SupportIntent.DOUBLE_SUPPORT)
    command = _reference()
    for _ in range(200):
        command = machine.update(_reference(), no_touchdown, dt_s=0.01)
        diagnostics = machine.last_diagnostics
        if (
            machine.phase is SupportPhase.CENTER_WEIGHT
            and diagnostics is not None
            and diagnostics.abort_reason == "touchdown_timeout"
        ):
            break
    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "touchdown_timeout"
    return no_touchdown, command


@pytest.mark.parametrize(
    ("intent", "state", "shift_values_deg", "lift_values_deg"),
    (
        (
            SupportIntent.RIGHT_SWING,
            _loads(right_n=1.0, left_n=27.0),
            {8: 25.0, 9: -25.0, 16: -25.0, 17: 25.0},
            {10: 20.0, 12: 20.0, 14: -20.0},
        ),
        (
            SupportIntent.LEFT_SWING,
            _loads(right_n=27.0, left_n=1.0),
            {8: -28.0, 9: 28.0, 16: 28.0, 17: -28.0},
            {11: 20.0, 13: 30.0, 15: -10.0},
        ),
    ),
)
def test_verified_offsets_are_force_gated_and_ramped_in_phase_order(
    intent: SupportIntent,
    state: SimpleNamespace,
    shift_values_deg: dict[int, float],
    lift_values_deg: dict[int, float],
) -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    command = _admit_from_double_support(machine, intent)
    phases: list[SupportPhase] = [SupportPhase.SHIFT_WEIGHT]

    for _ in range(100):
        command = machine.update(_reference(), state, dt_s=0.01)
        if not phases or phases[-1] is not machine.phase:
            phases.append(machine.phase)
        if machine.phase is SupportPhase.HOLD_SWING:
            break

    assert machine.phase is SupportPhase.HOLD_SWING
    assert phases == [
        SupportPhase.SHIFT_WEIGHT,
        SupportPhase.VERIFY_STANCE,
        SupportPhase.LIFT_SWING,
        SupportPhase.HOLD_SWING,
    ]
    expected_shift, expected_lift = support_offsets(intent)
    expected = np.clip(HOME + expected_shift + expected_lift, LOWER, UPPER)
    np.testing.assert_allclose(command.positions_rad, expected, atol=1e-12)
    for index, value in shift_values_deg.items():
        assert np.degrees(expected_shift[index]) == pytest.approx(value)
    for index, value in lift_values_deg.items():
        assert np.degrees(expected_lift[index]) == pytest.approx(value)


def test_minimum_hold_dwell_precedes_normal_camera_release() -> None:
    config = replace(_fast_config(), minimum_hold_duration_s=0.05)
    machine = SupportStateMachine(LOWER, UPPER, config)
    loaded = _loads(right_n=1.0, left_n=27.0)
    _admit_from_double_support(machine, SupportIntent.RIGHT_SWING)
    for _ in range(100):
        machine.update(_reference(), loaded, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING

    machine.set_intent(SupportIntent.DOUBLE_SUPPORT)
    for _ in range(4):
        machine.update(_reference(), loaded, dt_s=0.01)
        assert machine.phase is SupportPhase.HOLD_SWING
    machine.update(_reference(), loaded, dt_s=0.01)
    assert machine.phase is SupportPhase.LOWER_SWING


def test_leg_never_lifts_without_confirmed_stance_load_and_failure_latches() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    machine.set_intent(SupportIntent.RIGHT_SWING)
    evenly_loaded = _loads(right_n=14.0, left_n=14.0)
    observed_phases: set[SupportPhase] = set()

    for _ in range(100):
        command = machine.update(_reference(), evenly_loaded, dt_s=0.01)
        observed_phases.add(machine.phase)

    assert SupportPhase.LIFT_SWING not in observed_phases
    assert SupportPhase.HOLD_SWING not in observed_phases
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.blocked_intent is SupportIntent.RIGHT_SWING
    assert machine.last_diagnostics.abort_reason == "stance_load_timeout"
    assert machine.last_diagnostics.swing_progress == 0.0
    np.testing.assert_allclose(command.positions_rad, HOME, atol=1e-12)

    # Repeating the failed intent cannot immediately restart the lift.  An
    # explicit double-support acknowledgement clears the latch.
    for _ in range(10):
        machine.update(_reference(), evenly_loaded, dt_s=0.01)
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    machine.set_intent(SupportIntent.DOUBLE_SUPPORT)
    assert machine.last_diagnostics.blocked_intent is SupportIntent.RIGHT_SWING
    machine.update(_reference(), evenly_loaded, dt_s=0.01)
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.blocked_intent is None


def test_stance_load_loss_lowers_swing_before_recentering() -> None:
    config = _fast_config()
    machine = SupportStateMachine(LOWER, UPPER, config)
    loaded = _loads(right_n=27.0, left_n=1.0)
    lost = _loads(right_n=2.0, left_n=2.0)
    _admit_from_double_support(machine, SupportIntent.LEFT_SWING)

    for _ in range(100):
        raised = machine.update(_reference(), loaded, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING
    raised_offset = raised.positions_rad - HOME

    for _ in range(10):
        lowering = machine.update(_reference(), lost, dt_s=0.01)
        if machine.phase is SupportPhase.LOWER_SWING:
            break
    assert machine.phase is SupportPhase.LOWER_SWING
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "stance_load_lost"
    assert machine.last_diagnostics.blocked_intent is SupportIntent.LEFT_SWING
    # Failure does not discontinuously remove the raised-leg target.
    assert np.max(np.abs((lowering.positions_rad - HOME) - raised_offset)) < np.radians(2.0)

    phases: list[SupportPhase] = []
    two_feet = _loads(right_n=14.0, left_n=14.0)
    for _ in range(150):
        returned = machine.update(_reference(), two_feet, dt_s=0.01)
        if not phases or phases[-1] is not machine.phase:
            phases.append(machine.phase)
        if machine.phase is SupportPhase.DOUBLE_SUPPORT:
            break
    assert SupportPhase.CENTER_WEIGHT in phases
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    np.testing.assert_allclose(returned.positions_rad, HOME, atol=1e-12)


def test_late_lowering_contact_centers_without_driving_profile_to_zero() -> None:
    config = replace(_fast_config(), lower_duration_s=0.50)
    machine = SupportStateMachine(LOWER, UPPER, config)
    loaded = _loads(right_n=1.0, left_n=27.0)
    _admit_from_double_support(machine, SupportIntent.RIGHT_SWING)
    for _ in range(100):
        machine.update(_reference(), loaded, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING

    machine.set_intent(SupportIntent.DOUBLE_SUPPORT)
    machine.update(_reference(), loaded, dt_s=0.01)
    assert machine.phase is SupportPhase.LOWER_SWING
    while (
        machine.phase is SupportPhase.LOWER_SWING
        and machine.last_diagnostics is not None
        and machine.last_diagnostics.swing_progress
        > config.early_touchdown_max_swing_progress
    ):
        machine.update(_reference(), loaded, dt_s=0.01)
    assert machine.phase is SupportPhase.LOWER_SWING

    two_feet = _loads(right_n=14.0, left_n=14.0)
    for _ in range(20):
        machine.update(_reference(), two_feet, dt_s=0.01)
        if machine.phase is SupportPhase.CENTER_WEIGHT:
            break

    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    assert 0.0 < machine.last_diagnostics.swing_progress <= (
        config.early_touchdown_max_swing_progress
    )


def test_touchdown_timeout_holds_known_stance_until_contact_and_rearm() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    no_touchdown, timed_out = _reach_touchdown_fault(machine)

    diagnostics = machine.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.blocked_intent is SupportIntent.RIGHT_SWING
    assert diagnostics.swing_progress == 0.0
    # VERIFY_TOUCHDOWN may not unload the only sole with measured contact.
    assert diagnostics.shift_progress == pytest.approx(1.0)
    expected_shift, _ = support_offsets(SupportIntent.RIGHT_SWING)
    np.testing.assert_allclose(
        timed_out.positions_rad,
        np.clip(HOME + expected_shift, LOWER, UPPER),
        atol=1e-12,
    )

    # Time alone cannot leave the fault hold, erase the support shift, or
    # admit an opposite lift.
    for _ in range(100):
        held = machine.update(
            _reference(),
            no_touchdown,
            dt_s=0.01,
            intent=SupportIntent.LEFT_SWING,
        )
    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.shift_progress == pytest.approx(1.0)
    np.testing.assert_allclose(held.positions_rad, timed_out.positions_rad, atol=1e-12)

    # A delayed physical touchdown permits centering, but an unacknowledged
    # fault still blocks every new lift after DOUBLE_SUPPORT is reached.
    two_feet = _loads(right_n=14.0, left_n=14.0)
    for _ in range(100):
        machine.update(
            _reference(),
            two_feet,
            dt_s=0.01,
            intent=SupportIntent.LEFT_SWING,
        )
        if machine.phase is SupportPhase.DOUBLE_SUPPORT:
            break
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.blocked_intent is SupportIntent.RIGHT_SWING

    for _ in range(10):
        machine.update(
            _reference(),
            two_feet,
            dt_s=0.01,
            intent=SupportIntent.LEFT_SWING,
        )
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT

    machine.update(
        _reference(),
        two_feet,
        dt_s=0.01,
        intent=SupportIntent.DOUBLE_SUPPORT,
    )
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.blocked_intent is None
    machine.update(
        _reference(),
        two_feet,
        dt_s=0.01,
        intent=SupportIntent.LEFT_SWING,
    )
    assert machine.phase is SupportPhase.SHIFT_WEIGHT


def test_centering_reverifies_contact_before_removing_more_weight_shift() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    loaded = _loads(right_n=1.0, left_n=27.0)
    _admit_from_double_support(machine, SupportIntent.RIGHT_SWING)
    for _ in range(100):
        machine.update(_reference(), loaded, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING

    machine.set_intent(SupportIntent.DOUBLE_SUPPORT)
    two_feet = _loads(right_n=14.0, left_n=14.0)
    for _ in range(200):
        machine.update(_reference(), two_feet, dt_s=0.01)
        if machine.phase is SupportPhase.CENTER_WEIGHT:
            break
    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    confirmed_shift = machine.last_diagnostics.shift_progress

    # Losing the returning sole on the very next tick must not remove any
    # additional lateral support. CENTER remains externally visible while the
    # state machine waits for load to return.
    contact_lost = _loads(right_n=0.5, left_n=27.5)
    machine.update(_reference(), contact_lost, dt_s=0.01)
    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.shift_progress == pytest.approx(confirmed_shift)

    # A loaded tick may progress, but another loss must freeze precisely at the
    # new value rather than continuing to unload the known stance.
    machine.update(_reference(), two_feet, dt_s=0.01)
    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    progressed_shift = machine.last_diagnostics.shift_progress
    assert progressed_shift < confirmed_shift
    machine.update(_reference(), contact_lost, dt_s=0.01)
    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.shift_progress == pytest.approx(progressed_shift)

    machine.update(_reference(), two_feet, dt_s=0.01)
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.shift_progress < progressed_shift


def test_touchdown_fault_recovery_freezes_again_if_contact_is_lost() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    no_touchdown, _ = _reach_touchdown_fault(machine)
    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    held_shift = machine.last_diagnostics.shift_progress

    two_feet = _loads(right_n=14.0, left_n=14.0)
    for _ in range(20):
        machine.update(_reference(), two_feet, dt_s=0.01)
        if machine.last_diagnostics is not None and (
            machine.last_diagnostics.shift_progress < held_shift
        ):
            break
    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    progressed_shift = machine.last_diagnostics.shift_progress
    assert progressed_shift < held_shift

    contact_lost = _loads(right_n=0.5, left_n=27.5)
    machine.update(_reference(), contact_lost, dt_s=0.01)
    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.shift_progress == pytest.approx(progressed_shift)


def test_contact_loss_fault_keeps_lowering_residual_lift_without_centering() -> None:
    config = replace(
        _fast_config(),
        lower_duration_s=0.50,
        touchdown_timeout_s=0.08,
    )
    machine = SupportStateMachine(LOWER, UPPER, config)
    one_foot = _loads(right_n=1.0, left_n=27.0)
    _admit_from_double_support(machine, SupportIntent.RIGHT_SWING)
    for _ in range(100):
        machine.update(_reference(), one_foot, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING

    machine.set_intent(SupportIntent.DOUBLE_SUPPORT)
    two_feet = _loads(right_n=14.0, left_n=14.0)
    for _ in range(100):
        machine.update(_reference(), two_feet, dt_s=0.01)
        if machine.phase is SupportPhase.CENTER_WEIGHT:
            break
    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    initial_shift = machine.last_diagnostics.shift_progress
    initial_swing = machine.last_diagnostics.swing_progress
    assert initial_swing > 0.0

    observed_swing = [initial_swing]
    contact_lost = _loads(right_n=0.5, left_n=27.5)
    for _ in range(100):
        machine.update(_reference(), contact_lost, dt_s=0.01)
        assert machine.last_diagnostics is not None
        observed_swing.append(machine.last_diagnostics.swing_progress)
        if observed_swing[-1] <= 0.0:
            break

    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "touchdown_timeout"
    assert machine.last_diagnostics.shift_progress == pytest.approx(initial_shift)
    assert all(
        current <= previous + 1e-12
        for previous, current in zip(observed_swing, observed_swing[1:])
    )
    assert machine.last_diagnostics.swing_progress == pytest.approx(0.0)
    _, lift_offset = support_offsets(SupportIntent.RIGHT_SWING)
    lift_indices = np.flatnonzero(np.abs(lift_offset) > 0.0)
    assert np.max(
        np.abs(machine.last_diagnostics.applied_offset_rad[lift_indices])
    ) < np.radians(1.0)

def test_touchdown_timeout_is_seen_by_latch_and_requires_fresh_camera_edge() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    latch = SupportIntentLatch()
    no_touchdown = _loads(right_n=1.0, left_n=27.0)
    timestamp_s = 0.0

    def step(observed: SupportIntent, state: SimpleNamespace) -> None:
        nonlocal timestamp_s
        diagnostics = machine.last_diagnostics
        aborting = bool(
            diagnostics is not None
            and diagnostics.abort_reason is not None
            and machine.phase in {SupportPhase.LOWER_SWING, SupportPhase.CENTER_WEIGHT}
        )
        request = latch.update(
            observed,
            machine.phase,
            aborted=aborting,
            timestamp_s=timestamp_s,
        )
        machine.update(_reference(), state, dt_s=0.01, intent=request)
        timestamp_s += 0.01

    step(
        SupportIntent.RIGHT_SWING,
        _loads(right_n=14.0, left_n=14.0),
    )
    for _ in range(100):
        step(SupportIntent.RIGHT_SWING, no_touchdown)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING

    for _ in range(200):
        step(SupportIntent.DOUBLE_SUPPORT, no_touchdown)
        if (
            machine.phase is SupportPhase.CENTER_WEIGHT
            and machine.last_diagnostics is not None
            and machine.last_diagnostics.abort_reason == "touchdown_timeout"
        ):
            break
    # The next controller tick observes CENTER_WEIGHT + abort_reason, exactly
    # the existing teleop contract, and hard-cancels the latch/pending queue.
    step(SupportIntent.RIGHT_SWING, no_touchdown)
    assert latch.blocked_intent is SupportIntent.RIGHT_SWING
    assert latch.pending_intent is None
    assert latch.intent is SupportIntent.DOUBLE_SUPPORT
    assert machine.phase is SupportPhase.CENTER_WEIGHT

    two_feet = _loads(right_n=14.0, left_n=14.0)
    for _ in range(100):
        step(SupportIntent.RIGHT_SWING, two_feet)
        if machine.phase is SupportPhase.DOUBLE_SUPPORT:
            break
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    # A continuously held failed gesture remains blocked after recovery.
    for _ in range(10):
        step(SupportIntent.RIGHT_SWING, two_feet)
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    assert latch.blocked_intent is SupportIntent.RIGHT_SWING

    step(SupportIntent.DOUBLE_SUPPORT, two_feet)
    assert latch.blocked_intent is None
    step(SupportIntent.RIGHT_SWING, two_feet)
    assert machine.phase is SupportPhase.SHIFT_WEIGHT


def test_reset_explicitly_clears_touchdown_fault_and_latch_state() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    _reach_touchdown_fault(machine)
    latch = SupportIntentLatch()
    latch.update(SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT)
    latch.update(
        SupportIntent.RIGHT_SWING,
        SupportPhase.CENTER_WEIGHT,
        aborted=True,
    )
    assert latch.blocked_intent is SupportIntent.RIGHT_SWING

    machine.reset()
    latch.reset()

    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    assert machine.intent is SupportIntent.DOUBLE_SUPPORT
    assert machine.active_intent is SupportIntent.DOUBLE_SUPPORT
    assert machine.last_diagnostics is None
    assert latch.blocked_intent is None
    command = machine.update(
        _reference(),
        _loads(right_n=14.0, left_n=14.0),
        dt_s=0.01,
        intent=SupportIntent.RIGHT_SWING,
    )
    assert machine.phase is SupportPhase.SHIFT_WEIGHT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason is None
    np.testing.assert_allclose(command.positions_rad, HOME, atol=1e-12)


def test_stale_reference_requests_safe_two_foot_return() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    state = _loads(right_n=1.0, left_n=27.0)
    _admit_from_double_support(machine, SupportIntent.RIGHT_SWING)
    for _ in range(100):
        machine.update(_reference(), state, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING

    machine.update(_reference(stale=True), state, dt_s=0.01)

    assert machine.intent is SupportIntent.DOUBLE_SUPPORT
    assert machine.phase is SupportPhase.LOWER_SWING
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "stale_reference"


@pytest.mark.parametrize(
    "state",
    (
        _loads(right_n=0.0, left_n=0.0),
        _loads(right_n=0.0, left_n=28.0),
        _loads(right_n=2.0, left_n=2.0),
    ),
)
def test_one_leg_cycle_cannot_start_without_loaded_bilateral_contact(
    state: SimpleNamespace,
) -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())

    command = machine.update(
        _reference(),
        state,
        dt_s=0.01,
        intent=SupportIntent.RIGHT_SWING,
    )

    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    assert machine.active_intent is SupportIntent.DOUBLE_SUPPORT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.shift_progress == 0.0
    assert machine.last_diagnostics.abort_reason == "start_contact_missing"
    np.testing.assert_allclose(command.positions_rad, HOME, atol=1e-12)

    machine.update(
        _reference(),
        _loads(right_n=14.0, left_n=14.0),
        dt_s=0.01,
    )
    assert machine.phase is SupportPhase.SHIFT_WEIGHT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason is None


@pytest.mark.parametrize(
    "target_phase",
    (SupportPhase.SHIFT_WEIGHT, SupportPhase.VERIFY_STANCE),
)
@pytest.mark.parametrize(
    "lost_support",
    (
        _loads(right_n=0.0, left_n=0.0),
        _loads(right_n=0.0, left_n=28.0),
        _loads(right_n=1.0, left_n=1.0),
    ),
    ids=("both-contacts-lost", "one-contact-lost", "total-load-lost"),
)
def test_pre_lift_contact_loss_uses_grace_then_aborts_without_more_shift(
    target_phase: SupportPhase,
    lost_support: SimpleNamespace,
) -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    two_feet = _loads(right_n=14.0, left_n=14.0)
    _admit_from_double_support(machine, SupportIntent.RIGHT_SWING)
    while machine.phase is not target_phase:
        machine.update(_reference(), two_feet, dt_s=0.01)

    # Two 10 ms dropouts remain below the configured 30 ms grace and do not
    # create a fault or prematurely leave the active phase.
    for _ in range(2):
        machine.update(_reference(), lost_support, dt_s=0.01)
        assert machine.phase is target_phase
        assert machine.last_diagnostics is not None
        assert machine.last_diagnostics.abort_reason is None

    # A good bilateral sample resets the grace timer.  A second short dropout
    # is therefore also tolerated rather than being accumulated forever.
    machine.update(_reference(), two_feet, dt_s=0.01)
    for _ in range(2):
        machine.update(_reference(), lost_support, dt_s=0.01)
        assert machine.phase is target_phase
        assert machine.last_diagnostics is not None
        assert machine.last_diagnostics.abort_reason is None

    for _ in range(4):
        machine.update(_reference(), lost_support, dt_s=0.01)
        if machine.phase is SupportPhase.CENTER_WEIGHT:
            break

    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "support_contact_lost"
    assert machine.last_diagnostics.blocked_intent is SupportIntent.RIGHT_SWING
    shift_at_abort = machine.last_diagnostics.shift_progress

    # Recovery may retain or reduce the lateral shift, but must never continue
    # increasing it after bilateral support has been lost.
    for _ in range(5):
        machine.update(_reference(), lost_support, dt_s=0.01)
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.shift_progress <= shift_at_abort


@pytest.mark.parametrize(
    ("target_phase", "recovery_phase"),
    (
        (SupportPhase.SHIFT_WEIGHT, SupportPhase.CENTER_WEIGHT),
        (SupportPhase.VERIFY_STANCE, SupportPhase.CENTER_WEIGHT),
        (SupportPhase.LIFT_SWING, SupportPhase.LOWER_SWING),
        (SupportPhase.HOLD_SWING, SupportPhase.LOWER_SWING),
    ),
)
def test_stale_reference_is_an_explicit_abort_in_every_active_phase(
    target_phase: SupportPhase,
    recovery_phase: SupportPhase,
) -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    one_foot = _loads(right_n=1.0, left_n=27.0)
    _admit_from_double_support(machine, SupportIntent.RIGHT_SWING)

    for _ in range(100):
        if machine.phase is target_phase:
            break
        machine.update(_reference(), one_foot, dt_s=0.01)
    assert machine.phase is target_phase
    if target_phase is SupportPhase.LIFT_SWING:
        machine.update(_reference(), one_foot, dt_s=0.01)
        assert machine.phase is SupportPhase.LIFT_SWING

    machine.update(_reference(stale=True), one_foot, dt_s=0.01)

    assert machine.phase is recovery_phase
    assert machine.intent is SupportIntent.DOUBLE_SUPPORT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "stale_reference"
    assert machine.last_diagnostics.blocked_intent is SupportIntent.RIGHT_SWING

    two_feet = _loads(right_n=14.0, left_n=14.0)
    for _ in range(150):
        machine.update(_reference(), two_feet, dt_s=0.01)
        if machine.phase is SupportPhase.DOUBLE_SUPPORT:
            break
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "stale_reference"


def test_upstream_forced_return_uses_explicit_abort_reason() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    _admit_from_double_support(machine, SupportIntent.LEFT_SWING)

    machine.update(
        _reference(),
        _loads(right_n=14.0, left_n=14.0),
        dt_s=0.01,
        force_return_reason="stale_support_intent",
    )

    assert machine.phase is SupportPhase.CENTER_WEIGHT
    assert machine.intent is SupportIntent.DOUBLE_SUPPORT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "stale_support_intent"
    assert machine.last_diagnostics.blocked_intent is SupportIntent.LEFT_SWING


def test_stale_support_in_hold_forces_return_after_double_support_was_requested() -> None:
    config = replace(_fast_config(), minimum_hold_duration_s=1.0)
    machine = SupportStateMachine(LOWER, UPPER, config)
    one_foot = _loads(right_n=1.0, left_n=27.0)
    _admit_from_double_support(machine, SupportIntent.RIGHT_SWING)
    for _ in range(100):
        machine.update(_reference(), one_foot, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING

    # A valid camera-side release has changed the requested intent, but the
    # lifted foot has not physically begun returning and the ordinary path
    # would remain in the configured HOLD dwell.
    machine.set_intent(SupportIntent.DOUBLE_SUPPORT)
    assert machine.intent is SupportIntent.DOUBLE_SUPPORT
    reason = _stale_support_force_return_reason(
        support_estimate_stale=True,
        reference_stale=False,
        prior_phase=machine.phase,
    )
    assert reason == "stale_support_intent"

    machine.update(
        _reference(),
        one_foot,
        dt_s=0.01,
        force_return_reason=reason,
    )

    assert machine.phase is SupportPhase.LOWER_SWING
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "stale_support_intent"
    assert machine.last_diagnostics.blocked_intent is SupportIntent.RIGHT_SWING


@pytest.mark.parametrize("reason", ("", "   ", 3))
def test_forced_return_reason_must_be_a_non_empty_string(reason: object) -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    with pytest.raises(ValueError, match="force_return_reason"):
        machine.update(
            _reference(),
            _loads(right_n=14.0, left_n=14.0),
            dt_s=0.01,
            force_return_reason=reason,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("state", "reason"),
    (
        (
            _loads(right_n=14.0, left_n=14.0, tilt_rad=np.radians(13.0)),
            "start_tilt_limit",
        ),
        (
            _loads(right_n=14.0, left_n=14.0, angular_speed_rad_s=1.1),
            "start_angular_speed_limit",
        ),
    ),
)
def test_unstable_base_cannot_start_weight_shift(
    state: SimpleNamespace,
    reason: str,
) -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())

    command = machine.update(
        _reference(),
        state,
        dt_s=0.01,
        intent=SupportIntent.RIGHT_SWING,
    )

    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    assert machine.active_intent is SupportIntent.DOUBLE_SUPPORT
    assert machine.last_diagnostics is not None
    assert not machine.last_diagnostics.start_stable
    assert machine.last_diagnostics.active_stable
    assert machine.last_diagnostics.abort_reason == reason
    np.testing.assert_allclose(command.positions_rad, HOME, atol=1e-12)


@pytest.mark.parametrize(
    ("target_phase", "recovery_phase"),
    (
        (SupportPhase.SHIFT_WEIGHT, SupportPhase.CENTER_WEIGHT),
        (SupportPhase.VERIFY_STANCE, SupportPhase.CENTER_WEIGHT),
        (SupportPhase.LIFT_SWING, SupportPhase.LOWER_SWING),
        (SupportPhase.HOLD_SWING, SupportPhase.LOWER_SWING),
    ),
)
def test_active_tilt_limit_aborts_every_non_recovery_phase(
    target_phase: SupportPhase,
    recovery_phase: SupportPhase,
) -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    loaded = _loads(right_n=1.0, left_n=27.0)
    _admit_from_double_support(machine, SupportIntent.RIGHT_SWING)

    for _ in range(100):
        machine.update(_reference(), loaded, dt_s=0.01)
        if machine.phase is target_phase:
            break
    assert machine.phase is target_phase
    if target_phase is SupportPhase.LIFT_SWING:
        # Establish non-zero lift progress so the safe abort path must lower
        # the foot before it can center the stance.
        machine.update(_reference(), loaded, dt_s=0.01)
        assert machine.phase is SupportPhase.LIFT_SWING

    unstable = _loads(
        right_n=1.0,
        left_n=27.0,
        tilt_rad=np.radians(19.0),
    )
    machine.update(_reference(), unstable, dt_s=0.01)

    assert machine.phase is recovery_phase
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "active_tilt_limit"
    assert not machine.last_diagnostics.active_stable
    assert machine.last_diagnostics.blocked_intent is SupportIntent.RIGHT_SWING


def test_active_angular_speed_limit_aborts_and_recovery_is_not_restarted() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    loaded = _loads(right_n=1.0, left_n=27.0)
    _admit_from_double_support(machine, SupportIntent.RIGHT_SWING)
    for _ in range(100):
        machine.update(_reference(), loaded, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING

    unstable = _loads(
        right_n=14.0,
        left_n=14.0,
        angular_speed_rad_s=3.1,
    )
    machine.update(_reference(), unstable, dt_s=0.01)
    assert machine.phase is SupportPhase.LOWER_SWING
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "active_angular_speed_limit"

    phases: set[SupportPhase] = set()
    for _ in range(150):
        machine.update(_reference(), unstable, dt_s=0.01)
        phases.add(machine.phase)
        if machine.phase is SupportPhase.DOUBLE_SUPPORT:
            break
    assert SupportPhase.CENTER_WEIGHT in phases
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    assert machine.last_diagnostics is not None
    assert machine.last_diagnostics.abort_reason == "active_angular_speed_limit"


def test_latch_releases_request_rejected_by_double_support_admission_gate() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    latch = SupportIntentLatch()
    unstable = _loads(
        right_n=14.0,
        left_n=14.0,
        tilt_rad=np.radians(13.0),
    )

    request = latch.update(SupportIntent.RIGHT_SWING, machine.phase)
    machine.update(_reference(), unstable, dt_s=0.01, intent=request)
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT

    request = latch.update(SupportIntent.DOUBLE_SUPPORT, machine.phase)
    assert request is SupportIntent.DOUBLE_SUPPORT
    machine.update(_reference(), unstable, dt_s=0.01, intent=request)
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    assert machine.intent is SupportIntent.DOUBLE_SUPPORT


@pytest.mark.parametrize(
    ("settings", "message"),
    (
        (
            {"start_max_tilt_rad": 0.3, "active_max_tilt_rad": 0.3},
            "start_max_tilt_rad",
        ),
        (
            {
                "start_max_angular_speed_rad_s": 2.0,
                "active_max_angular_speed_rad_s": 1.0,
            },
            "start_max_angular_speed_rad_s",
        ),
        ({"active_max_tilt_rad": float("nan")}, "stability limits"),
    ),
)
def test_stability_threshold_validation(
    settings: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SupportControlConfig(**settings)


def test_base_motion_observations_are_validated() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    invalid_quaternion = _loads(right_n=14.0, left_n=14.0)
    invalid_quaternion.base_orientation_wxyz = np.zeros(4)
    with pytest.raises(ValueError, match="non-zero norm"):
        machine.update(_reference(), invalid_quaternion, dt_s=0.01)

    invalid_velocity = _loads(right_n=14.0, left_n=14.0)
    invalid_velocity.base_angular_velocity_rad_s = np.zeros(2)
    with pytest.raises(ValueError, match="finite 3-vector"):
        machine.update(_reference(), invalid_velocity, dt_s=0.01)


def _tilt_rad(quaternion_wxyz: np.ndarray) -> float:
    _, x, y, _ = quaternion_wxyz
    return float(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))


@pytest.mark.parametrize("intent", (SupportIntent.RIGHT_SWING, SupportIntent.LEFT_SWING))
def test_free_base_shifts_load_lifts_and_returns_each_leg(intent: SupportIntent) -> None:
    with HumanoidSimulation("free") as simulation:
        assert simulation.model.neq == 0
        standing = StandingBalanceController.from_simulation(
            simulation, load_standing_balance_config("config/balance.yaml")
        )
        support = SupportStateMachine.from_simulation(
            simulation, load_support_control_config("config/balance.yaml")
        )
        reference = RobotJointCommand.humanoid(0.0, simulation.home_positions_rad, 1.0)
        dt_s = float(simulation.model.opt.timestep)
        maximum_tilt = 0.0

        def physics_step() -> object:
            nonlocal maximum_tilt
            state = simulation.get_state()
            standing_command = standing.update(reference, state, dt_s=dt_s)
            motor_command = support.update(standing_command, state, dt_s=dt_s)
            simulation.apply_joint_command(motor_command)
            stepped = simulation.step()
            maximum_tilt = max(maximum_tilt, _tilt_rad(stepped.base_orientation_wxyz))
            return stepped

        # Establish the verified two-foot standing equilibrium first.
        for _ in range(1_000):
            state = physics_step()

        support.set_intent(intent)
        for _ in range(3_000):
            state = physics_step()
            if support.phase is SupportPhase.HOLD_SWING:
                break
        assert support.phase is SupportPhase.HOLD_SWING

        # The requested leg remains physically clear of the floor while the
        # opposite sole carries the robot weight.
        for _ in range(500):
            state = physics_step()
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
        assert swing_height - stance_height > 0.025
        assert stance_force > 20.0
        assert swing_force < 3.0
        assert state.base_position_m[2] > 0.82
        assert np.degrees(maximum_tilt) < 18.0

        support.set_intent(SupportIntent.DOUBLE_SUPPORT)
        for _ in range(4_000):
            state = physics_step()
            if support.phase is SupportPhase.DOUBLE_SUPPORT:
                break
        assert support.phase is SupportPhase.DOUBLE_SUPPORT
        for _ in range(500):
            state = physics_step()

        assert state.base_position_m[2] > 0.85
        assert state.right_foot_normal_force_n > 5.0
        assert state.left_foot_normal_force_n > 5.0
        assert _tilt_rad(state.base_orientation_wxyz) < np.radians(10.0)


def test_support_yaml_is_runtime_source_for_phase_timing_and_force_gates() -> None:
    config = load_support_control_config("config/balance.yaml")
    assert config.shift_duration_s == 2.0
    assert config.load_confirm_duration_s == 0.15
    assert config.min_stance_force_n == 12.0
    assert np.degrees(config.start_max_tilt_rad) == pytest.approx(12.0)
    assert np.degrees(config.active_max_tilt_rad) == pytest.approx(18.0)
    assert config.start_max_angular_speed_rad_s == 1.0
    assert config.active_max_angular_speed_rad_s == 3.0
    assert config.swing_reference_blend == 0.25
    assert np.degrees(config.max_swing_reference_delta_rad) == pytest.approx(12.0)
    assert config.single_support_upper_body_scale == 0.15
    assert config.upper_body_rate_limit_rad_s == 2.5
    assert config.lower_body_rate_limit_rad_s == 1.2
    assert config.early_touchdown_max_swing_progress == 0.45
    assert config.minimum_hold_duration_s == 0.25


def test_frozen_cycle_applies_live_capture_delta_exactly_once() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    loaded = _loads(right_n=1.0, left_n=27.0)
    sagittal = np.asarray((10, 11, 12, 13, 14, 15))

    def command(recovery_rad: float) -> BalancedJointCommand:
        recovery = np.zeros(20)
        recovery[10:12] = -0.5 * recovery_rad
        recovery[12:14] = 0.5 * recovery_rad
        recovery[14:16] = recovery_rad
        return BalancedJointCommand(
            0.0,
            tuple(spec.name for spec in DEFAULT_JOINT_SPECS),
            HOME + recovery,
            1.0,
            False,
            pose_reference_positions_rad=HOME,
            capture_recovery_positions_rad=recovery,
        )

    admitted = command(0.04)
    first = machine.update(
        admitted,
        _loads(right_n=14.0, left_n=14.0),
        dt_s=0.01,
        intent=SupportIntent.RIGHT_SWING,
    )
    assert machine.phase is SupportPhase.SHIFT_WEIGHT
    np.testing.assert_allclose(first.positions_rad[sagittal], admitted.positions_rad[sagittal])

    changed = command(0.10)
    second = machine.update(changed, loaded, dt_s=0.01)
    expected_delta = (
        changed.capture_recovery_positions_rad
        - admitted.capture_recovery_positions_rad
    )
    np.testing.assert_allclose(
        second.positions_rad[sagittal],
        first.positions_rad[sagittal] + expected_delta[sagittal],
    )


def test_swing_pose_reference_waits_for_load_gate_then_remains_correlated() -> None:
    config = _fast_config()
    machine = SupportStateMachine(LOWER, UPPER, config)
    baseline = _reference()
    loaded = _loads(right_n=1.0, left_n=27.0)
    _admit_from_double_support(
        machine,
        SupportIntent.RIGHT_SWING,
        reference=baseline,
    )

    pose = HOME.copy()
    pose_delta = np.asarray((0.18, -0.24, 0.30, -0.16, 0.0, -0.10))
    right_leg = np.asarray((6, 8, 10, 12, 14, 16))
    left_leg = np.asarray((7, 9, 11, 13, 15, 17))
    pose[right_leg] += pose_delta
    changing = BalancedJointCommand(
        0.1,
        baseline.joint_names,
        baseline.positions_rad,
        1.0,
        False,
        pose_reference_positions_rad=pose,
    )

    # Weight transfer is based on the quiet admitted command, not an early
    # visually inferred leg lift.
    during_shift = machine.update(changing, loaded, dt_s=0.01)
    shift, _ = support_offsets(SupportIntent.RIGHT_SWING)
    progress = machine.last_diagnostics.shift_progress
    ramp = progress * progress * (3.0 - 2.0 * progress)
    expected_shift = np.clip(HOME + ramp * shift, LOWER, UPPER)
    protected = np.asarray((6, 8, 10, 12, 16))
    np.testing.assert_allclose(
        during_shift.positions_rad[protected],
        expected_shift[protected],
        atol=1e-12,
    )

    for _ in range(100):
        held = machine.update(changing, loaded, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING
    shift, lift = support_offsets(SupportIntent.RIGHT_SWING)
    capped_pose_delta = np.clip(
        pose_delta,
        -config.max_swing_reference_delta_rad,
        config.max_swing_reference_delta_rad,
    )
    expected = np.clip(
        HOME + shift + (1.0 - config.swing_reference_blend) * lift,
        LOWER,
        UPPER,
    )
    expected[right_leg] += config.swing_reference_blend * capped_pose_delta
    expected = np.clip(expected, LOWER, UPPER)
    np.testing.assert_allclose(held.positions_rad, expected, atol=1e-12)
    profile_without_pose = np.clip(
        HOME + shift + (1.0 - config.swing_reference_blend) * lift,
        LOWER,
        UPPER,
    )
    admitted_delta = held.positions_rad[right_leg] - profile_without_pose[right_leg]
    # Every unclipped swing component keeps the sign and ordering of the
    # continuous reference; the stance leg remains on the verified profile.
    assert np.dot(admitted_delta, pose_delta) > 0.0
    assert np.ptp(admitted_delta) > np.radians(5.0)
    np.testing.assert_allclose(
        held.positions_rad[left_leg], profile_without_pose[left_leg], atol=1e-12
    )


def test_shift_admission_preserves_projected_lower_pose_without_a_home_reset() -> None:
    config = replace(
        _fast_config(),
        upper_body_rate_limit_rad_s=2.5,
        lower_body_rate_limit_rad_s=1.2,
    )
    machine = SupportStateMachine(
        LOWER,
        UPPER,
        config,
        home_positions_rad=HOME,
    )
    crouched = HOME.copy()
    crouched[10:12] += np.radians(7.0)
    crouched[12:14] += np.radians(10.0)
    crouched[14:16] += np.radians(3.0)
    reference = RobotJointCommand.humanoid(0.0, crouched, 1.0)
    two_feet = _loads(right_n=14.0, left_n=14.0)
    loaded = _loads(right_n=1.0, left_n=27.0)
    for _ in range(100):
        previous = machine.update(reference, two_feet, dt_s=0.01)

    shifted = machine.update(
        reference,
        two_feet,
        dt_s=0.01,
        intent=SupportIntent.RIGHT_SWING,
    )

    assert machine.phase is SupportPhase.SHIFT_WEIGHT
    step = np.abs(shifted.positions_rad - previous.positions_rad)
    rate_limits = np.full(20, config.lower_body_rate_limit_rad_s)
    rate_limits[np.asarray((0, 1, 2, 3, 4, 5, 18, 19))] = (
        config.upper_body_rate_limit_rad_s
    )
    assert np.max(step - rate_limits * 0.01) <= 1e-12
    # SHIFT changes only the verified roll profile.  The already safe
    # sagittal crouch must not be overwritten with home at admission.
    sagittal = np.asarray((10, 11, 12, 13, 14, 15))
    np.testing.assert_allclose(
        shifted.positions_rad[sagittal], previous.positions_rad[sagittal], atol=1e-12
    )


def test_single_support_smoothly_reduces_moving_upper_body_envelope() -> None:
    config = _fast_config()
    machine = SupportStateMachine(LOWER, UPPER, config)
    loaded = _loads(right_n=1.0, left_n=27.0)
    machine.update(_reference(), loaded, dt_s=0.01)
    machine.set_intent(SupportIntent.RIGHT_SWING)
    upper = HOME.copy()
    upper_indices = np.asarray((0, 1, 2, 3, 4, 5, 18, 19))
    upper_delta = np.asarray((0.8, -0.7, 0.5, -0.4, 0.3, -0.2, 1.0, 0.4))
    upper[upper_indices] += upper_delta
    moving = RobotJointCommand.humanoid(0.1, upper, 1.0)

    first = machine.update(
        moving,
        _loads(right_n=14.0, left_n=14.0),
        dt_s=0.01,
    )
    # Admission is continuous: the first transfer tick is still nearly the
    # current IK pose, rather than snapping to the smaller envelope.
    assert np.max(np.abs(first.positions_rad[upper_indices] - upper[upper_indices])) < 0.02

    for _ in range(100):
        held = machine.update(moving, loaded, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING
    expected_upper = (
        HOME[upper_indices]
        + config.single_support_upper_body_scale * upper_delta
    )
    np.testing.assert_allclose(
        held.positions_rad[upper_indices], expected_upper, atol=1e-12
    )


def test_final_slew_limit_covers_touchdown_to_center_transition() -> None:
    config = replace(
        _fast_config(),
        upper_body_rate_limit_rad_s=2.5,
        lower_body_rate_limit_rad_s=1.2,
    )
    machine = SupportStateMachine(LOWER, UPPER, config)
    loaded = _loads(right_n=1.0, left_n=27.0)
    two_feet = _loads(right_n=14.0, left_n=14.0)
    baseline = machine.update(_reference(), two_feet, dt_s=0.01)
    machine.set_intent(SupportIntent.RIGHT_SWING)
    upper = HOME.copy()
    upper_indices = np.asarray((0, 1, 2, 3, 4, 5, 18, 19))
    upper[upper_indices] += np.asarray((0.8, -0.7, 0.5, -0.4, 0.3, -0.2, 1.0, 0.4))
    moving = RobotJointCommand.humanoid(0.1, upper, 1.0)
    previous = baseline.positions_rad.copy()
    maximum_ratio = 0.0

    def update_and_check(state: SimpleNamespace) -> None:
        nonlocal maximum_ratio, previous
        command = machine.update(moving, state, dt_s=0.01)
        rate_limits = np.full(20, config.lower_body_rate_limit_rad_s)
        rate_limits[upper_indices] = config.upper_body_rate_limit_rad_s
        ratio = np.max(np.abs(command.positions_rad - previous) / (rate_limits * 0.01))
        maximum_ratio = max(maximum_ratio, float(ratio))
        previous = command.positions_rad.copy()

    update_and_check(two_feet)
    for _ in range(100):
        update_and_check(loaded)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING
    machine.set_intent(SupportIntent.DOUBLE_SUPPORT)
    observed: set[SupportPhase] = set()
    for _ in range(200):
        update_and_check(two_feet)
        observed.add(machine.phase)
        if machine.phase is SupportPhase.DOUBLE_SUPPORT:
            break

    assert SupportPhase.CENTER_WEIGHT in observed
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    assert maximum_ratio <= 1.0 + 1e-12


def test_camera_intent_is_latched_until_the_safe_lift_reaches_hold() -> None:
    latch = SupportIntentLatch()

    assert latch.update(
        SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT
    ) is SupportIntent.RIGHT_SWING
    # A short visual gesture cannot cancel halfway through weight transfer.
    assert latch.update(
        SupportIntent.DOUBLE_SUPPORT, SupportPhase.SHIFT_WEIGHT
    ) is SupportIntent.RIGHT_SWING
    assert latch.update(
        SupportIntent.DOUBLE_SUPPORT, SupportPhase.LIFT_SWING
    ) is SupportIntent.RIGHT_SWING
    assert latch.update(
        SupportIntent.DOUBLE_SUPPORT, SupportPhase.HOLD_SWING
    ) is SupportIntent.DOUBLE_SUPPORT


def test_latched_intent_releases_immediately_on_stale_or_controller_abort() -> None:
    latch = SupportIntentLatch()
    latch.update(SupportIntent.LEFT_SWING, SupportPhase.DOUBLE_SUPPORT)
    assert latch.update(
        SupportIntent.LEFT_SWING, SupportPhase.SHIFT_WEIGHT, stale=True
    ) is SupportIntent.DOUBLE_SUPPORT

    latch.update(SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT)
    assert latch.update(
        SupportIntent.RIGHT_SWING, SupportPhase.VERIFY_STANCE, aborted=True
    ) is SupportIntent.DOUBLE_SUPPORT
    assert latch.blocked_intent is SupportIntent.RIGHT_SWING
    # The same still-raised leg cannot create an abort/retry loop.
    assert latch.update(
        SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT
    ) is SupportIntent.DOUBLE_SUPPORT
    assert latch.update(
        SupportIntent.DOUBLE_SUPPORT, SupportPhase.DOUBLE_SUPPORT
    ) is SupportIntent.DOUBLE_SUPPORT
    assert latch.blocked_intent is None
    assert latch.update(
        SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT
    ) is SupportIntent.RIGHT_SWING


def test_persistent_failed_camera_intent_does_not_retry_without_a_new_edge() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    latch = SupportIntentLatch()
    evenly_loaded = _loads(right_n=14.0, left_n=14.0)
    shift_entries = 0
    previous_phase = machine.phase

    for _ in range(250):
        diagnostics = machine.last_diagnostics
        aborting = bool(
            diagnostics is not None
            and diagnostics.abort_reason is not None
            and machine.phase in {SupportPhase.LOWER_SWING, SupportPhase.CENTER_WEIGHT}
        )
        request = latch.update(
            SupportIntent.RIGHT_SWING,
            machine.phase,
            aborted=aborting,
        )
        machine.update(_reference(), evenly_loaded, dt_s=0.01, intent=request)
        if machine.phase is SupportPhase.SHIFT_WEIGHT and previous_phase is not machine.phase:
            shift_entries += 1
        previous_phase = machine.phase

    assert shift_entries == 1
    assert latch.blocked_intent is SupportIntent.RIGHT_SWING
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT


def test_opposite_swing_during_safe_return_is_queued_and_consumed_once() -> None:
    latch = SupportIntentLatch(pending_max_age_s=6.0)

    assert latch.update(
        SupportIntent.RIGHT_SWING,
        SupportPhase.DOUBLE_SUPPORT,
        timestamp_s=0.0,
    ) is SupportIntent.RIGHT_SWING
    # An opposite observation at HOLD requests a safe lower, never a direct
    # side switch.  It becomes pending only once the FSM enters its return path.
    assert latch.update(
        SupportIntent.LEFT_SWING,
        SupportPhase.HOLD_SWING,
        timestamp_s=0.1,
    ) is SupportIntent.DOUBLE_SUPPORT
    assert latch.pending_intent is None
    assert latch.update(
        SupportIntent.LEFT_SWING,
        SupportPhase.LOWER_SWING,
        timestamp_s=0.2,
    ) is SupportIntent.DOUBLE_SUPPORT
    assert latch.pending_intent is SupportIntent.LEFT_SWING
    assert latch.pending_since_s == pytest.approx(0.2)

    # The visual gesture may already be over while touchdown and centering run.
    assert latch.update(
        SupportIntent.DOUBLE_SUPPORT,
        SupportPhase.VERIFY_TOUCHDOWN,
        timestamp_s=1.0,
    ) is SupportIntent.DOUBLE_SUPPORT
    assert latch.update(
        SupportIntent.DOUBLE_SUPPORT,
        SupportPhase.CENTER_WEIGHT,
        timestamp_s=2.0,
    ) is SupportIntent.DOUBLE_SUPPORT
    assert latch.update(
        SupportIntent.DOUBLE_SUPPORT,
        SupportPhase.DOUBLE_SUPPORT,
        timestamp_s=3.0,
    ) is SupportIntent.LEFT_SWING
    assert latch.pending_intent is None


def test_pending_swing_expires_and_is_not_refreshed_by_a_held_observation() -> None:
    latch = SupportIntentLatch(pending_max_age_s=1.0)
    latch.update(SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT, timestamp_s=0.0)
    latch.update(SupportIntent.DOUBLE_SUPPORT, SupportPhase.HOLD_SWING, timestamp_s=0.1)
    latch.update(SupportIntent.LEFT_SWING, SupportPhase.LOWER_SWING, timestamp_s=0.2)

    # Repeated LEFT observations do not extend the original one-second lease.
    latch.update(SupportIntent.LEFT_SWING, SupportPhase.CENTER_WEIGHT, timestamp_s=0.8)
    latch.update(SupportIntent.LEFT_SWING, SupportPhase.CENTER_WEIGHT, timestamp_s=1.3)
    assert latch.pending_intent is None

    # Once the old observation has ended, returning to DOUBLE cannot replay it.
    assert latch.update(
        SupportIntent.DOUBLE_SUPPORT,
        SupportPhase.DOUBLE_SUPPORT,
        timestamp_s=1.4,
    ) is SupportIntent.DOUBLE_SUPPORT


def test_pending_queue_ignores_same_side_and_clears_on_safety_events() -> None:
    latch = SupportIntentLatch()
    latch.update(SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT, timestamp_s=0.0)
    latch.update(SupportIntent.DOUBLE_SUPPORT, SupportPhase.HOLD_SWING, timestamp_s=0.1)
    latch.update(SupportIntent.RIGHT_SWING, SupportPhase.LOWER_SWING, timestamp_s=0.2)
    assert latch.pending_intent is None

    latch.update(SupportIntent.LEFT_SWING, SupportPhase.LOWER_SWING, timestamp_s=0.3)
    assert latch.pending_intent is SupportIntent.LEFT_SWING
    latch.update(
        SupportIntent.DOUBLE_SUPPORT,
        SupportPhase.VERIFY_TOUCHDOWN,
        stale=True,
        timestamp_s=0.4,
    )
    assert latch.pending_intent is None

    # Abort and reset are independent hard cancellation paths too.
    latch.reset()
    latch.update(SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT, timestamp_s=1.0)
    latch.update(SupportIntent.DOUBLE_SUPPORT, SupportPhase.HOLD_SWING, timestamp_s=1.1)
    latch.update(SupportIntent.LEFT_SWING, SupportPhase.CENTER_WEIGHT, timestamp_s=1.2)
    assert latch.pending_intent is SupportIntent.LEFT_SWING
    latch.update(
        SupportIntent.LEFT_SWING,
        SupportPhase.CENTER_WEIGHT,
        aborted=True,
        timestamp_s=1.3,
    )
    assert latch.pending_intent is None
    latch.reset()
    assert latch.pending_intent is None


def test_pending_queue_validates_age_and_timestamp_but_timestamp_is_optional() -> None:
    with pytest.raises(ValueError, match="pending_max_age_s"):
        SupportIntentLatch(pending_max_age_s=0.0)

    latch = SupportIntentLatch(clock=lambda: 10.0)
    assert latch.update(
        SupportIntent.RIGHT_SWING,
        SupportPhase.DOUBLE_SUPPORT,
    ) is SupportIntent.RIGHT_SWING
    with pytest.raises(ValueError, match="monotonic"):
        latch.update(
            SupportIntent.DOUBLE_SUPPORT,
            SupportPhase.HOLD_SWING,
            timestamp_s=9.0,
        )
