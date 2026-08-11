from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

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
        min_touchdown_total_force_n=10.0,
    )


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
    machine.set_intent(intent)
    phases: list[SupportPhase] = []
    command = _reference()

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
    machine.set_intent(SupportIntent.LEFT_SWING)
    loaded = _loads(right_n=27.0, left_n=1.0)
    lost = _loads(right_n=2.0, left_n=2.0)

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
    assert SupportPhase.VERIFY_TOUCHDOWN in phases
    assert SupportPhase.CENTER_WEIGHT in phases
    assert machine.phase is SupportPhase.DOUBLE_SUPPORT
    np.testing.assert_allclose(returned.positions_rad, HOME, atol=1e-12)


def test_stale_reference_requests_safe_two_foot_return() -> None:
    machine = SupportStateMachine(LOWER, UPPER, _fast_config())
    machine.set_intent(SupportIntent.RIGHT_SWING)
    state = _loads(right_n=1.0, left_n=27.0)
    for _ in range(100):
        machine.update(_reference(), state, dt_s=0.01)
        if machine.phase is SupportPhase.HOLD_SWING:
            break
    assert machine.phase is SupportPhase.HOLD_SWING

    machine.update(_reference(stale=True), state, dt_s=0.01)

    assert machine.intent is SupportIntent.DOUBLE_SUPPORT
    assert machine.phase is SupportPhase.LOWER_SWING


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
    machine.set_intent(SupportIntent.RIGHT_SWING)

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
    machine.set_intent(SupportIntent.RIGHT_SWING)
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
    assert SupportPhase.VERIFY_TOUCHDOWN in phases
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


def test_camera_intent_is_latched_until_the_safe_lift_reaches_hold() -> None:
    latch = SupportIntentLatch()

    assert latch.update(SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT) is SupportIntent.RIGHT_SWING
    # A short visual gesture cannot cancel halfway through weight transfer.
    assert latch.update(SupportIntent.DOUBLE_SUPPORT, SupportPhase.SHIFT_WEIGHT) is SupportIntent.RIGHT_SWING
    assert latch.update(SupportIntent.DOUBLE_SUPPORT, SupportPhase.LIFT_SWING) is SupportIntent.RIGHT_SWING
    assert latch.update(SupportIntent.DOUBLE_SUPPORT, SupportPhase.HOLD_SWING) is SupportIntent.DOUBLE_SUPPORT


def test_latched_intent_releases_immediately_on_stale_or_controller_abort() -> None:
    latch = SupportIntentLatch()
    latch.update(SupportIntent.LEFT_SWING, SupportPhase.DOUBLE_SUPPORT)
    assert latch.update(SupportIntent.LEFT_SWING, SupportPhase.SHIFT_WEIGHT, stale=True) is SupportIntent.DOUBLE_SUPPORT

    latch.update(SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT)
    assert latch.update(SupportIntent.RIGHT_SWING, SupportPhase.VERIFY_STANCE, aborted=True) is SupportIntent.DOUBLE_SUPPORT
    assert latch.blocked_intent is SupportIntent.RIGHT_SWING
    # The same still-raised leg cannot create an abort/retry loop.
    assert latch.update(SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT) is SupportIntent.DOUBLE_SUPPORT
    assert latch.update(SupportIntent.DOUBLE_SUPPORT, SupportPhase.DOUBLE_SUPPORT) is SupportIntent.DOUBLE_SUPPORT
    assert latch.blocked_intent is None
    assert latch.update(SupportIntent.RIGHT_SWING, SupportPhase.DOUBLE_SUPPORT) is SupportIntent.RIGHT_SWING


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
