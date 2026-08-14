from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robot_human_interface.simulation import HumanoidSimulation
from tools import evaluate_freebase_robustness as robustness


def _motion(span_deg: float) -> dict[str, dict[str, object]]:
    return {
        group: {
            "maximum_span_deg": span_deg,
            "joint_span_deg": [span_deg] * len(indices),
            "minimum_excursion_deg": [-span_deg] * len(indices),
            "maximum_excursion_deg": [span_deg] * len(indices),
        }
        for group, indices in robustness.MOTION_GROUPS.items()
    }


def _support_phase_evidence(
    scenario: robustness.ScenarioSpec,
) -> tuple[list[str], list[dict[str, str]]]:
    if scenario.support_intent is None:
        return [], []
    sequence = [
        "double_support",
        "shift_weight",
        "verify_stance",
        "lift_swing",
        "hold_swing",
        "lower_swing",
        "center_weight",
        "double_support",
    ]
    events = [
        {
            "phase": phase,
            "active_intent": (
                "double_support"
                if index in (0, len(sequence) - 1)
                else scenario.support_intent.value
            ),
            "requested_intent": (
                "double_support" if index == 0 else scenario.support_intent.value
            ),
        }
        for index, phase in enumerate(sequence)
    ]
    return sequence, events


def _passing_evaluation(
    trial: robustness.TrialSpec,
    _thresholds: robustness.RobustnessThresholds,
) -> dict[str, object]:
    scenario = trial.scenario
    body_names = list(robustness.CANONICAL_POSITIVE_MASS_BODY_NAMES)
    actuator_names = list(robustness.JOINT_NAMES)
    parameters = robustness._sample_variation_from_names(
        body_names,
        actuator_names,
        seed=trial.seed,
        randomized=trial.randomized,
    )
    force_impulse = [0.0, 0.0, 0.0]
    torque_impulse = [0.0, 0.0, 0.0]
    peak_force = [0.0, 0.0, 0.0]
    peak_torque = [0.0, 0.0, 0.0]
    if scenario.kind == "push_sagittal":
        peak_force[0] = robustness.PUSH_FORCE_N
        peak_torque[1] = robustness.PUSH_TORQUE_N_M
        force_impulse[0] = (
            robustness.PUSH_FORCE_N
            * robustness.PUSH_DURATION_S
            * scenario.direction
        )
        torque_impulse[1] = (
            -robustness.PUSH_TORQUE_N_M
            * robustness.PUSH_DURATION_S
            * scenario.direction
        )
    elif scenario.kind == "push_lateral":
        peak_force[1] = robustness.PUSH_FORCE_N
        peak_torque[0] = robustness.PUSH_TORQUE_N_M
        force_impulse[1] = (
            robustness.PUSH_FORCE_N
            * robustness.PUSH_DURATION_S
            * scenario.direction
        )
        torque_impulse[0] = (
            robustness.PUSH_TORQUE_N_M
            * robustness.PUSH_DURATION_S
            * scenario.direction
        )
    phase_sequence, phase_events = _support_phase_evidence(scenario)
    recovery_start = scenario.duration_s - 1.0
    return {
        "trial_id": trial.trial_id,
        "scenario": scenario.name,
        "seed": trial.seed,
        "randomized": trial.randomized,
        "critical": trial.critical,
        "parameters": parameters,
        "model": {
            "scene": "free",
            "scene_path": robustness.CANONICAL_SCENE_PATH,
            "equality_constraint_count": 0,
            "base_joint_type": "free",
            "base_joint_id": robustness.CANONICAL_BASE_JOINT_ID,
            "total_mass_kg": 2.93,
            "timestep_s": 0.002,
            "actuator_count": robustness.CANONICAL_ACTUATOR_COUNT,
            "joint_count": robustness.CANONICAL_JOINT_COUNT,
            "positive_mass_body_names": body_names,
            "actuator_names": actuator_names,
            "state_mutation_policy": robustness.CANONICAL_STATE_MUTATION_POLICY,
        },
        "metrics": {
            "requested_duration_s": scenario.duration_s,
            "simulated_duration_s": scenario.duration_s,
            "duration_scale": 1.0,
            "completed_duration": True,
            "fell": False,
            "minimum_base_height_m": 0.89,
            "maximum_tilt_deg": 7.0,
            "maximum_horizontal_drift_m": 0.04,
            "final_horizontal_drift_m": 0.04,
            "maximum_loaded_foot_slip_speed_m_s": 0.04,
            "right_loaded_foot_slip_distance_m": 0.02,
            "left_loaded_foot_slip_distance_m": 0.02,
            "maximum_non_foot_ground_contacts": 0,
            "final_base_height_m": 0.91,
            "final_tilt_deg": 2.0,
            "motion": {
                "motor_command": _motion(30.0),
                "actual_joint": _motion(25.0),
            },
            "perturbation": {
                "kind": scenario.kind if scenario.kind.startswith("push") else None,
                "direction": scenario.direction,
                "peak_force_abs_n": peak_force,
                "peak_torque_abs_n_m": peak_torque,
                "signed_force_impulse_n_s": force_impulse,
                "signed_torque_impulse_n_m_s": torque_impulse,
            },
            "support": {
                "requested": (
                    scenario.support_intent.value
                    if scenario.support_intent is not None
                    else None
                ),
                "completed": True,
                "final_phase": "double_support",
                "phase_sequence": phase_sequence,
                "phase_events": phase_events,
                "abort_reasons": [],
                "maximum_right_foot_clearance_m": 0.04,
                "maximum_left_foot_clearance_m": 0.04,
                "maximum_right_foot_clearance_over_left_m": (
                    0.04
                    if scenario.support_intent is not robustness.SupportIntent.LEFT_SWING
                    else 0.0
                ),
                "maximum_left_foot_clearance_over_right_m": (
                    0.04
                    if scenario.support_intent is not robustness.SupportIntent.RIGHT_SWING
                    else 0.0
                ),
                "hold_observations": 100,
                "minimum_hold_stance_force_n": 20.0,
                "maximum_hold_swing_force_n": 1.0,
                "minimum_hold_stance_load_fraction": 0.90,
                "maximum_hold_swing_load_fraction": 0.10,
                "support_frame_observations": 100,
                "maximum_support_frame_com_to_stance_x_error_m": 0.04,
                "maximum_support_frame_com_to_stance_y_error_m": 0.04,
                "maximum_support_frame_stance_foot_travel_m": 0.005,
                "swing_contact_episodes": 1,
                "maximum_swing_foot_impact_speed_m_s": 0.10,
                "maximum_swing_foot_precontact_vertical_speed_m_s": 0.10,
                "maximum_swing_foot_impact_force_n": 20.0,
                "maximum_swing_foot_contact_impulse_n_s": 2.0,
            },
            "recovery": {
                "observation_s": 1.0,
                "window_start_time_s": recovery_start,
                "window_end_time_s": scenario.duration_s,
                "sample_period_s": 0.002,
                "expected_samples": 500,
                "samples": 500,
                "first_sample_time_s": recovery_start + 0.002,
                "last_sample_time_s": scenario.duration_s,
                "maximum_sample_gap_s": 0.002,
                "minimum_base_height_m": 0.90,
                "maximum_tilt_deg": 3.0,
                "maximum_horizontal_drift_m": 0.04,
                "maximum_base_speed_m_s": 0.03,
                "maximum_base_angular_speed_rad_s": 0.20,
                "maximum_joint_speed_rad_s": 0.20,
                "maximum_tracking_error_rad": 0.06,
                "maximum_loaded_foot_slip_speed_m_s": 0.01,
                "maximum_capture_point_error_m": 0.02,
                "minimum_right_foot_force_n": 10.0,
                "minimum_left_foot_force_n": 10.0,
                "minimum_total_support_force_n": 25.0,
            },
        },
    }


def _provenance() -> dict[str, object]:
    return {
        "captured_at_utc": "2026-08-13T00:00:00+00:00",
        "git_revision": "0" * 40,
        "git_worktree": {"dirty": True},
        "versions": {"mujoco": "test", "numpy": "test"},
        "runtime_input_sha256": robustness._runtime_input_hashes(),
    }


def test_default_matrix_predeclares_signed_and_bidirectional_cases() -> None:
    scenarios = robustness.default_scenarios()
    names = {scenario.name for scenario in scenarios}

    assert {
        "neutral_settle",
        "combined_upper_body_slow",
        "crouch_positive",
        "crouch_negative",
        "push_sagittal_positive",
        "push_sagittal_negative",
        "push_lateral_positive",
        "push_lateral_negative",
        "right_single_support",
        "left_single_support",
    } == names
    assert all(scenario.critical for scenario in scenarios)
    assert robustness.default_scenarios(include_one_leg=False) == scenarios[:-2]


def test_trial_matrix_has_nominal_critical_coverage_and_reproducible_random_seeds() -> None:
    scenarios = robustness.default_scenarios(include_one_leg=False)
    first = robustness.build_trial_specs(
        randomized_trials=17, seed=1234, scenarios=scenarios
    )
    second = robustness.build_trial_specs(
        randomized_trials=17, seed=1234, scenarios=scenarios
    )

    assert first == second
    nominal = [trial for trial in first if not trial.randomized]
    randomized = [trial for trial in first if trial.randomized]
    assert [trial.scenario for trial in nominal] == list(scenarios)
    assert all(trial.critical for trial in nominal)
    assert len(randomized) == 17
    assert len({trial.seed for trial in first}) == len(first)
    assert [trial.scenario for trial in randomized[: len(scenarios)]] == list(
        scenarios
    )


def test_canonical_matrix_adds_independent_twenty_per_side_holdout() -> None:
    trials = robustness.build_trial_specs(
        randomized_trials=robustness.DEFAULT_RANDOMIZED_TRIALS,
        seed=robustness.CANONICAL_SEED,
        scenarios=robustness.SCENARIOS,
    )

    critical = [trial for trial in trials if trial.cohort == "critical"]
    broad = [trial for trial in trials if trial.cohort == "broad_randomized"]
    holdout = [
        trial for trial in trials if trial.cohort == "single_support_holdout"
    ]
    assert len(critical) == len(robustness.SCENARIOS)
    assert len(broad) == robustness.DEFAULT_RANDOMIZED_TRIALS
    assert len(holdout) == (
        2 * robustness.DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
    )
    assert [trial.scenario.name for trial in holdout[:4]] == [
        "right_single_support",
        "left_single_support",
        "right_single_support",
        "left_single_support",
    ]
    assert [trial.seed for trial in holdout] == list(
        range(
            robustness.CANONICAL_SEED
            + robustness.SINGLE_SUPPORT_HOLDOUT_SEED_OFFSET,
            robustness.CANONICAL_SEED
            + robustness.SINGLE_SUPPORT_HOLDOUT_SEED_OFFSET
            + len(holdout),
        )
    )
    assert len({trial.seed for trial in trials}) == len(trials)

    for invalid in (True, 1.0, "1"):
        with pytest.raises(ValueError, match="randomized_trials"):
            robustness.build_trial_specs(randomized_trials=invalid)
        with pytest.raises(ValueError, match="holdout"):
            robustness.build_trial_specs(
                single_support_holdout_trials_per_side=invalid
            )
        with pytest.raises(ValueError, match="seed"):
            robustness.build_trial_specs(seed=invalid)


def test_each_single_support_holdout_side_requires_twenty_of_twenty() -> None:
    thresholds = robustness.RobustnessThresholds()
    trials = robustness.build_trial_specs(
        randomized_trials=robustness.DEFAULT_RANDOMIZED_TRIALS,
        seed=robustness.CANONICAL_SEED,
        scenarios=robustness.SCENARIOS,
    )
    results = robustness.evaluate_suite(
        trials, thresholds=thresholds, evaluator=_passing_evaluation
    )
    left_holdout = [
        result
        for result in results
        if result["cohort"] == "single_support_holdout"
        and result["scenario"] == "left_single_support"
    ]
    left_holdout[0]["passed"] = False

    report = robustness.build_report(
        results,
        thresholds=thresholds,
        selected_scenarios=robustness.SCENARIOS,
        randomized_trials_requested=robustness.DEFAULT_RANDOMIZED_TRIALS,
        single_support_holdout_trials_per_side_requested=(
            robustness.DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
        ),
        seed=robustness.CANONICAL_SEED,
        provenance=_provenance(),
    )
    gates = {gate["name"]: gate for gate in report["aggregate"]["gates"]}

    assert report["aggregate"]["randomized"]["pass_rate"] == 1.0
    assert (
        report["aggregate"]["single_support_holdout"]["by_side"]
        ["right_single_support"]["successes"]
        == 20
    )
    assert (
        report["aggregate"]["single_support_holdout"]["by_side"]
        ["left_single_support"]["successes"]
        == 19
    )
    assert gates["right_single_support_holdout_wilson_95_lower_bound"][
        "passed"
    ] is True
    assert gates["left_single_support_holdout_wilson_95_lower_bound"][
        "passed"
    ] is False
    assert report["overall_passed"] is False


def test_report_recomputes_trial_pass_from_nested_gates() -> None:
    thresholds = robustness.RobustnessThresholds()
    trials = robustness.build_trial_specs(
        randomized_trials=robustness.DEFAULT_RANDOMIZED_TRIALS,
        seed=robustness.CANONICAL_SEED,
        scenarios=robustness.SCENARIOS,
    )
    results = robustness.evaluate_suite(
        trials, thresholds=thresholds, evaluator=_passing_evaluation
    )
    tampered = next(
        result
        for result in results
        if result["cohort"] == "single_support_holdout"
        and result["scenario"] == "right_single_support"
    )
    assert tampered["passed"] is True
    tampered["gates"][0]["passed"] = False

    report = robustness.build_report(
        results,
        thresholds=thresholds,
        selected_scenarios=robustness.SCENARIOS,
        randomized_trials_requested=robustness.DEFAULT_RANDOMIZED_TRIALS,
        seed=robustness.CANONICAL_SEED,
        provenance=_provenance(),
    )
    gates = {gate["name"]: gate for gate in report["aggregate"]["gates"]}

    assert gates["trial_assessments_consistent"]["passed"] is False
    assert tampered["trial_id"] in gates["trial_assessments_consistent"]["observed"]
    assert (
        report["aggregate"]["single_support_holdout"]["by_side"]
        ["right_single_support"]["successes"]
        == 19
    )
    assert report["overall_passed"] is False

    for invalid in (True, 20.0, "20"):
        with pytest.raises(ValueError, match="randomized_trials_requested"):
            robustness.build_report(
                results,
                thresholds=thresholds,
                selected_scenarios=robustness.SCENARIOS,
                randomized_trials_requested=invalid,
                seed=robustness.CANONICAL_SEED,
                provenance=_provenance(),
            )
        with pytest.raises(ValueError, match="holdout"):
            robustness.build_report(
                results,
                thresholds=thresholds,
                selected_scenarios=robustness.SCENARIOS,
                randomized_trials_requested=robustness.DEFAULT_RANDOMIZED_TRIALS,
                single_support_holdout_trials_per_side_requested=invalid,
                seed=robustness.CANONICAL_SEED,
                provenance=_provenance(),
            )
        with pytest.raises(ValueError, match="seed"):
            robustness.build_report(
                results,
                thresholds=thresholds,
                selected_scenarios=robustness.SCENARIOS,
                randomized_trials_requested=robustness.DEFAULT_RANDOMIZED_TRIALS,
                seed=invalid,
                provenance=_provenance(),
            )


def test_model_variation_is_seeded_bounded_and_keeps_mass_inertia_scaling_physical() -> None:
    with HumanoidSimulation("free") as simulation:
        first = robustness.sample_model_variation(
            simulation, seed=9876, randomized=True
        )
        second = robustness.sample_model_variation(
            simulation, seed=9876, randomized=True
        )
        assert first == second
        assert first["realization_sha256"] == second["realization_sha256"]

        mass_before = simulation.model.body_mass.copy()
        inertia_before = simulation.model.body_inertia.copy()
        actuator_force_before = simulation.model.actuator_forcerange.copy()
        actuator_gain_before = simulation.model.actuator_gainprm[:, 0].copy()
        actuator_position_bias_before = simulation.model.actuator_biasprm[:, 1].copy()
        actuator_velocity_bias_before = simulation.model.actuator_biasprm[:, 2].copy()
        dof_damping_before = simulation.model.dof_damping.copy()
        right_foot_id = mujoco.mj_name2id(
            simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_rl_geom"
        )
        friction_before = simulation.model.geom_friction[right_foot_id].copy()
        robustness._apply_model_variation(simulation, first)

        factors = first["body_mass_inertia_factor"]
        assert isinstance(factors, dict)
        for body_id in range(1, simulation.model.nbody):
            name = mujoco.mj_id2name(
                simulation.model, mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if name is None or mass_before[body_id] <= 0.0:
                continue
            factor = float(factors[name])
            assert 0.90 <= factor <= 1.10
            assert simulation.model.body_mass[body_id] == pytest.approx(
                mass_before[body_id] * factor
            )
            np.testing.assert_allclose(
                simulation.model.body_inertia[body_id],
                inertia_before[body_id] * factor,
            )
        evidence = robustness._model_evidence(simulation)
        assert evidence["equality_constraint_count"] == 0
        assert evidence["base_joint_type"] == "free"
        assert evidence["scene_path"] == robustness.CANONICAL_SCENE_PATH
        assert evidence["base_joint_id"] == robustness.CANONICAL_BASE_JOINT_ID
        assert evidence["actuator_count"] == robustness.CANONICAL_ACTUATOR_COUNT
        assert evidence["joint_count"] == robustness.CANONICAL_JOINT_COUNT
        assert evidence["positive_mass_body_names"] == list(
            robustness.CANONICAL_POSITIVE_MASS_BODY_NAMES
        )
        assert (
            evidence["state_mutation_policy"]
            == robustness.CANONICAL_STATE_MUTATION_POLICY
        )
        assert evidence["actuator_names"] == list(robustness.JOINT_NAMES)
        np.testing.assert_allclose(
            simulation.model.geom_friction[right_foot_id] / friction_before,
            first["foot_ground_friction_factor"],
        )
        for actuator_id, actuator_name in enumerate(robustness.JOINT_NAMES):
            np.testing.assert_allclose(
                simulation.model.actuator_forcerange[actuator_id]
                / actuator_force_before[actuator_id],
                first["actuator_strength_factor"][actuator_name],
            )
            assert simulation.model.actuator_gainprm[actuator_id, 0] / actuator_gain_before[
                actuator_id
            ] == pytest.approx(first["actuator_kp_factor"][actuator_name])
            assert (
                simulation.model.actuator_biasprm[actuator_id, 1]
                / actuator_position_bias_before[actuator_id]
            ) == pytest.approx(first["actuator_kp_factor"][actuator_name])
            assert (
                simulation.model.actuator_biasprm[actuator_id, 2]
                / actuator_velocity_bias_before[actuator_id]
            ) == pytest.approx(first["actuator_kv_factor"][actuator_name])
            joint_id = int(simulation.model.actuator_trnid[actuator_id, 0])
            dof_id = int(simulation.model.jnt_dofadr[joint_id])
            assert simulation.model.dof_damping[dof_id] / dof_damping_before[
                dof_id
            ] == pytest.approx(first["joint_damping_factor"][actuator_name])


def test_fake_suite_passes_all_explicit_trial_and_aggregate_gates() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenarios = robustness.default_scenarios()
    trials = robustness.build_trial_specs(
        randomized_trials=20,
        seed=robustness.CANONICAL_SEED,
        scenarios=scenarios,
    )
    results = robustness.evaluate_suite(
        trials, thresholds=thresholds, evaluator=_passing_evaluation
    )
    report = robustness.build_report(
        results,
        thresholds=thresholds,
        selected_scenarios=scenarios,
        randomized_trials_requested=20,
        seed=robustness.CANONICAL_SEED,
        provenance=_provenance(),
    )

    assert all(result["passed"] for result in results)
    assert report["overall_passed"] is True
    assert report["aggregate"]["randomized"]["pass_rate"] == 1.0
    low, high = report["aggregate"]["randomized"]["wilson_95_ci"]
    assert 0.80 < low < high == 1.0
    assert all(gate["passed"] for gate in report["aggregate"]["gates"])


def test_noncanonical_seed_cannot_claim_canonical_acceptance() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenarios = robustness.default_scenarios()
    seed = robustness.CANONICAL_SEED + 1
    trials = robustness.build_trial_specs(
        randomized_trials=20, seed=seed, scenarios=scenarios
    )
    results = robustness.evaluate_suite(
        trials, thresholds=thresholds, evaluator=_passing_evaluation
    )
    report = robustness.build_report(
        results,
        thresholds=thresholds,
        selected_scenarios=scenarios,
        randomized_trials_requested=20,
        seed=seed,
        provenance=_provenance(),
    )

    gates = {gate["name"]: gate for gate in report["aggregate"]["gates"]}
    assert gates["canonical_seed"]["passed"] is False
    assert gates["exact_ordered_trial_contracts"]["passed"] is False
    assert report["overall_passed"] is False


def test_aggregate_rejects_weaker_thresholds_and_noncanonical_scenario_contracts() -> None:
    scenarios = robustness.default_scenarios()
    seed = robustness.CANONICAL_SEED
    trials = robustness.build_trial_specs(
        randomized_trials=20, seed=seed, scenarios=scenarios
    )

    weaker = replace(robustness.RobustnessThresholds(), maximum_tilt_deg=90.0)
    weaker_results = robustness.evaluate_suite(
        trials, thresholds=weaker, evaluator=_passing_evaluation
    )
    weaker_report = robustness.build_report(
        weaker_results,
        thresholds=weaker,
        selected_scenarios=scenarios,
        randomized_trials_requested=20,
        seed=seed,
        provenance=_provenance(),
    )
    weaker_gates = {
        gate["name"]: gate for gate in weaker_report["aggregate"]["gates"]
    }
    assert weaker_gates["canonical_or_stricter_thresholds"]["passed"] is False
    assert weaker_report["overall_passed"] is False

    invalid = replace(robustness.RobustnessThresholds(), maximum_tilt_deg=math.nan)
    assert "maximum_tilt_deg" in robustness._threshold_policy_violations(invalid)

    stricter = replace(robustness.RobustnessThresholds(), maximum_tilt_deg=19.0)
    stricter_results = robustness.evaluate_suite(
        trials, thresholds=stricter, evaluator=_passing_evaluation
    )
    stricter_report = robustness.build_report(
        stricter_results,
        thresholds=stricter,
        selected_scenarios=scenarios,
        randomized_trials_requested=20,
        seed=seed,
        provenance=_provenance(),
    )
    assert stricter_report["overall_passed"] is True

    modified = (
        replace(scenarios[0], active_duration_s=0.1),
        *scenarios[1:],
    )
    modified_trials = robustness.build_trial_specs(
        randomized_trials=20, seed=seed, scenarios=modified
    )
    modified_results = robustness.evaluate_suite(
        modified_trials,
        thresholds=robustness.RobustnessThresholds(),
        evaluator=_passing_evaluation,
    )
    modified_report = robustness.build_report(
        modified_results,
        thresholds=robustness.RobustnessThresholds(),
        selected_scenarios=modified,
        randomized_trials_requested=20,
        seed=seed,
        provenance=_provenance(),
    )
    modified_gates = {
        gate["name"]: gate for gate in modified_report["aggregate"]["gates"]
    }
    assert modified_gates["exact_canonical_scenario_contracts"]["passed"] is False
    assert modified_report["overall_passed"] is False


def test_runtime_hashing_fails_closed_when_an_input_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing-runtime-input.py"

    with pytest.raises(FileNotFoundError, match="runtime input is missing"):
        robustness._sha256(missing)


def test_aggregate_requires_all_twenty_for_wilson_lower_bound_and_every_critical() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenarios = robustness.default_scenarios()
    trials = robustness.build_trial_specs(
        randomized_trials=20,
        seed=robustness.CANONICAL_SEED,
        scenarios=scenarios,
    )
    results = robustness.evaluate_suite(
        trials, thresholds=thresholds, evaluator=_passing_evaluation
    )
    randomized = [result for result in results if result["randomized"]]
    randomized[0]["passed"] = False
    report = robustness.build_report(
        results,
        thresholds=thresholds,
        selected_scenarios=scenarios,
        randomized_trials_requested=20,
        seed=robustness.CANONICAL_SEED,
        provenance=_provenance(),
    )
    assert report["aggregate"]["randomized"]["pass_rate"] == pytest.approx(0.95)
    assert report["aggregate"]["randomized"]["wilson_95_ci"][0] < 0.80
    assert report["overall_passed"] is False
    wilson_gate = next(
        gate
        for gate in report["aggregate"]["gates"]
        if gate["name"] == "randomized_wilson_95_lower_bound"
    )
    assert wilson_gate["passed"] is False

    randomized[0]["passed"] = True
    report = robustness.build_report(
        results,
        thresholds=thresholds,
        selected_scenarios=scenarios,
        randomized_trials_requested=20,
        seed=robustness.CANONICAL_SEED,
        provenance=_provenance(),
    )
    assert report["overall_passed"] is True

    critical = next(result for result in results if result["critical"])
    critical["passed"] = False
    report = robustness.build_report(
        results,
        thresholds=thresholds,
        selected_scenarios=scenarios,
        randomized_trials_requested=20,
        seed=robustness.CANONICAL_SEED,
        provenance=_provenance(),
    )
    assert report["overall_passed"] is False


def test_one_random_trial_and_reordered_matrix_cannot_claim_canonical_acceptance() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenarios = robustness.default_scenarios()
    short_trials = robustness.build_trial_specs(
        randomized_trials=1, seed=71, scenarios=scenarios
    )
    short_results = robustness.evaluate_suite(
        short_trials, thresholds=thresholds, evaluator=_passing_evaluation
    )
    short_report = robustness.build_report(
        short_results,
        thresholds=thresholds,
        selected_scenarios=scenarios,
        randomized_trials_requested=1,
        seed=71,
        provenance=_provenance(),
    )
    assert short_report["overall_passed"] is False
    short_gates = {gate["name"]: gate for gate in short_report["aggregate"]["gates"]}
    assert short_gates["randomized_trial_count"]["passed"] is False
    assert short_gates["randomized_scenario_coverage"]["passed"] is False
    assert short_gates["randomized_wilson_95_lower_bound"]["passed"] is False

    full_trials = robustness.build_trial_specs(
        randomized_trials=20, seed=71, scenarios=scenarios
    )
    full_results = robustness.evaluate_suite(
        full_trials, thresholds=thresholds, evaluator=_passing_evaluation
    )
    critical = [result for result in full_results if result["critical"]]
    randomized = [result for result in full_results if result["randomized"]]
    reordered_report = robustness.build_report(
        [*critical, *reversed(randomized)],
        thresholds=thresholds,
        selected_scenarios=scenarios,
        randomized_trials_requested=20,
        seed=71,
        provenance=_provenance(),
    )
    reordered_gates = {
        gate["name"]: gate for gate in reordered_report["aggregate"]["gates"]
    }
    assert reordered_gates["randomized_scenario_coverage"]["passed"] is True
    assert reordered_gates["randomized_matrix_exact"]["passed"] is False
    assert reordered_report["overall_passed"] is False


def test_reduced_single_support_holdout_cannot_claim_acceptance() -> None:
    thresholds = robustness.RobustnessThresholds()
    trials = robustness.build_trial_specs(
        randomized_trials=robustness.DEFAULT_RANDOMIZED_TRIALS,
        single_support_holdout_trials_per_side=19,
        seed=robustness.CANONICAL_SEED,
        scenarios=robustness.SCENARIOS,
    )
    results = robustness.evaluate_suite(
        trials, thresholds=thresholds, evaluator=_passing_evaluation
    )
    report = robustness.build_report(
        results,
        thresholds=thresholds,
        selected_scenarios=robustness.SCENARIOS,
        randomized_trials_requested=robustness.DEFAULT_RANDOMIZED_TRIALS,
        single_support_holdout_trials_per_side_requested=19,
        seed=robustness.CANONICAL_SEED,
        provenance=_provenance(),
    )
    gates = {gate["name"]: gate for gate in report["aggregate"]["gates"]}

    assert gates["single_support_holdout_trial_count"]["passed"] is False
    assert gates["single_support_holdout_matrix_exact"]["passed"] is False
    assert report["overall_passed"] is False


def test_incomplete_scenario_selection_cannot_create_passing_acceptance() -> None:
    thresholds = robustness.RobustnessThresholds()
    selected = (robustness.SCENARIO_BY_NAME["neutral_settle"],)
    trials = robustness.build_trial_specs(
        randomized_trials=20, seed=10, scenarios=selected
    )
    results = robustness.evaluate_suite(
        trials, thresholds=thresholds, evaluator=_passing_evaluation
    )
    report = robustness.build_report(
        results,
        thresholds=thresholds,
        selected_scenarios=selected,
        randomized_trials_requested=20,
        seed=10,
        provenance=_provenance(),
    )

    assert report["overall_passed"] is False
    coverage_gate = next(
        gate
        for gate in report["aggregate"]["gates"]
        if gate["name"] == "critical_scenarios_complete"
    )
    assert coverage_gate["passed"] is False
    assert "push_sagittal_negative" in coverage_gate["observed"]["missing"]


def test_one_leg_requires_ordered_side_specific_cycle_no_abort_clearance_and_motion() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenario = robustness.SCENARIO_BY_NAME["right_single_support"]
    trial = robustness.TrialSpec(scenario, 33, False, True, 0)
    passing = robustness.assess_trial(
        trial, _passing_evaluation(trial, thresholds), thresholds
    )
    assert passing["passed"] is True

    evaluation = _passing_evaluation(trial, thresholds)
    support = evaluation["metrics"]["support"]
    assert isinstance(support, dict)
    support["phase_sequence"] = ["double_support"]
    support["phase_events"] = support["phase_events"][:1]
    support["abort_reasons"] = ["stance_load_timeout"]
    support["maximum_right_foot_clearance_over_left_m"] = 0.0
    support["maximum_left_foot_clearance_over_right_m"] = 0.04
    support["hold_observations"] = 0
    support["minimum_hold_stance_force_n"] = 0.0
    support["minimum_hold_stance_load_fraction"] = 0.4
    support["maximum_hold_swing_load_fraction"] = 0.6
    motion = evaluation["metrics"]["motion"]
    assert isinstance(motion, dict)
    for source in ("motor_command", "actual_joint"):
        side = motion[source]["right_leg"]
        side["joint_span_deg"][2:5] = [0.0, 0.0, 0.0]

    result = robustness.assess_trial(trial, evaluation, thresholds)
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert {
        "support_phase_sequence_and_side",
        "support_abort_reasons",
        "swing_foot_clearance",
        "swing_foot_side_dominance",
        "single_support_hold_observations",
        "single_support_hold_stance_force",
        "single_support_hold_stance_load_fraction",
        "single_support_hold_swing_load_fraction",
        "motor_command_swing_leg_joint_coverage",
        "actual_joint_swing_leg_joint_coverage",
    } <= failed


def test_one_leg_gates_support_frame_and_recovery_not_required_world_transfer() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenario = robustness.SCENARIO_BY_NAME["right_single_support"]
    trial = robustness.TrialSpec(scenario, 34, False, True, 0)

    world_transfer = _passing_evaluation(trial, thresholds)
    metrics = world_transfer["metrics"]
    assert isinstance(metrics, dict)
    metrics["maximum_horizontal_drift_m"] = 0.28
    result = robustness.assess_trial(trial, world_transfer, thresholds)
    assert result["passed"] is True
    descriptive = next(
        gate
        for gate in result["gates"]
        if gate["name"] == "maximum_horizontal_drift_m_descriptive"
    )
    assert descriptive["passed"] is True
    assert descriptive["observed"] == pytest.approx(0.28)

    support_cases = (
        (
            "support_frame_observations",
            0,
            "single_support_support_frame_observations",
        ),
        (
            "maximum_support_frame_com_to_stance_x_error_m",
            thresholds.maximum_single_support_com_to_stance_x_error_m + 1e-6,
            "single_support_com_to_stance_x_error_m",
        ),
        (
            "maximum_support_frame_com_to_stance_y_error_m",
            thresholds.maximum_single_support_com_to_stance_y_error_m + 1e-6,
            "single_support_com_to_stance_y_error_m",
        ),
        (
            "maximum_support_frame_stance_foot_travel_m",
            thresholds.maximum_single_support_stance_foot_travel_m + 1e-6,
            "single_support_stance_foot_travel_m",
        ),
    )
    for field, observed, expected_gate in support_cases:
        evaluation = _passing_evaluation(trial, thresholds)
        support = evaluation["metrics"]["support"]
        assert isinstance(support, dict)
        support[field] = observed
        failed = {
            gate["name"]
            for gate in robustness.assess_trial(
                trial, evaluation, thresholds
            )["gates"]
            if not gate["passed"]
        }
        assert expected_gate in failed

    final_drift = _passing_evaluation(trial, thresholds)
    final_metrics = final_drift["metrics"]
    assert isinstance(final_metrics, dict)
    final_metrics["final_horizontal_drift_m"] = (
        thresholds.maximum_single_support_final_base_drift_m + 1e-6
    )
    failed = {
        gate["name"]
        for gate in robustness.assess_trial(
            trial, final_drift, thresholds
        )["gates"]
        if not gate["passed"]
    }
    assert "single_support_final_base_drift_m" in failed

    recovery_drift = _passing_evaluation(trial, thresholds)
    recovery = recovery_drift["metrics"]["recovery"]
    assert isinstance(recovery, dict)
    recovery["maximum_horizontal_drift_m"] = (
        thresholds.maximum_single_support_final_base_drift_m + 1e-6
    )
    failed = {
        gate["name"]
        for gate in robustness.assess_trial(
            trial, recovery_drift, thresholds
        )["gates"]
        if not gate["passed"]
    }
    assert "single_support_recovery_base_drift_m" in failed


def test_one_leg_rejects_missing_or_excessive_swing_contact_evidence() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenario = robustness.SCENARIO_BY_NAME["left_single_support"]
    trial = robustness.TrialSpec(scenario, 35, False, True, 0)
    cases = (
        ("swing_contact_episodes", 0, "swing_contact_episode_observed"),
        (
            "maximum_swing_foot_precontact_vertical_speed_m_s",
            thresholds.maximum_swing_precontact_vertical_speed_m_s + 1e-6,
            "swing_precontact_vertical_speed",
        ),
        (
            "maximum_swing_foot_impact_force_n",
            thresholds.maximum_swing_impact_force_n + 1e-6,
            "swing_impact_force",
        ),
        (
            "maximum_swing_foot_contact_impulse_n_s",
            thresholds.maximum_swing_contact_impulse_n_s + 1e-6,
            "swing_contact_impulse",
        ),
    )

    for field, observed, expected_gate in cases:
        evaluation = _passing_evaluation(trial, thresholds)
        support = evaluation["metrics"]["support"]
        assert isinstance(support, dict)
        support[field] = observed
        failed = {
            gate["name"]
            for gate in robustness.assess_trial(
                trial, evaluation, thresholds
            )["gates"]
            if not gate["passed"]
        }
        assert expected_gate in failed


def test_phase_aware_slip_excludes_swing_impact_but_preserves_impact_telemetry() -> None:
    phase = robustness.SupportPhase.HOLD_SWING
    intent = robustness.SupportIntent.RIGHT_SWING
    forces = np.asarray([10.0, 20.0])
    speeds = np.asarray([0.40, 0.02])
    stance_mask = robustness._phase_aware_loaded_mask(forces, phase, intent)

    assert stance_mask.tolist() == [False, True]
    assert float(np.max(speeds[stance_mask])) == pytest.approx(0.02)
    assert robustness._phase_aware_loaded_mask(
        forces, robustness.SupportPhase.CENTER_WEIGHT, intent
    ).tolist() == [True, True]
    assert robustness._phase_aware_loaded_mask(
        forces, robustness.SupportPhase.SHIFT_WEIGHT, intent
    ).tolist() == [True, True]
    assert robustness._phase_aware_loaded_mask(
        forces, robustness.SupportPhase.VERIFY_STANCE, intent
    ).tolist() == [True, True]

    telemetry = robustness._SwingContactTelemetry()
    telemetry.update(
        phase=phase,
        active_intent=intent,
        right_force_n=0.0,
        left_force_n=20.0,
        right_velocity_m_s=np.asarray([0.40, 0.0, -0.20]),
        left_velocity_m_s=np.asarray([0.02, 0.0, 0.0]),
        dt_s=0.002,
    )
    telemetry.update(
        phase=phase,
        active_intent=intent,
        right_force_n=10.0,
        left_force_n=20.0,
        right_velocity_m_s=np.asarray([0.40, 0.0, -0.10]),
        left_velocity_m_s=np.asarray([0.02, 0.0, 0.0]),
        dt_s=0.002,
    )

    assert telemetry.episodes == 1
    assert telemetry.maximum_impact_speed_m_s == pytest.approx(0.40)
    assert telemetry.maximum_impact_force_n == pytest.approx(10.0)
    assert telemetry.maximum_contact_impulse_n_s == pytest.approx(0.02)

    telemetry.update(
        phase=robustness.SupportPhase.CENTER_WEIGHT,
        active_intent=intent,
        right_force_n=20.0,
        left_force_n=20.0,
        right_velocity_m_s=np.zeros(3),
        left_velocity_m_s=np.zeros(3),
        dt_s=0.5,
    )
    assert telemetry.maximum_contact_impulse_n_s == pytest.approx(0.02)
    assert not telemetry.right.contact_active


def test_swing_telemetry_updates_precontact_speed_after_contact_bounce() -> None:
    telemetry = robustness._SwingContactTelemetry()
    phase = robustness.SupportPhase.LOWER_SWING
    intent = robustness.SupportIntent.RIGHT_SWING

    for force_n, vertical_speed in (
        (0.0, -0.1),
        (1.0, -0.1),
        (0.0, -2.0),
        (20.0, -2.0),
    ):
        telemetry.update(
            phase=phase,
            active_intent=intent,
            right_force_n=force_n,
            left_force_n=20.0,
            right_velocity_m_s=np.asarray([0.0, 0.0, vertical_speed]),
            left_velocity_m_s=np.zeros(3),
            dt_s=0.01,
        )

    assert telemetry.episodes == 1
    assert telemetry.maximum_precontact_vertical_speed_m_s == pytest.approx(2.0)
    assert telemetry.maximum_impact_force_n == pytest.approx(20.0)


def test_single_support_com_limit_stays_inside_physical_sole() -> None:
    thresholds = robustness.RobustnessThresholds()

    assert thresholds.maximum_single_support_com_to_stance_x_error_m == pytest.approx(
        0.090
    )
    assert thresholds.maximum_single_support_com_to_stance_x_error_m < 0.105
    assert thresholds.maximum_single_support_com_to_stance_y_error_m == pytest.approx(
        0.065
    )
    assert thresholds.maximum_single_support_com_to_stance_y_error_m < 0.066


def test_recovery_must_be_contiguous_final_full_window_and_include_angular_speed() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenario = robustness.SCENARIO_BY_NAME["neutral_settle"]
    trial = robustness.TrialSpec(scenario, 40, False, True, 0)
    evaluation = _passing_evaluation(trial, thresholds)
    metrics = evaluation["metrics"]
    recovery = metrics["recovery"]
    assert isinstance(metrics, dict)
    assert isinstance(recovery, dict)
    metrics["simulated_duration_s"] = 0.0
    recovery.update(
        {
            "observation_s": 0.0,
            "window_start_time_s": 0.0,
            "window_end_time_s": 0.0,
            "expected_samples": 1,
            "samples": 1,
            "first_sample_time_s": 0.0,
            "last_sample_time_s": 0.0,
            "maximum_sample_gap_s": 1.0,
            "maximum_base_angular_speed_rad_s": 999.0,
        }
    )

    result = robustness.assess_trial(trial, evaluation, thresholds)
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert {
        "simulated_duration_s",
        "recovery_continuous_final_interval",
        "recovery_samples",
        "recovery_maximum_base_angular_speed_rad_s",
    } <= failed


def test_variation_must_replay_exactly_from_seed_names_flags_bounds_and_hash() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenario = robustness.SCENARIO_BY_NAME["neutral_settle"]
    trial = robustness.TrialSpec(scenario, 12345, True, False, 0)
    evaluation = _passing_evaluation(trial, thresholds)
    parameters = evaluation["parameters"]
    assert isinstance(parameters, dict)
    parameters["seed"] = 99
    kp = parameters["actuator_kp_factor"]
    assert isinstance(kp, dict)
    kp[robustness.JOINT_NAMES[0]] = 1.0
    parameters["realization_sha256"] = robustness._variation_realization_sha256(
        parameters
    )

    result = robustness.assess_trial(trial, evaluation, thresholds)
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert "variation_seed" in failed
    assert "variation_seed_replay" in failed
    assert "variation_realization_sha256" not in failed
    assert "variation_factor_bounds" not in failed


def test_push_requires_matching_force_and_torque_peaks_and_signed_impulses() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenario = robustness.SCENARIO_BY_NAME["push_sagittal_positive"]
    trial = robustness.TrialSpec(scenario, 8, False, True, 0)
    evaluation = _passing_evaluation(trial, thresholds)
    perturbation = evaluation["metrics"]["perturbation"]
    assert isinstance(perturbation, dict)
    perturbation["peak_torque_abs_n_m"] = [0.0, 0.0, 0.0]
    perturbation["signed_torque_impulse_n_m_s"] = [0.0, 0.0, 0.0]

    result = robustness.assess_trial(trial, evaluation, thresholds)
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert "finite_signed_perturbation" in failed


def test_non_push_scenario_rejects_undeclared_force_or_torque() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenario = robustness.SCENARIO_BY_NAME["neutral_settle"]
    trial = robustness.TrialSpec(scenario, 9, False, True, 0)
    evaluation = _passing_evaluation(trial, thresholds)
    perturbation = evaluation["metrics"]["perturbation"]
    assert isinstance(perturbation, dict)
    perturbation["peak_force_abs_n"] = [1.0, 0.0, 0.0]

    result = robustness.assess_trial(trial, evaluation, thresholds)
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert "no_undeclared_perturbation" in failed


def test_excitation_requires_every_declared_joint_and_signed_crouch_direction() -> None:
    thresholds = robustness.RobustnessThresholds()
    upper_scenario = robustness.SCENARIO_BY_NAME["combined_upper_body_slow"]
    upper_trial = robustness.TrialSpec(upper_scenario, 15, False, True, 0)
    upper = _passing_evaluation(upper_trial, thresholds)
    upper_motion = upper["metrics"]["motion"]
    assert isinstance(upper_motion, dict)
    for source in ("motor_command", "actual_joint"):
        upper_motion[source]["arms"]["joint_span_deg"] = [30.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    upper_result = robustness.assess_trial(upper_trial, upper, thresholds)
    upper_failed = {
        gate["name"] for gate in upper_result["gates"] if not gate["passed"]
    }
    assert "motor_command_arms_joint_coverage" in upper_failed
    assert "actual_joint_arms_joint_coverage" in upper_failed

    crouch_scenario = robustness.SCENARIO_BY_NAME["crouch_positive"]
    crouch_trial = robustness.TrialSpec(crouch_scenario, 16, False, True, 0)
    crouch = _passing_evaluation(crouch_trial, thresholds)
    crouch_motion = crouch["metrics"]["motion"]
    assert isinstance(crouch_motion, dict)
    for source in ("motor_command", "actual_joint"):
        crouch_motion[source]["legs"]["maximum_excursion_deg"][4:8] = [
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    crouch_result = robustness.assess_trial(crouch_trial, crouch, thresholds)
    crouch_failed = {
        gate["name"] for gate in crouch_result["gates"] if not gate["passed"]
    }
    assert "motor_command_crouch_signed_excursion" in crouch_failed
    assert "actual_joint_crouch_signed_excursion" in crouch_failed


def test_trial_assessment_rejects_fixed_base_fall_nonfinite_and_missing_excitation() -> None:
    thresholds = robustness.RobustnessThresholds()
    scenario = robustness.SCENARIO_BY_NAME["push_lateral_negative"]
    trial = robustness.TrialSpec(scenario, 1, False, True, 0)
    evaluation = _passing_evaluation(trial, thresholds)
    model = evaluation["model"]
    metrics = evaluation["metrics"]
    assert isinstance(model, dict)
    assert isinstance(metrics, dict)
    model.update({"scene": "fixed", "equality_constraint_count": 1, "base_joint_type": "slide"})
    metrics.update(
        {
            "duration_scale": 0.1,
            "completed_duration": False,
            "fell": True,
            "minimum_base_height_m": 0.4,
            "maximum_tilt_deg": 70.0,
            "maximum_horizontal_drift_m": math.nan,
            "maximum_loaded_foot_slip_speed_m_s": 0.5,
            "right_loaded_foot_slip_distance_m": 0.3,
            "left_loaded_foot_slip_distance_m": 0.3,
            "maximum_non_foot_ground_contacts": 3,
            "final_base_height_m": 0.4,
        }
    )
    perturbation = metrics["perturbation"]
    assert isinstance(perturbation, dict)
    perturbation["signed_force_impulse_n_s"] = [0.0, 0.0, 0.0]
    recovery = metrics["recovery"]
    assert isinstance(recovery, dict)
    recovery.update(
        {
            "samples": 0,
            "minimum_base_height_m": 0.4,
            "maximum_tilt_deg": 30.0,
            "maximum_base_speed_m_s": 1.0,
            "maximum_base_angular_speed_rad_s": 3.0,
            "maximum_joint_speed_rad_s": 4.0,
            "maximum_tracking_error_rad": 1.0,
            "maximum_loaded_foot_slip_speed_m_s": 0.5,
            "maximum_capture_point_error_m": 0.5,
            "minimum_right_foot_force_n": 0.0,
            "minimum_left_foot_force_n": 0.0,
            "minimum_total_support_force_n": 0.0,
        }
    )

    result = robustness.assess_trial(trial, evaluation, thresholds)
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}

    assert result["passed"] is False
    assert {
        "scene",
        "equality_constraint_count",
        "base_joint_type",
        "duration_scale",
        "completed_duration",
        "fell",
        "minimum_base_height_m",
        "maximum_tilt_deg",
        "maximum_horizontal_drift_m",
        "maximum_loaded_foot_slip_speed_m_s",
        "right_loaded_foot_slip_distance_m",
        "left_loaded_foot_slip_distance_m",
        "maximum_non_foot_ground_contacts",
        "final_base_height_m",
        "recovery_samples",
        "recovery_continuous_final_interval",
        "recovery_maximum_base_angular_speed_rad_s",
        "recovery_maximum_capture_point_error_m",
        "recovery_minimum_right_foot_force_n",
        "recovery_minimum_left_foot_force_n",
        "recovery_minimum_total_support_force_n",
        "finite_signed_perturbation",
        "finite_numeric_metrics",
    } <= failed
    sanitized = robustness._sanitize_json(result)
    assert sanitized["metrics"]["maximum_horizontal_drift_m"] is None
    json.dumps(sanitized, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "expected_gate"),
    (
        ("scene_path", "models/humanoid/not-canonical.xml", "canonical_scene_path"),
        ("base_joint_id", 1, "canonical_base_joint_id"),
        ("actuator_count", 19, "canonical_actuator_count"),
        ("joint_count", 20, "canonical_joint_count"),
        ("positive_mass_body_names", ["torso"], "canonical_positive_mass_bodies"),
        ("state_mutation_policy", "unverified", "state_mutation_policy"),
    ),
)
def test_trial_assessment_binds_exact_canonical_model_contract(
    field: str,
    value: object,
    expected_gate: str,
) -> None:
    thresholds = robustness.RobustnessThresholds()
    scenario = robustness.SCENARIO_BY_NAME["neutral_settle"]
    trial = robustness.TrialSpec(scenario, 91, False, True, 0)
    evaluation = _passing_evaluation(trial, thresholds)
    model = evaluation["model"]
    assert isinstance(model, dict)
    model[field] = value

    result = robustness.assess_trial(trial, evaluation, thresholds)
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}

    assert expected_gate in failed
    assert result["passed"] is False


def test_evaluator_exception_is_retained_as_trial_failure() -> None:
    scenario = robustness.SCENARIO_BY_NAME["neutral_settle"]
    trial = robustness.TrialSpec(scenario, 5, True, False, 0)

    def failing(
        _trial: robustness.TrialSpec,
        _thresholds: robustness.RobustnessThresholds,
    ) -> dict[str, object]:
        raise RuntimeError("synthetic robustness failure")

    result = robustness.evaluate_suite(
        [trial], thresholds=robustness.RobustnessThresholds(), evaluator=failing
    )[0]

    assert result["passed"] is False
    assert "synthetic robustness failure" in result["error"]
    assert result["gates"] == [
        {
            "name": "evaluation_completed",
            "passed": False,
            "observed": False,
            "required": True,
        }
    ]


def test_wilson_interval_and_validation() -> None:
    low, high = robustness.wilson_interval(19, 20)
    assert low == pytest.approx(0.7638688066)
    assert high == pytest.approx(0.9911185512)
    assert robustness.wilson_interval(0, 0) == (None, None)
    with pytest.raises(ValueError, match="0 <="):
        robustness.wilson_interval(2, 1)


def test_main_writes_strict_json_with_fake_evaluator(tmp_path: Path) -> None:
    output = tmp_path / "robustness.json"

    exit_code = robustness.main(
        ["--randomized-trials", "20", "--output", str(output)],
        evaluator=_passing_evaluation,
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["overall_passed"] is True
    assert report["configuration"]["qpos_or_base_repair_after_initial_reset"] is False
    assert report["aggregate"]["randomized"]["trials"] == 20
    assert report["aggregate"]["single_support_holdout"]["trials"] == 40
    assert report["schema_version"] == 3
    assert "tools/evaluate_freebase_robustness.py" in report["runtime_input_sha256"]
    assert "config/balance.yaml" in report["runtime_input_sha256"]
    assert "models/humanoid/robot.xml" in report["runtime_input_sha256"]
    assert (
        "src/robot_human_interface/skeleton/types.py"
        in report["runtime_input_sha256"]
    )
    assert report["runtime_inputs_unchanged_during_run"] is True


def test_real_freebase_smoke_uses_real_model_and_cannot_claim_full_acceptance() -> None:
    scenario = robustness.SCENARIO_BY_NAME["neutral_settle"]
    trial = robustness.TrialSpec(scenario, 77, False, True, 0)
    thresholds = robustness.RobustnessThresholds()

    evaluation = robustness.run_trial(trial, thresholds, duration_scale=0.02)
    result = robustness.assess_trial(trial, evaluation, thresholds)

    assert evaluation["model"]["scene"] == "free"
    assert evaluation["model"]["equality_constraint_count"] == 0
    assert evaluation["model"]["base_joint_type"] == "free"
    assert "no qpos/base repair" in evaluation["model"]["state_mutation_policy"]
    assert evaluation["metrics"]["simulated_duration_s"] > 0.0
    assert math.isfinite(evaluation["metrics"]["minimum_base_height_m"])
    assert result["passed"] is False
    assert any(
        gate["name"] == "duration_scale" and not gate["passed"]
        for gate in result["gates"]
    )
