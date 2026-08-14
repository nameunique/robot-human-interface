from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_human_interface.pose import MediaPipePoseLandmarker, make_synthetic_skeleton
from robot_human_interface.control import (
    HumanSupportIntentEstimator,
    SupportControlConfig,
    SupportStateMachine,
)
from robot_human_interface.simulation import HumanoidSimulation
from robot_human_interface.skeleton import RobotJointCommand
from tools import evaluate_safe_pose_fidelity as fidelity


def _passing_trajectory() -> dict[str, object]:
    stage = {
        "amplitude_deg": 30.0,
        "amplitude_ratio": 0.75,
        "zero_lag_correlation": 0.85,
        "best_correlation": 0.90,
        "best_lag_s": 0.10,
    }
    return {
        "sample_count": 100,
        "duration_s": 3.3,
        "reliable": True,
        "reason": None,
        "human_amplitude_deg": 40.0,
        "safe_command": dict(stage),
        "actual_qpos": dict(stage),
    }


def _passing_evaluation(clip: fidelity.ClipSpec) -> dict[str, object]:
    groups = {
        name: {"count": 100, "mean_deg": 20.0, "p50_deg": 18.0, "p90_deg": 35.0}
        for name in fidelity.GROUPS
    }
    return {
        "clip": clip.name,
        "video": fidelity._project_path(clip.path.resolve()),
        "video_sha256": fidelity._sha256(clip.path.resolve()),
        "teleop_stats": {
            "base_mode": "free",
            "frames": clip.expected_frames,
            "fell": False,
            "maximum_non_foot_ground_contacts": 0,
            "support_abort_count": 0,
            "support_abort_reasons": [],
            "calibration": {
                "mode": (
                    "explicit_replay_frame"
                    if clip.calibration_video is not None
                    else "automatic_window"
                ),
                "source": (
                    fidelity._project_path(clip.calibration_video.resolve())
                    if clip.calibration_video is not None
                    else None
                ),
                "source_sha256": (
                    fidelity._sha256(clip.calibration_video.resolve())
                    if clip.calibration_video is not None
                    else None
                ),
                "frame_index": clip.calibration_frame,
            },
        },
        "metrics": {
            "coverage": {
                "source_frames": clip.expected_frames,
                "evaluated_source_fraction": 0.90,
                "directions": {
                    name: {
                        "valid_frames": 100,
                        "valid_frame_fraction": 0.95,
                    }
                    for name in fidelity.DIRECTION_NAMES
                },
                "groups": {
                    name: {
                        "frames_with_any_direction": 100,
                        "valid_direction_fraction": 0.95,
                    }
                    for name in fidelity.GROUPS
                },
            },
            "instrumentation": {
                "same_simulation": True,
                "equality_constraint_count": 0,
                "base_joint_type": "free",
            },
            "fidelity": {
                "safe_command": {"groups": groups},
                "actual_qpos": {"groups": groups},
            },
            "trajectories": {
                name: _passing_trajectory() for name in clip.semantic_channels
            },
        },
    }


def _valid_runtime_snapshot(
    clips: tuple[fidelity.ClipSpec, ...] | list[fidelity.ClipSpec],
) -> dict[str, str]:
    return {
        fidelity._project_path(path): "0" * 64
        for path in fidelity._runtime_input_paths(clips)
    }


def test_runtime_manifest_includes_imported_package_reexports() -> None:
    paths = fidelity._runtime_input_paths(fidelity.default_clips())

    for package in (
        "camera",
        "control",
        "pose",
        "retargeting",
        "simulation",
        "skeleton",
    ):
        package_init = (
            fidelity.PROJECT_ROOT
            / "src"
            / "robot_human_interface"
            / package
            / "__init__.py"
        )
        assert package_init in paths


def test_default_matrix_declares_scientific_task_contracts() -> None:
    clips = fidelity.default_clips()

    assert [clip.name for clip in clips] == [
        "slow-balance",
        "jumping-jacks",
        "arm-circles",
        "frontal-leg-swing",
        "stationary-squat",
        "trunk-circles",
    ]
    assert all(clip.path.is_file() for clip in clips)
    assert [clip.expected_frames for clip in clips] == [1961, 194, 796, 836, 817, 867]
    assert all(clip.semantic_channels for clip in clips)
    assert all(len(set(clip.semantic_channels)) == len(clip.semantic_channels) for clip in clips)
    assert all(clip.task_label.strip() for clip in clips)
    assert all(clip.capability.strip() for clip in clips)
    explicit = {
        clip.name: (clip.calibration_video.name, clip.calibration_frame)
        for clip in clips
        if clip.calibration_video is not None
    }
    assert explicit == {
        "jumping-jacks": ("jumping_jacks_demo.mp4", 2),
        "frontal-leg-swing": ("dvids_arm_circles.mp4", 29),
        "trunk-circles": ("dvids_arm_circles.mp4", 29),
    }
    event_contracts = {
        clip.name: clip.required_leg_event_sides
        for clip in clips
        if clip.required_leg_event_sides
    }
    assert event_contracts == {
        "slow-balance": ("right_swing", "left_swing"),
        "frontal-leg-swing": ("right_swing",),
    }
    assert clips[0].gated_p90_groups == ("arms", "head")
    assert clips[3].gated_p90_groups == ()
    assert clips[5].semantic_channels == ("head",)
    assert clips[5].gated_p90_groups == ("head",)
    assert "waist DOF" in clips[5].unsupported_limitations[0]

    # Unsupported flight is disclosed, but the supported grounded leg
    # projection remains an acceptance requirement rather than being removed
    # after observing a failing result.
    jumping = clips[1]
    assert jumping.semantic_channels == (
        "right_arm",
        "left_arm",
        "right_leg",
        "left_leg",
    )
    assert jumping.gated_p90_groups == ("arms", "legs")
    assert "remain acceptance gates" in jumping.unsupported_limitations[0]


def test_instrumentation_context_restores_every_production_method() -> None:
    original_estimate = MediaPipePoseLandmarker.estimate
    original_intent = HumanSupportIntentEstimator.update
    original_support = SupportStateMachine.update
    original_apply = HumanoidSimulation.apply_joint_command
    original_step = HumanoidSimulation.step

    with fidelity._record_end_to_end_trace():
        assert MediaPipePoseLandmarker.estimate is not original_estimate
        assert HumanSupportIntentEstimator.update is not original_intent
        assert SupportStateMachine.update is not original_support
        assert HumanoidSimulation.apply_joint_command is not original_apply
        assert HumanoidSimulation.step is not original_step
        assert "force_return_reason" in inspect.signature(
            SupportStateMachine.update
        ).parameters

    assert MediaPipePoseLandmarker.estimate is original_estimate
    assert HumanSupportIntentEstimator.update is original_intent
    assert SupportStateMachine.update is original_support
    assert HumanoidSimulation.apply_joint_command is original_apply
    assert HumanoidSimulation.step is original_step


def test_trace_uses_last_safe_and_actual_sample_for_each_input_frame() -> None:
    recorder = fidelity._TeleopTraceRecorder()
    skeleton = make_synthetic_skeleton(0.0)
    with HumanoidSimulation("free") as simulation:
        first = simulation.home_positions_rad.copy()
        second = first.copy()
        first[0] += 0.01
        second[0] += 0.02
        recorder.begin_frame(0.0)
        recorder.record_skeleton(skeleton)
        recorder.record_safe_command(
            simulation,
            RobotJointCommand.humanoid(0.0, first, 1.0),
            first,
        )
        recorder.record_state(simulation, simulation.reset(first))
        recorder.record_safe_command(
            simulation,
            RobotJointCommand.humanoid(0.0, second, 1.0),
            second,
        )
        recorder.record_state(simulation, simulation.reset(second))
        recorder.begin_frame(1.0 / 30.0)
        recorder.record_skeleton(None)
        recorder.finish_frame()

    assert len(recorder.samples) == 2
    np.testing.assert_allclose(recorder.samples[0].safe_positions_rad, second)
    np.testing.assert_allclose(recorder.samples[0].actual_positions_rad, second)
    assert np.isfinite(
        recorder.samples[0].safe_right_minus_left_foot_height_m
    )
    assert np.isfinite(
        recorder.samples[0].actual_right_minus_left_foot_height_m
    )
    assert recorder.samples[1].safe_positions_rad is None
    assert len(recorder.simulation_instances) == 1
    assert len(recorder.data_instances) == 1


def test_trace_analysis_reports_safe_and_actual_coverage_and_percentiles() -> None:
    recorder = fidelity._TeleopTraceRecorder()
    with HumanoidSimulation("free") as simulation:
        for index in range(40):
            timestamp_s = index / 30.0
            skeleton = make_synthetic_skeleton(
                timestamp_s,
                phase_rad=0.8 * np.sin(index / 8.0),
            )
            positions = simulation.home_positions_rad.copy()
            positions[0] += 0.15 * np.sin(index / 8.0)
            positions[1] += 0.12 * np.sin(index / 8.0)
            recorder.begin_frame(timestamp_s)
            recorder.record_skeleton(skeleton)
            recorder.record_safe_command(
                simulation,
                RobotJointCommand.humanoid(timestamp_s, positions, 1.0),
                positions,
            )
            recorder.record_state(simulation, simulation.reset(positions))
        recorder.finish_frame()

    metrics = fidelity.analyze_trace(
        recorder,
        policy=fidelity.FidelityGatePolicy(),
    )
    coverage = metrics["coverage"]
    assert coverage["source_frames"] == 40
    assert coverage["evaluated_nonstale_frames"] == 40
    assert coverage["evaluated_source_fraction"] == 1.0
    assert metrics["instrumentation"]["same_simulation"] is True
    assert metrics["instrumentation"]["equality_constraint_count"] == 0
    assert metrics["instrumentation"]["base_joint_type"] == "free"
    for stage in ("safe_command", "actual_qpos"):
        for group in fidelity.GROUPS:
            summary = metrics["fidelity"][stage]["groups"][group]
            assert summary["count"] == 40
            assert np.isfinite(summary["mean_deg"])
            assert np.isfinite(summary["p50_deg"])
            assert np.isfinite(summary["p90_deg"])


def test_trace_analysis_preserves_safe_vs_actual_stage_distinction() -> None:
    recorder = fidelity._TeleopTraceRecorder()
    with HumanoidSimulation("free") as simulation:
        safe_positions = simulation.home_positions_rad.copy()
        actual_positions = simulation.home_positions_rad.copy()
        safe_positions[0] += np.radians(35.0)
        actual_positions[0] -= np.radians(20.0)
        for index in range(35):
            timestamp_s = index / 30.0
            recorder.begin_frame(timestamp_s)
            recorder.record_skeleton(
                make_synthetic_skeleton(timestamp_s, phase_rad=np.pi / 2.0)
            )
            recorder.record_safe_command(
                simulation,
                RobotJointCommand.humanoid(timestamp_s, safe_positions, 1.0),
                safe_positions,
            )
            recorder.record_state(simulation, simulation.reset(actual_positions))
        recorder.finish_frame()

    metrics = fidelity.analyze_trace(
        recorder,
        policy=fidelity.FidelityGatePolicy(),
    )
    safe_error = metrics["fidelity"]["safe_command"]["directions"]["right_arm"][
        "mean_deg"
    ]
    actual_error = metrics["fidelity"]["actual_qpos"]["directions"]["right_arm"][
        "mean_deg"
    ]
    assert abs(safe_error - actual_error) > 5.0


def test_dominant_direction_trajectory_recovers_amplitude_correlation_and_lag() -> None:
    timestamps = np.linspace(0.0, 4.0, 121)
    period_s = 2.0

    def vectors(delay_s: float, scale: float) -> np.ndarray:
        angles = np.radians(
            scale * 30.0 * np.sin(2.0 * np.pi * (timestamps - delay_s) / period_s)
        )
        return np.column_stack(
            (np.sin(angles), np.zeros_like(angles), np.cos(angles))
        )

    result = fidelity._direction_trajectory_metrics(
        timestamps,
        vectors(0.0, 1.0),
        vectors(0.0, 0.8),
        vectors(0.2, 0.7),
        policy=fidelity.FidelityGatePolicy(),
    )

    assert result["reliable"] is True
    assert result["human_amplitude_deg"] > 50.0
    assert result["safe_command"]["amplitude_ratio"] == pytest.approx(0.8, abs=0.03)
    assert result["safe_command"]["best_correlation"] > 0.99
    assert result["actual_qpos"]["amplitude_ratio"] == pytest.approx(0.7, abs=0.03)
    assert result["actual_qpos"]["best_correlation"] > 0.99
    assert result["actual_qpos"]["best_lag_s"] == pytest.approx(0.2, abs=0.04)


def test_lag_search_extends_beyond_gate_and_exposes_late_tracking() -> None:
    timestamps = np.linspace(0.0, 8.0, 241)

    def vectors(delay_s: float) -> np.ndarray:
        angles = np.radians(
            30.0 * np.sin(2.0 * np.pi * (timestamps - delay_s) / 6.0)
        )
        return np.column_stack(
            (np.sin(angles), np.zeros_like(angles), np.cos(angles))
        )

    policy = fidelity.FidelityGatePolicy(
        maximum_actual_lag_s=0.5,
        lag_search_horizon_s=1.5,
    )
    late = fidelity._direction_trajectory_metrics(
        timestamps,
        vectors(0.0),
        vectors(0.0),
        vectors(0.9),
        policy=policy,
    )

    assert late["actual_qpos"]["best_correlation"] > 0.99
    assert late["actual_qpos"]["best_lag_s"] == pytest.approx(0.9, abs=0.04)
    clip = fidelity.default_clips()[2]
    trajectories = {
        name: _passing_trajectory() for name in clip.semantic_channels
    }
    trajectories[clip.semantic_channels[0]] = late
    gate = fidelity.evaluate_semantic_gate(clip, trajectories, policy=policy)
    assert gate["passed"] is False
    assert gate["channels"][clip.semantic_channels[0]]["stages"]["actual_qpos"][
        "checks"
    ]["lag"] is False


def test_semantic_gate_cannot_pass_with_only_one_moving_channel() -> None:
    clip = fidelity.default_clips()[1]
    policy = fidelity.FidelityGatePolicy()
    trajectories = {
        name: _passing_trajectory() for name in clip.semantic_channels
    }

    passed = fidelity.evaluate_semantic_gate(clip, trajectories, policy=policy)
    assert passed["passed"] is True
    assert passed["minimum_channels_required"] == len(clip.semantic_channels)

    for name in clip.semantic_channels[1:]:
        trajectories[name] = {
            **_passing_trajectory(),
            "reliable": False,
            "reason": "insufficient_human_amplitude",
        }
    rejected = fidelity.evaluate_semantic_gate(clip, trajectories, policy=policy)
    assert rejected["passed"] is False
    assert rejected["passed_channels"] == 1


def test_clip_assessment_requires_both_fidelity_stages_and_semantic_motion() -> None:
    clip = fidelity.default_clips()[1]
    policy = fidelity.FidelityGatePolicy()
    evaluation = _passing_evaluation(clip)

    passed = fidelity.assess_clip(clip, evaluation, policy=policy)
    assert passed["passed"] is True

    evaluation["metrics"]["fidelity"]["actual_qpos"]["groups"]["legs"][
        "p90_deg"
    ] = 100.0
    failed = fidelity.assess_clip(clip, evaluation, policy=policy)
    assert failed["passed"] is False
    assert "actual_qpos_legs_p90_deg" in {
        gate["name"] for gate in failed["gates"] if not gate["passed"]
    }


def test_clip_assessment_gates_abort_nonfoot_and_group_coverage() -> None:
    clip = fidelity.default_clips()[1]
    policy = fidelity.FidelityGatePolicy()
    evaluation = _passing_evaluation(clip)

    evaluation["teleop_stats"]["support_abort_count"] = 1
    evaluation["teleop_stats"]["support_abort_reasons"] = ["touchdown_timeout"]
    aborted = fidelity.assess_clip(clip, evaluation, policy=policy)
    assert {
        gate["name"] for gate in aborted["gates"] if not gate["passed"]
    } >= {"support_abort_count", "support_abort_reasons"}

    evaluation = _passing_evaluation(clip)
    evaluation["teleop_stats"]["maximum_non_foot_ground_contacts"] = 1
    contacted = fidelity.assess_clip(clip, evaluation, policy=policy)
    assert "maximum_non_foot_ground_contacts" in {
        gate["name"] for gate in contacted["gates"] if not gate["passed"]
    }

    evaluation = _passing_evaluation(clip)
    evaluation["metrics"]["coverage"]["directions"]["right_arm"][
        "valid_frames"
    ] = 29
    evaluation["metrics"]["coverage"]["directions"]["right_leg"][
        "valid_frame_fraction"
    ] = 0.79
    uncovered = fidelity.assess_clip(clip, evaluation, policy=policy)
    assert {
        gate["name"] for gate in uncovered["gates"] if not gate["passed"]
    } >= {
        "coverage_right_arm_valid_frames",
        "coverage_right_leg_valid_frame_fraction",
    }


def test_clip_assessment_rejects_early_eof_even_with_perfect_relative_coverage() -> None:
    clip = fidelity.default_clips()[2]
    evaluation = _passing_evaluation(clip)
    evaluation["teleop_stats"]["frames"] = clip.expected_frames - 1
    evaluation["metrics"]["coverage"]["source_frames"] = clip.expected_frames - 1
    evaluation["metrics"]["coverage"]["evaluated_source_fraction"] = 1.0

    result = fidelity.assess_clip(
        clip, evaluation, policy=fidelity.FidelityGatePolicy()
    )
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}

    assert result["passed"] is False
    assert {"frames", "source_frames"} <= failed


def test_clip_assessment_gates_model_video_and_calibration_provenance() -> None:
    clip = fidelity.default_clips()[1]
    policy = fidelity.FidelityGatePolicy()
    evaluation = _passing_evaluation(clip)
    evaluation["video_sha256"] = "0" * 64
    evaluation["metrics"]["instrumentation"].update(
        {"equality_constraint_count": 1, "base_joint_type": "fixed"}
    )
    evaluation["teleop_stats"]["calibration"].update(
        {"source_sha256": "f" * 64, "frame_index": 3}
    )

    failed = fidelity.assess_clip(clip, evaluation, policy=policy)
    assert {
        "video_sha256",
        "equality_constraint_count",
        "base_joint_type",
        "calibration_source_sha256",
        "calibration_frame_index",
    } <= {gate["name"] for gate in failed["gates"] if not gate["passed"]}


def test_unilateral_clip_replaces_3d_leg_gate_with_event_gate() -> None:
    clip = fidelity.default_clips()[0]
    policy = fidelity.FidelityGatePolicy()
    evaluation = _passing_evaluation(clip)
    evaluation["metrics"]["fidelity"]["safe_command"]["groups"]["legs"][
        "p90_deg"
    ] = 179.0
    evaluation["metrics"]["fidelity"]["actual_qpos"]["groups"]["legs"][
        "p90_deg"
    ] = 179.0
    evaluation["metrics"]["leg_events"] = {
        "applicable": True,
        "passed": True,
        "completed_side_counts": {"right_swing": 1, "left_swing": 1},
    }

    passed = fidelity.assess_clip(clip, evaluation, policy=policy)
    assert passed["passed"] is True
    assert passed["semantic_gate"]["required_channels"] == [
        "right_arm",
        "left_arm",
    ]
    assert set(passed["semantic_gate"]["event_replaced_3d_leg_channels"]) == {
        "right_leg",
        "left_leg",
    }
    assert not any(
        gate["name"].endswith(("legs_p90_deg", "end_effectors_p90_deg"))
        for gate in passed["gates"]
    )

    evaluation["metrics"]["leg_events"]["passed"] = False
    failed = fidelity.assess_clip(clip, evaluation, policy=policy)
    assert failed["passed"] is False
    assert "unilateral_leg_event_semantics" in {
        gate["name"] for gate in failed["gates"] if not gate["passed"]
    }


def test_trunk_proxy_accepts_head_only_and_keeps_other_groups_descriptive() -> None:
    clip = fidelity.default_clips()[5]
    policy = fidelity.FidelityGatePolicy()
    evaluation = _passing_evaluation(clip)
    for group in ("arms", "legs", "end_effectors"):
        evaluation["metrics"]["coverage"]["groups"][group].update(
            frames_with_any_direction=5,
            valid_direction_fraction=0.10,
        )
        for stage in ("safe_command", "actual_qpos"):
            evaluation["metrics"]["fidelity"][stage]["groups"][group].update(
                count=5,
                p90_deg=179.0,
            )

    result = fidelity.assess_clip(clip, evaluation, policy=policy)

    assert result["passed"] is True
    assert result["global_acceptance"]["passed"] is True
    assert result["task_acceptance"]["passed"] is True
    assert result["task_contract"]["acceptance_direction_channels"] == ["head"]
    assert result["task_contract"]["acceptance_p90_groups"] == ["head"]
    assert result["whole_body_fidelity"]["role"] == (
        "descriptive_full_body_context"
    )
    assert {
        warning["group"] for warning in result["whole_body_fidelity"]["warnings"]
    } >= {"arms", "legs", "end_effectors"}
    assert all(
        warning["acceptance_effect"] == "none"
        for warning in result["whole_body_fidelity"]["warnings"]
    )
    assert result["whole_body_fidelity"]["fidelity"]["actual_qpos"]["groups"][
        "legs"
    ]["p90_deg"] == 179.0


def test_trunk_proxy_still_fails_head_or_global_evidence_gates() -> None:
    clip = fidelity.default_clips()[5]
    policy = fidelity.FidelityGatePolicy()
    head_missing = _passing_evaluation(clip)
    head_missing["metrics"]["coverage"]["directions"]["head"][
        "valid_frames"
    ] = 29

    failed_head = fidelity.assess_clip(clip, head_missing, policy=policy)
    assert failed_head["passed"] is False
    assert failed_head["global_acceptance"]["passed"] is True
    assert failed_head["task_acceptance"]["passed"] is False
    assert "coverage_head_valid_frames" in {
        gate["name"] for gate in failed_head["gates"] if not gate["passed"]
    }

    source_missing = _passing_evaluation(clip)
    source_missing["metrics"]["coverage"]["evaluated_source_fraction"] = 0.69
    failed_source = fidelity.assess_clip(clip, source_missing, policy=policy)
    assert failed_source["passed"] is False
    assert failed_source["global_acceptance"]["passed"] is False
    assert failed_source["task_acceptance"]["passed"] is True


def test_jumping_jacks_cannot_post_hoc_drop_required_leg_channels() -> None:
    clip = fidelity.default_clips()[1]
    policy = fidelity.FidelityGatePolicy()
    evaluation = _passing_evaluation(clip)
    for name in ("right_leg", "left_leg"):
        evaluation["metrics"]["trajectories"][name] = {
            **_passing_trajectory(),
            "reliable": False,
            "reason": "unsupported_motion_was_not_observed",
        }

    result = fidelity.assess_clip(clip, evaluation, policy=policy)

    assert result["passed"] is False
    assert result["task_acceptance"]["passed"] is False
    assert result["task_acceptance"]["trajectory_semantics"][
        "required_channels"
    ] == ["right_arm", "left_arm", "right_leg", "left_leg"]
    assert result["task_acceptance"]["trajectory_semantics"][
        "passed_channels"
    ] == 2
    assert "remain acceptance gates" in result["task_contract"][
        "unsupported_limitations"
    ][0]


def test_production_arguments_are_free_base_ik_and_explicitly_exclude_settling() -> None:
    arguments = fidelity._teleop_arguments(fidelity.default_clips()[0], max_frames=123)
    parsed = fidelity.build_teleop_parser().parse_args(arguments)

    assert parsed.free_base
    assert parsed.balance_controller
    assert parsed.retargeting == "ik"
    assert parsed.headless
    assert parsed.max_frames == 123
    assert parsed.settle_seconds == 0.0


def test_policy_and_clip_specs_allow_one_channel_but_reject_empty_contracts() -> None:
    single_channel = fidelity.ClipSpec(
        "proxy",
        "replay",
        fidelity.PROJECT_ROOT / "proxy.mp4",
        None,
        ("head",),
        "orientation proxy",
        "head orientation tracking",
        100,
        gated_p90_groups=("head",),
    )
    assert single_channel.semantic_channels == ("head",)

    with pytest.raises(ValueError, match="requires a semantic channel"):
        fidelity.ClipSpec(
            "bad",
            "replay",
            fidelity.PROJECT_ROOT / "bad.mp4",
            None,
            (),
            "bad task",
            "bad capability",
            100,
        )
    with pytest.raises(ValueError, match="task_label"):
        fidelity.ClipSpec(
            "bad-label",
            "replay",
            fidelity.PROJECT_ROOT / "bad.mp4",
            None,
            ("head",),
            " ",
            "head orientation tracking",
            100,
        )
    with pytest.raises(ValueError, match="expected_frames"):
        fidelity.ClipSpec(
            "bad-frames",
            "replay",
            fidelity.PROJECT_ROOT / "bad.mp4",
            None,
            ("head",),
            "orientation proxy",
            "head orientation tracking",
            0,
        )
    with pytest.raises(ValueError, match="at least three"):
        fidelity.FidelityGatePolicy(minimum_trajectory_samples=2)
    with pytest.raises(ValueError, match="must exceed"):
        fidelity.FidelityGatePolicy(
            maximum_actual_lag_s=0.5,
            lag_search_horizon_s=0.5,
        )


def _event_config() -> SupportControlConfig:
    return SupportControlConfig(
        shift_duration_s=0.20,
        load_confirm_duration_s=0.05,
        stance_load_timeout_s=0.30,
        lift_duration_s=0.20,
        minimum_hold_duration_s=0.05,
        lower_duration_s=0.30,
        touchdown_confirm_duration_s=0.05,
        touchdown_preload_duration_s=0.20,
        touchdown_timeout_s=0.40,
        center_duration_s=0.20,
        support_loss_grace_s=0.05,
    )


def _event_sample(
    timestamp_s: float,
    *,
    human: str = "double_support",
    active: str = "double_support",
    phase: str = "double_support",
    safe_difference_m: float = 0.0,
    actual_difference_m: float = 0.0,
    abort_reason: str | None = None,
) -> fidelity._TraceSample:
    ratio = 0.0
    if human == "right_swing":
        ratio = 0.30
    elif human == "left_swing":
        ratio = -0.30
    return fidelity._TraceSample(
        timestamp_s=timestamp_s,
        human_support_intent=human,
        human_support_lift_ratio=ratio,
        human_support_calibrated=True,
        human_support_stale=False,
        support_phase=phase,
        support_active_intent=active,
        support_requested_intent=active,
        support_abort_reason=abort_reason,
        safe_right_minus_left_foot_height_m=safe_difference_m,
        actual_right_minus_left_foot_height_m=actual_difference_m,
    )


def _completed_two_side_trace() -> list[fidelity._TraceSample]:
    return [
        _event_sample(0.0),
        _event_sample(
            0.1,
            human="right_swing",
            active="right_swing",
            phase="shift_weight",
        ),
        _event_sample(
            0.2,
            human="right_swing",
            active="right_swing",
            phase="lift_swing",
            safe_difference_m=0.01,
            actual_difference_m=0.01,
        ),
        _event_sample(
            0.3,
            human="right_swing",
            active="right_swing",
            phase="hold_swing",
            safe_difference_m=0.04,
            actual_difference_m=0.03,
        ),
        _event_sample(
            0.4,
            active="right_swing",
            phase="lower_swing",
        ),
        _event_sample(
            0.5,
            active="right_swing",
            phase="verify_touchdown",
        ),
        _event_sample(
            0.6,
            active="right_swing",
            phase="center_weight",
        ),
        _event_sample(0.7),
        _event_sample(
            0.8,
            human="left_swing",
            active="left_swing",
            phase="shift_weight",
        ),
        _event_sample(
            0.9,
            human="left_swing",
            active="left_swing",
            phase="lift_swing",
            safe_difference_m=-0.01,
            actual_difference_m=-0.01,
        ),
        _event_sample(
            1.0,
            human="left_swing",
            active="left_swing",
            phase="hold_swing",
            safe_difference_m=-0.04,
            actual_difference_m=-0.03,
        ),
        _event_sample(
            1.1,
            active="left_swing",
            phase="lower_swing",
        ),
        _event_sample(
            1.2,
            active="left_swing",
            phase="verify_touchdown",
        ),
        _event_sample(
            1.3,
            active="left_swing",
            phase="center_weight",
        ),
        _event_sample(1.4),
    ]


def test_leg_event_gate_matches_idle_left_and_right_events() -> None:
    result = fidelity._analyze_leg_events(
        _completed_two_side_trace(),
        required_sides=("right_swing", "left_swing"),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    assert result["passed"] is True
    assert result["completed_side_counts"] == {
        "right_swing": 1,
        "left_swing": 1,
    }
    assert [event["status"] for event in result["events"]] == [
        "passed",
        "passed",
    ]
    assert not result["unexpected_cycle_indices"]


def test_leg_event_gate_preserves_queued_opposite_fifo_order() -> None:
    samples = [
        _event_sample(0.0),
        _event_sample(
            0.1,
            human="right_swing",
            active="right_swing",
            phase="shift_weight",
        ),
        _event_sample(
            0.2,
            human="right_swing",
            active="right_swing",
            phase="hold_swing",
            safe_difference_m=0.04,
            actual_difference_m=0.03,
        ),
        _event_sample(
            0.3,
            active="right_swing",
            phase="lower_swing",
            safe_difference_m=0.02,
            actual_difference_m=0.02,
        ),
        _event_sample(
            0.4,
            human="left_swing",
            active="right_swing",
            phase="center_weight",
        ),
        _event_sample(
            0.5,
            human="left_swing",
            active="right_swing",
            phase="center_weight",
        ),
        _event_sample(
            0.6,
            human="left_swing",
            active="left_swing",
            phase="shift_weight",
        ),
        _event_sample(
            0.7,
            human="left_swing",
            active="left_swing",
            phase="hold_swing",
            safe_difference_m=-0.04,
            actual_difference_m=-0.03,
        ),
        _event_sample(
            0.8,
            active="left_swing",
            phase="lower_swing",
        ),
        _event_sample(
            0.9,
            active="left_swing",
            phase="center_weight",
        ),
        _event_sample(1.0),
    ]

    result = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing", "left_swing"),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    assert result["passed"] is True
    assert [event["matched_cycle_side"] for event in result["events"]] == [
        "right_swing",
        "left_swing",
    ]
    queued = result["events"][1]
    assert queued["queued_behind_opposite_cycle"] is True
    assert queued["acceptance_latency_s"] == pytest.approx(0.2)
    assert result["deadlines"]["queued_acceptance_s"] == pytest.approx(0.95)
    assert result["deadlines"]["own_clearance_s"] == pytest.approx(0.70)


def test_leg_event_gate_coalesces_same_side_chatter_inside_one_active_cycle() -> None:
    samples = [
        _event_sample(0.0),
        _event_sample(
            0.1,
            human="right_swing",
            active="right_swing",
            phase="shift_weight",
        ),
        _event_sample(
            0.2,
            active="right_swing",
            phase="lift_swing",
            safe_difference_m=0.02,
            actual_difference_m=0.02,
        ),
        _event_sample(
            0.3,
            human="right_swing",
            active="right_swing",
            phase="hold_swing",
            safe_difference_m=0.04,
            actual_difference_m=0.03,
        ),
        _event_sample(
            0.4,
            active="right_swing",
            phase="lower_swing",
        ),
        _event_sample(
            0.5,
            human="right_swing",
            active="right_swing",
            phase="lower_swing",
        ),
        _event_sample(
            0.6,
            active="right_swing",
            phase="center_weight",
        ),
        _event_sample(1.5),
    ]
    result = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing",),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    assert result["passed"] is True
    assert len(result["events"]) == 1
    assert result["events"][0]["intent_segment_count"] == 3
    assert result["events"][0]["coalesced_interruption_frames"] == 2


def test_leg_event_gate_never_coalesces_across_an_opposite_active_cycle() -> None:
    samples = [
        _event_sample(0.0),
        _event_sample(
            0.1,
            human="right_swing",
            active="right_swing",
            phase="hold_swing",
            safe_difference_m=0.04,
            actual_difference_m=0.03,
        ),
        _event_sample(
            0.2,
            active="left_swing",
            phase="shift_weight",
        ),
        _event_sample(
            0.3,
            human="right_swing",
            active="left_swing",
            phase="hold_swing",
        ),
        _event_sample(2.0),
    ]
    events = fidelity._intent_events(samples)

    assert len(events) == 2
    assert all(event["intent_segment_count"] == 1 for event in events)


def test_leg_event_gate_rejects_wrong_side_fifo_cycle() -> None:
    samples = [
        _event_sample(0.0),
        _event_sample(
            0.1,
            human="right_swing",
            active="left_swing",
            phase="shift_weight",
        ),
        _event_sample(
            0.2,
            human="right_swing",
            active="left_swing",
            phase="hold_swing",
            safe_difference_m=-0.04,
            actual_difference_m=-0.03,
        ),
        _event_sample(2.0),
    ]

    result = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing",),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    assert result["passed"] is False
    assert "wrong_side_fifo" in result["events"][0]["failure_reasons"]


@pytest.mark.parametrize(
    ("safe_clearance", "actual_clearance", "expected_reason"),
    (
        (0.01, 0.03, "safe_clearance_below_threshold"),
        (0.03, 0.01, "actual_clearance_below_threshold"),
    ),
)
def test_leg_event_gate_rejects_completed_low_clearance(
    safe_clearance: float,
    actual_clearance: float,
    expected_reason: str,
) -> None:
    samples = [
        _event_sample(0.0),
        _event_sample(
            0.1,
            human="right_swing",
            active="right_swing",
            phase="shift_weight",
        ),
        _event_sample(
            0.2,
            human="right_swing",
            active="right_swing",
            phase="hold_swing",
            safe_difference_m=safe_clearance,
            actual_difference_m=actual_clearance,
        ),
        _event_sample(2.0),
    ]

    result = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing",),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    assert result["passed"] is False
    assert expected_reason in result["events"][0]["failure_reasons"]
    assert result["events"][0]["status"] == "failed"


def test_leg_event_gate_rejects_support_abort_even_before_eof() -> None:
    samples = [
        _event_sample(0.0),
        _event_sample(
            0.1,
            human="right_swing",
            active="right_swing",
            phase="lift_swing",
        ),
        _event_sample(
            0.2,
            human="right_swing",
            active="right_swing",
            phase="center_weight",
            abort_reason="touchdown_timeout",
        ),
        _event_sample(0.3),
    ]

    result = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing",),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    assert result["events"][0]["status"] == "failed"
    assert "support_abort" in result["events"][0]["failure_reasons"]


def test_leg_event_gate_censors_only_short_eof_tail_after_required_completion() -> None:
    samples = _completed_two_side_trace()[:8]
    samples.extend(
        [
            _event_sample(0.8),
            _event_sample(1.0, human="right_swing"),
            _event_sample(1.1, human="right_swing"),
        ]
    )
    censored = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing",),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    assert censored["passed"] is True
    assert censored["censored_event_count"] == 1
    assert censored["events"][-1]["status"] == "censored_eof"

    samples.extend(
        _event_sample(timestamp)
        for timestamp in np.arange(1.2, 2.2, 0.1)
    )
    expired = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing",),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )
    assert expired["passed"] is False
    assert expired["events"][-1]["status"] == "failed"
    assert "missing_support_cycle" in expired["events"][-1]["failure_reasons"]


def test_leg_event_gate_does_not_count_hold_at_eof_as_a_completed_cycle() -> None:
    samples = [
        _event_sample(0.0),
        _event_sample(
            0.1,
            human="right_swing",
            active="right_swing",
            phase="shift_weight",
        ),
        _event_sample(
            0.2,
            human="right_swing",
            active="right_swing",
            phase="hold_swing",
            safe_difference_m=0.04,
            actual_difference_m=0.03,
        ),
    ]

    result = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing",),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    assert result["passed"] is False
    assert result["completed_side_counts"]["right_swing"] == 0
    assert result["events"][0]["status"] == "censored_eof"
    assert result["events"][0]["failure_reasons"] == []
    assert result["events"][0]["return_trigger_timestamp_s"] is None


def test_leg_event_gate_censors_extra_hold_at_eof_after_required_cycles() -> None:
    samples = _completed_two_side_trace()
    samples.extend(
        (
            _event_sample(
                1.5,
                human="right_swing",
                active="right_swing",
                phase="shift_weight",
            ),
            _event_sample(
                1.6,
                human="right_swing",
                active="right_swing",
                phase="hold_swing",
                safe_difference_m=0.04,
                actual_difference_m=0.03,
            ),
        )
    )

    result = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing", "left_swing"),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    assert result["passed"] is True
    assert result["completed_side_counts"] == {
        "right_swing": 1,
        "left_swing": 1,
    }
    assert result["failed_event_count"] == 0
    assert result["censored_event_count"] == 1
    assert result["events"][-1]["status"] == "censored_eof"


def test_leg_event_gate_never_censors_low_clearance_after_hold() -> None:
    samples = _completed_two_side_trace()
    samples.extend(
        (
            _event_sample(
                1.5,
                human="right_swing",
                active="right_swing",
                phase="shift_weight",
            ),
            _event_sample(
                1.6,
                human="right_swing",
                active="right_swing",
                phase="hold_swing",
                safe_difference_m=0.001,
                actual_difference_m=0.001,
            ),
        )
    )

    result = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing", "left_swing"),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    event = result["events"][-1]
    assert result["passed"] is False
    assert result["failed_event_count"] == 1
    assert event["status"] == "failed"
    assert "safe_clearance_below_threshold" in event["failure_reasons"]
    assert "actual_clearance_below_threshold" in event["failure_reasons"]


def test_leg_event_gate_fails_incomplete_return_after_full_return_window() -> None:
    samples = [
        _event_sample(0.0),
        _event_sample(
            0.1,
            human="right_swing",
            active="right_swing",
            phase="shift_weight",
        ),
        _event_sample(
            0.2,
            human="right_swing",
            active="right_swing",
            phase="hold_swing",
            safe_difference_m=0.04,
            actual_difference_m=0.03,
        ),
        _event_sample(
            0.3,
            active="right_swing",
            phase="hold_swing",
            safe_difference_m=0.04,
            actual_difference_m=0.03,
        ),
        _event_sample(
            1.3,
            active="right_swing",
            phase="hold_swing",
            safe_difference_m=0.04,
            actual_difference_m=0.03,
        ),
    ]

    result = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing",),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    event = result["events"][0]
    assert event["return_window_observed"] is True
    assert event["status"] == "failed"
    assert "support_cycle_incomplete" in event["failure_reasons"]


def test_leg_event_gate_requires_a_completed_event_for_every_declared_side() -> None:
    samples = _completed_two_side_trace()[:8]
    samples.extend(
        [
            _event_sample(0.8),
            _event_sample(0.9, human="left_swing"),
            _event_sample(1.0, human="left_swing"),
        ]
    )
    result = fidelity._analyze_leg_events(
        samples,
        required_sides=("right_swing", "left_swing"),
        support_config=_event_config(),
        policy=fidelity.FidelityGatePolicy(),
    )

    assert result["events"][-1]["status"] == "censored_eof"
    assert result["required_sides_met"] is False
    assert result["passed"] is False


def test_overall_pass_requires_exact_six_full_length_canonical_clips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fidelity, "_runtime_hashes", lambda _clips: {})
    clips = fidelity.default_clips()
    results = [{"clip": clip.name, "passed": True} for clip in clips]
    policy = fidelity.FidelityGatePolicy()
    runtime_snapshot = _valid_runtime_snapshot(clips)

    complete = fidelity.build_report(
        results,
        clips,
        policy=policy,
        max_frames=0,
        runtime_hashes_at_start=runtime_snapshot,
        runtime_hashes_at_completion=runtime_snapshot,
    )
    assert complete["overall_passed"] is True
    assert complete["configuration"]["canonical_complete_matrix"] is True
    contracts = complete["configuration"]["task_contracts"]
    trunk_contract = next(
        contract for contract in contracts if contract["clip"] == "trunk-circles"
    )
    assert trunk_contract["semantic_channels"] == ["head"]
    assert trunk_contract["acceptance_p90_groups"] == ["head"]
    assert trunk_contract["unsupported_limitations"]

    subset = fidelity.build_report(
        results[:1],
        clips[:1],
        policy=policy,
        max_frames=0,
        runtime_hashes_at_start=_valid_runtime_snapshot(clips[:1]),
        runtime_hashes_at_completion=_valid_runtime_snapshot(clips[:1]),
    )
    assert subset["overall_passed"] is False
    assert "exact_canonical_clip_matrix" in {
        gate["name"] for gate in subset["suite_gates"] if not gate["passed"]
    }

    truncated = fidelity.build_report(
        results,
        clips,
        policy=policy,
        max_frames=100,
        runtime_hashes_at_start=runtime_snapshot,
        runtime_hashes_at_completion=runtime_snapshot,
    )
    assert truncated["overall_passed"] is False
    assert "full_length_replays" in {
        gate["name"] for gate in truncated["suite_gates"] if not gate["passed"]
    }

    changed_snapshot = dict(runtime_snapshot)
    changed_snapshot[next(iter(changed_snapshot))] = "1" * 64
    changed = fidelity.build_report(
        results,
        clips,
        policy=policy,
        max_frames=0,
        runtime_hashes_at_start=runtime_snapshot,
        runtime_hashes_at_completion=changed_snapshot,
    )
    assert changed["overall_passed"] is False
    assert "runtime_inputs_unchanged" in {
        gate["name"] for gate in changed["suite_gates"] if not gate["passed"]
    }

    weakened_policy = replace(policy, minimum_evaluated_source_fraction=0.10)
    weakened = fidelity.build_report(
        results,
        clips,
        policy=weakened_policy,
        max_frames=0,
        runtime_hashes_at_start=runtime_snapshot,
        runtime_hashes_at_completion=runtime_snapshot,
    )
    assert weakened["overall_passed"] is False
    assert "canonical_or_stricter_policy" in {
        gate["name"] for gate in weakened["suite_gates"] if not gate["passed"]
    }

    stricter_policy = replace(
        policy,
        minimum_evaluated_source_fraction=0.75,
        maximum_actual_lag_s=0.40,
    )
    stricter = fidelity.build_report(
        results,
        clips,
        policy=stricter_policy,
        max_frames=0,
        runtime_hashes_at_start=runtime_snapshot,
        runtime_hashes_at_completion=runtime_snapshot,
    )
    assert stricter["overall_passed"] is True

    incomplete_snapshot = fidelity.build_report(
        results,
        clips,
        policy=policy,
        max_frames=0,
        runtime_hashes_at_start={},
        runtime_hashes_at_completion={},
    )
    assert incomplete_snapshot["overall_passed"] is False
    assert {
        "runtime_snapshot_at_start_complete",
        "runtime_snapshot_at_completion_complete",
    } <= {
        gate["name"]
        for gate in incomplete_snapshot["suite_gates"]
        if not gate["passed"]
    }


def test_main_hashes_runtime_before_and_after_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    clip = fidelity.default_clips()[0]
    hash_snapshots = iter(({"runtime.py": "before"}, {"runtime.py": "after"}))
    captured: dict[str, object] = {}
    result = {
        "clip": clip.name,
        "passed": True,
        "metrics": {"coverage": {"evaluated_nonstale_frames": 1}},
        "semantic_gate": {
            "passed_channels": 1,
            "minimum_channels_required": 1,
        },
        "leg_event_gate": {"applicable": False},
    }

    monkeypatch.setattr(fidelity, "default_clips", lambda: (clip,))
    monkeypatch.setattr(
        fidelity,
        "_runtime_hashes",
        lambda _clips: next(hash_snapshots),
    )
    monkeypatch.setattr(
        fidelity,
        "evaluate_suite",
        lambda clips, *, policy, max_frames: [result],
    )

    def fake_build_report(
        results,
        clips,
        *,
        policy,
        max_frames,
        runtime_hashes_at_start,
        runtime_hashes_at_completion,
    ):
        captured["start"] = runtime_hashes_at_start
        captured["completion"] = runtime_hashes_at_completion
        return {"overall_passed": True}

    monkeypatch.setattr(fidelity, "build_report", fake_build_report)

    assert fidelity.main(["--output", str(tmp_path / "report.json")]) == 0
    assert captured == {
        "start": {"runtime.py": "before"},
        "completion": {"runtime.py": "after"},
    }
