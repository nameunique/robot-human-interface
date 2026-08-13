from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import evaluate_freebase_stability as stability


def _motion(span: float = 100.0) -> dict[str, object]:
    groups = {
        name: {"maximum_span_deg": span, "active_joints_over_2deg": len(indices)}
        for name, indices in stability.MOTION_GROUPS.items()
    }
    return {
        "safe_command": groups,
        "actual_joint": groups,
    }


def _passing_evaluation(
    clip: stability.ClipSpec, settling_seconds: float
) -> dict[str, object]:
    expectation = clip.expectation
    right = (
        expectation.right_swing_completed
        if expectation.right_swing_completed is not None
        else False
    )
    left = (
        expectation.left_swing_completed
        if expectation.left_swing_completed is not None
        else False
    )
    return {
        "exit_code": 0,
        "video": clip.path.as_posix(),
        "video_size_bytes": 123,
        "video_sha256": "0" * 64,
        "teleop_arguments": ["--headless", "--free-base", "--retargeting", "ik"],
        "metrics": {
            "base_mode": "free",
            "equality_constraint_count": 0,
            "base_joint_type": "free",
            "frames": clip.expected_frames,
            "skeleton_frames": clip.expected_frames,
            "skeleton_fraction": 1.0,
            "stale_commands": 0,
            "stale_fraction": 0.0,
            "minimum_base_height_m": 0.82,
            "final_base_height_m": 0.86,
            "maximum_tilt_deg": 12.0,
            "fell": False,
            "support_transitions": max(20, expectation.minimum_support_transitions),
            "right_swing_completed": right,
            "left_swing_completed": left,
            "maximum_right_foot_clearance_m": max(
                0.04, expectation.minimum_right_clearance_m
            ),
            "maximum_left_foot_clearance_m": max(
                0.04, expectation.minimum_left_clearance_m
            ),
            "simulation_time_s": 8.301,
            "input_simulation_time_s": 3.300,
            "media_time_s": 3.301,
            "media_sync_error_s": 0.001,
            "raw_command_span_deg": 150.0,
            "safe_command_span_deg": 100.0,
            "maximum_non_foot_ground_contacts": 0,
            "maximum_loaded_foot_slip_speed_m_s": 0.05,
            "right_foot_slip_distance_m": 0.03,
            "left_foot_slip_distance_m": 0.04,
            "maximum_swing_foot_impact_speed_m_s": 0.32,
            "maximum_swing_foot_impact_force_n": 21.0,
            "maximum_swing_foot_contact_impulse_n_s": 0.042,
            "motion": _motion(),
        },
        "settling": {
            "requested_duration_s": settling_seconds,
            "elapsed_s": settling_seconds + 0.001,
            "stable_s": settling_seconds,
            "completed": True,
            "status": "passed",
            "same_simulation": True,
            "simulation_instance_count": 1,
            "data_instance_count": 1,
            "minimum_base_height_m": 0.82,
            "maximum_tilt_deg": 8.0,
        },
    }


def test_default_matrix_contains_bundled_and_external_replays() -> None:
    clips = stability.default_clips()

    assert [clip.name for clip in clips] == [
        "slow-balance",
        "jumping-jacks",
        "arm-circles",
        "frontal-leg-swing",
        "stationary-squat",
        "trunk-circles",
    ]
    assert all(clip.path.is_file() for clip in clips)
    assert [clip.source for clip in clips[:2]] == ["mp4", "mp4"]
    assert all(clip.source == "replay" for clip in clips[2:])


def test_motion_evidence_excludes_same_simulation_settling_tail() -> None:
    recorder = stability._SimulationRecorder(timestep_s=0.002)
    zero = np.zeros(20)
    input_motion = zero.copy()
    input_motion[0] = np.radians(10.0)
    input_motion[6] = np.radians(20.0)
    settling_return = np.full(20, np.radians(90.0))
    recorder.safe_samples_rad.extend((zero, input_motion, settling_return))
    recorder.actual_samples_rad.extend((zero, input_motion, settling_return))

    motion = recorder.motion(settling_elapsed_s=0.002)

    assert motion["settling_steps_excluded"] == 1
    assert motion["safe_command"]["arms"]["maximum_span_deg"] == pytest.approx(10.0)
    assert motion["safe_command"]["legs"]["maximum_span_deg"] == pytest.approx(20.0)
    assert motion["actual_joint"]["arms"]["maximum_span_deg"] == pytest.approx(10.0)


def test_same_simulation_evidence_uses_simulation_and_data_object_identity() -> None:
    recorder = stability._SimulationRecorder()
    first = SimpleNamespace(data=object())
    replacement = SimpleNamespace(data=object())

    recorder.record_instance(first)
    recorder.record_instance(first)
    assert recorder.same_simulation is True
    assert len(recorder.simulation_instances) == 1
    assert len(recorder.data_instances) == 1

    recorder.record_instance(replacement)
    assert recorder.same_simulation is False
    assert len(recorder.simulation_instances) == 2
    assert len(recorder.data_instances) == 2


def test_fake_evaluator_passes_every_explicit_gate_without_running_mujoco() -> None:
    clips = stability.default_clips()
    calls: list[tuple[str, float]] = []

    def fake(
        clip: stability.ClipSpec, settling_seconds: float
    ) -> dict[str, object]:
        calls.append((clip.name, settling_seconds))
        return _passing_evaluation(clip, settling_seconds)

    results = stability.evaluate_suite(
        clips,
        thresholds=stability.StabilityThresholds(),
        settling_seconds=5.0,
        evaluator=fake,
    )
    report = stability.build_report(
        results,
        thresholds=stability.StabilityThresholds(),
        settling_seconds=5.0,
    )

    assert calls == [(clip.name, 5.0) for clip in clips]
    assert report["overall_passed"] is True
    assert all(result["passed"] for result in results)
    assert all(all(gate["passed"] for gate in result["gates"]) for result in results)
    assert all(result["settling"]["status"] == "passed" for result in results)
    assert all(result["settling"]["same_simulation"] is True for result in results)


def test_gates_reject_fixed_fallen_nonfinite_unsynchronized_and_missing_motion() -> None:
    clip = stability.default_clips()[0]
    evaluation = _passing_evaluation(clip, 5.0)
    metrics = evaluation["metrics"]
    assert isinstance(metrics, dict)
    metrics.update(
        {
            "frames": clip.expected_frames - 1,
            "base_mode": "fixed",
            "equality_constraint_count": 1,
            "base_joint_type": "slide",
            "fell": True,
            "minimum_base_height_m": 0.40,
            "final_base_height_m": math.nan,
            "maximum_tilt_deg": 70.0,
            "media_sync_error_s": 0.2,
            "stale_fraction": 0.5,
            "maximum_non_foot_ground_contacts": 2,
            "maximum_loaded_foot_slip_speed_m_s": 0.5,
            "right_foot_slip_distance_m": 0.25,
            "left_foot_slip_distance_m": 0.30,
            "motion": _motion(0.0),
        }
    )
    evaluation["exit_code"] = 1
    settling = evaluation["settling"]
    assert isinstance(settling, dict)
    settling.update(
        {
            "same_simulation": False,
            "simulation_instance_count": 2,
            "data_instance_count": 2,
            "status": "failed",
            "completed": False,
            "stable_s": 1.0,
            "minimum_base_height_m": 0.45,
            "maximum_tilt_deg": 60.0,
        }
    )

    result = stability.assess_clip(
        clip,
        evaluation,
        stability.StabilityThresholds(),
        settling_seconds=5.0,
    )
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}

    assert result["passed"] is False
    assert {
        "base_mode",
        "equality_constraint_count",
        "base_joint_type",
        "exit_code",
        "frames",
        "fell",
        "minimum_base_height_m",
        "final_base_height_m",
        "maximum_tilt_deg",
        "media_sync_error_s",
        "stale_fraction",
        "maximum_non_foot_ground_contacts",
        "maximum_loaded_foot_slip_speed_m_s",
        "right_foot_slip_distance_m",
        "left_foot_slip_distance_m",
        "finite_metrics",
        "settling_same_simulation",
        "settling_simulation_instance_count",
        "settling_data_instance_count",
        "settling_status",
        "settling_completed",
        "settling_stable_duration_s",
        "settling_minimum_base_height_m",
        "settling_maximum_tilt_deg",
        "safe_arms_motion",
        "actual_legs_motion",
    } <= failed
    sanitized = stability._sanitize_json(result)
    assert isinstance(sanitized, dict)
    assert sanitized["metrics"]["final_base_height_m"] is None
    json.dumps(sanitized, allow_nan=False)


def test_evaluator_exception_is_retained_as_machine_readable_failure() -> None:
    clip = stability.default_clips()[0]

    def failing(
        _clip: stability.ClipSpec, _settling_seconds: float
    ) -> dict[str, object]:
        raise RuntimeError("synthetic evaluator failure")

    results = stability.evaluate_suite(
        [clip],
        thresholds=stability.StabilityThresholds(),
        settling_seconds=5.0,
        evaluator=failing,
    )

    assert len(results) == 1
    assert results[0]["passed"] is False
    assert results[0]["metrics"] is None
    assert "synthetic evaluator failure" in results[0]["error"]
    assert results[0]["gates"] == [
        {
            "name": "evaluation_completed",
            "passed": False,
            "observed": False,
            "required": True,
        }
    ]


def test_main_writes_valid_json_and_returns_success_with_fake_evaluator(
    tmp_path: Path,
) -> None:
    output = tmp_path / "acceptance.json"

    exit_code = stability.main(
        ["--clip", "slow-balance", "--output", str(output)],
        evaluator=_passing_evaluation,
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["overall_passed"] is True
    assert report["configuration"]["base_mode"] == "free"
    assert isinstance(report["git_worktree"]["dirty"], bool)
    assert isinstance(report["git_worktree"]["tracked_diff_sha256"], str)
    assert "tools/evaluate_freebase_stability.py" in report["runtime_input_sha256"]
    assert report["runtime_inputs_unchanged_during_run"] is True
    assert report["clips"][0]["name"] == "slow-balance"
    assert report["clips"][0]["settling"]["status"] == "passed"


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--maximum-tilt-deg", "90"], "within"),
        (["--maximum-stale-fraction", "1.1"], "within"),
        (["--minimum-skeleton-fraction", "-0.1"], "within"),
        (["--maximum-loaded-foot-slip-speed-m-s", "-0.1"], "non-negative"),
        (["--maximum-foot-slip-distance-m", "-0.1"], "non-negative"),
        (["--settling-seconds", "-1"], "non-negative"),
    ],
)
def test_threshold_validation_rejects_invalid_cli_values(
    arguments: list[str], message: str
) -> None:
    args = stability.build_parser().parse_args(arguments)
    with pytest.raises(ValueError, match=message):
        stability._validated_thresholds(args)


def test_teleop_arguments_always_select_real_free_base_ik() -> None:
    for clip in stability.default_clips():
        arguments = stability._teleop_arguments(clip, 5.0)
        assert "--headless" in arguments
        assert "--free-base" in arguments
        assert "--fixed-base" not in arguments
        assert arguments[arguments.index("--retargeting") + 1] == "ik"
        assert "--balance-controller" in arguments
        assert arguments[arguments.index("--settle-seconds") + 1] == "5"
        assert arguments[arguments.index("--settle-timeout-s") + 1] == "20"


@pytest.mark.parametrize(
    ("fell", "requested_s", "settled", "expected"),
    (
        (False, 5.0, True, 0),
        (True, 5.0, True, 3),
        (False, 5.0, False, 3),
        (False, 0.0, False, 0),
    ),
)
def test_direct_evaluator_derives_the_same_safety_exit_code_as_teleop_main(
    fell: bool, requested_s: float, settled: bool, expected: int
) -> None:
    stats = SimpleNamespace(
        fell=fell,
        settling_requested_s=requested_s,
        settling_completed=settled,
    )

    assert stability._teleop_exit_code(stats) == expected
