from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools import evaluate_pose_fidelity as fidelity


def _metrics(count: int) -> dict[str, object]:
    return {
        group: {
            "geometric": {"count": count},
            "ik": {"count": count},
        }
        for group in fidelity.GROUPS
    }


def _valid_runtime_snapshot(
    clips: tuple[fidelity.ClipSpec, ...] | list[fidelity.ClipSpec],
) -> dict[str, str]:
    return {
        fidelity._project_path(path): "0" * 64
        for path in fidelity._runtime_input_paths(clips)
    }


def _valid_provenance(snapshot: dict[str, str]) -> dict[str, object]:
    return {
        "captured_at_utc": "2026-08-14T00:00:00+00:00",
        "git_revision": "a" * 40,
        "git_worktree": {
            "dirty": True,
            "status_porcelain": [" M tools/evaluate_pose_fidelity.py"],
            "tracked_diff_sha256": "b" * 64,
        },
        "runtime_input_sha256": snapshot,
    }


def test_default_matrix_declares_only_required_controlled_calibrations() -> None:
    clips = fidelity.default_clips()

    assert [clip.name for clip in clips] == [
        "slow-balance",
        "arm-circles",
        "frontal-leg-swing",
        "stationary-squat",
        "trunk-circles",
    ]
    assert [clip.expected_frames for clip in clips] == [1961, 796, 836, 817, 867]
    assert all(clip.path.is_file() for clip in clips)
    explicit = {
        clip.name: (clip.calibration_video.name, clip.calibration_frame)
        for clip in clips
        if clip.calibration_video is not None
    }
    assert explicit == {
        "frontal-leg-swing": ("dvids_arm_circles.mp4", 29),
        "trunk-circles": ("dvids_arm_circles.mp4", 29),
    }


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_clip_spec_rejects_invalid_expected_frame_counts(value: object) -> None:
    with pytest.raises(ValueError, match="expected_frames"):
        fidelity.ClipSpec(
            "clip", Path("clip.mp4"), expected_frames=value  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_clip_spec_rejects_invalid_calibration_frames(value: object) -> None:
    with pytest.raises(ValueError, match="calibration_frame"):
        fidelity.ClipSpec(
            "clip",
            Path("clip.mp4"),
            calibration_video=Path("neutral.mp4"),
            calibration_frame=value,  # type: ignore[arg-type]
        )


def test_controlled_calibration_records_content_provenance_and_resets_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration_video = tmp_path / "neutral.mp4"
    content = b"controlled-neutral-reference"
    calibration_video.write_bytes(content)

    class FakeSource:
        instance: "FakeSource | None" = None

        def __init__(self, path: Path, *, realtime: bool) -> None:
            assert path == calibration_video.resolve()
            assert realtime is False
            self.index = 0
            self.closed = False
            FakeSource.instance = self

        def read(self) -> object | None:
            if self.index >= 4:
                return None
            frame = SimpleNamespace(sequence=self.index)
            self.index += 1
            return frame

        def close(self) -> None:
            self.closed = True

    class FakePose:
        def __init__(self) -> None:
            self.closed = 0

        def estimate(self, frame: object) -> str:
            return f"skeleton-{frame.sequence}"  # type: ignore[attr-defined]

        def close(self) -> None:
            self.closed += 1

    class FakeRetargeter:
        def __init__(self) -> None:
            self.calibrated_with: object | None = None

        def calibrate(self, skeleton: object) -> bool:
            self.calibrated_with = skeleton
            return True

    monkeypatch.setattr(fidelity, "OpenCVVideoSource", FakeSource)
    pose = FakePose()
    geometric = FakeRetargeter()
    ik = FakeRetargeter()

    result = fidelity._controlled_calibration(
        calibration_video,
        2,
        pose=pose,  # type: ignore[arg-type]
        geometric=geometric,  # type: ignore[arg-type]
        ik=ik,  # type: ignore[arg-type]
    )

    assert geometric.calibrated_with == "skeleton-2"
    assert ik.calibrated_with == "skeleton-2"
    assert FakeSource.instance is not None and FakeSource.instance.closed
    assert pose.closed == 1
    assert result == {
        "mode": "explicit_replay_frame",
        "source": str(calibration_video.resolve()),
        "source_size_bytes": len(content),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "frame_index": 2,
        "geometric_accepted": True,
        "ik_accepted": True,
        "status": "passed",
        "failure_reason": None,
    }


def test_zero_evaluated_frames_fail_coverage_instead_of_silently_passing() -> None:
    coverage = fidelity._coverage_assessment(
        expected_frames=100,
        source_frames=100,
        sampled_frames=20,
        detected_frames=19,
        post_calibration_frames=0,
        evaluated_nonstale_frames=0,
        stale_frames=0,
        metrics=_metrics(0),
        calibration_passed=False,
    )

    assert coverage["coverage_passed"] is False
    assert coverage["evaluated_nonstale_sample_fraction"] == 0.0
    assert set(coverage["failure_reasons"]) == {
        "calibration_completed",
        "evaluated_nonstale_sample_fraction",
        "arms_valid_evaluated_fraction",
        "legs_valid_evaluated_fraction",
        "end_effectors_valid_evaluated_fraction",
        "head_valid_evaluated_fraction",
    }


def test_complete_coverage_passes_all_declared_gates() -> None:
    coverage = fidelity._coverage_assessment(
        expected_frames=100,
        source_frames=100,
        sampled_frames=50,
        detected_frames=49,
        post_calibration_frames=45,
        evaluated_nonstale_frames=40,
        stale_frames=5,
        metrics=_metrics(38),
        calibration_passed=True,
    )

    assert coverage["coverage_passed"] is True
    assert coverage["failure_reasons"] == []
    assert coverage["detected_sample_fraction"] == pytest.approx(0.98)
    assert coverage["evaluated_nonstale_sample_fraction"] == pytest.approx(0.80)
    assert coverage["stale_post_calibration_fraction"] == pytest.approx(5 / 45)
    assert coverage["group_valid_evaluated_fraction"] == {
        group: pytest.approx(0.95) for group in fidelity.PRIMARY_GROUPS
    }


def test_incomplete_automatic_calibration_marks_video_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "main.mp4"
    video.write_bytes(b"video")

    class FakeSource:
        def __init__(self, path: Path, *, realtime: bool) -> None:
            assert path == video.resolve()
            assert realtime is False
            self.index = 0

        def read(self) -> object | None:
            if self.index >= 3:
                return None
            frame = SimpleNamespace(timestamp_s=self.index / 30.0)
            self.index += 1
            return frame

        def close(self) -> None:
            pass

    class FakePose:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def estimate(self, frame: object) -> object:
            return frame

        def close(self) -> None:
            pass

    class StuckRetargeter:
        def __init__(self, *_: object) -> None:
            self.is_calibrating = True
            self.model = object()

        def retarget(self, *_: object, **__: object) -> object:
            return SimpleNamespace(stale=False, positions_rad=())

    class UnusedEvaluator:
        def __init__(self, _: object) -> None:
            pass

        def evaluate(self, *_: object) -> object:
            raise AssertionError("calibrating samples must not be evaluated")

    monkeypatch.setattr(
        fidelity,
        "load_retargeting_config",
        lambda _: SimpleNamespace(auto_calibration_frames=2),
    )
    monkeypatch.setattr(fidelity, "load_joint_specs", lambda _: object())
    monkeypatch.setattr(fidelity, "GeometricRetargeter", StuckRetargeter)
    monkeypatch.setattr(fidelity, "MujocoIKRetargeter", StuckRetargeter)
    monkeypatch.setattr(fidelity, "MujocoPoseFidelityEvaluator", UnusedEvaluator)
    monkeypatch.setattr(fidelity, "OpenCVVideoSource", FakeSource)
    monkeypatch.setattr(fidelity, "MediaPipePoseLandmarker", FakePose)

    result = fidelity.evaluate_video(video, stride=1, expected_frames=3)

    assert result["measurement_status"] == "incomplete"
    assert result["measurement_complete"] is False
    assert result["coverage_passed"] is False
    assert result["fidelity_acceptance"] == "not_defined"
    assert result["evaluated_frames"] == 0
    assert result["calibration"] == {
        "mode": "automatic_window",
        "source": None,
        "source_size_bytes": None,
        "source_sha256": None,
        "frame_index": None,
        "geometric_accepted": False,
        "ik_accepted": False,
        "status": "failed",
        "failure_reason": "automatic_calibration_incomplete",
    }
    assert "evaluated_nonstale_sample_fraction" in result["coverage"][  # type: ignore[index]
        "failure_reasons"
    ]


def test_runtime_inputs_cover_tool_configs_model_mjcf_code_and_replays() -> None:
    clips = fidelity.default_clips()
    paths = fidelity._runtime_input_paths(clips)

    required = {
        Path(fidelity.__file__).resolve(),
        fidelity.JOINT_CONFIG.resolve(),
        fidelity.RETARGETING_CONFIG.resolve(),
        fidelity.POSE_MODEL.resolve(),
        fidelity.MUJOCO_MODEL.resolve(),
        (fidelity.PROJECT_ROOT / "models" / "humanoid" / "robot.xml").resolve(),
        (
            fidelity.PROJECT_ROOT
            / "src"
            / "robot_human_interface"
            / "retargeting"
            / "mujoco_ik.py"
        ).resolve(),
    }
    assert required <= paths
    assert any(path.name == "torso.obj" for path in paths)
    for clip in clips:
        assert clip.path.resolve() in paths
        if clip.calibration_video is not None:
            assert clip.calibration_video.resolve() in paths


def test_runtime_snapshot_validation_is_exact_and_rejects_malformed_hashes() -> None:
    clips = fidelity.default_clips()
    snapshot = _valid_runtime_snapshot(clips)

    assert fidelity._runtime_snapshot_violations(snapshot, clips) == {
        "missing_keys": [],
        "unexpected_keys": [],
        "invalid_sha256_keys": [],
    }

    missing_key = next(iter(snapshot))
    malformed = dict(snapshot)
    malformed.pop(missing_key)
    malformed["unexpected.file"] = "0" * 64
    another_key = next(iter(malformed))
    malformed[another_key] = "NOT-A-SHA256"
    violations = fidelity._runtime_snapshot_violations(malformed, clips)

    assert violations["missing_keys"] == [missing_key]
    assert violations["unexpected_keys"] == ["unexpected.file"]
    assert violations["invalid_sha256_keys"] == [another_key]


def test_report_gates_provenance_and_never_claims_fidelity_from_coverage() -> None:
    clips = fidelity.default_clips()
    snapshot = _valid_runtime_snapshot(clips)
    evaluations = [
        {"video": fidelity._project_path(clip.path), "coverage_passed": True}
        for clip in clips
    ]
    config = fidelity.load_retargeting_config(fidelity.RETARGETING_CONFIG)

    report = fidelity.build_report(
        evaluations,
        clips,
        stride=5,
        retargeting_config=config,
        command_argv=["python", "tools/evaluate_pose_fidelity.py", "--stride", "5"],
        provenance=_valid_provenance(snapshot),
        runtime_hashes_at_completion=snapshot,
    )

    assert report["coverage_passed"] is True
    assert report["provenance_complete"] is True
    assert report["measurement_complete"] is True
    assert report["measurement_status"] == "complete"
    assert report["report_kind"] == "raw_pose_measurement_diagnostic"
    assert report["fidelity_acceptance"] == "not_defined"
    assert "overall_passed" not in report
    assert "fidelity" in report["fidelity_acceptance_reason"]

    changed_snapshot = dict(snapshot)
    changed_snapshot[next(iter(changed_snapshot))] = "1" * 64
    changed = fidelity.build_report(
        evaluations,
        clips,
        stride=5,
        retargeting_config=config,
        command_argv=["python", "tools/evaluate_pose_fidelity.py"],
        provenance=_valid_provenance(snapshot),
        runtime_hashes_at_completion=changed_snapshot,
    )
    assert changed["measurement_complete"] is False
    assert "runtime_inputs_unchanged_during_run" in {
        gate["name"]
        for gate in changed["provenance_gates"]
        if not gate["passed"]
    }

    incomplete_snapshot = dict(snapshot)
    incomplete_snapshot.pop(next(iter(incomplete_snapshot)))
    incomplete = fidelity.build_report(
        evaluations,
        clips,
        stride=5,
        retargeting_config=config,
        command_argv=["python", "tools/evaluate_pose_fidelity.py"],
        provenance=_valid_provenance(incomplete_snapshot),
        runtime_hashes_at_completion=incomplete_snapshot,
    )
    assert incomplete["measurement_complete"] is False
    assert "runtime_snapshot_at_start_complete" in {
        gate["name"]
        for gate in incomplete["provenance_gates"]
        if not gate["passed"]
    }


def test_stale_command_frame_is_excluded_from_measurement_and_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "stale.mp4"
    video.write_bytes(b"video")

    class FakeSource:
        def __init__(self, path: Path, *, realtime: bool) -> None:
            assert path == video.resolve()
            assert realtime is False
            self.index = 0

        def read(self) -> object | None:
            if self.index >= 3:
                return None
            frame = SimpleNamespace(timestamp_s=float(self.index))
            self.index += 1
            return frame

        def close(self) -> None:
            pass

    class FakePose:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def estimate(self, frame: object) -> object:
            return frame

        def close(self) -> None:
            pass

    class FakeGeometric:
        is_calibrating = False

        def __init__(self, *_: object) -> None:
            pass

        def retarget(self, _skeleton: object, *, timestamp_s: float) -> object:
            return SimpleNamespace(stale=False, positions_rad=(timestamp_s,))

    class FakeIK(FakeGeometric):
        def __init__(self, *_: object) -> None:
            self.model = object()

        def retarget(self, _skeleton: object, *, timestamp_s: float) -> object:
            return SimpleNamespace(
                stale=timestamp_s == 1.0,
                positions_rad=(timestamp_s,),
            )

    class FakeFidelity:
        def mean_error_deg(self, _directions: object) -> float:
            return 1.0

    class FakeEvaluator:
        def __init__(self, _model: object) -> None:
            pass

        def evaluate(self, _skeleton: object, _positions: object) -> FakeFidelity:
            return FakeFidelity()

    monkeypatch.setattr(
        fidelity,
        "load_retargeting_config",
        lambda _: SimpleNamespace(auto_calibration_frames=0),
    )
    monkeypatch.setattr(fidelity, "load_joint_specs", lambda _: object())
    monkeypatch.setattr(fidelity, "GeometricRetargeter", FakeGeometric)
    monkeypatch.setattr(fidelity, "MujocoIKRetargeter", FakeIK)
    monkeypatch.setattr(fidelity, "MujocoPoseFidelityEvaluator", FakeEvaluator)
    monkeypatch.setattr(fidelity, "OpenCVVideoSource", FakeSource)
    monkeypatch.setattr(fidelity, "MediaPipePoseLandmarker", FakePose)

    result = fidelity.evaluate_video(video, stride=1, expected_frames=3)

    assert result["post_calibration_frames"] == 3
    assert result["evaluated_frames"] == 2
    assert result["stale_frames"] == 1
    assert result["stale_commands"] == {"geometric": 0, "ik": 1}
    assert result["metrics"]["arms"]["geometric"]["count"] == 2  # type: ignore[index]
    assert result["coverage"][  # type: ignore[index]
        "evaluated_nonstale_sample_fraction"
    ] == pytest.approx(2 / 3)
    assert result["coverage_passed"] is False
    assert result["measurement_complete"] is False


def test_main_records_custom_video_and_output_in_actual_command_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "custom video.mp4"
    output = tmp_path / "custom report.json"
    video.write_bytes(b"video")

    def fake_capture(clips: list[fidelity.ClipSpec]) -> dict[str, object]:
        return _valid_provenance(_valid_runtime_snapshot(clips))

    def fake_evaluate(path: Path, **_: object) -> dict[str, object]:
        return {
            "video": str(path.resolve()),
            "coverage_passed": True,
            "measurement_status": "complete",
            "metrics": {
                group: {"ik_mean_improvement_percent": None}
                for group in ("arms", "legs", "end_effectors")
            },
            "evaluated_frames": 1,
            "sampled_frames": 1,
        }

    monkeypatch.setattr(fidelity, "_capture_run_provenance", fake_capture)
    monkeypatch.setattr(
        fidelity,
        "_runtime_input_hashes",
        lambda clips: _valid_runtime_snapshot(clips),
    )
    monkeypatch.setattr(fidelity, "evaluate_video", fake_evaluate)
    argv = [
        "--video",
        str(video),
        "--stride",
        "7",
        "--output",
        str(output),
    ]

    exit_code = fidelity.main(argv)
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["command_argv"] == [
        sys.executable,
        "tools/evaluate_pose_fidelity.py",
        *argv,
    ]
    assert report["videos"][0]["video"] == str(video.resolve())
    assert report["measurement_complete"] is True
    assert report["fidelity_acceptance"] == "not_defined"
