from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import Event, Thread

import numpy as np
import pytest
import yaml

import robot_human_interface.experiments.recorder as recorder_module

from robot_human_interface.experiments import (
    ExperimentRecorder,
    ExperimentSample,
    ExperimentSpec,
    RecorderState,
    recover_interrupted_experiments,
    sha256_file,
)
from robot_human_interface.skeleton import JOINT_NAMES, LANDMARK_COUNT


NOW = datetime(2026, 8, 27, 17, 45, 0, tzinfo=timezone.utc)


def _spec(**overrides: object) -> ExperimentSpec:
    values: dict[str, object] = {
        "participant_code": "P001",
        "movement": "squat",
        "attempt": 1,
        "method_id": "baseline-ik",
        "seed": 42,
        "note": "test run",
        "consent": True,
        "record_video": False,
    }
    values.update(overrides)
    return ExperimentSpec(**values)


def _sample(sequence: int, *, rich: bool = False) -> ExperimentSample:
    if not rich:
        return ExperimentSample(timestamp_s=sequence / 30.0, sequence=sequence)
    landmarks_2d = np.arange(LANDMARK_COUNT * 2, dtype=np.float64).reshape(
        LANDMARK_COUNT, 2
    )
    landmarks_2d[0, 0] = np.nan
    landmarks_3d = np.zeros((LANDMARK_COUNT, 3), dtype=np.float64)
    visibility = np.ones(LANDMARK_COUNT, dtype=np.float64)
    presence = np.ones(LANDMARK_COUNT, dtype=np.float64)
    return ExperimentSample(
        timestamp_s=sequence / 30.0,
        sequence=sequence,
        landmarks_2d=landmarks_2d,
        landmarks_3d=landmarks_3d,
        visibility=visibility,
        presence=presence,
        raw_angles_rad=np.linspace(-0.5, 0.5, len(JOINT_NAMES)),
        safe_angles_rad=None,
        actual_angles_rad=np.zeros(len(JOINT_NAMES)),
        joint_velocities_rad_s=np.ones(len(JOINT_NAMES)),
        base_position_m=[0.0, 0.0, 0.72],
        base_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        center_of_mass_position_m=[0.0, 0.0, 0.4],
        right_foot_normal_force_n=51.0,
        left_foot_normal_force_n=49.0,
        contact_count=2,
        non_foot_ground_contact_count=0,
        support_intent="double_support",
        support_phase="standing",
        diagnostics={"margin_m": 0.03, "reason": None},
        tracking_quality=0.95,
        calibrating=False,
        calibration_progress=1.0,
        command_stale=False,
        safe_valid=False,
        free_base_active=True,
        balance_active=True,
    )


def _recorder(tmp_path: Path, **kwargs: object) -> ExperimentRecorder:
    return ExperimentRecorder(
        tmp_path / "experiments",
        utc_now=lambda: NOW,
        repo_root=tmp_path / "not-a-repository",
        recover_partials=False,
        **kwargs,
    )


def test_video_requires_explicit_consent() -> None:
    with pytest.raises(ValueError, match="explicit consent"):
        _spec(consent=False, record_video=True)


def test_129_samples_make_two_safe_npz_chunks_and_hashed_manifest(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "private" / "source-video.mp4"
    artifact.parent.mkdir()
    artifact.write_bytes(b"reference-video")
    private_source_path = (tmp_path / "private" / "user-video.mp4").resolve()
    recorder = _recorder(tmp_path)
    run_id = recorder.start(
        _spec(note=str((tmp_path / "private" / "notes.txt").resolve())),
        session_config={
            "source": {"path": private_source_path, "source_id": "stock-01"},
            "retargeting": "ik",
        },
        source_id="stock-01",
        artifact_files={"source": artifact},
        software_versions={"robot-human-interface": "test"},
        git_metadata={"revision": "abc", "dirty": False, "dirty_hash": None},
        run_id="stable-test-run",
    )
    assert run_id == "stable-test-run"
    for sequence in range(129):
        assert recorder.append(_sample(sequence, rich=sequence == 0))
    summary = recorder.stop("manual")

    assert summary.state is RecorderState.COMPLETE
    assert summary.sample_count == 129
    assert summary.accepted_samples == 129
    assert summary.dropped_samples == 0
    assert summary.chunk_count == 2
    assert summary.path.name == run_id
    assert not summary.path.name.endswith(".partial")

    with (summary.path / "manifest.yaml").open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    assert manifest["complete"] is True
    assert manifest["incomplete"] is False
    assert manifest["counts"]["written_samples"] == 129
    assert [chunk["samples"] for chunk in manifest["chunks"]] == [128, 1]
    assert manifest["files"]["source"]["sha256"] == hashlib.sha256(
        b"reference-video"
    ).hexdigest()

    rendered_manifest = (summary.path / "manifest.yaml").read_text(encoding="utf-8")
    assert str(tmp_path.resolve()) not in rendered_manifest
    assert str(private_source_path) not in rendered_manifest
    assert manifest["session_config"]["source"]["path"] == private_source_path.name

    for chunk_metadata in manifest["chunks"]:
        chunk_path = summary.path / chunk_metadata["file"]
        assert hashlib.sha256(chunk_path.read_bytes()).hexdigest() == chunk_metadata["sha256"]
        with np.load(chunk_path, allow_pickle=False) as chunk:
            for key in chunk.files:
                assert not chunk[key].dtype.hasobject, key

    with np.load(summary.path / "chunks" / "000000.npz", allow_pickle=False) as first:
        assert first["sequence"].shape == (128,)
        assert first["landmarks_2d"].shape == (128, LANDMARK_COUNT, 2)
        assert np.isnan(first["landmarks_2d"][0, 0, 0])
        assert not first["landmarks_2d_mask"][0, 0, 0]
        assert not first["landmark_valid_mask"][0, 0]
        assert np.isnan(first["safe_angles_rad"][0]).all()
        assert not first["safe_angles_rad_mask"][0].any()
        assert first["actual_angles_rad_mask"][0].all()
        assert np.isnan(first["landmarks_3d"][1]).all()
        assert not first["landmarks_3d_mask"][1].any()
        assert first["diagnostics_json"].dtype.kind == "U"

    with np.load(summary.path / "chunks" / "000001.npz", allow_pickle=False) as second:
        assert second["sequence"].tolist() == [128]


def test_events_include_run_source_sequence_and_sanitize_paths(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    run_id = recorder.start(_spec(), source_id="camera-0", run_id="events-run")
    private_path = (tmp_path / "secret" / "frame.json").resolve()
    assert recorder.record_event(
        "TRACKING_CHANGED",
        sequence=7,
        level="warning",
        subsystem="tracking",
        details={"debug_path": private_path},
    )
    summary = recorder.stop("test")

    lines = (summary.path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    payloads = [__import__("json").loads(line) for line in lines]
    event = next(item for item in payloads if item["code"] == "TRACKING_CHANGED")
    assert event["run_id"] == run_id
    assert event["source_id"] == "camera-0"
    assert event["sequence"] == 7
    assert event["details"]["debug_path"] == private_path.name
    assert str(tmp_path.resolve()) not in lines[1]


def test_event_publication_cannot_be_overtaken_by_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_put_started = Event()
    release_event_put = Event()
    stop_enqueued = Event()

    class GatedQueue(Queue):
        def put_nowait(self, item: object) -> None:
            if type(item).__name__ == "_EventItem":
                event_put_started.set()
                assert release_event_put.wait(timeout=5.0)
            super().put_nowait(item)

        def put(
            self,
            item: object,
            block: bool = True,
            timeout: float | None = None,
        ) -> None:
            super().put(item, block=block, timeout=timeout)
            if type(item).__name__ == "_StopItem":
                stop_enqueued.set()

    monkeypatch.setattr(recorder_module, "Queue", GatedQueue)
    recorder = _recorder(tmp_path, queue_capacity=4)
    recorder.start(_spec(), run_id="event-stop-race-run")
    event_result: list[bool] = []
    stop_result: list[object] = []
    producer = Thread(
        target=lambda: event_result.append(recorder.record_event("RACING_EVENT"))
    )
    producer.start()
    assert event_put_started.wait(timeout=2.0)

    stopper = Thread(target=lambda: stop_result.append(recorder.stop("test")))
    stopper.start()
    # stop() must remain behind the in-flight event publication rather than
    # enqueueing its sentinel while record_event() is paused in put_nowait().
    assert not stop_enqueued.wait(timeout=0.2)
    release_event_put.set()
    producer.join(timeout=2.0)
    stopper.join(timeout=5.0)

    assert not producer.is_alive()
    assert not stopper.is_alive()
    assert event_result == [True]
    summary = stop_result[0]
    payloads = [
        json.loads(line)
        for line in (summary.path / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    codes = [payload["code"] for payload in payloads]
    assert codes.index("RACING_EVENT") < codes.index("RECORDER_STOPPED")


def test_partial_recovery_marks_interrupted_without_deleting_data(tmp_path: Path) -> None:
    partial = tmp_path / "experiments" / "2026-08-27" / "crashed.partial"
    chunks = partial / "chunks"
    chunks.mkdir(parents=True)
    sample_bytes = b"still here"
    (chunks / "000000.npz").write_bytes(sample_bytes)
    with (partial / "manifest.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "run_id": "crashed",
                "state": "recording",
                "complete": False,
            },
            stream,
        )

    recovered = recover_interrupted_experiments(tmp_path / "experiments")

    assert recovered == (partial.resolve(),)
    assert (chunks / "000000.npz").read_bytes() == sample_bytes
    with (partial / "manifest.yaml").open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    assert manifest["state"] == "interrupted"
    assert manifest["stop_reason"] == "interrupted"
    assert manifest["incomplete"] is True
    assert "RECORDER_INTERRUPTED_RECOVERED" in (
        partial / "events.jsonl"
    ).read_text(encoding="utf-8")


def test_recovery_preserves_terminal_error_partial(tmp_path: Path) -> None:
    partial = tmp_path / "experiments" / "2026-08-27" / "failed.partial"
    partial.mkdir(parents=True)
    manifest_path = partial / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "run_id": "failed",
                "state": "error",
                "complete": False,
                "incomplete": True,
                "ended_utc": "2026-08-27T17:46:00.000Z",
                "stop_reason": "recorder_error",
                "errors": ["io_error: disk full"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_bytes()

    assert recover_interrupted_experiments(tmp_path / "experiments") == ()
    assert manifest_path.read_bytes() == original_manifest
    assert not (partial / "events.jsonl").exists()


def test_default_artifacts_hash_recursive_mujoco_inputs_without_paths(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models" / "humanoid"
    (model_root / "meshes").mkdir(parents=True)
    (model_root / "textures").mkdir()
    (model_root / "robot.xml").write_text(
        '<mujoco><asset><mesh file="meshes/body.obj"/></asset></mujoco>',
        encoding="utf-8",
    )
    (model_root / "meshes" / "body.obj").write_bytes(b"mesh-content")
    (model_root / "textures" / "skin.png").write_bytes(b"texture-content")
    recorder = ExperimentRecorder(
        tmp_path / "experiments",
        utc_now=lambda: NOW,
        repo_root=tmp_path,
        recover_partials=False,
    )
    recorder.start(_spec(), run_id="mujoco-artifacts-run")
    summary = recorder.stop("test")

    manifest_path = summary.path / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    assert files["model/humanoid/robot.xml"]["sha256"] == sha256_file(
        model_root / "robot.xml"
    )
    assert files["model/humanoid/meshes/body.obj"]["sha256"] == sha256_file(
        model_root / "meshes" / "body.obj"
    )
    assert files["model/humanoid/textures/skin.png"]["sha256"] == sha256_file(
        model_root / "textures" / "skin.png"
    )
    assert str(tmp_path.resolve()) not in manifest_path.read_text(encoding="utf-8")


def test_git_dirty_hash_includes_untracked_file_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    untracked = tmp_path / "new-input.yaml"
    untracked.write_text("value: one\n", encoding="utf-8")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        arguments = command[3:]
        outputs = {
            ("rev-parse", "HEAD"): b"abc123\n",
            ("status", "--porcelain=v1", "-z"): b"?? new-input.yaml\0",
            ("diff", "--binary", "HEAD"): b"",
            ("diff", "--binary", "--cached"): b"",
            ("ls-files", "--others", "--exclude-standard", "-z"): (
                b"new-input.yaml\0"
            ),
        }
        return subprocess.CompletedProcess(command, 0, stdout=outputs[tuple(arguments)])

    monkeypatch.setattr(recorder_module.subprocess, "run", fake_run)
    first = recorder_module._git_metadata(tmp_path)
    untracked.write_text("value: two\n", encoding="utf-8")
    second = recorder_module._git_metadata(tmp_path)

    assert first["dirty"] is True
    assert first["dirty_hash"] != second["dirty_hash"]


def test_disk_error_stays_partial_and_does_not_escape_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path, chunk_size=1)

    def fail_write(_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(recorder, "_write_chunk", fail_write)
    recorder.start(_spec(), run_id="disk-error-run")
    assert recorder.append(_sample(0))
    summary = recorder.stop("manual")

    assert summary.state is RecorderState.ERROR
    assert summary.incomplete
    assert "disk full" in (summary.error or "")
    assert summary.path.name.endswith(".partial")
    assert summary.path.exists()
    with (summary.path / "manifest.yaml").open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    assert manifest["complete"] is False
    assert manifest["counts"]["written_samples"] == 0


def test_queue_overflow_marks_run_incomplete_without_blocking_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(tmp_path, queue_capacity=1)
    gate = Event()
    original_writer = recorder._writer_main

    def delayed_writer() -> None:
        assert gate.wait(timeout=5.0)
        original_writer()

    monkeypatch.setattr(recorder, "_writer_main", delayed_writer)
    recorder.start(_spec(), run_id="queue-overflow-run")
    assert recorder.append(_sample(0))
    assert recorder.append(_sample(1)) is False
    gate.set()
    summary = recorder.stop("manual")

    assert summary.state is RecorderState.ERROR
    assert summary.incomplete
    assert summary.accepted_samples == 1
    assert summary.dropped_samples == 1
    assert "sample_queue_overflow" in (summary.error or "")
    assert summary.path.name.endswith(".partial")


def test_mark_incomplete_preserves_partial_error_manifest_with_companion_reason(
    tmp_path: Path,
) -> None:
    recorder = _recorder(tmp_path)
    recorder.start(_spec(), run_id="companion-io-run")
    assert recorder.append(_sample(0))

    recorder.mark_incomplete("video_queue_overflow")
    summary = recorder.stop("manual")

    assert summary.state is RecorderState.ERROR
    assert summary.incomplete
    assert summary.path.name.endswith(".partial")
    manifest = yaml.safe_load(
        (summary.path / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["state"] == "error"
    assert manifest["complete"] is False
    assert manifest["incomplete"] is True
    assert "companion_io:video_queue_overflow" in manifest["errors"]


def test_media_scaffolding_is_opt_in_and_uses_plain_basenames(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder.start(
        _spec(record_video=True),
        run_id="video-run",
    )
    media_path = recorder.reserve_media_path("processed.mp4")
    assert media_path.parent.name == "media"
    assert media_path.name == "processed.mp4"
    media_path.write_bytes(b"camera-video-placeholder")
    with pytest.raises(ValueError, match="basename"):
        recorder.reserve_media_path(os.path.join("nested", "processed.mp4"))
    summary = recorder.stop("manual")
    manifest = yaml.safe_load((summary.path / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["files"]["media/processed.mp4"]["sha256"] == sha256_file(media_path if media_path.exists() else summary.path / "media" / "processed.mp4")
