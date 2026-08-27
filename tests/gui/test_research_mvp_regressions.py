from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import radians
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QDialog

from robot_human_interface.experiments import (
    ExperimentRecorder,
    ExperimentSpec,
    RecorderState,
    RecorderSummary,
)
from robot_human_interface.gui.main_window import MainWindow, RobotCallbacks
from robot_human_interface.gui.resources import ResourceLocator, SourceItem, UserSourceStore
from robot_human_interface.gui.runtime import (
    ReadinessReason,
    RobotReadiness,
    RobotUiState,
    RuntimeStatus,
)
from robot_human_interface.gui.worker import DemoSession, PipelineWorker
from robot_human_interface.playback import PlaybackDiscontinuity, PlaybackState
from robot_human_interface.protocol import (
    OperatorSafetyAcknowledgement,
    SafeRobotController,
    finalize_safe_command,
)
from robot_human_interface.skeleton import JOINT_NAMES, RobotJointCommand


class _StreamingSession:
    def __init__(self, source: SourceItem) -> None:
        self.source = source
        self.config = {"source": {"source_id": source.source_id}}
        self.running = False
        self.closed = False
        self.sequence = 0

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def pause(self) -> None:
        self.running = False

    def resume(self) -> None:
        self.running = True

    def close(self) -> None:
        self.running = False
        self.closed = True

    def step(self):
        if not self.running:
            return None
        snapshot = SimpleNamespace(
            sequence=self.sequence,
            timestamp_s=monotonic(),
            status="RUNNING",
            source=self.source,
            frame=None,
            skeleton=None,
            raw_command=None,
            safe_command=None,
            tracking_quality=0.9,
            telemetry={
                "calibrating": False,
                "calibration_progress": 1.0,
                "free_base_active": True,
                "balance_active": True,
            },
            playback=None,
        )
        self.sequence += 1
        return snapshot


class _NonSeekablePlaybackSession(_StreamingSession):
    def step(self):
        snapshot = super().step()
        if snapshot is not None:
            snapshot.playback = PlaybackState(
                seekable=False,
                position_s=0.0,
                duration_s=None,
                frame_index=0,
                frame_count=None,
                fps=30.0,
                eof=False,
            )
        return snapshot

    def request_seek(self, _position_s: float) -> None:
        raise RuntimeError("source is not seekable")


class _BlockingErrorRecorder(ExperimentRecorder):
    """Minimal real-type recorder double with deliberately slow boundaries."""

    def __init__(self, path: Path) -> None:
        # Deliberately do not initialize ExperimentRecorder's filesystem writer.
        self.path = path
        self._test_state = RecorderState.IDLE
        self.start_entered = Event()
        self.start_release = Event()
        self.stop_entered = Event()
        self.stop_release = Event()
        self.sample_count = 0

    @property
    def state(self) -> RecorderState:
        return self._test_state

    @property
    def run_id(self) -> str:
        return "slow-run"

    @property
    def summary(self) -> RecorderSummary:
        return RecorderSummary(
            run_id=self.run_id,
            state=self._test_state,
            path=self.path,
            sample_count=self.sample_count,
            accepted_samples=self.sample_count,
            dropped_samples=0,
            chunk_count=0,
            event_count=0,
            started_utc="2026-08-27T00:00:00Z",
            ended_utc=None,
            stop_reason="writer_error" if self._test_state is RecorderState.ERROR else None,
            incomplete=self._test_state is RecorderState.ERROR,
            error="disk failure" if self._test_state is RecorderState.ERROR else None,
        )

    def start(self, *_args, **_kwargs) -> str:
        self._test_state = RecorderState.PREPARING
        self.start_entered.set()
        if not self.start_release.wait(5.0):
            raise TimeoutError("test did not release recorder.start")
        self._test_state = RecorderState.RECORDING
        return self.run_id

    def append(self, _snapshot: object) -> bool:
        self.sample_count += 1
        return True

    def record_event(self, *_args, **_kwargs) -> bool:
        return True

    def stop(self, _reason: str = "manual", **_kwargs) -> RecorderSummary:
        self._test_state = RecorderState.FINALIZING
        self.stop_entered.set()
        if not self.stop_release.wait(5.0):
            raise TimeoutError("test did not release recorder.stop")
        self._test_state = RecorderState.ERROR
        return self.summary


class _GatedWriterErrorRecorder(ExperimentRecorder):
    """Recorder double whose writer fails only after the test opens its gate."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._test_state = RecorderState.IDLE
        self.fail_after_arm = Event()
        self.error_observed = Event()
        self.sample_count = 0

    @property
    def state(self) -> RecorderState:
        return self._test_state

    @property
    def run_id(self) -> str:
        return "armed-writer-error"

    @property
    def summary(self) -> RecorderSummary:
        failed = self._test_state is RecorderState.ERROR
        return RecorderSummary(
            run_id=self.run_id,
            state=self._test_state,
            path=self.path,
            sample_count=self.sample_count,
            accepted_samples=self.sample_count,
            dropped_samples=1 if failed else 0,
            chunk_count=0,
            event_count=0,
            started_utc="2026-08-27T00:00:00Z",
            ended_utc="2026-08-27T00:00:01Z" if failed else None,
            stop_reason="writer_error" if failed else None,
            incomplete=failed,
            error="simulated disk failure" if failed else None,
        )

    def start(self, *_args, **_kwargs) -> str:
        self._test_state = RecorderState.RECORDING
        return self.run_id

    def append(self, _snapshot: object) -> bool:
        if self.fail_after_arm.is_set():
            self._test_state = RecorderState.ERROR
            self.error_observed.set()
            return False
        self.sample_count += 1
        return True

    def record_event(self, *_args, **_kwargs) -> bool:
        return True

    def stop(self, _reason: str = "manual", **_kwargs) -> RecorderSummary:
        self._test_state = RecorderState.ERROR
        return self.summary


class _RepeatedEofSession(_StreamingSession):
    def step(self):
        if not self.running:
            return None
        return SimpleNamespace(
            sequence=41,
            timestamp_s=monotonic(),
            status="ENDED",
            source=self.source,
            frame=None,
            skeleton=None,
            raw_command=None,
            safe_command=None,
            tracking_quality=0.0,
            telemetry={},
            playback=PlaybackState(
                seekable=True,
                position_s=2.0,
                duration_s=2.0,
                frame_index=59,
                frame_count=60,
                fps=30.0,
                eof=True,
            ),
        )


@dataclass
class _PayloadTransport:
    connected: bool = False
    payloads: list[str] = field(default_factory=list)

    def connect(self) -> None:
        self.connected = True

    def send(self, payload: str) -> None:
        self.payloads.append(payload)

    def close(self) -> None:
        self.connected = False


class _BarrierSafetySession(_StreamingSession):
    """Emit one baseline frame, then form one gated safety-critical frame."""

    def __init__(
        self,
        source: SourceItem,
        *,
        next_status: str = "RUNNING",
        discontinuity: PlaybackDiscontinuity | None = None,
    ) -> None:
        super().__init__(source)
        self.next_status = next_status
        self.discontinuity = discontinuity
        self.ready_for_baseline = Event()
        self.release_baseline = Event()
        self.baseline_emitted = Event()
        self.ready_for_next = Event()
        self.release_next = Event()
        self.next_emitted = Event()

    @staticmethod
    def _safe_command(degrees: float, sequence: int):
        positions = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        positions[0] = radians(degrees)
        command = RobotJointCommand.humanoid(float(sequence), positions, 1.0)
        return finalize_safe_command(
            command,
            free_base_active=True,
            balance_active=True,
        )

    def step(self):
        if not self.running:
            return None
        if not self.baseline_emitted.is_set():
            self.ready_for_baseline.set()
            if not self.release_baseline.is_set():
                return None
            self.baseline_emitted.set()
            self.ready_for_next.set()
            return self._snapshot(sequence=0, status="RUNNING", degrees=7.0)
        if self.next_emitted.is_set() or not self.release_next.is_set():
            return None
        self.next_emitted.set()
        return self._snapshot(
            sequence=1,
            status=self.next_status,
            degrees=37.0,
            discontinuity=self.discontinuity,
        )

    def _snapshot(
        self,
        *,
        sequence: int,
        status: str,
        degrees: float,
        discontinuity: PlaybackDiscontinuity | None = None,
    ) -> SimpleNamespace:
        playback = None
        if status == "ENDED" or discontinuity is not None:
            playback = PlaybackState(
                seekable=True,
                position_s=0.0 if discontinuity is not None else 2.0,
                duration_s=2.0,
                frame_index=0 if discontinuity is not None else 59,
                frame_count=60,
                fps=30.0,
                loop_enabled=discontinuity is PlaybackDiscontinuity.LOOP_WRAP,
                loop_start_s=0.0,
                loop_end_s=1.5 if discontinuity is not None else None,
                eof=status == "ENDED",
                discontinuity_reason=discontinuity,
            )
        return SimpleNamespace(
            sequence=sequence,
            timestamp_s=monotonic(),
            status=status,
            source=self.source,
            frame=None,
            skeleton=None,
            raw_command=None,
            safe_valid=True,
            safe_command=self._safe_command(degrees, sequence),
            tracking_quality=1.0,
            telemetry={
                "calibrating": False,
                "calibration_progress": 1.0,
                "free_base_active": True,
                "balance_active": True,
            },
            playback=playback,
        )


class _RecorderErrorSafetySession(_BarrierSafetySession):
    """Gate the first valid snapshot, then stream only when explicitly released."""

    def __init__(self, source: SourceItem) -> None:
        super().__init__(source)
        self.release_stream = Event()
        self.restore_emitted = Event()
        self.continue_after_arm = Event()

    def step(self):
        if not self.running:
            return None
        if not self.baseline_emitted.is_set():
            self.ready_for_baseline.set()
            if not self.release_baseline.is_set():
                return None
            self.baseline_emitted.set()
            self.sequence = 1
            return self._snapshot(sequence=0, status="RUNNING", degrees=7.0)
        if not self.release_stream.is_set():
            return None
        if self.restore_emitted.is_set() and not self.continue_after_arm.is_set():
            return None
        snapshot = self._snapshot(
            sequence=self.sequence,
            status="RUNNING",
            degrees=11.0,
        )
        self.sequence += 1
        self.restore_emitted.set()
        return snapshot


def _source(kind: str = "stock") -> SourceItem:
    path = None if kind == "camera" else "input.mp4"
    return SourceItem(f"{kind}:regression", "Regression", kind, path=path)


def _snapshot(source: SourceItem, sequence: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        sequence=sequence,
        timestamp_s=monotonic(),
        status="RUNNING",
        source=source,
        frame=None,
        skeleton=None,
        raw_command=None,
        safe_command=None,
        tracking_quality=0.9,
        telemetry={
            "calibrating": False,
            "calibration_progress": 1.0,
            "free_base_active": True,
            "balance_active": True,
        },
        playback=None,
    )


def test_slow_recorder_boundaries_do_not_block_worker_and_error_completes(qtbot, tmp_path: Path) -> None:
    recorder = _BlockingErrorRecorder(tmp_path / "slow-run")
    worker = PipelineWorker(session_factory=_StreamingSession, experiment_root=tmp_path)
    worker._recorder = recorder
    sequences: list[int] = []
    states: list[str] = []
    completions: list[RecorderSummary] = []
    worker.snapshot_ready.connect(lambda snapshot: sequences.append(snapshot.sequence))
    worker.recorder_state_changed.connect(lambda state, _payload: states.append(state))
    worker.experiment_completed.connect(completions.append)
    worker.start()
    worker.select_source(_source())
    worker.start_pipeline()

    try:
        qtbot.waitUntil(lambda: len(sequences) >= 3, timeout=1500)
        worker.start_recording(
            ExperimentSpec("P001", "squat", 1, "baseline", 7, consent=True)
        )
        qtbot.waitUntil(recorder.start_entered.is_set, timeout=1000)
        preparing_sequence = sequences[-1]
        qtbot.waitUntil(lambda: sequences[-1] >= preparing_sequence + 3, timeout=1000)

        recorder.start_release.set()
        qtbot.waitUntil(lambda: "RECORDING" in states, timeout=1500)
        worker.stop_recording("manual")
        qtbot.waitUntil(recorder.stop_entered.is_set, timeout=1000)
        finalizing_sequence = sequences[-1]
        qtbot.waitUntil(lambda: sequences[-1] >= finalizing_sequence + 3, timeout=1000)

        recorder.stop_release.set()
        qtbot.waitUntil(lambda: states and states[-1] == "ERROR", timeout=1500)
        qtbot.waitUntil(lambda: len(completions) == 1, timeout=1000)
        assert completions[0].state is RecorderState.ERROR
        assert worker._pipeline_state == "RUNNING"
        assert worker.isRunning()
    finally:
        recorder.start_release.set()
        recorder.stop_release.set()
        assert worker.shutdown_and_wait(2500)


def test_recorder_writer_error_finalizes_without_disarming_robot(
    qtbot,
    tmp_path: Path,
) -> None:
    sessions: list[_RecorderErrorSafetySession] = []
    transport = _PayloadTransport()
    controller = SafeRobotController(transport, max_command_age_s=5.0)
    recorder = _GatedWriterErrorRecorder(tmp_path / "armed-writer-error")

    def factory(source: SourceItem) -> _RecorderErrorSafetySession:
        session = _RecorderErrorSafetySession(source)
        sessions.append(session)
        return session

    worker = PipelineWorker(
        session_factory=factory,
        robot_factory=lambda _endpoint: controller,
        experiment_root=tmp_path,
        max_snapshot_age_s=5.0,
    )
    worker._recorder = recorder
    recorder_states: list[str] = []
    completions: list[RecorderSummary] = []
    worker.recorder_state_changed.connect(
        lambda state, _payload: recorder_states.append(state)
    )
    worker.experiment_completed.connect(completions.append)
    worker.start()
    worker.select_source(_source())
    worker.start_pipeline()

    try:
        qtbot.waitUntil(lambda: len(sessions) == 1, timeout=1000)
        session = sessions[0]
        qtbot.waitUntil(session.ready_for_baseline.is_set, timeout=1000)
        worker.connect_robot()
        qtbot.waitUntil(
            lambda: controller.status().state.value == "connected_disarmed",
            timeout=1000,
        )
        session.release_baseline.set()
        qtbot.waitUntil(
            lambda: worker._latest_snapshot is not None
            and worker._latest_snapshot.sequence == 0,
            timeout=1000,
        )

        worker.start_recording(
            ExperimentSpec("P002", "reach", 1, "writer-fault", 11, consent=True)
        )
        qtbot.waitUntil(lambda: "RECORDING" in recorder_states, timeout=1000)
        session.release_stream.set()
        qtbot.waitUntil(session.restore_emitted.is_set, timeout=1000)
        qtbot.waitUntil(
            lambda: worker.robot_readiness is not None and worker.robot_readiness.ready,
            timeout=1000,
        )
        worker.arm_robot(
            acknowledgement=OperatorSafetyAcknowledgement(True, True, True)
        )
        qtbot.waitUntil(lambda: controller.is_armed, timeout=1000)
        qtbot.waitUntil(lambda: controller.status().positions_sent >= 1, timeout=1000)
        generation_before_error = controller.status().command_generation
        sends_before_error = controller.status().positions_sent

        recorder.fail_after_arm.set()
        session.continue_after_arm.set()
        qtbot.waitUntil(recorder.error_observed.is_set, timeout=1000)
        qtbot.waitUntil(
            lambda: recorder_states and recorder_states[-1] == "ERROR",
            timeout=1000,
        )
        qtbot.waitUntil(lambda: len(completions) == 1, timeout=1000)

        summary = completions[0]
        assert summary.state is RecorderState.ERROR
        assert summary.incomplete
        assert controller.is_armed
        qtbot.waitUntil(
            lambda: controller.status().positions_sent > sends_before_error,
            timeout=1500,
        )
        assert controller.status().command_generation > generation_before_error
        assert controller.is_armed
        assert len(transport.payloads) > sends_before_error
    finally:
        recorder.fail_after_arm.set()
        if sessions:
            sessions[0].release_baseline.set()
            sessions[0].release_stream.set()
            sessions[0].continue_after_arm.set()
        assert worker.shutdown_and_wait(2500)


@pytest.mark.parametrize(
    ("session_type", "source_kind", "warning_code"),
    (
        (_StreamingSession, "camera", "PLAYBACK_UNSUPPORTED"),
        (_NonSeekablePlaybackSession, "stock", "PLAYBACK_UNSUPPORTED"),
    ),
)
def test_unsupported_playback_warns_without_pausing_or_invalidating_pipeline(
    qtbot,
    session_type,
    source_kind: str,
    warning_code: str,
) -> None:
    worker = PipelineWorker(session_factory=session_type)
    states: list[str] = []
    events: list[dict[str, object]] = []
    sequences: list[int] = []
    worker.state_changed.connect(states.append)
    worker.event_ready.connect(events.append)
    worker.snapshot_ready.connect(lambda snapshot: sequences.append(snapshot.sequence))
    worker.start()
    worker.select_source(_source(source_kind))
    worker.start_pipeline()

    try:
        qtbot.waitUntil(lambda: len(sequences) >= 2, timeout=1500)
        prior_sequence = sequences[-1]
        states.clear()
        worker.seek(1.25)
        qtbot.waitUntil(
            lambda: warning_code in {str(event.get("event_code")) for event in events},
            timeout=1000,
        )
        qtbot.waitUntil(lambda: sequences[-1] > prior_sequence, timeout=1000)
        assert "PAUSED" not in states
        assert "DEGRADED" not in states
        assert worker._pipeline_state == "RUNNING"
        assert worker._latest_snapshot is not None
    finally:
        assert worker.shutdown_and_wait(2000)


def test_seekable_eof_emits_ended_once_even_when_session_repeats_snapshot(qtbot) -> None:
    worker = PipelineWorker(session_factory=_RepeatedEofSession)
    states: list[str] = []
    ended_snapshots: list[object] = []
    worker.state_changed.connect(states.append)
    worker.snapshot_ready.connect(
        lambda snapshot: ended_snapshots.append(snapshot)
        if str(snapshot.status).upper() == "ENDED"
        else None
    )
    worker.start()
    worker.select_source(_source())
    worker.start_pipeline()

    try:
        qtbot.waitUntil(lambda: "ENDED" in states, timeout=1500)
        qtbot.wait(150)
        assert states.count("ENDED") == 1
        assert len(ended_snapshots) == 1
        assert worker._pipeline_state == "ENDED"
        assert worker._session is not None
    finally:
        assert worker.shutdown_and_wait(2000)


def _start_armed_barrier_worker(
    qtbot,
    *,
    next_status: str = "RUNNING",
    discontinuity: PlaybackDiscontinuity | None = None,
) -> tuple[PipelineWorker, _BarrierSafetySession, SafeRobotController, _PayloadTransport]:
    sessions: list[_BarrierSafetySession] = []
    transport = _PayloadTransport()
    controller = SafeRobotController(
        transport,
        rate_hz=10.0,
        max_command_age_s=5.0,
    )

    def factory(source: SourceItem) -> _BarrierSafetySession:
        session = _BarrierSafetySession(
            source,
            next_status=next_status,
            discontinuity=discontinuity,
        )
        sessions.append(session)
        return session

    worker = PipelineWorker(
        session_factory=factory,
        robot_factory=lambda _endpoint: controller,
    )
    worker.start()
    worker.select_source(_source())
    worker.start_pipeline()
    worker.connect_robot()
    qtbot.waitUntil(lambda: len(sessions) == 1, timeout=1000)
    session = sessions[0]
    qtbot.waitUntil(session.ready_for_baseline.is_set, timeout=1000)
    qtbot.waitUntil(
        lambda: controller.status().state.value == "connected_disarmed",
        timeout=1000,
    )

    session.release_baseline.set()
    qtbot.waitUntil(lambda: controller.status().command_generation == 1, timeout=1000)
    qtbot.waitUntil(
        lambda: worker.robot_readiness is not None and worker.robot_readiness.ready,
        timeout=1000,
    )
    worker.arm_robot(
        acknowledgement=OperatorSafetyAcknowledgement(True, True, True)
    )
    qtbot.waitUntil(lambda: controller.is_armed, timeout=1000)
    qtbot.waitUntil(lambda: bool(transport.payloads), timeout=1000)
    qtbot.waitUntil(session.ready_for_next.is_set, timeout=1000)
    return worker, session, controller, transport


@pytest.mark.parametrize(
    ("terminal_status", "expected_pipeline_state"),
    (("ENDED", "ENDED"), ("ERROR", "DEGRADED")),
)
def test_terminal_snapshot_invalidates_before_accepting_its_unique_safe_command(
    qtbot,
    terminal_status: str,
    expected_pipeline_state: str,
) -> None:
    worker, session, controller, transport = _start_armed_barrier_worker(
        qtbot,
        next_status=terminal_status,
    )
    generation_before = controller.status().command_generation

    try:
        session.release_next.set()
        qtbot.waitUntil(session.next_emitted.is_set, timeout=1000)
        qtbot.waitUntil(
            lambda: worker._pipeline_state == expected_pipeline_state,
            timeout=1000,
        )
        qtbot.waitUntil(
            lambda: controller.status().last_disarm_reason
            == f"pipeline_{terminal_status.lower()}",
            timeout=1000,
        )
        assert controller.status().command_generation == generation_before
        assert all('"params":[37.0,' not in payload for payload in transport.payloads)
    finally:
        session.release_next.set()
        assert worker.shutdown_and_wait(2000)


def test_loop_wrap_invalidates_before_accepting_its_unique_safe_command(qtbot) -> None:
    worker, session, controller, transport = _start_armed_barrier_worker(
        qtbot,
        discontinuity=PlaybackDiscontinuity.LOOP_WRAP,
    )
    generation_before = controller.status().command_generation
    events: list[dict[str, object]] = []
    worker.event_ready.connect(events.append)

    try:
        session.release_next.set()
        qtbot.waitUntil(session.next_emitted.is_set, timeout=1000)
        qtbot.waitUntil(
            lambda: controller.status().last_disarm_reason == "playback_loop_wrap",
            timeout=1000,
        )
        qtbot.waitUntil(
            lambda: "PLAYBACK_DISCONTINUITY"
            in {str(event.get("event_code")) for event in events},
            timeout=1000,
        )
        assert worker._pipeline_state == "RUNNING"
        assert controller.status().command_generation == generation_before
        assert all('"params":[37.0,' not in payload for payload in transport.payloads)
    finally:
        session.release_next.set()
        assert worker.shutdown_and_wait(2000)


def test_paused_overlay_survives_late_snapshot(qtbot, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    worker = PipelineWorker(session_factory=DemoSession)
    window = MainWindow(
        locator=ResourceLocator(root),
        user_store=UserSourceStore(tmp_path / "app-data"),
        worker=worker,
        log_dir=tmp_path / "logs",
    )
    qtbot.addWidget(window)

    try:
        window._on_pipeline_state("PAUSED")
        assert "ПАУЗА" in window.preview._overlay_text
        window._on_snapshot(_snapshot(_source()))
        assert window._pipeline_state == "PAUSED"
        assert "ПАУЗА" in window.preview._overlay_text
    finally:
        window.close()
        qtbot.waitUntil(lambda: not worker.isRunning(), timeout=2500)


def test_recorder_error_and_completion_unlock_window_controls(qtbot, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    worker = PipelineWorker(session_factory=DemoSession)
    window = MainWindow(
        locator=ResourceLocator(root),
        user_store=UserSourceStore(tmp_path / "app-data"),
        worker=worker,
        log_dir=tmp_path / "logs",
    )
    qtbot.addWidget(window)
    summary = RecorderSummary(
        run_id="failed-run",
        state=RecorderState.ERROR,
        path=tmp_path / "failed-run",
        sample_count=0,
        accepted_samples=0,
        dropped_samples=0,
        chunk_count=0,
        event_count=0,
        started_utc="2026-08-27T00:00:00Z",
        ended_utc="2026-08-27T00:00:01Z",
        stop_reason="writer_error",
        incomplete=True,
        error="disk failure",
    )

    try:
        window._on_recorder_state("PREPARING", None)
        assert window._recording_active
        assert not window.source_panel.add_button.isEnabled()

        window._on_recorder_state("ERROR", summary)
        window._on_experiment_complete(summary)
        assert not window._recording_active
        assert window.source_panel.add_button.isEnabled()
        assert window.telemetry.reset_button.isEnabled()
        assert window.telemetry.calibrate_button.isEnabled()
    finally:
        window.close()
        qtbot.waitUntil(lambda: not worker.isRunning(), timeout=2500)


def test_arm_callback_type_error_is_not_retried(qtbot, tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def arm(*args, **kwargs):
        calls.append((args, kwargs))
        raise TypeError("adapter failed after starting its arm operation")

    readiness = RobotReadiness(
        ready=True,
        reason_code=ReadinessReason.READY,
        reason="",
        evaluated_at_s=monotonic(),
        runtime=RuntimeStatus.production(),
        pipeline_state="RUNNING",
        robot_state=RobotUiState.CONNECTED_DISARMED,
        source_id="stock:regression",
        snapshot_sequence=1,
        snapshot_age_s=0.01,
        command_generation=3,
        safe_command_valid=True,
        free_base_active=True,
        balance_active=True,
    )
    worker = PipelineWorker(session_factory=DemoSession)
    window = MainWindow(
        locator=ResourceLocator(root),
        user_store=UserSourceStore(tmp_path / "app-data"),
        worker=worker,
        robot_callbacks=RobotCallbacks(arm=arm, status=lambda: readiness),
        log_dir=tmp_path / "logs",
    )
    qtbot.addWidget(window)
    monkeypatch.setattr(
        "robot_human_interface.gui.main_window.ArmConfirmationDialog.exec",
        lambda _self: QDialog.DialogCode.Accepted,
    )

    try:
        window.set_robot_state("CONNECTED_DISARMED")
        window._on_robot_interlock(True)
        assert len(calls) == 1
        assert window._robot_state == "DEGRADED"
    finally:
        window.close()
        qtbot.waitUntil(lambda: not worker.isRunning(), timeout=2500)
