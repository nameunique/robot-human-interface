from __future__ import annotations

import os
from pathlib import Path
from threading import Event, get_ident
from time import monotonic
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import numpy as np

pytest.importorskip("PyQt6")

from robot_human_interface.experiments import CameraVideoSink, ExperimentSpec
from robot_human_interface.gui.main_window import MainWindow
from robot_human_interface.gui.research_widgets import SystemBannerState
from robot_human_interface.gui.resources import ResourceLocator, SourceItem, UserSourceStore
from robot_human_interface.gui.worker import PipelineWorker
from robot_human_interface.playback import PlaybackState


class _ProductionSession:
    def __init__(self, source: SourceItem) -> None:
        self.source = source
        self.running = False
        self.sequence = 0
        self.calls: list[tuple[object, ...]] = []
        self._next_frame = 0.0
        self.config = {"source": {"source_id": source.source_id}}

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

    def step(self):
        now = monotonic()
        if not self.running or now < self._next_frame:
            return None
        self._next_frame = now + 0.02
        snapshot = SimpleNamespace(
            sequence=self.sequence,
            timestamp_s=now,
            status="running",
            source=self.source,
            frame=None,
            skeleton=None,
            raw_command=None,
            safe_command=None,
            tracking_quality=0.8,
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

    def request_seek(self, position_s: float) -> None:
        self.calls.append(("seek", position_s))

    def request_step_frame(self, delta: int) -> None:
        self.calls.append(("step", delta))

    def request_set_playback_rate(self, rate: float) -> None:
        self.calls.append(("rate", rate))

    def request_set_loop(self, enabled: bool, start_s: float, end_s: float) -> None:
        self.calls.append(("loop", enabled, start_s, end_s))


class _EndedSeekableSession(_ProductionSession):
    def __init__(self, source: SourceItem) -> None:
        super().__init__(source)
        self.closed = False

    def close(self) -> None:
        self.closed = True
        super().close()

    def step(self):
        if not self.running:
            return None
        return SimpleNamespace(
            sequence=0,
            timestamp_s=monotonic(),
            status="ended",
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


def test_worker_forwards_playback_and_records_without_blocking_pipeline(qtbot, tmp_path: Path) -> None:
    sessions: list[_ProductionSession] = []

    def factory(source: SourceItem) -> _ProductionSession:
        session = _ProductionSession(source)
        sessions.append(session)
        return session

    worker = PipelineWorker(
        session_factory=factory,
        experiment_root=tmp_path / "experiments",
    )
    source = SourceItem("stock:test", "Test", "stock", path=str(tmp_path / "input.mp4"))
    worker.start()
    worker.select_source(source)
    with qtbot.waitSignal(worker.snapshot_ready, timeout=2000):
        worker.start_pipeline()
    assert sessions

    worker.seek(1.25)
    qtbot.waitUntil(lambda: ("seek", 1.25) in sessions[0].calls, timeout=1000)
    with qtbot.waitSignal(worker.snapshot_ready, timeout=2000):
        worker.resume_pipeline()
    qtbot.waitUntil(lambda: worker.runtime_status.physical_output_allowed, timeout=1000)

    states: list[str] = []
    worker.recorder_state_changed.connect(lambda state, _payload: states.append(state))
    worker.start_recording(
        ExperimentSpec("P001", "squat", 1, "baseline", 7, consent=True)
    )
    qtbot.waitUntil(lambda: "RECORDING" in states, timeout=3000)
    qtbot.wait(100)
    with qtbot.waitSignal(worker.experiment_completed, timeout=5000) as completed:
        worker.stop_recording("manual")
    summary = completed.args[0]
    assert summary.sample_count > 0
    assert summary.path.is_dir()
    assert worker.isRunning()
    assert worker.shutdown_and_wait(3000)


def test_compact_layout_and_stale_watchdog_remain_operator_safe(qtbot, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    worker = PipelineWorker(session_factory=_ProductionSession)
    window = MainWindow(
        locator=ResourceLocator(root),
        user_store=UserSourceStore(tmp_path / "app-data"),
        worker=worker,
        log_dir=tmp_path / "logs",
    )
    qtbot.addWidget(window)
    window.resize(1024, 640)
    window.show()
    qtbot.waitUntil(lambda: window._compact, timeout=1000)
    assert window.preview.isVisible()
    assert window.telemetry.isVisible()
    assert window.telemetry.connect_button.isVisible()
    assert not window.source_panel.isVisible()
    assert window.log_panel.collapsed
    window.telemetry.tabs.setCurrentIndex(0)
    qtbot.wait(20)
    assert not window.telemetry.angles_table.horizontalScrollBar().isVisible()

    window._pipeline_state = "RUNNING"
    window._snapshot_received_at = monotonic() - 0.6
    window._watchdog_tick()
    assert window.system_banner.banner_state is SystemBannerState.STALE
    assert "УСТАРЕЛИ" in window.preview._overlay_text

    window.close()
    qtbot.waitUntil(lambda: not worker.isRunning(), timeout=3000)


def test_worker_keeps_seekable_source_open_at_eof(qtbot) -> None:
    sessions: list[_EndedSeekableSession] = []

    def factory(source: SourceItem) -> _EndedSeekableSession:
        session = _EndedSeekableSession(source)
        sessions.append(session)
        return session

    worker = PipelineWorker(session_factory=factory)
    worker.start()
    worker.select_source(SourceItem("stock:eof", "EOF", "stock", path="eof.mp4"))
    with qtbot.waitSignal(worker.snapshot_ready, timeout=2000):
        worker.start_pipeline()
    qtbot.waitUntil(lambda: worker._pipeline_state == "ENDED", timeout=1000)
    assert not sessions[0].closed

    worker.seek(0.5)
    qtbot.waitUntil(lambda: ("seek", 0.5) in sessions[0].calls, timeout=1000)
    assert not sessions[0].closed
    assert worker.shutdown_and_wait(2000)


def test_camera_video_writer_never_blocks_pipeline_worker(qtbot, tmp_path: Path) -> None:
    write_entered = Event()
    release_write = Event()
    writer_thread_ids: list[int] = []
    session_thread_ids: list[int] = []

    class BlockingWriter:
        def isOpened(self) -> bool:  # noqa: N802
            writer_thread_ids.append(get_ident())
            return True

        def write(self, _frame) -> None:
            writer_thread_ids.append(get_ident())
            write_entered.set()
            assert release_write.wait(5.0)

        def release(self) -> None:
            writer_thread_ids.append(get_ident())

    class CameraSession(_ProductionSession):
        def step(self):
            snapshot = super().step()
            if snapshot is not None:
                session_thread_ids.append(get_ident())
                snapshot.frame = SimpleNamespace(
                    image_bgr=np.zeros((24, 32, 3), dtype=np.uint8)
                )
            return snapshot

    def sink_factory(path: Path, fps: float, frame_size: tuple[int, int]):
        def writer_factory(filename: str, *_args):
            Path(filename).touch()
            writer_thread_ids.append(get_ident())
            return BlockingWriter()

        return CameraVideoSink(
            path,
            fps,
            frame_size,
            queue_capacity=4,
            writer_factory=writer_factory,
            fourcc=0,
        )

    worker = PipelineWorker(
        session_factory=CameraSession,
        experiment_root=tmp_path / "experiments",
        camera_video_sink_factory=sink_factory,
    )
    sequences: list[int] = []
    states: list[str] = []
    completions: list[object] = []
    worker.snapshot_ready.connect(lambda snapshot: sequences.append(snapshot.sequence))
    worker.recorder_state_changed.connect(lambda state, _payload: states.append(state))
    worker.experiment_completed.connect(completions.append)
    worker.start()
    worker.select_source(SourceItem("camera:test", "Camera", "camera", fps=30.0))
    worker.start_pipeline()

    try:
        qtbot.waitUntil(lambda: len(sequences) >= 2, timeout=1500)
        worker.start_recording(
            ExperimentSpec(
                "P-CAM",
                "squat",
                1,
                "baseline",
                11,
                consent=True,
                record_video=True,
            )
        )
        qtbot.waitUntil(lambda: "RECORDING" in states, timeout=3000)
        qtbot.waitUntil(write_entered.is_set, timeout=2000)
        while_blocked = sequences[-1]
        qtbot.waitUntil(lambda: sequences[-1] >= while_blocked + 3, timeout=1500)

        worker.stop_recording("manual")
        qtbot.waitUntil(lambda: "FINALIZING" in states, timeout=1000)
        while_finalizing = sequences[-1]
        qtbot.waitUntil(
            lambda: sequences[-1] >= while_finalizing + 3,
            timeout=1500,
        )
        assert not completions
        release_write.set()
        qtbot.waitUntil(lambda: len(completions) == 1, timeout=5000)
        assert Path(getattr(completions[0], "path")).is_dir()
        assert writer_thread_ids
        assert session_thread_ids
        assert len(set(writer_thread_ids)) == 1
        assert writer_thread_ids[0] not in set(session_thread_ids)
    finally:
        release_write.set()
        assert worker.shutdown_and_wait(4000)


def test_experiment_package_waits_for_camera_sink_to_really_close(
    qtbot, tmp_path: Path
) -> None:
    frame_accepted = Event()
    close_attempted = Event()
    allow_close = Event()
    completions: list[object] = []
    sequences: list[int] = []
    states: list[str] = []

    class CameraSession(_ProductionSession):
        def step(self):
            snapshot = super().step()
            if snapshot is not None:
                snapshot.frame = SimpleNamespace(
                    image_bgr=np.zeros((24, 32, 3), dtype=np.uint8)
                )
            return snapshot

    class DelayedCloseSink:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.path.touch()
            self.status = SimpleNamespace(
                closed=False,
                incomplete=False,
                dropped=0,
                error=None,
            )

        def append(self, _frame) -> bool:
            frame_accepted.set()
            return True

        def close(self, *, timeout_s: float = 30.0):
            assert timeout_s > 0.0
            close_attempted.set()
            closed = allow_close.is_set()
            self.status = SimpleNamespace(
                closed=closed,
                incomplete=not closed,
                dropped=0,
                error=None if closed else "video_close_timeout",
            )
            return self.status

    def sink_factory(path: Path, _fps: float, _frame_size: tuple[int, int]):
        return DelayedCloseSink(path)

    worker = PipelineWorker(
        session_factory=CameraSession,
        experiment_root=tmp_path / "experiments",
        camera_video_sink_factory=sink_factory,
    )
    worker.experiment_completed.connect(completions.append)
    worker.snapshot_ready.connect(lambda snapshot: sequences.append(snapshot.sequence))
    worker.recorder_state_changed.connect(lambda state, _payload: states.append(state))
    worker.start()
    worker.select_source(SourceItem("camera:test", "Camera", "camera", fps=30.0))
    worker.start_pipeline()

    try:
        qtbot.waitUntil(lambda: len(sequences) >= 2, timeout=2000)
        worker.start_recording(
            ExperimentSpec(
                "P-CLOSE",
                "squat",
                1,
                "baseline",
                12,
                consent=True,
                record_video=True,
            )
        )
        qtbot.waitUntil(lambda: "RECORDING" in states, timeout=3000)
        qtbot.waitUntil(frame_accepted.is_set, timeout=3000)
        worker.stop_recording("manual")
        qtbot.waitUntil(close_attempted.is_set, timeout=1500)

        assert not completions
        assert worker._recording_task is not None
        assert worker._recording_task.is_alive()
        assert str(getattr(worker._recorder.state, "value", "")).lower() == "recording"
        assert worker._recorder.summary.path.name.endswith(".partial")

        allow_close.set()
        qtbot.waitUntil(lambda: len(completions) == 1, timeout=3000)
        summary = completions[0]
        assert summary.incomplete
        assert summary.path.name.endswith(".partial")
    finally:
        allow_close.set()
        assert worker.shutdown_and_wait(4000)
