from __future__ import annotations

import os
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt

from robot_human_interface.gui.main_window import MainWindow
from robot_human_interface.gui.resources import ResourceLocator, SourceItem, UserSourceStore
from robot_human_interface.gui.widgets import ArmConfirmationDialog, TelemetryPanel
from robot_human_interface.gui.worker import DemoSession, PipelineWorker, build_session


def test_worker_emits_snapshot_and_stops_cleanly(qtbot) -> None:
    worker = PipelineWorker(session_factory=DemoSession)
    source = SourceItem("synthetic:test", "Synthetic", "synthetic")
    worker.start()
    worker.select_source(source)
    with qtbot.waitSignal(worker.snapshot_ready, timeout=2000) as signal:
        worker.start_pipeline()
    snapshot = signal.args[0]
    assert snapshot.sequence == 0
    assert len(snapshot.landmarks) == 33
    assert len(snapshot.angles_rad) == 20
    with qtbot.waitSignal(worker.state_changed, timeout=1000) as paused:
        worker.pause_pipeline()
    assert paused.args == ["PAUSED"]
    with qtbot.waitSignal(worker.state_changed, timeout=1000) as resumed:
        worker.resume_pipeline()
    assert resumed.args == ["RUNNING"]
    assert worker.shutdown_and_wait(2000)
    assert not worker.isRunning()


def test_lazy_adapter_runs_real_core_on_synthetic_source(qtbot) -> None:
    del qtbot  # its QApplication fixture is the ordering guarantee under test
    source = SourceItem("synthetic:gui-core", "GUI core smoke", "synthetic")
    session = build_session(source)
    assert not isinstance(session, DemoSession)
    try:
        session.start()
        snapshots = [session.step() for _ in range(3)]
        assert all(snapshot.frame is not None for snapshot in snapshots)
        assert any(
            snapshot.safe_command is not None
            and len(snapshot.safe_command.positions_rad) == 20
            for snapshot in snapshots
        )
    finally:
        session.close()


def test_source_change_closes_previous_session(qtbot) -> None:
    sessions: list[DemoSession] = []

    class TrackedSession(DemoSession):
        closed = False

        def close(self) -> None:
            super().close()
            self.closed = True

    def factory(source: SourceItem) -> TrackedSession:
        session = TrackedSession(source)
        sessions.append(session)
        return session

    worker = PipelineWorker(session_factory=factory)
    worker.start()
    worker.select_source(SourceItem("synthetic:a", "A", "synthetic"))
    worker.start_pipeline()
    qtbot.waitUntil(lambda: len(sessions) == 1, timeout=1500)
    worker.select_source(SourceItem("synthetic:b", "B", "synthetic"))
    qtbot.waitUntil(lambda: len(sessions) == 2, timeout=1500)
    assert sessions[0].closed
    assert worker.shutdown_and_wait(2000)


def test_arm_dialog_requires_both_physical_safety_acknowledgements(qtbot) -> None:
    dialog = ArmConfirmationDialog()
    qtbot.addWidget(dialog)
    assert not dialog.ok_button.isEnabled()
    dialog.zone_ack.setChecked(True)
    assert not dialog.ok_button.isEnabled()
    dialog.estop_ack.setChecked(True)
    assert dialog.ok_button.isEnabled()
    assert not dialog.send_velocities


def test_telemetry_labels_mujoco_actual_and_highlights_joint_limits(qtbot) -> None:
    panel = TelemetryPanel()
    qtbot.addWidget(panel)
    count = panel.angles_table.rowCount()
    snapshot = SimpleNamespace(
        tracking_quality=0.9,
        raw_command=SimpleNamespace(positions_rad=(0.0,) * count),
        safe_command=SimpleNamespace(positions_rad=(0.95,) * count),
        balance_active=True,
        free_base_active=True,
        telemetry={
            "joint_positions_rad": (0.0,) * count,
            "joint_lower_limits_rad": (-1.0,) * count,
            "joint_upper_limits_rad": (1.0,) * count,
            "right_foot_position_m": (0.0, -0.1, 0.0),
            "left_foot_position_m": (0.0, 0.1, 0.0),
            "center_of_mass_position_m": (0.0, 0.0, 0.8),
            "right_foot_in_contact": True,
            "left_foot_in_contact": False,
            "support_phase": "double_support",
        },
    )

    panel.update_snapshot(snapshot)

    assert "MUJOCO" in panel.angles_table.horizontalHeaderItem(3).text()
    assert panel.angles_table.verticalHeaderItem(0).text() == "РУКИ"
    assert panel.angles_table.verticalHeaderItem(6).text() == "НОГИ"
    assert panel.angles_table.verticalHeaderItem(18).text() == "ГОЛОВА"
    assert panel.angles_table.item(0, 2).background().color().alpha() > 0
    assert "энкодеров" in panel.real_feedback_note.text()
    assert panel.support_polygon._com == (0.0, 0.0)
    assert panel.support_polygon.active_contact_count == 1


def test_main_window_smoke_and_clean_close(qtbot, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    worker = PipelineWorker(
        session_factory=DemoSession,
        video_cache_dir=str(tmp_path / "cache"),
    )
    window = MainWindow(
        locator=ResourceLocator(root),
        user_store=UserSourceStore(tmp_path / "app-data"),
        worker=worker,
        log_dir=tmp_path / "logs",
    )
    qtbot.addWidget(window)
    window.show()
    qtbot.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._pipeline_state == "RUNNING", timeout=2000)
    qtbot.waitUntil(lambda: window._latest_snapshot is not None, timeout=2000)
    assert window.source_panel.reference_list.count() == 6
    assert window.telemetry.angles_table.rowCount() == 20
    window.close()
    qtbot.waitUntil(lambda: not worker.isRunning(), timeout=2500)


def test_close_remains_pending_until_delayed_worker_finishes(qtbot, tmp_path: Path) -> None:
    class DelayedShutdownWorker(PipelineWorker):
        def __init__(self) -> None:
            super().__init__(session_factory=DemoSession)
            self.release_cleanup = Event()

        def run(self) -> None:
            self.state_changed.emit("STOPPED")
            while not self._shutdown.is_set():
                self.msleep(5)
            # Model a camera/MuJoCo close that is cooperative but delayed.
            while not self.release_cleanup.is_set():
                self.msleep(5)
            self.clean_stopped.emit()

    root = Path(__file__).resolve().parents[2]
    worker = DelayedShutdownWorker()
    window = MainWindow(
        locator=ResourceLocator(root),
        user_store=UserSourceStore(tmp_path / "app-data"),
        worker=worker,
        log_dir=tmp_path / "logs",
    )
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(worker.isRunning, timeout=1000)

    try:
        assert window.close() is False
        qtbot.waitUntil(lambda: window._closing, timeout=500)
        assert window.isVisible()
        assert worker.isRunning()
        assert not window._worker_stopped_confirmed

        # A second close request must remain ignored while cleanup is blocked.
        assert window.close() is False
        assert window.isVisible()
        assert worker.isRunning()
        assert not window._worker_stopped_confirmed

        # Exercise the cooperative retry path without waiting four seconds.
        window._next_shutdown_retry_s = monotonic() - 1.0
        window._poll_worker_shutdown()
        assert window._shutdown_retry_count == 1
        assert window.isVisible()
    finally:
        worker.release_cleanup.set()

    qtbot.waitUntil(lambda: not worker.isRunning(), timeout=2000)
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=2000)
    assert window._worker_stopped_confirmed
    assert window._logging_closed
