from __future__ import annotations

import os
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QThread, QTimer
from PyQt6.QtWidgets import QApplication

from robot_human_interface.gui.app import _exec_with_worker_shutdown


def test_qapplication_quit_cooperatively_stops_and_waits_for_worker(qtbot) -> None:
    class CooperativeWorker(QThread):
        def __init__(self) -> None:
            super().__init__()
            self.shutdown_requested = Event()
            self.cleanup_finished = Event()

        def request_shutdown(self) -> None:
            self.shutdown_requested.set()

        def run(self) -> None:
            assert self.shutdown_requested.wait(timeout=5.0)
            self.msleep(25)
            self.cleanup_finished.set()

    app = QApplication.instance()
    assert isinstance(app, QApplication)
    worker = CooperativeWorker()
    worker.start()
    qtbot.waitUntil(worker.isRunning, timeout=1000)
    QTimer.singleShot(0, app.quit)

    assert _exec_with_worker_shutdown(app, worker) == 0
    assert worker.shutdown_requested.is_set()
    assert worker.cleanup_finished.is_set()
    assert not worker.isRunning()
