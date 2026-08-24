"""Desktop entry point that establishes Qt before optional CV imports."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

from .theme import APP_STYLESHEET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyQt-пульт Humanoid Interface")
    parser.add_argument(
        "--maximized",
        action="store_true",
        help="открыть главное окно развёрнутым",
    )
    parser.add_argument(
        "--robot-url",
        default="ws://leonardo.local:1233",
        help="WebSocket-адрес legacy Unity-протокола (соединение не включает отправку)",
    )
    return parser


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Create/configure QApplication before importing the application window."""

    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    qt_argv = list(argv) if argv is not None else list(sys.argv)
    app = QApplication(qt_argv)
    QCoreApplication.setOrganizationName("Humanoid Interface")
    QCoreApplication.setOrganizationDomain("humanoid-interface.local")
    QCoreApplication.setApplicationName("Humanoid Interface")
    QCoreApplication.setApplicationVersion("0.1.0")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    return app


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else list(sys.argv[1:])
    options = build_parser().parse_args(arguments)
    app = create_application([sys.argv[0], *arguments])

    # This import intentionally happens after QApplication exists. The window
    # creates the QThread worker; only that worker imports OpenCV/MediaPipe.
    from .main_window import MainWindow
    from .worker import PipelineWorker

    window = MainWindow(worker=PipelineWorker(robot_endpoint=options.robot_url))
    if options.maximized:
        window.showMaximized()
    else:
        window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
