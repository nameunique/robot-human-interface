"""PyQt6 operator interface for the humanoid teleoperation pipeline.

The package intentionally does not import Qt widgets at module import time.
Use :func:`robot_human_interface.gui.app.main` (or ``python -m
robot_human_interface``) so ``QApplication`` exists before any optional
OpenCV/MediaPipe integration is imported.
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Launch the desktop application with lazy GUI imports."""

    from .app import main as _main

    return _main(argv)
