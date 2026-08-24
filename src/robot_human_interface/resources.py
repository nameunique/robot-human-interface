"""CWD-independent runtime resource discovery with no GUI dependency."""

from __future__ import annotations

from pathlib import Path
import sys


class ResourceLocator:
    """Locate repository or frozen-bundle assets/config deterministically.

    Resolution order is explicit root, PyInstaller-style ``sys._MEIPASS``,
    then an ancestor of this module containing both ``assets`` and ``config``.
    No branch consults the process current working directory.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._explicit_root = (
            None if root is None else Path(root).expanduser().resolve()
        )

    @staticmethod
    def _is_resource_root(path: Path) -> bool:
        return (path / "assets").is_dir() and (path / "config").is_dir()

    @property
    def project_root(self) -> Path:
        if self._explicit_root is not None:
            return self._explicit_root
        frozen_root = getattr(sys, "_MEIPASS", None)
        if frozen_root:
            candidate = Path(frozen_root).resolve()
            if self._is_resource_root(candidate):
                return candidate
        anchor = Path(__file__).resolve()
        for parent in anchor.parents:
            if self._is_resource_root(parent):
                return parent
        # Deterministic source-layout fallback.  Missing resources then fail at
        # the concrete requested path rather than depending on os.getcwd().
        return anchor.parents[2]

    def locate(self, *parts: str | Path) -> Path:
        return self.project_root.joinpath(*parts).resolve()

    def asset(self, *parts: str | Path) -> Path:
        return self.locate("assets", *parts)

    def config(self, name: str | Path) -> Path:
        return self.locate("config", name)

    def model(self, *parts: str | Path) -> Path:
        return self.locate("models", *parts)


__all__ = ["ResourceLocator"]
