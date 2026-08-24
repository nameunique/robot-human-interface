from __future__ import annotations

from pathlib import Path
import sys

from robot_human_interface.resources import ResourceLocator


def test_resource_locator_is_cwd_independent_for_checkout() -> None:
    locator = ResourceLocator()

    assert locator.asset("models", "pose_landmarker_full.task").is_file()
    assert locator.config("joints.yaml").is_file()
    assert locator.project_root != Path.cwd() or (Path.cwd() / "assets").is_dir()


def test_resource_locator_prefers_explicit_and_frozen_roots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    explicit = tmp_path / "explicit"
    assert ResourceLocator(explicit).project_root == explicit.resolve()

    frozen = tmp_path / "bundle"
    (frozen / "assets").mkdir(parents=True)
    (frozen / "config").mkdir()
    monkeypatch.setattr(sys, "_MEIPASS", str(frozen), raising=False)

    assert ResourceLocator().project_root == frozen.resolve()
