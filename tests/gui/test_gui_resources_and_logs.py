from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QStandardPaths

from robot_human_interface.gui.logs import LogEntry, LogFilterProxyModel, LogTableModel
from robot_human_interface.gui.resources import ResourceLocator, UserSourceStore
from robot_human_interface.gui.video_probe import VideoProbeCache, default_video_cache_dir


def test_reference_catalog_contains_all_six_mp4_files() -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = ResourceLocator(root).stock_videos()
    assert len(catalog) == 6
    assert all(item.available for item in catalog)
    assert len({item.source_id for item in catalog}) == 6


def test_user_source_store_persists_absolute_paths_without_copying(tmp_path: Path) -> None:
    video = tmp_path / "outside" / "my clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"not copied")
    store = UserSourceStore(tmp_path / "app-data")
    items = store.add(video)
    assert items[0].path == str(video.resolve())
    assert Path(items[0].path).read_bytes() == b"not copied"
    assert list((tmp_path / "app-data").iterdir()) == [store.path]
    assert store.remove(items[0].source_id) == ()


def test_video_probe_metadata_and_thumbnail_cache(qtbot, tmp_path: Path) -> None:
    del qtbot  # guarantees QApplication exists before the lazy cv2 import
    root = Path(__file__).resolve().parents[2]
    source = ResourceLocator(root).stock_videos()[0]
    cache = VideoProbeCache(tmp_path / "cache")
    metadata = cache.probe(source.source_id, source.path or "")
    assert metadata.error is None
    assert metadata.duration_s > 0.0
    assert metadata.width > 0 and metadata.height > 0
    assert metadata.thumbnail_path is not None
    thumbnail = Path(metadata.thumbnail_path)
    assert thumbnail.is_file()
    assert thumbnail.is_relative_to(cache.cache_dir)
    assert list(cache.cache_dir.glob("*.json"))
    assert cache.probe(source.source_id, source.path or "") == metadata

    qt_cache = Path(
        QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.CacheLocation
        )
    ).resolve()
    assert default_video_cache_dir().is_relative_to(qt_cache)


def test_log_proxy_combines_severity_subsystem_and_search(qtbot) -> None:
    model = LogTableModel(capacity=3)
    proxy = LogFilterProxyModel()
    proxy.setSourceModel(model)
    model.append(LogEntry.now("INFO", "PIPELINE", "SESSION_STARTED", "Сессия запущена"))
    model.append(LogEntry.now("ERROR", "ROBOT", "NETWORK_ERROR", "Сеть недоступна"))
    model.append(LogEntry.now("WARNING", "SOURCE", "FILE_MISSING", "Файл недоступен"))
    proxy.set_severity("ERROR")
    assert proxy.rowCount() == 1
    proxy.set_severity("ALL")
    proxy.set_subsystem("SOURCE")
    assert proxy.rowCount() == 1
    proxy.set_subsystem("ALL")
    proxy.set_search("network_error")
    assert proxy.rowCount() == 1
    model.append(LogEntry.now("DEBUG", "GUI", "RESIZE", "Окно изменено"))
    assert model.rowCount() == 3
    assert model.entries[0].event_code == "NETWORK_ERROR"
