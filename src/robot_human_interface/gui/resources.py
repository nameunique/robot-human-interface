"""CWD-independent runtime resources and user source persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

from PyQt6.QtCore import QStandardPaths

from robot_human_interface.resources import ResourceLocator as CoreResourceLocator


@dataclass(frozen=True, slots=True)
class SourceItem:
    """One selectable video or camera source."""

    source_id: str
    title: str
    kind: str
    path: str | None = None
    camera_index: int = 0
    camera_backend: str = "auto"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    loop: bool = False
    mirror: bool = False

    @property
    def available(self) -> bool:
        return self.path is None or Path(self.path).is_file()

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


class ResourceLocator(CoreResourceLocator):
    """Core unified locator extended with GUI stock-video descriptors."""

    def stock_videos(self) -> tuple[SourceItem, ...]:
        video_root = self.locate("assets", "videos")
        paths = sorted(video_root.rglob("*.mp4")) if video_root.is_dir() else []
        display_names = {
            "jumping_jacks_demo": "Jumping Jacks",
            "slow_balance_demo": "Медленный баланс",
            "dvids_stationary_squat": "Приседание",
            "dvids_arm_circles": "Круги руками",
            "dvids_frontal_leg_swing": "Махи ногой",
            "dvids_trunk_circles": "Круги корпусом",
        }
        return tuple(
            SourceItem(
                source_id=f"reference:{path.relative_to(video_root).as_posix()}",
                title=display_names.get(path.stem, path.stem.replace("_", " ").title()),
                kind="reference",
                path=str(path.resolve()),
                loop=False,
            )
            for path in paths
        )


class UserSourceStore:
    """Persist absolute user-video paths without copying their contents."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir is None:
            location = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppLocalDataLocation
            )
            data_dir = location or str(Path.home() / ".robot-human-interface")
        self.data_dir = Path(data_dir).resolve()
        self.path = self.data_dir / "user_sources.json"

    def load(self) -> tuple[SourceItem, ...]:
        if not self.path.is_file():
            return ()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return ()
        items: list[SourceItem] = []
        for value in raw if isinstance(raw, list) else []:
            try:
                path = Path(str(value)).expanduser().resolve()
            except (OSError, RuntimeError):
                continue
            items.append(
                SourceItem(
                    source_id=f"user:{path.as_posix()}",
                    title=path.stem,
                    kind="user",
                    path=str(path),
                    loop=False,
                )
            )
        return tuple(items)

    def save_paths(self, paths: Iterable[str | Path]) -> None:
        normalized = sorted({str(Path(path).expanduser().resolve()) for path in paths})
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, path: str | Path) -> tuple[SourceItem, ...]:
        entries = list(self.load())
        resolved = Path(path).expanduser().resolve()
        if all(item.path != str(resolved) for item in entries):
            entries.append(
                SourceItem(
                    source_id=f"user:{resolved.as_posix()}",
                    title=resolved.stem,
                    kind="user",
                    path=str(resolved),
                    loop=False,
                )
            )
        self.save_paths(item.path for item in entries if item.path)
        return self.load()

    def remove(self, source_id: str) -> tuple[SourceItem, ...]:
        entries = [item for item in self.load() if item.source_id != source_id]
        self.save_paths(item.path for item in entries if item.path)
        return self.load()
