"""Lazy video metadata/thumbnail probe with an App Cache-only cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from PyQt6.QtCore import QStandardPaths, Qt
from PyQt6.QtGui import QImage


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    source_id: str
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    thumbnail_path: str | None = None
    error: str | None = None

    @property
    def duration_label(self) -> str:
        seconds = max(0, round(self.duration_s))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    @property
    def resolution_label(self) -> str:
        return f"{self.width}×{self.height}" if self.width > 0 and self.height > 0 else "—"


def default_video_cache_dir() -> Path:
    """Return a location strictly beneath QStandardPaths.CacheLocation."""

    location = QStandardPaths.writableLocation(
        QStandardPaths.StandardLocation.CacheLocation
    )
    if not location:
        raise RuntimeError("Qt CacheLocation is unavailable")
    return (Path(location).resolve() / "video-thumbnails").resolve()


class VideoProbeCache:
    """Probe a video only on cache miss; OpenCV is imported inside probe()."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = (
            default_video_cache_dir()
            if cache_dir is None
            else Path(cache_dir).resolve()
        )

    @staticmethod
    def _key(path: Path) -> str:
        stat = path.stat()
        material = f"{path.resolve()}\0{stat.st_mtime_ns}\0{stat.st_size}".encode()
        return sha256(material).hexdigest()[:24]

    def probe(self, source_id: str, video_path: str | Path) -> VideoMetadata:
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            return VideoMetadata(source_id, error=f"Файл недоступен: {path}")
        try:
            key = self._key(path)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = self.cache_dir / f"{key}.json"
            thumbnail_path = self.cache_dir / f"{key}.jpg"
            cached = self._load_cached(source_id, metadata_path, thumbnail_path)
            if cached is not None:
                return cached

            # QApplication is already alive and this method is called only by
            # PipelineWorker. Importing cv2 here avoids Qt plugin conflicts.
            import cv2  # type: ignore[import-not-found]

            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                return VideoMetadata(source_id, error=f"Видео не открыто: {path}")
            try:
                width = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
                height = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                fps = max(0.0, float(capture.get(cv2.CAP_PROP_FPS)))
                frame_count = max(0, int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
                duration_s = frame_count / fps if fps > 0.0 else 0.0
                ok, frame = capture.read()
            finally:
                capture.release()

            saved_thumbnail: str | None = None
            if ok and frame is not None:
                frame_height, frame_width = frame.shape[:2]
                image = QImage(
                    frame.data,
                    int(frame_width),
                    int(frame_height),
                    int(frame.strides[0]),
                    QImage.Format.Format_BGR888,
                ).copy()
                image = image.scaled(
                    240,
                    135,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                if image.save(str(thumbnail_path), "JPG", 82):
                    saved_thumbnail = str(thumbnail_path)

            result = VideoMetadata(
                source_id=source_id,
                duration_s=duration_s,
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                thumbnail_path=saved_thumbnail,
            )
            metadata_path.write_text(
                json.dumps(asdict(result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return result
        except Exception as error:
            return VideoMetadata(source_id, error=str(error))

    @staticmethod
    def _load_cached(
        source_id: str,
        metadata_path: Path,
        thumbnail_path: Path,
    ) -> VideoMetadata | None:
        if not metadata_path.is_file():
            return None
        try:
            values = json.loads(metadata_path.read_text(encoding="utf-8"))
            values["source_id"] = source_id
            if values.get("thumbnail_path") and not thumbnail_path.is_file():
                return None
            return VideoMetadata(**values)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
