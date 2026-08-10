"""Cross-platform camera sources with OpenCV imported only when opened."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from robot_human_interface.skeleton import CameraFrame


class CameraError(RuntimeError):
    """Base error for capture initialization and reads."""


class CameraUnavailableError(CameraError):
    """Raised when the requested camera/video cannot be opened."""


class CameraReadError(CameraError):
    """Raised when a live capture unexpectedly stops producing frames."""


@runtime_checkable
class CameraSource(Protocol):
    def read(self) -> CameraFrame | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class OpenCVCameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    # Processing is unmirrored by default so anatomical LEFT/RIGHT remains
    # unambiguous. Mirror only the display in the UI when possible.
    mirror: bool = False
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("camera index must be non-negative")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0.0:
            raise ValueError("width, height, and fps must be positive")


def _cv2():
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise CameraUnavailableError(
            "OpenCV is required for camera/video input; install opencv-contrib-python"
        ) from error
    return cv2


def _backend_id(cv2: object, name: str) -> int | None:
    normalized = name.strip().lower()
    if normalized in {"", "auto", "any"}:
        return None
    choices = {
        "dshow": "CAP_DSHOW",
        "msmf": "CAP_MSMF",
        "v4l2": "CAP_V4L2",
        "avfoundation": "CAP_AVFOUNDATION",
        "gstreamer": "CAP_GSTREAMER",
    }
    attribute = choices.get(normalized)
    if attribute is None:
        raise ValueError(f"unsupported OpenCV backend: {name!r}")
    if not hasattr(cv2, attribute):
        raise CameraUnavailableError(f"this OpenCV build has no {attribute} backend")
    return int(getattr(cv2, attribute))


class OpenCVCameraSource:
    """Live ``cv2.VideoCapture`` source with monotonic capture timestamps."""

    def __init__(
        self,
        config: OpenCVCameraConfig | None = None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.config = config or OpenCVCameraConfig()
        self._clock = clock
        self._capture = None
        self._sequence = 0

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    def open(self) -> "OpenCVCameraSource":
        if self._capture is not None:
            return self
        cv2 = _cv2()
        backend = _backend_id(cv2, self.config.backend)
        capture = (
            cv2.VideoCapture(self.config.index)
            if backend is None
            else cv2.VideoCapture(self.config.index, backend)
        )
        if not capture.isOpened():
            capture.release()
            raise CameraUnavailableError(
                f"cannot open camera index {self.config.index} "
                f"with backend {self.config.backend!r}"
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._capture = capture
        self._sequence = 0
        return self

    def read(self) -> CameraFrame:
        if self._capture is None:
            self.open()
        assert self._capture is not None
        ok, image = self._capture.read()
        timestamp = self._clock()
        if not ok or image is None:
            raise CameraReadError(f"camera {self.config.index} did not return a frame")
        if self.config.mirror:
            image = np.ascontiguousarray(image[:, ::-1])
        frame = CameraFrame(
            image_bgr=image,
            timestamp_s=timestamp,
            sequence=self._sequence,
            mirrored=self.config.mirror,
        )
        self._sequence += 1
        return frame

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __enter__(self) -> "OpenCVCameraSource":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class SyntheticCameraConfig:
    width: int = 640
    height: int = 480
    fps: float = 30.0
    max_frames: int | None = None
    realtime: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0.0:
            raise ValueError("width, height, and fps must be positive")
        if self.max_frames is not None and self.max_frames < 0:
            raise ValueError("max_frames must be non-negative or None")


class SyntheticCameraSource:
    """Deterministic moving test pattern requiring no camera or OpenCV."""

    def __init__(
        self,
        config: SyntheticCameraConfig | None = None,
        *,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self.config = config or SyntheticCameraConfig()
        self._clock = clock
        self._sleep = sleeper
        self._sequence = 0
        self._start_timestamp: float | None = None
        self._next_deadline: float | None = None
        self.closed = False

    def read(self) -> CameraFrame | None:
        if self.closed:
            return None
        if self.config.max_frames is not None and self._sequence >= self.config.max_frames:
            return None
        if self._start_timestamp is None:
            self._start_timestamp = self._clock()
            self._next_deadline = self._start_timestamp
        if self.config.realtime:
            assert self._next_deadline is not None
            delay = self._next_deadline - self._clock()
            if delay > 0.0:
                self._sleep(delay)
            timestamp = self._clock()
            self._next_deadline += 1.0 / self.config.fps
        else:
            timestamp = self._start_timestamp + self._sequence / self.config.fps

        height, width = self.config.height, self.config.width
        image = np.zeros((height, width, 3), dtype=np.uint8)
        x_gradient = np.linspace(0, 100, width, dtype=np.uint8)
        image[:, :, 0] = x_gradient
        image[:, :, 1] = np.uint8((self._sequence * 7) % 180)
        block = max(4, min(width, height) // 12)
        x0 = (self._sequence * max(1, block // 3)) % max(1, width - block)
        y0 = height // 2 - block // 2
        image[y0 : y0 + block, x0 : x0 + block] = (30, 220, 255)
        frame = CameraFrame(image, timestamp, self._sequence, mirrored=False)
        self._sequence += 1
        return frame

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "SyntheticCameraSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class OpenCVVideoSource:
    """Replay a video file through the same frame API as the live camera.

    Media timestamps are derived from the source FPS rather than wall-clock
    decode time.  Optional real-time pacing only delays frame delivery; it
    therefore cannot introduce timestamp jitter into pose estimation.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mirror: bool = False,
        loop: bool = False,
        realtime: bool = False,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        self.path = Path(path)
        self.mirror = mirror
        self.loop = loop
        self.realtime = realtime
        self._clock = clock
        self._sleep = sleeper
        self._capture = None
        self._sequence = 0
        self._fps = 30.0
        self._start_timestamp: float | None = None
        self._last_delivery_timestamp: float | None = None

    def open(self) -> "OpenCVVideoSource":
        if self._capture is not None:
            return self
        cv2 = _cv2()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise CameraUnavailableError(f"cannot open replay video: {self.path}")
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        self._fps = reported_fps if np.isfinite(reported_fps) and reported_fps > 0.0 else 30.0
        self._capture = capture
        self._sequence = 0
        self._start_timestamp = self._clock()
        self._last_delivery_timestamp = None
        return self

    def _pace_delivery(self) -> None:
        if not self.realtime:
            return
        now = self._clock()
        if self._last_delivery_timestamp is not None:
            deadline = self._last_delivery_timestamp + 1.0 / self._fps
            delay = deadline - now
            if delay > 0.0:
                self._sleep(delay)
                now = self._clock()
        # A long UI pause makes ``now`` later than the old deadline. Starting
        # the next interval here avoids a burst of catch-up frames on resume.
        self._last_delivery_timestamp = now

    def read(self) -> CameraFrame | None:
        if self._capture is None:
            self.open()
        assert self._capture is not None
        ok, image = self._capture.read()
        if not ok and self.loop:
            cv2 = _cv2()
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._capture.read()
        if not ok or image is None:
            return None
        self._pace_delivery()
        if self.mirror:
            image = np.ascontiguousarray(image[:, ::-1])
        assert self._start_timestamp is not None
        timestamp = self._start_timestamp + self._sequence / self._fps
        frame = CameraFrame(image, timestamp, self._sequence, self.mirror)
        self._sequence += 1
        return frame

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
        self._last_delivery_timestamp = None

    def __enter__(self) -> "OpenCVVideoSource":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()
