"""Cross-platform camera sources with OpenCV imported only when opened."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from robot_human_interface.playback import PlaybackDiscontinuity, PlaybackState
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


@runtime_checkable
class SeekableVideoSource(CameraSource, Protocol):
    """Optional capability implemented by file-backed frame sources."""

    @property
    def playback_state(self) -> PlaybackState: ...

    def seek(self, position_s: float) -> PlaybackState: ...

    def step_relative(self, delta_frames: int) -> PlaybackState: ...

    def set_rate(self, rate: float) -> PlaybackState: ...

    def set_loop(
        self,
        enabled: bool,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> PlaybackState: ...


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

    Frame timestamps form a deterministic monotonic delivery timeline.  Media
    position is reported separately through :attr:`playback_state`, so seek
    and A/B wrap never send a backwards timestamp to pose estimation.
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
        self._rate = 1.0
        self._frame_count: int | None = None
        self._duration_s: float | None = None
        self._next_frame_index = 0
        self._current_frame_index = -1
        self._loop_start_frame = 0
        self._loop_end_frame_exclusive: int | None = None
        self._eof = False
        self._pending_discontinuity: PlaybackDiscontinuity | None = None
        self._last_discontinuity: PlaybackDiscontinuity | None = None
        self._timeline_origin_s: float | None = None
        self._last_frame_timestamp_s: float | None = None
        self._last_delivery_clock_s: float | None = None

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
        frame_count_property = getattr(cv2, "CAP_PROP_FRAME_COUNT", None)
        reported_count = (
            float(capture.get(frame_count_property))
            if frame_count_property is not None
            else float("nan")
        )
        self._frame_count = (
            int(round(reported_count))
            if np.isfinite(reported_count) and reported_count >= 1.0
            else None
        )
        self._duration_s = (
            None
            if self._frame_count is None
            else float(self._frame_count / self._fps)
        )
        self._capture = capture
        self._sequence = 0
        self._next_frame_index = 0
        self._current_frame_index = -1
        self._loop_start_frame = 0
        self._loop_end_frame_exclusive = self._frame_count
        self._eof = False
        self._pending_discontinuity = None
        self._last_discontinuity = None
        self._timeline_origin_s = self._clock()
        self._last_frame_timestamp_s = None
        self._last_delivery_clock_s = None
        return self

    @property
    def playback_state(self) -> PlaybackState:
        frame_index = (
            self._current_frame_index
            if self._current_frame_index >= 0
            else self._next_frame_index
        )
        if self._frame_count is not None:
            frame_index = min(frame_index, max(0, self._frame_count - 1))
        position_s = frame_index / self._fps
        if self._eof and self._duration_s is not None:
            position_s = self._duration_s
        elif self._duration_s is not None:
            position_s = min(position_s, self._duration_s)
        loop_end_s = (
            None
            if self._loop_end_frame_exclusive is None
            else self._loop_end_frame_exclusive / self._fps
        )
        return PlaybackState(
            seekable=True,
            position_s=position_s,
            duration_s=self._duration_s,
            frame_index=max(0, frame_index),
            frame_count=self._frame_count,
            fps=self._fps,
            rate=self._rate,
            loop_enabled=bool(self.loop),
            loop_start_s=self._loop_start_frame / self._fps,
            loop_end_s=loop_end_s,
            eof=self._eof,
            discontinuity_reason=(
                self._last_discontinuity or self._pending_discontinuity
            ),
        )

    def _clamp_frame_index(self, frame_index: int) -> int:
        target = max(0, int(frame_index))
        if self._frame_count is not None:
            target = min(target, max(0, self._frame_count - 1))
        return target

    def _seek_frame(
        self,
        frame_index: int,
        reason: PlaybackDiscontinuity,
        *,
        reset_pacing: bool = True,
    ) -> PlaybackState:
        if self._capture is None:
            self.open()
        assert self._capture is not None
        target = self._clamp_frame_index(frame_index)
        cv2 = _cv2()
        if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, target):
            raise CameraReadError(f"video source rejected seek to frame {target}")
        self._next_frame_index = target
        self._current_frame_index = target
        self._eof = False
        self._pending_discontinuity = reason
        self._last_discontinuity = None
        if reset_pacing:
            self._last_delivery_clock_s = None
        return self.playback_state

    def seek(self, position_s: float) -> PlaybackState:
        position = float(position_s)
        if not np.isfinite(position):
            raise ValueError("position_s must be finite")
        if self._capture is None:
            self.open()
        position = max(0.0, position)
        was_eof = self._eof
        frame_index = int(np.floor(position * self._fps + 1e-9))
        target = self._clamp_frame_index(frame_index)
        reason = (
            PlaybackDiscontinuity.RESTART
            if was_eof and target == 0
            else PlaybackDiscontinuity.SEEK
        )
        return self._seek_frame(target, reason)

    def step_relative(self, delta_frames: int) -> PlaybackState:
        if (
            isinstance(delta_frames, bool)
            or int(delta_frames) != delta_frames
            or int(delta_frames) not in {-1, 1}
        ):
            raise ValueError("delta_frames must be exactly -1 or +1")
        base = (
            self._current_frame_index
            if self._current_frame_index >= 0
            else self._next_frame_index
        )
        return self._seek_frame(
            self._clamp_frame_index(base + int(delta_frames)),
            PlaybackDiscontinuity.STEP,
        )

    def set_rate(self, rate: float) -> PlaybackState:
        requested = float(rate)
        supported = (0.25, 0.5, 1.0, 1.5, 2.0)
        accepted = next(
            (value for value in supported if abs(requested - value) < 1e-12),
            None,
        )
        if accepted is None:
            raise ValueError(f"rate must be one of {supported}")
        self._rate = accepted
        # Resume pacing from the next decode instead of trying to catch up to
        # a deadline calculated for the previous rate.
        self._last_delivery_clock_s = None
        return self.playback_state

    def set_loop(
        self,
        enabled: bool,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> PlaybackState:
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if self._capture is None:
            self.open()
        start = float(start_s)
        end = self._duration_s if end_s is None else float(end_s)
        if not np.isfinite(start) or start < 0.0:
            raise ValueError("loop start must be finite and non-negative")
        if end is not None and (not np.isfinite(end) or end <= start):
            raise ValueError("loop end must be finite and after loop start")
        if enabled and end is None:
            raise ValueError("loop end is required when video duration is unknown")
        if end is not None and end - start < 1.0 / self._fps - 1e-9:
            raise ValueError("A/B loop must contain at least one frame")
        if self._duration_s is not None:
            if start >= self._duration_s:
                raise ValueError("loop start must be before video duration")
            if end is not None and end > self._duration_s + 1e-9:
                raise ValueError("loop end must not exceed video duration")

        start_frame = int(np.floor(start * self._fps + 1e-9))
        end_frame = (
            None if end is None else int(np.ceil(end * self._fps - 1e-9))
        )
        if self._frame_count is not None:
            start_frame = min(start_frame, max(0, self._frame_count - 1))
            if end_frame is not None:
                end_frame = min(end_frame, self._frame_count)
        if end_frame is not None and end_frame <= start_frame:
            raise ValueError("A/B loop must contain at least one frame")

        self.loop = enabled
        self._loop_start_frame = max(0, start_frame)
        self._loop_end_frame_exclusive = end_frame
        return self.playback_state

    def _pace_delivery(self) -> None:
        if not self.realtime:
            return
        now = self._clock()
        if self._last_delivery_clock_s is not None:
            deadline = self._last_delivery_clock_s + 1.0 / (self._fps * self._rate)
            delay = deadline - now
            if delay > 0.0:
                self._sleep(delay)
                now = self._clock()
        # A long UI pause makes ``now`` later than the old deadline. Starting
        # the next interval here avoids a burst of catch-up frames on resume.
        self._last_delivery_clock_s = now

    def _frame_timestamp(self) -> float:
        assert self._timeline_origin_s is not None
        if self._last_frame_timestamp_s is None:
            timestamp = self._timeline_origin_s
        else:
            timestamp = self._last_frame_timestamp_s + 1.0 / (
                self._fps * self._rate
            )
        self._last_frame_timestamp_s = timestamp
        return timestamp

    def _loop_boundary_reached(self) -> bool:
        return bool(
            self.loop
            and self._loop_end_frame_exclusive is not None
            and self._next_frame_index >= self._loop_end_frame_exclusive
        )

    def _rewind_loop(self) -> None:
        self._seek_frame(
            self._loop_start_frame,
            PlaybackDiscontinuity.LOOP_WRAP,
            reset_pacing=False,
        )

    def read(self) -> CameraFrame | None:
        if self._capture is None:
            self.open()
        assert self._capture is not None
        self._last_discontinuity = None
        if self._loop_boundary_reached():
            self._rewind_loop()
        ok, image = self._capture.read()
        if not ok and self.loop:
            self._rewind_loop()
            ok, image = self._capture.read()
        if not ok or image is None:
            self._eof = True
            return None
        self._pace_delivery()
        if self.mirror:
            image = np.ascontiguousarray(image[:, ::-1])
        current_frame_index = self._next_frame_index
        self._current_frame_index = current_frame_index
        self._next_frame_index = current_frame_index + 1
        self._eof = False
        self._last_discontinuity = self._pending_discontinuity
        self._pending_discontinuity = None
        timestamp = self._frame_timestamp()
        frame = CameraFrame(image, timestamp, self._sequence, self.mirror)
        self._sequence += 1
        return frame

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()
        self._last_delivery_clock_s = None

    def __enter__(self) -> "OpenCVVideoSource":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()
