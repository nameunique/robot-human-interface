"""Qt-independent, non-blocking camera video recording.

``CameraVideoSink`` keeps OpenCV and all ``VideoWriter`` operations on one
private thread.  Camera/pipeline threads only copy frames and attempt a
non-blocking enqueue into a bounded queue.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from threading import RLock, Thread, current_thread
from typing import Protocol, TypeAlias

import numpy as np


DEFAULT_VIDEO_QUEUE_CAPACITY = 64
DEFAULT_VIDEO_CODEC = "mp4v"


class _VideoWriter(Protocol):
    def isOpened(self) -> bool: ...

    def write(self, frame: np.ndarray) -> object: ...

    def release(self) -> object: ...


WriterFactory: TypeAlias = Callable[
    [str, int, float, tuple[int, int]], _VideoWriter
]
FourccFactory: TypeAlias = Callable[[str, str, str, str], int]


@dataclass(frozen=True, slots=True)
class CameraVideoStatus:
    """Immutable point-in-time state of a :class:`CameraVideoSink`."""

    error: str | None
    dropped: int
    incomplete: bool
    accepted: int
    written: int
    closed: bool

    @property
    def dropped_frames(self) -> int:
        """Readable alias for integrations that expose frame counters."""

        return self.dropped

    @property
    def accepted_frames(self) -> int:
        return self.accepted

    @property
    def written_frames(self) -> int:
        return self.written


# The longer name is convenient for callers without creating a second status
# representation.
CameraVideoSinkStatus = CameraVideoStatus


_STOP = object()


class CameraVideoSink:
    """Write camera frames to MP4 without blocking the producing thread.

    Args:
        path: Destination MP4 path.
        fps: Video frame rate passed to ``VideoWriter``.
        frame_size: ``(width, height)`` passed to ``VideoWriter``.
        queue_capacity: Maximum number of waiting frames.  A full queue makes
            :meth:`append` return ``False`` immediately and marks the status
            incomplete.
        codec: Four-character codec used when ``fourcc`` is not supplied.
        writer_factory: Optional ``VideoWriter``-compatible factory.  It is
            invoked on the sink thread, which makes fake writers suitable for
            tests without importing OpenCV.
        fourcc: An encoded integer or a four-character callable compatible
            with ``cv2.VideoWriter_fourcc``.  It is resolved on the sink
            thread.

    The factory, ``isOpened()``, ``write()`` and ``release()`` all run on the
    same private thread.  Encoder failures are captured in :attr:`status` and
    never propagate from :meth:`append` or :meth:`close`.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        *,
        queue_capacity: int = DEFAULT_VIDEO_QUEUE_CAPACITY,
        codec: str = DEFAULT_VIDEO_CODEC,
        writer_factory: WriterFactory | None = None,
        fourcc: int | FourccFactory | None = None,
    ) -> None:
        rate = float(fps)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("fps must be finite and positive")
        if len(frame_size) != 2:
            raise ValueError("frame_size must be (width, height)")
        width, height = frame_size
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or int(width) != width
            or int(height) != height
            or int(width) <= 0
            or int(height) <= 0
        ):
            raise ValueError("frame_size dimensions must be positive integers")
        if (
            isinstance(queue_capacity, bool)
            or int(queue_capacity) != queue_capacity
            or int(queue_capacity) <= 0
        ):
            raise ValueError("queue_capacity must be a positive integer")
        codec_text = str(codec)
        if len(codec_text) != 4:
            raise ValueError("codec must contain exactly four characters")

        self._path = Path(path)
        self._fps = rate
        self._frame_size = (int(width), int(height))
        self._codec = codec_text
        self._writer_factory = writer_factory
        self._fourcc = fourcc
        self._queue: Queue[np.ndarray | object] = Queue(
            maxsize=int(queue_capacity)
        )

        self._lock = RLock()
        self._accepting = True
        self._close_requested = False
        self._stop_enqueued = False
        self._closed = False
        self._accepted = 0
        self._written = 0
        self._dropped = 0
        self._incomplete = False
        self._errors: list[str] = []

        self._thread = Thread(
            target=self._writer_main,
            name=f"camera-video-sink-{self._path.stem or 'video'}",
            daemon=True,
        )
        self._thread.start()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def status(self) -> CameraVideoStatus:
        """Return an immutable snapshot safe to retain across updates."""

        with self._lock:
            return CameraVideoStatus(
                error="; ".join(self._errors) if self._errors else None,
                dropped=self._dropped,
                incomplete=self._incomplete,
                accepted=self._accepted,
                written=self._written,
                closed=self._closed,
            )

    def append(self, frame: np.ndarray) -> bool:
        """Copy and enqueue one frame without waiting for the writer.

        ``False`` means the queue was full or the sink no longer accepts
        frames.  Failures in the encoder thread are reported through
        :attr:`status`, never raised here.
        """

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        expected_shape = (self._frame_size[1], self._frame_size[0], 3)
        with self._lock:
            if not self._accepting:
                return False
            if frame.shape != expected_shape or frame.dtype != np.uint8:
                self._record_error(
                    "video_frame_error: expected a uint8 BGR frame with shape "
                    f"{expected_shape}, got shape {frame.shape} and dtype "
                    f"{frame.dtype}",
                    stop_accepting=False,
                )
                return False
            # Avoid an expensive frame copy in the common overload case.  A
            # second put_nowait below remains authoritative because a second
            # producer can fill the queue after this advisory check.
            if self._queue.full():
                self._dropped += 1
                self._incomplete = True
                return False

        # The writer must never observe subsequent mutation or reuse of a
        # camera buffer by the producer.  order="C" also normalizes strided
        # camera views to the layout expected by OpenCV.
        copied = frame.copy(order="C")
        with self._lock:
            if not self._accepting:
                return False
            try:
                self._queue.put_nowait(copied)
            except Full:
                self._dropped += 1
                self._incomplete = True
                return False
            self._accepted += 1
            return True

    def close(self, *, timeout_s: float = 30.0) -> CameraVideoStatus:
        """Flush queued frames, join the writer thread and return its status.

        Calls are idempotent.  A timeout is recorded as an asynchronous sink
        error instead of being raised; a later call may join the thread after
        a temporarily blocked writer resumes.
        """

        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")

        with self._lock:
            self._accepting = False
            self._close_requested = True
            enqueue_stop = not self._stop_enqueued and not self._closed
            if enqueue_stop:
                try:
                    self._queue.put_nowait(_STOP)
                except Full:
                    # A full queue already guarantees that the writer wakes.
                    # It observes _close_requested after draining the frames.
                    pass
                else:
                    self._stop_enqueued = True

        if current_thread() is self._thread:
            self._record_error(
                "video_close_error: sink thread cannot join itself",
                stop_accepting=True,
            )
            return self.status

        # Concurrent callers may join the same Python thread independently;
        # no mutex is held across join, so each caller's timeout bounds its own
        # close call.
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            self._record_error(
                "video_close_timeout: writer thread did not stop",
                stop_accepting=True,
            )
        return self.status

    def finalize(self, *, timeout_s: float = 30.0) -> CameraVideoStatus:
        """Alias for :meth:`close` used by experiment finalizers."""

        return self.close(timeout_s=timeout_s)

    def __enter__(self) -> CameraVideoSink:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _record_error(self, message: str, *, stop_accepting: bool) -> None:
        rendered = str(message)
        with self._lock:
            if rendered not in self._errors:
                self._errors.append(rendered)
            self._incomplete = True
            if stop_accepting:
                self._accepting = False

    def _resolve_writer(self) -> _VideoWriter | None:
        factory = self._writer_factory
        fourcc = self._fourcc

        if factory is None:
            try:
                import cv2
            except Exception as error:
                self._record_error(
                    self._format_error("video_codec_error", error),
                    stop_accepting=True,
                )
                return None
            factory = cv2.VideoWriter
            if fourcc is None:
                try:
                    fourcc = int(cv2.VideoWriter_fourcc(*self._codec))
                except Exception as error:
                    self._record_error(
                        self._format_error("video_codec_error", error),
                        stop_accepting=True,
                    )
                    return None

        if fourcc is None:
            # OpenCV's fourcc macro packs four one-byte character codes into
            # a little-endian integer.  Using the same operation here keeps an
            # injected fake factory independent from the cv2 package.
            try:
                values = [ord(character) for character in self._codec]
                if any(value > 0xFF for value in values):
                    raise ValueError("codec characters must fit in one byte")
                fourcc = sum(value << (8 * index) for index, value in enumerate(values))
            except Exception as error:
                self._record_error(
                    self._format_error("video_codec_error", error),
                    stop_accepting=True,
                )
                return None
        elif callable(fourcc):
            try:
                fourcc = int(fourcc(*self._codec))
            except Exception as error:
                self._record_error(
                    self._format_error("video_codec_error", error),
                    stop_accepting=True,
                )
                return None
        else:
            try:
                fourcc = int(fourcc)
            except (TypeError, ValueError, OverflowError) as error:
                self._record_error(
                    self._format_error("video_codec_error", error),
                    stop_accepting=True,
                )
                return None

        try:
            writer = factory(
                str(self._path),
                fourcc,
                self._fps,
                self._frame_size,
            )
        except Exception as error:
            self._record_error(
                self._format_error("video_open_error", error),
                stop_accepting=True,
            )
            return None
        if writer is None:
            self._record_error(
                "video_open_error: writer factory returned None",
                stop_accepting=True,
            )
            return None
        return writer

    def _writer_main(self) -> None:
        writer: _VideoWriter | None = None
        try:
            writer = self._resolve_writer()
            if writer is None:
                return
            try:
                opened = bool(writer.isOpened())
            except Exception as error:
                self._record_error(
                    self._format_error("video_open_error", error),
                    stop_accepting=True,
                )
                return
            if not opened:
                self._record_error(
                    "video_open_error: VideoWriter.isOpened() returned False",
                    stop_accepting=True,
                )
                return

            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                assert isinstance(item, np.ndarray)
                try:
                    result = writer.write(item)
                except Exception as error:
                    self._record_error(
                        self._format_error("video_write_error", error),
                        stop_accepting=True,
                    )
                    break
                if isinstance(result, (bool, np.bool_)) and not bool(result):
                    self._record_error(
                        "video_write_error: VideoWriter.write() returned False",
                        stop_accepting=True,
                    )
                    break
                with self._lock:
                    self._written += 1
                    should_stop = self._close_requested and self._queue.empty()
                if should_stop:
                    break
        except Exception as error:
            # The public API remains failure-isolated even for an unexpected
            # implementation error in an injected writer.
            self._record_error(
                self._format_error("video_writer_error", error),
                stop_accepting=True,
            )
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception as error:
                    self._record_error(
                        self._format_error("video_release_error", error),
                        stop_accepting=True,
                    )
            with self._lock:
                self._accepting = False
                if self._accepted != self._written:
                    self._incomplete = True
                self._closed = True

    @staticmethod
    def _format_error(stage: str, error: BaseException) -> str:
        detail = str(error).strip()
        suffix = f": {detail}" if detail else ""
        return f"{stage}: {type(error).__name__}{suffix}"


__all__ = [
    "CameraVideoSink",
    "CameraVideoSinkStatus",
    "CameraVideoStatus",
    "DEFAULT_VIDEO_CODEC",
    "DEFAULT_VIDEO_QUEUE_CAPACITY",
    "FourccFactory",
    "WriterFactory",
]
