from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Event, Thread, get_ident
from time import perf_counter, sleep

import numpy as np
import pytest

from robot_human_interface.experiments import CameraVideoSink


class _BlockingWriter:
    def __init__(self, gate: Event, write_started: Event) -> None:
        self._gate = gate
        self._write_started = write_started
        self.frames: list[np.ndarray] = []
        self.call_threads: list[tuple[str, int]] = []

    def isOpened(self) -> bool:
        self.call_threads.append(("isOpened", get_ident()))
        return True

    def write(self, frame: np.ndarray) -> bool:
        self.call_threads.append(("write", get_ident()))
        self.frames.append(frame)
        self._write_started.set()
        assert self._gate.wait(timeout=5.0)
        return True

    def release(self) -> None:
        self.call_threads.append(("release", get_ident()))


class _CopyBomb(np.ndarray):
    def copy(self, *args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("a frame rejected by the full queue must not be copied")


def test_append_is_nonblocking_copies_frames_and_reports_overflow(
    tmp_path: Path,
) -> None:
    caller_thread = get_ident()
    gate = Event()
    write_started = Event()
    factory_threads: list[int] = []
    writer = _BlockingWriter(gate, write_started)

    def writer_factory(
        path: str,
        fourcc: int,
        fps: float,
        frame_size: tuple[int, int],
    ) -> _BlockingWriter:
        factory_threads.append(get_ident())
        assert path == str(tmp_path / "camera.mp4")
        assert fourcc == 1234
        assert fps == 20.0
        assert frame_size == (4, 3)
        return writer

    sink = CameraVideoSink(
        tmp_path / "camera.mp4",
        20.0,
        (4, 3),
        queue_capacity=1,
        writer_factory=writer_factory,
        fourcc=1234,
    )
    backing = np.full((3, 8, 3), 7, dtype=np.uint8)
    first = backing[:, ::2, :]
    assert not first.flags.c_contiguous
    assert sink.append(first)
    backing.fill(99)
    assert write_started.wait(timeout=5.0)

    assert sink.append(np.full((3, 4, 3), 8, dtype=np.uint8))
    started = perf_counter()
    rejected = np.full((3, 4, 3), 9, dtype=np.uint8).view(_CopyBomb)
    assert sink.append(rejected) is False
    assert perf_counter() - started < 0.25

    overflow = sink.status
    assert overflow.dropped == 1
    assert overflow.incomplete
    assert overflow.error is None
    with pytest.raises(FrozenInstanceError):
        overflow.dropped = 0  # type: ignore[misc]

    gate.set()
    status = sink.close()
    assert status.closed
    assert status.accepted == 2
    assert status.written == 2
    assert np.all(writer.frames[0] == 7)
    assert writer.frames[0].flags.c_contiguous
    assert factory_threads and factory_threads[0] != caller_thread
    assert {thread_id for _, thread_id in writer.call_threads} == {
        factory_threads[0]
    }
    assert [name for name, _ in writer.call_threads].count("release") == 1
    assert sink.close() == status


class _FailingWriter:
    def __init__(self) -> None:
        self.write_thread: int | None = None
        self.release_thread: int | None = None

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.write_thread = get_ident()
        raise OSError("encoder stopped")

    def release(self) -> None:
        self.release_thread = get_ident()


def test_writer_error_is_status_not_a_caller_exception(tmp_path: Path) -> None:
    caller_thread = get_ident()
    writer = _FailingWriter()
    sink = CameraVideoSink(
        tmp_path / "camera.mp4",
        30.0,
        (2, 2),
        writer_factory=lambda *_: writer,
        fourcc=0,
    )

    assert sink.append(np.zeros((2, 2, 3), dtype=np.uint8))
    status = sink.finalize()

    assert status.closed
    assert status.incomplete
    assert status.written == 0
    assert "video_write_error" in (status.error or "")
    assert "encoder stopped" in (status.error or "")
    assert writer.write_thread is not None
    assert writer.write_thread != caller_thread
    assert writer.release_thread == writer.write_thread
    assert sink.append(np.zeros((2, 2, 3), dtype=np.uint8)) is False


class _ClosedWriter:
    def __init__(self) -> None:
        self.factory_thread: int | None = None
        self.release_thread: int | None = None

    def isOpened(self) -> bool:
        return False

    def write(self, frame: np.ndarray) -> None:
        raise AssertionError("write must not run for a closed writer")

    def release(self) -> None:
        self.release_thread = get_ident()


def test_open_failure_is_reported_and_released_on_sink_thread(
    tmp_path: Path,
) -> None:
    writer = _ClosedWriter()

    def writer_factory(*_: object) -> _ClosedWriter:
        writer.factory_thread = get_ident()
        return writer

    sink = CameraVideoSink(
        tmp_path / "camera.mp4",
        25.0,
        (2, 2),
        writer_factory=writer_factory,
        fourcc=lambda *_: 42,
    )
    status = sink.close()

    assert status.closed
    assert status.incomplete
    assert "video_open_error" in (status.error or "")
    assert writer.factory_thread is not None
    assert writer.release_thread == writer.factory_thread


def test_codec_failure_is_reported_without_opening_writer(tmp_path: Path) -> None:
    factory_called = Event()

    def fail_fourcc(*_: str) -> int:
        raise ValueError("unsupported codec")

    def writer_factory(*_: object) -> _ClosedWriter:
        factory_called.set()
        return _ClosedWriter()

    sink = CameraVideoSink(
        tmp_path / "camera.mp4",
        25.0,
        (2, 2),
        writer_factory=writer_factory,
        fourcc=fail_fourcc,
    )
    status = sink.close()

    assert status.closed
    assert status.incomplete
    assert "video_codec_error" in (status.error or "")
    assert "unsupported codec" in (status.error or "")
    assert not factory_called.is_set()


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
    ],
    ids=["wrong-dtype", "wrong-size", "missing-channels"],
)
def test_invalid_frame_is_fail_isolated_and_marks_status_incomplete(
    tmp_path: Path,
    frame: np.ndarray,
) -> None:
    writer = _BlockingWriter(Event(), Event())
    sink = CameraVideoSink(
        tmp_path / "camera.mp4",
        25.0,
        (2, 2),
        writer_factory=lambda *_: writer,
        fourcc=0,
    )

    assert sink.append(frame) is False
    status = sink.close()

    assert status.closed
    assert status.incomplete
    assert status.accepted == 0
    assert status.dropped == 0
    assert "video_frame_error" in (status.error or "")
    assert writer.frames == []


def test_concurrent_close_obeys_each_callers_timeout(tmp_path: Path) -> None:
    gate = Event()
    write_started = Event()
    writer = _BlockingWriter(gate, write_started)
    sink = CameraVideoSink(
        tmp_path / "camera.mp4",
        25.0,
        (2, 2),
        writer_factory=lambda *_: writer,
        fourcc=0,
    )
    assert sink.append(np.zeros((2, 2, 3), dtype=np.uint8))
    assert write_started.wait(timeout=5.0)

    first_result: list[object] = []
    first_close = Thread(
        target=lambda: first_result.append(sink.close(timeout_s=1.0)),
        daemon=True,
    )
    first_close.start()
    deadline = perf_counter() + 1.0
    while not sink._close_requested:
        assert perf_counter() < deadline
        sleep(0.001)

    started = perf_counter()
    short_status = sink.close(timeout_s=0.05)
    elapsed = perf_counter() - started
    assert elapsed < 0.25
    assert not short_status.closed
    assert "video_close_timeout" in (short_status.error or "")

    gate.set()
    first_close.join(timeout=5.0)
    assert not first_close.is_alive()
    final_status = sink.close()
    assert final_status.closed
    assert first_result
