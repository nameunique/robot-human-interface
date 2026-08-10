from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import robot_human_interface.camera.sources as camera_sources
from robot_human_interface.camera import (
    OpenCVVideoSource,
    SyntheticCameraConfig,
    SyntheticCameraSource,
)
from robot_human_interface.pose import SyntheticPoseEstimator


class _ManualClock:
    def __init__(self, timestamp: float = 0.0) -> None:
        self.timestamp = timestamp
        self.sleep_calls: list[float] = []

    def __call__(self) -> float:
        return self.timestamp

    def advance(self, seconds: float) -> None:
        self.timestamp += seconds

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.advance(seconds)


class _FakeCapture:
    def __init__(self, frames: Sequence[np.ndarray], *, fps: float) -> None:
        self.frames = [frame.copy() for frame in frames]
        self.fps = fps
        self.position = 0
        self.released = False
        self.set_calls: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame

    def get(self, _property: int) -> float:
        return self.fps

    def set(self, property_id: int, value: float) -> bool:
        self.set_calls.append((property_id, value))
        self.position = int(value)
        return True

    def release(self) -> None:
        self.released = True


class _FakeCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_POS_FRAMES = 1

    def __init__(self, capture: _FakeCapture) -> None:
        self.capture = capture
        self.paths: list[str] = []

    def VideoCapture(self, path: str) -> _FakeCapture:
        self.paths.append(path)
        return self.capture


def test_synthetic_camera_has_deterministic_monotonic_timestamps() -> None:
    source = SyntheticCameraSource(
        SyntheticCameraConfig(width=64, height=48, fps=20.0, max_frames=3),
        clock=lambda: 10.0,
    )
    frames = [source.read(), source.read(), source.read()]
    assert source.read() is None
    assert [frame.timestamp_s for frame in frames if frame] == [10.0, 10.05, 10.1]
    assert all(frame.image_bgr.shape == (48, 64, 3) for frame in frames if frame)
    assert not np.array_equal(frames[0].image_bgr, frames[1].image_bgr)
    source.close()
    assert source.read() is None


def test_synthetic_image_and_pose_exercise_real_skeleton_pipeline() -> None:
    source = SyntheticCameraSource(SyntheticCameraConfig(max_frames=2), clock=lambda: 4.0)
    pose = SyntheticPoseEstimator()
    first = pose.estimate(source.read())
    second_frame = source.read()
    second = pose.estimate(second_frame)
    assert first.landmarks_3d.shape == (33, 3)
    assert first.timestamp_s == 4.0
    assert second.timestamp_s > first.timestamp_s
    pose.close()


def test_video_source_realtime_pacing_preserves_media_timestamps(monkeypatch) -> None:
    frames = [np.full((2, 3, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    capture = _FakeCapture(frames, fps=25.0)
    monkeypatch.setattr(camera_sources, "_cv2", lambda: _FakeCV2(capture))
    clock = _ManualClock(100.0)
    source = OpenCVVideoSource(
        "motion.mp4",
        realtime=True,
        clock=clock,
        sleeper=clock.sleep,
    )

    first = source.read()
    clock.advance(0.01)  # decoding/processing consumed part of the frame interval
    second = source.read()
    clock.advance(5.0)  # simulate a UI pause; replay must not catch up in a burst
    third = source.read()

    assert [frame.sequence for frame in (first, second, third)] == [0, 1, 2]
    np.testing.assert_allclose(
        [frame.timestamp_s for frame in (first, second, third)],
        [100.0, 100.04, 100.08],
    )
    np.testing.assert_allclose(clock.sleep_calls, [0.03])
    assert source.read() is None


def test_video_source_loop_rewinds_while_sequence_and_timestamps_stay_monotonic(
    monkeypatch,
) -> None:
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (11, 22)]
    capture = _FakeCapture(frames, fps=20.0)
    fake_cv2 = _FakeCV2(capture)
    monkeypatch.setattr(camera_sources, "_cv2", lambda: fake_cv2)
    source = OpenCVVideoSource("loop.mp4", loop=True, clock=lambda: 7.0)

    replayed = [source.read(), source.read(), source.read()]

    assert [frame.sequence for frame in replayed] == [0, 1, 2]
    np.testing.assert_allclose(
        [frame.timestamp_s for frame in replayed],
        [7.0, 7.05, 7.1],
    )
    assert [int(frame.image_bgr[0, 0, 0]) for frame in replayed] == [11, 22, 11]
    assert capture.set_calls == [(_FakeCV2.CAP_PROP_POS_FRAMES, 0)]
    source.close()
    assert capture.released
