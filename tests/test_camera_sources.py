from __future__ import annotations

import numpy as np

from robot_human_interface.camera import SyntheticCameraConfig, SyntheticCameraSource
from robot_human_interface.pose import SyntheticPoseEstimator


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
