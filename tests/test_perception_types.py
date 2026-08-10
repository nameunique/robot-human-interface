from __future__ import annotations

import builtins
import importlib
import sys

import numpy as np
import pytest

from robot_human_interface.pose import make_synthetic_skeleton
from robot_human_interface.skeleton import (
    JOINT_NAMES,
    PoseLandmark as L,
    RobotJointCommand,
    SkeletonEMAFilter,
    SkeletonFilterConfig,
    SkeletonFrame,
)


def test_skeleton_arrays_are_validated_and_immutable() -> None:
    frame = make_synthetic_skeleton(1.0)
    assert frame.landmarks_2d.shape == (33, 2)
    assert frame.landmarks_3d.shape == (33, 3)
    assert frame.coverage(range(33), 0.5) == 1.0
    with pytest.raises(ValueError):
        frame.landmarks_3d[0, 0] = 123.0
    with pytest.raises(ValueError):
        SkeletonFrame(0.0, np.zeros((32, 2)), np.zeros((33, 3)), np.ones(33))


def test_robot_command_preserves_canonical_order_and_radians() -> None:
    command = RobotJointCommand.humanoid(2.0, np.arange(20) / 10.0, 0.8)
    assert command.joint_names == JOINT_NAMES
    assert command.positions_rad.shape == (20,)
    with pytest.raises(ValueError):
        RobotJointCommand.humanoid(2.0, [0.0] * 19, 0.8)


def test_landmark_filter_smooths_valid_data_but_never_accepts_low_confidence() -> None:
    first = make_synthetic_skeleton(1.0, phase_rad=0.0)
    moved = make_synthetic_skeleton(1.1, phase_rad=0.8)
    filter_ = SkeletonEMAFilter(
        SkeletonFilterConfig(time_constant_s=0.1, confidence_threshold=0.5, max_gap_s=0.2)
    )
    filtered_first = filter_.update(first)
    filtered_moved = filter_.update(moved)
    wrist = int(L.RIGHT_WRIST)
    assert not np.allclose(filtered_moved.landmarks_3d[wrist], moved.landmarks_3d[wrist])

    scores = moved.visibility.copy()
    scores[wrist] = 0.0
    untrusted = SkeletonFrame(
        1.15,
        moved.landmarks_2d,
        moved.landmarks_3d + 100.0,
        scores,
        moved.presence,
    )
    filtered_untrusted = filter_.update(untrusted)
    np.testing.assert_allclose(
        filtered_untrusted.landmarks_3d[wrist], filtered_moved.landmarks_3d[wrist]
    )
    expired = SkeletonFrame(
        1.5,
        moved.landmarks_2d,
        moved.landmarks_3d,
        scores,
        moved.presence,
    )
    filtered_expired = filter_.update(expired)
    assert np.isnan(filtered_expired.landmarks_3d[wrist]).all()


def test_importing_public_camera_and_pose_packages_does_not_import_optional_modules(monkeypatch) -> None:
    for name in list(sys.modules):
        if name.startswith("robot_human_interface.camera") or name.startswith("robot_human_interface.pose"):
            sys.modules.pop(name)
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"cv2", "mediapipe"}:
            raise AssertionError(f"optional dependency imported eagerly: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    importlib.import_module("robot_human_interface.camera")
    importlib.import_module("robot_human_interface.pose")
