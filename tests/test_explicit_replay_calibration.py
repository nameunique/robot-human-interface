from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from robot_human_interface.app.teleop import (
    BUNDLED_VIDEO_PATHS,
    build_parser,
    run_teleop,
)


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (
            ["--source", "mp4", "--calibration-video", "clip.mp4"],
            "must be supplied together",
        ),
        (
            ["--source", "mp4", "--calibration-frame", "2"],
            "must be supplied together",
        ),
        (
            [
                "--source",
                "synthetic",
                "--calibration-video",
                "clip.mp4",
                "--calibration-frame",
                "2",
            ],
            "only with --source mp4/replay",
        ),
        (
            [
                "--source",
                "mp4",
                "--calibration-video",
                "clip.mp4",
                "--calibration-frame",
                "-1",
            ],
            "must be non-negative",
        ),
    ),
)
def test_controlled_replay_calibration_options_are_explicit_and_paired(
    arguments: list[str],
    message: str,
) -> None:
    args = build_parser().parse_args([*arguments, "--headless", "--max-frames", "1"])

    with pytest.raises(ValueError, match=message):
        run_teleop(args)


def test_selected_bundled_frame_calibrates_before_main_replay_and_is_provenanced() -> None:
    video = BUNDLED_VIDEO_PATHS["jumping-jacks"].resolve()
    args = build_parser().parse_args(
        [
            "--source",
            "mp4",
            "--demo-video",
            "jumping-jacks",
            "--headless",
            "--max-frames",
            "5",
            "--calibration-video",
            str(video),
            "--calibration-frame",
            "2",
        ]
    )

    stats = run_teleop(args)

    assert stats.frames == stats.skeleton_frames == 5
    assert stats.calibration_mode == "explicit_replay_frame"
    assert stats.calibration_source_path == str(video)
    assert stats.calibration_source_sha256 == hashlib.sha256(video.read_bytes()).hexdigest()
    assert stats.calibration_frame_index == 2


def test_missing_calibration_asset_fails_before_main_replay(tmp_path: Path) -> None:
    missing = tmp_path / "missing-neutral.mp4"
    args = build_parser().parse_args(
        [
            "--source",
            "mp4",
            "--headless",
            "--max-frames",
            "1",
            "--calibration-video",
            str(missing),
            "--calibration-frame",
            "0",
        ]
    )

    with pytest.raises(FileNotFoundError, match="calibration video"):
        run_teleop(args)
