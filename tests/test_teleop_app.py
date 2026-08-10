from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_human_interface.app.teleop import (
    BUNDLED_VIDEO_PATHS,
    DEMO_VIDEO_PATH,
    PROJECT_ROOT,
    _handle_key,
    _video_path,
    build_parser,
    main,
    run_teleop,
)


def test_parser_defaults_to_camera_and_fixed_base() -> None:
    args = build_parser().parse_args([])
    config = yaml.safe_load((PROJECT_ROOT / "config" / "camera.yaml").read_text(encoding="utf-8"))
    assert args.source == "camera"
    assert args.demo_video == "slow-balance"
    assert not args.free_base
    assert args.viewer_mode == "visual"
    assert args.camera_index == config["camera"]["index"]
    assert args.camera_width == config["camera"]["width"]
    assert args.camera_height == config["camera"]["height"]
    assert args.camera_fps == config["camera"]["fps"]
    assert args.camera_backend == config["camera"]["backend"]
    assert args.mirror_input is config["camera"]["mirror"]
    assert args.min_pose_detection_confidence == config["pose"]["min_pose_detection_confidence"]
    assert args.pose_model == PROJECT_ROOT / "assets" / "models" / "pose_landmarker_full.task"
    assert Path(args.pose_model).is_file()


def test_parser_selects_slow_bundled_demo() -> None:
    args = build_parser().parse_args(
        ["--source", "mp4", "--demo-video", "slow-balance"]
    )

    assert args.demo_video == "slow-balance"
    assert BUNDLED_VIDEO_PATHS == {
        "jumping-jacks": PROJECT_ROOT / "assets" / "videos" / "jumping_jacks_demo.mp4",
        "slow-balance": PROJECT_ROOT / "assets" / "videos" / "slow_balance_demo.mp4",
    }


def test_parser_can_select_original_fast_demo() -> None:
    args = build_parser().parse_args(
        ["--source", "mp4", "--demo-video", "jumping-jacks"]
    )

    assert args.demo_video == "jumping-jacks"


def test_video_path_uses_selected_bundled_demo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow_video = tmp_path / "slow.mp4"
    slow_video.touch()
    monkeypatch.setitem(BUNDLED_VIDEO_PATHS, "slow-balance", slow_video)
    args = build_parser().parse_args(
        ["--source", "mp4", "--demo-video", "slow-balance"]
    )

    assert _video_path(args) == slow_video.resolve()


def test_headless_synthetic_runs_the_full_command_pipeline() -> None:
    args = build_parser().parse_args(
        ["--source", "synthetic", "--headless", "--max-frames", "30"]
    )
    stats = run_teleop(args)
    assert stats.source == "synthetic"
    assert stats.base_mode == "fixed"
    assert stats.frames == stats.skeleton_frames == 30
    assert stats.stale_commands == 0
    assert stats.command_span_rad > 0.2
    # The grounded stabilizer allows vertical motion, so copied leg motion can
    # physically raise/lower the torso while both feet remain on the floor.
    assert 0.7 < stats.final_base_height_m < 1.0


def test_bundled_mp4_runs_media_pipe_retargeting_and_mujoco() -> None:
    assert DEMO_VIDEO_PATH.is_file()
    args = build_parser().parse_args(
        ["--source", "mp4", "--headless", "--max-frames", "90"]
    )

    stats = run_teleop(args)

    assert stats.source == "mp4"
    assert stats.frames == 90
    assert stats.skeleton_frames == 90
    assert stats.stale_commands == 0
    assert stats.command_span_rad > 0.01
    assert 0.7 < stats.final_base_height_m < 1.0


def test_main_prints_machine_readable_smoke_summary(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--source", "synthetic", "--headless", "--max-frames", "3"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "teleop complete:" in captured.out
    assert "source=synthetic" in captured.out
    assert "frames=3" in captured.out
    assert "base=fixed" in captured.out


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--source", "replay", "--headless", "--max-frames", "1"], "video-path"),
        (["--source", "synthetic", "--headless", "--max-frames", "-1"], "non-negative"),
        (
            [
                "--source",
                "synthetic",
                "--headless",
                "--max-frames",
                "1",
                "--physics-steps-per-frame",
                "0",
            ],
            "positive",
        ),
    ],
)
def test_invalid_runtime_options_are_rejected(arguments: list[str], message: str) -> None:
    args = build_parser().parse_args(arguments)
    with pytest.raises(ValueError, match=message):
        run_teleop(args)


class _FakeSimulation:
    def __init__(self) -> None:
        self.reset_count = 0
        self.viewer_mode = "visual"

    def reset(self) -> None:
        self.reset_count += 1

    def toggle_view_mode(self) -> str:
        self.viewer_mode = "joints" if self.viewer_mode == "visual" else "visual"
        return self.viewer_mode


class _FakeRetargeter:
    def __init__(self) -> None:
        self.calibration_samples: list[int] = []
        self.reset_count = 0

    def start_calibration(self, count: int) -> None:
        self.calibration_samples.append(count)

    def reset(self) -> None:
        self.reset_count += 1


def test_interactive_key_contract() -> None:
    simulation = _FakeSimulation()
    retargeter = _FakeRetargeter()

    keep_running, paused = _handle_key(
        ord(" "), simulation=simulation, retargeter=retargeter, paused=False
    )
    assert keep_running and paused

    keep_running, paused = _handle_key(
        ord("C"), simulation=simulation, retargeter=retargeter, paused=paused
    )
    assert keep_running and paused
    assert retargeter.calibration_samples == [15]

    keep_running, paused = _handle_key(
        ord("r"), simulation=simulation, retargeter=retargeter, paused=paused
    )
    assert keep_running and paused
    assert simulation.reset_count == retargeter.reset_count == 1

    keep_running, paused = _handle_key(
        ord("v"), simulation=simulation, retargeter=retargeter, paused=paused
    )
    assert keep_running and paused
    assert simulation.viewer_mode == "joints"

    keep_running, paused = _handle_key(
        27, simulation=simulation, retargeter=retargeter, paused=paused
    )
    assert not keep_running and paused
