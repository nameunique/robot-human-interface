from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_human_interface.app.teleop import (
    PROJECT_ROOT,
    _handle_key,
    build_parser,
    main,
    run_teleop,
)


def test_parser_defaults_to_camera_and_fixed_base() -> None:
    args = build_parser().parse_args([])
    config = yaml.safe_load((PROJECT_ROOT / "config" / "camera.yaml").read_text(encoding="utf-8"))
    assert args.source == "camera"
    assert not args.free_base
    assert args.camera_index == config["camera"]["index"]
    assert args.camera_width == config["camera"]["width"]
    assert args.camera_height == config["camera"]["height"]
    assert args.camera_fps == config["camera"]["fps"]
    assert args.camera_backend == config["camera"]["backend"]
    assert args.mirror_input is config["camera"]["mirror"]
    assert args.min_pose_detection_confidence == config["pose"]["min_pose_detection_confidence"]
    assert args.pose_model == PROJECT_ROOT / "assets" / "models" / "pose_landmarker_full.task"
    assert Path(args.pose_model).is_file()


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
    assert stats.final_base_height_m == pytest.approx(0.9275, abs=0.01)


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
        (["--source", "replay", "--headless", "--max-frames", "1"], "replay-path"),
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

    def reset(self) -> None:
        self.reset_count += 1


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
        27, simulation=simulation, retargeter=retargeter, paused=paused
    )
    assert not keep_running and paused
