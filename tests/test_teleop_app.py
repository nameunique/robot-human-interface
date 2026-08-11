from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
import robot_human_interface.app.teleop as teleop_module

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
    assert args.balance_controller
    assert args.physics_steps_per_frame == 0
    assert args.viewer_mode == "visual"
    assert args.robot_websocket_url == ""
    assert args.robot_websocket_timeout_s == 0.5
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
        ["--source", "synthetic", "--headless", "--max-frames", "60"]
    )
    stats = run_teleop(args)
    assert stats.source == "synthetic"
    assert stats.base_mode == "fixed"
    assert stats.frames == stats.skeleton_frames == 60
    assert stats.stale_commands == 0
    assert stats.command_span_rad > 0.2
    assert abs(stats.simulation_time_s - stats.media_time_s) <= 0.0021
    assert not stats.fell
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


def test_full_slow_mp4_lifts_both_legs_without_free_base_fall() -> None:
    args = build_parser().parse_args(
        [
            "--source",
            "mp4",
            "--demo-video",
            "slow-balance",
            "--headless",
            "--free-base",
        ]
    )

    stats = run_teleop(args)

    assert stats.frames == 1961
    assert stats.skeleton_frames >= 1960
    assert not stats.fell
    assert stats.final_base_height_m > 0.82
    assert np.degrees(stats.maximum_tilt_rad) < 18.0
    assert stats.right_swing_completed and stats.left_swing_completed
    assert stats.maximum_right_foot_clearance_m > 0.025
    assert stats.maximum_left_foot_clearance_m > 0.025
    assert abs(stats.simulation_time_s - stats.media_time_s) <= 0.0021


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
                "-1",
            ],
            "non-negative",
        ),
        (
            [
                "--source",
                "synthetic",
                "--headless",
                "--max-frames",
                "1",
                "--robot-websocket-timeout-s",
                "0",
            ],
            "finite and positive",
        ),
        (
            [
                "--source",
                "synthetic",
                "--headless",
                "--max-frames",
                "1",
                "--robot-websocket-url",
                "http://127.0.0.1:9000",
            ],
            "absolute ws",
        ),
        (
            [
                "--source",
                "synthetic",
                "--headless",
                "--max-frames",
                "1",
                "--robot-websocket-url",
                "ws://127.0.0.1:9000",
            ],
            "requires --free-base",
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


class _FakeResetter:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def test_interactive_key_contract() -> None:
    simulation = _FakeSimulation()
    retargeter = _FakeRetargeter()
    calibration_resetter = _FakeResetter()

    keep_running, paused = _handle_key(
        ord(" "), simulation=simulation, retargeter=retargeter, paused=False
    )
    assert keep_running and paused

    keep_running, paused = _handle_key(
        ord("C"),
        simulation=simulation,
        retargeter=retargeter,
        paused=paused,
        calibration_resetters=(calibration_resetter,),
    )
    assert keep_running and paused
    assert retargeter.calibration_samples == [30]
    assert calibration_resetter.reset_count == 1

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


def test_pause_is_rejected_while_physical_robot_output_is_active() -> None:
    simulation = _FakeSimulation()
    retargeter = _FakeRetargeter()

    keep_running, paused = _handle_key(
        ord(" "),
        simulation=simulation,
        retargeter=retargeter,
        paused=False,
        allow_pause=False,
    )

    assert keep_running
    assert not paused


def test_free_base_uses_motor_angle_balance_at_every_physics_tick() -> None:
    args = build_parser().parse_args(
        ["--source", "synthetic", "--headless", "--free-base", "--max-frames", "300"]
    )

    stats = run_teleop(args)

    assert stats.base_mode == "free"
    assert not stats.fell
    assert stats.final_base_height_m > 0.85
    assert np.degrees(stats.maximum_tilt_rad) < 10.0
    assert stats.safe_command_span_rad > 0.2
    assert abs(stats.simulation_time_s - stats.media_time_s) <= 0.0021


class _FakeRobotPublisher:
    def __init__(self) -> None:
        self.commands = []
        self.attempt_count = 3
        self.sent_count = 2
        self.last_error = ConnectionError("disconnected")
        self.stopped = False
        self.started = False

    def submit(self, command: object) -> int:
        self.commands.append(command)
        return len(self.commands)

    def stop(self) -> None:
        self.stopped = True

    def start(self) -> None:
        self.started = True


def test_optional_robot_output_receives_the_same_safe_motor_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_human_interface.simulation import HumanoidSimulation

    publisher = _FakeRobotPublisher()
    applied_commands = []
    original_apply = HumanoidSimulation.apply_joint_command

    def record_apply(simulation: HumanoidSimulation, command: object) -> None:
        applied_commands.append(command)
        original_apply(simulation, command)

    monkeypatch.setattr(HumanoidSimulation, "apply_joint_command", record_apply)
    monkeypatch.setattr(teleop_module, "_make_robot_publisher", lambda args: publisher)
    args = build_parser().parse_args(
        [
            "--source",
            "synthetic",
            "--headless",
            "--free-base",
            "--max-frames",
            "60",
            "--robot-websocket-url",
            "ws://127.0.0.1:9000",
        ]
    )

    stats = run_teleop(args)

    assert publisher.started and publisher.stopped
    assert publisher.commands
    assert len(publisher.commands) == len(applied_commands)
    assert all(sent is applied for sent, applied in zip(publisher.commands, applied_commands, strict=True))
    assert all(command.joint_names == publisher.commands[0].joint_names for command in publisher.commands)
    assert stats.robot_output_enabled
    assert stats.robot_send_attempts == 3
    assert stats.robot_commands_sent == 2
    assert stats.robot_last_error == "disconnected"
