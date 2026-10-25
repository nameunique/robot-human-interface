from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from robot_human_interface.app.teleop import _make_robot_publisher, _validate_args, build_parser
from robot_human_interface.model_profiles import get_model_profile
from robot_human_interface.protocol import (
    OperatorSafetyAcknowledgement,
    RobotConnectionState,
    SafeRobotController,
    finalize_safe_command,
)
from robot_human_interface.resources import ResourceLocator
from robot_human_interface.session import SessionConfig, SourceKind, SourceSpec
from robot_human_interface.skeleton import RobotJointCommand


@dataclass
class Transport:
    payloads: list[str] = field(default_factory=list)

    def connect(self):
        pass

    def send(self, payload):
        self.payloads.append(payload)

    def close(self):
        pass


def safe_command(verified: bool):
    return finalize_safe_command(
        RobotJointCommand.humanoid(0.0, np.zeros(20), 1.0),
        free_base_active=True,
        balance_active=True,
        hardware_mapping_verified=verified,
    )


def test_entrypoint_defaults_and_legacy_selection():
    source = SourceSpec(SourceKind.SYNTHETIC, "test", "Test")
    assert SessionConfig(source).model_profile == "volodka"
    assert build_parser().parse_args([]).model_profile == "volodka"
    assert build_parser().parse_args(["--model-profile", "unity_legacy"]).model_profile == "unity_legacy"
    with pytest.raises(ValueError, match="Unknown model profile"):
        SessionConfig(source, model_profile="missing")


def test_profile_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    resources = ResourceLocator(root)
    monkeypatch.chdir(tmp_path)
    profile = get_model_profile()
    assert profile.scene_path("free", resources) == root / "models/volodka/scene_free.xml"
    assert profile.joints_path(resources) == root / "config/joints_volodka.yaml"
    assert profile.balance_path(resources) == root / "config/balance_volodka.yaml"
    legacy = get_model_profile("unity_legacy")
    assert legacy.scene_path("fixed", resources) == root / "models/humanoid/scene_fixed.xml"
    assert legacy.joints_path(resources) == root / "config/joints.yaml"


@pytest.mark.parametrize("entrypoint", [_validate_args, _make_robot_publisher])
def test_cli_rejects_cad_hardware_before_constructing_transport(entrypoint, monkeypatch):
    import robot_human_interface.protocol as protocol

    def forbidden(*args, **kwargs):
        pytest.fail("unverified profile must fail before constructing transport")

    monkeypatch.setattr(protocol, "WebSocketTransport", forbidden)
    args = build_parser().parse_args(["--robot-websocket-url", "ws://localhost:8765"])
    with pytest.raises(ValueError, match="не проверено"):
        entrypoint(args)


def test_unverified_mapping_survives_copy_and_never_arms():
    transport = Transport()
    controller = SafeRobotController(transport, clock=lambda: 0.0)
    assert controller.connect()
    command = safe_command(False)
    assert command.copy().hardware_mapping_verified is False
    controller.submit_safe_command(command)
    allowed, reason = controller.can_arm()
    assert not allowed and "mapping is not verified" in reason
    assert not controller.arm(OperatorSafetyAcknowledgement(True, True, True))
    assert not controller.tick()
    assert transport.payloads == []


def test_switch_to_unverified_mapping_revokes_armed_output():
    transport = Transport()
    controller = SafeRobotController(transport, clock=lambda: 0.0)
    controller.connect()
    controller.submit_safe_command(safe_command(True))
    assert controller.arm(OperatorSafetyAcknowledgement(True, True, True))
    with pytest.raises(ValueError, match="mapping is not verified"):
        controller.submit_safe_command(safe_command(False))
    assert controller.state is RobotConnectionState.DEGRADED
    assert not controller.tick()
    assert transport.payloads == []


def test_hardware_mapping_flag_rejects_truthy_non_boolean():
    with pytest.raises(ValueError, match="hardware_mapping_verified"):
        safe_command("yes")


@pytest.mark.parametrize("retargeting", ["ik", "geometric"])
def test_default_pipeline_uses_one_cad_profile_end_to_end(retargeting):
    from robot_human_interface.pipeline import DefaultTeleopPipeline

    config = SessionConfig(
        SourceSpec(
            SourceKind.SYNTHETIC, "cad-smoke", "CAD smoke",
            width=160, height=120, fps=120.0,
        ),
        retargeting=retargeting,
        physics_steps_per_frame=1,
    )
    pipeline = DefaultTeleopPipeline(config)
    try:
        pipeline.start()
        simulation = pipeline.simulation
        assert float(simulation.model.body_mass.sum()) == pytest.approx(3.0942)
        assert simulation.model_path == get_model_profile().scene_path("free")
        snapshot = pipeline.step()
        assert snapshot is not None
        assert snapshot.frame is not None and snapshot.skeleton is not None
        assert snapshot.telemetry["model_profile"] == "volodka"
        assert snapshot.telemetry["hardware_mapping_verified"] is False
        assert snapshot.safe_command is not None
        assert snapshot.safe_command.hardware_mapping_verified is False
        assert np.isfinite(snapshot.safe_command.positions_rad).all()
        assert snapshot.telemetry["base_height_m"] < 0.4
    finally:
        pipeline.close()


def test_cad_pipeline_preserves_camera_configuration_and_cleanup(monkeypatch):
    import robot_human_interface.camera as camera
    import robot_human_interface.pose as pose
    from robot_human_interface.pipeline import DefaultTeleopPipeline

    observed = []
    closed = []

    class MockCamera(camera.SyntheticCameraSource):
        def close(self):
            closed.append("camera")
            super().close()

    def make_camera(config):
        observed.append(config)
        return MockCamera(camera.SyntheticCameraConfig(
            width=config.width, height=config.height, fps=config.fps,
            realtime=False,
        ))

    monkeypatch.setattr(camera, "OpenCVCameraSource", make_camera)
    monkeypatch.setattr(pose, "MediaPipePoseLandmarker", lambda *a, **k: pose.SyntheticPoseEstimator())
    source = SourceSpec(
        SourceKind.CAMERA, "camera:2", "Mock camera", camera_index=2,
        width=160, height=120, fps=30.0, mirror=True,
    )
    pipeline = DefaultTeleopPipeline(SessionConfig(source, physics_steps_per_frame=1))
    try:
        pipeline.start()
        snapshot = pipeline.step()
        assert snapshot.source.kind is SourceKind.CAMERA
        assert snapshot.telemetry["model_profile"] == "volodka"
        assert snapshot.frame.image_bgr.shape == (120, 160, 3)
        assert observed[0].index == 2
        assert observed[0].mirror is True
        assert observed[0].backend == "auto"
    finally:
        pipeline.close()
    assert closed == ["camera"]


def test_cad_cli_processes_bundled_video_with_real_mediapipe():
    from robot_human_interface.app.teleop import run_teleop

    args = build_parser().parse_args([
        "--source", "mp4", "--headless", "--max-frames", "60",
    ])
    stats = run_teleop(args)
    assert stats.frames == stats.skeleton_frames == 60
    assert stats.stale_commands == 0
    assert stats.command_span_rad > 0.01
    assert stats.safe_command_span_rad > 0.01
    assert 0.15 < stats.final_base_height_m < 0.3
    assert not stats.fell
    assert not stats.robot_output_enabled
