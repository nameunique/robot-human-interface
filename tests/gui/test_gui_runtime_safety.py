from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QImage

from robot_human_interface.gui.resources import SourceItem
from robot_human_interface.gui.runtime import (
    ReadinessReason,
    RobotUiState,
    RuntimeMode,
    RuntimeStatus,
)
from robot_human_interface.gui.worker import DemoSession, PipelineWorker
from robot_human_interface.protocol import (
    OperatorSafetyAcknowledgement,
    SafeRobotController,
    finalize_safe_command,
)
from robot_human_interface.skeleton import JOINT_NAMES, RobotJointCommand


@dataclass
class _Transport:
    connected: bool = False
    payloads: list[str] = field(default_factory=list)

    def connect(self) -> None:
        self.connected = True

    def send(self, payload: str) -> None:
        self.payloads.append(payload)

    def close(self) -> None:
        self.connected = False


class _ProductionSession:
    def __init__(self, source: SourceItem) -> None:
        self.source = source
        self.running = False
        self.sequence = 0
        self.freeze = False
        self.snapshot_status = "RUNNING"

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def close(self) -> None:
        self.running = False

    def reset(self) -> None:
        self.sequence = 0

    def calibrate(self) -> None:
        return None

    def pause(self) -> None:
        self.freeze = True

    def resume(self) -> None:
        self.freeze = False

    def step(self):
        if not self.running or self.freeze:
            return None
        command = RobotJointCommand.humanoid(
            float(self.sequence),
            np.zeros(len(JOINT_NAMES)),
            1.0,
        )
        safe = finalize_safe_command(
            command,
            free_base_active=True,
            balance_active=True,
        )
        snapshot = SimpleNamespace(
            sequence=self.sequence,
            timestamp_s=float(self.sequence),
            status=self.snapshot_status,
            source=self.source,
            frame=QImage(8, 8, QImage.Format.Format_RGB32),
            landmarks=(),
            tracking_quality=1.0,
            safe_valid=True,
            safe_command=safe,
            telemetry={"free_base_active": True, "balance_active": True},
        )
        self.sequence += 1
        return snapshot


def _ack() -> OperatorSafetyAcknowledgement:
    return OperatorSafetyAcknowledgement(True, True, True)


def test_demo_snapshot_is_explicitly_non_authoritative(qtbot) -> None:
    del qtbot  # QApplication must exist before DemoSession paints its frame.
    source = SourceItem("synthetic:demo", "Demo", "synthetic")
    session = DemoSession(source, fallback_reason="test fallback")
    session.start()

    snapshot = session.step()

    assert snapshot is not None
    assert snapshot.safe_valid is False
    assert snapshot.safe_command is None
    assert session.runtime_mode is RuntimeMode.DEMO
    assert session.fallback_reason == "test fallback"


def test_worker_blocks_connect_and_arm_in_demo_without_creating_transport(qtbot) -> None:
    factory_calls: list[str] = []

    def robot_factory(endpoint: str):
        factory_calls.append(endpoint)
        return SafeRobotController(_Transport(), endpoint=endpoint)

    worker = PipelineWorker(
        session_factory=DemoSession,
        robot_factory=robot_factory,
    )
    events: list[dict[str, object]] = []
    worker.event_ready.connect(events.append)
    worker.start()
    worker.select_source(SourceItem("synthetic:demo", "Demo", "synthetic"))
    worker.start_pipeline()
    qtbot.waitUntil(lambda: worker.robot_readiness is not None, timeout=1000)
    qtbot.waitUntil(
        lambda: worker.runtime_status.mode is RuntimeMode.DEMO,
        timeout=1000,
    )

    worker.connect_robot()
    worker.arm_robot(acknowledgement=_ack())
    qtbot.waitUntil(
        lambda: {
            event.get("event_code") for event in events
        }
        >= {"ROBOT_CONNECT_BLOCKED_DEMO", "ROBOT_ARM_BLOCKED_DEMO"},
        timeout=1500,
    )

    assert factory_calls == []
    assert worker.robot_readiness is not None
    assert not worker.robot_readiness.ready
    assert worker.robot_readiness.reason_code is ReadinessReason.RUNTIME_DEMO
    assert worker.shutdown_and_wait(2000)


def test_worker_rejects_arm_without_operator_acknowledgement(qtbot) -> None:
    controller = SafeRobotController(_Transport())
    arm_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def unexpected_arm(
        *args: object,
        **kwargs: object,
    ) -> bool:
        arm_calls.append((args, kwargs))
        return True

    controller.arm = unexpected_arm  # type: ignore[method-assign]
    worker = PipelineWorker(
        session_factory=_ProductionSession,
        robot_factory=lambda _endpoint: controller,
    )
    events: list[dict[str, object]] = []
    worker.event_ready.connect(events.append)
    worker.start()
    try:
        worker.select_source(
            SourceItem("synthetic:production", "Production", "synthetic")
        )
        worker.start_pipeline()
        qtbot.waitUntil(
            lambda: worker.runtime_status.mode is RuntimeMode.PRODUCTION,
            timeout=1000,
        )
        worker.connect_robot()
        qtbot.waitUntil(
            lambda: worker.robot_readiness is not None
            and worker.robot_readiness.ready,
            timeout=1500,
        )

        worker.arm_robot()
        qtbot.waitUntil(
            lambda: any(
                event.get("event_code") == "ROBOT_ARM_ACK_REQUIRED"
                for event in events
            ),
            timeout=1000,
        )

        assert arm_calls == []
        assert not controller.is_armed
        assert controller.status().state.value == "connected_disarmed"
    finally:
        assert worker.shutdown_and_wait(2000)


def test_connect_success_without_authoritative_status_is_rejected(qtbot) -> None:
    del qtbot

    class StatuslessController:
        disconnected_reason: str | None = None

        def connect(self) -> bool:
            return True

        def disconnect(self, reason: str) -> None:
            self.disconnected_reason = reason

    controller = StatuslessController()
    worker = PipelineWorker(robot_factory=lambda _endpoint: controller)
    worker._runtime_status = RuntimeStatus.production()
    states: list[str] = []
    worker.robot_state_changed.connect(lambda state, _details: states.append(state))

    worker._connect_robot()

    assert controller.disconnected_reason == "connect_status_unavailable"
    assert "CONNECTED_DISARMED" not in states
    assert states[-1] == "DISCONNECTED"


def test_production_readiness_transitions_and_stale_watchdog(qtbot) -> None:
    sessions: list[_ProductionSession] = []
    transport = _Transport()
    controller = SafeRobotController(transport, max_command_age_s=0.12)

    def session_factory(source: SourceItem) -> _ProductionSession:
        session = _ProductionSession(source)
        sessions.append(session)
        return session

    worker = PipelineWorker(
        session_factory=session_factory,
        robot_factory=lambda _endpoint: controller,
        max_snapshot_age_s=0.12,
    )
    states: list[str] = []
    readiness = []
    worker.robot_state_changed.connect(lambda state, _details: states.append(state))
    worker.robot_readiness_changed.connect(readiness.append)
    worker.start()
    worker.select_source(SourceItem("synthetic:production", "Production", "synthetic"))
    worker.start_pipeline()
    qtbot.waitUntil(
        lambda: worker.runtime_status.mode is RuntimeMode.PRODUCTION,
        timeout=1000,
    )
    worker.connect_robot()
    qtbot.waitUntil(lambda: "CONNECTED_DISARMED" in states, timeout=1000)
    qtbot.waitUntil(
        lambda: worker.robot_readiness is not None and worker.robot_readiness.ready,
        timeout=1500,
    )

    worker.arm_robot(acknowledgement=_ack())
    qtbot.waitUntil(lambda: "ARMED" in states, timeout=1000)
    assert "CONNECTING" in states
    assert "ARMING" in states
    assert any(value.ready for value in readiness)

    worker.reset_pipeline()
    qtbot.waitUntil(lambda: "DISARMING" in states, timeout=1000)
    qtbot.waitUntil(
        lambda: controller.status().last_disarm_reason == "pipeline_reset",
        timeout=1000,
    )
    qtbot.waitUntil(
        lambda: worker.robot_readiness is not None and worker.robot_readiness.ready,
        timeout=1500,
    )
    worker.arm_robot(acknowledgement=_ack())
    qtbot.waitUntil(lambda: states.count("ARMED") >= 2, timeout=1000)

    worker.calibrate()
    qtbot.waitUntil(
        lambda: controller.status().last_disarm_reason == "calibration_started",
        timeout=1000,
    )
    qtbot.waitUntil(
        lambda: worker.robot_readiness is not None and worker.robot_readiness.ready,
        timeout=1500,
    )
    worker.arm_robot(acknowledgement=_ack())
    qtbot.waitUntil(lambda: states.count("ARMED") >= 3, timeout=1000)

    sessions[0].freeze = True
    qtbot.waitUntil(lambda: "DEGRADED" in states, timeout=1500)
    assert controller.status().last_disarm_reason == "stale_command"
    assert worker.robot_readiness is not None
    assert not worker.robot_readiness.ready
    assert worker.robot_readiness.robot_state in {
        RobotUiState.DEGRADED,
        RobotUiState.ARMED,
    }

    worker.disconnect_robot()
    qtbot.waitUntil(lambda: "DISCONNECTING" in states, timeout=1000)
    qtbot.waitUntil(lambda: states[-1] == "DISCONNECTED", timeout=1000)
    assert worker.shutdown_and_wait(2000)


def test_source_change_and_stop_disarm_an_armed_output(qtbot) -> None:
    sessions: list[_ProductionSession] = []
    controller = SafeRobotController(_Transport())

    def session_factory(source: SourceItem) -> _ProductionSession:
        session = _ProductionSession(source)
        sessions.append(session)
        return session

    worker = PipelineWorker(
        session_factory=session_factory,
        robot_factory=lambda _endpoint: controller,
    )
    worker.start()
    worker.select_source(SourceItem("synthetic:first", "First", "synthetic"))
    worker.start_pipeline()
    qtbot.waitUntil(
        lambda: worker.runtime_status.mode is RuntimeMode.PRODUCTION,
        timeout=1000,
    )
    worker.connect_robot()
    qtbot.waitUntil(
        lambda: worker.robot_readiness is not None and worker.robot_readiness.ready,
        timeout=1500,
    )
    worker.arm_robot(acknowledgement=_ack())
    qtbot.waitUntil(lambda: controller.is_armed, timeout=1000)

    worker.select_source(SourceItem("synthetic:second", "Second", "synthetic"))
    qtbot.waitUntil(lambda: len(sessions) == 2, timeout=1500)
    qtbot.waitUntil(
        lambda: controller.status().last_disarm_reason == "source_changed",
        timeout=1000,
    )
    qtbot.waitUntil(
        lambda: worker.robot_readiness is not None and worker.robot_readiness.ready,
        timeout=1500,
    )
    worker.arm_robot(acknowledgement=_ack())
    qtbot.waitUntil(lambda: controller.is_armed, timeout=1000)

    worker.stop_pipeline()
    qtbot.waitUntil(
        lambda: controller.status().last_disarm_reason == "pipeline_stopped",
        timeout=1000,
    )
    qtbot.waitUntil(lambda: worker._pipeline_state == "STOPPED", timeout=1000)
    assert not controller.is_armed
    assert worker.shutdown_and_wait(2000)


@pytest.mark.parametrize(
    ("terminal_status", "pipeline_state", "reason"),
    (
        ("ENDED", "ENDED", "pipeline_ended"),
        ("ERROR", "DEGRADED", "pipeline_error"),
    ),
)
def test_terminal_snapshot_disarms_and_closes_session(
    qtbot,
    terminal_status: str,
    pipeline_state: str,
    reason: str,
) -> None:
    sessions: list[_ProductionSession] = []
    controller = SafeRobotController(_Transport())

    def session_factory(source: SourceItem) -> _ProductionSession:
        session = _ProductionSession(source)
        sessions.append(session)
        return session

    worker = PipelineWorker(
        session_factory=session_factory,
        robot_factory=lambda _endpoint: controller,
    )
    worker.start()
    worker.select_source(SourceItem("synthetic:terminal", "Terminal", "synthetic"))
    worker.start_pipeline()
    qtbot.waitUntil(
        lambda: worker.runtime_status.mode is RuntimeMode.PRODUCTION,
        timeout=1000,
    )
    worker.connect_robot()
    qtbot.waitUntil(
        lambda: worker.robot_readiness is not None and worker.robot_readiness.ready,
        timeout=1500,
    )
    worker.arm_robot(acknowledgement=_ack())
    qtbot.waitUntil(lambda: controller.is_armed, timeout=1000)

    sessions[0].snapshot_status = terminal_status
    qtbot.waitUntil(lambda: worker._pipeline_state == pipeline_state, timeout=1500)
    assert controller.status().last_disarm_reason == reason
    assert not controller.is_armed
    assert not sessions[0].running
    assert worker.shutdown_and_wait(2000)
