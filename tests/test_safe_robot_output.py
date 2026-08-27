from __future__ import annotations

from dataclasses import dataclass, field
from math import radians

import numpy as np
import pytest

from robot_human_interface.protocol import (
    OperatorSafetyAcknowledgement,
    RobotConnectionState,
    SafeRobotController,
    WebSocketTransport,
    encode_legacy_velocities,
    finalize_safe_command,
)
from robot_human_interface.skeleton import JOINT_NAMES, RobotJointCommand


ACK = OperatorSafetyAcknowledgement(True, True, True)


def _command(
    degrees: float = 0.0,
    *,
    timestamp_s: float = 0.0,
    stale: bool = False,
) -> RobotJointCommand:
    positions = np.zeros(len(JOINT_NAMES))
    positions[0] = radians(degrees)
    return RobotJointCommand.humanoid(
        timestamp_s,
        positions,
        1.0,
        stale=stale,
    )


def _safe_command(
    degrees: float = 0.0,
    *,
    timestamp_s: float = 0.0,
    stale: bool = False,
    free_base_active: bool = True,
    balance_active: bool = True,
):
    return finalize_safe_command(
        _command(degrees, timestamp_s=timestamp_s, stale=stale),
        free_base_active=free_base_active,
        balance_active=balance_active,
    )


@dataclass
class _Transport:
    fail_connect: bool = False
    fail_send: bool = False
    connected: bool = False
    closed: int = 0
    payloads: list[str] = field(default_factory=list)

    def connect(self) -> None:
        if self.fail_connect:
            raise ConnectionError("connect failed")
        self.connected = True

    def send(self, payload: str) -> None:
        if self.fail_send:
            raise ConnectionError("send failed")
        self.payloads.append(payload)

    def close(self) -> None:
        self.connected = False
        self.closed += 1


def test_connect_is_silent_and_arm_requires_every_interlock() -> None:
    now = [10.0]
    transport = _Transport()
    output = SafeRobotController(transport, clock=lambda: now[0])

    assert output.connect()
    assert output.state is RobotConnectionState.CONNECTED_DISARMED
    assert transport.connected
    assert transport.payloads == []
    assert not output.arm(ACK)

    output.submit_safe_command(
        _safe_command(10.0, free_base_active=False),
        free_base=False,
        balance_enabled=True,
    )
    assert not output.arm(ACK)

    output.submit_safe_command(
        _safe_command(10.0),
        free_base=True,
        balance_enabled=True,
    )
    assert not output.arm(OperatorSafetyAcknowledgement(True, True, False))
    assert output.arm(ACK)
    assert transport.payloads == []  # Arm itself performs no I/O.

    assert output.tick()
    assert len(transport.payloads) == 1
    assert '"method":"setPositions"' in transport.payloads[0]
    assert '"params":[10.0,' in transport.payloads[0]


def test_positions_are_capped_at_ten_hz_and_only_latest_safe_command_is_sent() -> None:
    now = [0.0]
    transport = _Transport()
    output = SafeRobotController(transport, clock=lambda: now[0])
    output.connect()
    output.submit_safe_command(
        _safe_command(10.0), free_base=True, balance_enabled=True
    )
    assert output.arm(ACK)
    assert output.tick()

    now[0] = 0.05
    output.submit_safe_command(
        _safe_command(20.0), free_base=True, balance_enabled=True
    )
    assert not output.tick()
    now[0] = 0.099
    output.submit_safe_command(
        _safe_command(30.0), free_base=True, balance_enabled=True
    )
    assert not output.tick()
    now[0] = 0.1
    assert output.tick()

    assert len(transport.payloads) == 2
    assert '"params":[30.0,' in transport.payloads[-1]
    assert output.position_attempts == output.positions_sent == 2


def test_stale_network_and_invalid_command_faults_require_manual_rearm() -> None:
    now = [1.0]
    transport = _Transport()
    output = SafeRobotController(transport, clock=lambda: now[0])
    output.connect()
    output.submit_safe_command(
        _safe_command(), free_base=True, balance_enabled=True
    )
    assert output.arm(ACK)
    assert output.tick()

    now[0] = 1.501
    assert not output.tick()
    assert output.state is RobotConnectionState.DEGRADED
    assert not output.arm(ACK)

    # Explicit reconnect plus a new safe command and acknowledgement is needed.
    assert output.connect()
    output.submit_safe_command(
        _safe_command(), free_base=True, balance_enabled=True
    )
    assert output.arm(ACK)
    transport.fail_send = True
    assert not output.tick()
    assert output.state is RobotConnectionState.DEGRADED
    assert isinstance(output.last_error, ConnectionError)

    transport.fail_send = False
    assert output.connect()
    output.submit_safe_command(
        _safe_command(), free_base=True, balance_enabled=True
    )
    assert output.arm(ACK)
    # Even a finite, canonical raw retargeting command has no final-stage
    # provenance and must fault an already armed output.
    with pytest.raises(TypeError, match="provenance"):
        output.submit_safe_command(
            _command(),
            free_base=True,
            balance_enabled=True,
        )
    assert output.state is RobotConnectionState.DEGRADED


def test_source_invalidation_disarms_and_discards_previous_pose() -> None:
    transport = _Transport()
    output = SafeRobotController(transport, clock=lambda: 0.0)
    output.connect()
    output.submit_safe_command(
        _safe_command(), free_base=True, balance_enabled=True
    )
    assert output.arm(ACK)

    output.invalidate("source_changed")

    assert output.state is RobotConnectionState.CONNECTED_DISARMED
    assert not output.arm(ACK)
    assert output.status().last_disarm_reason == "no validated safe command is available"


def test_optional_velocity_compatibility_message_is_off_by_default_and_once_when_on() -> None:
    assert encode_legacy_velocities() == (
        '{"id":0,"method":"setVelocities","params":['
        + ",".join(["100"] * len(JOINT_NAMES))
        + "]}"
    )

    default_transport = _Transport()
    default = SafeRobotController(default_transport, clock=lambda: 0.0)
    default.connect()
    default.submit_safe_command(
        _safe_command(), free_base=True, balance_enabled=True
    )
    assert default.arm(ACK)
    assert default.tick()
    assert all("setVelocities" not in payload for payload in default_transport.payloads)

    now = [0.0]
    compatible_transport = _Transport()
    compatible = SafeRobotController(
        compatible_transport,
        clock=lambda: now[0],
    )
    compatible.connect()
    compatible.submit_safe_command(
        _safe_command(), free_base=True, balance_enabled=True
    )
    # The operator may enable the advanced compatibility message for this arm
    # action; it remains disabled by default in every new controller.
    assert compatible.arm(ACK, send_velocities=True)
    assert compatible.tick()
    now[0] = 0.1
    compatible.submit_safe_command(
        _safe_command(), free_base=True, balance_enabled=True
    )
    assert compatible.tick()

    methods = ["setVelocities" in payload for payload in compatible_transport.payloads]
    assert methods == [True, False, False]
    assert compatible.status().velocity_messages_sent == 1


def test_connect_failure_enters_degraded_without_sending() -> None:
    transport = _Transport(fail_connect=True)
    output = SafeRobotController(transport, clock=lambda: 0.0)

    assert not output.connect()
    assert output.state is RobotConnectionState.DEGRADED
    assert transport.payloads == []


def test_rejected_raw_submit_revokes_retained_pose_even_while_disarmed() -> None:
    transport = _Transport()
    output = SafeRobotController(transport, clock=lambda: 0.0)
    output.connect()
    output.submit_safe_command(_safe_command())
    assert output.can_arm() == (True, None)

    with pytest.raises(TypeError, match="provenance"):
        output.submit_safe_command(_command())

    ready, reason = output.can_arm()
    assert not ready
    assert reason == "no validated safe command is available"
    assert not output.arm(ACK)
    assert transport.payloads == []


def test_arm_is_bound_to_the_authoritative_safe_command_generation() -> None:
    transport = _Transport()
    output = SafeRobotController(transport, clock=lambda: 0.0)
    assert output.connect()
    first_generation = output.submit_safe_command(_safe_command(10.0))
    second_generation = output.submit_safe_command(_safe_command(20.0))

    assert second_generation == first_generation + 1
    assert not output.arm(
        ACK,
        expected_command_generation=first_generation,
    )
    assert output.status().last_disarm_reason == "safe_command_generation_changed"
    assert output.arm(
        ACK,
        expected_command_generation=second_generation,
    )
    assert transport.payloads == []


class WebSocketTimeoutException(Exception):
    pass


@dataclass
class _ReceiveFaultConnection:
    restore_timeout_fails: bool = False
    recv_fails: bool = False
    sent: list[str] = field(default_factory=list)
    closed: bool = False

    def send(self, payload: str) -> None:
        self.sent.append(payload)

    def recv(self) -> str:
        if self.recv_fails:
            raise ConnectionError("receive channel failed")
        raise WebSocketTimeoutException("drain complete")

    def settimeout(self, timeout_s: float) -> None:
        if self.restore_timeout_fails and timeout_s > 0.0:
            raise RuntimeError("timeout restoration failed")

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    "connection, message",
    (
        (_ReceiveFaultConnection(recv_fails=True), "receive channel failed"),
        (
            _ReceiveFaultConnection(restore_timeout_fails=True),
            "timeout restoration failed",
        ),
    ),
)
def test_receive_side_websocket_fault_degrades_without_armed_auto_reconnect(
    connection: _ReceiveFaultConnection,
    message: str,
) -> None:
    now = [0.0]
    factory_calls = 0

    def factory(_url: str, _timeout_s: float) -> _ReceiveFaultConnection:
        nonlocal factory_calls
        factory_calls += 1
        return connection

    transport = WebSocketTransport(
        "ws://127.0.0.1:9000",
        connection_factory=factory,
    )
    output = SafeRobotController(transport, clock=lambda: now[0])
    assert output.connect()
    output.submit_safe_command(_safe_command())
    assert output.arm(ACK)

    assert not output.tick()
    assert output.state is RobotConnectionState.DEGRADED
    assert message in str(output.last_error)
    assert connection.closed
    assert factory_calls == 1

    now[0] = 1.0
    assert not output.tick()
    assert factory_calls == 1  # no reconnect without a new manual connect/arm
