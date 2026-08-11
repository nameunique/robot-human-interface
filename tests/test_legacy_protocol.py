from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, radians

import numpy as np
import pytest

from robot_human_interface.protocol import (
    LatestCommandPublisher,
    LegacyWebSocketEncoder,
    WebSocketTransport,
)
from robot_human_interface.skeleton import JOINT_NAMES, RobotJointCommand


def _command(positions_rad: list[float], *, timestamp_s: float = 0.0) -> RobotJointCommand:
    return RobotJointCommand.humanoid(timestamp_s, positions_rad, 1.0)


def test_encoder_emits_exact_compact_unity_json_in_degrees() -> None:
    positions = [0.0] * len(JOINT_NAMES)
    positions[0] = pi / 2
    positions[1] = -pi / 2
    positions[2] = pi / 4

    payload = LegacyWebSocketEncoder().encode(_command(positions))

    assert payload == (
        '{"id":0,"method":"setPositions","params":['
        "90.0,-90.0,45.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"
        "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}"
    )


def test_encoder_reorders_a_complete_named_command_to_canonical_order() -> None:
    canonical_degrees = [0.0] * len(JOINT_NAMES)
    canonical_degrees[0:6] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    canonical_degrees[18:20] = [-10.0, 25.0]
    names = tuple(reversed(JOINT_NAMES))
    by_name = dict(zip(JOINT_NAMES, canonical_degrees, strict=True))
    positions = [radians(by_name[name]) for name in names]
    command = RobotJointCommand(0.0, names, np.asarray(positions), 1.0)

    payload = LegacyWebSocketEncoder().encode(command)

    assert payload == (
        '{"id":0,"method":"setPositions","params":['
        "10.0,20.0,30.0,40.0,50.0,60.0,0.0,0.0,0.0,0.0,"
        "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-10.0,25.0]}"
    )


def test_encoder_rejects_wrong_schema_nonfinite_and_out_of_limit_targets() -> None:
    encoder = LegacyWebSocketEncoder()
    incomplete = RobotJointCommand(
        0.0,
        JOINT_NAMES[:-1],
        np.zeros(len(JOINT_NAMES) - 1),
        1.0,
    )
    with pytest.raises(ValueError, match="joint names do not match schema"):
        encoder.encode(incomplete)

    outside = [0.0] * len(JOINT_NAMES)
    outside[-1] = radians(71.0)
    with pytest.raises(ValueError, match="head.*outside"):
        encoder.encode(_command(outside))

    corrupted = _command([0.0] * len(JOINT_NAMES))
    object.__setattr__(corrupted, "positions_rad", np.full(len(JOINT_NAMES), np.nan))
    with pytest.raises(ValueError, match="finite"):
        encoder.encode(corrupted)


def test_encoder_accepts_explicit_named_radian_limits() -> None:
    limits = {name: (-0.5, 0.5) for name in JOINT_NAMES}
    encoder = LegacyWebSocketEncoder(limits_rad=limits)
    positions = [0.0] * len(JOINT_NAMES)
    positions[0] = 0.25
    assert '"params":[14.323944878,' in encoder.encode(_command(positions))

    positions[0] = 0.51
    with pytest.raises(ValueError, match="shoulder_rh.*outside"):
        encoder.encode(_command(positions))


@dataclass
class _RecordingTransport:
    failures_remaining: int = 0
    payloads: list[str] = field(default_factory=list)

    def send(self, payload: str) -> None:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise ConnectionError("robot disconnected")
        self.payloads.append(payload)


def test_publisher_rate_gate_uses_latest_command_and_isolates_send_errors() -> None:
    now = [0.0]
    transport = _RecordingTransport()
    publisher = LatestCommandPublisher(transport, clock=lambda: now[0])

    first = [0.0] * len(JOINT_NAMES)
    first[0] = radians(10.0)
    publisher.submit(_command(first))
    assert publisher.tick()
    assert len(transport.payloads) == 1

    second = [0.0] * len(JOINT_NAMES)
    second[0] = radians(20.0)
    third = [0.0] * len(JOINT_NAMES)
    third[0] = radians(30.0)
    publisher.submit(_command(second))
    publisher.submit(_command(third))
    now[0] = 0.099
    assert not publisher.tick()
    assert len(transport.payloads) == 1

    now[0] = 0.1
    assert publisher.tick()
    assert len(transport.payloads) == 2
    assert '"params":[30.0,' in transport.payloads[-1]
    assert not publisher.has_pending_command

    fourth = [0.0] * len(JOINT_NAMES)
    fourth[0] = radians(40.0)
    publisher.submit(_command(fourth))
    transport.failures_remaining = 1
    now[0] = 0.2
    assert not publisher.tick()  # A disconnect is data, not a control-loop exception.
    assert isinstance(publisher.last_error, ConnectionError)
    assert publisher.has_pending_command

    now[0] = 0.3
    assert publisher.tick()
    assert publisher.last_error is None
    assert not publisher.has_pending_command
    assert publisher.attempt_count == 4
    assert publisher.sent_count == 3


def test_optional_repeat_latest_matches_the_legacy_ten_hz_heartbeat() -> None:
    now = [0.0]
    transport = _RecordingTransport()
    publisher = LatestCommandPublisher(
        transport,
        repeat_latest=True,
        clock=lambda: now[0],
    )
    publisher.submit(_command([0.0] * len(JOINT_NAMES)))

    assert publisher.tick()
    now[0] = 0.099
    assert not publisher.tick()
    now[0] = 0.1
    assert publisher.tick()

    assert len(transport.payloads) == 2
    assert transport.payloads[0] == transport.payloads[1]


class WebSocketTimeoutException(Exception):
    pass


@dataclass
class _ReplyingConnection:
    replies: list[str]
    sent: list[str] = field(default_factory=list)
    timeouts: list[float] = field(default_factory=list)
    closed: bool = False

    def send(self, payload: str) -> None:
        self.sent.append(payload)

    def recv(self) -> str:
        if self.replies:
            return self.replies.pop(0)
        raise WebSocketTimeoutException("no queued frame")

    def settimeout(self, timeout_s: float) -> None:
        self.timeouts.append(timeout_s)

    def close(self) -> None:
        self.closed = True


def test_websocket_transport_drains_server_replies_without_blocking() -> None:
    connection = _ReplyingConnection(["ack-1", "ack-2"])
    transport = WebSocketTransport(
        "ws://127.0.0.1:9000",
        timeout_s=0.5,
        connection_factory=lambda url, timeout: connection,
    )

    transport.send("positions")

    assert connection.sent == ["positions"]
    assert transport.received_count == 2
    assert transport.last_receive_error is None
    assert connection.timeouts == [0.0, 0.5]
    assert transport.is_connected
