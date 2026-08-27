"""Operator-gated output for the legacy Unity robot WebSocket.

This module deliberately sits above :mod:`legacy_websocket`.  The legacy
publisher remains useful for tests and backwards-compatible CLI runs, while
``SafeRobotController`` implements the stricter lifecycle required by the
operator GUI:

* connecting never sends a motor command;
* arming is explicit and requires a fresh, validated *safe* command;
* only canonical 20-joint commands can cross the protocol boundary;
* every fault stops output and requires a manual reconnect/re-arm;
* the positional stream is capped at 10 Hz.

This is an output interlock, not a software emergency stop.  It cannot prove
that the hardware reached or retained any particular pose after output stops.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from time import monotonic
from typing import Protocol

from robot_human_interface.skeleton import JOINT_NAMES, RobotJointCommand

from .legacy_websocket import (
    DEFAULT_RATE_HZ,
    LEGACY_REQUEST_ID,
    CommandTransport,
    LegacyWebSocketEncoder,
)


VELOCITY_METHOD = "setVelocities"
DEFAULT_MAX_COMMAND_AGE_S = 0.5
DEFAULT_VELOCITY_DEG_S = 100
FINAL_SAFETY_STAGE = "balance_support_final"
_FINAL_COMMAND_PROOF = object()


class RobotConnectionState(str, Enum):
    """Observable physical-output lifecycle."""

    DISCONNECTED = "disconnected"
    CONNECTED_DISARMED = "connected_disarmed"
    ARMED = "armed"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class FinalizedSafeCommand:
    """A copied motor command minted only by the final safety stage.

    A plain :class:`RobotJointCommand` is intentionally not accepted by the
    physical-output controller.  The private proof prevents callers from
    merely relabelling a raw retargeting result as safe by constructing this
    dataclass directly; custom pipelines must explicitly pass through
    :func:`finalize_safe_command` after their balance/support layer.
    """

    _command: RobotJointCommand
    free_base_active: bool
    balance_active: bool
    stage: str = FINAL_SAFETY_STAGE
    _proof: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._proof is not _FINAL_COMMAND_PROOF:
            raise TypeError(
                "FinalizedSafeCommand must be created by finalize_safe_command"
            )
        if not isinstance(self._command, RobotJointCommand):
            raise TypeError("finalized command must wrap a RobotJointCommand")
        if self._command.joint_names != JOINT_NAMES:
            raise ValueError("finalized command must use canonical Unity joint order")
        if type(self.free_base_active) is not bool or type(self.balance_active) is not bool:
            raise ValueError("finalized safety mode flags must be booleans")
        if self.stage != FINAL_SAFETY_STAGE:
            raise ValueError("unknown final safety stage")
        copied = RobotJointCommand(
            self._command.timestamp_s,
            self._command.joint_names,
            self._command.positions_rad.copy(),
            self._command.confidence,
            self._command.stale,
        )
        object.__setattr__(self, "_command", copied)

    @property
    def timestamp_s(self) -> float:
        return self._command.timestamp_s

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._command.joint_names

    @property
    def positions_rad(self):
        return self._command.positions_rad

    @property
    def confidence(self) -> float:
        return self._command.confidence

    @property
    def stale(self) -> bool:
        return self._command.stale

    def as_joint_command(self) -> RobotJointCommand:
        """Return an isolated low-level command for encoder/simulator APIs."""

        return RobotJointCommand(
            self.timestamp_s,
            self.joint_names,
            self.positions_rad.copy(),
            self.confidence,
            self.stale,
        )

    def copy(self) -> "FinalizedSafeCommand":
        return finalize_safe_command(
            self._command,
            free_base_active=self.free_base_active,
            balance_active=self.balance_active,
        )


def finalize_safe_command(
    command: RobotJointCommand,
    *,
    free_base_active: bool,
    balance_active: bool,
) -> FinalizedSafeCommand:
    """Mint provenance after the final balance/support command is selected."""

    if not isinstance(command, RobotJointCommand):
        raise TypeError("command must be a RobotJointCommand")
    return FinalizedSafeCommand(
        command,
        free_base_active,
        balance_active,
        _proof=_FINAL_COMMAND_PROOF,
    )


@dataclass(frozen=True, slots=True)
class OperatorSafetyAcknowledgement:
    """The three confirmations required for one manual arm action."""

    operator_acknowledged: bool
    free_zone_confirmed: bool
    hardware_estop_available: bool

    def __post_init__(self) -> None:
        for name in (
            "operator_acknowledged",
            "free_zone_confirmed",
            "hardware_estop_available",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    @property
    def complete(self) -> bool:
        return (
            self.operator_acknowledged
            and self.free_zone_confirmed
            and self.hardware_estop_available
        )


@dataclass(frozen=True, slots=True)
class RobotOutputStatus:
    """Immutable status suitable for copying into a GUI snapshot."""

    state: RobotConnectionState
    endpoint: str | None
    armed: bool
    command_age_s: float | None
    command_generation: int
    position_attempts: int
    positions_sent: int
    velocity_messages_sent: int
    last_error: str | None
    last_disarm_reason: str | None


class ConnectableCommandTransport(CommandTransport, Protocol):
    def connect(self) -> None: ...

    def close(self) -> None: ...


def encode_legacy_velocities(
    velocity_deg_s: int = DEFAULT_VELOCITY_DEG_S,
) -> str:
    """Encode the optional Unity compatibility velocity message.

    The firmware contract uses a single integer velocity per canonical motor.
    This helper is intentionally separate from positional encoding so callers
    cannot accidentally substitute velocity data for a safe pose.
    """

    if isinstance(velocity_deg_s, bool) or int(velocity_deg_s) != velocity_deg_s:
        raise ValueError("velocity_deg_s must be a positive integer")
    velocity = int(velocity_deg_s)
    if velocity <= 0:
        raise ValueError("velocity_deg_s must be a positive integer")
    return json.dumps(
        {
            "id": LEGACY_REQUEST_ID,
            "method": VELOCITY_METHOD,
            "params": [velocity] * len(JOINT_NAMES),
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )


class SafeRobotController:
    """Fail-closed, manually armed 10 Hz positional output controller.

    ``submit_safe_command`` is the only command ingestion API.  It validates
    canonical order, shape, finiteness and configured limits immediately, but
    performs no network I/O.  ``tick`` is intended to run on the same worker
    that owns the rest of the teleoperation pipeline.
    """

    def __init__(
        self,
        transport: CommandTransport,
        encoder: LegacyWebSocketEncoder | None = None,
        *,
        endpoint: str | None = None,
        rate_hz: float = DEFAULT_RATE_HZ,
        max_command_age_s: float = DEFAULT_MAX_COMMAND_AGE_S,
        send_velocities: bool = False,
        velocity_deg_s: int = DEFAULT_VELOCITY_DEG_S,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        rate_hz = float(rate_hz)
        max_command_age_s = float(max_command_age_s)
        if not math.isfinite(rate_hz) or rate_hz <= 0.0 or rate_hz > DEFAULT_RATE_HZ:
            raise ValueError(f"rate_hz must be finite and within (0, {DEFAULT_RATE_HZ}]")
        if not math.isfinite(max_command_age_s) or max_command_age_s <= 0.0:
            raise ValueError("max_command_age_s must be finite and positive")
        if type(send_velocities) is not bool:
            raise ValueError("send_velocities must be a boolean")

        self.transport = transport
        self.encoder = encoder or LegacyWebSocketEncoder()
        self.endpoint = None if endpoint is None else str(endpoint)
        self.rate_hz = rate_hz
        self.period_s = 1.0 / rate_hz
        self.max_command_age_s = max_command_age_s
        self.send_velocities = send_velocities
        self._velocity_payload = encode_legacy_velocities(velocity_deg_s)
        self._clock = clock
        self._lock = RLock()

        self._state = RobotConnectionState.DISCONNECTED
        self._safe_command: FinalizedSafeCommand | None = None
        self._safe_payload: str | None = None
        self._safe_received_s: float | None = None
        self._safe_free_base = False
        self._safe_balance_enabled = False
        self._command_generation = 0
        self._last_position_attempt_s: float | None = None
        self._position_attempts = 0
        self._positions_sent = 0
        self._velocity_messages_sent = 0
        self._velocity_pending = False
        self._last_error: Exception | None = None
        self._last_disarm_reason: str | None = None

    @staticmethod
    def _finite_time(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    def _now(self, value: float | None) -> float:
        return self._finite_time(self._clock() if value is None else value, "clock")

    @property
    def state(self) -> RobotConnectionState:
        with self._lock:
            return self._state

    @property
    def is_armed(self) -> bool:
        return self.state is RobotConnectionState.ARMED

    @property
    def last_error(self) -> Exception | None:
        with self._lock:
            return self._last_error

    @property
    def position_attempts(self) -> int:
        with self._lock:
            return self._position_attempts

    @property
    def positions_sent(self) -> int:
        with self._lock:
            return self._positions_sent

    def _command_age_locked(self, now_s: float) -> float | None:
        if self._safe_received_s is None:
            return None
        return max(0.0, now_s - self._safe_received_s)

    def status(self, now_s: float | None = None) -> RobotOutputStatus:
        now = self._now(now_s)
        with self._lock:
            error = None if self._last_error is None else str(self._last_error)
            return RobotOutputStatus(
                state=self._state,
                endpoint=self.endpoint,
                armed=self._state is RobotConnectionState.ARMED,
                command_age_s=self._command_age_locked(now),
                command_generation=self._command_generation,
                position_attempts=self._position_attempts,
                positions_sent=self._positions_sent,
                velocity_messages_sent=self._velocity_messages_sent,
                last_error=error,
                last_disarm_reason=self._last_disarm_reason,
            )

    def _clear_command_locked(self) -> None:
        self._safe_command = None
        self._safe_payload = None
        self._safe_received_s = None
        self._safe_free_base = False
        self._safe_balance_enabled = False
        self._velocity_pending = False

    def connect(self) -> bool:
        """Establish the transport without sending any application message."""

        with self._lock:
            if self._state in {
                RobotConnectionState.CONNECTED_DISARMED,
                RobotConnectionState.ARMED,
            }:
                return True
            # A degraded connection must not be silently reused.
            close = getattr(self.transport, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            try:
                connect = getattr(self.transport, "connect", None)
                if callable(connect):
                    connect()
            except Exception as error:
                self._state = RobotConnectionState.DEGRADED
                self._last_error = error
                self._last_disarm_reason = "connect_error"
                self._clear_command_locked()
                return False
            self._state = RobotConnectionState.CONNECTED_DISARMED
            self._last_error = None
            self._last_disarm_reason = None
            self._last_position_attempt_s = None
            return True

    def submit_safe_command(
        self,
        command: FinalizedSafeCommand,
        *,
        free_base: bool | None = None,
        balance_enabled: bool | None = None,
        received_at_s: float | None = None,
    ) -> int:
        """Validate and retain the latest final post-safety command.

        Invalid input is never retained.  If invalid data arrives while armed,
        output becomes degraded immediately and manual reconnect/re-arm is
        required.
        """

        now = self._now(received_at_s)
        try:
            if not isinstance(command, FinalizedSafeCommand):
                raise TypeError(
                    "safe command must have final balance/support provenance"
                )
            if free_base is not None and type(free_base) is not bool:
                raise ValueError("free_base must be a boolean or None")
            if balance_enabled is not None and type(balance_enabled) is not bool:
                raise ValueError("balance_enabled must be a boolean or None")
            if (
                free_base is not None
                and free_base is not command.free_base_active
            ):
                raise ValueError("free_base does not match safe-command provenance")
            if (
                balance_enabled is not None
                and balance_enabled is not command.balance_active
            ):
                raise ValueError("balance_enabled does not match safe-command provenance")
            if command.joint_names != JOINT_NAMES:
                raise ValueError("safe command must use the canonical Unity joint order")
            if command.stale:
                raise ValueError("safe command must not be marked stale")
            low_level = command.as_joint_command()
            # This validates exact shape, finiteness and configured joint limits.
            self.encoder.ordered_positions_rad(low_level)
            payload = self.encoder.encode(low_level)
        except Exception as error:
            with self._lock:
                if self._state is RobotConnectionState.ARMED:
                    self._degrade_locked(error, "invalid_safe_command")
                else:
                    # A rejected replacement must revoke a previously retained
                    # pose as well; otherwise a later arm could authorize stale
                    # data even though the most recent pipeline output failed.
                    self._clear_command_locked()
                    self._last_error = error
                    self._last_disarm_reason = "invalid_safe_command"
            raise

        safe_copy = command.copy()
        with self._lock:
            self._safe_command = safe_copy
            self._safe_payload = payload
            self._safe_received_s = now
            self._safe_free_base = command.free_base_active
            self._safe_balance_enabled = command.balance_active
            self._command_generation += 1
            return self._command_generation

    def _arm_error_locked(self, now_s: float) -> str | None:
        if self._state is not RobotConnectionState.CONNECTED_DISARMED:
            return f"robot state is {self._state.value}, not connected_disarmed"
        if self._safe_command is None or self._safe_payload is None:
            return "no validated safe command is available"
        age = self._command_age_locked(now_s)
        if age is None or age > self.max_command_age_s:
            return "safe command is stale"
        if not self._safe_free_base or not self._safe_balance_enabled:
            return "free-base and balance modes must both be active"
        return None

    def can_arm(self, now_s: float | None = None) -> tuple[bool, str | None]:
        """Return current interlock readiness without changing controller state."""

        now = self._now(now_s)
        with self._lock:
            error = self._arm_error_locked(now)
            return error is None, error

    def arm(
        self,
        acknowledgement: OperatorSafetyAcknowledgement,
        *,
        send_velocities: bool | None = None,
        expected_command_generation: int | None = None,
        now_s: float | None = None,
    ) -> bool:
        """Arm output without sending; the next due ``tick`` performs I/O.

        ``expected_command_generation`` lets the worker bind an operator arm
        decision to the exact safe command used by its authoritative readiness
        check.  This closes the check/use gap if another producer replaces or
        invalidates the retained command before the controller lock is taken.
        """

        if not isinstance(acknowledgement, OperatorSafetyAcknowledgement):
            raise TypeError("acknowledgement must be OperatorSafetyAcknowledgement")
        if send_velocities is not None and type(send_velocities) is not bool:
            raise ValueError("send_velocities must be a boolean or None")
        if expected_command_generation is not None:
            if (
                isinstance(expected_command_generation, bool)
                or int(expected_command_generation) != expected_command_generation
                or int(expected_command_generation) < 0
            ):
                raise ValueError(
                    "expected_command_generation must be a non-negative integer or None"
                )
            expected_command_generation = int(expected_command_generation)
        now = self._now(now_s)
        with self._lock:
            if not acknowledgement.complete:
                self._last_disarm_reason = "operator_confirmation_incomplete"
                return False
            if (
                expected_command_generation is not None
                and expected_command_generation != self._command_generation
            ):
                self._last_disarm_reason = "safe_command_generation_changed"
                return False
            error = self._arm_error_locked(now)
            if error is not None:
                self._last_disarm_reason = error
                return False
            self._state = RobotConnectionState.ARMED
            self._last_error = None
            self._last_disarm_reason = None
            self._last_position_attempt_s = None
            if send_velocities is not None:
                self.send_velocities = send_velocities
            self._velocity_pending = self.send_velocities
            return True

    def _degrade_locked(self, error: Exception, reason: str) -> None:
        self._state = RobotConnectionState.DEGRADED
        self._last_error = error
        self._last_disarm_reason = reason
        self._clear_command_locked()
        close = getattr(self.transport, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def invalidate(self, reason: str) -> None:
        """Disarm after source/reset/pipeline changes and discard the command."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        with self._lock:
            if self._state is RobotConnectionState.ARMED:
                self._state = RobotConnectionState.CONNECTED_DISARMED
            self._last_disarm_reason = reason.strip()
            self._clear_command_locked()
            self._last_position_attempt_s = None

    def disarm(self, reason: str = "operator_disarm") -> None:
        self.invalidate(reason)

    def tick(self, now_s: float | None = None) -> bool:
        """Send at most one due position; all transport failures fail closed."""

        now = self._now(now_s)
        with self._lock:
            if self._state is not RobotConnectionState.ARMED:
                return False
            age = self._command_age_locked(now)
            if age is None or age > self.max_command_age_s:
                self._degrade_locked(RuntimeError("safe command is stale"), "stale_command")
                return False
            if (
                self._last_position_attempt_s is not None
                and now - self._last_position_attempt_s + 1e-12 < self.period_s
            ):
                return False
            payload = self._safe_payload
            if payload is None:
                self._degrade_locked(RuntimeError("safe command is unavailable"), "missing_command")
                return False

            try:
                if self._velocity_pending:
                    self.transport.send(self._velocity_payload)
                    self._velocity_messages_sent += 1
                    self._velocity_pending = False
                self._last_position_attempt_s = now
                self._position_attempts += 1
                self.transport.send(payload)
            except Exception as error:
                self._degrade_locked(error, "network_error")
                return False
            self._positions_sent += 1
            self._last_error = None
            return True

    def disconnect(self, reason: str = "disconnect") -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        with self._lock:
            close = getattr(self.transport, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    self._last_error = error
            self._state = RobotConnectionState.DISCONNECTED
            self._last_disarm_reason = reason.strip()
            self._clear_command_locked()
            self._last_position_attempt_s = None

    def close(self) -> None:
        self.disconnect("controller_closed")

    def __enter__(self) -> "SafeRobotController":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def create_legacy_websocket_controller(
    url: str,
    *,
    timeout_s: float = 0.5,
    rate_hz: float = DEFAULT_RATE_HZ,
    max_command_age_s: float = DEFAULT_MAX_COMMAND_AGE_S,
    send_velocities: bool = False,
    velocity_deg_s: int = DEFAULT_VELOCITY_DEG_S,
    encoder: LegacyWebSocketEncoder | None = None,
    clock: Callable[[], float] = monotonic,
) -> SafeRobotController:
    """Build the production safe controller without opening its WebSocket."""

    from .legacy_websocket import WebSocketTransport

    transport = WebSocketTransport(url, timeout_s=timeout_s)
    return SafeRobotController(
        transport,
        encoder,
        endpoint=url,
        rate_hz=rate_hz,
        max_command_age_s=max_command_age_s,
        send_velocities=send_velocities,
        velocity_deg_s=velocity_deg_s,
        clock=clock,
    )
