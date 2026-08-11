"""Safe adapter for the Unity-era motor WebSocket protocol.

The rest of the application uses named joint targets in radians.  This module
is the only boundary that converts those targets to the positional JSON used
by the existing robot server::

    {"id":0,"method":"setPositions","params":[... 20 degrees ...]}

``LatestCommandPublisher.submit`` performs no network I/O.  A balance or
camera loop can therefore publish targets without ever being interrupted by a
socket disconnect.  Network work belongs in ``tick`` or in the optional
background worker started with ``start``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import Lock, Thread, current_thread
from time import monotonic
from types import TracebackType
from typing import Any, Protocol
from urllib.parse import urlsplit

import numpy as np

from robot_human_interface.retargeting import load_joint_specs
from robot_human_interface.skeleton import JOINT_NAMES, RobotJointCommand


LEGACY_REQUEST_ID = 0
LEGACY_METHOD = "setPositions"
DEFAULT_RATE_HZ = 10.0
DEFAULT_DEGREE_PRECISION = 9


class CommandTransport(Protocol):
    """Minimal text transport consumed by :class:`LatestCommandPublisher`."""

    def send(self, payload: str) -> None:
        """Send one complete UTF-8 JSON text message or raise an exception."""


class _JointLimitSpec(Protocol):
    name: str
    lower_rad: float
    upper_rad: float


class _WebSocketConnection(Protocol):
    def send(self, payload: str) -> Any: ...

    def close(self) -> Any: ...


ConnectionFactory = Callable[[str, float], _WebSocketConnection]


def _default_joint_config() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "joints.yaml"


def _limits_from_specs(
    specs: Sequence[_JointLimitSpec],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    records = tuple(specs)
    names = tuple(str(spec.name) for spec in records)
    if names != JOINT_NAMES:
        raise ValueError("joint specs must use the canonical JOINT_NAMES order")
    lower = tuple(float(spec.lower_rad) for spec in records)
    upper = tuple(float(spec.upper_rad) for spec in records)
    return _validate_limits(lower, upper)


def _limits_from_mapping(
    limits_rad: Mapping[str, Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    names = {str(name) for name in limits_rad}
    expected = set(JOINT_NAMES)
    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        raise ValueError(f"joint limits do not match schema; missing={missing}, extra={extra}")

    lower: list[float] = []
    upper: list[float] = []
    for name in JOINT_NAMES:
        pair = limits_rad[name]
        if isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError(f"joint {name!r} must have exactly two radian limits")
        lower.append(float(pair[0]))
        upper.append(float(pair[1]))
    return _validate_limits(tuple(lower), tuple(upper))


def _validate_limits(
    lower: tuple[float, ...],
    upper: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(lower) != len(JOINT_NAMES) or len(upper) != len(JOINT_NAMES):
        raise ValueError(f"joint limits must contain {len(JOINT_NAMES)} entries")
    for name, minimum, maximum in zip(JOINT_NAMES, lower, upper, strict=True):
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError(f"limits for joint {name!r} must be finite")
        if minimum >= maximum:
            raise ValueError(f"lower limit must be below upper limit for joint {name!r}")
    return lower, upper


class LegacyWebSocketEncoder:
    """Validate and encode one internal radian command for the legacy server.

    Limits can come from ``config/joints.yaml`` (the default), a sequence of
    JointSpec-like objects, or an explicit complete name-to-pair mapping.  The
    incoming command may use any order, but it must contain every canonical
    joint exactly once.  The emitted ``params`` list is always canonical.
    """

    def __init__(
        self,
        joint_specs: Sequence[_JointLimitSpec] | None = None,
        *,
        joint_config_path: str | Path | None = None,
        limits_rad: Mapping[str, Sequence[float]] | None = None,
        degree_precision: int = DEFAULT_DEGREE_PRECISION,
    ) -> None:
        selected_sources = sum(
            source is not None for source in (joint_specs, joint_config_path, limits_rad)
        )
        if selected_sources > 1:
            raise ValueError(
                "choose only one limit source: joint_specs, joint_config_path, or limits_rad"
            )
        if isinstance(degree_precision, bool) or not 0 <= int(degree_precision) <= 15:
            raise ValueError("degree_precision must be an integer within [0, 15]")
        if int(degree_precision) != degree_precision:
            raise ValueError("degree_precision must be an integer within [0, 15]")
        self.degree_precision = int(degree_precision)

        if limits_rad is not None:
            lower, upper = _limits_from_mapping(limits_rad)
        else:
            if joint_config_path is not None and not Path(joint_config_path).is_file():
                raise FileNotFoundError(f"Joint configuration not found: {joint_config_path}")
            config_path = joint_config_path
            if joint_specs is None and config_path is None:
                default_path = _default_joint_config()
                config_path = default_path if default_path.is_file() else None
            specs = tuple(
                joint_specs if joint_specs is not None else load_joint_specs(config_path)
            )
            lower, upper = _limits_from_specs(specs)
        self._lower_rad = np.asarray(lower, dtype=np.float64)
        self._upper_rad = np.asarray(upper, dtype=np.float64)
        self._lower_rad.setflags(write=False)
        self._upper_rad.setflags(write=False)

    @property
    def lower_limits_rad(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self._lower_rad.copy()

    @property
    def upper_limits_rad(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self._upper_rad.copy()

    def ordered_positions_rad(
        self,
        command: RobotJointCommand,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return a validated copy of ``command`` in canonical motor order."""

        if not isinstance(command, RobotJointCommand):
            raise TypeError("command must be a RobotJointCommand")
        names = tuple(str(name) for name in command.joint_names)
        positions = np.asarray(command.positions_rad, dtype=np.float64)
        if positions.shape != (len(names),):
            raise ValueError("positions_rad must have one value per joint name")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")

        expected = set(JOINT_NAMES)
        actual = set(names)
        if actual != expected or len(names) != len(JOINT_NAMES):
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"joint names do not match schema; missing={missing}, extra={extra}")
        if not np.isfinite(positions).all():
            raise ValueError("positions_rad must contain only finite values")

        by_name = dict(zip(names, positions, strict=True))
        ordered = np.asarray([by_name[name] for name in JOINT_NAMES], dtype=np.float64)
        outside = (ordered < self._lower_rad) | (ordered > self._upper_rad)
        if np.any(outside):
            index = int(np.flatnonzero(outside)[0])
            name = JOINT_NAMES[index]
            raise ValueError(
                f"joint {name!r} target {ordered[index]!r} rad is outside "
                f"[{self._lower_rad[index]!r}, {self._upper_rad[index]!r}]"
            )
        return ordered

    def encode(self, command: RobotJointCommand) -> str:
        """Return compact deterministic JSON, converting radians only here."""

        ordered = self.ordered_positions_rad(command)
        params: list[float] = []
        for radians in ordered:
            degrees = round(math.degrees(float(radians)), self.degree_precision)
            # JSON distinguishes -0.0 textually even though the values compare
            # equal.  Canonicalize it so byte-for-byte payload tests are stable.
            params.append(0.0 if degrees == 0.0 else degrees)
        document = {
            "id": LEGACY_REQUEST_ID,
            "method": LEGACY_METHOD,
            "params": params,
        }
        return json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )


def encode_legacy_command(
    command: RobotJointCommand,
    *,
    joint_specs: Sequence[_JointLimitSpec] | None = None,
    joint_config_path: str | Path | None = None,
    limits_rad: Mapping[str, Sequence[float]] | None = None,
) -> str:
    """One-shot convenience wrapper around :class:`LegacyWebSocketEncoder`."""

    return LegacyWebSocketEncoder(
        joint_specs,
        joint_config_path=joint_config_path,
        limits_rad=limits_rad,
    ).encode(command)


class WebSocketTransport:
    """Lazy reconnecting synchronous WebSocket text transport.

    The optional ``websocket-client`` package is imported only when the first
    connection is made.  Put this transport behind ``LatestCommandPublisher``
    so connection latency and failures remain off the balance-control thread.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float = 1.0,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        parts = urlsplit(str(url))
        if parts.scheme not in {"ws", "wss"} or not parts.netloc:
            raise ValueError("url must be an absolute ws:// or wss:// URL")
        if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        self.url = str(url)
        self.timeout_s = float(timeout_s)
        self._connection_factory = connection_factory
        self._connection: _WebSocketConnection | None = None
        self._lock = Lock()
        self._received_count = 0
        self._last_receive_error: Exception | None = None

    def _default_connection_factory(self, url: str, timeout_s: float) -> _WebSocketConnection:
        try:
            import websocket  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "real WebSocket transport requires websocket-client: "
                "python -m pip install websocket-client"
            ) from error
        return websocket.create_connection(
            url,
            timeout=timeout_s,
            enable_multithread=True,
        )

    def _connect_locked(self) -> _WebSocketConnection:
        if self._connection is None:
            factory = self._connection_factory or self._default_connection_factory
            self._connection = factory(self.url, self.timeout_s)
        return self._connection

    def send(self, payload: str) -> None:
        if not isinstance(payload, str):
            raise TypeError("payload must be a text string")
        with self._lock:
            connection = self._connect_locked()
            try:
                connection.send(payload)
            except Exception:
                self._close_locked()
                raise
            self._drain_incoming_locked(connection)

    @staticmethod
    def _is_nonblocking_timeout(error: Exception) -> bool:
        return isinstance(error, (TimeoutError, BlockingIOError)) or (
            error.__class__.__name__ == "WebSocketTimeoutException"
        )

    def _drain_incoming_locked(self, connection: _WebSocketConnection) -> None:
        """Non-blockingly drain replies/control frames after each 10 Hz send.

        The Unity client continuously calls ``Receive``.  Draining here avoids
        accumulating firmware acknowledgements while keeping every socket call
        on the publisher thread, never on the motor-control loop.
        """

        receive = getattr(connection, "recv", None)
        set_timeout = getattr(connection, "settimeout", None)
        if not callable(receive) or not callable(set_timeout):
            return
        try:
            set_timeout(0.0)
            for _ in range(64):
                try:
                    message = receive()
                except Exception as error:
                    if self._is_nonblocking_timeout(error):
                        break
                    self._last_receive_error = error
                    self._close_locked()
                    return
                self._received_count += 1
                self._last_receive_error = None
                if message is None or message == "":
                    break
        finally:
            if self._connection is connection:
                try:
                    set_timeout(self.timeout_s)
                except Exception as error:
                    self._last_receive_error = error
                    self._close_locked()

    def _close_locked(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                # Closing is best-effort; the original send error is more useful.
                pass

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connection is not None

    @property
    def received_count(self) -> int:
        with self._lock:
            return self._received_count

    @property
    def last_receive_error(self) -> Exception | None:
        with self._lock:
            return self._last_receive_error


class LatestCommandPublisher:
    """A 10 Hz latest-only, failure-isolated command publisher.

    ``submit`` validates/encodes and atomically replaces the queued payload,
    without touching the network.  ``tick`` sends at most one payload per rate
    interval and catches all transport errors.  ``start`` runs ``tick`` on a
    daemon worker for integration with a real balance loop.
    """

    def __init__(
        self,
        transport: CommandTransport,
        encoder: LegacyWebSocketEncoder | None = None,
        *,
        rate_hz: float = DEFAULT_RATE_HZ,
        repeat_latest: bool = False,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        rate_hz = float(rate_hz)
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be finite and positive")
        self.transport = transport
        self.encoder = encoder or LegacyWebSocketEncoder()
        self.rate_hz = rate_hz
        self.period_s = 1.0 / rate_hz
        self.repeat_latest = bool(repeat_latest)
        self._clock = clock
        self._lock = Lock()
        self._latest_payload: str | None = None
        self._latest_generation = 0
        self._sent_generation = 0
        self._last_attempt_s: float | None = None
        self._last_error: Exception | None = None
        self._attempt_count = 0
        self._sent_count = 0
        self._stop_requested = False
        self._worker: Thread | None = None

    def submit(self, command: RobotJointCommand) -> int:
        """Replace the queued command and return its monotonically increasing id."""

        payload = self.encoder.encode(command)
        with self._lock:
            self._latest_payload = payload
            self._latest_generation += 1
            return self._latest_generation

    def tick(self, now_s: float | None = None) -> bool:
        """Attempt one due send; return success and never raise transport errors."""

        now = float(self._clock() if now_s is None else now_s)
        if not math.isfinite(now):
            raise ValueError("publisher clock must return a finite value")
        with self._lock:
            if self._latest_payload is None:
                return False
            if (
                not self.repeat_latest
                and self._latest_generation <= self._sent_generation
            ):
                return False
            if (
                self._last_attempt_s is not None
                and now - self._last_attempt_s + 1e-12 < self.period_s
            ):
                return False
            payload = self._latest_payload
            generation = self._latest_generation
            self._last_attempt_s = now
            self._attempt_count += 1

        try:
            self.transport.send(payload)
        except Exception as error:
            with self._lock:
                self._last_error = error
            return False

        with self._lock:
            # A newer submit may occur while the old payload is in flight.  In
            # that case only mark the captured generation as delivered.
            self._sent_generation = max(self._sent_generation, generation)
            self._sent_count += 1
            self._last_error = None
        return True

    def publish(self, command: RobotJointCommand, now_s: float | None = None) -> bool:
        """Submit then synchronously tick; useful for simple/manual integrations."""

        self.submit(command)
        return self.tick(now_s)

    @property
    def has_pending_command(self) -> bool:
        with self._lock:
            return self._latest_generation > self._sent_generation

    @property
    def last_error(self) -> Exception | None:
        with self._lock:
            return self._last_error

    @property
    def attempt_count(self) -> int:
        with self._lock:
            return self._attempt_count

    @property
    def sent_count(self) -> int:
        with self._lock:
            return self._sent_count

    def _run(self) -> None:
        from time import sleep

        while True:
            with self._lock:
                if self._stop_requested:
                    return
            self.tick()
            sleep(self.period_s)

    def start(self) -> None:
        """Start a daemon network worker; calling this twice is harmless."""

        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop_requested = False
            self._worker = Thread(
                target=self._run,
                name="legacy-websocket-publisher",
                daemon=True,
            )
            worker = self._worker
        worker.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Stop the background worker and best-effort close the transport."""

        if not math.isfinite(float(timeout_s)) or float(timeout_s) < 0.0:
            raise ValueError("timeout_s must be finite and non-negative")
        with self._lock:
            self._stop_requested = True
            worker = self._worker
        if worker is not None and worker is not current_thread():
            worker.join(float(timeout_s))
        with self._lock:
            if self._worker is worker and (worker is None or not worker.is_alive()):
                self._worker = None

        close = getattr(self.transport, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:
                with self._lock:
                    self._last_error = error

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "LatestCommandPublisher":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


# Descriptive aliases retained for callers that prefer explicit protocol/rate names.
LegacyCommandEncoder = LegacyWebSocketEncoder
RateLimitedCommandPublisher = LatestCommandPublisher
LegacyWebSocketPublisher = LatestCommandPublisher
WebSocketClientTransport = WebSocketTransport
