"""Worker-agnostic lifecycle for interactive humanoid teleoperation.

``TeleopSession`` contains no Qt dependency.  A GUI thread enqueues immutable
``SessionCommand`` objects, while one worker thread calls :meth:`step`.  The
injected pipeline remains the sole owner of camera/video, MediaPipe,
retargeting, controllers and MuJoCo resources.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from threading import RLock
from time import monotonic
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from robot_human_interface.protocol import (
    FinalizedSafeCommand,
    OperatorSafetyAcknowledgement,
    RobotOutputStatus,
    SafeRobotController,
)
from robot_human_interface.skeleton import CameraFrame, RobotJointCommand, SkeletonFrame


class SourceKind(str, Enum):
    STOCK_VIDEO = "stock_video"
    USER_VIDEO = "user_video"
    CAMERA = "camera"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Portable description of a video, camera or deterministic test source."""

    kind: SourceKind
    source_id: str
    display_name: str
    path: Path | None = None
    camera_index: int = 0
    camera_backend: str = "auto"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    loop: bool = False
    mirror: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = SourceKind(self.kind)
        source_id = str(self.source_id).strip()
        display_name = str(self.display_name).strip()
        if not source_id:
            raise ValueError("source_id must be non-empty")
        if not display_name:
            raise ValueError("display_name must be non-empty")
        if isinstance(self.camera_index, bool) or int(self.camera_index) != self.camera_index:
            raise ValueError("camera_index must be a non-negative integer")
        if int(self.camera_index) < 0:
            raise ValueError("camera_index must be a non-negative integer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("source width and height must be positive")
        fps = float(self.fps)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("source fps must be finite and positive")
        if type(self.loop) is not bool or type(self.mirror) is not bool:
            raise ValueError("loop and mirror must be booleans")
        path = None if self.path is None else Path(self.path).expanduser().resolve()
        if kind in {SourceKind.STOCK_VIDEO, SourceKind.USER_VIDEO} and path is None:
            raise ValueError("video sources require an absolute path")
        if kind in {SourceKind.CAMERA, SourceKind.SYNTHETIC} and path is not None:
            raise ValueError("camera and synthetic sources do not accept a path")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "camera_index", int(self.camera_index))
        object.__setattr__(self, "camera_backend", str(self.camera_backend).strip() or "auto")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SessionConfig:
    source: SourceSpec
    free_base: bool = True
    balance_enabled: bool = True
    retargeting: str = "ik"
    snapshot_rate_hz: float = 20.0
    calibration_samples: int = 30
    physics_steps_per_frame: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceSpec):
            raise TypeError("source must be a SourceSpec")
        if type(self.free_base) is not bool or type(self.balance_enabled) is not bool:
            raise ValueError("free_base and balance_enabled must be booleans")
        mode = str(self.retargeting).strip().lower()
        if mode not in {"ik", "geometric"}:
            raise ValueError("retargeting must be 'ik' or 'geometric'")
        rate = float(self.snapshot_rate_hz)
        if not math.isfinite(rate) or rate <= 0.0 or rate > 20.0:
            raise ValueError("snapshot_rate_hz must be finite and within (0, 20]")
        if (
            isinstance(self.calibration_samples, bool)
            or int(self.calibration_samples) != self.calibration_samples
            or int(self.calibration_samples) <= 0
        ):
            raise ValueError("calibration_samples must be a positive integer")
        if (
            isinstance(self.physics_steps_per_frame, bool)
            or int(self.physics_steps_per_frame) != self.physics_steps_per_frame
            or int(self.physics_steps_per_frame) < 0
        ):
            raise ValueError("physics_steps_per_frame must be a non-negative integer")
        object.__setattr__(self, "retargeting", mode)
        object.__setattr__(self, "snapshot_rate_hz", rate)
        object.__setattr__(self, "calibration_samples", int(self.calibration_samples))
        object.__setattr__(self, "physics_steps_per_frame", int(self.physics_steps_per_frame))


class SessionStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ENDED = "ended"
    ERROR = "error"
    CLOSED = "closed"


def _freeze_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        copied = value.copy()
        copied.setflags(write=False)
        return copied
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {str(key): _freeze_value(item) for key, item in dict(value).items()}
    )


def _copy_frame(frame: CameraFrame | None) -> CameraFrame | None:
    if frame is None:
        return None
    image = np.ascontiguousarray(frame.image_bgr.copy())
    image.setflags(write=False)
    return CameraFrame(
        image,
        frame.timestamp_s,
        frame.sequence,
        frame.mirrored,
    )


def _copy_skeleton(frame: SkeletonFrame | None) -> SkeletonFrame | None:
    if frame is None:
        return None
    return SkeletonFrame(
        frame.timestamp_s,
        frame.landmarks_2d.copy(),
        frame.landmarks_3d.copy(),
        frame.visibility.copy(),
        frame.presence.copy(),
        frame.image_size,
        frame.sequence,
    )


def _copy_command(command: RobotJointCommand | None) -> RobotJointCommand | None:
    if command is None:
        return None
    return RobotJointCommand(
        command.timestamp_s,
        command.joint_names,
        command.positions_rad.copy(),
        command.confidence,
        command.stale,
    )


def _copy_safe_command(
    command: FinalizedSafeCommand | None,
) -> FinalizedSafeCommand | None:
    if command is None:
        return None
    if not isinstance(command, FinalizedSafeCommand):
        raise TypeError("safe_command must have final balance/support provenance")
    return command.copy()


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    """Deep-copied immutable data crossing from the worker to the UI."""

    sequence: int
    timestamp_s: float
    status: SessionStatus
    source: SourceSpec
    frame: CameraFrame | None = None
    skeleton: SkeletonFrame | None = None
    raw_command: RobotJointCommand | None = None
    safe_command: FinalizedSafeCommand | None = None
    tracking_quality: float = 0.0
    telemetry: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or int(self.sequence) != self.sequence:
            raise ValueError("snapshot sequence must be a non-negative integer")
        if int(self.sequence) < 0:
            raise ValueError("snapshot sequence must be a non-negative integer")
        timestamp = float(self.timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("snapshot timestamp_s must be finite and non-negative")
        quality = float(self.tracking_quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("tracking_quality must be finite and within [0, 1]")
        if not isinstance(self.source, SourceSpec):
            raise TypeError("snapshot source must be a SourceSpec")
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "status", SessionStatus(self.status))
        object.__setattr__(self, "tracking_quality", quality)
        object.__setattr__(self, "frame", _copy_frame(self.frame))
        object.__setattr__(self, "skeleton", _copy_skeleton(self.skeleton))
        object.__setattr__(self, "raw_command", _copy_command(self.raw_command))
        object.__setattr__(self, "safe_command", _copy_safe_command(self.safe_command))
        object.__setattr__(self, "telemetry", _freeze_mapping(self.telemetry))


class SessionEventLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SessionEvent:
    """Structured event with stable English code and localized UI message."""

    timestamp_s: float
    level: SessionEventLevel
    subsystem: str
    code: str
    message_ru: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("event timestamp_s must be finite and non-negative")
        subsystem = str(self.subsystem).strip()
        code = str(self.code).strip()
        message = str(self.message_ru).strip()
        if not subsystem or not code or not message:
            raise ValueError("event subsystem, code and message_ru must be non-empty")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "level", SessionEventLevel(self.level))
        object.__setattr__(self, "subsystem", subsystem)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message_ru", message)
        object.__setattr__(self, "details", _freeze_mapping(self.details))


class SessionCommandKind(str, Enum):
    START = "start"
    STOP = "stop"
    RESET = "reset"
    CALIBRATE = "calibrate"
    CHANGE_SOURCE = "change_source"
    PAUSE = "pause"
    RESUME = "resume"
    OPEN_VIEWER = "open_viewer"
    CLOSE_VIEWER = "close_viewer"
    CONNECT_ROBOT = "connect_robot"
    ARM_ROBOT = "arm_robot"
    DISARM_ROBOT = "disarm_robot"
    DISCONNECT_ROBOT = "disconnect_robot"


@dataclass(frozen=True, slots=True)
class SessionCommand:
    kind: SessionCommandKind
    requested_at_s: float
    source: SourceSpec | None = None
    calibration_samples: int | None = None
    acknowledgement: OperatorSafetyAcknowledgement | None = None
    send_velocities: bool | None = None

    def __post_init__(self) -> None:
        timestamp = float(self.requested_at_s)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("requested_at_s must be finite and non-negative")
        kind = SessionCommandKind(self.kind)
        if kind is SessionCommandKind.CHANGE_SOURCE and not isinstance(
            self.source, SourceSpec
        ):
            raise ValueError("change_source command requires a SourceSpec")
        if self.calibration_samples is not None and (
            isinstance(self.calibration_samples, bool)
            or int(self.calibration_samples) != self.calibration_samples
            or int(self.calibration_samples) <= 0
        ):
            raise ValueError("calibration_samples must be a positive integer")
        if kind is SessionCommandKind.ARM_ROBOT and not isinstance(
            self.acknowledgement, OperatorSafetyAcknowledgement
        ):
            raise ValueError("arm_robot command requires a safety acknowledgement")
        if self.send_velocities is not None and type(self.send_velocities) is not bool:
            raise ValueError("send_velocities must be a boolean or None")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "requested_at_s", timestamp)
        if self.calibration_samples is not None:
            object.__setattr__(self, "calibration_samples", int(self.calibration_samples))


@runtime_checkable
class SessionPipeline(Protocol):
    """The resource-owning implementation that runs on one worker thread."""

    def step(self) -> PipelineSnapshot | None: ...

    def reset(self) -> None: ...

    def calibrate(self, sample_count: int) -> None: ...

    def close(self) -> None: ...


PipelineFactory = Callable[[SessionConfig], SessionPipeline]


class TeleopSession:
    """Serialize lifecycle commands and guarantee deterministic cleanup."""

    def __init__(
        self,
        config: SessionConfig,
        pipeline_factory: PipelineFactory,
        *,
        robot_output: SafeRobotController | None = None,
        clock: Callable[[], float] = monotonic,
        event_capacity: int = 5000,
    ) -> None:
        if not isinstance(config, SessionConfig):
            raise TypeError("config must be a SessionConfig")
        if not callable(pipeline_factory):
            raise TypeError("pipeline_factory must be callable")
        if isinstance(event_capacity, bool) or int(event_capacity) <= 0:
            raise ValueError("event_capacity must be positive")
        self._config = config
        self._pipeline_factory = pipeline_factory
        self._robot_output = robot_output
        self._clock = clock
        self._commands: Queue[SessionCommand] = Queue()
        self._events: deque[SessionEvent] = deque(maxlen=int(event_capacity))
        self._lock = RLock()
        self._pipeline: SessionPipeline | None = None
        self._status = SessionStatus.STOPPED
        self._sequence = 0
        now = self._now()
        self._latest_snapshot = PipelineSnapshot(
            0,
            now,
            SessionStatus.STOPPED,
            config.source,
        )

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("session clock must return a finite non-negative value")
        return value

    @property
    def config(self) -> SessionConfig:
        with self._lock:
            return self._config

    @property
    def status(self) -> SessionStatus:
        with self._lock:
            return self._status

    @property
    def latest_snapshot(self) -> PipelineSnapshot:
        with self._lock:
            # Reconstruction makes array ownership explicit for every consumer.
            return replace(self._latest_snapshot)

    @property
    def snapshot(self) -> PipelineSnapshot:
        """Short QThread-friendly alias for :attr:`latest_snapshot`."""

        return self.latest_snapshot

    def _emit(
        self,
        level: SessionEventLevel,
        subsystem: str,
        code: str,
        message_ru: str,
        **details: object,
    ) -> None:
        event = SessionEvent(
            self._now(),
            level,
            subsystem,
            code,
            message_ru,
            details,
        )
        with self._lock:
            self._events.append(event)

    def drain_events(self) -> tuple[SessionEvent, ...]:
        with self._lock:
            result = tuple(self._events)
            self._events.clear()
            return result

    def submit_command(self, command: SessionCommand) -> None:
        if not isinstance(command, SessionCommand):
            raise TypeError("command must be a SessionCommand")
        with self._lock:
            if self._status is SessionStatus.CLOSED:
                raise RuntimeError("session is closed")
        self._commands.put(command)

    def enqueue(self, command: SessionCommand) -> None:
        """Alias used by adapters that expose a generic command queue."""

        self.submit_command(command)

    def _request(self, kind: SessionCommandKind, **kwargs: object) -> None:
        self.submit_command(SessionCommand(kind, self._now(), **kwargs))

    def request_start(self) -> None:
        self._request(SessionCommandKind.START)

    def request_stop(self) -> None:
        self._request(SessionCommandKind.STOP)

    def request_reset(self) -> None:
        self._request(SessionCommandKind.RESET)

    def request_calibrate(self, sample_count: int | None = None) -> None:
        self._request(
            SessionCommandKind.CALIBRATE,
            calibration_samples=(
                self.config.calibration_samples if sample_count is None else sample_count
            ),
        )

    def request_change_source(self, source: SourceSpec) -> None:
        self._request(SessionCommandKind.CHANGE_SOURCE, source=source)

    def request_pause(self) -> None:
        self._request(SessionCommandKind.PAUSE)

    def request_resume(self) -> None:
        self._request(SessionCommandKind.RESUME)

    def request_open_viewer(self) -> None:
        self._request(SessionCommandKind.OPEN_VIEWER)

    def request_close_viewer(self) -> None:
        self._request(SessionCommandKind.CLOSE_VIEWER)

    def request_connect_robot(self) -> None:
        self._request(SessionCommandKind.CONNECT_ROBOT)

    def request_arm_robot(
        self,
        acknowledgement: OperatorSafetyAcknowledgement,
        *,
        send_velocities: bool | None = None,
    ) -> None:
        self._request(
            SessionCommandKind.ARM_ROBOT,
            acknowledgement=acknowledgement,
            send_velocities=send_velocities,
        )

    def request_disarm_robot(self) -> None:
        self._request(SessionCommandKind.DISARM_ROBOT)

    def request_disconnect_robot(self) -> None:
        self._request(SessionCommandKind.DISCONNECT_ROBOT)

    def _close_pipeline(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is None:
            return
        try:
            pipeline.close()
        except Exception as error:
            self._emit(
                SessionEventLevel.WARNING,
                "pipeline",
                "PIPELINE_CLOSE_FAILED",
                "Не удалось полностью закрыть ресурсы конвейера",
                error=str(error),
            )

    def _set_terminal_snapshot(self, status: SessionStatus) -> None:
        self._sequence += 1
        self._latest_snapshot = PipelineSnapshot(
            self._sequence,
            self._now(),
            status,
            self._config.source,
            telemetry=self._latest_snapshot.telemetry,
        )

    def start(self) -> None:
        with self._lock:
            if self._status is SessionStatus.CLOSED:
                raise RuntimeError("session is closed")
            if self._status in {SessionStatus.RUNNING, SessionStatus.PAUSED}:
                return
        try:
            pipeline = self._pipeline_factory(self.config)
            if not isinstance(pipeline, SessionPipeline):
                raise TypeError("pipeline_factory returned an incompatible pipeline")
            start = getattr(pipeline, "start", None)
            if callable(start):
                start()
        except Exception as error:
            close = getattr(locals().get("pipeline"), "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            with self._lock:
                self._status = SessionStatus.ERROR
                self._set_terminal_snapshot(SessionStatus.ERROR)
            self._emit(
                SessionEventLevel.ERROR,
                "pipeline",
                "PIPELINE_START_FAILED",
                "Конвейер не запущен",
                error=str(error),
            )
            raise
        with self._lock:
            self._pipeline = pipeline
            self._status = SessionStatus.RUNNING
            self._set_terminal_snapshot(SessionStatus.RUNNING)
        self._emit(
            SessionEventLevel.INFO,
            "pipeline",
            "PIPELINE_STARTED",
            "Конвейер запущен",
            source_id=self.config.source.source_id,
        )

    def stop(self, reason: str = "pipeline_stopped") -> None:
        if self._robot_output is not None:
            self._robot_output.invalidate(reason)
        self._close_pipeline()
        with self._lock:
            if self._status is SessionStatus.CLOSED:
                return
            self._status = SessionStatus.STOPPED
            self._set_terminal_snapshot(SessionStatus.STOPPED)
        self._emit(
            SessionEventLevel.INFO,
            "pipeline",
            "PIPELINE_STOPPED",
            "Конвейер остановлен",
            reason=reason,
        )

    def reset(self) -> None:
        if self._robot_output is not None:
            self._robot_output.invalidate("pipeline_reset")
        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("pipeline is not running")
        pipeline.reset()
        self._emit(
            SessionEventLevel.INFO,
            "pipeline",
            "PIPELINE_RESET",
            "Состояние конвейера сброшено",
        )

    def calibrate(self, sample_count: int | None = None) -> None:
        if self._robot_output is not None:
            self._robot_output.invalidate("calibration_started")
        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("pipeline is not running")
        count = self.config.calibration_samples if sample_count is None else int(sample_count)
        if count <= 0:
            raise ValueError("sample_count must be positive")
        pipeline.calibrate(count)
        self._emit(
            SessionEventLevel.INFO,
            "retargeting",
            "CALIBRATION_STARTED",
            "Калибровка запущена",
            sample_count=count,
        )

    def change_source(self, source: SourceSpec) -> None:
        if not isinstance(source, SourceSpec):
            raise TypeError("source must be a SourceSpec")
        was_active = self.status in {SessionStatus.RUNNING, SessionStatus.PAUSED}
        if self._robot_output is not None:
            self._robot_output.invalidate("source_changed")
        self._close_pipeline()
        with self._lock:
            old_source = self._config.source
            self._config = replace(self._config, source=source)
            self._status = SessionStatus.STOPPED
            self._set_terminal_snapshot(SessionStatus.STOPPED)
        self._emit(
            SessionEventLevel.INFO,
            "source",
            "SOURCE_CHANGED",
            "Источник изменён",
            old_source_id=old_source.source_id,
            source_id=source.source_id,
        )
        if was_active:
            self.start()

    def pause(self) -> None:
        with self._lock:
            if self._status is not SessionStatus.RUNNING:
                return
            self._status = SessionStatus.PAUSED
            self._set_terminal_snapshot(SessionStatus.PAUSED)
        if self._robot_output is not None:
            self._robot_output.invalidate("pipeline_paused")

    def resume(self) -> None:
        with self._lock:
            if self._status is not SessionStatus.PAUSED:
                return
            self._status = SessionStatus.RUNNING
            self._set_terminal_snapshot(SessionStatus.RUNNING)

    def _viewer_call(self, method_name: str, code: str, message: str) -> None:
        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("pipeline is not running")
        method = getattr(pipeline, method_name, None)
        if not callable(method):
            raise RuntimeError(f"pipeline does not support {method_name}")
        method()
        self._emit(SessionEventLevel.INFO, "mujoco", code, message)

    def open_viewer(self) -> None:
        self._viewer_call("open_viewer", "MUJOCO_VIEWER_OPENED", "Окно MuJoCo открыто")

    def close_viewer(self) -> None:
        self._viewer_call("close_viewer", "MUJOCO_VIEWER_CLOSED", "Окно MuJoCo закрыто")

    def _handle_command(self, command: SessionCommand) -> None:
        kind = command.kind
        if kind is SessionCommandKind.START:
            self.start()
        elif kind is SessionCommandKind.STOP:
            self.stop()
        elif kind is SessionCommandKind.RESET:
            self.reset()
        elif kind is SessionCommandKind.CALIBRATE:
            self.calibrate(command.calibration_samples)
        elif kind is SessionCommandKind.CHANGE_SOURCE:
            assert command.source is not None
            self.change_source(command.source)
        elif kind is SessionCommandKind.PAUSE:
            self.pause()
        elif kind is SessionCommandKind.RESUME:
            self.resume()
        elif kind is SessionCommandKind.OPEN_VIEWER:
            self.open_viewer()
        elif kind is SessionCommandKind.CLOSE_VIEWER:
            self.close_viewer()
        elif kind is SessionCommandKind.CONNECT_ROBOT:
            if self._robot_output is None:
                raise RuntimeError("robot output is not configured")
            connected = self._robot_output.connect()
            self._emit(
                SessionEventLevel.INFO if connected else SessionEventLevel.ERROR,
                "robot",
                "ROBOT_CONNECTED" if connected else "ROBOT_CONNECT_FAILED",
                "Соединение с роботом установлено"
                if connected
                else "Не удалось подключиться к роботу",
                error=(
                    None
                    if self._robot_output.last_error is None
                    else str(self._robot_output.last_error)
                ),
            )
        elif kind is SessionCommandKind.ARM_ROBOT:
            if self._robot_output is None:
                raise RuntimeError("robot output is not configured")
            assert command.acknowledgement is not None
            armed = self._robot_output.arm(
                command.acknowledgement,
                send_velocities=command.send_velocities,
            )
            status = self._robot_output.status(self._now())
            self._emit(
                SessionEventLevel.INFO if armed else SessionEventLevel.WARNING,
                "robot",
                "ROBOT_ARMED" if armed else "ROBOT_ARM_REJECTED",
                "Отправка на реального робота разрешена"
                if armed
                else "Включение отправки на робота отклонено",
                reason=status.last_disarm_reason,
            )
        elif kind is SessionCommandKind.DISARM_ROBOT:
            if self._robot_output is not None:
                self._robot_output.disarm()
                self._emit(
                    SessionEventLevel.INFO,
                    "robot",
                    "ROBOT_DISARMED",
                    "Отправка на реального робота выключена",
                )
        elif kind is SessionCommandKind.DISCONNECT_ROBOT:
            if self._robot_output is not None:
                self._robot_output.disconnect()
                self._emit(
                    SessionEventLevel.INFO,
                    "robot",
                    "ROBOT_DISCONNECTED",
                    "Соединение с роботом закрыто",
                )

    def process_commands(self, *, limit: int | None = None) -> int:
        """Process queued commands on the resource-owning worker thread."""

        if limit is not None and (
            isinstance(limit, bool) or int(limit) != limit or int(limit) <= 0
        ):
            raise ValueError("limit must be a positive integer or None")
        handled = 0
        while limit is None or handled < int(limit):
            try:
                command = self._commands.get_nowait()
            except Empty:
                break
            try:
                self._handle_command(command)
            except Exception as error:
                self._emit(
                    SessionEventLevel.ERROR,
                    "session",
                    "SESSION_COMMAND_FAILED",
                    "Команда сессии завершилась ошибкой",
                    command=command.kind.value,
                    error=str(error),
                )
            finally:
                self._commands.task_done()
            handled += 1
        return handled

    def step(self) -> PipelineSnapshot:
        """Run one worker iteration and return an isolated immutable snapshot."""

        self.process_commands()
        with self._lock:
            status = self._status
            pipeline = self._pipeline
        if status is not SessionStatus.RUNNING or pipeline is None:
            return self.latest_snapshot

        try:
            produced = pipeline.step()
            if produced is not None and not isinstance(produced, PipelineSnapshot):
                raise TypeError(
                    "pipeline.step() must return PipelineSnapshot or None"
                )
        except Exception as error:
            if self._robot_output is not None:
                self._robot_output.invalidate("pipeline_error")
            self._close_pipeline()
            with self._lock:
                self._status = SessionStatus.ERROR
                self._set_terminal_snapshot(SessionStatus.ERROR)
            self._emit(
                SessionEventLevel.ERROR,
                "pipeline",
                "PIPELINE_STEP_FAILED",
                "Ошибка обработки кадра",
                error=str(error),
            )
            return self.latest_snapshot

        if produced is None:
            if self._robot_output is not None:
                self._robot_output.invalidate("source_ended")
            self._close_pipeline()
            with self._lock:
                self._status = SessionStatus.ENDED
                self._set_terminal_snapshot(SessionStatus.ENDED)
            self._emit(
                SessionEventLevel.INFO,
                "source",
                "SOURCE_ENDED",
                "Источник завершён",
            )
            return self.latest_snapshot

        assert isinstance(produced, PipelineSnapshot)

        telemetry = dict(produced.telemetry)
        if self._robot_output is not None:
            if produced.safe_command is not None:
                try:
                    self._robot_output.submit_safe_command(
                        produced.safe_command,
                        free_base=self.config.free_base,
                        balance_enabled=self.config.balance_enabled,
                        received_at_s=self._now(),
                    )
                except Exception as error:
                    self._emit(
                        SessionEventLevel.ERROR,
                        "robot",
                        "ROBOT_SAFE_COMMAND_REJECTED",
                        "Безопасная команда робота отклонена",
                        error=str(error),
                    )
            self._robot_output.tick(self._now())
            robot_status: RobotOutputStatus = self._robot_output.status(self._now())
            telemetry["robot_output"] = robot_status

        with self._lock:
            self._sequence += 1
            self._latest_snapshot = PipelineSnapshot(
                self._sequence,
                produced.timestamp_s,
                SessionStatus.RUNNING,
                self._config.source,
                produced.frame,
                produced.skeleton,
                produced.raw_command,
                produced.safe_command,
                produced.tracking_quality,
                telemetry,
            )
        return self.latest_snapshot

    def run_once(self) -> PipelineSnapshot:
        """Explicit worker-loop alias for :meth:`step`."""

        return self.step()

    def close(self) -> None:
        with self._lock:
            if self._status is SessionStatus.CLOSED:
                return
        if self._robot_output is not None:
            self._robot_output.close()
        self._close_pipeline()
        with self._lock:
            self._status = SessionStatus.CLOSED
            self._set_terminal_snapshot(SessionStatus.CLOSED)
        self._emit(
            SessionEventLevel.INFO,
            "session",
            "SESSION_CLOSED",
            "Сессия закрыта",
        )

    def __enter__(self) -> "TeleopSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def default_pipeline_factory(config: SessionConfig) -> SessionPipeline:
    """Lazy concrete factory, safe to import before ``QApplication`` exists."""

    from robot_human_interface.pipeline import DefaultTeleopPipeline

    return DefaultTeleopPipeline(config)


def create_default_session(
    config: SessionConfig,
    *,
    robot_output: SafeRobotController | None = None,
    clock: Callable[[], float] = monotonic,
) -> TeleopSession:
    """Create the production session without opening hardware yet."""

    return TeleopSession(
        config,
        default_pipeline_factory,
        robot_output=robot_output,
        clock=clock,
    )


__all__ = [
    "PipelineFactory",
    "PipelineSnapshot",
    "SessionCommand",
    "SessionCommandKind",
    "SessionConfig",
    "SessionEvent",
    "SessionEventLevel",
    "SessionPipeline",
    "SessionStatus",
    "SourceKind",
    "SourceSpec",
    "TeleopSession",
    "create_default_session",
    "default_pipeline_factory",
]
