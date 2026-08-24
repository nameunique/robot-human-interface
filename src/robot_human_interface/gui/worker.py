"""Command-queued pipeline worker owned by a single ``QThread``."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from math import radians, sin
from queue import Empty, Queue
from threading import Event
from time import monotonic, sleep
from typing import Any, Callable, Mapping

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter

from .resources import SourceItem


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    kind: str
    payload: object | None = None


@dataclass(frozen=True, slots=True)
class GuiPipelineSnapshot:
    """Synthetic fallback snapshot compatible with the core snapshot surface."""

    sequence: int
    timestamp_s: float
    status: str
    source: SourceItem
    frame: QImage
    landmarks: tuple[tuple[float, float, float], ...]
    tracking_quality: float
    angles_rad: tuple[float, ...]
    safe_valid: bool = True
    balance_active: bool = True
    free_base_active: bool = True
    telemetry: Mapping[str, object] = field(default_factory=dict)


class DemoSession:
    """Dependency-free GUI demo used until/when the concrete core is unavailable."""

    def __init__(self, source: SourceItem, *, clock: Callable[[], float] = monotonic) -> None:
        self.source = source
        self.clock = clock
        self.running = False
        self.paused = False
        self.viewer_open = False
        self.sequence = 0
        self._next_frame = 0.0

    def start(self) -> None:
        self.running = True
        self.paused = False
        self._next_frame = self.clock()

    def stop(self) -> None:
        self.running = False
        self.paused = False

    def pause(self) -> None:
        if self.running:
            self.paused = True

    def resume(self) -> None:
        if self.running:
            self.paused = False

    def close(self) -> None:
        self.running = False
        self.paused = False
        self.viewer_open = False

    def reset(self) -> None:
        self.sequence = 0

    def calibrate(self) -> None:
        return None

    def open_viewer(self) -> None:
        self.viewer_open = True

    def close_viewer(self) -> None:
        self.viewer_open = False

    def step(self) -> GuiPipelineSnapshot | None:
        if not self.running or self.paused:
            return None
        now = self.clock()
        if now < self._next_frame:
            return None
        self._next_frame = now + 0.05
        phase = self.sequence * 0.075
        frame = self._make_frame(phase)
        landmarks = self._make_landmarks(phase)
        angles = tuple(
            radians(8.0 * sin(phase + index * 0.31)) for index in range(20)
        )
        snapshot = GuiPipelineSnapshot(
            sequence=self.sequence,
            timestamp_s=now,
            status="RUNNING",
            source=self.source,
            frame=frame,
            landmarks=landmarks,
            tracking_quality=0.93,
            angles_rad=angles,
            telemetry={"pipeline_hz": 20.0, "mode": "synthetic-fallback"},
        )
        self.sequence += 1
        return snapshot

    def _make_frame(self, phase: float) -> QImage:
        image = QImage(960, 540, QImage.Format.Format_RGB32)
        image.fill(QColor("#030608"))
        painter = QPainter(image)
        gradient = QLinearGradient(0, 0, 960, 540)
        gradient.setColorAt(0.0, QColor(9, 27, 39))
        gradient.setColorAt(1.0, QColor(2, 7, 10))
        painter.fillRect(image.rect(), gradient)
        x = int(460 + 170 * sin(phase))
        painter.fillRect(x, 430, 90, 4, QColor("#35C7F2"))
        painter.setPen(QColor("#91A4B7"))
        painter.setFont(QFont("Inter", 12))
        kind = "КАМЕРА" if self.source.kind == "camera" else "ДЕМО-РЕЖИМ"
        painter.drawText(24, 514, f"{kind} · безопасный синтетический источник")
        painter.end()
        return image

    @staticmethod
    def _make_landmarks(phase: float) -> tuple[tuple[float, float, float], ...]:
        sway = 0.015 * sin(phase)
        points = [
            (.50 + sway, .14), (.49 + sway, .13), (.485 + sway, .13), (.48 + sway, .135),
            (.51 + sway, .13), (.515 + sway, .13), (.52 + sway, .135), (.465 + sway, .15),
            (.535 + sway, .15), (.49 + sway, .175), (.51 + sway, .175),
            (.42 + sway, .28), (.58 + sway, .28), (.34 + sway, .42), (.66 + sway, .42),
            (.29 + sway, .55), (.71 + sway, .55), (.275 + sway, .56), (.725 + sway, .56),
            (.28 + sway, .54), (.72 + sway, .54), (.29 + sway, .53), (.71 + sway, .53),
            (.45 + sway, .52), (.55 + sway, .52), (.43 + sway, .69), (.57 + sway, .69),
            (.41 + sway, .87), (.59 + sway, .87), (.40 + sway, .89), (.60 + sway, .89),
            (.43 + sway, .91), (.57 + sway, .91),
        ]
        return tuple((x, y, .93) for x, y in points)


def _enum_value(enum_type: object, preferred: tuple[str, ...], fallback: str) -> object:
    members = getattr(enum_type, "__members__", {})
    for name in preferred:
        if name in members:
            return members[name]
    try:
        return enum_type(fallback)  # type: ignore[operator]
    except (TypeError, ValueError):
        return fallback


def _core_source(module: object, source: SourceItem) -> object:
    source_type = getattr(module, "SourceSpec")
    enum_type = getattr(module, "SourceKind", str)
    if source.kind == "camera":
        kind = _enum_value(enum_type, ("CAMERA", "LIVE_CAMERA"), "camera")
    elif source.kind == "synthetic":
        kind = _enum_value(enum_type, ("SYNTHETIC",), "synthetic")
    elif source.kind == "user":
        kind = _enum_value(enum_type, ("USER_VIDEO", "VIDEO", "REPLAY", "FILE"), "user_video")
    else:
        kind = _enum_value(enum_type, ("STOCK_VIDEO", "VIDEO", "REPLAY", "FILE"), "stock_video")
    return source_type(
        kind=kind,
        source_id=source.source_id,
        display_name=source.title,
        path=source.path,
        camera_index=source.camera_index,
        camera_backend=source.camera_backend,
        width=source.width,
        height=source.height,
        fps=source.fps,
        loop=source.loop,
        mirror=source.mirror,
    )


def build_session(source: SourceItem) -> object:
    """Lazily bind to the extracted core, otherwise return a safe demo."""

    try:
        module = importlib.import_module("robot_human_interface.session")
        source_spec = _core_source(module, source)
        config = module.SessionConfig(
            source=source_spec,
            free_base=True,
            balance_enabled=True,
            snapshot_rate_hz=20.0,
        )
        builder = getattr(module, "create_default_session", None)
        if callable(builder):
            return builder(config)
        pipeline_factory = getattr(module, "default_pipeline_factory", None)
        if callable(pipeline_factory):
            return module.TeleopSession(config, pipeline_factory)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        # Startup must remain useful when optional MediaPipe/MuJoCo resources
        # are absent. The fallback is explicitly surfaced in the event log.
        pass
    return DemoSession(source)


class PipelineWorker(QThread):
    """Own all pipeline/session resources in one long-lived worker thread."""

    snapshot_ready = pyqtSignal(object)
    event_ready = pyqtSignal(object)
    state_changed = pyqtSignal(str)
    viewer_changed = pyqtSignal(bool)
    source_changed = pyqtSignal(object)
    robot_state_changed = pyqtSignal(str, str)
    safety_flags_changed = pyqtSignal(bool, bool, bool)
    video_metadata_ready = pyqtSignal(object)
    clean_stopped = pyqtSignal()

    def __init__(
        self,
        parent=None,
        *,
        session_factory: Callable[[SourceItem], object] = build_session,
        robot_endpoint: str = "ws://leonardo.local:1233",
        video_cache_dir: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._factory = session_factory
        self._commands: Queue[WorkerCommand] = Queue()
        self._shutdown = Event()
        self._session: object | None = None
        self._source: SourceItem | None = None
        self._running = False
        self._viewer_open = False
        self._robot_endpoint = robot_endpoint
        self._robot: object | None = None
        self._latest_snapshot: object | None = None
        self._last_snapshot_key: tuple[int, str] | None = None
        self._safety_warning_emitted = False
        self._video_cache_dir = video_cache_dir
        self._video_probe: object | None = None

    def submit(self, kind: str, payload: object | None = None) -> None:
        self._commands.put(WorkerCommand(kind, payload))

    def select_source(self, source: SourceItem) -> None:
        self.submit("source", source)

    def start_pipeline(self) -> None:
        self.submit("start")

    def stop_pipeline(self) -> None:
        self.submit("stop")

    def reset_pipeline(self) -> None:
        self.submit("reset")

    def calibrate(self) -> None:
        self.submit("calibrate")

    def pause_pipeline(self) -> None:
        self.submit("pause")

    def resume_pipeline(self) -> None:
        self.submit("resume")

    def set_viewer_open(self, enabled: bool) -> None:
        self.submit("viewer", bool(enabled))

    def connect_robot(self) -> None:
        self.submit("robot_connect")

    def arm_robot(self, *, send_velocities: bool = False) -> None:
        self.submit("robot_arm", {"send_velocities": bool(send_velocities)})

    def disarm_robot(self, reason: str = "operator_disarm") -> None:
        self.submit("robot_disarm", reason)

    def disconnect_robot(self) -> None:
        self.submit("robot_disconnect")

    def probe_video(self, source: SourceItem) -> None:
        if source.path:
            self.submit("video_probe", source)

    def request_shutdown(self) -> None:
        self._shutdown.set()
        self.submit("shutdown")

    def shutdown_and_wait(self, timeout_ms: int = 4000) -> bool:
        self.request_shutdown()
        return self.wait(timeout_ms)

    def run(self) -> None:
        self.state_changed.emit("STOPPED")
        try:
            while not self._shutdown.is_set():
                self._drain_commands()
                if self._shutdown.is_set():
                    break
                if self._running and self._session is not None:
                    try:
                        snapshot = self._invoke_step(self._session)
                        if snapshot is not None:
                            self._latest_snapshot = snapshot
                            sequence = int(getattr(snapshot, "sequence", -1))
                            raw_status = getattr(snapshot, "status", "running")
                            status = str(getattr(raw_status, "value", raw_status)).lower()
                            key = (sequence, status)
                            if key != self._last_snapshot_key:
                                self._last_snapshot_key = key
                                flags = self._resolve_safety_flags(snapshot)
                                self.safety_flags_changed.emit(
                                    flags is not None,
                                    False if flags is None else flags[0],
                                    False if flags is None else flags[1],
                                )
                                self._update_robot_command(snapshot)
                                self.snapshot_ready.emit(snapshot)
                            else:
                                self._tick_robot_only()
                            if status in {"ended", "stopped", "closed", "error"}:
                                self._running = False
                                self._invalidate_robot(f"pipeline_{status}")
                                self.state_changed.emit("DEGRADED" if status == "error" else "STOPPED")
                        self._drain_core_events()
                    except Exception as error:  # worker boundary must survive plugin failures
                        self._emit("ERROR", "PIPELINE", "PIPELINE_STEP_FAILED", "Ошибка обработки кадра", str(error))
                        self._running = False
                        self.state_changed.emit("DEGRADED")
                        self._invalidate_robot("pipeline_error")
                        self._close_session()
                sleep(0.005)
        finally:
            self._close_session()
            self._close_robot()
            self.clean_stopped.emit()

    def _drain_commands(self) -> None:
        while True:
            if self._shutdown.is_set():
                return
            try:
                command = self._commands.get_nowait()
            except Empty:
                return
            if command.kind == "shutdown":
                self._shutdown.set()
                return
            try:
                self._handle(command)
            except Exception as error:
                self._emit("ERROR", "PIPELINE", "COMMAND_FAILED", f"Команда {command.kind} не выполнена", str(error))
                self._running = False
                self.state_changed.emit("DEGRADED")
                self._invalidate_robot("command_error")
                self._close_session()

    def _handle(self, command: WorkerCommand) -> None:
        if command.kind == "source":
            if not isinstance(command.payload, SourceItem):
                raise TypeError("source command requires SourceItem")
            restart = self._running
            self._running = False
            self._close_session()
            self._source = command.payload
            self._last_snapshot_key = None
            self._safety_warning_emitted = False
            self.safety_flags_changed.emit(False, False, False)
            self._invalidate_robot("source_changed")
            self.source_changed.emit(self._source)
            self._emit("INFO", "SOURCE", "SOURCE_SELECTED", "Источник выбран", self._source.title)
            if restart:
                self._start_current()
            return
        if command.kind == "start":
            self._start_current()
            return
        if command.kind == "stop":
            self._running = False
            self._call(self._session, ("stop", "request_stop"))
            self._invalidate_robot("pipeline_stopped")
            self.state_changed.emit("STOPPED")
            self._viewer_open = False
            self.viewer_changed.emit(False)
            self._emit("INFO", "PIPELINE", "SESSION_STOPPED", "Сессия остановлена")
            return
        if command.kind == "reset":
            self._call(self._session, ("reset", "request_reset"))
            self._invalidate_robot("pipeline_reset")
            self._emit("INFO", "PIPELINE", "SESSION_RESET", "Состояние сброшено")
            return
        if command.kind == "calibrate":
            self._call(self._session, ("request_calibrate", "calibrate"))
            self._invalidate_robot("calibration_started")
            self._emit("INFO", "PIPELINE", "CALIBRATION_STARTED", "Калибровка запущена")
            return
        if command.kind == "pause":
            self._call(self._session, ("pause", "request_pause"), required=True)
            self._invalidate_robot("pipeline_paused")
            self.state_changed.emit("PAUSED")
            self._emit("INFO", "PIPELINE", "SESSION_PAUSED", "Сессия приостановлена")
            return
        if command.kind == "resume":
            self._call(self._session, ("resume", "request_resume"), required=True)
            self.state_changed.emit("RUNNING")
            self._emit("INFO", "PIPELINE", "SESSION_RESUMED", "Сессия продолжена")
            return
        if command.kind == "viewer":
            enabled = bool(command.payload)
            if self._session is None or not self._running:
                raise RuntimeError("сначала запустите pipeline")
            names = ("open_viewer", "request_open_viewer") if enabled else ("close_viewer", "request_close_viewer")
            self._call(self._session, names)
            self._viewer_open = enabled
            self.viewer_changed.emit(enabled)
            self._emit("INFO", "MUJOCO", "VIEWER_OPENED" if enabled else "VIEWER_CLOSED", "Окно MuJoCo открыто" if enabled else "Окно MuJoCo закрыто")
            return
        if command.kind == "video_probe":
            if not isinstance(command.payload, SourceItem) or not command.payload.path:
                raise TypeError("video_probe requires a file SourceItem")
            if self._video_probe is None:
                from .video_probe import VideoProbeCache

                self._video_probe = VideoProbeCache(self._video_cache_dir)
            metadata = self._video_probe.probe(
                command.payload.source_id,
                command.payload.path,
            )
            self.video_metadata_ready.emit(metadata)
            return
        if command.kind == "robot_connect":
            self._connect_robot()
            return
        if command.kind == "robot_arm":
            payload = command.payload if isinstance(command.payload, dict) else {}
            self._arm_robot(bool(payload.get("send_velocities", False)))
            return
        if command.kind == "robot_disarm":
            self._invalidate_robot(str(command.payload or "operator_disarm"))
            return
        if command.kind == "robot_disconnect":
            self._disconnect_robot()

    def _start_current(self) -> None:
        if self._source is None:
            raise RuntimeError("источник не выбран")
        if self._session is None:
            self._session = self._factory(self._source)
            if isinstance(self._session, DemoSession):
                self._emit("WARNING", "PIPELINE", "SYNTHETIC_FALLBACK", "Используется безопасный демонстрационный pipeline", "Concrete TeleopSession недоступен")
        self._call(self._session, ("start", "request_start"), required=True)
        self._running = True
        self.state_changed.emit("RUNNING")
        self._emit("INFO", "PIPELINE", "SESSION_STARTED", "Сессия запущена")

    @staticmethod
    def _call(target: object | None, names: tuple[str, ...], *, required: bool = False) -> object | None:
        if target is not None:
            for name in names:
                method = getattr(target, name, None)
                if callable(method):
                    return method()
        if required:
            raise RuntimeError(f"session has none of {names}")
        return None

    @staticmethod
    def _invoke_step(session: object) -> object | None:
        process = getattr(session, "process_commands", None)
        if callable(process):
            process()
        step = getattr(session, "step", None)
        if not callable(step):
            raise RuntimeError("session has no step()")
        return step()

    def _drain_core_events(self) -> None:
        if self._session is None:
            return
        drain = getattr(self._session, "drain_events", None)
        if not callable(drain):
            return
        for event in drain() or ():
            self.event_ready.emit(event)

    def _close_session(self) -> None:
        session, self._session = self._session, None
        self._viewer_open = False
        if session is not None:
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    self._emit("WARNING", "PIPELINE", "SESSION_CLOSE_FAILED", "Не удалось полностью закрыть сессию", str(error))
        self.viewer_changed.emit(False)

    def _connect_robot(self) -> None:
        if self._robot is None:
            from robot_human_interface.protocol import create_legacy_websocket_controller

            self._robot = create_legacy_websocket_controller(self._robot_endpoint)
        connected = bool(self._robot.connect())
        status = self._robot.status()
        state = getattr(status.state, "value", str(status.state)).upper()
        details = "" if connected else str(status.last_error or "connection failed")
        self.robot_state_changed.emit(state, details)
        self._emit(
            "INFO" if connected else "ERROR",
            "ROBOT",
            "ROBOT_CONNECTED_DISARMED" if connected else "ROBOT_CONNECT_FAILED",
            "Соединение установлено; команды не отправляются" if connected else "Соединение с роботом не установлено",
            details,
        )

    def _arm_robot(self, send_velocities: bool) -> None:
        if self._robot is None:
            raise RuntimeError("robot controller is not connected")
        from robot_human_interface.protocol import OperatorSafetyAcknowledgement

        # This option is intentionally only changed on the worker before one
        # explicit arm action. It remains disabled by default.
        self._robot.send_velocities = bool(send_velocities)
        acknowledgement = OperatorSafetyAcknowledgement(True, True, True)
        armed = bool(self._robot.arm(acknowledgement))
        status = self._robot.status()
        state = getattr(status.state, "value", str(status.state)).upper()
        details = "" if armed else str(status.last_disarm_reason or "interlock rejected arm")
        self.robot_state_changed.emit(state, details)
        self._emit(
            "WARNING" if armed else "ERROR",
            "ROBOT",
            "ROBOT_ARMED" if armed else "ROBOT_ARM_BLOCKED",
            "Отправка safe_command разрешена оператором" if armed else "Контроллер отклонил включение отправки",
            details,
        )

    def _update_robot_command(self, snapshot: object) -> None:
        if self._robot is None:
            return
        safe_command = getattr(snapshot, "safe_command", None)
        if safe_command is not None:
            flags = self._resolve_safety_flags(snapshot)
            if flags is None:
                self._invalidate_robot("safety_provenance_missing")
                if not self._safety_warning_emitted:
                    self._safety_warning_emitted = True
                    self._emit(
                        "ERROR",
                        "ROBOT",
                        "SAFETY_PROVENANCE_MISSING",
                        "Отправка заблокирована: режимы free-base/balance не подтверждены",
                    )
                return
            free_base, balance_enabled = flags
            try:
                self._robot.submit_safe_command(
                    safe_command,
                    free_base=free_base,
                    balance_enabled=balance_enabled,
                )
            except Exception as error:
                status = self._robot.status()
                state = getattr(status.state, "value", str(status.state)).upper()
                self.robot_state_changed.emit(state, str(error))
                self._emit("ERROR", "ROBOT", "SAFE_COMMAND_REJECTED", "Safe-команда не прошла выходной контроллер", str(error))
                return
        self._robot.tick()
        status = self._robot.status()
        state = getattr(status.state, "value", str(status.state)).upper()
        if state == "DEGRADED":
            self.robot_state_changed.emit(state, str(status.last_error or status.last_disarm_reason or ""))

    def _resolve_safety_flags(self, snapshot: object) -> tuple[bool, bool] | None:
        direct_free = getattr(snapshot, "free_base_active", None)
        direct_balance = getattr(snapshot, "balance_active", None)
        if type(direct_free) is bool and type(direct_balance) is bool:
            return direct_free, direct_balance
        telemetry = getattr(snapshot, "telemetry", None)
        if isinstance(telemetry, Mapping):
            free_value = telemetry.get("free_base_active", telemetry.get("free_base"))
            balance_value = telemetry.get("balance_active", telemetry.get("balance_enabled"))
            if type(free_value) is bool and type(balance_value) is bool:
                return free_value, balance_value
        config = getattr(self._session, "config", None)
        free_value = getattr(config, "free_base", None)
        balance_value = getattr(config, "balance_enabled", None)
        if type(free_value) is bool and type(balance_value) is bool:
            return free_value, balance_value
        return None

    def _tick_robot_only(self) -> None:
        if self._robot is None:
            return
        self._robot.tick()
        status = self._robot.status()
        state = getattr(status.state, "value", str(status.state)).upper()
        if state == "DEGRADED":
            self.robot_state_changed.emit(state, str(status.last_error or status.last_disarm_reason or ""))

    def _invalidate_robot(self, reason: str) -> None:
        if self._robot is None:
            return
        self._robot.invalidate(reason)
        status = self._robot.status()
        state = getattr(status.state, "value", str(status.state)).upper()
        self.robot_state_changed.emit(state, str(status.last_disarm_reason or ""))

    def _disconnect_robot(self) -> None:
        if self._robot is None:
            self.robot_state_changed.emit("DISCONNECTED", "")
            return
        self._robot.disconnect("operator_disconnect")
        self.robot_state_changed.emit("DISCONNECTED", "")
        self._emit("INFO", "ROBOT", "ROBOT_DISCONNECTED", "Соединение закрыто")

    def _close_robot(self) -> None:
        controller, self._robot = self._robot, None
        if controller is not None:
            try:
                controller.close()
            except Exception as error:
                self._emit("WARNING", "ROBOT", "ROBOT_CLOSE_FAILED", "Ошибка закрытия контроллера робота", str(error))
        self.robot_state_changed.emit("DISCONNECTED", "")

    def _emit(
        self,
        severity: str,
        subsystem: str,
        event_code: str,
        message: str,
        details: str = "",
    ) -> None:
        self.event_ready.emit(
            {
                "severity": severity,
                "subsystem": subsystem,
                "event_code": event_code,
                "message": message,
                "details": details,
            }
        )
