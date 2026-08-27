"""Command-queued pipeline worker owned by a single ``QThread``."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from inspect import Parameter, signature
from math import isfinite, radians, sin
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic, sleep
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter

from .resources import SourceItem
from .runtime import (
    ReadinessReason,
    RobotReadiness,
    RobotUiState,
    RuntimeMode,
    RuntimeStatus,
)


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    kind: str
    payload: object | None = None


@dataclass(frozen=True, slots=True)
class _ArmRequest:
    """One-use envelope proving that the GUI supplied operator acknowledgements."""

    acknowledgement: object | None
    send_velocities: bool
    token: str = field(default_factory=lambda: uuid4().hex)


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
    safe_valid: bool = False
    balance_active: bool = True
    free_base_active: bool = True
    telemetry: Mapping[str, object] = field(default_factory=dict)
    safe_command: object | None = None


@dataclass(frozen=True, slots=True)
class _SnapshotSafetyFacts:
    valid: bool
    reason_code: ReadinessReason
    reason: str
    command: object | None = None
    free_base_active: bool | None = None
    balance_active: bool | None = None


class DemoSession:
    """Dependency-free GUI demo used until/when the concrete core is unavailable."""

    def __init__(
        self,
        source: SourceItem,
        *,
        clock: Callable[[], float] = monotonic,
        fallback_reason: str = "явно выбран демонстрационный backend",
    ) -> None:
        self.source = source
        self.clock = clock
        self.runtime_mode = RuntimeMode.DEMO
        self.fallback_reason = str(fallback_reason).strip() or "причина DEMO fallback не указана"
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

    fallback_reason = "сборщик production-сессии недоступен"
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
            session = builder(config)
            if session is None:
                raise RuntimeError("create_default_session returned None")
            return session
        pipeline_factory = getattr(module, "default_pipeline_factory", None)
        if callable(pipeline_factory):
            return module.TeleopSession(config, pipeline_factory)
        fallback_reason = "модуль сессии не содержит поддерживаемого production-сборщика"
    except Exception as error:
        # Startup must remain useful when optional MediaPipe/MuJoCo resources
        # are absent.  The exact fallback is surfaced to the UI and also makes
        # physical output impossible until a production session is created.
        message = str(error).strip()
        fallback_reason = type(error).__name__ + (f": {message}" if message else "")
    return DemoSession(source, fallback_reason=fallback_reason)


class PipelineWorker(QThread):
    """Own all pipeline/session resources in one long-lived worker thread."""

    snapshot_ready = pyqtSignal(object)
    event_ready = pyqtSignal(object)
    state_changed = pyqtSignal(str)
    viewer_changed = pyqtSignal(bool)
    viewer_status_changed = pyqtSignal(str, str)
    source_changed = pyqtSignal(object)
    robot_state_changed = pyqtSignal(str, str)
    robot_status_changed = pyqtSignal(object)
    runtime_status_changed = pyqtSignal(object)
    robot_readiness_changed = pyqtSignal(object)
    safety_flags_changed = pyqtSignal(bool, bool, bool)
    video_metadata_ready = pyqtSignal(object)
    recorder_state_changed = pyqtSignal(str, object)
    recorder_progress = pyqtSignal(object)
    experiment_completed = pyqtSignal(object)
    clean_stopped = pyqtSignal()

    def __init__(
        self,
        parent=None,
        *,
        session_factory: Callable[[SourceItem], object] = build_session,
        robot_endpoint: str = "ws://leonardo.local:1233",
        robot_factory: Callable[[str], object] | None = None,
        video_cache_dir: str | None = None,
        experiment_root: str | Path | None = None,
        camera_video_sink_factory: Callable[..., object] | None = None,
        max_snapshot_age_s: float = 0.5,
        state_log_interval_s: float = 2.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        super().__init__(parent)
        max_snapshot_age_s = float(max_snapshot_age_s)
        state_log_interval_s = float(state_log_interval_s)
        if not isfinite(max_snapshot_age_s) or max_snapshot_age_s <= 0.0:
            raise ValueError("max_snapshot_age_s must be positive")
        if not isfinite(state_log_interval_s) or state_log_interval_s <= 0.0:
            raise ValueError("state_log_interval_s must be positive")
        self._factory = session_factory
        self._robot_factory = robot_factory
        self._clock = clock
        self._max_snapshot_age_s = max_snapshot_age_s
        self._state_log_interval_s = state_log_interval_s
        self._commands: Queue[WorkerCommand] = Queue()
        self._shutdown = Event()
        self._session: object | None = None
        self._source: SourceItem | None = None
        self._running = False
        self._viewer_open = False
        self._viewer_state = "UNAVAILABLE"
        self._viewer_details = "backend ещё не запущен"
        self._robot_endpoint = robot_endpoint
        self._robot: object | None = None
        self._latest_snapshot: object | None = None
        self._latest_snapshot_received_s: float | None = None
        self._last_snapshot_key: tuple[int, str, str] | None = None
        self._safety_warning_emitted = False
        self._runtime_status = RuntimeStatus.demo(
            "production-сессия ещё не создана"
        )
        self._pipeline_state = "STOPPED"
        self._last_robot_ui_state = RobotUiState.DISCONNECTED
        self._last_robot_state_details = ""
        self._last_robot_status: object | None = None
        self._consumed_arm_tokens: set[str] = set()
        self._last_readiness: RobotReadiness | None = None
        self._last_readiness_key: tuple[object, ...] | None = None
        self._state_log_times: dict[str, float] = {}
        self._video_cache_dir = video_cache_dir
        self._video_probe: object | None = None
        self._experiment_root = (
            None if experiment_root is None else Path(experiment_root).expanduser().resolve()
        )
        self._camera_video_sink_factory = camera_video_sink_factory
        self._recorder: object | None = None
        self._recording_spec: object | None = None
        self._active_run_id: str | None = None
        self._recording_phase = "IDLE"
        self._recording_task: Thread | None = None
        self._recording_cancel = Event()
        self._recording_cancel_reason = "application_closed"
        self._last_recorder_progress_s = 0.0
        self._camera_video_sink: object | None = None
        self._camera_video_unavailable = False
        self._camera_video_last_dropped = 0
        self._camera_video_last_error = ""

    @property
    def runtime_status(self) -> RuntimeStatus:
        return self._runtime_status

    @property
    def robot_readiness(self) -> RobotReadiness | None:
        return self._last_readiness

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

    def arm_robot(
        self,
        *,
        acknowledgement: object | None = None,
        send_velocities: bool = False,
    ) -> None:
        self.submit(
            "robot_arm",
            _ArmRequest(acknowledgement, bool(send_velocities)),
        )

    def disarm_robot(self, reason: str = "operator_disarm") -> None:
        self.submit("robot_disarm", reason)

    def disconnect_robot(self) -> None:
        self.submit("robot_disconnect")

    def probe_video(self, source: SourceItem) -> None:
        if source.path:
            self.submit("video_probe", source)

    def seek(self, position_s: float) -> None:
        self.submit("playback_seek", float(position_s))

    def step_frame(self, delta_frames: int) -> None:
        self.submit("playback_step", int(delta_frames))

    def set_playback_rate(self, rate: float) -> None:
        self.submit("playback_rate", float(rate))

    def set_playback_loop(
        self,
        enabled: bool,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> None:
        self.submit(
            "playback_loop",
            (bool(enabled), float(start_s), None if end_s is None else float(end_s)),
        )

    def start_recording(self, spec: object) -> None:
        self.submit("recording_start", spec)

    def stop_recording(self, reason: str = "manual") -> None:
        self.submit("recording_stop", str(reason))

    def request_shutdown(self) -> None:
        self._shutdown.set()
        self.submit("shutdown")

    def shutdown_and_wait(self, timeout_ms: int = 4000) -> bool:
        self.request_shutdown()
        return self.wait(timeout_ms)

    def run(self) -> None:
        self._initialize_recorder()
        self.runtime_status_changed.emit(self._runtime_status)
        self._set_pipeline_state("STOPPED")
        self._publish_viewer_status("UNAVAILABLE", "backend ещё не запущен", force=True)
        self._publish_robot_state(RobotUiState.DISCONNECTED, force=True)
        self._publish_readiness(force=True)
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
                            self._latest_snapshot_received_s = self._clock()
                            sequence = int(getattr(snapshot, "sequence", -1))
                            raw_status = getattr(snapshot, "status", "running")
                            status = str(getattr(raw_status, "value", raw_status)).lower()
                            playback = getattr(snapshot, "playback", None)
                            raw_discontinuity = getattr(
                                playback, "discontinuity_reason", None
                            )
                            discontinuity = str(
                                getattr(raw_discontinuity, "value", raw_discontinuity)
                            ).lower() if raw_discontinuity is not None else ""
                            terminal = status in {"ended", "stopped", "closed", "error"}
                            rejects_output = terminal or bool(discontinuity)
                            key = (sequence, status, discontinuity)
                            if key != self._last_snapshot_key:
                                self._last_snapshot_key = key
                                if rejects_output:
                                    reason = (
                                        f"pipeline_{status}"
                                        if terminal
                                        else f"playback_{discontinuity}"
                                    )
                                    # A terminal or time-discontinuous frame is
                                    # preview-only. Never submit/tick its command.
                                    self._invalidate_robot(reason)
                                    self.safety_flags_changed.emit(False, False, False)
                                    if discontinuity:
                                        self._stop_recording(reason)
                                        self._emit_rate_limited(
                                            f"playback-discontinuity:{discontinuity}",
                                            "WARNING",
                                            "PLAYBACK",
                                            "PLAYBACK_DISCONTINUITY",
                                            "Временной скачок прекратил отправку на робота",
                                            discontinuity,
                                        )
                                else:
                                    flags = self._resolve_safety_flags(snapshot)
                                    self.safety_flags_changed.emit(
                                        flags is not None,
                                        False if flags is None else flags[0],
                                        False if flags is None else flags[1],
                                    )
                                    self._update_robot_command(snapshot)
                                    self._append_recording_snapshot(snapshot)
                                self.snapshot_ready.emit(snapshot)
                            elif not rejects_output:
                                self._tick_robot_only()
                            if terminal and not (
                                status == "ended" and self._pipeline_state == "ENDED"
                            ):
                                self._stop_recording(
                                    "eof" if status == "ended" else f"pipeline_{status}"
                                )
                                self._invalidate_robot(f"pipeline_{status}")
                                seekable_eof = bool(
                                    status == "ended"
                                    and getattr(playback, "seekable", False)
                                )
                                if seekable_eof:
                                    # Keep the decoder/session alive.  The
                                    # worker continues processing queued seek
                                    # and step commands while the UI truthfully
                                    # displays ENDED and no safe authority.
                                    self._running = True
                                    self.safety_flags_changed.emit(False, False, False)
                                    self._set_pipeline_state("ENDED")
                                else:
                                    self._running = False
                                    self._clear_snapshot_authority()
                                    self._set_pipeline_state(
                                        "DEGRADED" if status == "error" else status.upper()
                                    )
                                    self._close_session()
                        else:
                            # A paused/stalled source still needs the physical
                            # output watchdog to observe command age.
                            self._tick_robot_only()
                        self._drain_core_events()
                        self._publish_readiness()
                    except Exception as error:  # worker boundary must survive plugin failures
                        self._emit("ERROR", "PIPELINE", "PIPELINE_STEP_FAILED", "Ошибка обработки кадра", str(error))
                        self._stop_recording("pipeline_error")
                        self._running = False
                        self._invalidate_robot("pipeline_error")
                        self._clear_snapshot_authority()
                        self._set_pipeline_state("DEGRADED")
                        self._close_session()
                self._poll_recorder_state()
                self._poll_camera_video_sink()
                sleep(0.005)
        finally:
            # Physical output is the first shutdown responsibility. Recorder
            # helpers may legitimately need tens of seconds to flush a large
            # package and must never delay robot invalidation/disconnect.
            self._invalidate_robot("application_closed")
            self._close_robot()
            self._close_session()
            self._shutdown_recording()
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
                self._invalidate_robot("command_error")
                self._clear_snapshot_authority()
                self._set_pipeline_state("DEGRADED")
                self._close_session()

    def _handle(self, command: WorkerCommand) -> None:
        if command.kind == "source":
            if not isinstance(command.payload, SourceItem):
                raise TypeError("source command requires SourceItem")
            restart = self._running
            if self._recording_active():
                self._emit(
                    "WARNING",
                    "RECORDER",
                    "SOURCE_CHANGE_BLOCKED_RECORDING",
                    "Смена источника заблокирована до завершения записи опыта",
                )
                return
            self._running = False
            self._close_session()
            self._source = command.payload
            self._clear_snapshot_authority()
            self._set_runtime_status(
                RuntimeStatus.demo("для выбранного источника production-сессия ещё не запущена")
            )
            self._safety_warning_emitted = False
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
            self._stop_recording("pipeline_stopped")
            self._running = False
            self._call(self._session, ("stop", "request_stop"))
            self._invalidate_robot("pipeline_stopped")
            self._clear_snapshot_authority()
            self._set_pipeline_state("STOPPED")
            self._publish_viewer_status("UNAVAILABLE", "pipeline остановлен")
            self._emit("INFO", "PIPELINE", "SESSION_STOPPED", "Сессия остановлена")
            return
        if command.kind == "reset":
            if self._recording_active():
                self._emit("WARNING", "RECORDER", "RESET_BLOCKED_RECORDING", "Сброс заблокирован во время записи опыта")
                return
            self._invalidate_robot("pipeline_reset")
            self._clear_snapshot_authority()
            self._call(self._session, ("reset", "request_reset"))
            self._emit("INFO", "PIPELINE", "SESSION_RESET", "Состояние сброшено")
            return
        if command.kind == "calibrate":
            if self._recording_active():
                self._emit("WARNING", "RECORDER", "CALIBRATION_BLOCKED_RECORDING", "Калибровка заблокирована во время записи опыта")
                return
            self._invalidate_robot("calibration_started")
            self._clear_snapshot_authority()
            self._call(self._session, ("request_calibrate", "calibrate"))
            self._emit("INFO", "PIPELINE", "CALIBRATION_STARTED", "Калибровка запущена")
            return
        if command.kind == "pause":
            self._stop_recording("pipeline_paused")
            self._call(self._session, ("pause", "request_pause"), required=True)
            self._invalidate_robot("pipeline_paused")
            self._clear_snapshot_authority()
            self._set_pipeline_state("PAUSED")
            self._emit("INFO", "PIPELINE", "SESSION_PAUSED", "Сессия приостановлена")
            return
        if command.kind == "resume":
            self._call(self._session, ("resume", "request_resume"), required=True)
            self._set_pipeline_state("RUNNING")
            self._emit("INFO", "PIPELINE", "SESSION_RESUMED", "Сессия продолжена")
            return
        if command.kind == "viewer":
            enabled = bool(command.payload)
            if self._session is None or not self._running:
                self._publish_viewer_status(
                    "UNAVAILABLE", "сначала запустите production pipeline"
                )
                self._emit(
                    "WARNING",
                    "MUJOCO",
                    "VIEWER_NOT_RUNNING",
                    "Сначала запустите pipeline",
                )
                return
            if self._runtime_status.mode is not RuntimeMode.PRODUCTION:
                self._publish_viewer_status(
                    "UNAVAILABLE", "MuJoCo недоступен в DEMO backend"
                )
                self._emit(
                    "WARNING",
                    "MUJOCO",
                    "VIEWER_UNAVAILABLE_DEMO",
                    "MuJoCo viewer недоступен в демонстрационном режиме",
                )
                return
            names = (
                ("request_open_viewer", "open_viewer")
                if enabled
                else ("request_close_viewer", "close_viewer")
            )
            selected_name = next(
                (
                    name
                    for name in names
                    if callable(getattr(self._session, name, None))
                ),
                None,
            )
            if selected_name is None:
                self._publish_viewer_status(
                    "UNAVAILABLE", "backend не поддерживает MuJoCo viewer"
                )
                self._emit(
                    "WARNING",
                    "MUJOCO",
                    "VIEWER_UNAVAILABLE",
                    "Backend не поддерживает MuJoCo viewer",
                )
                return
            self._publish_viewer_status(
                "INITIALIZING",
                "открытие окна" if enabled else "закрытие окна",
            )
            try:
                getattr(self._session, selected_name)()
            except Exception as error:
                self._publish_viewer_status("ERROR", str(error))
                self._emit(
                    "ERROR",
                    "MUJOCO",
                    "VIEWER_COMMAND_FAILED",
                    "Команда MuJoCo viewer завершилась ошибкой",
                    str(error),
                )
                return
            # Queued TeleopSession calls publish an authoritative core event
            # after execution. Direct adapters have completed synchronously.
            if not selected_name.startswith("request_"):
                self._publish_viewer_status("OPEN" if enabled else "READY", "")
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
        if command.kind == "playback_seek":
            self._handle_playback_command(
                "seek", (float(command.payload),), pause_after=True
            )
            return
        if command.kind == "playback_step":
            delta = int(command.payload)
            if delta not in {-1, 1}:
                self._emit("WARNING", "PLAYBACK", "PLAYBACK_STEP_INVALID", "Допустим шаг только на один кадр", str(delta))
                return
            self._handle_playback_command("step_frame", (delta,), pause_after=True)
            return
        if command.kind == "playback_rate":
            self._handle_playback_command(
                "set_playback_rate", (float(command.payload),), pause_after=False
            )
            return
        if command.kind == "playback_loop":
            payload = command.payload
            if not isinstance(payload, tuple) or len(payload) != 3:
                raise TypeError("playback_loop requires (enabled, start, end)")
            self._handle_playback_command("set_loop", payload, pause_after=False)
            return
        if command.kind == "recording_start":
            self._start_recording(command.payload)
            return
        if command.kind == "recording_stop":
            self._stop_recording(str(command.payload or "manual"))
            return
        if command.kind == "recording_prepared":
            self._recording_prepared(command.payload)
            return
        if command.kind == "recording_finalized":
            self._recording_finalized(command.payload)
            return
        if command.kind == "robot_connect":
            self._connect_robot()
            return
        if command.kind == "robot_arm":
            request = command.payload
            if not isinstance(request, _ArmRequest):
                self._emit(
                    "ERROR",
                    "ROBOT",
                    "ROBOT_ARM_REQUEST_INVALID",
                    "Запрос arm не содержит одноразового подтверждения оператора",
                )
                return
            if request.token in self._consumed_arm_tokens:
                self._emit(
                    "ERROR",
                    "ROBOT",
                    "ROBOT_ARM_TOKEN_REUSED",
                    "Повторное использование подтверждения arm запрещено",
                )
                return
            self._consumed_arm_tokens.add(request.token)
            self._arm_robot(request.send_velocities, request.acknowledgement)
            return
        if command.kind == "robot_disarm":
            self._invalidate_robot(str(command.payload or "operator_disarm"))
            return
        if command.kind == "robot_disconnect":
            self._disconnect_robot()

    def _handle_playback_command(
        self,
        operation: str,
        arguments: tuple[object, ...],
        *,
        pause_after: bool,
    ) -> None:
        """Forward an optional seekable-source command without degrading cameras."""

        if self._recording_active():
            self._emit(
                "WARNING",
                "PLAYBACK",
                "PLAYBACK_BLOCKED_RECORDING",
                "Управление timeline заблокировано во время записи опыта",
                operation,
            )
            return
        if operation == "set_loop" and self._last_robot_ui_state in {
            RobotUiState.ARMING,
            RobotUiState.ARMED,
        }:
            self._emit(
                "WARNING",
                "PLAYBACK",
                "PLAYBACK_LOOP_BLOCKED_ARMED",
                "Сначала остановите отправку на реального робота",
            )
            return
        if self._session is None or not self._running:
            self._emit(
                "WARNING",
                "PLAYBACK",
                "PLAYBACK_NOT_RUNNING",
                "Сначала запустите файловый источник",
                operation,
            )
            return
        playback = (
            None
            if self._latest_snapshot is None
            else getattr(self._latest_snapshot, "playback", None)
        )
        source_is_camera = self._source is not None and self._source.kind == "camera"
        if source_is_camera or (
            playback is not None and not bool(getattr(playback, "seekable", False))
        ):
            self._emit(
                "WARNING",
                "PLAYBACK",
                "PLAYBACK_UNSUPPORTED",
                "Источник не поддерживает эту команду воспроизведения",
                operation,
            )
            return
        method_names = {
            "seek": ("request_seek", "seek"),
            "step_frame": ("request_step_frame", "step_relative"),
            "set_playback_rate": ("request_set_playback_rate", "set_playback_rate", "set_rate"),
            "set_loop": ("request_set_loop", "set_loop"),
        }.get(operation, ())
        method = next(
            (
                getattr(self._session, name)
                for name in method_names
                if callable(getattr(self._session, name, None))
            ),
            None,
        )
        if method is None:
            self._emit(
                "WARNING",
                "PLAYBACK",
                "PLAYBACK_UNSUPPORTED",
                "Источник не поддерживает эту команду воспроизведения",
                operation,
            )
            return
        self._invalidate_robot(f"playback_{operation}")
        self._clear_snapshot_authority()
        try:
            method(*arguments)
        except (TypeError, ValueError, RuntimeError) as error:
            # Optional media controls are operator mistakes/capability gaps,
            # not reasons to tear down an otherwise healthy camera pipeline.
            self._emit(
                "WARNING",
                "PLAYBACK",
                "PLAYBACK_COMMAND_REJECTED",
                "Команда воспроизведения отклонена",
                str(error),
            )
            return
        if pause_after:
            self._set_pipeline_state("PAUSED")
        self._emit(
            "INFO",
            "PLAYBACK",
            f"PLAYBACK_{operation.upper()}",
            "Параметры воспроизведения изменены",
            repr(arguments),
        )

    def _recording_active(self) -> bool:
        return self._recording_phase in {"PREPARING", "RECORDING", "FINALIZING"}

    def _poll_recorder_state(self) -> None:
        """Surface writer failures even while video/pipeline frames are stalled."""

        recorder = self._recorder
        if recorder is None or self._recording_phase != "RECORDING":
            return
        state = getattr(recorder, "state", None)
        state_value = str(getattr(state, "value", state)).upper()
        if state_value == "ERROR":
            self._emit_rate_limited(
                "recording-io-error",
                "ERROR",
                "RECORDER",
                "RECORDING_IO_FAILED",
                "Пакет опыта помечен неполным; pipeline продолжает работу",
            )
            self._stop_recording("recorder_error")
        elif state_value == "COMPLETE":
            self._recording_finalized(
                {"ok": True, "summary": getattr(recorder, "summary", None)}
            )

    def _resolved_experiment_root(self) -> Path:
        if self._experiment_root is not None:
            return self._experiment_root
        from PyQt6.QtCore import QStandardPaths

        documents = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        if not documents:
            documents = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.AppLocalDataLocation
            )
        if not documents:
            raise RuntimeError("Qt did not provide a writable standard path")
        base = Path(documents)
        self._experiment_root = (base / "HumanoidInterface" / "experiments").resolve()
        return self._experiment_root

    def _initialize_recorder(self) -> object | None:
        if self._recorder is not None:
            return self._recorder
        try:
            from robot_human_interface.experiments import ExperimentRecorder

            recorder = ExperimentRecorder(self._resolved_experiment_root())
            self._recorder = recorder
            recovered = tuple(getattr(recorder, "recovered_paths", ()))
            if recovered:
                self._emit(
                    "WARNING",
                    "RECORDER",
                    "INTERRUPTED_EXPERIMENTS_RECOVERED",
                    "Незавершённые пакеты опытов помечены interrupted",
                    f"count={len(recovered)}",
                )
            return recorder
        except Exception as error:
            self._emit(
                "WARNING",
                "RECORDER",
                "RECORDER_INITIALIZATION_FAILED",
                "Каталог опытов пока недоступен; pipeline продолжает работу",
                str(error),
            )
            return None

    def _recording_block_reason(self) -> str:
        if not self._runtime_status.physical_output_allowed:
            return "Запись исследования недоступна в DEMO-режиме."
        if not self._running or self._pipeline_state != "RUNNING":
            return "Production pipeline должен быть в состоянии RUNNING."
        if self._latest_snapshot is None or self._latest_snapshot_received_s is None:
            return "Нет свежего снимка pipeline."
        if self._clock() - self._latest_snapshot_received_s > self._max_snapshot_age_s:
            return "Последний снимок pipeline устарел."
        telemetry = getattr(self._latest_snapshot, "telemetry", {})
        if not isinstance(telemetry, Mapping):
            return "Pipeline не сообщил состояние калибровки."
        calibrating = telemetry.get("calibrating")
        progress = telemetry.get("calibration_progress")
        try:
            complete = calibrating is False and float(progress) >= 1.0
        except (TypeError, ValueError):
            complete = False
        if not complete:
            return "Сначала завершите калибровку (100%)."
        return ""

    def _start_recording(self, raw_spec: object) -> None:
        from robot_human_interface.experiments import (
            ExperimentRecorder,
            ExperimentSpec,
        )

        if not isinstance(raw_spec, ExperimentSpec):
            self.recorder_state_changed.emit("ERROR", "Некорректное описание опыта")
            self._emit("ERROR", "RECORDER", "RECORDING_SPEC_INVALID", "Некорректное описание опыта")
            return
        if self._recording_active():
            self._emit("WARNING", "RECORDER", "RECORDING_ALREADY_ACTIVE", "Запись опыта уже активна")
            return
        reason = self._recording_block_reason()
        if reason:
            self.recorder_state_changed.emit("ERROR", reason)
            self._emit("WARNING", "RECORDER", "RECORDING_START_BLOCKED", "Запись опыта не запущена", reason)
            return
        recorder = self._initialize_recorder()
        if not isinstance(recorder, ExperimentRecorder):
            self.recorder_state_changed.emit("ERROR", "Каталог опытов недоступен")
            return
        # Artifact hashing and optional source-video copying can take seconds.
        # Keep them off the sole pipeline QThread so frame processing and the
        # physical-output watchdog remain responsive throughout PREPARING.
        # Recorder lifecycle alone never mutates the physical controller.
        self._recording_spec = raw_spec
        self._active_run_id = None
        self._recording_phase = "PREPARING"
        self._recording_cancel.clear()
        self._recording_cancel_reason = "application_closed"
        self._camera_video_unavailable = False
        self._camera_video_last_dropped = 0
        self._camera_video_last_error = ""
        if self._camera_video_sink is not None:
            self._recording_phase = "ERROR"
            self._recording_spec = None
            self.recorder_state_changed.emit(
                "ERROR", "Предыдущий video sink ещё не завершён"
            )
            return
        self.recorder_state_changed.emit("PREPARING", None)
        media_source = (
            self._source.path
            if raw_spec.record_video
            and self._source is not None
            and self._source.kind != "camera"
            and self._source.path
            else None
        )
        session_config = getattr(self._session, "config", None)
        source_id = None if self._source is None else self._source.source_id

        def prepare() -> None:
            try:
                run_id = recorder.start(
                    raw_spec,
                    session_config=session_config,
                    source_id=source_id,
                    media_source_file=media_source,
                )
            except Exception as error:
                self.submit(
                    "recording_prepared",
                    {
                        "ok": False,
                        "error": str(error),
                        "summary": recorder.summary,
                    },
                )
                return
            self.submit(
                "recording_prepared",
                {
                    "ok": True,
                    "run_id": run_id,
                    "summary": recorder.summary,
                },
            )

        task = Thread(
            target=prepare,
            name="experiment-recorder-prepare",
            daemon=True,
        )
        self._recording_task = task
        try:
            task.start()
        except Exception as error:
            self._recording_task = None
            self._recording_phase = "ERROR"
            self._recording_spec = None
            self.recorder_state_changed.emit("ERROR", str(error))
            self._emit(
                "ERROR",
                "RECORDER",
                "RECORDING_START_FAILED",
                "Запись опыта не запущена",
                str(error),
            )

    def _recording_prepared(self, payload: object) -> None:
        self._recording_task = None
        result = payload if isinstance(payload, Mapping) else {}
        summary = result.get("summary")
        if not bool(result.get("ok", False)):
            error = str(result.get("error") or "неизвестная ошибка подготовки")
            self._active_run_id = (
                None if summary is None else str(getattr(summary, "run_id", "") or "") or None
            )
            self._recording_phase = "ERROR"
            self._recording_spec = None
            self._recording_cancel.clear()
            self.recorder_state_changed.emit("ERROR", error)
            if summary is not None:
                self.experiment_completed.emit(summary)
            self._emit(
                "ERROR",
                "RECORDER",
                "RECORDING_START_FAILED",
                "Запись опыта не запущена",
                error,
            )
            self._active_run_id = None
            return
        if self._recording_cancel.is_set() or self._recording_phase == "FINALIZING":
            self._begin_recording_finalization(self._recording_cancel_reason)
            return
        self._recording_phase = "RECORDING"
        self._active_run_id = str(result.get("run_id") or "") or None
        self.recorder_state_changed.emit("RECORDING", summary)
        self._emit(
            "INFO",
            "RECORDER",
            "RECORDING_STARTED",
            "Запись опыта начата",
            str(result.get("run_id") or ""),
        )

    def _append_recording_snapshot(self, snapshot: object) -> None:
        recorder = self._recorder
        if recorder is None or self._recording_phase != "RECORDING":
            return
        recorder_state = getattr(recorder, "state", None)
        recorder_state_value = str(
            getattr(recorder_state, "value", recorder_state)
        ).upper()
        if recorder_state_value == "ERROR":
            self._emit_rate_limited(
                "recording-io-error",
                "ERROR",
                "RECORDER",
                "RECORDING_IO_FAILED",
                "Пакет опыта помечен неполным; pipeline продолжает работу",
            )
            self._stop_recording("recorder_error")
            return
        if recorder_state_value != "RECORDING":
            return
        try:
            accepted = bool(recorder.append(snapshot))
        except Exception as error:
            self._emit_rate_limited(
                f"recording-append:{type(error).__name__}",
                "ERROR",
                "RECORDER",
                "RECORDING_SAMPLE_REJECTED",
                "Числовой кадр опыта не записан; pipeline продолжает работу",
                str(error),
            )
            accepted = False
        if not accepted:
            self._emit_rate_limited(
                "recording-drop",
                "WARNING",
                "RECORDER",
                "RECORDING_SAMPLE_DROPPED",
                "Очередь записи переполнена; опыт будет помечен неполным",
            )
        if bool(getattr(self._recording_spec, "record_video", False)) and self._source is not None and self._source.kind == "camera":
            self._append_camera_video_frame(snapshot)
        recorder_state = getattr(recorder, "state", None)
        if str(getattr(recorder_state, "value", recorder_state)).upper() == "ERROR":
            self._stop_recording("recorder_error")
            return
        now = self._clock()
        if now - self._last_recorder_progress_s >= 0.2:
            self._last_recorder_progress_s = now
            summary = recorder.summary
            self.recorder_progress.emit(summary)
            self.recorder_state_changed.emit("RECORDING", summary)

    def _append_camera_video_frame(self, snapshot: object) -> None:
        if self._camera_video_unavailable:
            return
        frame = getattr(snapshot, "frame", None)
        image = getattr(frame, "image_bgr", None)
        shape = getattr(image, "shape", None)
        if image is None or shape is None or len(shape) != 3:
            self._camera_video_unavailable = True
            self._emit("WARNING", "RECORDER", "VIDEO_FRAME_UNAVAILABLE", "Видео камеры недоступно; числовая запись продолжается")
            self._mark_recorder_incomplete("video_frame_unavailable")
            return
        try:
            if self._camera_video_sink is None:
                factory = self._camera_video_sink_factory
                if factory is None:
                    from robot_human_interface.experiments import CameraVideoSink

                    factory = CameraVideoSink

                assert self._recorder is not None
                target = self._recorder.reserve_media_path("processed.mp4")
                height, width = int(shape[0]), int(shape[1])
                fps = 20.0 if self._source is None else float(self._source.fps or 20.0)
                self._camera_video_sink = factory(
                    target,
                    fps,
                    (width, height),
                )
            accepted = bool(self._camera_video_sink.append(image))
            if not accepted:
                self._poll_camera_video_sink()
        except Exception as error:
            self._camera_video_unavailable = True
            self._mark_recorder_incomplete(
                f"video_sink_error:{type(error).__name__}"
            )
            self._emit("WARNING", "RECORDER", "VIDEO_CODEC_UNAVAILABLE", "Видео не записывается; числовая запись продолжается", str(error))

    def _mark_recorder_incomplete(self, reason: str) -> None:
        recorder = self._recorder
        marker = getattr(recorder, "mark_incomplete", None)
        if callable(marker):
            try:
                marker(reason)
            except Exception:
                pass

    def _poll_camera_video_sink(self) -> None:
        sink = self._camera_video_sink
        if sink is None:
            return
        status = getattr(sink, "status", None)
        if status is None:
            return
        dropped = int(getattr(status, "dropped", 0) or 0)
        error = str(getattr(status, "error", "") or "")
        if dropped > self._camera_video_last_dropped:
            self._camera_video_last_dropped = dropped
            self._mark_recorder_incomplete("video_queue_overflow")
            self._emit_rate_limited(
                "camera-video-drop",
                "WARNING",
                "RECORDER",
                "VIDEO_FRAME_DROPPED",
                "Очередь MP4 переполнена; числовая запись продолжается",
                f"dropped={dropped}",
            )
        if error and error != self._camera_video_last_error:
            self._camera_video_last_error = error
            self._camera_video_unavailable = True
            self._mark_recorder_incomplete("video_encoder_error")
            self._emit(
                "WARNING",
                "RECORDER",
                "VIDEO_CODEC_UNAVAILABLE",
                "Видео не записывается; числовая запись продолжается",
                error,
            )

    def _detach_camera_video_sink(self) -> object | None:
        sink, self._camera_video_sink = self._camera_video_sink, None
        return sink

    def _close_camera_video(self) -> object | None:
        sink = self._detach_camera_video_sink()
        if sink is None:
            return None
        try:
            status = sink.close()
        except Exception as error:
            self._mark_recorder_incomplete(
                f"video_close_error:{type(error).__name__}"
            )
            return None
        if bool(getattr(status, "incomplete", False)):
            self._mark_recorder_incomplete("video_incomplete")
        return status

    def _stop_recording(self, reason: str) -> None:
        recorder = self._recorder
        if recorder is None or not self._recording_active():
            return
        reason = str(reason or "manual")
        if self._recording_phase == "PREPARING":
            # The preparation helper will report back on the command queue;
            # its result is then finalized without ever blocking this QThread.
            self._recording_cancel_reason = reason
            self._recording_cancel.set()
            self._recording_phase = "FINALIZING"
            self.recorder_state_changed.emit("FINALIZING", recorder.summary)
            return
        if self._recording_phase == "FINALIZING":
            return
        self._begin_recording_finalization(reason)

    def _begin_recording_finalization(self, reason: str) -> None:
        recorder = self._recorder
        if recorder is None:
            self._recording_phase = "ERROR"
            self._recording_spec = None
            self.recorder_state_changed.emit("ERROR", "Recorder недоступен")
            return
        self._recording_phase = "FINALIZING"
        self._recording_cancel_reason = str(reason or "manual")
        self._recording_cancel.set()
        self.recorder_state_changed.emit("FINALIZING", recorder.summary)
        video_sink = self._detach_camera_video_sink()

        def finalize() -> None:
            video_error = ""
            if video_sink is not None:
                # The recorder hashes media and atomically renames the package
                # in stop().  It therefore must not see an MP4 that can still
                # be mutated by the encoder thread.  A close timeout leaves the
                # package in FINALIZING/.partial and this daemon helper keeps
                # waiting; shutdown may abandon the helper, but never publishes
                # a falsely stable media hash.
                video_closed = False
                while not video_closed:
                    try:
                        video_status = video_sink.close(timeout_s=1.0)
                    except Exception as error:
                        video_error = str(error)
                        self._mark_recorder_incomplete(
                            f"video_close_error:{type(error).__name__}"
                        )
                        sleep(0.05)
                        continue
                    video_error = str(
                        getattr(video_status, "error", "") or video_error
                    )
                    if bool(getattr(video_status, "incomplete", False)):
                        self._mark_recorder_incomplete("video_incomplete")
                    video_closed = bool(getattr(video_status, "closed", True))
                    if not video_closed:
                        self._mark_recorder_incomplete("video_close_timeout")
                        sleep(0.05)
            try:
                summary = recorder.stop(self._recording_cancel_reason)
            except Exception as error:
                self.submit(
                    "recording_finalized",
                    {
                        "ok": False,
                        "error": str(error),
                        "summary": recorder.summary,
                        "video_error": video_error,
                    },
                )
                return
            self.submit(
                "recording_finalized",
                {"ok": True, "summary": summary, "video_error": video_error},
            )

        task = Thread(
            target=finalize,
            name="experiment-recorder-finalize",
            daemon=True,
        )
        self._recording_task = task
        try:
            task.start()
        except Exception as error:
            self._recording_task = None
            self._recording_finalized(
                {"ok": False, "error": str(error), "summary": recorder.summary}
            )

    def _recording_finalized(self, payload: object) -> None:
        self._recording_task = None
        result = payload if isinstance(payload, Mapping) else {}
        summary = result.get("summary")
        error = str(result.get("error") or "")
        video_error = str(result.get("video_error") or "")
        if video_error and video_error != self._camera_video_last_error:
            self._camera_video_last_error = video_error
            self._emit(
                "WARNING",
                "RECORDER",
                "VIDEO_FINALIZE_FAILED",
                "Видео завершено с ошибкой; числовой пакет сохранён",
                video_error,
            )
        if summary is None:
            self._recording_phase = "ERROR"
            self.recorder_state_changed.emit(
                "ERROR", error or "Recorder не вернул итог опыта"
            )
            self._emit(
                "ERROR",
                "RECORDER",
                "RECORDING_FINALIZE_FAILED",
                "Опыт завершён с ошибкой",
                error,
            )
            self._recording_spec = None
            self._active_run_id = None
            self._recording_cancel.clear()
            return
        state = str(getattr(getattr(summary, "state", None), "value", "complete")).upper()
        self._recording_phase = state if state in {"COMPLETE", "ERROR"} else "ERROR"
        self.recorder_state_changed.emit(state, summary)
        self.experiment_completed.emit(summary)
        self._emit(
            "INFO" if not bool(getattr(summary, "incomplete", True)) else "WARNING",
            "RECORDER",
            "RECORDING_FINALIZED",
            "Запись опыта завершена",
            f"reason={getattr(summary, 'stop_reason', self._recording_cancel_reason)}; "
            f"samples={getattr(summary, 'sample_count', 0)}; "
            f"path={getattr(summary, 'path', '')}; error={error}",
        )
        self._recording_spec = None
        self._active_run_id = None
        self._recording_cancel.clear()

    def _shutdown_recording(self) -> None:
        """Cooperatively finish recorder helpers during application shutdown."""

        if self._recording_active() and self._recording_task is None:
            # Reuse the regular helper so camera media is fully closed before
            # ExperimentRecorder.stop() hashes or renames the package.
            self._begin_recording_finalization("application_closed")
        elif self._recording_active():
            self._recording_cancel_reason = "application_closed"
            self._recording_cancel.set()
            self._recording_phase = "FINALIZING"
        task = self._recording_task
        if task is not None and task.is_alive():
            task.join(timeout=35.0)
        if task is not None and task.is_alive():
            self._emit(
                "ERROR",
                "RECORDER",
                "RECORDING_HELPER_TIMEOUT",
                "Фоновая операция записи не завершилась вовремя; пакет останется partial",
            )
            return
        self._recording_task = None
        recorder = self._recorder
        if recorder is None:
            return
        state = getattr(recorder, "state", None)
        state_value = str(getattr(state, "value", state)).upper()
        if state_value in {"PREPARING", "RECORDING", "FINALIZING", "ERROR"}:
            try:
                recorder.stop("application_closed", timeout_s=30.0)
            except Exception as error:
                self._emit(
                    "ERROR",
                    "RECORDER",
                    "RECORDING_SHUTDOWN_FAILED",
                    "Не удалось полностью завершить пакет опыта",
                    str(error),
                )

    def _start_current(self) -> None:
        if self._source is None:
            raise RuntimeError("источник не выбран")
        if self._session is None:
            self._session = self._factory(self._source)
            if isinstance(self._session, DemoSession):
                reason = self._session.fallback_reason
                self._set_runtime_status(RuntimeStatus.demo(reason))
                self._emit(
                    "WARNING",
                    "PIPELINE",
                    "SYNTHETIC_FALLBACK",
                    "Используется безопасный демонстрационный pipeline",
                    reason,
                )
            else:
                self._set_runtime_status(RuntimeStatus.production())
        self._call(self._session, ("start", "request_start"), required=True)
        self._running = True
        self._set_pipeline_state("RUNNING")
        self._publish_viewer_status(
            "READY"
            if self._runtime_status.mode is RuntimeMode.PRODUCTION
            else "UNAVAILABLE",
            "" if self._runtime_status.mode is RuntimeMode.PRODUCTION else "DEMO backend",
        )
        self._emit("INFO", "PIPELINE", "SESSION_STARTED", "Сессия запущена")

    def _set_runtime_status(self, status: RuntimeStatus) -> None:
        if not isinstance(status, RuntimeStatus):
            raise TypeError("status must be RuntimeStatus")
        changed = status != self._runtime_status
        self._runtime_status = status
        if status.mode is RuntimeMode.DEMO:
            # Retained commands from a prior production session are never
            # reusable after any fallback to the synthetic/demo pipeline.
            self._invalidate_robot("runtime_demo")
        if changed:
            self.runtime_status_changed.emit(status)
        self._publish_readiness(force=changed)

    def _set_pipeline_state(self, state: str) -> None:
        normalized = str(state).upper()
        self._pipeline_state = normalized
        self.state_changed.emit(normalized)
        self._publish_readiness(force=True)

    def _publish_viewer_status(
        self,
        state: str,
        details: str = "",
        *,
        force: bool = False,
    ) -> None:
        normalized = str(state).upper()
        if normalized not in {"UNAVAILABLE", "INITIALIZING", "READY", "OPEN", "ERROR"}:
            raise ValueError(f"unsupported viewer state: {state}")
        detail_text = str(details or "")
        changed = (
            normalized != self._viewer_state or detail_text != self._viewer_details
        )
        self._viewer_state = normalized
        self._viewer_details = detail_text
        self._viewer_open = normalized == "OPEN"
        if force or changed:
            # Keep the historical bool signal for embedders while exposing an
            # explicit authoritative lifecycle to the first-party GUI.
            self.viewer_changed.emit(self._viewer_open)
            self.viewer_status_changed.emit(normalized, detail_text)

    def _clear_snapshot_authority(self) -> None:
        self._latest_snapshot = None
        self._latest_snapshot_received_s = None
        self._last_snapshot_key = None
        self.safety_flags_changed.emit(False, False, False)
        self._publish_readiness(force=True)

    def _publish_robot_state(
        self,
        state: RobotUiState | str,
        details: str = "",
        *,
        force: bool = False,
    ) -> None:
        resolved = RobotUiState(str(getattr(state, "value", state)).lower())
        detail_text = str(details or "")
        changed = (
            resolved is not self._last_robot_ui_state
            or detail_text != self._last_robot_state_details
        )
        self._last_robot_ui_state = resolved
        self._last_robot_state_details = detail_text
        if force or changed:
            self.robot_state_changed.emit(resolved.value.upper(), detail_text)
            self._publish_readiness(force=True)

    def _emit_rate_limited(
        self,
        key: str,
        severity: str,
        subsystem: str,
        event_code: str,
        message: str,
        details: str = "",
    ) -> None:
        now = self._clock()
        last = self._state_log_times.get(key)
        if last is not None and now - last < self._state_log_interval_s:
            return
        self._state_log_times[key] = now
        self._emit(severity, subsystem, event_code, message, details)

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
            code = str(
                getattr(
                    event,
                    "code",
                    event.get("code", event.get("event_code", ""))
                    if isinstance(event, Mapping)
                    else "",
                )
            ).upper()
            details = getattr(
                event,
                "details",
                event.get("details", {}) if isinstance(event, Mapping) else {},
            )
            details = details if isinstance(details, Mapping) else {}
            if code == "MUJOCO_VIEWER_OPENED":
                self._publish_viewer_status("OPEN", "")
            elif code == "MUJOCO_VIEWER_CLOSED":
                self._publish_viewer_status("READY", "")
            elif code == "SESSION_COMMAND_FAILED" and str(
                details.get("command", "")
            ) in {"open_viewer", "close_viewer"}:
                self._publish_viewer_status(
                    "ERROR", str(details.get("error", "viewer command failed"))
                )
            recorder = self._recorder
            recorder_state = None if recorder is None else getattr(recorder, "state", None)
            if (
                recorder is not None
                and str(getattr(recorder_state, "value", recorder_state)).upper()
                == "RECORDING"
            ):
                try:
                    recorder.record_event(event)
                except Exception:
                    pass
            level = getattr(
                event,
                "level",
                event.get("severity", "INFO") if isinstance(event, Mapping) else "INFO",
            )
            subsystem = getattr(
                event,
                "subsystem",
                event.get("subsystem", "PIPELINE") if isinstance(event, Mapping) else "PIPELINE",
            )
            message = getattr(
                event,
                "message_ru",
                event.get("message", str(event)) if isinstance(event, Mapping) else str(event),
            )
            sequence = (
                None
                if self._latest_snapshot is None
                else getattr(self._latest_snapshot, "sequence", None)
            )
            self.event_ready.emit(
                {
                    "severity": str(getattr(level, "value", level)).upper(),
                    "subsystem": str(getattr(subsystem, "value", subsystem)).upper(),
                    "event_code": code or "SESSION_EVENT",
                    "message": str(message),
                    "details": dict(details),
                    "run_id": self._active_run_id or "",
                    "source_id": "" if self._source is None else self._source.source_id,
                    "sequence": sequence,
                }
            )

    def _close_session(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    self._emit("WARNING", "PIPELINE", "SESSION_CLOSE_FAILED", "Не удалось полностью закрыть сессию", str(error))
        self._publish_viewer_status("UNAVAILABLE", "pipeline не запущен")

    def _connect_robot(self) -> None:
        if not self._runtime_status.physical_output_allowed:
            if self._robot is not None:
                self._disconnect_robot("runtime_demo")
            reason = self._runtime_status.fallback_reason or "demo runtime"
            self._publish_robot_state(RobotUiState.DISCONNECTED, reason, force=True)
            self._emit_rate_limited(
                "robot-connect-demo",
                "ERROR",
                "ROBOT",
                "ROBOT_CONNECT_BLOCKED_DEMO",
                "Подключение к реальному роботу запрещено в демонстрационном режиме",
                reason,
            )
            return

        self._publish_robot_state(RobotUiState.CONNECTING, self._robot_endpoint, force=True)
        try:
            if self._robot is None:
                factory = self._robot_factory
                if factory is None:
                    from robot_human_interface.protocol import (
                        create_legacy_websocket_controller,
                    )

                    factory = create_legacy_websocket_controller
                self._robot = factory(self._robot_endpoint)
            connected = bool(self._robot.connect())
            status = self._controller_status()
        except Exception as error:
            connected = False
            status = None
            details = str(error)
            self._publish_robot_state(RobotUiState.DEGRADED, details, force=True)
        else:
            state = self._controller_ui_state(status)
            if connected and state is not RobotUiState.CONNECTED_DISARMED:
                connected = False
                details = (
                    "controller did not report authoritative connected_disarmed state"
                )
                disconnect = getattr(self._robot, "disconnect", None)
                if callable(disconnect):
                    try:
                        disconnect("connect_status_unavailable")
                    except Exception:
                        pass
                status = self._controller_status()
                state = self._controller_ui_state(status)
            else:
                details = (
                    ""
                    if connected
                    else self._status_details(status, "connection failed")
                )
            self._publish_robot_state(state, details, force=True)

        self._emit(
            "INFO" if connected else "ERROR",
            "ROBOT",
            "ROBOT_CONNECTED_DISARMED" if connected else "ROBOT_CONNECT_FAILED",
            (
                "Соединение установлено; команды не отправляются"
                if connected
                else "Соединение с роботом не установлено"
            ),
            details,
        )

    def _arm_robot(self, send_velocities: bool, acknowledgement: object | None) -> None:
        from robot_human_interface.protocol import OperatorSafetyAcknowledgement

        if not isinstance(acknowledgement, OperatorSafetyAcknowledgement) or not acknowledgement.complete:
            self._publish_robot_state(
                self._controller_ui_state(self._controller_status()),
                "operator safety acknowledgement is missing or incomplete",
                force=True,
            )
            self._emit_rate_limited(
                "robot-arm-ack-required",
                "ERROR",
                "ROBOT",
                "ROBOT_ARM_ACK_REQUIRED",
                "Включение выхода требует нового подтверждения свободной зоны и аппаратного E-stop",
            )
            return
        if not self._runtime_status.physical_output_allowed:
            reason = self._runtime_status.fallback_reason or "demo runtime"
            self._publish_robot_state(
                self._controller_ui_state(self._controller_status()), reason, force=True
            )
            self._emit_rate_limited(
                "robot-arm-demo",
                "ERROR",
                "ROBOT",
                "ROBOT_ARM_BLOCKED_DEMO",
                "Включение реального выхода запрещено в демонстрационном режиме",
                reason,
            )
            return
        if self._robot is None:
            self._publish_robot_state(
                RobotUiState.DISCONNECTED, "robot controller is not connected", force=True
            )
            self._emit_rate_limited(
                "robot-arm-disconnected",
                "ERROR",
                "ROBOT",
                "ROBOT_ARM_BLOCKED",
                "Включение реального выхода заблокировано",
                "robot controller is not connected",
            )
            return

        # This is the first worker-owned check after the GUI confirmation and
        # command queue boundary.  Widgets are never the arm authority.
        readiness = self._compute_readiness()
        self._publish_readiness(readiness, force=True)
        if not readiness.ready:
            self._publish_robot_state(
                self._controller_ui_state(self._controller_status()),
                readiness.reason,
                force=True,
            )
            self._emit_rate_limited(
                f"robot-arm-blocked:{readiness.reason_code.value}",
                "ERROR",
                "ROBOT",
                "ROBOT_ARM_BLOCKED",
                "Включение реального выхода заблокировано",
                readiness.reason,
            )
            return

        self._publish_robot_state(RobotUiState.ARMING, "", force=True)

        # Re-evaluate immediately before changing controller state.  The
        # readiness contains the command generation checked below atomically
        # by SafeRobotController.arm().
        readiness = self._compute_readiness()
        self._publish_readiness(readiness, force=True)
        if not readiness.ready:
            self._publish_robot_state(
                self._controller_ui_state(self._controller_status()),
                readiness.reason,
                force=True,
            )
            self._emit_rate_limited(
                f"robot-arm-race:{readiness.reason_code.value}",
                "ERROR",
                "ROBOT",
                "ROBOT_ARM_READINESS_CHANGED",
                "Готовность изменилась до включения реального выхода",
                readiness.reason,
            )
            return

        arm_method = getattr(self._robot, "arm", None)
        if not callable(arm_method):
            raise RuntimeError("robot controller has no arm()")
        parameters: Mapping[str, Parameter]
        try:
            parameters = signature(arm_method).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind is Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        arm_kwargs: dict[str, object] = {}
        if accepts_kwargs or "send_velocities" in parameters:
            arm_kwargs["send_velocities"] = bool(send_velocities)
        else:
            # Backwards-compatible adapters may still expose the historical
            # mutable option.  Readiness/status remains mandatory regardless.
            self._robot.send_velocities = bool(send_velocities)
        if accepts_kwargs or "expected_command_generation" in parameters:
            arm_kwargs["expected_command_generation"] = readiness.command_generation

        armed = bool(arm_method(acknowledgement, **arm_kwargs))
        status = self._controller_status()
        state = self._controller_ui_state(status)
        if armed and state is not RobotUiState.ARMED:
            armed = False
            details = "controller did not report authoritative armed state"
            invalidate = getattr(self._robot, "invalidate", None)
            if callable(invalidate):
                try:
                    invalidate("arm_status_unavailable")
                except Exception:
                    pass
            status = self._controller_status()
            state = self._controller_ui_state(status)
        else:
            details = (
                ""
                if armed
                else self._status_details(status, "interlock rejected arm")
            )
        self._publish_robot_state(state, details, force=True)
        self._emit(
            "WARNING" if armed else "ERROR",
            "ROBOT",
            "ROBOT_ARMED" if armed else "ROBOT_ARM_BLOCKED",
            (
                "Отправка safe_command разрешена оператором"
                if armed
                else "Контроллер отклонил включение отправки"
            ),
            details,
        )

    def _update_robot_command(self, snapshot: object) -> None:
        if self._robot is None:
            return
        status = self._controller_status()
        state = self._controller_ui_state(status)
        if state not in {RobotUiState.CONNECTED_DISARMED, RobotUiState.ARMED}:
            return

        facts = self._snapshot_safety_facts(snapshot)
        if not facts.valid:
            self._invalidate_robot(facts.reason_code.value)
            self._emit_rate_limited(
                f"snapshot-invalid:{facts.reason_code.value}",
                "ERROR",
                "ROBOT",
                "SAFE_COMMAND_UNAVAILABLE",
                "Отправка заблокирована: нет авторитетной safe-команды",
                facts.reason,
            )
            return

        try:
            self._robot.submit_safe_command(
                facts.command,
                free_base=facts.free_base_active,
                balance_enabled=facts.balance_active,
            )
        except Exception as error:
            status = self._controller_status()
            self._publish_robot_state(
                self._controller_ui_state(status), str(error), force=True
            )
            self._emit_rate_limited(
                f"safe-command-rejected:{type(error).__name__}",
                "ERROR",
                "ROBOT",
                "SAFE_COMMAND_REJECTED",
                "Safe-команда не прошла выходной контроллер",
                str(error),
            )
            return
        self._tick_robot_only()

    def _snapshot_safety_facts(self, snapshot: object) -> _SnapshotSafetyFacts:
        declared_valid = getattr(snapshot, "safe_valid", None)
        if declared_valid is not None and type(declared_valid) is not bool:
            return _SnapshotSafetyFacts(
                False,
                ReadinessReason.SAFE_COMMAND_INVALID,
                "snapshot safe_valid flag is not boolean",
            )
        if declared_valid is False:
            return _SnapshotSafetyFacts(
                False,
                ReadinessReason.SAFE_VALID_FALSE,
                "pipeline explicitly marked safe command invalid",
            )

        command = getattr(snapshot, "safe_command", None)
        if command is None:
            return _SnapshotSafetyFacts(
                False,
                ReadinessReason.SAFE_COMMAND_MISSING,
                "snapshot has no finalized safe command",
            )

        from robot_human_interface.protocol import FinalizedSafeCommand

        if not isinstance(command, FinalizedSafeCommand) or bool(
            getattr(command, "stale", True)
        ):
            return _SnapshotSafetyFacts(
                False,
                ReadinessReason.SAFE_COMMAND_INVALID,
                "safe command lacks final provenance or is marked stale",
            )

        flags = self._resolve_safety_flags(snapshot)
        if flags is None:
            return _SnapshotSafetyFacts(
                False,
                ReadinessReason.SAFETY_PROVENANCE_MISSING,
                "free-base/balance provenance is missing or inconsistent",
                command,
            )
        free_base, balance_active = flags
        if not free_base:
            return _SnapshotSafetyFacts(
                False,
                ReadinessReason.FREE_BASE_INACTIVE,
                "free-base mode is not active",
                command,
                free_base,
                balance_active,
            )
        if not balance_active:
            return _SnapshotSafetyFacts(
                False,
                ReadinessReason.BALANCE_INACTIVE,
                "balance mode is not active",
                command,
                free_base,
                balance_active,
            )
        return _SnapshotSafetyFacts(
            True,
            ReadinessReason.READY,
            "",
            command,
            free_base,
            balance_active,
        )

    @staticmethod
    def _resolve_safety_flags(snapshot: object) -> tuple[bool, bool] | None:
        command = getattr(snapshot, "safe_command", None)
        command_free = getattr(command, "free_base_active", None)
        command_balance = getattr(command, "balance_active", None)
        if type(command_free) is not bool or type(command_balance) is not bool:
            return None

        observed: list[tuple[bool, bool]] = []
        direct_free = getattr(snapshot, "free_base_active", None)
        direct_balance = getattr(snapshot, "balance_active", None)
        if direct_free is not None or direct_balance is not None:
            if type(direct_free) is not bool or type(direct_balance) is not bool:
                return None
            observed.append((direct_free, direct_balance))

        telemetry = getattr(snapshot, "telemetry", None)
        if isinstance(telemetry, Mapping):
            sentinel = object()
            free_value = telemetry.get(
                "free_base_active", telemetry.get("free_base", sentinel)
            )
            balance_value = telemetry.get(
                "balance_active", telemetry.get("balance_enabled", sentinel)
            )
            if free_value is not sentinel or balance_value is not sentinel:
                if type(free_value) is not bool or type(balance_value) is not bool:
                    return None
                observed.append((free_value, balance_value))

        command_flags = (command_free, command_balance)
        if any(value != command_flags for value in observed):
            return None
        return command_flags

    def _compute_readiness(self, now_s: float | None = None) -> RobotReadiness:
        now = self._clock() if now_s is None else float(now_s)
        status = self._controller_status()
        controller_state = self._controller_ui_state(status)
        generation = int(getattr(status, "command_generation", 0) or 0)
        source_id = None if self._source is None else self._source.source_id
        snapshot = self._latest_snapshot
        sequence_value = None if snapshot is None else getattr(snapshot, "sequence", None)
        try:
            sequence = None if sequence_value is None else int(sequence_value)
        except (TypeError, ValueError):
            sequence = None
        if sequence is not None and sequence < 0:
            sequence = None
        age = (
            None
            if self._latest_snapshot_received_s is None
            else max(0.0, now - self._latest_snapshot_received_s)
        )
        facts = (
            _SnapshotSafetyFacts(
                False,
                ReadinessReason.SAFE_COMMAND_MISSING,
                "snapshot has no finalized safe command",
            )
            if snapshot is None
            else self._snapshot_safety_facts(snapshot)
        )

        def result(
            code: ReadinessReason,
            reason: str,
            *,
            ready: bool = False,
        ) -> RobotReadiness:
            return RobotReadiness(
                ready=ready,
                reason_code=code,
                reason=reason,
                evaluated_at_s=now,
                runtime=self._runtime_status,
                pipeline_state=self._pipeline_state,
                robot_state=self._last_robot_ui_state,
                source_id=source_id,
                snapshot_sequence=sequence,
                snapshot_age_s=age,
                command_generation=generation,
                safe_command_valid=facts.valid,
                free_base_active=facts.free_base_active,
                balance_active=facts.balance_active,
            )

        if self._runtime_status.mode is RuntimeMode.DEMO:
            return result(
                ReadinessReason.RUNTIME_DEMO,
                self._runtime_status.fallback_reason or "demo runtime",
            )
        if self._pipeline_state != "RUNNING" or not self._running:
            return result(
                ReadinessReason.PIPELINE_NOT_RUNNING,
                "production pipeline is not running",
            )
        if controller_state is not RobotUiState.CONNECTED_DISARMED:
            return result(
                ReadinessReason.ROBOT_NOT_CONNECTED,
                f"robot state is {controller_state.value}, not connected_disarmed",
            )
        if snapshot is None or self._latest_snapshot_received_s is None:
            return result(ReadinessReason.NO_SNAPSHOT, "no pipeline snapshot is available")
        if age is None or age > self._max_snapshot_age_s:
            return result(
                ReadinessReason.SNAPSHOT_STALE,
                f"pipeline snapshot is older than {self._max_snapshot_age_s:.3f} s",
            )
        raw_status = getattr(snapshot, "status", "")
        snapshot_status = str(getattr(raw_status, "value", raw_status)).lower()
        if snapshot_status != "running":
            return result(
                ReadinessReason.SNAPSHOT_STATUS_INVALID,
                f"snapshot status is {snapshot_status or 'unknown'}",
            )
        if not facts.valid:
            return result(facts.reason_code, facts.reason)

        can_arm = getattr(self._robot, "can_arm", None)
        if not callable(can_arm):
            return result(
                ReadinessReason.CONTROLLER_NOT_READY,
                "robot controller has no authoritative can_arm status",
            )
        try:
            controller_ready, controller_reason = can_arm()
        except Exception as error:
            return result(
                ReadinessReason.CONTROLLER_NOT_READY,
                f"robot readiness query failed: {error}",
            )
        if type(controller_ready) is not bool or not controller_ready:
            return result(
                ReadinessReason.CONTROLLER_NOT_READY,
                str(controller_reason or "robot controller rejected readiness"),
            )
        return result(ReadinessReason.READY, "", ready=True)

    def _publish_readiness(
        self,
        readiness: RobotReadiness | None = None,
        *,
        force: bool = False,
    ) -> RobotReadiness:
        resolved = self._compute_readiness() if readiness is None else readiness
        if not isinstance(resolved, RobotReadiness):
            raise TypeError("readiness must be RobotReadiness")
        key = resolved.semantic_key()
        self._last_readiness = resolved
        if force or key != self._last_readiness_key:
            self._last_readiness_key = key
            self.robot_readiness_changed.emit(resolved)
        return resolved

    @staticmethod
    def _controller_ui_state(status: object | None) -> RobotUiState:
        if status is None:
            return RobotUiState.DISCONNECTED
        raw_state = getattr(status, "state", RobotUiState.DEGRADED)
        value = str(getattr(raw_state, "value", raw_state)).lower()
        try:
            return RobotUiState(value)
        except ValueError:
            return RobotUiState.DEGRADED

    def _controller_status(self) -> object | None:
        if self._robot is None:
            return None
        status = getattr(self._robot, "status", None)
        if not callable(status):
            return None
        try:
            resolved = status()
        except Exception as error:
            self._emit_rate_limited(
                f"robot-status:{type(error).__name__}",
                "ERROR",
                "ROBOT",
                "ROBOT_STATUS_FAILED",
                "Не удалось получить авторитетное состояние робота",
                str(error),
            )
            return None
        if resolved != self._last_robot_status:
            self._last_robot_status = resolved
            self.robot_status_changed.emit(resolved)
        return resolved

    @staticmethod
    def _status_details(status: object | None, fallback: str = "") -> str:
        if status is None:
            return fallback
        return str(
            getattr(status, "last_error", None)
            or getattr(status, "last_disarm_reason", None)
            or fallback
        )

    def _tick_robot_only(self) -> None:
        if self._robot is None:
            self._publish_readiness()
            return
        try:
            self._robot.tick()
        except Exception as error:
            self._emit_rate_limited(
                f"robot-tick:{type(error).__name__}",
                "ERROR",
                "ROBOT",
                "ROBOT_TICK_FAILED",
                "Ошибка watchdog реального выхода",
                str(error),
            )
        status = self._controller_status()
        state = self._controller_ui_state(status)
        details = self._status_details(status)
        self._publish_robot_state(state, details)
        if state is RobotUiState.DEGRADED:
            self._emit_rate_limited(
                f"robot-degraded:{details}",
                "ERROR",
                "ROBOT",
                "ROBOT_DEGRADED",
                "Реальный выход перешёл в отказобезопасное состояние",
                details,
            )
        self._publish_readiness()

    def _invalidate_robot(self, reason: str) -> None:
        if self._robot is None:
            self._publish_readiness()
            return
        status_before = self._controller_status()
        state_before = self._controller_ui_state(status_before)
        was_active = (
            state_before is RobotUiState.ARMED
            or self._last_robot_ui_state is RobotUiState.ARMING
        )
        if was_active:
            self._publish_robot_state(RobotUiState.DISARMING, reason, force=True)
        self._robot.invalidate(reason)
        status = self._controller_status()
        state = self._controller_ui_state(status)
        details = self._status_details(status, reason)
        self._publish_robot_state(state, details, force=was_active)
        if state_before is RobotUiState.ARMED:
            self._emit_rate_limited(
                f"robot-disarmed:{reason}",
                "WARNING",
                "ROBOT",
                "ROBOT_DISARMED",
                "Отправка на реального робота прекращена",
                reason,
            )
        self._publish_readiness(force=was_active)

    def _disconnect_robot(self, reason: str = "operator_disconnect") -> None:
        if self._robot is None:
            self._publish_robot_state(RobotUiState.DISCONNECTED, "", force=True)
            return
        self._publish_robot_state(RobotUiState.DISCONNECTING, reason, force=True)
        try:
            self._robot.disconnect(reason)
        except Exception as error:
            self._publish_robot_state(RobotUiState.DEGRADED, str(error), force=True)
            self._emit(
                "ERROR",
                "ROBOT",
                "ROBOT_DISCONNECT_FAILED",
                "Соединение с роботом не закрыто штатно",
                str(error),
            )
            return
        self._publish_robot_state(RobotUiState.DISCONNECTED, "", force=True)
        self._emit("INFO", "ROBOT", "ROBOT_DISCONNECTED", "Соединение закрыто")

    def _close_robot(self) -> None:
        controller, self._robot = self._robot, None
        if controller is not None:
            self._publish_robot_state(
                RobotUiState.DISCONNECTING, "controller_closed", force=True
            )
            try:
                controller.close()
            except Exception as error:
                self._emit(
                    "WARNING",
                    "ROBOT",
                    "ROBOT_CLOSE_FAILED",
                    "Ошибка закрытия контроллера робота",
                    str(error),
                )
        self._publish_robot_state(RobotUiState.DISCONNECTED, "", force=True)

    def _emit(
        self,
        severity: str,
        subsystem: str,
        event_code: str,
        message: str,
        details: str = "",
    ) -> None:
        sequence = (
            None
            if self._latest_snapshot is None
            else getattr(self._latest_snapshot, "sequence", None)
        )
        source_id = None if self._source is None else self._source.source_id
        recorder = self._recorder
        recorder_state = None if recorder is None else getattr(recorder, "state", None)
        if (
            recorder is not None
            and str(getattr(recorder_state, "value", recorder_state)).upper()
            == "RECORDING"
        ):
            try:
                recorder.record_event(
                    event_code,
                    sequence=sequence,
                    level=severity.lower(),
                    subsystem=subsystem.lower(),
                    message=message,
                    details={"text": details} if details else None,
                )
            except Exception:
                # Journal failures are reflected by recorder summary and must
                # never recurse through the worker's own event boundary.
                pass
        self.event_ready.emit(
            {
                "severity": severity,
                "subsystem": subsystem,
                "event_code": event_code,
                "message": message,
                "details": details,
                "run_id": self._active_run_id or "",
                "source_id": source_id or "",
                "sequence": sequence,
            }
        )
