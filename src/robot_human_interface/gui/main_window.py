"""Figma-derived 1366 x 768 Humanoid Interface main window."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from time import monotonic
from typing import Callable

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QCloseEvent, QResizeEvent
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .logs import LogEntry, LogPanel, QtLogHandler, build_rotating_file_handler
from .preview import PreviewWidget
from .resources import ResourceLocator, SourceItem, UserSourceStore
from .widgets import ArmConfirmationDialog, SourcePanel, StatusBadge, TelemetryPanel
from .worker import PipelineWorker


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RobotCallbacks:
    """Optional adapter to the guarded physical-robot controller."""

    connect: Callable[[], object] | None = None
    disconnect: Callable[[], object] | None = None
    arm: Callable[..., object] | None = None
    disarm: Callable[[], object] | None = None


class MainWindow(QMainWindow):
    """Operator console; all heavy processing remains in ``PipelineWorker``."""

    def __init__(
        self,
        *,
        locator: ResourceLocator | None = None,
        user_store: UserSourceStore | None = None,
        worker: PipelineWorker | None = None,
        robot_callbacks: RobotCallbacks | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Humanoid Interface")
        self.resize(1366, 768)
        self.setMinimumSize(1024, 640)
        self.locator = locator or ResourceLocator()
        self.user_store = user_store or UserSourceStore()
        self.worker = worker or PipelineWorker(self)
        if self.worker.parent() is None:
            self.worker.setParent(self)
        self.robot_callbacks = robot_callbacks or RobotCallbacks()
        self._log_dir = log_dir
        self._latest_snapshot: object | None = None
        self._snapshot_received_at = 0.0
        self._pipeline_state = "STOPPED"
        self._robot_state = "DISCONNECTED"
        self._viewer_open = False
        self._safety_flags: tuple[bool, bool] | None = None
        self._auto_collapsed_logs = False
        self._closing = False
        self._worker_stopped_confirmed = False
        self._logging_closed = False
        self._next_shutdown_retry_s = 0.0
        self._shutdown_retry_count = 0
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setInterval(250)
        self._shutdown_timer.timeout.connect(self._poll_worker_shutdown)
        self._build_ui()
        self._install_logging()
        self._connect_signals()
        self.worker.start()
        self.source_panel.select_initial()
        self.source_panel.request_all_probes()
        self._log("INFO", "GUI", "GUI_STARTED", "Интерфейс готов")
        stock_count = len(self.locator.stock_videos())
        if stock_count != 6:
            self._log(
                "WARNING",
                "SOURCE",
                "REFERENCE_CATALOG_INCOMPLETE",
                f"Найдено эталонных видео: {stock_count}; ожидалось 6",
            )

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_top_bar())

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 8)
        content_layout.setSpacing(12)
        self.source_panel = SourcePanel(self.locator, self.user_store)
        self.source_panel.setMinimumWidth(238)
        self.source_panel.setMaximumWidth(290)
        self.preview = PreviewWidget()
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.telemetry = TelemetryPanel()
        self.telemetry.setMinimumWidth(300)
        self.telemetry.setMaximumWidth(370)
        content_layout.addWidget(self.source_panel, 270)
        content_layout.addWidget(self.preview, 698)
        content_layout.addWidget(self.telemetry, 342)
        root_layout.addWidget(content, 1)

        log_wrap = QWidget()
        log_layout = QVBoxLayout(log_wrap)
        log_layout.setContentsMargins(12, 0, 12, 12)
        self.log_panel = LogPanel()
        self.log_panel.setMinimumHeight(150)
        self.log_panel.setMaximumHeight(184)
        log_layout.addWidget(self.log_panel)
        root_layout.addWidget(log_wrap)
        self.setCentralWidget(root)

    def _build_top_bar(self) -> QWidget:
        top = QWidget()
        top.setObjectName("topBar")
        top.setFixedHeight(64)
        layout = QHBoxLayout(top)
        layout.setContentsMargins(12, 8, 16, 8)
        layout.setSpacing(10)
        brand_box = QVBoxLayout()
        brand_box.setSpacing(0)
        brand = QLabel("HUMANOID")
        brand.setObjectName("brand")
        brand_accent = QLabel("INTERFACE")
        brand_accent.setObjectName("brandAccent")
        brand_box.addWidget(brand)
        brand_box.addWidget(brand_accent)
        brand_widget = QWidget()
        brand_widget.setLayout(brand_box)
        brand_widget.setFixedWidth(170)
        layout.addWidget(brand_widget)
        self.pipeline_badge = StatusBadge("PIPELINE OFF", "neutral")
        self.tracking_badge = StatusBadge("TRACKING 0%", "neutral")
        self.mujoco_badge = StatusBadge("MUJOCO READY", "info")
        self.robot_badge = StatusBadge("ROBOT OFF", "neutral")
        layout.addWidget(self.pipeline_badge)
        layout.addWidget(self.tracking_badge)
        layout.addWidget(self.mujoco_badge)
        layout.addWidget(self.robot_badge)
        layout.addStretch(1)
        self.viewer_button = QPushButton("Открыть MuJoCo")
        self.viewer_button.setEnabled(False)
        self.viewer_button.setMinimumWidth(136)
        self.start_button = QPushButton("Запустить")
        self.start_button.setProperty("primary", True)
        self.start_button.setMinimumWidth(146)
        layout.addWidget(self.viewer_button)
        layout.addWidget(self.start_button)
        return top

    def _install_logging(self) -> None:
        self._qt_log_handler = QtLogHandler(self)
        self._qt_log_handler.bridge.entry_ready.connect(self.log_panel.append)
        self._file_log_handler = build_rotating_file_handler(self._log_dir)
        root_logger = logging.getLogger()
        root_logger.addHandler(self._qt_log_handler)
        root_logger.addHandler(self._file_log_handler)

    def _connect_signals(self) -> None:
        self.source_panel.source_selected.connect(self._select_source)
        self.source_panel.probe_requested.connect(self.worker.probe_video)
        self.start_button.clicked.connect(self._toggle_pipeline)
        self.viewer_button.clicked.connect(self._toggle_viewer)
        self.telemetry.reset_requested.connect(self.worker.reset_pipeline)
        self.telemetry.calibrate_requested.connect(self.worker.calibrate)
        self.telemetry.pause_requested.connect(self._toggle_pause)
        self.telemetry.robot_connect_requested.connect(self._toggle_robot_connection)
        self.telemetry.robot_arm_changed.connect(self._on_robot_interlock)
        self.worker.snapshot_ready.connect(self._on_snapshot)
        self.worker.event_ready.connect(self._on_event)
        self.worker.state_changed.connect(self._on_pipeline_state)
        self.worker.viewer_changed.connect(self._on_viewer_state)
        self.worker.robot_state_changed.connect(self.set_robot_state)
        self.worker.safety_flags_changed.connect(self._on_safety_flags)
        self.worker.video_metadata_ready.connect(self.source_panel.apply_video_metadata)
        self.worker.finished.connect(self._on_worker_finished)

    def set_robot_callbacks(self, callbacks: RobotCallbacks) -> None:
        self.robot_callbacks = callbacks

    def set_robot_state(self, state: str, details: str = "") -> None:
        state = state.upper()
        if state not in {"DISCONNECTED", "CONNECTED_DISARMED", "ARMED", "DEGRADED"}:
            raise ValueError(f"unsupported robot state: {state}")
        if state in {"DISCONNECTED", "CONNECTED_DISARMED", "DEGRADED"}:
            self.telemetry.set_interlock_checked(False)
        self._robot_state = state
        self.telemetry.set_robot_state(state)
        if state == "ARMED":
            self.robot_badge.set_status("ROBOT ARMED", "success")
        elif state == "DEGRADED":
            self.robot_badge.set_status("ROBOT DEGRADED", "critical")
        elif state == "CONNECTED_DISARMED":
            self.robot_badge.set_status("ROBOT READY", "info")
        else:
            self.robot_badge.set_status("ROBOT OFF", "neutral")
        if details:
            self._log(
                "ERROR" if state == "DEGRADED" else "INFO",
                "ROBOT",
                f"ROBOT_{state}",
                "Состояние реального робота изменено",
                details,
            )

    def _select_source(self, source: SourceItem) -> None:
        self._force_robot_off("SOURCE_CHANGED", "Смена источника прекращает отправку")
        self.preview.set_source_label(source.title)
        self._safety_flags = None
        self.worker.select_source(source)
        if not source.available:
            self._log("ERROR", "SOURCE", "SOURCE_UNAVAILABLE", "Файл источника недоступен", source.path or "")

    def _toggle_pipeline(self) -> None:
        if self._pipeline_state in {"RUNNING", "PAUSED"}:
            self._force_robot_off("PIPELINE_STOPPING", "Остановка pipeline прекращает отправку")
            self.worker.stop_pipeline()
        else:
            self.worker.start_pipeline()

    def _toggle_pause(self) -> None:
        if self._pipeline_state == "RUNNING":
            self.worker.pause_pipeline()
        elif self._pipeline_state == "PAUSED":
            self.worker.resume_pipeline()

    def _toggle_viewer(self) -> None:
        if self._pipeline_state not in {"RUNNING", "PAUSED"}:
            self._log("WARNING", "MUJOCO", "VIEWER_BLOCKED", "Сначала запустите pipeline")
            return
        self.worker.set_viewer_open(not self._viewer_open)

    def _on_pipeline_state(self, state: str) -> None:
        self._pipeline_state = state.upper()
        if self._pipeline_state == "RUNNING":
            self.pipeline_badge.set_status("PIPELINE", "success")
            self.start_button.setText("Остановить")
            self.start_button.setProperty("primary", False)
            self.viewer_button.setEnabled(True)
            self.telemetry.pause_button.setEnabled(True)
            self.telemetry.pause_button.setText("Пауза")
        elif self._pipeline_state == "PAUSED":
            self.pipeline_badge.set_status("PIPELINE PAUSED", "warning")
            self.start_button.setText("Остановить")
            self.start_button.setProperty("primary", False)
            self.viewer_button.setEnabled(True)
            self.telemetry.pause_button.setEnabled(True)
            self.telemetry.pause_button.setText("Продолжить")
        elif self._pipeline_state == "DEGRADED":
            self.pipeline_badge.set_status("PIPELINE ERROR", "critical")
            self.start_button.setText("Перезапустить")
            self._force_robot_off("PIPELINE_DEGRADED", "Ошибка pipeline прекращает отправку")
            self.viewer_button.setEnabled(False)
            self.telemetry.pause_button.setEnabled(False)
        else:
            self.pipeline_badge.set_status("PIPELINE OFF", "neutral")
            self.start_button.setText("Запустить")
            self.start_button.setProperty("primary", True)
            self.viewer_button.setEnabled(False)
            self.telemetry.pause_button.setEnabled(False)
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)

    def _on_viewer_state(self, opened: bool) -> None:
        self._viewer_open = bool(opened)
        self.viewer_button.setText("Закрыть MuJoCo" if opened else "Открыть MuJoCo")
        self.mujoco_badge.set_status("MUJOCO OPEN" if opened else "MUJOCO READY", "success" if opened else "info")

    def _on_snapshot(self, snapshot: object) -> None:
        self._latest_snapshot = snapshot
        self._snapshot_received_at = monotonic()
        self.preview.set_snapshot(snapshot)
        self.telemetry.update_snapshot(snapshot)
        quality = float(getattr(snapshot, "tracking_quality", 0.0) or 0.0)
        kind = "success" if quality >= .7 else ("warning" if quality >= .4 else "critical")
        self.tracking_badge.set_status(f"TRACKING {round(quality * 100)}%", kind)

    def _on_safety_flags(self, known: bool, free_base: bool, balance: bool) -> None:
        self._safety_flags = (bool(free_base), bool(balance)) if known else None
        self.telemetry.set_safety_flags(known, free_base, balance)

    def _on_event(self, raw: object) -> None:
        if isinstance(raw, LogEntry):
            self.log_panel.append(raw)
            return
        if isinstance(raw, dict):
            self._log(
                str(raw.get("severity", "INFO")),
                str(raw.get("subsystem", "PIPELINE")),
                str(raw.get("event_code", "SESSION_EVENT")),
                str(raw.get("message", "Событие сессии")),
                str(raw.get("details", "")),
            )
            return
        severity = getattr(raw, "severity", getattr(raw, "level", "INFO"))
        severity = getattr(severity, "value", severity)
        subsystem = getattr(raw, "subsystem", "PIPELINE")
        subsystem = getattr(subsystem, "value", subsystem)
        code = getattr(raw, "code", getattr(raw, "event_code", "SESSION_EVENT"))
        code = getattr(code, "value", code)
        self._log(
            str(severity),
            str(subsystem),
            str(code),
            str(getattr(raw, "message", getattr(raw, "message_ru", raw))),
            str(getattr(raw, "details", "")),
        )

    def _toggle_robot_connection(self) -> None:
        uses_external_callbacks = any(
            callback is not None
            for callback in (
                self.robot_callbacks.connect,
                self.robot_callbacks.disconnect,
                self.robot_callbacks.arm,
                self.robot_callbacks.disarm,
            )
        )
        if not uses_external_callbacks:
            if self._robot_state == "DISCONNECTED":
                self.worker.connect_robot()
            else:
                self._force_robot_off("ROBOT_DISCONNECT", "Оператор отключил соединение")
                self.worker.disconnect_robot()
            return
        if self._robot_state == "DISCONNECTED":
            try:
                if self.robot_callbacks.connect is not None:
                    result = self.robot_callbacks.connect()
                    if result is False:
                        raise RuntimeError("controller rejected connection")
                self.set_robot_state("CONNECTED_DISARMED")
                self._log("INFO", "ROBOT", "ROBOT_CONNECTED_DISARMED", "Соединение установлено; команды не отправляются")
            except Exception as error:
                self.set_robot_state("DEGRADED", str(error))
        else:
            self._force_robot_off("ROBOT_DISCONNECT", "Оператор отключил соединение")
            try:
                if self.robot_callbacks.disconnect is not None:
                    self.robot_callbacks.disconnect()
            except Exception as error:
                self._log("WARNING", "ROBOT", "ROBOT_DISCONNECT_FAILED", "Ошибка закрытия соединения", str(error))
            self.set_robot_state("DISCONNECTED")

    def _on_robot_interlock(self, enabled: bool) -> None:
        if not enabled:
            self._force_robot_off("ROBOT_DISARMED", "Отправка выключена оператором")
            return
        reason = self._arm_block_reason()
        if reason:
            self.telemetry.set_interlock_checked(False)
            self._log("WARNING", "ROBOT", "ROBOT_ARM_BLOCKED", "Включение отправки заблокировано", reason)
            QMessageBox.warning(self, "Отправка заблокирована", reason)
            return
        dialog = ArmConfirmationDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.telemetry.set_interlock_checked(False)
            return
        try:
            uses_external_callbacks = self.robot_callbacks.arm is not None
            if uses_external_callbacks:
                try:
                    result = self.robot_callbacks.arm(send_velocities=dialog.send_velocities)
                except TypeError:
                    result = self.robot_callbacks.arm(dialog.send_velocities)
                if result is False:
                    raise RuntimeError("controller rejected arm")
                self.set_robot_state("ARMED")
                self.telemetry.set_interlock_checked(True)
                self._log("WARNING", "ROBOT", "ROBOT_ARMED", "Отправка safe_command разрешена оператором", f"setVelocities={dialog.send_velocities}")
            else:
                self.telemetry.set_interlock_checked(False)
                self.worker.arm_robot(send_velocities=dialog.send_velocities)
        except Exception as error:
            self.telemetry.set_interlock_checked(False)
            self.set_robot_state("DEGRADED", str(error))

    def _arm_block_reason(self) -> str:
        if self._robot_state != "CONNECTED_DISARMED":
            return "Сначала установите соединение с роботом."
        if self._latest_snapshot is None or monotonic() - self._snapshot_received_at > .5:
            return "Нет свежей safe-команды (допустимый возраст — 500 мс)."
        snapshot = self._latest_snapshot
        safe_valid = getattr(snapshot, "safe_valid", None)
        safe_command = getattr(snapshot, "safe_command", None)
        if safe_valid is None:
            safe_valid = safe_command is not None and not bool(getattr(safe_command, "stale", False))
            if hasattr(snapshot, "angles_rad"):
                safe_valid = bool(getattr(snapshot, "safe_valid", True))
        if not safe_valid:
            return "Safe-команда отсутствует, устарела или не прошла ограничения."
        if self._safety_flags is None:
            return "Нет подтверждённого происхождения режимов balance/free-base."
        free_base, balance = self._safety_flags
        if not balance or not free_base:
            return "Для реального выхода требуются активные balance и free-base режимы."
        return ""

    def _force_robot_off(self, event_code: str, message: str) -> None:
        was_armed = self._robot_state == "ARMED" or self.telemetry.robot_interlock.isChecked()
        self.telemetry.set_interlock_checked(False)
        if was_armed and self.robot_callbacks.disarm is not None:
            try:
                self.robot_callbacks.disarm()
            except Exception as error:
                self.set_robot_state("DEGRADED", str(error))
                return
        elif was_armed:
            self.worker.disarm_robot(event_code.casefold())
        if was_armed:
            self.set_robot_state("CONNECTED_DISARMED")
            self._log("WARNING", "ROBOT", event_code, message)

    def _log(
        self,
        severity: str,
        subsystem: str,
        event_code: str,
        message: str,
        details: str = "",
    ) -> None:
        self.log_panel.append(LogEntry.now(severity, subsystem, event_code, message, details))
        level = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }.get(severity.upper(), logging.INFO)
        record = logging.LogRecord(
            name=f"humanoid.{subsystem.lower()}",
            level=level,
            pathname="",
            lineno=0,
            msg=f"{event_code} {message} {details}".strip(),
            args=(),
            exc_info=None,
        )
        self._file_log_handler.handle(record)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.height() < 710 and not self.log_panel.collapsed:
            self._auto_collapsed_logs = True
            self.log_panel.set_collapsed(True)
        elif self.height() >= 740 and self._auto_collapsed_logs and self.log_panel.collapsed:
            self._auto_collapsed_logs = False
            self.log_panel.set_collapsed(False)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        # A QThread must never be destroyed while running. Only a confirmed
        # stopped worker is allowed through the final close event.
        if self._worker_stopped_confirmed or not self.worker.isRunning():
            self._worker_stopped_confirmed = True
            self._shutdown_timer.stop()
            self._close_logging_once()
            event.accept()
            return

        event.ignore()
        if self._closing:
            # Repeated OS/user close requests remain pending. Accepting here
            # would destroy the window (and its child QThread) too early.
            return

        self._closing = True
        self._next_shutdown_retry_s = monotonic() + 4.0
        self._shutdown_retry_count = 0
        self._force_robot_off("APP_CLOSING", "Закрытие приложения прекращает отправку")
        try:
            if self.robot_callbacks.disconnect is not None and self._robot_state != "DISCONNECTED":
                self.robot_callbacks.disconnect()
        except Exception as error:
            self._log("WARNING", "ROBOT", "ROBOT_DISCONNECT_FAILED", "Ошибка закрытия соединения", str(error))
        self._log(
            "INFO",
            "GUI",
            "APP_SHUTDOWN_PENDING",
            "Ожидание безопасного завершения worker",
        )
        self.start_button.setText("Завершение…")
        central = self.centralWidget()
        if central is not None:
            central.setEnabled(False)
        try:
            self.worker.request_shutdown()
        except Exception as error:
            self._log(
                "ERROR",
                "PIPELINE",
                "WORKER_SHUTDOWN_REQUEST_FAILED",
                "Не удалось передать запрос завершения worker",
                str(error),
            )
        self._shutdown_timer.start()

    def _on_worker_finished(self) -> None:
        """Finish a pending close only after Qt reports QThread.finished."""

        if not self._closing:
            self._worker_stopped_confirmed = True
            return
        # ``finished`` may be emitted just before isRunning() flips to false;
        # queue the authoritative poll on the GUI event loop.
        QTimer.singleShot(0, self._poll_worker_shutdown)

    def _poll_worker_shutdown(self) -> None:
        if not self._closing:
            self._shutdown_timer.stop()
            return
        if not self.worker.isRunning():
            self._worker_stopped_confirmed = True
            self._shutdown_timer.stop()
            self._close_logging_once()
            # Generate a new close event. The guarded branch above is now the
            # only branch allowed to accept it.
            QTimer.singleShot(0, self.close)
            return

        now = monotonic()
        if now < self._next_shutdown_retry_s:
            return
        self._shutdown_retry_count += 1
        self._next_shutdown_retry_s = now + 4.0
        self._log(
            "WARNING",
            "PIPELINE",
            "WORKER_SHUTDOWN_DELAYED",
            "Worker ещё завершает ресурсы; закрытие окна остаётся отложенным",
            f"retry={self._shutdown_retry_count}",
        )
        LOGGER.warning(
            "pipeline worker shutdown is delayed; retry=%d",
            self._shutdown_retry_count,
        )
        # Reissue the cooperative request. Never call terminate(): camera,
        # MediaPipe, MuJoCo and WebSocket cleanup must run in their owner.
        try:
            self.worker.request_shutdown()
        except Exception as error:
            self._log(
                "ERROR",
                "PIPELINE",
                "WORKER_SHUTDOWN_RETRY_FAILED",
                "Повторный запрос завершения worker не передан",
                str(error),
            )

    def _close_logging_once(self) -> None:
        if self._logging_closed:
            return
        self._logging_closed = True
        root_logger = logging.getLogger()
        root_logger.removeHandler(self._qt_log_handler)
        root_logger.removeHandler(self._file_log_handler)
        self._file_log_handler.close()
