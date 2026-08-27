"""Figma-derived 1366 x 768 Humanoid Interface main window."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
import logging
from pathlib import Path
from time import monotonic
from typing import Callable

from PyQt6.QtCore import QSettings, QStandardPaths, Qt, QTimer, QUrl
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QResizeEvent
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .logs import LogEntry, LogPanel, QtLogHandler, build_rotating_file_handler
from .preview import PreviewWidget
from .research_widgets import (
    ExperimentPanel,
    PlaybackBar,
    ReadinessChecklist,
    SystemBanner,
    SystemBannerState,
    TelemetrySparkline,
)
from .resources import ResourceLocator, SourceItem, UserSourceStore
from .runtime import RobotReadiness, RobotUiState, RuntimeMode, RuntimeStatus
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
    status: Callable[[], object] | None = None


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
        self._settings = QSettings()
        self._log_dir = log_dir
        self._latest_snapshot: object | None = None
        self._playback_state: object | None = None
        self._snapshot_received_at = 0.0
        self._pipeline_state = "STOPPED"
        self._robot_state = "DISCONNECTED"
        self._robot_status: object | None = None
        self._runtime_status = RuntimeStatus.demo("backend ещё не запущен")
        self._robot_readiness: RobotReadiness | None = None
        self._viewer_open = False
        self._viewer_state = "UNAVAILABLE"
        self._safety_flags: tuple[bool, bool] | None = None
        self._auto_collapsed_logs = False
        self._compact = False
        self._stale_active = False
        self._last_external_status_poll_s = 0.0
        self._recording_active = False
        self._active_run_id: str | None = None
        self._last_experiment_path: Path | None = None
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
        self._restore_settings()
        self._watchdog_timer = QTimer(self)
        self._watchdog_timer.setInterval(100)
        self._watchdog_timer.timeout.connect(self._watchdog_tick)
        self._watchdog_timer.start()
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
        self.system_banner = SystemBanner()
        root_layout.addWidget(self.system_banner)

        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.setChildrenCollapsible(False)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 6)
        content_layout.setSpacing(8)
        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.source_panel = SourcePanel(self.locator, self.user_store)
        self.source_panel.setMinimumWidth(185)
        self.preview = PreviewWidget()
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(7)
        center_layout.addWidget(self.preview, 1)
        self.playback_bar = PlaybackBar()
        center_layout.addWidget(self.playback_bar)
        self.telemetry = TelemetryPanel()
        self.telemetry.setMinimumWidth(300)
        self.readiness = ReadinessChecklist()
        self.experiment_panel = ExperimentPanel()
        self.telemetry.tabs.addTab(self.readiness, "Робот")
        self.telemetry.tabs.addTab(self.experiment_panel, "Эксперимент")
        self._install_balance_charts()
        self.workspace_splitter.addWidget(self.source_panel)
        self.workspace_splitter.addWidget(center)
        self.workspace_splitter.addWidget(self.telemetry)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setStretchFactor(2, 0)
        self.workspace_splitter.setSizes((238, 720, 342))
        content_layout.addWidget(self.workspace_splitter, 1)
        self.vertical_splitter.addWidget(content)

        log_wrap = QWidget()
        log_layout = QVBoxLayout(log_wrap)
        log_layout.setContentsMargins(12, 0, 12, 12)
        self.log_panel = LogPanel()
        self.log_panel.setMinimumHeight(150)
        log_layout.addWidget(self.log_panel)
        self.vertical_splitter.addWidget(log_wrap)
        self.vertical_splitter.setStretchFactor(0, 1)
        self.vertical_splitter.setStretchFactor(1, 0)
        self.vertical_splitter.setSizes((548, 178))
        root_layout.addWidget(self.vertical_splitter, 1)
        self.setCentralWidget(root)

    def _install_balance_charts(self) -> None:
        charts = QHBoxLayout()
        self.tilt_chart = TelemetrySparkline(
            title="Наклон базы · 10 с", unit="rad", window_s=10.0
        )
        self.force_chart = TelemetrySparkline(
            title="Сила стоп · 10 с", unit="N", window_s=10.0
        )
        self.tilt_chart.setToolTip("Наклон базы за 10 секунд")
        self.force_chart.setToolTip("Суммарная сила стоп за 10 секунд")
        charts.addWidget(self.tilt_chart)
        charts.addWidget(self.force_chart)
        # Insert above the action-row stretch while keeping TelemetryPanel's
        # public surface backwards compatible.
        balance_page = self.telemetry.tabs.widget(1)
        if balance_page is not None and balance_page.layout() is not None:
            balance_page.layout().insertLayout(
                max(0, balance_page.layout().count() - 2), charts
            )

    def _build_top_bar(self) -> QWidget:
        top = QWidget()
        top.setObjectName("topBar")
        top.setFixedHeight(56)
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
        self.brand_widget = QWidget()
        self.brand_widget.setLayout(brand_box)
        self.brand_widget.setFixedWidth(170)
        layout.addWidget(self.brand_widget)
        self.pipeline_badge = StatusBadge("PIPELINE OFF", "neutral")
        self.tracking_badge = StatusBadge("TRACKING 0%", "neutral")
        self.mujoco_badge = StatusBadge("MUJOCO · НЕ ПРОВЕРЕНО", "neutral")
        self.robot_badge = StatusBadge("ROBOT OFF", "neutral")
        layout.addWidget(self.pipeline_badge)
        layout.addWidget(self.tracking_badge)
        layout.addWidget(self.mujoco_badge)
        layout.addWidget(self.robot_badge)
        layout.addStretch(1)
        self.sources_toggle = QToolButton()
        self.sources_toggle.setText("Источники")
        self.sources_toggle.setCheckable(True)
        self.sources_toggle.setChecked(True)
        self.sources_toggle.setVisible(False)
        self.sources_toggle.toggled.connect(self._toggle_sources_panel)
        layout.addWidget(self.sources_toggle)
        self.viewer_button = QPushButton("Открыть MuJoCo")
        self.viewer_button.setEnabled(False)
        self.viewer_button.setMinimumWidth(136)
        self.start_button = QPushButton("Запустить")
        self.start_button.setProperty("primary", True)
        self.start_button.setMinimumWidth(146)
        layout.addWidget(self.viewer_button)
        layout.addWidget(self.start_button)
        return top

    def _toggle_sources_panel(self, visible: bool) -> None:
        panel = getattr(self, "source_panel", None)
        if panel is not None:
            panel.setVisible(bool(visible))

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
        self.telemetry.reset_requested.connect(self._request_reset)
        self.telemetry.calibrate_requested.connect(self._request_calibration)
        self.telemetry.pause_requested.connect(self._toggle_pause)
        self.telemetry.robot_connect_requested.connect(self._toggle_robot_connection)
        self.telemetry.robot_arm_changed.connect(self._on_robot_interlock)
        self.system_banner.stop_sending_requested.connect(
            lambda: self._force_robot_off(
                "OPERATOR_STOP_SENDING",
                "Оператор остановил отправку — это не аппаратный E-stop",
            )
        )
        self.playback_bar.play_requested.connect(self._playback_play)
        self.playback_bar.pause_requested.connect(self._playback_pause)
        self.playback_bar.step_requested.connect(self._playback_step)
        self.playback_bar.seek_requested.connect(self._playback_seek)
        self.playback_bar.rate_requested.connect(self._playback_rate)
        self.playback_bar.loop_requested.connect(self._playback_loop)
        self.experiment_panel.start_requested.connect(self.worker.start_recording)
        self.experiment_panel.stop_requested.connect(
            lambda: self.worker.stop_recording("manual")
        )
        self.experiment_panel.open_directory_requested.connect(
            self._open_experiment_directory
        )
        self.worker.snapshot_ready.connect(self._on_snapshot)
        self.worker.event_ready.connect(self._on_event)
        self.worker.state_changed.connect(self._on_pipeline_state)
        self.worker.viewer_changed.connect(self._on_viewer_state)
        if hasattr(self.worker, "viewer_status_changed"):
            self.worker.viewer_status_changed.connect(self._on_viewer_status)
        self.worker.robot_state_changed.connect(self.set_robot_state)
        if hasattr(self.worker, "robot_status_changed"):
            self.worker.robot_status_changed.connect(self._on_robot_status)
        if hasattr(self.worker, "runtime_status_changed"):
            self.worker.runtime_status_changed.connect(self._on_runtime_status)
        if hasattr(self.worker, "robot_readiness_changed"):
            self.worker.robot_readiness_changed.connect(self._on_robot_readiness)
        self.worker.safety_flags_changed.connect(self._on_safety_flags)
        if hasattr(self.worker, "recorder_state_changed"):
            self.worker.recorder_state_changed.connect(self._on_recorder_state)
        if hasattr(self.worker, "recorder_progress"):
            self.worker.recorder_progress.connect(self._on_recorder_progress)
        if hasattr(self.worker, "experiment_completed"):
            self.worker.experiment_completed.connect(self._on_experiment_complete)
        self.worker.video_metadata_ready.connect(self.source_panel.apply_video_metadata)
        self.worker.finished.connect(self._on_worker_finished)

    def set_robot_callbacks(self, callbacks: RobotCallbacks) -> None:
        self.robot_callbacks = callbacks

    def set_robot_state(self, state: str, details: str = "") -> None:
        state = state.upper()
        if state not in {
            "DISCONNECTED",
            "CONNECTING",
            "CONNECTED_DISARMED",
            "ARMING",
            "ARMED",
            "DISARMING",
            "DISCONNECTING",
            "DEGRADED",
        }:
            raise ValueError(f"unsupported robot state: {state}")
        if state in {"DISCONNECTED", "CONNECTED_DISARMED", "DEGRADED", "DISARMING", "DISCONNECTING"}:
            self.telemetry.set_interlock_checked(False)
        self._robot_state = state
        self.telemetry.set_robot_state(state)
        if state == "ARMED":
            self.robot_badge.set_status("ROBOT ARMED", "warning")
            self._refresh_system_banner()
        elif state == "DEGRADED":
            self.robot_badge.set_status("ROBOT DEGRADED", "critical")
            self.system_banner.show_error("Реальный выход перешёл в DEGRADED", details)
        elif state == "CONNECTED_DISARMED":
            self.robot_badge.set_status("ROBOT READY", "info")
            self._refresh_system_banner()
        elif state in {"CONNECTING", "ARMING", "DISARMING", "DISCONNECTING"}:
            labels = {
                "CONNECTING": "ROBOT CONNECTING",
                "ARMING": "ROBOT ARMING",
                "DISARMING": "ROBOT DISARMING",
                "DISCONNECTING": "ROBOT DISCONNECTING",
            }
            self.robot_badge.set_status(labels[state], "warning")
        else:
            self.robot_badge.set_status("ROBOT OFF", "neutral")
            self._refresh_system_banner()
        if details:
            self._log(
                "ERROR" if state == "DEGRADED" else "INFO",
                "ROBOT",
                f"ROBOT_{state}",
                "Состояние реального робота изменено",
                details,
            )
        self._update_robot_controls()

    def _select_source(self, source: SourceItem) -> None:
        if self._recording_active:
            self._log("WARNING", "RECORDER", "SOURCE_CHANGE_BLOCKED_RECORDING", "Завершите запись опыта перед сменой источника")
            return
        self._force_robot_off("SOURCE_CHANGED", "Смена источника прекращает отправку")
        self.preview.set_source_label(source.title)
        self.preview.clear(overlay="ИСТОЧНИК ВЫБРАН · НАЖМИТЕ «ЗАПУСТИТЬ»")
        self.telemetry.clear_snapshot()
        self.playback_bar.set_playback_state(
            None,
            session_state="STOPPED",
            live=source.kind == "camera",
            enabled=True,
        )
        self._latest_snapshot = None
        self._playback_state = None
        self._snapshot_received_at = 0.0
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

    def _request_reset(self) -> None:
        if self._recording_active:
            self._log("WARNING", "RECORDER", "RESET_BLOCKED_RECORDING", "Сброс заблокирован во время записи опыта")
            return
        self._force_robot_off("PIPELINE_RESET", "Сброс прекращает отправку")
        self.preview.clear(keep_frame=True, overlay="СБРОС ВРЕМЕННОГО СОСТОЯНИЯ…")
        self.telemetry.clear_snapshot()
        self.worker.reset_pipeline()

    def _request_calibration(self) -> None:
        if self._recording_active:
            self._log("WARNING", "RECORDER", "CALIBRATION_BLOCKED_RECORDING", "Калибровка заблокирована во время записи опыта")
            return
        self._force_robot_off("CALIBRATION_STARTED", "Калибровка прекращает отправку")
        self.preview.clear(keep_frame=True, overlay="КАЛИБРОВКА · СТОЙТЕ В ИСХОДНОЙ ПОЗЕ")
        self.telemetry.clear_snapshot()
        self.worker.calibrate()

    def _playback_play(self) -> None:
        state = self.playback_bar.playback_state
        if state is not None and state.eof:
            self._force_robot_off("PLAYBACK_RESTART", "Переход к началу прекращает отправку")
            self.worker.seek(0.0)
            return
        if self._pipeline_state == "PAUSED":
            self.worker.resume_pipeline()
        elif self._pipeline_state not in {"RUNNING"}:
            self.worker.start_pipeline()

    def _playback_pause(self) -> None:
        if self._pipeline_state == "RUNNING":
            self.worker.pause_pipeline()

    def _playback_step(self, delta: int) -> None:
        self._force_robot_off("PLAYBACK_STEP", "Шаг по кадру прекращает отправку")
        self.worker.step_frame(int(delta))

    def _playback_seek(self, position_s: float) -> None:
        self._force_robot_off("PLAYBACK_SEEK", "Переход по timeline прекращает отправку")
        self.preview.set_overlay("ПЕРЕХОД ПО ВИДЕО…")
        self.worker.seek(float(position_s))

    def _playback_rate(self, rate: float) -> None:
        self._force_robot_off("PLAYBACK_RATE_CHANGED", "Смена скорости прекращает отправку")
        self.worker.set_playback_rate(float(rate))

    def _playback_loop(self, enabled: bool, start_s: float, end_s: float) -> None:
        if self._robot_state in {"ARMED", "ARMING"}:
            self._log("WARNING", "PLAYBACK", "PLAYBACK_LOOP_BLOCKED_ARMED", "Сначала остановите отправку на реального робота")
            self._refresh_playback_bar()
            return
        self._force_robot_off("PLAYBACK_LOOP_CHANGED", "Изменение цикла прекращает отправку")
        self.worker.set_playback_loop(bool(enabled), float(start_s), float(end_s))

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
            self.viewer_button.setEnabled(self._viewer_state in {"READY", "OPEN", "ERROR"})
            self.telemetry.pause_button.setEnabled(True)
            self.telemetry.pause_button.setText("Пауза")
            self.preview.set_overlay("")
        elif self._pipeline_state == "PAUSED":
            self.pipeline_badge.set_status("PIPELINE PAUSED", "warning")
            self.start_button.setText("Остановить")
            self.start_button.setProperty("primary", False)
            self.viewer_button.setEnabled(self._viewer_state in {"READY", "OPEN", "ERROR"})
            self.telemetry.pause_button.setEnabled(True)
            self.telemetry.pause_button.setText("Продолжить")
            self.preview.clear(
                keep_frame=True,
                overlay="ПАУЗА · ДАННЫЕ НЕ СВЕЖИЕ",
            )
        elif self._pipeline_state == "DEGRADED":
            self.pipeline_badge.set_status("PIPELINE ERROR", "critical")
            self.start_button.setText("Перезапустить")
            self._force_robot_off("PIPELINE_DEGRADED", "Ошибка pipeline прекращает отправку")
            self.viewer_button.setEnabled(False)
            self.telemetry.pause_button.setEnabled(False)
            self._clear_runtime_data("ОШИБКА PIPELINE · СМОТРИТЕ ЖУРНАЛ")
            self.system_banner.show_error("Ошибка pipeline", "Смотрите журнал событий")
        else:
            self.pipeline_badge.set_status("PIPELINE OFF", "neutral")
            self.start_button.setText("Запустить")
            self.start_button.setProperty("primary", True)
            self.viewer_button.setEnabled(False)
            self.telemetry.pause_button.setEnabled(False)
            self._clear_runtime_data("PIPELINE ОСТАНОВЛЕН")
            self._viewer_state = "UNAVAILABLE"
            self.mujoco_badge.set_status("MUJOCO UNAVAILABLE", "neutral")
        self._update_recording_gate()
        self._refresh_playback_bar()
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)

    def _on_viewer_state(self, opened: bool) -> None:
        """Compatibility handler for embedders that only expose a bool signal."""

        self._viewer_open = bool(opened)
        if hasattr(self.worker, "viewer_status_changed"):
            return
        self.viewer_button.setText("Закрыть MuJoCo" if opened else "Открыть MuJoCo")
        if opened:
            self.mujoco_badge.set_status("MUJOCO OPEN", "success")
        elif self._pipeline_state in {"RUNNING", "PAUSED"}:
            self.mujoco_badge.set_status("MUJOCO READY", "info")
        else:
            self.mujoco_badge.set_status("MUJOCO UNAVAILABLE", "neutral")

    def _on_viewer_status(self, state: str, details: str = "") -> None:
        normalized = str(state).upper()
        if normalized not in {"UNAVAILABLE", "INITIALIZING", "READY", "OPEN", "ERROR"}:
            self._log(
                "ERROR",
                "MUJOCO",
                "VIEWER_STATE_INVALID",
                "Получено неизвестное состояние MuJoCo viewer",
                normalized,
            )
            return
        self._viewer_state = normalized
        self._viewer_open = normalized == "OPEN"
        labels = {
            "UNAVAILABLE": ("MUJOCO UNAVAILABLE", "neutral"),
            "INITIALIZING": ("MUJOCO INITIALIZING", "warning"),
            "READY": ("MUJOCO READY", "info"),
            "OPEN": ("MUJOCO OPEN", "success"),
            "ERROR": ("MUJOCO ERROR", "critical"),
        }
        label, kind = labels[normalized]
        self.mujoco_badge.set_status(label, kind)
        self.viewer_button.setText(
            "Закрыть MuJoCo"
            if normalized == "OPEN"
            else ("Повторить MuJoCo" if normalized == "ERROR" else "Открыть MuJoCo")
        )
        self.viewer_button.setEnabled(
            normalized in {"READY", "OPEN", "ERROR"}
            and self._pipeline_state in {"RUNNING", "PAUSED"}
        )
        if details and normalized == "ERROR":
            self._log(
                "ERROR",
                "MUJOCO",
                "VIEWER_ERROR",
                "MuJoCo viewer завершился ошибкой",
                details,
            )

    def _on_snapshot(self, snapshot: object) -> None:
        self._latest_snapshot = snapshot
        self._snapshot_received_at = monotonic()
        self._stale_active = False
        raw_status = getattr(snapshot, "status", "running")
        status = str(getattr(raw_status, "value", raw_status)).upper()
        playback = getattr(snapshot, "playback", None)
        self._playback_state = playback
        source = getattr(snapshot, "source", None)
        source_kind = str(getattr(getattr(source, "kind", None), "value", getattr(source, "kind", ""))).lower()
        is_live = source_kind in {"camera", "live_camera"} or (
            self.source_panel.current_source is not None
            and self.source_panel.current_source.kind == "camera"
        )
        self.playback_bar.set_playback_state(
            playback,
            session_state=status,
            live=is_live,
            enabled=True,
            locked=self._recording_active,
        )
        if status in {"ENDED", "STOPPED", "CLOSED", "ERROR"} or bool(getattr(playback, "eof", False)):
            self._clear_runtime_data(
                "КОНЕЦ ВИДЕО" if status == "ENDED" or bool(getattr(playback, "eof", False)) else "ДАННЫЕ НЕДОСТУПНЫ"
            )
            self._update_recording_gate()
            return
        self.preview.set_snapshot(snapshot)
        self.telemetry.update_snapshot(snapshot)
        quality = float(getattr(snapshot, "tracking_quality", 0.0) or 0.0)
        kind = "success" if quality >= .7 else ("warning" if quality >= .4 else "critical")
        self.tracking_badge.set_status(f"TRACKING {round(quality * 100)}%", kind)
        telemetry = getattr(snapshot, "telemetry", {}) or {}
        if isinstance(telemetry, dict) or hasattr(telemetry, "get"):
            timestamp = monotonic()
            try:
                self.tilt_chart.append(float(telemetry.get("base_tilt_rad")), timestamp)
            except (TypeError, ValueError):
                pass
        if self._pipeline_state == "PAUSED":
            self.preview.set_overlay("ПАУЗА · ДАННЫЕ НЕ СВЕЖИЕ")
        elif bool(getattr(telemetry, "get", lambda *_: False)("calibrating", False)):
            try:
                progress = float(telemetry.get("calibration_progress", 0.0))
                suffix = f" · {round(progress * 100)}%"
            except (TypeError, ValueError):
                suffix = ""
            self.preview.set_overlay(
                f"КАЛИБРОВКА · СТОЙТЕ В ИСХОДНОЙ ПОЗЕ{suffix}"
            )
        else:
            self.preview.set_overlay("")
            try:
                total_force = float(telemetry.get("right_foot_force_n")) + float(telemetry.get("left_foot_force_n"))
                self.force_chart.append(total_force, timestamp)
            except (TypeError, ValueError):
                pass
        self._update_recording_gate()
        self._refresh_system_banner()

    def _on_safety_flags(self, known: bool, free_base: bool, balance: bool) -> None:
        self._safety_flags = (bool(free_base), bool(balance)) if known else None
        self.telemetry.set_safety_flags(known, free_base, balance)

    def _on_runtime_status(self, status: object) -> None:
        if not isinstance(status, RuntimeStatus):
            self._log("ERROR", "SAFETY", "RUNTIME_STATUS_INVALID", "Worker передал некорректный RuntimeStatus")
            return
        self._runtime_status = status
        production = status.mode is RuntimeMode.PRODUCTION
        if not production:
            self.telemetry.set_interlock_checked(False)
            self.telemetry.connect_button.setEnabled(
                self._robot_state not in {"DISCONNECTED", "CONNECTING", "ARMING", "DISARMING", "DISCONNECTING"}
            )
            self.system_banner.show_demo(status.fallback_reason or "Неизвестная причина DEMO fallback")
        elif self._robot_state not in {"CONNECTING", "ARMING", "DISARMING", "DISCONNECTING"}:
            self.telemetry.connect_button.setEnabled(True)
        self._update_robot_controls()
        self._update_recording_gate()
        self._refresh_system_banner()

    def _update_robot_controls(self) -> None:
        transition = self._robot_state in {
            "CONNECTING",
            "ARMING",
            "DISARMING",
            "DISCONNECTING",
        }
        can_disconnect = self._robot_state != "DISCONNECTED"
        can_connect = self._runtime_status.mode is RuntimeMode.PRODUCTION
        self.telemetry.connect_button.setEnabled(
            not transition and (can_disconnect or can_connect)
        )
        if self._robot_state != "ARMED":
            self.telemetry.robot_interlock.setEnabled(
                self._robot_state == "CONNECTED_DISARMED"
                and isinstance(self._robot_readiness, RobotReadiness)
                and self._robot_readiness.ready
            )

    def _on_robot_readiness(self, readiness: object) -> None:
        if not isinstance(readiness, RobotReadiness):
            self._robot_readiness = None
            self.readiness.clear()
            self.telemetry.robot_interlock.setEnabled(False)
            return
        self._robot_readiness = readiness
        self.readiness.set_readiness(readiness)
        if self.robot_callbacks.status is not None:
            authoritative_state = readiness.robot_state.value.upper()
            if authoritative_state != self._robot_state:
                self.set_robot_state(authoritative_state)
        self.telemetry.robot_interlock.setEnabled(
            readiness.ready or self._robot_state == "ARMED"
        )
        self._update_robot_controls()
        self._update_recording_gate()

    def _on_robot_status(self, status: object) -> None:
        self._robot_status = status
        self._refresh_system_banner()

    def _on_recorder_state(self, state: str, payload: object) -> None:
        normalized = str(state).upper()
        summary = payload if hasattr(payload, "run_id") else None
        message = str(payload) if payload is not None and summary is None else None
        if summary is not None and normalized in {"PREPARING", "RECORDING", "FINALIZING"}:
            self._active_run_id = str(getattr(summary, "run_id", "") or "") or None
        try:
            self.experiment_panel.set_recorder_state(
                normalized,
                summary,
                message=message,
            )
        except (KeyError, ValueError):
            self._log("ERROR", "RECORDER", "RECORDER_STATE_INVALID", "Получено неизвестное состояние записи", normalized)
            return
        self._recording_active = normalized in {"PREPARING", "RECORDING", "FINALIZING"}
        if normalized in {"COMPLETE", "ERROR", "IDLE"}:
            self._active_run_id = None
        self._set_recording_locks(self._recording_active)
        self._refresh_system_banner()

    def _on_recorder_progress(self, summary: object) -> None:
        if summary is None:
            return
        self.experiment_panel.set_progress(
            accepted_samples=int(getattr(summary, "accepted_samples", 0) or 0),
            dropped_samples=int(getattr(summary, "dropped_samples", 0) or 0),
            run_id=str(getattr(summary, "run_id", "") or ""),
        )

    def _on_experiment_complete(self, summary: object) -> None:
        path = getattr(summary, "path", None)
        if path:
            self._last_experiment_path = Path(path)
        self._recording_active = False
        self._active_run_id = None
        self._set_recording_locks(False)
        self._update_recording_gate()
        self._refresh_system_banner()

    def _set_recording_locks(self, locked: bool) -> None:
        self.source_panel.set_controls_locked(locked)
        self.playback_bar.set_locked(locked)
        self.telemetry.pause_button.setEnabled(
            self._pipeline_state in {"RUNNING", "PAUSED"}
        )
        self.telemetry.reset_button.setEnabled(not locked)
        self.telemetry.calibrate_button.setEnabled(not locked)

    def _update_recording_gate(self) -> None:
        reason = ""
        allowed = True
        if self._runtime_status.mode is not RuntimeMode.PRODUCTION:
            allowed, reason = False, "Нужен production backend; DEMO не записывается"
        elif self._pipeline_state != "RUNNING":
            allowed, reason = False, "Pipeline должен быть в состоянии RUNNING"
        elif self._latest_snapshot is None or not self._snapshot_received_at:
            allowed, reason = False, "Нет свежего кадра"
        elif monotonic() - self._snapshot_received_at > 0.5:
            allowed, reason = False, "Кадр устарел (более 500 мс)"
        else:
            telemetry = getattr(self._latest_snapshot, "telemetry", {}) or {}
            try:
                calibrated = telemetry.get("calibrating") is False and float(telemetry.get("calibration_progress")) >= 1.0
            except (AttributeError, TypeError, ValueError):
                calibrated = False
            if not calibrated:
                allowed, reason = False, "Сначала завершите калибровку до 100%"
        self.experiment_panel.set_start_allowed(allowed and not self._recording_active, reason)

    def _open_experiment_directory(self) -> None:
        if self._last_experiment_path is not None:
            target = self._last_experiment_path
        else:
            documents = QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.DocumentsLocation
            )
            if not documents:
                documents = QStandardPaths.writableLocation(
                    QStandardPaths.StandardLocation.AppLocalDataLocation
                )
            if not documents:
                QMessageBox.warning(self, "Каталог недоступен", "Qt не предоставил стандартный каталог данных")
                return
            target = Path(documents) / "HumanoidInterface" / "experiments"
        try:
            target.mkdir(parents=True, exist_ok=True)
            opened = QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(target.resolve()))
            )
            if not opened:
                raise RuntimeError("операционная система отклонила открытие каталога")
        except (OSError, RuntimeError) as error:
            self._log("ERROR", "RECORDER", "EXPERIMENT_DIRECTORY_OPEN_FAILED", "Не удалось открыть каталог опытов", str(error))
            QMessageBox.warning(self, "Каталог недоступен", str(error))

    def _clear_runtime_data(self, overlay: str) -> None:
        self.preview.clear(overlay=overlay)
        self.telemetry.clear_snapshot()
        self.tracking_badge.set_status("TRACKING · НЕТ ДАННЫХ", "neutral")
        self.tilt_chart.clear()
        self.force_chart.clear()
        self._latest_snapshot = None
        self._snapshot_received_at = 0.0
        self._safety_flags = None
        self._stale_active = False

    def _watchdog_tick(self) -> None:
        now = monotonic()
        if (
            self.robot_callbacks.status is not None
            and self._robot_state in {"CONNECTED_DISARMED", "ARMED"}
            and now - self._last_external_status_poll_s >= 0.25
        ):
            self._last_external_status_poll_s = now
            self._authoritative_readiness()
        if self._pipeline_state != "RUNNING" or not self._snapshot_received_at:
            return
        age = now - self._snapshot_received_at
        if age <= 0.5:
            if self._stale_active:
                self._stale_active = False
                self.preview.set_overlay("")
                self._refresh_system_banner()
            return
        if not self._stale_active:
            self._stale_active = True
            self.preview.clear(keep_frame=True, overlay="ДАННЫЕ УСТАРЕЛИ · ВЫХОД ЗАБЛОКИРОВАН")
            self.telemetry.clear_snapshot()
            self.tracking_badge.set_status("TRACKING STALE", "warning")
            self._force_robot_off("GUI_SNAPSHOT_STALE", "GUI не получал свежий snapshot более 500 мс")
            self._log("WARNING", "SAFETY", "GUI_SNAPSHOT_STALE", "Данные интерфейса устарели", f"age_ms={age * 1000.0:.0f}")
        self.system_banner.show_stale(age)
        self._update_recording_gate()

    def _refresh_playback_bar(self) -> None:
        source = self.source_panel.current_source
        self.playback_bar.set_playback_state(
            self._playback_state,
            session_state=self._pipeline_state,
            live=source is not None and source.kind == "camera",
            enabled=source is not None,
            locked=self._recording_active,
        )

    def _refresh_system_banner(self) -> None:
        if self._robot_state == "ARMED":
            status = self._robot_status
            self.system_banner.show_armed(
                endpoint=str(getattr(status, "endpoint", None) or getattr(self.worker, "_robot_endpoint", "robot")),
                rate_hz=10.0,
                command_age_s=getattr(status, "command_age_s", None),
                successful_sends=int(getattr(status, "positions_sent", 0) or 0),
            )
            return
        if self._recording_active:
            self.system_banner.show_recording(
                run_id=self.experiment_panel.run_id_label.text(),
            )
            return
        if self._stale_active:
            age = None if not self._snapshot_received_at else monotonic() - self._snapshot_received_at
            self.system_banner.show_stale(age)
            return
        if self._runtime_status.mode is RuntimeMode.DEMO:
            self.system_banner.show_demo(self._runtime_status.fallback_reason or "DEMO fallback")
            return
        self.system_banner.clear()

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
                run_id=str(raw.get("run_id", "")),
                source_id=str(raw.get("source_id", "")),
                sequence=raw.get("sequence"),
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
        if self._runtime_status.mode is RuntimeMode.DEMO and self._robot_state == "DISCONNECTED":
            reason = self._runtime_status.fallback_reason or "DEMO runtime"
            self._log("WARNING", "ROBOT", "ROBOT_CONNECT_BLOCKED_DEMO", "Подключение реального робота запрещено в DEMO", reason)
            QMessageBox.warning(self, "Подключение заблокировано", reason)
            return
        if self._robot_state in {"CONNECTING", "ARMING", "DISARMING", "DISCONNECTING"}:
            return
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
                self.set_robot_state("CONNECTING")
                if self.robot_callbacks.connect is None:
                    raise RuntimeError("external adapter has no connect callback")
                result = self.robot_callbacks.connect()
                if result is False:
                    raise RuntimeError("controller rejected connection")
                after = self._authoritative_readiness()
                if after is None or after.robot_state is not RobotUiState.CONNECTED_DISARMED:
                    raise RuntimeError(
                        "controller did not confirm authoritative CONNECTED_DISARMED"
                    )
                self._log("INFO", "ROBOT", "ROBOT_CONNECTED_DISARMED", "Соединение установлено; команды не отправляются")
            except Exception as error:
                self.set_robot_state("DEGRADED", str(error))
        else:
            self._force_robot_off("ROBOT_DISCONNECT", "Оператор отключил соединение")
            try:
                if self.robot_callbacks.disconnect is None:
                    raise RuntimeError("external adapter has no disconnect callback")
                self.set_robot_state("DISCONNECTING")
                self.robot_callbacks.disconnect()
                after = self._authoritative_readiness()
                if after is None or after.robot_state is not RobotUiState.DISCONNECTED:
                    raise RuntimeError(
                        "controller did not confirm authoritative DISCONNECTED"
                    )
            except Exception as error:
                self._log("WARNING", "ROBOT", "ROBOT_DISCONNECT_FAILED", "Ошибка закрытия соединения", str(error))
                self.set_robot_state("DEGRADED", str(error))

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
            before = self._authoritative_readiness()
            if before is None or not before.ready:
                raise RuntimeError(
                    "Authoritative readiness изменилась после подтверждения"
                    if before is None
                    else before.reason
                )
            uses_external_callbacks = self.robot_callbacks.arm is not None
            if uses_external_callbacks:
                from robot_human_interface.protocol import OperatorSafetyAcknowledgement

                acknowledgement = OperatorSafetyAcknowledgement(True, True, True)
                self.set_robot_state("ARMING")
                result = self._invoke_external_arm_once(
                    self.robot_callbacks.arm,
                    send_velocities=dialog.send_velocities,
                    readiness=before,
                    acknowledgement=acknowledgement,
                )
                if result is False:
                    raise RuntimeError("controller rejected arm")
                after = self._authoritative_readiness()
                if after is None or after.robot_state.value.upper() != "ARMED":
                    raise RuntimeError(
                        "controller did not report authoritative ARMED after arm"
                    )
                self.telemetry.set_interlock_checked(True)
                self._log("WARNING", "ROBOT", "ROBOT_ARMED", "Отправка safe_command разрешена оператором", f"setVelocities={dialog.send_velocities}")
            else:
                self.telemetry.set_interlock_checked(False)
                from robot_human_interface.protocol import OperatorSafetyAcknowledgement

                self.worker.arm_robot(
                    acknowledgement=OperatorSafetyAcknowledgement(True, True, True),
                    send_velocities=dialog.send_velocities,
                )
        except Exception as error:
            self.telemetry.set_interlock_checked(False)
            self.set_robot_state("DEGRADED", str(error))

    @staticmethod
    def _invoke_external_arm_once(
        callback: Callable[..., object],
        *,
        send_velocities: bool,
        readiness: RobotReadiness,
        acknowledgement: object,
    ) -> object:
        """Call a guarded external adapter exactly once or fail closed."""

        try:
            parameters = signature(callback).parameters
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "external arm adapter signature cannot be verified"
            ) from error
        accepts_kwargs = any(
            parameter.kind is Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        required_guards = {"readiness", "expected_command_generation"}
        if not accepts_kwargs and not required_guards.issubset(parameters):
            raise RuntimeError(
                "external arm adapter must accept readiness and expected_command_generation"
            )
        kwargs: dict[str, object] = {
            "readiness": readiness,
            "expected_command_generation": readiness.command_generation,
        }
        optional = {
            "send_velocities": bool(send_velocities),
            "acknowledgement": acknowledgement,
        }
        for name, value in optional.items():
            if accepts_kwargs or name in parameters:
                kwargs[name] = value
            elif name == "send_velocities" and bool(value):
                raise RuntimeError(
                    "external arm adapter cannot enable the requested setVelocities option"
                )
        for name in kwargs:
            parameter = parameters.get(name)
            if parameter is not None and parameter.kind is Parameter.POSITIONAL_ONLY:
                raise RuntimeError(
                    f"external arm adapter guard {name} must be keyword-capable"
                )
        return callback(**kwargs)

    def _arm_block_reason(self) -> str:
        if self._robot_state != "CONNECTED_DISARMED":
            return "Сначала установите соединение с роботом."
        readiness = self._authoritative_readiness()
        if readiness is None:
            return "Нет авторитетного RobotReadiness от worker/controller."
        if not readiness.ready:
            return readiness.reason
        return ""

    def _authoritative_readiness(self) -> RobotReadiness | None:
        if self.robot_callbacks.status is not None:
            try:
                value = self.robot_callbacks.status()
            except Exception as error:
                self._log("ERROR", "ROBOT", "ROBOT_READINESS_FAILED", "Не удалось получить авторитетную готовность", str(error))
                return None
            if not isinstance(value, RobotReadiness) or value.authoritative is not True:
                self._log("ERROR", "ROBOT", "ROBOT_READINESS_INVALID", "Внешний адаптер не предоставил RobotReadiness")
                return None
            self._on_robot_readiness(value)
            return value
        return self._robot_readiness if isinstance(self._robot_readiness, RobotReadiness) else None

    def _force_robot_off(self, event_code: str, message: str) -> None:
        was_armed = self._robot_state in {"ARMED", "ARMING"} or self.telemetry.robot_interlock.isChecked()
        self.telemetry.set_interlock_checked(False)
        if was_armed and self.robot_callbacks.disarm is not None:
            self.set_robot_state("DISARMING")
            try:
                self.robot_callbacks.disarm()
            except Exception as error:
                self.set_robot_state("DEGRADED", str(error))
                return
            after = self._authoritative_readiness()
            if after is None or after.robot_state.value.upper() not in {
                "CONNECTED_DISARMED",
                "DISCONNECTED",
            }:
                self.set_robot_state(
                    "DEGRADED", "controller did not confirm disarm"
                )
                return
        elif was_armed:
            self.set_robot_state("DISARMING")
            self.worker.disarm_robot(event_code.casefold())
        if was_armed:
            self._log(
                "WARNING",
                "ROBOT",
                event_code,
                message,
                "ожидается авторитетное подтверждение разоружения",
            )

    def _log(
        self,
        severity: str,
        subsystem: str,
        event_code: str,
        message: str,
        details: str = "",
        *,
        run_id: str = "",
        source_id: str = "",
        sequence: object | None = None,
    ) -> None:
        if not run_id and self._recording_active:
            run_id = self._active_run_id or ""
        if not source_id and self.source_panel.current_source is not None:
            source_id = self.source_panel.current_source.source_id
        if sequence is None and self._latest_snapshot is not None:
            sequence = getattr(self._latest_snapshot, "sequence", None)
        try:
            normalized_sequence = None if sequence is None else int(sequence)
        except (TypeError, ValueError):
            normalized_sequence = None
        self.log_panel.append(
            LogEntry.now(
                severity,
                subsystem,
                event_code,
                message,
                details,
                run_id,
                source_id,
                normalized_sequence,
            )
        )
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
        compact = self.width() <= 1100 or self.height() <= 660
        if compact != self._compact:
            self._compact = compact
            self.log_panel.set_compact(compact)
            self.sources_toggle.setVisible(compact)
            self.brand_widget.setVisible(not compact)
            if compact:
                self.sources_toggle.setChecked(False)
                self.source_panel.setVisible(False)
                self.log_panel.set_collapsed(True)
                self._auto_collapsed_logs = True
            else:
                self.sources_toggle.setChecked(True)
                self.source_panel.setVisible(True)
        if self.height() < 710 and not self.log_panel.collapsed:
            self._auto_collapsed_logs = True
            self.log_panel.set_collapsed(True)
        elif self.height() >= 740 and self._auto_collapsed_logs and self.log_panel.collapsed:
            self._auto_collapsed_logs = False
            self.log_panel.set_collapsed(False)

    def _restore_settings(self) -> None:
        workspace = self._settings.value("layout/workspace_splitter")
        vertical = self._settings.value("layout/vertical_splitter")
        if workspace is not None:
            self.workspace_splitter.restoreState(workspace)
        if vertical is not None:
            self.vertical_splitter.restoreState(vertical)
        vertical_sizes = self.vertical_splitter.sizes()
        if len(vertical_sizes) == 2 and vertical_sizes[1] < 120:
            self.vertical_splitter.setSizes((548, 178))
        try:
            self.telemetry.tabs.setCurrentIndex(
                int(self._settings.value("layout/telemetry_tab", 0))
            )
        except (TypeError, ValueError):
            pass
        camera = {
            "index": self._settings.value("camera/index", 0),
            "backend": self._settings.value("camera/backend", ""),
            "resolution": self._settings.value("camera/resolution", "1280×720"),
            "fps": self._settings.value("camera/fps", 30),
            "mirror": str(self._settings.value("camera/mirror", "false")).lower() in {"1", "true", "yes"},
        }
        self.source_panel.restore_camera_settings(camera)

    def _save_settings(self) -> None:
        if not self._compact:
            self._settings.setValue(
                "layout/workspace_splitter", self.workspace_splitter.saveState()
            )
            self._settings.setValue(
                "layout/vertical_splitter", self.vertical_splitter.saveState()
            )
        self._settings.setValue(
            "layout/telemetry_tab", self.telemetry.tabs.currentIndex()
        )
        camera = self.source_panel.camera_settings()
        for key, value in camera.items():
            self._settings.setValue(f"camera/{key}", value)
        self._settings.sync()

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
        self._save_settings()
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
