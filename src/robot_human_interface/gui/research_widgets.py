"""Research-oriented widgets for playback, safety, and experiment recording.

The widgets in this module are intentionally thin presentation adapters.  In
particular, :class:`ReadinessChecklist` only renders the authoritative
``RobotReadiness`` object produced by the worker and never makes an arming
decision itself.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from enum import Enum
import math
from time import monotonic

from PyQt6.QtCore import (
    QElapsedTimer,
    QPointF,
    QRectF,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from robot_human_interface.experiments import (
    ExperimentSpec,
    RecorderState,
    RecorderSummary,
)
from robot_human_interface.playback import PlaybackState

from .runtime import ReadinessReason, RobotReadiness, RobotUiState, RuntimeMode
from .theme import COLORS, status_style


def _coerce_enum(value: object, enum_type: type[Enum]) -> Enum:
    """Accept enum values and case-insensitive names for GUI-facing methods."""

    if isinstance(value, enum_type):
        return value
    text = str(value).strip()
    try:
        return enum_type(text.lower())
    except ValueError:
        return enum_type[text.upper()]


def _format_time(seconds: float | None, *, milliseconds: bool = False) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0.0:
        return "—"
    total = float(seconds)
    whole = int(total)
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if milliseconds:
        suffix = f".{int((total - whole) * 1000.0):03d}"
    else:
        suffix = ""
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}{suffix}"
    return f"{minutes:02d}:{secs:02d}{suffix}"


class SystemBannerState(str, Enum):
    """Mutually exclusive global states shown above the operator workspace."""

    HIDDEN = "hidden"
    DEMO = "demo"
    STALE = "stale"
    ERROR = "error"
    ARMED = "armed"
    RECORDING = "recording"


class SystemBanner(QFrame):
    """Persistent, truthful banner for exceptional or safety-critical states."""

    stop_sending_requested = pyqtSignal()

    _PRESENTATION = {
        SystemBannerState.DEMO: (
            "ДЕМО-РЕЖИМ",
            "Физический робот недоступен",
            "warning",
            "warning_subtle",
        ),
        SystemBannerState.STALE: (
            "ДАННЫЕ УСТАРЕЛИ",
            "Команды роботу заблокированы",
            "warning",
            "warning_subtle",
        ),
        SystemBannerState.ERROR: (
            "ОШИБКА",
            "Проверьте журнал событий",
            "critical",
            "critical_subtle",
        ),
        SystemBannerState.ARMED: (
            "РОБОТ ARMED",
            "Идёт отправка команд на реальный робот",
            "warning",
            "warning_subtle",
        ),
        SystemBannerState.RECORDING: (
            "● ЗАПИСЬ ОПЫТА",
            "Числовые данные сохраняются",
            "critical",
            "critical_subtle",
        ),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("systemBanner")
        self._state = SystemBannerState.HIDDEN

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 7, 10, 7)
        layout.setSpacing(10)
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size:11px;font-weight:700")
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.details_label = QLabel()
        self.details_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.stop_button = QPushButton("Остановить отправку — не E-stop")
        self.stop_button.setProperty("danger", True)
        self.stop_button.clicked.connect(self.stop_sending_requested)

        layout.addWidget(self.title_label)
        layout.addWidget(self.message_label, 1)
        layout.addWidget(self.details_label)
        layout.addWidget(self.stop_button)
        self.set_state(SystemBannerState.HIDDEN)

    @property
    def banner_state(self) -> SystemBannerState:
        return self._state

    def set_state(
        self,
        state: SystemBannerState | str | None,
        *,
        message: str | None = None,
        details: str | None = None,
    ) -> None:
        resolved = (
            SystemBannerState.HIDDEN
            if state is None
            else _coerce_enum(state, SystemBannerState)
        )
        assert isinstance(resolved, SystemBannerState)
        self._state = resolved
        if resolved is SystemBannerState.HIDDEN:
            self.setVisible(False)
            self.stop_button.setVisible(False)
            return

        title, default_message, color_key, background_key = self._PRESENTATION[resolved]
        foreground = COLORS[color_key]
        background = COLORS[background_key]
        self.title_label.setText(title)
        self.message_label.setText(message or default_message)
        self.details_label.setText(details or "")
        self.details_label.setVisible(bool(details))
        self.details_label.setStyleSheet(f"color:{foreground};font-size:10px")
        self.stop_button.setVisible(resolved is SystemBannerState.ARMED)
        self.setStyleSheet(
            "QFrame#systemBanner{"
            f"background:{background};border:1px solid {foreground};"
            "border-radius:6px;}"
            f"QFrame#systemBanner QLabel{{color:{foreground};}}"
        )
        self.setVisible(True)

    # Small named helpers make the integration call-sites unambiguous.
    def clear(self) -> None:
        self.set_state(SystemBannerState.HIDDEN)

    def show_demo(self, reason: str) -> None:
        self.set_state(
            SystemBannerState.DEMO,
            message="Используется безопасный демонстрационный pipeline",
            details=reason,
        )

    def show_stale(self, age_s: float | None = None) -> None:
        details = "" if age_s is None else f"Возраст команды: {age_s * 1000.0:.0f} мс"
        self.set_state(SystemBannerState.STALE, details=details)

    def show_error(self, message: str, details: str | None = None) -> None:
        self.set_state(SystemBannerState.ERROR, message=message, details=details)

    def show_armed(
        self,
        *,
        endpoint: str,
        rate_hz: float | None,
        command_age_s: float | None,
        successful_sends: int,
    ) -> None:
        rate = "—" if rate_hz is None else f"{rate_hz:g} Гц"
        age = "—" if command_age_s is None else f"{command_age_s * 1000.0:.0f} мс"
        self.set_state(
            SystemBannerState.ARMED,
            details=(
                f"{endpoint}  ·  {rate}  ·  команда {age}  ·  "
                f"отправлено {int(successful_sends)}"
            ),
        )

    def show_recording(
        self,
        *,
        run_id: str,
        elapsed_s: float | None = None,
    ) -> None:
        details = str(run_id)
        if elapsed_s is not None:
            details += f"  ·  {_format_time(elapsed_s)}"
        self.set_state(SystemBannerState.RECORDING, details=details)


class PlaybackBar(QFrame):
    """File timeline and live-source indicator.

    ``seek_requested`` is emitted only from ``QSlider.sliderReleased``.  Slider
    movement updates the preview label locally but never performs live
    scrubbing.
    """

    play_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    step_requested = pyqtSignal(int)
    seek_requested = pyqtSignal(float)
    rate_requested = pyqtSignal(float)
    loop_enabled_requested = pyqtSignal(bool)
    loop_range_requested = pyqtSignal(float, float)
    loop_requested = pyqtSignal(bool, float, float)

    _SLIDER_STEPS = 100_000
    _RATES = (0.25, 0.5, 1.0, 1.5, 2.0)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("playbackBar")
        self.setStyleSheet(
            "QFrame#playbackBar{"
            f"background:{COLORS['panel']};border:1px solid {COLORS['border']};"
            "border-radius:10px;}"
            f"QSlider::groove:horizontal{{height:4px;background:{COLORS['border']};"
            "border-radius:2px;}"
            f"QSlider::sub-page:horizontal{{background:{COLORS['accent']};"
            "border-radius:2px;}"
            f"QSlider::handle:horizontal{{width:14px;margin:-5px 0;"
            f"background:{COLORS['accent']};border-radius:7px;}}"
        )
        self._playback: PlaybackState | None = None
        self._running = False
        self._live = False
        self._controls_enabled = False
        self._locked = False
        self._loop_a_s = 0.0
        self._loop_b_s = 0.0
        self._display_position_s = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 7, 10, 7)
        root.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(6)
        self.mode_badge = QLabel("НЕТ ИСТОЧНИКА")
        self.mode_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mode_badge.setStyleSheet(status_style("neutral"))
        self.play_button = QPushButton("▶")
        self.play_button.setToolTip("Воспроизвести")
        self.play_button.setFixedWidth(42)
        self.back_button = QPushButton("−1")
        self.back_button.setToolTip("Предыдущий кадр")
        self.forward_button = QPushButton("+1")
        self.forward_button.setToolTip("Следующий кадр")
        self.back_button.setFixedWidth(42)
        self.forward_button.setFixedWidth(42)
        self.position_label = QLabel("— / —")
        self.position_label.setMinimumWidth(102)
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.position_label.setProperty("muted", True)

        self.rate_combo = QComboBox()
        self.rate_combo.setToolTip("Скорость воспроизведения")
        for rate in self._RATES:
            self.rate_combo.addItem(f"{rate:g}×", rate)
        self.rate_combo.setCurrentIndex(self._RATES.index(1.0))

        self.loop_checkbox = QCheckBox("Цикл A/B")
        self.set_a_button = QPushButton("A ←")
        self.set_a_button.setToolTip("Установить A по текущему кадру")
        self.set_b_button = QPushButton("B ←")
        self.set_b_button.setToolTip("Установить B по текущему кадру")
        self.loop_label = QLabel("A — · B —")
        self.loop_label.setProperty("muted", True)

        top.addWidget(self.mode_badge)
        top.addWidget(self.play_button)
        top.addWidget(self.back_button)
        top.addWidget(self.forward_button)
        top.addWidget(self.position_label)
        top.addWidget(self.rate_combo)
        top.addStretch(1)
        top.addWidget(self.loop_checkbox)
        top.addWidget(self.set_a_button)
        top.addWidget(self.set_b_button)
        top.addWidget(self.loop_label)
        root.addLayout(top)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._SLIDER_STEPS)
        self.slider.setTracking(True)
        root.addWidget(self.slider)

        self.play_button.clicked.connect(self._emit_play_pause)
        self.back_button.clicked.connect(lambda: self.step_requested.emit(-1))
        self.forward_button.clicked.connect(lambda: self.step_requested.emit(1))
        self.slider.sliderMoved.connect(self._show_slider_position)
        self.slider.sliderReleased.connect(self._commit_seek)
        self.rate_combo.currentIndexChanged.connect(self._request_rate)
        self.loop_checkbox.toggled.connect(self._request_loop_enabled)
        self.set_a_button.clicked.connect(self._set_loop_a)
        self.set_b_button.clicked.connect(self._set_loop_b)
        self.set_playback_state(None, enabled=False)

    @property
    def playback_state(self) -> PlaybackState | None:
        return self._playback

    @property
    def loop_range(self) -> tuple[float, float]:
        return self._loop_a_s, self._loop_b_s

    def set_playback_state(
        self,
        state: PlaybackState | None,
        *,
        session_state: str = "STOPPED",
        live: bool = False,
        enabled: bool = True,
        locked: bool = False,
    ) -> None:
        self._playback = state
        self._running = str(session_state).upper() == "RUNNING"
        self._live = bool(live)
        self._controls_enabled = bool(enabled)
        self._locked = bool(locked)

        with QSignalBlocker(self.rate_combo), QSignalBlocker(self.loop_checkbox):
            if state is not None:
                rate_index = self.rate_combo.findData(float(state.rate))
                if rate_index >= 0:
                    self.rate_combo.setCurrentIndex(rate_index)
                self.loop_checkbox.setChecked(state.loop_enabled)
                self._loop_a_s = state.loop_start_s
                fallback_end = state.duration_s
                if fallback_end is None:
                    fallback_end = max(state.position_s, state.loop_start_s) + 1.0 / state.fps
                self._loop_b_s = (
                    fallback_end if state.loop_end_s is None else state.loop_end_s
                )

        if state is None:
            self._display_position_s = 0.0
            self.slider.setValue(0)
            self.position_label.setText("LIVE" if live else "— / —")
        else:
            self._display_position_s = state.position_s
            self._set_slider_from_position(state.position_s)
            self.position_label.setText(
                f"{_format_time(state.position_s)} / {_format_time(state.duration_s)}"
            )

        self._update_mode_badge()
        self._update_loop_label()
        self._update_controls()

    # Short alias useful when connecting directly to a worker snapshot handler.
    set_playback = set_playback_state

    def set_locked(self, locked: bool) -> None:
        self._locked = bool(locked)
        self._update_controls()

    def _update_mode_badge(self) -> None:
        state = self._playback
        if not self._controls_enabled:
            text, kind = "ОТКЛЮЧЕНО", "neutral"
        elif self._live and state is None:
            text, kind = "LIVE", "critical"
        elif state is None:
            text, kind = "ФАЙЛ НЕ ЗАПУЩЕН", "neutral"
        elif state.eof:
            text, kind = "КОНЕЦ", "warning"
        elif not self._running:
            text, kind = "ПАУЗА", "warning"
        else:
            text, kind = "ФАЙЛ", "info"
        self.mode_badge.setText(text)
        self.mode_badge.setStyleSheet(status_style(kind))
        if self._running:
            self.play_button.setText("Ⅱ")
            self.play_button.setToolTip("Пауза")
        elif state is not None and state.eof:
            self.play_button.setText("↺")
            self.play_button.setToolTip("С начала")
        else:
            self.play_button.setText("▶")
            self.play_button.setToolTip("Воспроизвести")

    def _update_controls(self) -> None:
        usable = self._controls_enabled and not self._locked
        state = self._playback
        seekable = bool(usable and state is not None and state.seekable)
        self.play_button.setEnabled(usable and (state is not None or self._live))
        self.back_button.setEnabled(seekable)
        self.forward_button.setEnabled(seekable)
        self.slider.setEnabled(
            bool(seekable and state is not None and state.duration_s is not None)
        )
        self.rate_combo.setEnabled(seekable)
        self.loop_checkbox.setEnabled(seekable)
        loop_editable = seekable and not (
            state is not None and state.duration_s is None
        )
        self.set_a_button.setEnabled(loop_editable)
        self.set_b_button.setEnabled(loop_editable)

    def _position_from_slider(self, value: int | None = None) -> float:
        state = self._playback
        if state is None or state.duration_s is None:
            return 0.0
        slider_value = self.slider.value() if value is None else int(value)
        fraction = slider_value / self._SLIDER_STEPS
        return max(0.0, min(state.duration_s, fraction * state.duration_s))

    def _set_slider_from_position(self, position_s: float) -> None:
        state = self._playback
        if state is None or not state.duration_s:
            self.slider.setValue(0)
            return
        value = round(
            max(0.0, min(1.0, position_s / state.duration_s))
            * self._SLIDER_STEPS
        )
        with QSignalBlocker(self.slider):
            self.slider.setValue(value)

    def _show_slider_position(self, value: int) -> None:
        state = self._playback
        if state is None:
            return
        self._display_position_s = self._position_from_slider(value)
        self.position_label.setText(
            f"{_format_time(self._display_position_s)} / "
            f"{_format_time(state.duration_s)}"
        )

    def _commit_seek(self) -> None:
        if not self.slider.isEnabled():
            return
        position = self._position_from_slider()
        self._display_position_s = position
        state = self._playback
        duration = None if state is None else state.duration_s
        self.position_label.setText(
            f"{_format_time(position)} / {_format_time(duration)}"
        )
        self.seek_requested.emit(position)

    def _emit_play_pause(self) -> None:
        if self._running:
            self.pause_requested.emit()
        else:
            self.play_requested.emit()

    def _request_rate(self, index: int) -> None:
        value = self.rate_combo.itemData(index)
        if value is not None and self.rate_combo.isEnabled():
            self.rate_requested.emit(float(value))

    def _current_visible_position(self) -> float:
        return self._display_position_s

    def _set_loop_a(self) -> None:
        state = self._playback
        if state is None:
            return
        position = self._current_visible_position()
        minimum_span = 1.0 / state.fps
        self._loop_a_s = min(position, max(0.0, self._loop_b_s - minimum_span))
        self._update_loop_label()
        self._emit_loop_range()

    def _set_loop_b(self) -> None:
        state = self._playback
        if state is None:
            return
        position = self._current_visible_position()
        minimum_span = 1.0 / state.fps
        self._loop_b_s = max(position, self._loop_a_s + minimum_span)
        if state.duration_s is not None:
            self._loop_b_s = min(self._loop_b_s, state.duration_s)
        self._update_loop_label()
        self._emit_loop_range()

    def _request_loop_enabled(self, enabled: bool) -> None:
        if not self.loop_checkbox.isEnabled():
            return
        self.loop_enabled_requested.emit(bool(enabled))
        self.loop_requested.emit(bool(enabled), self._loop_a_s, self._loop_b_s)

    def _emit_loop_range(self) -> None:
        self.loop_range_requested.emit(self._loop_a_s, self._loop_b_s)
        self.loop_requested.emit(
            self.loop_checkbox.isChecked(), self._loop_a_s, self._loop_b_s
        )

    def _update_loop_label(self) -> None:
        if self._playback is None:
            self.loop_label.setText("A — · B —")
        else:
            self.loop_label.setText(
                f"A {_format_time(self._loop_a_s, milliseconds=True)} · "
                f"B {_format_time(self._loop_b_s, milliseconds=True)}"
            )


class _ReadinessRow(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(7)
        self.icon = QLabel("—")
        self.icon.setFixedWidth(14)
        self.title = QLabel(title)
        self.value = QLabel("Не проверено")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value.setProperty("muted", True)
        layout.addWidget(self.icon)
        layout.addWidget(self.title, 1)
        layout.addWidget(self.value)

    def set_result(self, result: bool | None, value: str | None = None) -> None:
        if result is True:
            icon, color, default = "✓", COLORS["success"], "Готово"
        elif result is False:
            icon, color, default = "×", COLORS["critical"], "Заблокировано"
        else:
            icon, color, default = "—", COLORS["muted"], "Не проверено"
        self.icon.setText(icon)
        self.icon.setStyleSheet(f"color:{color};font-weight:700")
        self.value.setText(value or default)
        self.value.setStyleSheet(f"color:{color};font-size:10px")


class ReadinessChecklist(QFrame):
    """Read-only representation of one authoritative arming evaluation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._readiness: RobotReadiness | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(4)
        heading = QHBoxLayout()
        title = QLabel("Готовность реального робота")
        title.setStyleSheet("font-weight:600")
        self.status_badge = QLabel("НЕ ПРОВЕРЕНО")
        self.status_badge.setStyleSheet(status_style("neutral"))
        heading.addWidget(title, 1)
        heading.addWidget(self.status_badge)
        layout.addLayout(heading)

        definitions = (
            ("runtime", "Рабочий режим"),
            ("connection", "Соединение"),
            ("pipeline", "Конвейер запущен"),
            ("fresh", "Свежая команда ≤ 500 мс"),
            ("safe", "Безопасная команда"),
            ("free_base", "Свободное основание"),
            ("balance", "Баланс"),
            ("controller", "Контроллер"),
        )
        self.rows: dict[str, _ReadinessRow] = {}
        for key, label in definitions:
            row = _ReadinessRow(label)
            self.rows[key] = row
            layout.addWidget(row)

        self.reason_label = QLabel("Подтверждённый статус ещё не получен")
        self.reason_label.setWordWrap(True)
        self.reason_label.setProperty("muted", True)
        layout.addWidget(self.reason_label)

    @property
    def readiness(self) -> RobotReadiness | None:
        return self._readiness

    def clear(self) -> None:
        self.set_readiness(None)

    def set_readiness(self, readiness: RobotReadiness | None) -> None:
        self._readiness = readiness
        if readiness is None:
            self.status_badge.setText("НЕ ПРОВЕРЕНО")
            self.status_badge.setStyleSheet(status_style("neutral"))
            for row in self.rows.values():
                row.set_result(None)
            self.reason_label.setText("Подтверждённый статус ещё не получен")
            return

        self.status_badge.setText("ГОТОВО" if readiness.ready else "БЛОКИРОВКА")
        self.status_badge.setStyleSheet(
            status_style("success" if readiness.ready else "critical")
        )
        self.reason_label.setText(
            "Все проверки пройдены"
            if readiness.ready
            else readiness.reason
        )
        self.reason_label.setToolTip(readiness.reason_code.value)

        runtime_ok = readiness.runtime.mode is RuntimeMode.PRODUCTION
        connected = readiness.robot_state in {
            RobotUiState.CONNECTED_DISARMED,
            RobotUiState.ARMING,
            RobotUiState.ARMED,
            RobotUiState.DISARMING,
        }
        pipeline_ok = readiness.pipeline_state.upper() == "RUNNING"
        fresh = (
            None
            if readiness.snapshot_age_s is None
            else readiness.snapshot_age_s <= 0.5
        )
        controller: bool | None
        if readiness.reason_code is ReadinessReason.CONTROLLER_NOT_READY:
            controller = False
        elif readiness.ready:
            controller = True
        else:
            controller = None

        self.rows["runtime"].set_result(
            runtime_ok,
            "PRODUCTION" if runtime_ok else "DEMO",
        )
        self.rows["connection"].set_result(connected, readiness.robot_state.value)
        self.rows["pipeline"].set_result(pipeline_ok, readiness.pipeline_state)
        self.rows["fresh"].set_result(
            fresh,
            "—"
            if readiness.snapshot_age_s is None
            else f"{readiness.snapshot_age_s * 1000.0:.0f} мс",
        )
        self.rows["safe"].set_result(readiness.safe_command_valid)
        self.rows["free_base"].set_result(readiness.free_base_active)
        self.rows["balance"].set_result(readiness.balance_active)
        self.rows["controller"].set_result(controller)

    # Compatibility with signal names used in worker/controller facades.
    update_readiness = set_readiness


class ExperimentPanel(QFrame):
    """Operator form and progress card for reproducible experiment capture."""

    start_requested = pyqtSignal(object)
    stop_requested = pyqtSignal()
    open_directory_requested = pyqtSignal()
    form_validity_changed = pyqtSignal(bool, str)

    _ACTIVE_STATES = {
        RecorderState.PREPARING,
        RecorderState.RECORDING,
        RecorderState.FINALIZING,
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._state = RecorderState.IDLE
        self._context_allowed = False
        self._context_reason = "Pipeline ещё не готов к записи"
        self._external_elapsed_s: float | None = None
        self._elapsed = QElapsedTimer()

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 9, 10, 9)
        root.setSpacing(7)
        header = QHBoxLayout()
        title = QLabel("Исследовательский опыт")
        title.setStyleSheet("font-size:14px;font-weight:600")
        self.state_badge = QLabel("ОЖИДАНИЕ")
        self.state_badge.setStyleSheet(status_style("neutral"))
        header.addWidget(title, 1)
        header.addWidget(self.state_badge)
        root.addLayout(header)

        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.participant_edit = QLineEdit()
        self.participant_edit.setPlaceholderText("например, P001")
        self.movement_edit = QLineEdit()
        self.movement_edit.setPlaceholderText("например, приседание")
        self.attempt_spin = QSpinBox()
        self.attempt_spin.setRange(1, 999_999)
        self.method_combo = QComboBox()
        self.method_combo.setEditable(True)
        self.method_combo.addItems(("baseline", "neural-residual"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-2_147_483_648, 2_147_483_647)
        self.note_edit = QPlainTextEdit()
        self.note_edit.setPlaceholderText("Необязательная заметка")
        self.note_edit.setMaximumHeight(62)
        form.addRow("Участник", self.participant_edit)
        form.addRow("Движение", self.movement_edit)
        form.addRow("Попытка", self.attempt_spin)
        form.addRow("Метод", self.method_combo)
        form.addRow("Seed", self.seed_spin)
        form.addRow("Заметка", self.note_edit)
        root.addWidget(form_host)

        self.consent_checkbox = QCheckBox("Согласие участника зафиксировано")
        self.video_checkbox = QCheckBox("Сохранять видео (требует согласия)")
        root.addWidget(self.consent_checkbox)
        root.addWidget(self.video_checkbox)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        root.addWidget(self.progress)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(8)
        self.run_id_label = QLabel("—")
        self.elapsed_label = QLabel("00:00")
        self.samples_label = QLabel("0")
        self.drops_label = QLabel("0")
        for label in (
            self.run_id_label,
            self.elapsed_label,
            self.samples_label,
            self.drops_label,
        ):
            label.setStyleSheet("font-weight:600")
        metric_items = (
            ("ID ОПЫТА", self.run_id_label),
            ("ВРЕМЯ", self.elapsed_label),
            ("ОБРАЗЦЫ", self.samples_label),
            ("ПОТЕРИ", self.drops_label),
        )
        for column, (caption, value) in enumerate(metric_items):
            caption_label = QLabel(caption)
            caption_label.setProperty("eyebrow", True)
            metrics.addWidget(caption_label, 0, column)
            metrics.addWidget(value, 1, column)
        root.addLayout(metrics)

        self.status_label = QLabel(self._context_reason)
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("muted", True)
        root.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("● Начать запись")
        self.start_button.setProperty("primary", True)
        self.stop_button = QPushButton("Остановить")
        self.stop_button.setProperty("danger", True)
        self.open_button = QPushButton("Открыть каталог")
        buttons.addWidget(self.start_button, 1)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.open_button)
        root.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._update_elapsed_label)
        self.start_button.clicked.connect(self._emit_start)
        self.stop_button.clicked.connect(self.stop_requested)
        self.open_button.clicked.connect(self.open_directory_requested)

        for widget in (self.participant_edit, self.movement_edit):
            widget.textChanged.connect(self._validate_form)
        self.method_combo.currentTextChanged.connect(self._validate_form)
        self.consent_checkbox.toggled.connect(self._consent_changed)
        self.video_checkbox.toggled.connect(self._validate_form)
        self.set_recorder_state(RecorderState.IDLE)

    @property
    def recorder_state(self) -> RecorderState:
        return self._state

    @property
    def recording_active(self) -> bool:
        return self._state in self._ACTIVE_STATES

    def set_start_allowed(self, allowed: bool, reason: str = "") -> None:
        """Apply the worker-owned recording gate without re-evaluating it here."""

        self._context_allowed = bool(allowed)
        self._context_reason = str(reason).strip()
        self._validate_form()

    set_context_ready = set_start_allowed

    def experiment_spec(self) -> ExperimentSpec:
        return ExperimentSpec(
            participant_code=self.participant_edit.text().strip(),
            movement=self.movement_edit.text().strip(),
            attempt=self.attempt_spin.value(),
            method_id=self.method_combo.currentText().strip(),
            seed=self.seed_spin.value(),
            note=self.note_edit.toPlainText().strip(),
            consent=self.consent_checkbox.isChecked(),
            record_video=self.video_checkbox.isChecked(),
        )

    current_spec = experiment_spec

    def _form_error(self) -> str:
        if not self.participant_edit.text().strip():
            return "Укажите код участника"
        if not self.movement_edit.text().strip():
            return "Укажите движение"
        if not self.method_combo.currentText().strip():
            return "Укажите method ID"
        if not self.consent_checkbox.isChecked():
            return "Подтвердите согласие участника"
        if self.video_checkbox.isChecked() and not self.consent_checkbox.isChecked():
            return "Для видео требуется согласие"
        return ""

    def _validate_form(self, *_args: object) -> None:
        form_error = self._form_error()
        active = self.recording_active
        valid = not form_error and self._context_allowed and not active
        self.start_button.setEnabled(bool(valid))
        self.video_checkbox.setEnabled(
            self.consent_checkbox.isChecked() and not active
        )
        if active:
            message = self.status_label.text()
        elif form_error:
            message = form_error
        elif not self._context_allowed:
            message = self._context_reason or "Запись сейчас недоступна"
        else:
            message = "Форма заполнена — можно начинать"
        self.status_label.setText(message)
        self.form_validity_changed.emit(bool(valid), message)

    def _consent_changed(self, checked: bool) -> None:
        if not checked:
            with QSignalBlocker(self.video_checkbox):
                self.video_checkbox.setChecked(False)
        self._validate_form()

    def _emit_start(self) -> None:
        if not self.start_button.isEnabled():
            return
        try:
            spec = self.experiment_spec()
        except ValueError as error:
            self.status_label.setText(str(error))
            return
        self.start_requested.emit(spec)

    def set_recorder_state(
        self,
        state: RecorderState | str,
        summary: RecorderSummary | None = None,
        *,
        elapsed_s: float | None = None,
        accepted_samples: int | None = None,
        dropped_samples: int | None = None,
        run_id: str | None = None,
        message: str | None = None,
    ) -> None:
        resolved = _coerce_enum(state, RecorderState)
        assert isinstance(resolved, RecorderState)
        previous = self._state
        self._state = resolved
        self._external_elapsed_s = elapsed_s

        if summary is not None:
            accepted_samples = summary.accepted_samples
            dropped_samples = summary.dropped_samples
            run_id = summary.run_id
            if message is None:
                message = summary.error or summary.stop_reason

        if run_id is not None:
            self.run_id_label.setText(str(run_id))
        if accepted_samples is not None:
            self.samples_label.setText(str(max(0, int(accepted_samples))))
        if dropped_samples is not None:
            self.drops_label.setText(str(max(0, int(dropped_samples))))
            self.drops_label.setStyleSheet(
                "font-weight:600;"
                f"color:{COLORS['critical'] if int(dropped_samples) else COLORS['text']}"
            )

        presentations = {
            RecorderState.IDLE: ("ОЖИДАНИЕ", "neutral", "Запись не запущена"),
            RecorderState.PREPARING: ("ПОДГОТОВКА", "warning", "Создание пакета опыта"),
            RecorderState.RECORDING: ("● ЗАПИСЬ", "critical", "Данные записываются"),
            RecorderState.FINALIZING: ("СОХРАНЕНИЕ", "warning", "Финализация файлов"),
            RecorderState.COMPLETE: ("ГОТОВО", "success", "Опыт успешно сохранён"),
            RecorderState.ERROR: ("ОШИБКА", "critical", "Опыт сохранён не полностью"),
        }
        badge, kind, default_message = presentations[resolved]
        self.state_badge.setText(badge)
        self.state_badge.setStyleSheet(status_style(kind))
        self.status_label.setText(message or default_message)

        busy = resolved in {RecorderState.PREPARING, RecorderState.FINALIZING}
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)

        if resolved is RecorderState.RECORDING:
            if previous is not RecorderState.RECORDING:
                self._elapsed.restart()
            self._timer.start()
        else:
            self._timer.stop()
        self._update_elapsed_label()

        active = resolved in self._ACTIVE_STATES
        for widget in (
            self.participant_edit,
            self.movement_edit,
            self.attempt_spin,
            self.method_combo,
            self.seed_spin,
            self.note_edit,
            self.consent_checkbox,
        ):
            widget.setEnabled(not active)
        self.stop_button.setEnabled(
            resolved in {RecorderState.PREPARING, RecorderState.RECORDING}
        )
        self.open_button.setEnabled(resolved not in {RecorderState.PREPARING})
        self._validate_form()

    update_state = set_recorder_state

    def set_progress(
        self,
        *,
        accepted_samples: int,
        dropped_samples: int,
        elapsed_s: float | None = None,
        run_id: str | None = None,
    ) -> None:
        self.samples_label.setText(str(max(0, int(accepted_samples))))
        drops = max(0, int(dropped_samples))
        self.drops_label.setText(str(drops))
        self.drops_label.setStyleSheet(
            "font-weight:600;"
            f"color:{COLORS['critical'] if drops else COLORS['text']}"
        )
        if elapsed_s is not None:
            self._external_elapsed_s = max(0.0, float(elapsed_s))
        if run_id is not None:
            self.run_id_label.setText(str(run_id))
        self._update_elapsed_label()

    def _update_elapsed_label(self) -> None:
        if self._external_elapsed_s is not None:
            elapsed_s = self._external_elapsed_s
        elif self._elapsed.isValid() and self._state is RecorderState.RECORDING:
            elapsed_s = self._elapsed.elapsed() / 1000.0
        else:
            elapsed_s = 0.0
        self.elapsed_label.setText(_format_time(elapsed_s))


class TelemetrySparkline(QWidget):
    """Small dependency-free chart retaining only a rolling time window."""

    def __init__(
        self,
        *,
        title: str = "",
        unit: str = "",
        window_s: float = 10.0,
        color: str | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not math.isfinite(window_s) or window_s <= 0.0:
            raise ValueError("window_s must be finite and positive")
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError("minimum must be less than maximum")
        self.title = str(title)
        self.unit = str(unit)
        self.window_s = float(window_s)
        self.line_color = QColor(color or COLORS["accent"])
        self.fixed_minimum = minimum
        self.fixed_maximum = maximum
        self._samples: deque[tuple[float, float]] = deque()
        self.setMinimumHeight(74)
        self.setSizePolicy(
            self.sizePolicy().horizontalPolicy(),
            self.sizePolicy().verticalPolicy(),
        )

    @property
    def samples(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._samples)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt virtual method
        return QSize(220, 84)

    def clear(self) -> None:
        self._samples.clear()
        self.update()

    def append(self, value: float, timestamp_s: float | None = None) -> None:
        number = float(value)
        timestamp = monotonic() if timestamp_s is None else float(timestamp_s)
        if not math.isfinite(number) or not math.isfinite(timestamp):
            return
        if self._samples and timestamp < self._samples[-1][0]:
            # A seek/restart is a discontinuity: never draw a misleading line
            # backwards through time.
            self._samples.clear()
        self._samples.append((timestamp, number))
        self._trim(timestamp)
        self.update()

    append_sample = append

    def set_samples(self, samples: Iterable[tuple[float, float]]) -> None:
        self._samples.clear()
        for timestamp, value in samples:
            timestamp_f = float(timestamp)
            value_f = float(value)
            if math.isfinite(timestamp_f) and math.isfinite(value_f):
                if self._samples and timestamp_f < self._samples[-1][0]:
                    raise ValueError("sparkline timestamps must be non-decreasing")
                self._samples.append((timestamp_f, value_f))
        if self._samples:
            self._trim(self._samples[-1][0])
        self.update()

    def _trim(self, newest_s: float) -> None:
        cutoff = newest_s - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt virtual
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(QColor(COLORS["border"]), 1.0))
        painter.setBrush(QColor(COLORS["raised"]))
        painter.drawRoundedRect(bounds, 6.0, 6.0)

        caption = self.title
        if self._samples:
            value = self._samples[-1][1]
            rendered = f"{value:.3g}"
            if self.unit:
                rendered += f" {self.unit}"
            caption = f"{caption}  {rendered}".strip()
        painter.setPen(QColor(COLORS["muted"]))
        painter.drawText(
            QRectF(8.0, 4.0, max(0.0, bounds.width() - 16.0), 17.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            caption,
        )

        plot = bounds.adjusted(8.0, 23.0, -8.0, -7.0)
        painter.setPen(QPen(QColor(COLORS["border"]), 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(
            QPointF(plot.left(), plot.center().y()),
            QPointF(plot.right(), plot.center().y()),
        )
        if not self._samples:
            painter.setPen(QColor(COLORS["muted"]))
            painter.drawText(
                plot,
                int(Qt.AlignmentFlag.AlignCenter),
                "Нет данных",
            )
            return

        values = [sample[1] for sample in self._samples]
        low = min(values) if self.fixed_minimum is None else self.fixed_minimum
        high = max(values) if self.fixed_maximum is None else self.fixed_maximum
        if math.isclose(low, high):
            padding = max(abs(low) * 0.05, 1.0)
            low -= padding
            high += padding
        newest = self._samples[-1][0]
        oldest = newest - self.window_s

        points: list[QPointF] = []
        for timestamp, value in self._samples:
            x_fraction = max(0.0, min(1.0, (timestamp - oldest) / self.window_s))
            y_fraction = max(0.0, min(1.0, (value - low) / (high - low)))
            points.append(
                QPointF(
                    plot.left() + x_fraction * plot.width(),
                    plot.bottom() - y_fraction * plot.height(),
                )
            )
        painter.setPen(QPen(self.line_color, 1.8))
        for first, second in zip(points, points[1:]):
            painter.drawLine(first, second)
        if len(points) == 1:
            painter.drawPoint(points[0])


# Shorter public alias for callers that do not care about the telemetry prefix.
SparklineWidget = TelemetrySparkline


__all__ = [
    "ExperimentPanel",
    "PlaybackBar",
    "ReadinessChecklist",
    "SparklineWidget",
    "SystemBanner",
    "SystemBannerState",
    "TelemetrySparkline",
]
