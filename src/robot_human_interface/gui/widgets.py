"""Reusable source, telemetry, and robot-safety widgets."""

from __future__ import annotations

from math import degrees
from pathlib import Path
import platform
from typing import Iterable, Mapping

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .resources import ResourceLocator, SourceItem, UserSourceStore
from .theme import COLORS, status_style


JOINT_NAMES: tuple[str, ...] = (
    "shoulder_rh", "shoulder_lh", "elbow_rh", "elbow_lh", "wrist_rh", "wrist_lh",
    "rotat_axis_rl", "rotat_axis_ll", "motors_thigh_rl", "motors_thigh_ll",
    "knee_rl", "knee_ll", "shin_rl", "shin_ll", "motors_feet_rl", "motors_feet_ll",
    "foot_rl", "foot_ll", "neck", "head",
)


class StatusBadge(QLabel):
    def __init__(self, text: str, kind: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_status(text, kind)

    def set_status(self, text: str, kind: str = "neutral") -> None:
        self.setText(f"●  {text}")
        self.setStyleSheet(status_style(kind))


class SourceCardWidget(QFrame):
    def __init__(self, source: SourceItem, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source = source
        self.setMinimumHeight(80)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 7, 9, 7)
        layout.setSpacing(8)
        self.thumbnail = QLabel("MP4")
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setFixedSize(56, 54)
        self.thumbnail.setStyleSheet(
            f"background:{COLORS['app']};border:1px solid {COLORS['border']};"
            f"border-radius:6px;color:{COLORS['muted']};font-size:9px"
        )
        layout.addWidget(self.thumbnail)
        copy = QVBoxLayout()
        copy.setSpacing(1)
        type_label = QLabel("КАМЕРА" if source.kind == "camera" else ("МОЁ ВИДЕО · MP4" if source.kind == "user" else "ЭТАЛОН · MP4"))
        type_label.setProperty("eyebrow", True)
        title = QLabel(source.title)
        title.setStyleSheet("font-size:13px;font-weight:600")
        title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        path = "Устройство" if source.path is None else Path(source.path).name
        self.meta_label = QLabel(path)
        self.meta_label.setProperty("muted", True)
        self.meta_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.availability = QLabel("Готов к запуску" if source.available else "Файл недоступен")
        self.availability.setStyleSheet(f"color:{COLORS['success'] if source.available else COLORS['critical']};font-size:10px")
        copy.addWidget(type_label)
        copy.addWidget(title)
        copy.addWidget(self.meta_label)
        copy.addWidget(self.availability)
        layout.addLayout(copy, 1)
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        border = COLORS["accent"] if selected else COLORS["border"]
        self.setStyleSheet(
            f"SourceCardWidget{{background:{COLORS['raised']};border:1px solid {border};border-radius:10px}}"
        )

    def set_metadata(self, metadata: object) -> None:
        error = getattr(metadata, "error", None)
        if error:
            self.availability.setText("Метаданные недоступны")
            self.availability.setStyleSheet(f"color:{COLORS['warning']};font-size:10px")
            self.setToolTip(str(error))
            return
        duration = str(getattr(metadata, "duration_label", "—"))
        resolution = str(getattr(metadata, "resolution_label", "—"))
        self.meta_label.setText(f"{duration} · {resolution}")
        thumbnail_path = getattr(metadata, "thumbnail_path", None)
        if thumbnail_path:
            pixmap = QPixmap(str(thumbnail_path))
            if not pixmap.isNull():
                self.thumbnail.setText("")
                self.thumbnail.setPixmap(
                    pixmap.scaled(
                        self.thumbnail.size(),
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )


class SourcePanel(QFrame):
    source_selected = pyqtSignal(object)
    probe_requested = pyqtSignal(object)

    def __init__(
        self,
        locator: ResourceLocator,
        user_store: UserSourceStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.locator = locator
        self.user_store = user_store
        self._current: SourceItem | None = None
        self._build_ui()
        self.refresh_references()
        self.refresh_users()

    @property
    def current_source(self) -> SourceItem | None:
        return self._current

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 10, 11, 10)
        layout.setSpacing(7)
        eyebrow = QLabel("ИСТОЧНИКИ")
        eyebrow.setProperty("eyebrow", True)
        title = QLabel("Пул движений")
        title.setProperty("section", True)
        layout.addWidget(eyebrow)
        layout.addWidget(title)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.reference_list = self._new_list()
        self.user_list = self._new_list()
        self.reference_list.itemClicked.connect(lambda item: self._select_item(self.reference_list, item))
        self.user_list.itemClicked.connect(lambda item: self._select_item(self.user_list, item))
        self.tabs.addTab(self.reference_list, "Эталоны")

        user_page = QWidget()
        user_layout = QVBoxLayout(user_page)
        user_layout.setContentsMargins(0, 0, 0, 0)
        user_layout.addWidget(self.user_list, 1)
        user_buttons = QHBoxLayout()
        add_button = QPushButton("Добавить")
        remove_button = QPushButton("Удалить путь")
        add_button.clicked.connect(self.add_user_video)
        remove_button.clicked.connect(self.remove_user_video)
        user_buttons.addWidget(add_button)
        user_buttons.addWidget(remove_button)
        user_layout.addLayout(user_buttons)
        self.tabs.addTab(user_page, "Мои")

        camera_page = QWidget()
        camera_layout = QFormLayout(camera_page)
        camera_layout.setContentsMargins(0, 8, 0, 0)
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 32)
        self.camera_backend = QComboBox()
        if platform.system() == "Windows":
            self.camera_backend.addItems(("auto", "msmf", "dshow"))
        elif platform.system() == "Linux":
            self.camera_backend.addItems(("v4l2", "auto", "gstreamer"))
        else:
            self.camera_backend.addItems(("auto", "avfoundation"))
        self.camera_resolution = QComboBox()
        self.camera_resolution.addItems(("1280×720", "1920×1080", "640×480"))
        self.camera_fps = QSpinBox()
        self.camera_fps.setRange(5, 120)
        self.camera_fps.setValue(30)
        self.camera_mirror = QCheckBox("Зеркальный preview")
        use_camera = QPushButton("Использовать камеру")
        use_camera.setProperty("primary", True)
        use_camera.clicked.connect(self.select_camera)
        camera_layout.addRow("Устройство", self.camera_index)
        camera_layout.addRow("Backend", self.camera_backend)
        camera_layout.addRow("Разрешение", self.camera_resolution)
        camera_layout.addRow("FPS", self.camera_fps)
        camera_layout.addRow("", self.camera_mirror)
        camera_layout.addRow("", use_camera)
        self.tabs.addTab(camera_page, "Камера")

    @staticmethod
    def _new_list() -> QListWidget:
        widget = QListWidget()
        widget.setSpacing(5)
        widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        return widget

    def _populate(self, widget: QListWidget, sources: Iterable[SourceItem]) -> None:
        widget.clear()
        for source in sources:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, source)
            card = SourceCardWidget(source)
            item.setSizeHint(card.sizeHint())
            widget.addItem(item)
            widget.setItemWidget(item, card)

    def refresh_references(self) -> None:
        sources = self.locator.stock_videos()
        self._populate(self.reference_list, sources)
        for source in sources:
            self.probe_requested.emit(source)

    def refresh_users(self) -> None:
        sources = self.user_store.load()
        self._populate(self.user_list, sources)
        for source in sources:
            self.probe_requested.emit(source)

    def file_sources(self) -> tuple[SourceItem, ...]:
        values: list[SourceItem] = []
        for widget in (self.reference_list, self.user_list):
            for row in range(widget.count()):
                source = widget.item(row).data(Qt.ItemDataRole.UserRole)
                if isinstance(source, SourceItem) and source.path:
                    values.append(source)
        return tuple(values)

    def request_all_probes(self) -> None:
        for source in self.file_sources():
            self.probe_requested.emit(source)

    def apply_video_metadata(self, metadata: object) -> None:
        source_id = getattr(metadata, "source_id", None)
        for widget in (self.reference_list, self.user_list):
            for row in range(widget.count()):
                item = widget.item(row)
                source = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(source, SourceItem) and source.source_id == source_id:
                    card = widget.itemWidget(item)
                    if isinstance(card, SourceCardWidget):
                        card.set_metadata(metadata)

    def select_initial(self) -> SourceItem | None:
        if self.reference_list.count() == 0:
            return None
        item = self.reference_list.item(0)
        self.reference_list.setCurrentItem(item)
        self._select_item(self.reference_list, item)
        return self._current

    def _select_item(self, owner: QListWidget, item: QListWidgetItem) -> None:
        source = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(source, SourceItem):
            return
        for widget in (self.reference_list, self.user_list):
            for row in range(widget.count()):
                card = widget.itemWidget(widget.item(row))
                if isinstance(card, SourceCardWidget):
                    card.set_selected(widget is owner and widget.item(row) is item)
        self._current = source
        self.source_selected.emit(source)

    def add_user_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Добавить видео", "", "Видео MP4 (*.mp4)")
        if path:
            self.user_store.add(path)
            self.refresh_users()

    def remove_user_video(self) -> None:
        item = self.user_list.currentItem()
        if item is None:
            return
        source = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(source, SourceItem):
            self.user_store.remove(source.source_id)
            self.refresh_users()

    def select_camera(self) -> None:
        width, height = (int(value) for value in self.camera_resolution.currentText().split("×"))
        source = SourceItem(
            source_id=f"camera:{self.camera_backend.currentText()}:{self.camera_index.value()}",
            title=f"Камера {self.camera_index.value()}",
            kind="camera",
            camera_index=self.camera_index.value(),
            camera_backend=self.camera_backend.currentText(),
            width=width,
            height=height,
            fps=float(self.camera_fps.value()),
            mirror=self.camera_mirror.isChecked(),
        )
        self._current = source
        self.source_selected.emit(source)


class MetricCard(QFrame):
    def __init__(self, label: str, value: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"MetricCard{{background:{COLORS['panel']};border:1px solid {COLORS['border']};border-radius:10px}}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(1)
        name = QLabel(label)
        name.setProperty("eyebrow", True)
        self.value = QLabel(value)
        self.value.setProperty("metric", True)
        self.status = QLabel("IDLE")
        self.status.setStyleSheet(f"color:{COLORS['muted']};font-size:10px;font-weight:600")
        layout.addWidget(name)
        layout.addWidget(self.value)
        layout.addWidget(self.status)

    def update_value(self, value: str, state: str = "success") -> None:
        self.value.setText(value)
        color = {"success": COLORS["success"], "warning": COLORS["warning"], "critical": COLORS["critical"]}.get(state, COLORS["muted"])
        self.status.setStyleSheet(f"color:{color};font-size:10px;font-weight:600")
        self.status.setText({"success": "TRACKING OK", "warning": "CHECK", "critical": "DEGRADED"}.get(state, "IDLE"))


class TelemetryPanel(QFrame):
    reset_requested = pyqtSignal()
    calibrate_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    robot_connect_requested = pyqtSignal()
    robot_disconnect_requested = pyqtSignal()
    robot_arm_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(7)
        eyebrow = QLabel("ТЕЛЕМЕТРИЯ")
        eyebrow.setProperty("eyebrow", True)
        title = QLabel("Скелет и команды")
        title.setProperty("section", True)
        layout.addWidget(eyebrow)
        layout.addWidget(title)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        skeleton = QWidget()
        skeleton_layout = QVBoxLayout(skeleton)
        skeleton_layout.setContentsMargins(0, 0, 0, 0)
        metrics = QHBoxLayout()
        self.quality_card = MetricCard("QUALITY", "0%")
        self.rate_card = MetricCard("PIPELINE", "0 Hz")
        metrics.addWidget(self.quality_card)
        metrics.addWidget(self.rate_card)
        skeleton_layout.addLayout(metrics)
        label = QLabel("КОМАНДНЫЕ УГЛЫ · 20")
        label.setProperty("eyebrow", True)
        skeleton_layout.addWidget(label)
        self.angles_table = QTableWidget(len(JOINT_NAMES), 2)
        self.angles_table.setHorizontalHeaderLabels(("UNITY JOINT", "DEG"))
        self.angles_table.verticalHeader().setVisible(False)
        self.angles_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.angles_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.angles_table.setAlternatingRowColors(True)
        self.angles_table.verticalHeader().setDefaultSectionSize(20)
        self.angles_table.horizontalHeader().setStretchLastSection(True)
        self.angles_table.setColumnWidth(0, 205)
        for row, name in enumerate(JOINT_NAMES):
            self.angles_table.setItem(row, 0, QTableWidgetItem(name))
            value = QTableWidgetItem("0.0°")
            value.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setForeground(QColor(COLORS["accent"]))
            self.angles_table.setItem(row, 1, value)
        skeleton_layout.addWidget(self.angles_table, 1)
        self.tabs.addTab(skeleton, "Скелет")

        balance = QWidget()
        balance_layout = QVBoxLayout(balance)
        balance_layout.setContentsMargins(2, 6, 2, 2)
        self.balance_status = StatusBadge("BALANCE READY", "success")
        self.free_base_status = StatusBadge("FREE BASE", "info")
        explanation = QLabel("Safe command проходит balance и ограничения суставов до симуляции и реального выхода.")
        explanation.setWordWrap(True)
        explanation.setProperty("muted", True)
        action_row = QHBoxLayout()
        reset = QPushButton("Сброс")
        calibrate = QPushButton("Калибровать")
        self.pause_button = QPushButton("Пауза")
        self.pause_button.setEnabled(False)
        reset.clicked.connect(self.reset_requested)
        calibrate.clicked.connect(self.calibrate_requested)
        self.pause_button.clicked.connect(self.pause_requested)
        action_row.addWidget(self.pause_button)
        action_row.addWidget(reset)
        action_row.addWidget(calibrate)
        balance_layout.addWidget(self.balance_status)
        balance_layout.addWidget(self.free_base_status)
        balance_layout.addWidget(explanation)
        balance_layout.addStretch(1)
        balance_layout.addLayout(action_row)
        self.tabs.addTab(balance, "Баланс")

        robot_box = QFrame()
        robot_box.setStyleSheet(f"QFrame{{background:{COLORS['raised']};border:1px solid {COLORS['border']};border-radius:10px}}")
        robot_layout = QGridLayout(robot_box)
        robot_layout.setContentsMargins(10, 6, 10, 6)
        robot_layout.setHorizontalSpacing(8)
        robot_label = QLabel("Реальный робот")
        self.robot_state = QLabel("DISCONNECTED")
        self.robot_state.setProperty("muted", True)
        self.connect_button = QPushButton("Подключить")
        self.connect_button.setFixedHeight(28)
        self.robot_interlock = QCheckBox("Отправка")
        self.robot_interlock.setChecked(False)
        self.robot_interlock.setEnabled(False)
        robot_layout.addWidget(robot_label, 0, 0)
        robot_layout.addWidget(self.robot_state, 1, 0)
        robot_layout.addWidget(self.connect_button, 0, 1)
        robot_layout.addWidget(self.robot_interlock, 1, 1)
        layout.addWidget(robot_box)
        self.connect_button.clicked.connect(self.robot_connect_requested)
        self.robot_interlock.toggled.connect(self.robot_arm_changed)

    def update_snapshot(self, snapshot: object) -> None:
        quality = float(getattr(snapshot, "tracking_quality", 0.0) or 0.0)
        self.quality_card.update_value(
            f"{round(quality * 100)}%",
            "success" if quality >= .7 else ("warning" if quality >= .4 else "critical"),
        )
        telemetry = getattr(snapshot, "telemetry", {}) or {}
        rate = telemetry.get("pipeline_hz", 20.0) if isinstance(telemetry, Mapping) else 20.0
        self.rate_card.update_value(f"{float(rate):.0f} Hz", "success")
        angles = getattr(snapshot, "angles_rad", None)
        if angles is None:
            command = getattr(snapshot, "safe_command", None)
            angles = getattr(command, "positions_rad", ())
        try:
            for row, value in enumerate(tuple(angles)[: len(JOINT_NAMES)]):
                self.angles_table.item(row, 1).setText(f"{degrees(float(value)):+.1f}°")
        except (TypeError, ValueError):
            pass
        balance_active = getattr(snapshot, "balance_active", None)
        free_base = getattr(snapshot, "free_base_active", None)
        if type(balance_active) is bool and type(free_base) is bool:
            self.set_safety_flags(True, free_base, balance_active)

    def set_safety_flags(self, known: bool, free_base: bool, balance_active: bool) -> None:
        if not known:
            self.balance_status.set_status("BALANCE UNKNOWN", "critical")
            self.free_base_status.set_status("BASE UNKNOWN", "critical")
            return
        self.balance_status.set_status("BALANCE ACTIVE" if balance_active else "BALANCE OFF", "success" if balance_active else "critical")
        self.free_base_status.set_status("FREE BASE" if free_base else "FIXED BASE", "info" if free_base else "warning")

    def set_robot_state(self, state: str) -> None:
        state = state.upper()
        self.robot_state.setText(state)
        self.connect_button.setText("Отключить" if state != "DISCONNECTED" else "Подключить")
        self.robot_interlock.setEnabled(state in {"CONNECTED_DISARMED", "ARMED"})
        if state == "ARMED" and not self.robot_interlock.isChecked():
            self.robot_interlock.blockSignals(True)
            self.robot_interlock.setChecked(True)
            self.robot_interlock.blockSignals(False)
        if state in {"DISCONNECTED", "DEGRADED", "CONNECTED_DISARMED"} and self.robot_interlock.isChecked():
            self.robot_interlock.blockSignals(True)
            self.robot_interlock.setChecked(False)
            self.robot_interlock.blockSignals(False)
        colors = {
            "DISCONNECTED": COLORS["muted"],
            "CONNECTED_DISARMED": COLORS["accent"],
            "ARMED": COLORS["success"],
            "DEGRADED": COLORS["critical"],
        }
        self.robot_state.setStyleSheet(f"color:{colors.get(state, COLORS['muted'])};font-size:10px;font-weight:600")

    def set_interlock_checked(self, checked: bool) -> None:
        self.robot_interlock.blockSignals(True)
        self.robot_interlock.setChecked(checked)
        self.robot_interlock.blockSignals(False)


class ArmConfirmationDialog(QDialog):
    """Explicit two-acknowledgement gate for physical robot output."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Подтверждение отправки на реального робота")
        self.setModal(True)
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        title = QLabel("Разрешить отправку safe_command?")
        title.setProperty("section", True)
        text = QLabel(
            "Будут отправляться 20 углов в Unity-порядке. Отключение прекращает отправку, "
            "но не является программным E-stop и не обещает нейтральную позу."
        )
        text.setWordWrap(True)
        text.setProperty("muted", True)
        self.zone_ack = QCheckBox("Свободная зона вокруг робота проверена")
        self.estop_ack = QCheckBox("Физический аппаратный E-stop доступен оператору")
        self.velocities = QCheckBox("Совместимость Unity: один раз setVelocities=[100]×20")
        self.velocities.setChecked(False)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        self.ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("Разрешить отправку")
        self.ok_button.setEnabled(False)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addSpacing(8)
        layout.addWidget(self.zone_ack)
        layout.addWidget(self.estop_ack)
        layout.addSpacing(8)
        layout.addWidget(self.velocities)
        layout.addWidget(self.buttons)
        self.zone_ack.toggled.connect(self._update_ok)
        self.estop_ack.toggled.connect(self._update_ok)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def _update_ok(self) -> None:
        self.ok_button.setEnabled(self.zone_ack.isChecked() and self.estop_ack.isChecked())

    @property
    def send_velocities(self) -> bool:
        return self.velocities.isChecked()
