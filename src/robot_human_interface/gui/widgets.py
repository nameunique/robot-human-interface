"""Reusable source, telemetry, and robot-safety widgets."""

from __future__ import annotations

from math import degrees
from pathlib import Path
import platform
from typing import Iterable, Mapping

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
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
    QHeaderView,
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
        self.add_button = QPushButton("Добавить")
        self.remove_button = QPushButton("Удалить путь")
        self.add_button.clicked.connect(self.add_user_video)
        self.remove_button.clicked.connect(self.remove_user_video)
        user_buttons.addWidget(self.add_button)
        user_buttons.addWidget(self.remove_button)
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
        self.use_camera_button = QPushButton("Использовать камеру")
        self.use_camera_button.setProperty("primary", True)
        self.use_camera_button.clicked.connect(self.select_camera)
        camera_layout.addRow("Устройство", self.camera_index)
        camera_layout.addRow("Backend", self.camera_backend)
        camera_layout.addRow("Разрешение", self.camera_resolution)
        camera_layout.addRow("FPS", self.camera_fps)
        camera_layout.addRow("", self.camera_mirror)
        camera_layout.addRow("", self.use_camera_button)
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

    def camera_settings(self) -> dict[str, object]:
        return {
            "index": self.camera_index.value(),
            "backend": self.camera_backend.currentText(),
            "resolution": self.camera_resolution.currentText(),
            "fps": self.camera_fps.value(),
            "mirror": self.camera_mirror.isChecked(),
        }

    def restore_camera_settings(self, values: Mapping[str, object]) -> None:
        try:
            self.camera_index.setValue(int(values.get("index", 0)))
            backend = str(values.get("backend", ""))
            if self.camera_backend.findText(backend) >= 0:
                self.camera_backend.setCurrentText(backend)
            resolution = str(values.get("resolution", ""))
            if self.camera_resolution.findText(resolution) >= 0:
                self.camera_resolution.setCurrentText(resolution)
            self.camera_fps.setValue(int(values.get("fps", 30)))
            self.camera_mirror.setChecked(bool(values.get("mirror", False)))
        except (TypeError, ValueError):
            return

    def set_controls_locked(self, locked: bool) -> None:
        enabled = not bool(locked)
        self.tabs.setEnabled(enabled)
        self.add_button.setEnabled(enabled)
        self.remove_button.setEnabled(enabled)
        self.use_camera_button.setEnabled(enabled)


class MetricCard(QFrame):
    def __init__(
        self,
        label: str,
        value: str,
        status: str = "Нет данных",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"MetricCard{{background:{COLORS['panel']};border:1px solid {COLORS['border']};border-radius:10px}}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(1)
        name = QLabel(label)
        name.setProperty("eyebrow", True)
        self.value = QLabel(value)
        self.value.setProperty("metric", True)
        self.status = QLabel(status)
        self.status.setStyleSheet(f"color:{COLORS['muted']};font-size:10px;font-weight:600")
        layout.addWidget(name)
        layout.addWidget(self.value)
        layout.addWidget(self.status)

    def update_value(
        self,
        value: str,
        state: str = "neutral",
        status: str | None = None,
    ) -> None:
        self.value.setText(value)
        color = {"success": COLORS["success"], "warning": COLORS["warning"], "critical": COLORS["critical"]}.get(state, COLORS["muted"])
        self.status.setStyleSheet(f"color:{color};font-size:10px;font-weight:600")
        if status is not None:
            self.status.setText(status)

    def clear(self) -> None:
        self.update_value("—", "neutral", "Нет данных")


class SupportPolygonWidget(QWidget):
    """Small honest top-view of measured MuJoCo feet and CoM projection."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(92)
        self.setMaximumHeight(112)
        self.setToolTip(
            "Проекция CoM и приближённый опорный многоугольник по положениям стоп MuJoCo"
        )
        self._right: tuple[float, float] | None = None
        self._left: tuple[float, float] | None = None
        self._com: tuple[float, float] | None = None
        self._right_contact: bool | None = None
        self._left_contact: bool | None = None
        self._phase = ""

    @staticmethod
    def _point(value: object | None) -> tuple[float, float] | None:
        try:
            values = tuple(value)  # type: ignore[arg-type]
            x, y = float(values[0]), float(values[1])
        except (TypeError, ValueError, IndexError):
            return None
        return x, y

    @staticmethod
    def _contact(
        telemetry: Mapping[str, object],
        state: object | None,
        side: str,
    ) -> bool | None:
        explicit = telemetry.get(f"{side}_foot_in_contact")
        if type(explicit) is bool:
            return explicit
        state_contact = getattr(state, f"{side}_foot_in_contact", None)
        if type(state_contact) is bool:
            return state_contact
        force = telemetry.get(
            f"{side}_foot_force_n",
            telemetry.get(
                f"{side}_foot_normal_force_n",
                getattr(state, f"{side}_foot_normal_force_n", None),
            ),
        )
        try:
            return float(force) > 1e-3
        except (TypeError, ValueError, OverflowError):
            return None

    @property
    def active_contact_count(self) -> int | None:
        """Number of measured load-bearing feet, or ``None`` when unknown."""

        if self._right_contact is None or self._left_contact is None:
            return None
        return int(self._right_contact) + int(self._left_contact)

    def update_telemetry(self, telemetry: Mapping[str, object]) -> None:
        state = telemetry.get("humanoid_state") or telemetry.get("simulation_state")
        self._right = self._point(
            telemetry.get(
                "right_foot_position_m",
                getattr(state, "right_foot_position_m", None),
            )
        )
        self._left = self._point(
            telemetry.get(
                "left_foot_position_m",
                getattr(state, "left_foot_position_m", None),
            )
        )
        self._com = self._point(
            telemetry.get(
                "center_of_mass_position_m",
                getattr(state, "center_of_mass_position_m", None),
            )
        )
        self._right_contact = self._contact(telemetry, state, "right")
        self._left_contact = self._contact(telemetry, state, "left")
        self._phase = str(telemetry.get("support_phase", "") or "")
        self.update()

    def clear(self) -> None:
        self._right = self._left = self._com = None
        self._right_contact = self._left_contact = None
        self._phase = ""
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor(COLORS["panel"]))
        painter.drawRoundedRect(bounds, 7, 7)
        painter.setPen(QColor(COLORS["muted"]))
        painter.drawText(8, 15, "ОПОРА · ВИД СВЕРХУ (MUJOCO)")
        points = [point for point in (self._right, self._left, self._com) if point]
        if (
            not points
            or self._right is None
            or self._left is None
            or self.active_contact_count is None
            or self.active_contact_count == 0
        ):
            painter.drawText(
                bounds,
                Qt.AlignmentFlag.AlignCenter,
                "Нет достоверных данных об опоре",
            )
            return

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0
        span = max(max(xs) - min(xs), max(ys) - min(ys), 0.32)
        scale = min(max(1.0, bounds.width() - 36), max(1.0, bounds.height() - 34)) / span

        def screen(point: tuple[float, float]) -> QPointF:
            return QPointF(
                bounds.center().x() + (point[0] - center_x) * scale,
                bounds.center().y() - (point[1] - center_y) * scale + 8,
            )

        right = screen(self._right)
        left = screen(self._left)
        foot_w = max(10.0, min(22.0, 0.09 * scale))
        foot_h = max(18.0, min(38.0, 0.18 * scale))
        foot_rects = (
            (
                QRectF(
                    right.x() - foot_w / 2,
                    right.y() - foot_h / 2,
                    foot_w,
                    foot_h,
                ),
                "R",
                bool(self._right_contact),
            ),
            (
                QRectF(
                    left.x() - foot_w / 2,
                    left.y() - foot_h / 2,
                    foot_w,
                    foot_h,
                ),
                "L",
                bool(self._left_contact),
            ),
        )
        support_rects = [rect for rect, _label, contact in foot_rects if contact]
        polygon = QRectF(support_rects[0])
        for rect in support_rects[1:]:
            polygon = polygon.united(rect)
        painter.setPen(QPen(QColor(COLORS["accent"]), 1))
        translucent = QColor(COLORS["accent"])
        translucent.setAlpha(35)
        painter.setBrush(translucent)
        painter.drawRoundedRect(polygon, 5, 5)
        for rect, label, contact in foot_rects:
            painter.setPen(
                QPen(QColor(COLORS["accent"] if contact else COLORS["muted"]), 1)
            )
            foot_fill = QColor(COLORS["raised"])
            foot_fill.setAlpha(255 if contact else 45)
            painter.setBrush(foot_fill)
            painter.drawRoundedRect(rect, 3, 3)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)
        if self._com is not None:
            com = screen(self._com)
            inside = polygon.adjusted(-2, -2, 2, 2).contains(com)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(COLORS["success"] if inside else COLORS["critical"]))
            painter.drawEllipse(com, 4.5, 4.5)
        painter.setPen(QColor(COLORS["muted"]))
        painter.drawText(
            bounds.adjusted(7, 0, -7, -4),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
            self._phase or "—",
        )


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
        self.quality_card = MetricCard("QUALITY", "—")
        self.rate_card = MetricCard("PIPELINE", "—")
        metrics.addWidget(self.quality_card)
        metrics.addWidget(self.rate_card)
        skeleton_layout.addLayout(metrics)
        label = QLabel("КОМАНДНЫЕ УГЛЫ · 20")
        label.setProperty("eyebrow", True)
        skeleton_layout.addWidget(label)
        self.angles_table = QTableWidget(len(JOINT_NAMES), 4)
        self.angles_table.setHorizontalHeaderLabels(
            ("СУСТАВ UNITY", "IK / RAW", "SAFE-ЦЕЛЬ", "ФАКТ MUJOCO")
        )
        group_labels = [""] * len(JOINT_NAMES)
        group_labels[0], group_labels[6], group_labels[18] = "РУКИ", "НОГИ", "ГОЛОВА"
        self.angles_table.setVerticalHeaderLabels(group_labels)
        self.angles_table.verticalHeader().setVisible(True)
        self.angles_table.verticalHeader().setMinimumWidth(44)
        self.angles_table.verticalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.angles_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.angles_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.angles_table.setAlternatingRowColors(True)
        self.angles_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.angles_table.verticalHeader().setDefaultSectionSize(20)
        self.angles_table.horizontalHeader().setStretchLastSection(True)
        self.angles_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.angles_table.setColumnWidth(0, 76)
        self.angles_table.setColumnWidth(1, 42)
        self.angles_table.setColumnWidth(2, 48)
        self.angles_table.setColumnWidth(3, 58)
        for column in range(self.angles_table.columnCount()):
            header = self.angles_table.horizontalHeaderItem(column)
            if header is not None:
                header.setToolTip(header.text())
        for row, name in enumerate(JOINT_NAMES):
            group = "РУКИ" if row < 6 else ("НОГИ" if row < 18 else "ГОЛОВА")
            joint = QTableWidgetItem(name)
            joint.setToolTip(group)
            self.angles_table.setItem(row, 0, joint)
            for column, color in (
                (1, COLORS["muted"]),
                (2, COLORS["accent"]),
                (3, COLORS["success"]),
            ):
                value = QTableWidgetItem("—")
                value.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                value.setForeground(QColor(color))
                self.angles_table.setItem(row, column, value)
        skeleton_layout.addWidget(self.angles_table, 1)
        self.real_feedback_note = QLabel(
            "РЕАЛЬНЫЙ ФАКТ: — · legacy WebSocket не передаёт данные энкодеров"
        )
        self.real_feedback_note.setProperty("muted", True)
        self.real_feedback_note.setWordWrap(True)
        skeleton_layout.addWidget(self.real_feedback_note)
        self.tabs.addTab(skeleton, "Скелет")

        balance = QWidget()
        balance_layout = QVBoxLayout(balance)
        balance_layout.setContentsMargins(2, 6, 2, 2)
        self.balance_status = StatusBadge("BALANCE · НЕ ПРОВЕРЕНО", "neutral")
        self.free_base_status = StatusBadge("BASE · НЕ ПРОВЕРЕНО", "neutral")
        explanation = QLabel("Safe command проходит balance и ограничения суставов до симуляции и реального выхода.")
        explanation.setWordWrap(True)
        explanation.setProperty("muted", True)
        action_row = QHBoxLayout()
        self.reset_button = QPushButton("Сброс")
        self.calibrate_button = QPushButton("Калибровать")
        self.pause_button = QPushButton("Пауза")
        self.pause_button.setEnabled(False)
        self.reset_button.clicked.connect(self.reset_requested)
        self.calibrate_button.clicked.connect(self.calibrate_requested)
        self.pause_button.clicked.connect(self.pause_requested)
        action_row.addWidget(self.pause_button)
        action_row.addWidget(self.reset_button)
        action_row.addWidget(self.calibrate_button)
        balance_layout.addWidget(self.balance_status)
        balance_layout.addWidget(self.free_base_status)
        balance_layout.addWidget(explanation)
        self.support_polygon = SupportPolygonWidget()
        balance_layout.addWidget(self.support_polygon)
        self.balance_metrics = QLabel(
            "Наклон / высота: — / —\n"
            "CoM: —\n"
            "Опора R/L: — / —\n"
            "Силы стоп R/L: — / —\n"
            "Фаза / контакты: — / —\n"
            "Калибровка: —"
        )
        self.balance_metrics.setProperty("muted", True)
        self.balance_metrics.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        balance_layout.addWidget(self.balance_metrics)
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
            "Хорошее слежение" if quality >= .7 else ("Проверьте позу" if quality >= .4 else "Слежение ненадёжно"),
        )
        telemetry = getattr(snapshot, "telemetry", {}) or {}
        rate = telemetry.get("pipeline_hz") if isinstance(telemetry, Mapping) else None
        if rate is None:
            self.rate_card.clear()
        else:
            try:
                self.rate_card.update_value(f"{float(rate):.1f} Hz", "success", "Измерено")
            except (TypeError, ValueError):
                self.rate_card.clear()
        raw = getattr(getattr(snapshot, "raw_command", None), "positions_rad", None)
        if raw is None:
            raw = getattr(snapshot, "angles_rad", None)
        safe = getattr(getattr(snapshot, "safe_command", None), "positions_rad", None)
        actual = telemetry.get("joint_positions_rad") if isinstance(telemetry, Mapping) else None
        lower = telemetry.get("joint_lower_limits_rad") if isinstance(telemetry, Mapping) else None
        upper = telemetry.get("joint_upper_limits_rad") if isinstance(telemetry, Mapping) else None
        self._set_angle_column(1, raw, lower, upper)
        self._set_angle_column(2, safe, lower, upper)
        self._set_angle_column(3, actual, lower, upper)
        delta = telemetry.get("neural_delta_q_rad") if isinstance(telemetry, Mapping) else None
        self._set_delta_column(delta)
        self._update_balance_metrics(telemetry)
        self.support_polygon.update_telemetry(telemetry)
        balance_active = getattr(snapshot, "balance_active", None)
        free_base = getattr(snapshot, "free_base_active", None)
        if type(balance_active) is bool and type(free_base) is bool:
            self.set_safety_flags(True, free_base, balance_active)

    def set_safety_flags(self, known: bool, free_base: bool, balance_active: bool) -> None:
        if not known:
            self.balance_status.set_status("BALANCE · НЕ ПРОВЕРЕНО", "neutral")
            self.free_base_status.set_status("BASE · НЕ ПРОВЕРЕНО", "neutral")
            return
        self.balance_status.set_status("BALANCE ACTIVE" if balance_active else "BALANCE OFF", "success" if balance_active else "critical")
        self.free_base_status.set_status("FREE BASE" if free_base else "FIXED BASE", "info" if free_base else "warning")

    def _set_angle_column(
        self,
        column: int,
        values: object | None,
        lower_limits: object | None = None,
        upper_limits: object | None = None,
    ) -> None:
        try:
            sequence = tuple(values) if values is not None else ()
        except TypeError:
            sequence = ()
        try:
            lower = tuple(lower_limits) if lower_limits is not None else ()
            upper = tuple(upper_limits) if upper_limits is not None else ()
        except TypeError:
            lower = upper = ()
        default_color = {
            1: COLORS["muted"],
            2: COLORS["accent"],
            3: COLORS["success"],
            4: COLORS["warning"],
        }.get(column, COLORS["muted"])
        for row in range(len(JOINT_NAMES)):
            item = self.angles_table.item(row, column)
            item.setForeground(QColor(default_color))
            item.setBackground(QColor("transparent"))
            if row >= len(sequence):
                item.setText("—")
                continue
            try:
                value = float(sequence[row])
                item.setText(f"{degrees(value):+.1f}°")
                if row < len(lower) and row < len(upper):
                    minimum, maximum = float(lower[row]), float(upper[row])
                    span = maximum - minimum
                    if span > 0.0:
                        margin = min(value - minimum, maximum - value) / span
                        if margin < 0.0:
                            item.setForeground(QColor(COLORS["critical"]))
                            background = QColor(COLORS["critical"])
                            background.setAlpha(55)
                            item.setBackground(background)
                        elif margin <= 0.10:
                            item.setForeground(QColor(COLORS["warning"]))
                            background = QColor(COLORS["warning"])
                            background.setAlpha(40)
                            item.setBackground(background)
            except (TypeError, ValueError):
                item.setText("—")

    def _set_delta_column(self, values: object | None) -> None:
        has_delta = values is not None
        if has_delta and self.angles_table.columnCount() == 4:
            self.angles_table.setColumnCount(5)
            self.angles_table.setHorizontalHeaderItem(4, QTableWidgetItem("Δq"))
            for row in range(len(JOINT_NAMES)):
                item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setForeground(QColor(COLORS["warning"]))
                self.angles_table.setItem(row, 4, item)
        elif not has_delta and self.angles_table.columnCount() == 5:
            self.angles_table.setColumnCount(4)
        if has_delta:
            self._set_angle_column(4, values)

    def _update_balance_metrics(self, telemetry: Mapping[str, object]) -> None:
        def number(key: str, suffix: str, digits: int = 2) -> str:
            value = telemetry.get(key)
            try:
                return f"{float(value):.{digits}f}{suffix}"
            except (TypeError, ValueError):
                return "—"

        tilt = telemetry.get("base_tilt_rad")
        try:
            tilt_text = f"{degrees(float(tilt)):.1f}°"
        except (TypeError, ValueError):
            tilt_text = "—"
        state = telemetry.get("humanoid_state") or telemetry.get("simulation_state")
        com = telemetry.get("center_of_mass_position_m", getattr(state, "center_of_mass_position_m", None))
        try:
            com_text = ", ".join(f"{float(value):+.3f}" for value in tuple(com)[:3]) + " m"
        except (TypeError, ValueError):
            com_text = "—"
        contacts = telemetry.get("contact_count", getattr(state, "contact_count", None))
        contacts_text = "—" if contacts is None else str(contacts)
        progress = telemetry.get("calibration_progress")
        try:
            progress_text = f"{max(0, min(100, round(float(progress) * 100)))}%"
        except (TypeError, ValueError):
            progress_text = "—"
        def point(attribute: str) -> str:
            values = telemetry.get(attribute, getattr(state, attribute, None))
            try:
                return "(" + ", ".join(f"{float(value):+.2f}" for value in tuple(values)[:2]) + ") m"
            except (TypeError, ValueError):
                return "—"
        self.balance_metrics.setText(
            f"Наклон / высота: {tilt_text} / {number('base_height_m', ' m')}\n"
            f"CoM: {com_text}\n"
            f"Опора R/L: {point('right_foot_position_m')} / {point('left_foot_position_m')}\n"
            f"Силы стоп R/L: {number('right_foot_force_n', ' N', 1)} / {number('left_foot_force_n', ' N', 1)}\n"
            f"Фаза / контакты: {telemetry.get('support_phase', '—') or '—'} / {contacts_text}\n"
            f"Калибровка: {progress_text}"
        )

    def clear_snapshot(self) -> None:
        self.quality_card.clear()
        self.rate_card.clear()
        for column in range(1, self.angles_table.columnCount()):
            for row in range(self.angles_table.rowCount()):
                self.angles_table.item(row, column).setText("—")
        self.balance_metrics.setText(
            "Наклон / высота: — / —\nCoM: —\nОпора R/L: — / —\nСилы стоп R/L: — / —\n"
            "Фаза / контакты: — / —\nКалибровка: —"
        )
        self.support_polygon.clear()
        self.set_safety_flags(False, False, False)

    def set_robot_state(self, state: str) -> None:
        state = state.upper()
        self.robot_state.setText(state)
        transitions = {"CONNECTING", "ARMING", "DISARMING", "DISCONNECTING"}
        self.connect_button.setText("Отключить" if state not in {"DISCONNECTED", "CONNECTING"} else ("Подключение…" if state == "CONNECTING" else "Подключить"))
        self.connect_button.setEnabled(state not in transitions)
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
            "CONNECTING": COLORS["warning"],
            "ARMING": COLORS["warning"],
            "DISARMING": COLORS["warning"],
            "DISCONNECTING": COLORS["warning"],
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
