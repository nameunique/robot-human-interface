"""Structured log model, filtering UI, and rotating file integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    QStandardPaths,
    Qt,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from .theme import COLORS


SEVERITY_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


@dataclass(frozen=True, slots=True)
class LogEntry:
    timestamp: str
    severity: str
    subsystem: str
    event_code: str
    message: str
    details: str = ""

    @classmethod
    def now(
        cls,
        severity: str,
        subsystem: str,
        event_code: str,
        message: str,
        details: str = "",
    ) -> "LogEntry":
        return cls(
            datetime.now().astimezone().strftime("%H:%M:%S.%f")[:-3],
            severity.upper(),
            subsystem.upper(),
            event_code.upper(),
            message,
            details,
        )


class LogTableModel(QAbstractTableModel):
    """Bounded structured event model suitable for a proxy model."""

    COLUMNS = (
        ("Время", "timestamp"),
        ("Уровень", "severity"),
        ("Подсистема", "subsystem"),
        ("Код события", "event_code"),
        ("Сообщение", "message"),
        ("Детали", "details"),
    )

    def __init__(self, parent: QObject | None = None, *, capacity: int = 5000) -> None:
        super().__init__(parent)
        self.capacity = max(1, int(capacity))
        self._entries: list[LogEntry] = []

    @property
    def entries(self) -> tuple[LogEntry, ...]:
        return tuple(self._entries)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._entries)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self.COLUMNS):
                return self.COLUMNS[section][0]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._entries)):
            return None
        entry = self._entries[index.row()]
        field = self.COLUMNS[index.column()][1]
        value = getattr(entry, field)
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return value
        if role == Qt.ItemDataRole.UserRole:
            if field == "severity":
                return SEVERITY_ORDER.get(entry.severity, 0)
            return value.casefold() if isinstance(value, str) else value
        if role == Qt.ItemDataRole.ForegroundRole:
            colors = {
                "DEBUG": COLORS["muted"],
                "INFO": COLORS["accent"],
                "WARNING": COLORS["warning"],
                "ERROR": COLORS["critical"],
                "CRITICAL": COLORS["critical"],
            }
            if field in {"severity", "event_code"}:
                from PyQt6.QtGui import QColor

                return QColor(colors.get(entry.severity, COLORS["text"]))
        return None

    def append(self, entry: LogEntry) -> None:
        overflow = len(self._entries) - self.capacity + 1
        if overflow > 0:
            self.beginRemoveRows(QModelIndex(), 0, overflow - 1)
            del self._entries[:overflow]
            self.endRemoveRows()
        row = len(self._entries)
        self.beginInsertRows(QModelIndex(), row, row)
        self._entries.append(entry)
        self.endInsertRows()

    def extend(self, entries: Iterable[LogEntry]) -> None:
        for entry in entries:
            self.append(entry)

    def clear(self) -> None:
        if not self._entries:
            return
        self.beginResetModel()
        self._entries.clear()
        self.endResetModel()


class LogFilterProxyModel(QSortFilterProxyModel):
    """Combined severity/subsystem/free-text filter."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._severity = "ALL"
        self._subsystem = "ALL"
        self._search = ""
        self.setDynamicSortFilter(True)
        self.setSortCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def set_severity(self, value: str) -> None:
        self._severity = value.upper()
        self.invalidateFilter()

    def set_subsystem(self, value: str) -> None:
        self._subsystem = value.upper()
        self.invalidateFilter()

    def set_search(self, value: str) -> None:
        self._search = value.casefold().strip()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:  # noqa: N802
        model = self.sourceModel()
        if model is None:
            return False
        values = [
            str(model.index(source_row, column, source_parent).data(Qt.ItemDataRole.DisplayRole) or "")
            for column in range(model.columnCount(source_parent))
        ]
        if self._severity != "ALL" and values[1].upper() != self._severity:
            return False
        if self._subsystem != "ALL" and values[2].upper() != self._subsystem:
            return False
        return not self._search or self._search in " ".join(values).casefold()

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        if left.column() == 1:
            return int(left.data(Qt.ItemDataRole.UserRole) or 0) < int(
                right.data(Qt.ItemDataRole.UserRole) or 0
            )
        return super().lessThan(left, right)


class LogPanel(QFrame):
    """Collapsible Figma-matched event journal."""

    collapse_requested = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("logPanel")
        self.model = LogTableModel(self)
        self.proxy = LogFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self._collapsed = False
        self._build_ui()

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 7, 12, 10)
        layout.setSpacing(7)
        toolbar = QHBoxLayout()
        title = QLabel("ЖУРНАЛ СОБЫТИЙ")
        title.setProperty("eyebrow", True)
        self.count_label = QLabel("0 / 5000")
        self.count_label.setProperty("muted", True)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск по коду или сообщению")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(238)
        self.severity = QComboBox()
        self.severity.addItems(("Все уровни", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
        self.subsystem = QComboBox()
        self.subsystem.addItems(("Все системы", "GUI", "PIPELINE", "SOURCE", "MUJOCO", "ROBOT"))
        self.export_button = QPushButton("Экспорт")
        self.clear_button = QPushButton("Очистить")
        self.collapse_button = QPushButton("Свернуть")
        self.collapse_button.setFixedWidth(84)
        toolbar.addWidget(title)
        toolbar.addSpacing(14)
        toolbar.addWidget(self.count_label)
        toolbar.addStretch(1)
        toolbar.addWidget(self.search)
        toolbar.addWidget(self.severity)
        toolbar.addWidget(self.subsystem)
        toolbar.addWidget(self.export_button)
        toolbar.addWidget(self.clear_button)
        toolbar.addWidget(self.collapse_button)
        layout.addLayout(toolbar)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(29)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        for column, width in enumerate((96, 72, 92, 168, 360, 240)):
            self.table.setColumnWidth(column, width)
        layout.addWidget(self.table, 1)

        self.search.textChanged.connect(self.proxy.set_search)
        self.severity.currentTextChanged.connect(
            lambda value: self.proxy.set_severity("ALL" if value.startswith("Все") else value)
        )
        self.subsystem.currentTextChanged.connect(
            lambda value: self.proxy.set_subsystem("ALL" if value.startswith("Все") else value)
        )
        self.clear_button.clicked.connect(self.model.clear)
        self.export_button.clicked.connect(self.export_interactive)
        self.collapse_button.clicked.connect(lambda: self.set_collapsed(not self._collapsed))
        self.model.rowsInserted.connect(self._update_count)
        self.model.rowsRemoved.connect(self._update_count)
        self.model.modelReset.connect(self._update_count)

    def _update_count(self) -> None:
        self.count_label.setText(f"{self.proxy.rowCount()} / {self.model.capacity}")

    def set_collapsed(self, collapsed: bool) -> None:
        collapsed = bool(collapsed)
        if self._collapsed == collapsed:
            return
        self._collapsed = collapsed
        self.table.setVisible(not collapsed)
        self.setMaximumHeight(46 if collapsed else 184)
        self.setMinimumHeight(46 if collapsed else 150)
        self.collapse_button.setText("Развернуть" if collapsed else "Свернуть")
        self.collapse_requested.emit(collapsed)

    def append(self, entry: LogEntry) -> None:
        self.model.append(entry)
        self._update_count()
        self.table.scrollToBottom()

    def export_interactive(self) -> None:
        default_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт журнала",
            str(Path(default_dir or ".") / "humanoid-interface-log.csv"),
            "CSV (*.csv)",
        )
        if path:
            self.export_csv(path)

    def export_csv(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow([label for label, _ in self.model.COLUMNS])
            for entry in self.model.entries:
                writer.writerow(
                    (
                        entry.timestamp,
                        entry.severity,
                        entry.subsystem,
                        entry.event_code,
                        entry.message,
                        entry.details,
                    )
                )


class _LogBridge(QObject):
    entry_ready = pyqtSignal(object)


class QtLogHandler(logging.Handler):
    """Route Python logging records to the GUI thread through a Qt signal."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__()
        self.bridge = _LogBridge(parent)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            subsystem = getattr(record, "subsystem", record.name.rsplit(".", 1)[-1])
            event_code = getattr(record, "event_code", "PYTHON_LOG")
            details = getattr(record, "details", "")
            self.bridge.entry_ready.emit(
                LogEntry.now(record.levelname, str(subsystem), str(event_code), record.getMessage(), str(details))
            )
        except Exception:
            self.handleError(record)


def build_rotating_file_handler(log_dir: str | Path | None = None) -> RotatingFileHandler:
    """Create the required 5 x 5 MiB UTF-8 log rotation handler."""

    if log_dir is None:
        location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
        log_dir = Path(location or ".") / "logs"
    directory = Path(log_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        directory / "humanoid-interface.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    return handler
