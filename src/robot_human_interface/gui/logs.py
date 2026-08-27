"""Structured log model, filtering UI, and rotating file integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from typing import Callable, Iterable, Iterator

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    QStandardPaths,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
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
    run_id: str = ""
    source_id: str = ""
    sequence: int | None = None

    @classmethod
    def now(
        cls,
        severity: str,
        subsystem: str,
        event_code: str,
        message: str,
        details: str = "",
        run_id: str = "",
        source_id: str = "",
        sequence: int | None = None,
    ) -> "LogEntry":
        return cls(
            datetime.now().astimezone().strftime("%H:%M:%S.%f")[:-3],
            severity.upper(),
            subsystem.upper(),
            event_code.upper(),
            message,
            details,
            run_id,
            source_id,
            sequence,
        )

    def as_dict(self, *, parse_details: bool = True) -> dict[str, object]:
        """Return a JSON-safe representation of the complete event."""

        details: object = self.details
        if parse_details and self.details.strip():
            try:
                details = json.loads(self.details)
            except (TypeError, ValueError, json.JSONDecodeError):
                details = self.details
        return {
            "timestamp": self.timestamp,
            "severity": self.severity,
            "subsystem": self.subsystem,
            "event_code": self.event_code,
            "message": self.message,
            "details": details,
            "run_id": self.run_id,
            "source_id": self.source_id,
            "sequence": self.sequence,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=indent)


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

    ENTRY_ROLE = int(Qt.ItemDataRole.UserRole) + 1

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        capacity: int = 5000,
        state_repeat_interval_s: float = 2.0,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(parent)
        self.capacity = max(1, int(capacity))
        self.state_repeat_interval_s = max(0.0, float(state_repeat_interval_s))
        self._monotonic_clock = monotonic_clock
        self._entries: list[LogEntry] = []
        self._last_state_events: dict[
            tuple[str, str, str, str], tuple[tuple[str, str, str], float]
        ] = {}

    @property
    def entries(self) -> tuple[LogEntry, ...]:
        return tuple(self._entries)

    def entry_at(self, row: int) -> LogEntry | None:
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None

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
        if role == self.ENTRY_ROLE:
            return entry
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

    @staticmethod
    def _is_state_event(entry: LogEntry) -> bool:
        code = entry.event_code.upper()
        return "STATE" in code or code.endswith("STATUS") or "READINESS" in code

    def _should_suppress(self, entry: LogEntry) -> bool:
        if not self._is_state_event(entry):
            return False
        now = self._monotonic_clock()
        key = (
            entry.subsystem.upper(),
            entry.event_code.upper(),
            entry.run_id,
            entry.source_id,
        )
        signature = (entry.severity.upper(), entry.message, entry.details)
        previous = self._last_state_events.get(key)
        if previous is not None:
            previous_signature, emitted_at = previous
            if (
                signature == previous_signature
                and now - emitted_at < self.state_repeat_interval_s
            ):
                return True
        self._last_state_events[key] = (signature, now)
        return False

    def append(self, entry: LogEntry) -> bool:
        """Append an event, returning ``False`` when a state repeat is rate-limited."""

        if self._should_suppress(entry):
            return False
        overflow = len(self._entries) - self.capacity + 1
        if overflow > 0:
            self.beginRemoveRows(QModelIndex(), 0, overflow - 1)
            del self._entries[:overflow]
            self.endRemoveRows()
        row = len(self._entries)
        self.beginInsertRows(QModelIndex(), row, row)
        self._entries.append(entry)
        self.endInsertRows()
        return True

    def extend(self, entries: Iterable[LogEntry]) -> None:
        for entry in entries:
            self.append(entry)

    def clear(self) -> None:
        if not self._entries:
            return
        self.beginResetModel()
        self._entries.clear()
        self._last_state_events.clear()
        self.endResetModel()


class LogFilterProxyModel(QSortFilterProxyModel):
    """Combined minimum-severity, subsystem, and free-text filter."""

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
        threshold = SEVERITY_ORDER.get(self._severity, 0)
        severity = SEVERITY_ORDER.get(values[1].upper(), 0)
        if self._severity != "ALL" and severity < threshold:
            return False
        if self._subsystem != "ALL" and values[2].upper() != self._subsystem:
            return False
        entry = model.index(source_row, 0, source_parent).data(LogTableModel.ENTRY_ROLE)
        metadata: list[str] = []
        if isinstance(entry, LogEntry):
            metadata = [entry.run_id, entry.source_id, "" if entry.sequence is None else str(entry.sequence)]
        # Timestamp digits make short sequence searches ambiguous; the search
        # box is intended for semantic fields plus run/source/sequence metadata.
        return not self._search or self._search in " ".join(values[1:] + metadata).casefold()

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        if left.column() == 1:
            return int(left.data(Qt.ItemDataRole.UserRole) or 0) < int(
                right.data(Qt.ItemDataRole.UserRole) or 0
            )
        return super().lessThan(left, right)


class LogDetailsDialog(QDialog):
    """Read-only structured view of one journal row with JSON copy support."""

    def __init__(self, entry: LogEntry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.entry = entry
        self.setWindowTitle(f"{entry.event_code} — детали события")
        self.resize(660, 440)

        layout = QVBoxLayout(self)
        heading = QLabel(f"{entry.severity} · {entry.subsystem} · {entry.timestamp}")
        heading.setProperty("eyebrow", True)
        layout.addWidget(heading)

        self.json_view = QPlainTextEdit(entry.to_json())
        self.json_view.setReadOnly(True)
        self.json_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.json_view, 1)

        actions = QHBoxLayout()
        self.copy_button = QPushButton("Копировать JSON")
        close_button = QPushButton("Закрыть")
        actions.addStretch(1)
        actions.addWidget(self.copy_button)
        actions.addWidget(close_button)
        layout.addLayout(actions)

        self.copy_button.clicked.connect(self.copy_json)
        close_button.clicked.connect(self.accept)

    def copy_json(self) -> None:
        QApplication.clipboard().setText(self.entry.to_json())


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
        self._details_dialog: LogDetailsDialog | None = None
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
        self.count_label = QLabel("Показано 0 из 0")
        self.count_label.setProperty("muted", True)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск по коду или сообщению")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(238)
        self.severity = QComboBox()
        self.severity.addItems(("Все уровни", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
        self.severity.setToolTip("Показывать выбранный уровень и все более важные")
        self.subsystem = QComboBox()
        self.subsystem.addItems(
            (
                "Все системы",
                "GUI",
                "PIPELINE",
                "SOURCE",
                "PLAYBACK",
                "MUJOCO",
                "ROBOT",
                "SAFETY",
                "RECORDER",
            )
        )
        self.export_button = QPushButton("Экспорт")
        export_menu = QMenu(self.export_button)
        self.export_filtered_action = QAction("Отфильтрованные строки…", export_menu)
        self.export_all_action = QAction("Все строки…", export_menu)
        export_menu.addAction(self.export_filtered_action)
        export_menu.addAction(self.export_all_action)
        self.export_button.setMenu(export_menu)
        self.details_button = QPushButton("Детали")
        self.details_button.setEnabled(False)
        self.follow_checkbox = QCheckBox("Автопрокрутка")
        self.follow_checkbox.setChecked(True)
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
        toolbar.addWidget(self.follow_checkbox)
        toolbar.addWidget(self.details_button)
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

        self.search.textChanged.connect(self._set_search_filter)
        self.severity.currentTextChanged.connect(self._set_severity_filter)
        self.subsystem.currentTextChanged.connect(self._set_subsystem_filter)
        self.clear_button.clicked.connect(self.model.clear)
        self.export_filtered_action.triggered.connect(
            lambda: self.export_interactive(filtered=True)
        )
        self.export_all_action.triggered.connect(
            lambda: self.export_interactive(filtered=False)
        )
        self.details_button.clicked.connect(self.show_selected_details)
        self.table.doubleClicked.connect(lambda _index: self.show_selected_details())
        self.table.selectionModel().selectionChanged.connect(
            lambda _selected, _deselected: self._update_details_button()
        )
        self.collapse_button.clicked.connect(lambda: self.set_collapsed(not self._collapsed))
        self.model.rowsInserted.connect(self._update_count)
        self.model.rowsRemoved.connect(self._update_count)
        self.model.modelReset.connect(self._update_count)
        self.proxy.rowsInserted.connect(self._update_count)
        self.proxy.rowsRemoved.connect(self._update_count)
        self.proxy.modelReset.connect(self._update_count)
        self.proxy.layoutChanged.connect(self._update_count)

    def _set_search_filter(self, value: str) -> None:
        self.proxy.set_search(value)
        self._update_count()

    def _set_severity_filter(self, value: str) -> None:
        self.proxy.set_severity("ALL" if value.startswith("Все") else value)
        self._update_count()

    def _set_subsystem_filter(self, value: str) -> None:
        self.proxy.set_subsystem("ALL" if value.startswith("Все") else value)
        self._update_count()

    def _update_count(self, *_args: object) -> None:
        self.count_label.setText(
            f"Показано {self.proxy.rowCount()} из {self.model.rowCount()}"
        )
        self._update_details_button()

    def _update_details_button(self) -> None:
        self.details_button.setEnabled(self._selected_entry() is not None)

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

    def set_compact(self, compact: bool) -> None:
        """Keep the collapsed journal toolbar usable at 1024 px width."""

        visible = not bool(compact)
        for widget in (
            self.search,
            self.subsystem,
            self.follow_checkbox,
            self.details_button,
            self.export_button,
            self.clear_button,
        ):
            widget.setVisible(visible)

    def append(self, entry: LogEntry) -> bool:
        appended = self.model.append(entry)
        self._update_count()
        if appended and self.follow_checkbox.isChecked():
            self.table.scrollToBottom()
        return appended

    def _selected_entry(self) -> LogEntry | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        source_index = self.proxy.mapToSource(rows[0])
        return self.model.entry_at(source_index.row())

    def show_selected_details(self) -> LogDetailsDialog | None:
        entry = self._selected_entry()
        if entry is None:
            return None
        dialog = LogDetailsDialog(entry, self)
        self._details_dialog = dialog
        dialog.open()
        return dialog

    def iter_entries(self, *, filtered: bool = False) -> Iterator[LogEntry]:
        """Iterate all rows or the current visible/sorted proxy rows."""

        if not filtered:
            yield from self.model.entries
            return
        for proxy_row in range(self.proxy.rowCount()):
            source_index = self.proxy.mapToSource(self.proxy.index(proxy_row, 0))
            entry = self.model.entry_at(source_index.row())
            if entry is not None:
                yield entry

    def export_interactive(self, *, filtered: bool = False) -> None:
        default_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        suffix = "filtered" if filtered else "all"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт журнала",
            str(Path(default_dir or ".") / f"humanoid-interface-log-{suffix}.csv"),
            "CSV (*.csv)",
        )
        if path:
            self.export_csv(path, filtered=filtered)

    def export_csv(self, path: str | Path, *, filtered: bool = False) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [label for label, _ in self.model.COLUMNS]
                + ["Run ID", "Source ID", "Sequence"]
            )
            for entry in self.iter_entries(filtered=filtered):
                writer.writerow(
                    (
                        entry.timestamp,
                        entry.severity,
                        entry.subsystem,
                        entry.event_code,
                        entry.message,
                        entry.details,
                        entry.run_id,
                        entry.source_id,
                        "" if entry.sequence is None else entry.sequence,
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
            run_id = getattr(record, "run_id", "")
            source_id = getattr(record, "source_id", "")
            raw_sequence = getattr(record, "sequence", None)
            try:
                sequence = None if raw_sequence is None else int(raw_sequence)
            except (TypeError, ValueError):
                sequence = None
            self.bridge.entry_ready.emit(
                LogEntry.now(
                    record.levelname,
                    str(subsystem),
                    str(event_code),
                    record.getMessage(),
                    str(details),
                    str(run_id),
                    str(source_id),
                    sequence,
                )
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
