from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from robot_human_interface.gui.logs import (
    LogDetailsDialog,
    LogEntry,
    LogFilterProxyModel,
    LogPanel,
    LogTableModel,
)


def _entry(
    severity: str,
    code: str,
    message: str,
    *,
    subsystem: str = "PIPELINE",
    details: str = "",
    sequence: int | None = None,
) -> LogEntry:
    return LogEntry.now(
        severity,
        subsystem,
        code,
        message,
        details,
        "run-17",
        "stock:squat",
        sequence,
    )


def test_severity_filter_is_a_threshold_and_searches_metadata(qtbot) -> None:
    del qtbot
    model = LogTableModel()
    proxy = LogFilterProxyModel()
    proxy.setSourceModel(model)
    for sequence, severity in enumerate(
        ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), start=1
    ):
        model.append(_entry(severity, f"EVENT_{severity}", severity, sequence=sequence))

    proxy.set_severity("WARNING")
    assert proxy.rowCount() == 3
    proxy.set_search("stock:squat")
    assert proxy.rowCount() == 3
    proxy.set_search("sequence-does-not-exist")
    assert proxy.rowCount() == 0
    proxy.set_search("5")
    assert proxy.rowCount() == 1


def test_state_repeats_are_rate_limited_but_changes_are_immediate(qtbot) -> None:
    del qtbot
    now = [10.0]
    model = LogTableModel(
        state_repeat_interval_s=2.0,
        monotonic_clock=lambda: now[0],
    )
    same = _entry("INFO", "ROBOT_STATE", "CONNECTED_DISARMED")
    changed = _entry("INFO", "ROBOT_STATE", "ARMED")

    assert model.append(same)
    assert not model.append(same)
    assert model.append(changed)
    assert model.rowCount() == 2

    now[0] += 1.0
    assert not model.append(changed)
    now[0] += 1.1
    assert model.append(changed)
    assert model.rowCount() == 3

    # Measurements are not state transitions and remain lossless.
    assert model.append(_entry("INFO", "SAMPLE", "42"))
    assert model.append(_entry("INFO", "SAMPLE", "42"))


def test_panel_count_details_json_and_export_scope(qtbot, tmp_path: Path) -> None:
    panel = LogPanel()
    qtbot.addWidget(panel)
    panel.show()
    info = _entry("INFO", "SESSION_STARTED", "Сессия запущена", sequence=1)
    error = _entry(
        "ERROR",
        "NETWORK_ERROR",
        "Сеть недоступна",
        subsystem="ROBOT",
        details='{"endpoint":"ws://robot.local:8080"}',
        sequence=2,
    )
    panel.append(info)
    panel.append(error)
    assert panel.count_label.text() == "Показано 2 из 2"

    panel.severity.setCurrentText("ERROR")
    assert panel.count_label.text() == "Показано 1 из 2"

    all_path = tmp_path / "all.csv"
    filtered_path = tmp_path / "filtered.csv"
    panel.export_csv(all_path)
    panel.export_csv(filtered_path, filtered=True)
    with all_path.open(encoding="utf-8-sig", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    with filtered_path.open(encoding="utf-8-sig", newline="") as stream:
        filtered_rows = list(csv.DictReader(stream))
    assert len(all_rows) == 2
    assert len(filtered_rows) == 1
    assert filtered_rows[0]["Run ID"] == "run-17"
    assert filtered_rows[0]["Source ID"] == "stock:squat"
    assert filtered_rows[0]["Sequence"] == "2"

    panel.table.selectRow(0)
    dialog = panel.show_selected_details()
    assert isinstance(dialog, LogDetailsDialog)
    dialog.copy_json()
    copied = json.loads(dialog.json_view.toPlainText())
    assert copied["details"]["endpoint"] == "ws://robot.local:8080"
    assert copied["run_id"] == "run-17"
    assert copied["sequence"] == 2


def test_autoscroll_can_be_disabled(qtbot) -> None:
    panel = LogPanel()
    qtbot.addWidget(panel)
    panel.resize(900, 184)
    panel.show()
    for index in range(30):
        panel.append(_entry("INFO", "SAMPLE", str(index), sequence=index))
    qtbot.waitUntil(
        lambda: panel.table.verticalScrollBar().value()
        == panel.table.verticalScrollBar().maximum()
    )

    panel.follow_checkbox.setChecked(False)
    panel.table.verticalScrollBar().setValue(0)
    panel.append(_entry("INFO", "SAMPLE", "new", sequence=31))
    assert panel.table.verticalScrollBar().value() == 0
