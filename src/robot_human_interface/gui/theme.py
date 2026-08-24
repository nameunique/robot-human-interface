"""Design tokens and the application-wide Qt stylesheet."""

from __future__ import annotations

from typing import Final


COLORS: Final[dict[str, str]] = {
    "app": "#0B1118",
    "panel": "#111A24",
    "raised": "#172331",
    "border": "#253548",
    "border_strong": "#35495F",
    "text": "#EAF2FA",
    "muted": "#91A4B7",
    "accent": "#35C7F2",
    "accent_subtle": "#143946",
    "success": "#43D17A",
    "success_subtle": "#183F2B",
    "warning": "#F4B740",
    "warning_subtle": "#493718",
    "critical": "#F25565",
    "critical_subtle": "#4A2027",
}


APP_STYLESHEET: Final[str] = r"""
* {
    font-family: "Inter", "Segoe UI", sans-serif;
    font-size: 12px;
    color: #EAF2FA;
}
QMainWindow, QWidget#appRoot { background: #0B1118; }
QWidget#topBar, QFrame#panel, QFrame#logPanel {
    background: #111A24;
    border: 1px solid #253548;
    border-radius: 10px;
}
QWidget#topBar { border-radius: 0; border-width: 0 0 1px 0; }
QLabel#brand { font-size: 16px; font-weight: 600; }
QLabel#brandAccent, QLabel[accent="true"] { color: #35C7F2; }
QLabel[muted="true"] { color: #91A4B7; }
QLabel[section="true"] { font-size: 16px; font-weight: 600; }
QLabel[eyebrow="true"] { color: #91A4B7; font-size: 10px; font-weight: 600; }
QLabel[metric="true"] { font-size: 22px; font-weight: 600; }
QPushButton {
    min-height: 34px;
    padding: 0 14px;
    background: #172331;
    border: 1px solid #35495F;
    border-radius: 6px;
}
QPushButton:hover { border-color: #35C7F2; background: #1D2C3B; }
QPushButton:pressed { background: #143946; }
QPushButton:disabled { color: #657789; border-color: #253548; }
QPushButton[primary="true"] {
    color: #0B1118;
    background: #35C7F2;
    border-color: #35C7F2;
    font-weight: 600;
}
QPushButton[primary="true"]:hover { background: #65D6F5; }
QPushButton[danger="true"] { color: #F25565; border-color: #F25565; }
QLineEdit, QComboBox, QSpinBox {
    min-height: 32px;
    padding: 0 10px;
    background: #172331;
    border: 1px solid #253548;
    border-radius: 6px;
    selection-background-color: #35C7F2;
    selection-color: #0B1118;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #35C7F2; }
QComboBox::drop-down { border: 0; width: 24px; }
QComboBox QAbstractItemView { background: #172331; border: 1px solid #253548; }
QTabWidget::pane { border: 0; background: transparent; }
QTabBar::tab {
    min-height: 30px;
    padding: 0 12px;
    color: #91A4B7;
    background: #111A24;
    border: 1px solid #253548;
    border-radius: 6px;
    margin-right: 5px;
}
QTabBar::tab:selected { color: #35C7F2; background: #143946; border-color: #35C7F2; }
QTableView, QListWidget {
    background: #111A24;
    alternate-background-color: #172331;
    border: 1px solid #253548;
    border-radius: 6px;
    gridline-color: #253548;
    outline: 0;
}
QTableView::item, QListWidget::item { padding: 4px; }
QTableView::item:selected, QListWidget::item:selected { background: #143946; color: #EAF2FA; }
QHeaderView::section {
    color: #91A4B7;
    background: #111A24;
    border: 0;
    border-bottom: 1px solid #253548;
    padding: 5px 8px;
    font-size: 10px;
}
QScrollBar:vertical { background: #111A24; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #35495F; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 34px; height: 18px; border-radius: 9px; background: #35495F; }
QCheckBox::indicator:checked { background: #43D17A; }
QCheckBox::indicator:disabled { background: #253548; }
QToolTip { color: #EAF2FA; background: #172331; border: 1px solid #35495F; }
QDialog { background: #111A24; }
QMessageBox { background: #111A24; }
QSplitter::handle { background: #0B1118; }
"""


def status_style(kind: str) -> str:
    """Return a compact badge style for one semantic status kind."""

    palette = {
        "success": (COLORS["success"], COLORS["success_subtle"]),
        "info": (COLORS["accent"], COLORS["accent_subtle"]),
        "warning": (COLORS["warning"], COLORS["warning_subtle"]),
        "critical": (COLORS["critical"], COLORS["critical_subtle"]),
        "neutral": (COLORS["muted"], COLORS["raised"]),
    }
    foreground, background = palette.get(kind, palette["neutral"])
    return (
        f"color:{foreground};background:{background};border:1px solid {foreground};"
        "border-radius:12px;padding:4px 10px;font-size:10px;font-weight:600;"
    )
