"""Video preview with a lightweight 33-landmark overlay."""

from __future__ import annotations

from math import isfinite
from typing import Iterable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from .theme import COLORS


POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


def _frame_to_qimage(frame: object | None) -> QImage | None:
    if frame is None:
        return None
    if isinstance(frame, QImage):
        return frame.copy()
    candidate = getattr(frame, "image_bgr", frame)
    if isinstance(candidate, QImage):
        return candidate.copy()
    shape = getattr(candidate, "shape", None)
    strides = getattr(candidate, "strides", None)
    data = getattr(candidate, "data", None)
    if shape is None or len(shape) != 3 or shape[2] != 3 or data is None:
        return None
    height, width = int(shape[0]), int(shape[1])
    bytes_per_line = int(strides[0]) if strides else width * 3
    try:
        return QImage(
            data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_BGR888,
        ).copy()
    except (TypeError, ValueError, BufferError):
        return None


def _landmarks_from_snapshot(snapshot: object) -> tuple[tuple[float, float, float], ...]:
    skeleton = getattr(snapshot, "skeleton", None)
    points = getattr(skeleton, "landmarks_2d", None) if skeleton is not None else None
    confidence = None
    if skeleton is not None:
        try:
            confidence = skeleton.confidence()
        except (AttributeError, TypeError):
            confidence = getattr(skeleton, "visibility", None)
    if points is None:
        points = getattr(snapshot, "landmarks", None)
    if points is None:
        return ()
    result: list[tuple[float, float, float]] = []
    try:
        for index, point in enumerate(points):
            x, y = float(point[0]), float(point[1])
            score = float(confidence[index]) if confidence is not None else 1.0
            if not (isfinite(x) and isfinite(y) and isfinite(score)):
                score = 0.0
            result.append((x, y, score))
    except (TypeError, ValueError, IndexError):
        return ()
    return tuple(result[:33])


class PreviewWidget(QWidget):
    """Aspect-fit video surface; painting never blocks the GUI thread."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 300)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._image: QImage | None = None
        self._landmarks: tuple[tuple[float, float, float], ...] = ()
        self._source_label = "Источник не запущен"
        self._quality = 0.0
        self._sequence = 0
        self._overlay_text = ""

    def set_source_label(self, value: str) -> None:
        self._source_label = value
        self.update()

    def set_snapshot(self, snapshot: object) -> None:
        image = _frame_to_qimage(getattr(snapshot, "frame", None))
        if image is not None:
            self._image = image
        self._landmarks = _landmarks_from_snapshot(snapshot)
        self._quality = float(getattr(snapshot, "tracking_quality", 0.0) or 0.0)
        self._sequence = int(getattr(snapshot, "sequence", self._sequence) or 0)
        source = getattr(snapshot, "source", None)
        display_name = getattr(source, "display_name", None)
        if display_name:
            self._source_label = str(display_name)
        self.update()

    def clear(self, *, keep_frame: bool = False, overlay: str = "") -> None:
        """Invalidate visual telemetry; optionally retain only the last image."""

        if not keep_frame:
            self._image = None
        self._landmarks = ()
        self._quality = 0.0
        self._overlay_text = str(overlay)
        self.update()

    def set_overlay(self, text: str = "") -> None:
        self._overlay_text = str(text)
        self.update()

    def _content_rect(self) -> QRectF:
        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        if self._image is None or self._image.isNull():
            return rect
        source_ratio = self._image.width() / max(1, self._image.height())
        target_ratio = rect.width() / max(1.0, rect.height())
        if source_ratio > target_ratio:
            height = rect.width() / source_ratio
            return QRectF(rect.left(), rect.center().y() - height / 2, rect.width(), height)
        width = rect.height() * source_ratio
        return QRectF(rect.center().x() - width / 2, rect.top(), width, rect.height())

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#030608"))
        content = self._content_rect()
        if self._image is not None and not self._image.isNull():
            painter.drawImage(content, self._image)
            painter.fillRect(content, QColor(0, 0, 0, 42))
        else:
            self._draw_grid(painter, content)
        if self._landmarks:
            self._draw_landmarks(painter, content, self._landmarks)
        self._draw_chrome(painter, content)
        if self._overlay_text:
            self._draw_overlay(painter, content, self._overlay_text)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)

    @staticmethod
    def _draw_overlay(painter: QPainter, rect: QRectF, text: str) -> None:
        painter.fillRect(rect, QColor(3, 6, 8, 150))
        box = QRectF(
            rect.center().x() - min(240.0, rect.width() * 0.42),
            rect.center().y() - 28.0,
            min(480.0, rect.width() * 0.84),
            56.0,
        )
        painter.setBrush(QColor(COLORS["raised"]))
        painter.setPen(QPen(QColor(COLORS["warning"]), 1))
        painter.drawRoundedRect(box, 10, 10)
        painter.setPen(QColor(COLORS["text"]))
        painter.setFont(QFont("Inter", 11, QFont.Weight.DemiBold))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)

    @staticmethod
    def _draw_grid(painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(23, 35, 49, 180), 1))
        for index in range(1, 8):
            x = rect.left() + rect.width() * index / 8
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        for index in range(1, 6):
            y = rect.top() + rect.height() * index / 6
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

    @staticmethod
    def _draw_demo_skeleton(painter: QPainter, rect: QRectF) -> None:
        points = (
            (.50, .15), (.42, .25), (.58, .25), (.34, .39), (.66, .39),
            (.29, .52), (.71, .52), (.45, .52), (.55, .52), (.43, .70),
            (.57, .70), (.41, .88), (.59, .88),
        )
        edges = ((0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
                 (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12))
        mapped = [QPointF(rect.left() + x * rect.width(), rect.top() + y * rect.height()) for x, y in points]
        painter.setPen(QPen(QColor(COLORS["accent"]), 1.4))
        for first, second in edges:
            painter.drawLine(mapped[first], mapped[second])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["accent"]))
        for point in mapped:
            painter.drawEllipse(point, 2.8, 2.8)

    @staticmethod
    def _draw_landmarks(
        painter: QPainter,
        rect: QRectF,
        landmarks: Iterable[tuple[float, float, float]],
    ) -> None:
        values = tuple(landmarks)
        mapped = [QPointF(rect.left() + x * rect.width(), rect.top() + y * rect.height()) for x, y, _ in values]
        painter.setPen(QPen(QColor(COLORS["accent"]), 1.5))
        for first, second in POSE_CONNECTIONS:
            if first < len(values) and second < len(values) and values[first][2] >= .45 and values[second][2] >= .45:
                painter.drawLine(mapped[first], mapped[second])
        painter.setPen(Qt.PenStyle.NoPen)
        for index, point in enumerate(mapped):
            confidence = values[index][2]
            if confidence < .35:
                continue
            painter.setBrush(QColor(COLORS["success"] if confidence >= .7 else COLORS["warning"]))
            painter.drawEllipse(point, 2.5, 2.5)

    def _draw_chrome(self, painter: QPainter, rect: QRectF) -> None:
        painter.setFont(QFont("Inter", 9))
        painter.setPen(QColor(COLORS["text"]))
        painter.drawText(QPointF(rect.left() + 14, rect.top() + 22), self._source_label)
        quality = max(0, min(100, round(self._quality * 100)))
        badge = QRectF(rect.right() - 126, rect.top() + 10, 112, 24)
        painter.setPen(QPen(QColor(COLORS["success"] if quality >= 70 else COLORS["warning"]), 1))
        painter.setBrush(QColor(COLORS["success_subtle"] if quality >= 70 else COLORS["warning_subtle"]))
        painter.drawRoundedRect(badge, 12, 12)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, f"QUALITY {quality}%")
        info = QRectF(rect.left() + 14, rect.top() + 38, 126, 24)
        painter.setPen(QPen(QColor(COLORS["accent"]), 1))
        painter.setBrush(QColor(COLORS["accent_subtle"]))
        painter.drawRoundedRect(info, 12, 12)
        landmark_count = sum(1 for _, _, score in self._landmarks if score >= .45)
        painter.drawText(info, Qt.AlignmentFlag.AlignCenter, f"VIDEO · {landmark_count}/33")
