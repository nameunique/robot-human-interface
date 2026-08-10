"""Confidence-aware temporal filtering for pose landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from .types import LANDMARK_COUNT, SkeletonFrame


@dataclass(frozen=True, slots=True)
class SkeletonFilterConfig:
    time_constant_s: float = 0.08
    confidence_threshold: float = 0.5
    max_gap_s: float = 0.25

    def __post_init__(self) -> None:
        if self.time_constant_s < 0.0:
            raise ValueError("time_constant_s must be non-negative")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if self.max_gap_s < 0.0:
            raise ValueError("max_gap_s must be non-negative")


class SkeletonEMAFilter:
    """EMA that never incorporates low-confidence coordinates into its state."""

    def __init__(self, config: SkeletonFilterConfig | None = None) -> None:
        self.config = config or SkeletonFilterConfig()
        self.reset()

    def reset(self) -> None:
        self._points_2d = np.full((LANDMARK_COUNT, 2), np.nan)
        self._points_3d = np.full((LANDMARK_COUNT, 3), np.nan)
        self._last_seen = np.full(LANDMARK_COUNT, -np.inf)
        self._last_timestamp: float | None = None

    def update(self, frame: SkeletonFrame) -> SkeletonFrame:
        if self._last_timestamp is not None and frame.timestamp_s < self._last_timestamp:
            self.reset()
        if self._last_timestamp is None or self.config.time_constant_s == 0.0:
            alpha = 1.0
        else:
            dt = max(0.0, frame.timestamp_s - self._last_timestamp)
            alpha = 1.0 - exp(-dt / self.config.time_constant_s)

        raw_valid = frame.valid_mask(self.config.confidence_threshold)
        for index in np.flatnonzero(raw_valid):
            if not np.isfinite(self._points_3d[index]).all():
                local_alpha = 1.0
            else:
                local_alpha = alpha
            self._points_2d[index] += local_alpha * (
                frame.landmarks_2d[index] - self._points_2d[index]
            )
            self._points_3d[index] += local_alpha * (
                frame.landmarks_3d[index] - self._points_3d[index]
            )
            if local_alpha == 1.0:
                self._points_2d[index] = frame.landmarks_2d[index]
                self._points_3d[index] = frame.landmarks_3d[index]
            self._last_seen[index] = frame.timestamp_s

        expired = frame.timestamp_s - self._last_seen > self.config.max_gap_s
        self._points_2d[expired] = np.nan
        self._points_3d[expired] = np.nan
        self._last_timestamp = frame.timestamp_s
        return SkeletonFrame(
            timestamp_s=frame.timestamp_s,
            landmarks_2d=self._points_2d,
            landmarks_3d=self._points_3d,
            visibility=frame.visibility,
            presence=frame.presence,
            image_size=frame.image_size,
            sequence=frame.sequence,
        )
