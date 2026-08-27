"""Immutable, Qt-independent state for seekable media playback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class PlaybackDiscontinuity(str, Enum):
    """Reason why consecutive delivered frames do not follow media order."""

    SEEK = "seek"
    STEP = "step"
    RESTART = "restart"
    LOOP_WRAP = "loop_wrap"


@dataclass(frozen=True, slots=True)
class PlaybackState:
    """A snapshot of file-playback state safe to pass between threads.

    ``position_s`` and ``frame_index`` describe the displayed (or requested,
    before the first decode) media frame.  Camera-frame timestamps deliberately
    remain a separate monotonic delivery timeline so MediaPipe never receives a
    backwards timestamp after seek or loop wrap.
    """

    seekable: bool
    position_s: float
    duration_s: float | None
    frame_index: int
    frame_count: int | None
    fps: float
    rate: float = 1.0
    loop_enabled: bool = False
    loop_start_s: float = 0.0
    loop_end_s: float | None = None
    eof: bool = False
    discontinuity_reason: PlaybackDiscontinuity | None = None

    def __post_init__(self) -> None:
        if type(self.seekable) is not bool or type(self.loop_enabled) is not bool:
            raise ValueError("seekable and loop_enabled must be booleans")
        if type(self.eof) is not bool:
            raise ValueError("eof must be a boolean")

        position = float(self.position_s)
        if not math.isfinite(position) or position < 0.0:
            raise ValueError("position_s must be finite and non-negative")
        duration = None if self.duration_s is None else float(self.duration_s)
        if duration is not None and (
            not math.isfinite(duration) or duration < 0.0
        ):
            raise ValueError("duration_s must be finite and non-negative or None")
        if duration is not None and position > duration + 1e-9:
            raise ValueError("position_s must not exceed duration_s")

        if isinstance(self.frame_index, bool) or int(self.frame_index) != self.frame_index:
            raise ValueError("frame_index must be a non-negative integer")
        frame_index = int(self.frame_index)
        if frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        frame_count = None if self.frame_count is None else int(self.frame_count)
        if (
            self.frame_count is not None
            and (
                isinstance(self.frame_count, bool)
                or frame_count != self.frame_count
                or frame_count < 0
            )
        ):
            raise ValueError("frame_count must be a non-negative integer or None")

        fps = float(self.fps)
        rate = float(self.rate)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be finite and positive")
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("rate must be finite and positive")

        loop_start = float(self.loop_start_s)
        loop_end = None if self.loop_end_s is None else float(self.loop_end_s)
        if not math.isfinite(loop_start) or loop_start < 0.0:
            raise ValueError("loop_start_s must be finite and non-negative")
        if loop_end is not None and (
            not math.isfinite(loop_end) or loop_end <= loop_start
        ):
            raise ValueError("loop_end_s must be after loop_start_s or None")
        if duration is not None:
            if loop_start > duration + 1e-9:
                raise ValueError("loop_start_s must not exceed duration_s")
            if loop_end is not None and loop_end > duration + 1e-9:
                raise ValueError("loop_end_s must not exceed duration_s")

        reason = self.discontinuity_reason
        if reason is not None:
            reason = PlaybackDiscontinuity(reason)

        object.__setattr__(self, "position_s", position)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "loop_start_s", loop_start)
        object.__setattr__(self, "loop_end_s", loop_end)
        object.__setattr__(self, "discontinuity_reason", reason)


__all__ = ["PlaybackDiscontinuity", "PlaybackState"]
