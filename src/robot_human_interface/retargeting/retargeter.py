"""Configuration, calibration, filtering, and safe stale-pose behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, pi
from pathlib import Path
from time import monotonic
from typing import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from robot_human_interface.skeleton import JOINT_NAMES, RobotJointCommand, SkeletonFrame

from .geometry import (
    UPPER_BODY_REQUIRED,
    WHOLE_BODY_REQUIRED,
    compute_human_joint_angles,
    joint_landmark_validity,
)


@dataclass(frozen=True, slots=True)
class JointSpec:
    index: int
    name: str
    axis: str
    lower_rad: float
    upper_rad: float
    start_rad: float
    zero_offset_rad: float = 0.0
    retarget_sign: float = 1.0

    def __post_init__(self) -> None:
        if self.lower_rad >= self.upper_rad:
            raise ValueError(f"invalid limits for {self.name}")
        if not self.lower_rad <= self.start_rad <= self.upper_rad:
            raise ValueError(f"start pose outside limits for {self.name}")
        if self.retarget_sign not in (-1.0, 1.0):
            raise ValueError("retarget_sign must be -1 or +1")


_UNITY_JOINT_DATA = (
    ("shoulder_rh", "-X", -180, 180, 30),
    ("shoulder_lh", "-X", -180, 180, 30),
    ("elbow_rh", "+Z", -50, 160, 15),
    ("elbow_lh", "-Z", -50, 160, 15),
    ("wrist_rh", "-Z", -90, 90, 15),
    ("wrist_lh", "+Z", -90, 90, 15),
    ("rotat_axis_rl", "+Y", -40, 40, 0),
    ("rotat_axis_ll", "-Y", -40, 40, 0),
    ("motors_thigh_rl", "+Z", -40, 40, 5),
    ("motors_thigh_ll", "-Z", -40, 40, 5),
    ("knee_rl", "-X", -30, 90, 28),
    ("knee_ll", "-X", -30, 90, 28),
    ("shin_rl", "+X", -10, 150, 45),
    ("shin_ll", "+X", -10, 150, 45),
    ("motors_feet_rl", "-X", -75, 75, 20),
    ("motors_feet_ll", "-X", -75, 75, 20),
    ("foot_rl", "+Z", -60, 45, -5),
    ("foot_ll", "-Z", -60, 45, -5),
    ("neck", "-Y", -180, 180, 0),
    ("head", "-X", -25, 70, 0),
)


DEFAULT_JOINT_SPECS: tuple[JointSpec, ...] = tuple(
    JointSpec(index, name, axis, lower * pi / 180, upper * pi / 180, start * pi / 180)
    for index, (name, axis, lower, upper, start) in enumerate(_UNITY_JOINT_DATA)
)


@dataclass(frozen=True, slots=True)
class RetargetingConfig:
    mode: str = "whole_body"
    auto_calibration_frames: int = 30
    confidence_threshold: float = 0.55
    minimum_coverage: float = 0.75
    smoothing_time_constant_s: float = 0.10
    hold_seconds: float = 0.35
    return_seconds: float = 1.0
    mirrored_input: bool = False
    joint_scales: Mapping[str, float] = field(default_factory=dict)
    joint_signs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"upper_body", "whole_body"}:
            raise ValueError("mode must be upper_body or whole_body")
        if (
            isinstance(self.auto_calibration_frames, bool)
            or int(self.auto_calibration_frames) != self.auto_calibration_frames
            or self.auto_calibration_frames < 0
        ):
            raise ValueError("auto_calibration_frames must be a non-negative integer")
        object.__setattr__(self, "auto_calibration_frames", int(self.auto_calibration_frames))
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be within [0, 1]")
        if self.smoothing_time_constant_s < 0.0 or self.hold_seconds < 0.0:
            raise ValueError("time constants must be non-negative")
        if self.return_seconds <= 0.0:
            raise ValueError("return_seconds must be positive")
        unknown = (set(self.joint_scales) | set(self.joint_signs)) - set(JOINT_NAMES)
        if unknown:
            raise ValueError(f"unknown joint names in retargeting config: {sorted(unknown)}")


def _yaml_load(path: Path) -> object:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("loading YAML configuration requires PyYAML") from error
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_joint_specs(path: str | Path | None = None) -> tuple[JointSpec, ...]:
    """Load the shared joint file, falling back to verified Unity constants."""

    if path is None or not Path(path).is_file():
        return DEFAULT_JOINT_SPECS
    data = _yaml_load(Path(path))
    if not isinstance(data, Mapping):
        raise ValueError("joint YAML root must be a mapping")
    records = data.get("joints", data)
    if isinstance(records, Mapping):
        iterable = [dict(records[name], name=name) for name in JOINT_NAMES]
    elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        iterable = list(records)
    else:
        raise ValueError("joints must be a sequence or name-to-spec mapping")
    specs: list[JointSpec] = []
    for default_index, record in enumerate(iterable):
        if not isinstance(record, Mapping):
            raise ValueError("each joint record must be a mapping")
        name = str(record.get("name", ""))
        index = int(record.get("index", default_index))
        limits_rad = record.get("limits_rad", record.get("limit_rad"))
        limits_deg = record.get("limits_deg", record.get("limit_deg", record.get("range_deg")))
        if limits_rad is None and limits_deg is None:
            limits_deg = (record.get("min_deg"), record.get("max_deg"))
        limits = limits_rad if limits_rad is not None else limits_deg
        if not isinstance(limits, Sequence) or len(limits) != 2:
            raise ValueError(f"joint {name} needs two angular limits")
        if limits_rad is not None:
            lower_rad, upper_rad = float(limits[0]), float(limits[1])
        else:
            lower_rad, upper_rad = float(limits[0]) * pi / 180, float(limits[1]) * pi / 180
        if "home_rad" in record:
            start_rad = float(record["home_rad"])
        else:
            start_rad = float(record.get("home_deg", record.get("start_deg", 0.0))) * pi / 180
        if "zero_offset_rad" in record:
            zero_offset_rad = float(record["zero_offset_rad"])
        else:
            zero_offset_rad = float(record.get("zero_offset_deg", 0.0)) * pi / 180
        axis_value = record.get("axis", record.get("unity_axis", ""))
        if isinstance(axis_value, Sequence) and not isinstance(axis_value, (str, bytes)):
            vector = tuple(float(value) for value in axis_value)
            labels = {(1.0, 0.0, 0.0): "+X", (-1.0, 0.0, 0.0): "-X",
                      (0.0, 1.0, 0.0): "+Y", (0.0, -1.0, 0.0): "-Y",
                      (0.0, 0.0, 1.0): "+Z", (0.0, 0.0, -1.0): "-Z"}
            axis = labels.get(vector, str(list(vector)))
        else:
            axis = str(axis_value)
        specs.append(
            JointSpec(
                index=index,
                name=name,
                axis=axis,
                lower_rad=lower_rad,
                upper_rad=upper_rad,
                start_rad=start_rad,
                zero_offset_rad=zero_offset_rad,
                retarget_sign=float(record.get("retarget_sign", 1.0)),
            )
        )
    specs.sort(key=lambda item: item.index)
    names = tuple(item.name for item in specs)
    indices = tuple(item.index for item in specs)
    if names != JOINT_NAMES or indices != tuple(range(len(JOINT_NAMES))):
        raise ValueError("joint YAML must preserve the canonical 20-joint order")
    return tuple(specs)


def load_retargeting_config(path: str | Path | None = None) -> RetargetingConfig:
    if path is None or not Path(path).is_file():
        return RetargetingConfig()
    data = _yaml_load(Path(path))
    if data is None:
        return RetargetingConfig()
    if not isinstance(data, Mapping):
        raise ValueError("retargeting YAML root must be a mapping")
    settings = data.get("retargeting", data)
    if not isinstance(settings, Mapping):
        raise ValueError("retargeting settings must be a mapping")
    allowed = {
        "mode", "auto_calibration_frames", "confidence_threshold", "minimum_coverage",
        "smoothing_time_constant_s", "hold_seconds", "return_seconds",
        "mirrored_input", "joint_scales", "joint_signs",
    }
    return RetargetingConfig(**{key: value for key, value in settings.items() if key in allowed})


class GeometricRetargeter:
    """Stateful geometric baseline with confidence gating and safe fallback."""

    def __init__(
        self,
        joint_specs: Sequence[JointSpec] | None = None,
        config: RetargetingConfig | None = None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.joint_specs = tuple(joint_specs or DEFAULT_JOINT_SPECS)
        if tuple(spec.name for spec in self.joint_specs) != JOINT_NAMES:
            raise ValueError("joint_specs must use canonical joint order")
        self.config = config or RetargetingConfig()
        self._clock = clock
        self._starts = np.array([spec.start_rad for spec in self.joint_specs])
        self._lower = np.array([spec.lower_rad for spec in self.joint_specs])
        self._upper = np.array([spec.upper_rad for spec in self.joint_specs])
        self._offsets = np.array([spec.zero_offset_rad for spec in self.joint_specs])
        self._signs = np.array([
            spec.retarget_sign * float(self.config.joint_signs.get(spec.name, 1.0))
            for spec in self.joint_specs
        ])
        self._scales = np.array([
            float(self.config.joint_scales.get(spec.name, 1.0))
            for spec in self.joint_specs
        ])
        if not np.isfinite(self._scales).all() or np.any(self._scales < 0.0):
            raise ValueError("joint scales must be finite and non-negative")
        if not np.all(np.isin(self._signs, (-1.0, 1.0))):
            raise ValueError("joint signs must be -1 or +1")
        self._active = np.ones(len(JOINT_NAMES), dtype=bool)
        if self.config.mode == "upper_body":
            self._active[6:18] = False
        self.reset()

    @classmethod
    def from_yaml(
        cls,
        *,
        joints_path: str | Path | None = None,
        retargeting_path: str | Path | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> "GeometricRetargeter":
        return cls(load_joint_specs(joints_path), load_retargeting_config(retargeting_path), clock=clock)

    @property
    def neutral_positions_rad(self) -> NDArray[np.float64]:
        return np.clip(self._starts + self._offsets, self._lower, self._upper).copy()

    @property
    def calibration_progress(self) -> float:
        if self._calibration_target == 0:
            return 1.0
        return min(1.0, len(self._calibration_samples) / self._calibration_target)

    @property
    def is_calibrating(self) -> bool:
        return self._calibration_target > 0

    def reset(self) -> None:
        self._calibration_reference = np.zeros(len(JOINT_NAMES))
        self._calibration_target = self.config.auto_calibration_frames
        self._calibration_samples: list[NDArray[np.float64]] = []
        self._last_output: NDArray[np.float64] | None = None
        self._last_output_timestamp: float | None = None
        self._last_valid_positions: NDArray[np.float64] | None = None
        self._last_valid_timestamp: float | None = None

    def start_calibration(self, sample_count: int = 15) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self._calibration_target = int(sample_count)
        self._calibration_samples = []
        self._last_output = None
        self._last_output_timestamp = None
        self._last_valid_positions = None
        self._last_valid_timestamp = None

    def calibrate(self, frame: SkeletonFrame) -> bool:
        """Immediately use one confident neutral frame as the human reference."""

        if not self._frame_is_valid(frame):
            return False
        raw = compute_human_joint_angles(frame, mirrored_input=self.config.mirrored_input)
        raw[~joint_landmark_validity(
            frame, self.config.confidence_threshold, mirrored_input=self.config.mirrored_input
        )] = np.nan
        finite = np.isfinite(raw) & self._active
        if not np.any(finite):
            return False
        self._calibration_reference[finite] = raw[finite]
        self._calibration_target = 0
        self._calibration_samples = []
        self._last_output = None
        return True

    def _required_landmarks(self):
        return UPPER_BODY_REQUIRED if self.config.mode == "upper_body" else WHOLE_BODY_REQUIRED

    def _frame_is_valid(self, frame: SkeletonFrame) -> bool:
        required = self._required_landmarks()
        return (
            frame.coverage(required, self.config.confidence_threshold) >= self.config.minimum_coverage
            and frame.mean_confidence(required) >= self.config.confidence_threshold
        )

    def _observe_calibration(self, raw: NDArray[np.float64]) -> None:
        if self._calibration_target == 0:
            return
        self._calibration_samples.append(raw.copy())
        if len(self._calibration_samples) < self._calibration_target:
            return
        samples = np.stack(self._calibration_samples)
        # Circular mean avoids the +/-pi discontinuity for head/foot headings.
        reference = np.arctan2(np.nanmean(np.sin(samples), axis=0), np.nanmean(np.cos(samples), axis=0))
        finite = np.isfinite(reference) & self._active
        self._calibration_reference[finite] = reference[finite]
        self._calibration_target = 0
        self._calibration_samples = []
        self._last_output = None

    def _desired(self, raw: NDArray[np.float64]) -> NDArray[np.float64]:
        desired = self.neutral_positions_rad
        finite = np.isfinite(raw) & self._active
        # Every source value is an angle.  Using the shortest circular delta is
        # essential for hip/foot/head headings near the -pi/+pi branch cut.
        # A direct subtraction can otherwise request an almost 2*pi motor jump
        # from two physically adjacent poses.
        delta = np.arctan2(
            np.sin(raw - self._calibration_reference),
            np.cos(raw - self._calibration_reference),
        )
        desired[finite] += self._signs[finite] * self._scales[finite] * (
            delta[finite]
        )
        if self._last_valid_positions is not None:
            missing = self._active & ~finite
            desired[missing] = self._last_valid_positions[missing]
        return np.clip(desired, self._lower, self._upper)

    def _smooth(self, desired: NDArray[np.float64], timestamp_s: float) -> NDArray[np.float64]:
        if self._last_output is None or self._last_output_timestamp is None:
            result = desired.copy()
        elif timestamp_s < self._last_output_timestamp:
            result = desired.copy()
        elif self.config.smoothing_time_constant_s == 0.0:
            result = desired.copy()
        else:
            dt = timestamp_s - self._last_output_timestamp
            alpha = 1.0 - exp(-dt / self.config.smoothing_time_constant_s)
            result = self._last_output + alpha * (desired - self._last_output)
        self._last_output = result
        self._last_output_timestamp = timestamp_s
        return result

    def _fallback(self, timestamp_s: float) -> RobotJointCommand:
        neutral = self.neutral_positions_rad
        if self._last_valid_positions is None or self._last_valid_timestamp is None:
            positions = neutral
        else:
            stale_for = max(0.0, timestamp_s - self._last_valid_timestamp)
            if stale_for <= self.config.hold_seconds:
                positions = self._last_valid_positions.copy()
            else:
                progress = min(
                    1.0,
                    (stale_for - self.config.hold_seconds) / self.config.return_seconds,
                )
                positions = (1.0 - progress) * self._last_valid_positions + progress * neutral
        positions = np.clip(positions, self._lower, self._upper)
        self._last_output = positions.copy()
        self._last_output_timestamp = timestamp_s
        return RobotJointCommand.humanoid(timestamp_s, positions, 0.0, stale=True)

    def retarget(
        self,
        frame: SkeletonFrame | None,
        *,
        timestamp_s: float | None = None,
    ) -> RobotJointCommand:
        """Produce one safe command; ``None`` advances stale-pose handling."""

        now = float(timestamp_s if timestamp_s is not None else (frame.timestamp_s if frame else self._clock()))
        if frame is None or not self._frame_is_valid(frame):
            return self._fallback(now)
        raw = compute_human_joint_angles(frame, mirrored_input=self.config.mirrored_input)
        raw[~joint_landmark_validity(
            frame, self.config.confidence_threshold, mirrored_input=self.config.mirrored_input
        )] = np.nan
        if not np.isfinite(raw[self._active]).any():
            return self._fallback(now)
        was_calibrating = self.is_calibrating
        self._observe_calibration(raw)
        if was_calibrating:
            # Calibration is a deliberate safe state: collect only confident
            # observations and keep every motor at its configured home target.
            # The first pose-relative command is emitted on the following
            # frame, after the full reference window has been accepted.
            positions = self.neutral_positions_rad
            self._last_output = positions.copy()
            self._last_output_timestamp = now
            self._last_valid_positions = positions.copy()
            self._last_valid_timestamp = now
            confidence = min(
                1.0,
                max(0.0, frame.mean_confidence(self._required_landmarks())),
            )
            return RobotJointCommand.humanoid(
                now,
                positions,
                confidence,
                stale=False,
            )
        desired = self._desired(raw)
        positions = self._smooth(desired, now)
        confidence = min(1.0, max(0.0, frame.mean_confidence(self._required_landmarks())))
        self._last_valid_positions = positions.copy()
        self._last_valid_timestamp = now
        return RobotJointCommand.humanoid(now, positions, confidence, stale=False)
