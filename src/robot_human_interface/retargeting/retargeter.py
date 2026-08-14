"""Configuration, calibration, filtering, and safe stale-pose behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, pi
from numbers import Real
from pathlib import Path
from time import monotonic
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from robot_human_interface.pose.calibration import NeutralCalibrationGate
from robot_human_interface.skeleton import (
    JOINT_NAMES,
    PoseLandmark as L,
    RobotJointCommand,
    SkeletonFrame,
)

from .geometry import (
    UPPER_BODY_REQUIRED,
    WHOLE_BODY_REQUIRED,
    compute_human_joint_angles,
    joint_landmark_validity,
)


def _finite_real(value: object, *, field_name: str) -> float:
    """Return one schema number without accepting YAML booleans or NaN/Inf."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"{field_name} must be a real number (booleans are not accepted)"
        )
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _exact_int(value: object, *, field_name: str) -> int:
    """Return one schema integer, rejecting bool, float, and string coercion."""

    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer (without coercion)")
    return value


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
        _exact_int(self.index, field_name="joint.index")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("joint.name must be a non-empty string")
        if not isinstance(self.axis, str) or not self.axis:
            raise ValueError(f"joint {self.name} axis must be a non-empty string")
        for field_name in (
            "lower_rad",
            "upper_rad",
            "start_rad",
            "zero_offset_rad",
            "retarget_sign",
        ):
            _finite_real(
                getattr(self, field_name),
                field_name=f"joint {self.name} {field_name}",
            )
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

_CALIBRATION_LOWER_BODY = (
    L.LEFT_KNEE,
    L.RIGHT_KNEE,
    L.LEFT_ANKLE,
    L.RIGHT_ANKLE,
)


@dataclass(frozen=True, slots=True)
class RetargetingConfig:
    mode: str = "whole_body"
    auto_calibration_frames: int = 30
    calibration_max_observations: int = 150
    calibration_max_pose_spread_ratio: float = 0.06
    calibration_max_ankle_offset_ratio: float = 0.08
    calibration_max_ankle_spread_ratio: float = 0.035
    calibration_max_arm_deviation_rad: float = pi / 3.0
    calibration_max_upper_arm_deviation_rad: float = pi / 3.0
    calibration_max_elbow_flexion_rad: float = pi / 3.0
    calibration_max_knee_flexion_rad: float = pi / 4.0
    confidence_threshold: float = 0.55
    minimum_coverage: float = 0.75
    smoothing_time_constant_s: float = 0.10
    hold_seconds: float = 0.35
    return_seconds: float = 1.0
    mirrored_input: bool = False
    joint_scales: Mapping[str, float] = field(default_factory=dict)
    joint_signs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.mirrored_input) is not bool:
            raise ValueError("retargeting.mirrored_input must be a boolean")
        _require_real_config_fields(
            self,
            (
                name
                for name in type(self).__dataclass_fields__
                if name not in {"mode", "mirrored_input", "joint_scales", "joint_signs"}
            ),
            section="retargeting",
        )
        if self.mode not in {"upper_body", "whole_body"}:
            raise ValueError("mode must be upper_body or whole_body")
        if (
            not np.isfinite(float(self.auto_calibration_frames))
            or int(self.auto_calibration_frames) != self.auto_calibration_frames
            or self.auto_calibration_frames < 0
        ):
            raise ValueError("auto_calibration_frames must be a non-negative integer")
        object.__setattr__(self, "auto_calibration_frames", int(self.auto_calibration_frames))
        if (
            not np.isfinite(float(self.calibration_max_observations))
            or int(self.calibration_max_observations)
            != self.calibration_max_observations
            or self.calibration_max_observations <= 0
        ):
            raise ValueError("calibration_max_observations must be a positive integer")
        object.__setattr__(
            self,
            "calibration_max_observations",
            int(self.calibration_max_observations),
        )
        if (
            self.auto_calibration_frames > 0
            and self.calibration_max_observations < self.auto_calibration_frames
        ):
            raise ValueError(
                "calibration_max_observations must cover auto_calibration_frames"
            )
        calibration_limits = (
            self.calibration_max_pose_spread_ratio,
            self.calibration_max_ankle_offset_ratio,
            self.calibration_max_ankle_spread_ratio,
            self.calibration_max_arm_deviation_rad,
            self.calibration_max_upper_arm_deviation_rad,
            self.calibration_max_elbow_flexion_rad,
            self.calibration_max_knee_flexion_rad,
        )
        if not np.isfinite(calibration_limits).all() or any(
            value <= 0.0 for value in calibration_limits
        ):
            raise ValueError("calibration stillness/neutral limits must be finite and positive")
        if (
            self.calibration_max_arm_deviation_rad >= pi
            or self.calibration_max_upper_arm_deviation_rad >= pi
            or self.calibration_max_elbow_flexion_rad >= pi
            or self.calibration_max_knee_flexion_rad >= pi
        ):
            raise ValueError("calibration posture angle limits must be below pi")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be within [0, 1]")
        if self.smoothing_time_constant_s < 0.0 or self.hold_seconds < 0.0:
            raise ValueError("time constants must be non-negative")
        if self.return_seconds <= 0.0:
            raise ValueError("return_seconds must be positive")
        if not isinstance(self.joint_scales, Mapping):
            raise ValueError("joint_scales must be a mapping")
        if not isinstance(self.joint_signs, Mapping):
            raise ValueError("joint_signs must be a mapping")
        unknown = (set(self.joint_scales) | set(self.joint_signs)) - set(JOINT_NAMES)
        if unknown:
            raise ValueError(f"unknown joint names in retargeting config: {sorted(unknown)}")
        for setting_name, values in (
            ("joint_scales", self.joint_scales),
            ("joint_signs", self.joint_signs),
        ):
            for joint_name, value in values.items():
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(
                        f"retargeting.{setting_name}.{joint_name} must be a real "
                        "number (booleans are not accepted)"
                    )
        scales = np.asarray(tuple(self.joint_scales.values()), dtype=np.float64)
        if scales.size and (not np.isfinite(scales).all() or np.any(scales < 0.0)):
            raise ValueError("joint_scales values must be finite and non-negative")
        if any(float(value) not in (-1.0, 1.0) for value in self.joint_signs.values()):
            raise ValueError("joint_signs values must be -1 or +1")


def _require_real_config_fields(
    config: object,
    field_names: Iterable[str],
    *,
    section: str,
) -> None:
    for name in field_names:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"{section}.{name} must be a real number "
                "(booleans are not accepted)"
            )


def _yaml_load(path: Path) -> object:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("loading YAML configuration requires PyYAML") from error
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


_JOINT_YAML_ROOT_KEYS = {
    "schema_version",
    "robot_model",
    "parameter_status",
    "source",
    "units",
    "coordinates",
    "physical_parameters",
    "joint_order",
    "joints",
}

_JOINT_YAML_RECORD_KEYS = {
    "index",
    "name",
    "parent",
    "axis",
    "unity_axis",
    "unity_anchor",
    "mujoco_axis",
    "limits_rad",
    "limit_rad",
    "limits_deg",
    "limit_deg",
    "range_deg",
    "min_deg",
    "max_deg",
    "home_rad",
    "home_deg",
    "start_deg",
    "zero_offset_rad",
    "zero_offset_deg",
    "retarget_sign",
    "mass_kg",
}

_JOINT_YAML_NUMERIC_SCALARS = {
    "min_deg",
    "max_deg",
    "home_rad",
    "home_deg",
    "start_deg",
    "zero_offset_rad",
    "zero_offset_deg",
    "retarget_sign",
    "mass_kg",
}

_JOINT_YAML_NUMERIC_VECTORS = {
    "unity_anchor",
    "mujoco_axis",
    "limits_rad",
    "limit_rad",
    "limits_deg",
    "limit_deg",
    "range_deg",
}


def _validated_numeric_sequence(
    value: object,
    *,
    field_name: str,
    length: int | None = None,
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a numeric sequence")
    if length is not None and len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} numbers")
    return tuple(
        _finite_real(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _validate_joint_record(record: Mapping[object, object], *, position: int) -> None:
    non_string_keys = [key for key in record if not isinstance(key, str)]
    if non_string_keys:
        raise ValueError(
            f"joint record {position} contains non-string key(s): {non_string_keys}"
        )
    unknown = set(record) - _JOINT_YAML_RECORD_KEYS
    if unknown:
        raise ValueError(
            f"joint record {position} contains unknown key(s): {sorted(unknown)}"
        )
    if "index" in record:
        _exact_int(record["index"], field_name=f"joint record {position} index")
    for key in _JOINT_YAML_NUMERIC_SCALARS & set(record):
        _finite_real(record[key], field_name=f"joint record {position} {key}")
    for key in _JOINT_YAML_NUMERIC_VECTORS & set(record):
        _validated_numeric_sequence(
            record[key],
            field_name=f"joint record {position} {key}",
            length=2 if "limit" in key or key == "range_deg" else 3,
        )
    if "unity_axis" in record:
        _validated_numeric_sequence(
            record["unity_axis"],
            field_name=f"joint record {position} unity_axis",
            length=3,
        )
    if "axis" in record and not isinstance(record["axis"], str):
        _validated_numeric_sequence(
            record["axis"],
            field_name=f"joint record {position} axis",
            length=3,
        )
    for key in ("name", "parent"):
        if key in record and not isinstance(record[key], str):
            raise ValueError(f"joint record {position} {key} must be a string")


def load_joint_specs(path: str | Path | None = None) -> tuple[JointSpec, ...]:
    """Load the shared joint file, falling back to verified Unity constants."""

    if path is None or not Path(path).is_file():
        return DEFAULT_JOINT_SPECS
    data = _yaml_load(Path(path))
    if not isinstance(data, Mapping):
        raise ValueError("joint YAML root must be a mapping")
    if "joints" in data:
        non_string_root_keys = [key for key in data if not isinstance(key, str)]
        if non_string_root_keys:
            raise ValueError(
                "joint YAML contains non-string top-level key(s): "
                f"{non_string_root_keys}"
            )
        unknown_root = set(data) - _JOINT_YAML_ROOT_KEYS
        if unknown_root:
            raise ValueError(
                "joint YAML contains unknown top-level key(s): "
                f"{sorted(unknown_root)}"
            )
        if "schema_version" in data:
            _exact_int(data["schema_version"], field_name="joint YAML schema_version")
        records = data["joints"]
    else:
        unknown_root = set(data) - set(JOINT_NAMES)
        missing_root = set(JOINT_NAMES) - set(data)
        if unknown_root or missing_root:
            details = []
            if unknown_root:
                details.append(f"unknown key(s): {sorted(unknown_root)}")
            if missing_root:
                details.append(f"missing joint(s): {sorted(missing_root)}")
            raise ValueError("joint YAML name mapping has " + "; ".join(details))
        records = data
    if isinstance(records, Mapping):
        iterable = []
        for name in JOINT_NAMES:
            record = records[name]
            if not isinstance(record, Mapping):
                raise ValueError(f"joint {name} record must be a mapping")
            if "name" in record and record["name"] != name:
                raise ValueError(
                    f"joint {name} record has conflicting name {record['name']!r}"
                )
            iterable.append(dict(record, name=name))
    elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        iterable = list(records)
    else:
        raise ValueError("joints must be a sequence or name-to-spec mapping")
    specs: list[JointSpec] = []
    for default_index, record in enumerate(iterable):
        if not isinstance(record, Mapping):
            raise ValueError("each joint record must be a mapping")
        _validate_joint_record(record, position=default_index)
        name_value = record.get("name", "")
        if not isinstance(name_value, str):
            raise ValueError(f"joint record {default_index} name must be a string")
        name = name_value
        index = _exact_int(
            record.get("index", default_index),
            field_name=f"joint {name or default_index} index",
        )
        limits_rad = record.get("limits_rad", record.get("limit_rad"))
        limits_deg = record.get("limits_deg", record.get("limit_deg", record.get("range_deg")))
        if limits_rad is None and limits_deg is None:
            limits_deg = (record.get("min_deg"), record.get("max_deg"))
        limits = limits_rad if limits_rad is not None else limits_deg
        if not isinstance(limits, Sequence) or len(limits) != 2:
            raise ValueError(f"joint {name} needs two angular limits")
        if limits_rad is not None:
            lower_rad, upper_rad = (
                _finite_real(value, field_name=f"joint {name} limits_rad[{offset}]")
                for offset, value in enumerate(limits)
            )
        else:
            lower_rad, upper_rad = (
                _finite_real(value, field_name=f"joint {name} limits_deg[{offset}]")
                * pi
                / 180
                for offset, value in enumerate(limits)
            )
        if "home_rad" in record:
            start_rad = _finite_real(
                record["home_rad"], field_name=f"joint {name} home_rad"
            )
        else:
            start_rad = (
                _finite_real(
                    record.get("home_deg", record.get("start_deg", 0.0)),
                    field_name=f"joint {name} home_deg",
                )
                * pi
                / 180
            )
        if "zero_offset_rad" in record:
            zero_offset_rad = _finite_real(
                record["zero_offset_rad"],
                field_name=f"joint {name} zero_offset_rad",
            )
        else:
            zero_offset_rad = (
                _finite_real(
                    record.get("zero_offset_deg", 0.0),
                    field_name=f"joint {name} zero_offset_deg",
                )
                * pi
                / 180
            )
        axis_value = record.get("axis", record.get("unity_axis", ""))
        if isinstance(axis_value, Sequence) and not isinstance(axis_value, (str, bytes)):
            vector = _validated_numeric_sequence(
                axis_value,
                field_name=f"joint {name} axis",
                length=3,
            )
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
                retarget_sign=_finite_real(
                    record.get("retarget_sign", 1.0),
                    field_name=f"joint {name} retarget_sign",
                ),
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
    if "retargeting" in data:
        unknown_root = set(data) - {"retargeting"}
        if unknown_root:
            raise ValueError(
                "retargeting YAML has unknown top-level key(s): "
                f"{sorted(unknown_root)}"
            )
        settings = data["retargeting"]
    else:
        settings = data
    if not isinstance(settings, Mapping):
        raise ValueError("retargeting settings must be a mapping")
    allowed = {
        "mode", "auto_calibration_frames", "confidence_threshold", "minimum_coverage",
        "smoothing_time_constant_s", "hold_seconds", "return_seconds",
        "mirrored_input", "joint_scales", "joint_signs",
        "calibration_max_observations", "calibration_max_pose_spread_ratio",
        "calibration_max_ankle_offset_ratio", "calibration_max_ankle_spread_ratio",
        "calibration_max_arm_deviation_rad", "calibration_max_knee_flexion_rad",
        "calibration_max_upper_arm_deviation_rad",
        "calibration_max_elbow_flexion_rad",
    }
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(
            "retargeting settings contain unknown key(s): " f"{sorted(unknown)}"
        )
    return RetargetingConfig(**dict(settings))


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
        assert self._calibration_gate is not None
        return self._calibration_gate.progress

    @property
    def is_calibrating(self) -> bool:
        return self._calibration_target > 0

    def reset(self) -> None:
        self._calibration_reference = np.zeros(len(JOINT_NAMES))
        self._calibration_target = self.config.auto_calibration_frames
        # With calibration explicitly disabled, retain the historical
        # zero-angle reference behavior.  Once any calibration is requested,
        # the accepted sample window becomes authoritative channel-by-channel.
        self._calibration_reference_valid = np.full(
            len(JOINT_NAMES),
            self._calibration_target == 0,
            dtype=bool,
        )
        self._calibration_reference_valid &= self._active
        self._calibration_samples: list[NDArray[np.float64]] = []
        self._calibration_gate = self._new_calibration_gate(
            self._calibration_target
        )
        self._last_output: NDArray[np.float64] | None = None
        self._last_output_timestamp: float | None = None
        self._last_valid_positions: NDArray[np.float64] | None = None
        self._last_valid_timestamp: float | None = None

    def start_calibration(self, sample_count: int = 15) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self._calibration_target = int(sample_count)
        self._calibration_reference = np.zeros(len(JOINT_NAMES))
        self._calibration_reference_valid = np.zeros(len(JOINT_NAMES), dtype=bool)
        self._calibration_samples = []
        self._calibration_gate = self._new_calibration_gate(self._calibration_target)
        self._last_output = None
        self._last_output_timestamp = None
        self._last_valid_positions = None
        self._last_valid_timestamp = None

    def _new_calibration_gate(
        self, sample_count: int
    ) -> NeutralCalibrationGate | None:
        if sample_count == 0:
            return None
        landmarks = self._required_landmarks()
        if self.config.mode == "whole_body":
            # Heel/toe confidence is not needed to prove stillness; the major
            # leg chain plus the explicit ankle-level gate is sufficient and
            # remains observable on ordinary camera footage.
            landmarks = UPPER_BODY_REQUIRED + _CALIBRATION_LOWER_BODY
        return NeutralCalibrationGate(
            sample_count=sample_count,
            max_observations=max(
                sample_count, self.config.calibration_max_observations
            ),
            landmark_indices=landmarks,
            confidence_threshold=self.config.confidence_threshold,
            max_pose_spread_ratio=self.config.calibration_max_pose_spread_ratio,
            require_double_support=self.config.mode == "whole_body",
            max_ankle_offset_ratio=self.config.calibration_max_ankle_offset_ratio,
            max_ankle_spread_ratio=self.config.calibration_max_ankle_spread_ratio,
            max_arm_deviation_rad=self.config.calibration_max_arm_deviation_rad,
            max_upper_arm_deviation_rad=(
                self.config.calibration_max_upper_arm_deviation_rad
            ),
            max_elbow_flexion_rad=self.config.calibration_max_elbow_flexion_rad,
            max_knee_flexion_rad=self.config.calibration_max_knee_flexion_rad,
            require_extended_legs=self.config.mode == "whole_body",
            mirrored_input=self.config.mirrored_input,
            label="retargeting neutral-pose",
        )

    def calibrate(self, frame: SkeletonFrame) -> bool:
        """Explicitly override auto-admission with one caller-approved frame."""

        if not self._frame_is_valid(frame):
            return False
        gate = self._new_calibration_gate(1)
        assert gate is not None
        if not gate.accepts_explicit(frame):
            return False
        raw = compute_human_joint_angles(frame, mirrored_input=self.config.mirrored_input)
        raw[~joint_landmark_validity(
            frame, self.config.confidence_threshold, mirrored_input=self.config.mirrored_input
        )] = np.nan
        finite = np.isfinite(raw) & self._active
        if not np.any(finite):
            return False
        self._calibration_reference[:] = 0.0
        self._calibration_reference[finite] = raw[finite]
        self._calibration_reference_valid = finite.copy()
        self._calibration_target = 0
        self._calibration_samples = []
        self._calibration_gate = None
        self._last_output = None
        self._last_output_timestamp = None
        # An explicit calibration starts a new command-reference epoch.  A
        # subsequently missing frame must fall back to the newly calibrated
        # neutral pose, never resurrect a valid command from the old epoch.
        self._last_valid_positions = None
        self._last_valid_timestamp = None
        return True

    def _required_landmarks(self):
        return UPPER_BODY_REQUIRED if self.config.mode == "upper_body" else WHOLE_BODY_REQUIRED

    def _frame_is_valid(self, frame: SkeletonFrame) -> bool:
        required = self._required_landmarks()
        return (
            frame.coverage(required, self.config.confidence_threshold) >= self.config.minimum_coverage
            and frame.mean_confidence(required) >= self.config.confidence_threshold
        )

    def _observe_calibration(self, frame: SkeletonFrame) -> None:
        if self._calibration_target == 0:
            return
        assert self._calibration_gate is not None
        accepted = self._calibration_gate.observe(frame)
        if accepted is None:
            return
        accepted_raw: list[NDArray[np.float64]] = []
        for sample in accepted:
            raw = compute_human_joint_angles(
                sample, mirrored_input=self.config.mirrored_input
            )
            raw[~joint_landmark_validity(
                sample,
                self.config.confidence_threshold,
                mirrored_input=self.config.mirrored_input,
            )] = np.nan
            accepted_raw.append(raw)
        samples = np.stack(accepted_raw)
        # Circular mean avoids the +/-pi discontinuity for head/foot headings.
        sample_valid = np.isfinite(samples)
        reference = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        # A neutral reference must be supported by the complete accepted
        # window.  One intermittent optional landmark is not enough evidence
        # to arm a motor channel from a single potentially noisy angle.
        finite = np.all(sample_valid, axis=0) & self._active
        reference[finite] = np.arctan2(
            np.mean(np.sin(samples[:, finite]), axis=0),
            np.mean(np.cos(samples[:, finite]), axis=0),
        )
        self._calibration_reference[:] = 0.0
        self._calibration_reference[finite] = reference[finite]
        self._calibration_reference_valid = finite.copy()
        self._calibration_target = 0
        self._calibration_samples = []
        self._calibration_gate = None
        self._last_output = None

    def _desired(self, raw: NDArray[np.float64]) -> NDArray[np.float64]:
        desired = self.neutral_positions_rad
        available = self._active & self._calibration_reference_valid
        finite = np.isfinite(raw) & available
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
            # Observation loss may hold the last safe command only for
            # channels that actually acquired a neutral reference.  Channels
            # absent from the accepted calibration window remain at home.
            missing = available & ~finite
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
        self._observe_calibration(frame)
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
