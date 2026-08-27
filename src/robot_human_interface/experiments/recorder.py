"""Crash-tolerant recording of reproducible humanoid experiments.

The recorder deliberately has no Qt dependency.  A GUI or worker supplies the
application-specific documents directory and appends immutable numeric
samples.  One private I/O thread owns all writes for an active run.

The on-disk format is intentionally boring and inspectable: YAML metadata,
JSON Lines events, and NumPy ``.npz`` chunks containing no object arrays.  A
run is written below ``<run_id>.partial`` and is renamed only after every
queued sample and the final manifest have reached disk successfully.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from queue import Full, Queue
from threading import RLock, Thread
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np
import yaml

from robot_human_interface.skeleton import JOINT_NAMES, LANDMARK_COUNT


EXPERIMENT_SCHEMA_VERSION = "1.0"
DEFAULT_SAMPLE_QUEUE_CAPACITY = 512
DEFAULT_CHUNK_SIZE = 128
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


class RecorderState(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    RECORDING = "recording"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    ERROR = "error"


def _required_text(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    if any(ord(character) < 32 for character in text):
        raise ValueError(f"{name} must not contain control characters")
    return text


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Operator-entered identity and reproducibility metadata for one run."""

    participant_code: str
    movement: str
    attempt: int
    method_id: str
    seed: int
    note: str = ""
    consent: bool = False
    checkpoint: str | None = None
    checkpoint_hash: str | None = None
    record_video: bool = False

    def __post_init__(self) -> None:
        participant = _required_text(self.participant_code, "participant_code")
        movement = _required_text(self.movement, "movement")
        method = _required_text(self.method_id, "method_id")
        if isinstance(self.attempt, bool) or int(self.attempt) != self.attempt:
            raise ValueError("attempt must be a positive integer")
        if int(self.attempt) <= 0:
            raise ValueError("attempt must be a positive integer")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise ValueError("seed must be an integer")
        if type(self.consent) is not bool or type(self.record_video) is not bool:
            raise ValueError("consent and record_video must be booleans")
        if self.record_video and not self.consent:
            raise ValueError("video recording requires explicit consent")
        note = str(self.note).strip()
        checkpoint = (
            None
            if self.checkpoint is None
            else _required_text(self.checkpoint, "checkpoint")
        )
        checkpoint_hash = (
            None
            if self.checkpoint_hash is None
            else _required_text(self.checkpoint_hash, "checkpoint_hash")
        )
        object.__setattr__(self, "participant_code", participant)
        object.__setattr__(self, "movement", movement)
        object.__setattr__(self, "attempt", int(self.attempt))
        object.__setattr__(self, "method_id", method)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "note", note)
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(self, "checkpoint_hash", checkpoint_hash)

    @property
    def video_enabled(self) -> bool:
        """Readable alias used by the GUI layer."""

        return self.record_video


def _optional_array(
    value: object | None,
    *,
    name: str,
    shape: tuple[int | None, ...],
) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(result.shape, shape, strict=True)
    ):
        rendered = tuple("*" if item is None else item for item in shape)
        raise ValueError(f"{name} must have shape {rendered}, got {result.shape}")
    if np.isinf(result).any():
        raise ValueError(f"{name} must not contain infinite values")
    result = result.copy()
    result.setflags(write=False)
    return result


def _optional_scalar(value: object | None, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite when supplied")
    return number


def _optional_bool(value: object | None, name: str) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean or None")
    return value


def _freeze_mapping(value: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if value is None:
        return None
    return MappingProxyType({str(key): _copy_value(item) for key, item in value.items()})


def _copy_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.setflags(write=False)
        return result
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _copy_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_copy_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ExperimentSample:
    """One normalized sample accepted by :class:`ExperimentRecorder`.

    Optional numeric values are stored as NaN plus an equally shaped ``_mask``
    array in NPZ.  This keeps every chunk numeric and safe to load with
    ``allow_pickle=False`` while distinguishing missing data from real zeros.
    """

    timestamp_s: float
    sequence: int
    landmarks_2d: np.ndarray | None = None
    landmarks_3d: np.ndarray | None = None
    visibility: np.ndarray | None = None
    presence: np.ndarray | None = None
    raw_angles_rad: np.ndarray | None = None
    safe_angles_rad: np.ndarray | None = None
    actual_angles_rad: np.ndarray | None = None
    joint_velocities_rad_s: np.ndarray | None = None
    base_position_m: np.ndarray | None = None
    base_orientation_wxyz: np.ndarray | None = None
    base_linear_velocity_m_s: np.ndarray | None = None
    base_angular_velocity_rad_s: np.ndarray | None = None
    center_of_mass_position_m: np.ndarray | None = None
    right_foot_position_m: np.ndarray | None = None
    left_foot_position_m: np.ndarray | None = None
    right_foot_linear_velocity_m_s: np.ndarray | None = None
    left_foot_linear_velocity_m_s: np.ndarray | None = None
    right_foot_normal_force_n: float | None = None
    left_foot_normal_force_n: float | None = None
    actuator_forces: np.ndarray | None = None
    contact_count: int | None = None
    non_foot_ground_contact_count: int | None = None
    support_intent: str | None = None
    support_phase: str | None = None
    diagnostics: Mapping[str, object] | None = None
    tracking_quality: float | None = None
    calibrating: bool | None = None
    calibration_progress: float | None = None
    command_stale: bool | None = None
    safe_valid: bool | None = None
    free_base_active: bool | None = None
    balance_active: bool | None = None

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if isinstance(self.sequence, bool) or int(self.sequence) != self.sequence:
            raise ValueError("sequence must be a non-negative integer")
        if int(self.sequence) < 0:
            raise ValueError("sequence must be a non-negative integer")
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "sequence", int(self.sequence))

        shapes: dict[str, tuple[int | None, ...]] = {
            "landmarks_2d": (LANDMARK_COUNT, 2),
            "landmarks_3d": (LANDMARK_COUNT, 3),
            "visibility": (LANDMARK_COUNT,),
            "presence": (LANDMARK_COUNT,),
            "raw_angles_rad": (None,),
            "safe_angles_rad": (None,),
            "actual_angles_rad": (None,),
            "joint_velocities_rad_s": (None,),
            "base_position_m": (3,),
            "base_orientation_wxyz": (4,),
            "base_linear_velocity_m_s": (3,),
            "base_angular_velocity_rad_s": (3,),
            "center_of_mass_position_m": (3,),
            "right_foot_position_m": (3,),
            "left_foot_position_m": (3,),
            "right_foot_linear_velocity_m_s": (3,),
            "left_foot_linear_velocity_m_s": (3,),
            "actuator_forces": (None,),
        }
        for name, shape in shapes.items():
            object.__setattr__(
                self,
                name,
                _optional_array(getattr(self, name), name=name, shape=shape),
            )

        for name in (
            "right_foot_normal_force_n",
            "left_foot_normal_force_n",
            "tracking_quality",
            "calibration_progress",
        ):
            object.__setattr__(
                self,
                name,
                _optional_scalar(getattr(self, name), name),
            )
        for name in (
            "calibrating",
            "command_stale",
            "safe_valid",
            "free_base_active",
            "balance_active",
        ):
            object.__setattr__(self, name, _optional_bool(getattr(self, name), name))
        for name in ("contact_count", "non_foot_ground_contact_count"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or int(value) != value or int(value) < 0:
                    raise ValueError(f"{name} must be a non-negative integer or None")
                object.__setattr__(self, name, int(value))
        for name in ("support_intent", "support_phase"):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                None if value is None or not str(value).strip() else str(value).strip(),
            )
        object.__setattr__(self, "diagnostics", _freeze_mapping(self.diagnostics))

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "ExperimentSample":
        """Build a sample from a PipelineSnapshot-like object by duck typing.

        The method intentionally avoids importing the session module, which
        keeps the recorder reusable in synthetic and offline research tools.
        Future pipelines may expose a complete simulation state as
        ``telemetry['humanoid_state']``; current individual telemetry keys are
        supported as a fallback.
        """

        timestamp_s = getattr(snapshot, "timestamp_s")
        sequence = getattr(snapshot, "sequence")
        skeleton = getattr(snapshot, "skeleton", None)
        raw_command = getattr(snapshot, "raw_command", None)
        safe_command = getattr(snapshot, "safe_command", None)
        telemetry = getattr(snapshot, "telemetry", {})
        if not isinstance(telemetry, Mapping):
            telemetry = {}
        state = next(
            (
                telemetry[key]
                for key in ("humanoid_state", "simulation_state", "state")
                if key in telemetry and telemetry[key] is not None
            ),
            None,
        )

        def telemetry_or_state(key: str, state_attribute: str | None = None) -> object:
            if key in telemetry:
                return telemetry[key]
            return getattr(state, state_attribute or key, None) if state is not None else None

        diagnostics: dict[str, object] = {}
        for key in ("support_diagnostics", "balance_diagnostics", "diagnostics"):
            if key in telemetry and telemetry[key] is not None:
                diagnostics[key] = telemetry[key]
        safe_valid = telemetry.get("safe_valid")
        if safe_valid is None:
            safe_valid = safe_command is not None and not bool(
                getattr(safe_command, "stale", True)
            )
        return cls(
            timestamp_s=timestamp_s,
            sequence=sequence,
            landmarks_2d=getattr(skeleton, "landmarks_2d", None),
            landmarks_3d=getattr(skeleton, "landmarks_3d", None),
            visibility=getattr(skeleton, "visibility", None),
            presence=getattr(skeleton, "presence", None),
            raw_angles_rad=getattr(raw_command, "positions_rad", None),
            safe_angles_rad=getattr(safe_command, "positions_rad", None),
            actual_angles_rad=telemetry_or_state(
                "joint_positions_rad", "joint_positions_rad"
            ),
            joint_velocities_rad_s=telemetry_or_state(
                "joint_velocities_rad_s", "joint_velocities_rad_s"
            ),
            base_position_m=telemetry_or_state("base_position_m"),
            base_orientation_wxyz=telemetry_or_state("base_orientation_wxyz"),
            base_linear_velocity_m_s=telemetry_or_state("base_linear_velocity_m_s"),
            base_angular_velocity_rad_s=telemetry_or_state(
                "base_angular_velocity_rad_s"
            ),
            center_of_mass_position_m=telemetry_or_state(
                "center_of_mass_position_m"
            ),
            right_foot_position_m=telemetry_or_state("right_foot_position_m"),
            left_foot_position_m=telemetry_or_state("left_foot_position_m"),
            right_foot_linear_velocity_m_s=telemetry_or_state(
                "right_foot_linear_velocity_m_s"
            ),
            left_foot_linear_velocity_m_s=telemetry_or_state(
                "left_foot_linear_velocity_m_s"
            ),
            right_foot_normal_force_n=telemetry_or_state(
                "right_foot_force_n", "right_foot_normal_force_n"
            ),
            left_foot_normal_force_n=telemetry_or_state(
                "left_foot_force_n", "left_foot_normal_force_n"
            ),
            actuator_forces=telemetry_or_state("actuator_forces"),
            contact_count=telemetry_or_state("contact_count"),
            non_foot_ground_contact_count=telemetry_or_state(
                "non_foot_ground_contact_count"
            ),
            support_intent=telemetry.get("support_intent"),
            support_phase=telemetry.get("support_phase"),
            diagnostics=diagnostics or None,
            tracking_quality=getattr(snapshot, "tracking_quality", None),
            calibrating=telemetry.get("calibrating"),
            calibration_progress=telemetry.get("calibration_progress"),
            command_stale=telemetry.get("command_stale"),
            safe_valid=safe_valid,
            free_base_active=telemetry.get(
                "free_base_active", getattr(safe_command, "free_base_active", None)
            ),
            balance_active=telemetry.get(
                "balance_active", getattr(safe_command, "balance_active", None)
            ),
        )


@dataclass(frozen=True, slots=True)
class RecorderSummary:
    run_id: str
    state: RecorderState
    path: Path
    sample_count: int
    accepted_samples: int
    dropped_samples: int
    chunk_count: int
    event_count: int
    started_utc: str
    ended_utc: str | None
    stop_reason: str | None
    incomplete: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _SampleItem:
    sample: ExperimentSample


@dataclass(frozen=True, slots=True)
class _EventItem:
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _StopItem:
    reason: str


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file."""

    candidate = Path(path)
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _looks_absolute_path(value: str) -> bool:
    if "://" in value:
        return False
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _sanitize_string(value: str) -> str:
    if not _looks_absolute_path(value):
        return value
    windows_name = PureWindowsPath(value).name
    posix_name = PurePosixPath(value).name
    return windows_name if windows_name != value else posix_name


def _serializable(value: object) -> object:
    """Convert arbitrary metadata to safe builtins and redact absolute paths."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Enum):
        return _serializable(value.value)
    if isinstance(value, np.generic):
        return _serializable(value.item())
    if isinstance(value, np.ndarray):
        return _serializable(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _serializable(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serializable(item) for item in value]
    return _sanitize_string(str(value))


def _utc_text(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _slug(value: str, fallback: str) -> str:
    rendered = _SLUG_PATTERN.sub("-", value.strip()).strip("-_")
    return rendered[:24] or fallback


def _package_versions() -> dict[str, str]:
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in (
        "robot-human-interface",
        "numpy",
        "PyYAML",
        "opencv-contrib-python",
        "mediapipe",
        "mujoco",
        "PyQt6",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "unavailable"
    return result


def _git_metadata(repo_root: Path | None) -> dict[str, object]:
    if repo_root is None or not (repo_root / ".git").exists():
        return {"revision": None, "dirty": None, "dirty_hash": None}

    def run(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        return completed.stdout

    try:
        revision = run("rev-parse", "HEAD").decode("ascii", "replace").strip()
        status = run("status", "--porcelain=v1", "-z")
        dirty = bool(status)
        dirty_hash = None
        if dirty:
            digest = hashlib.sha256()
            digest.update(status)
            digest.update(run("diff", "--binary", "HEAD"))
            digest.update(run("diff", "--binary", "--cached"))
            # ``git diff`` has no payload for untracked files.  Include both
            # their repository-relative names and bytes so two worktrees with
            # the same ``?? path`` status but different new inputs cannot
            # produce the same provenance digest.
            untracked = run("ls-files", "--others", "--exclude-standard", "-z")
            for relative_bytes in untracked.split(b"\0"):
                if not relative_bytes:
                    continue
                digest.update(b"\0untracked\0")
                digest.update(relative_bytes)
                digest.update(b"\0")
                untracked_path = repo_root / os.fsdecode(relative_bytes)
                with untracked_path.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
            dirty_hash = digest.hexdigest()
        return {"revision": revision, "dirty": dirty, "dirty_hash": dirty_hash}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None, "dirty_hash": None}


def _default_artifact_files(
    repo_root: Path | None,
    session_config: object | None,
    spec: ExperimentSpec,
) -> dict[str, Path]:
    """Discover stable project inputs without leaking their locations."""

    result: dict[str, Path] = {}
    if repo_root is not None:
        lock = repo_root / "uv.lock"
        if lock.is_file():
            result["lock/uv"] = lock
        config_directory = repo_root / "config"
        if config_directory.is_dir():
            for config in sorted(config_directory.glob("*.yaml")):
                if config.is_file():
                    result[f"config/{config.name}"] = config
        model_directory = repo_root / "assets" / "models"
        if model_directory.is_dir():
            for model in sorted(model_directory.iterdir()):
                if model.is_file():
                    result[f"model/{model.name}"] = model
        mujoco_directory = repo_root / "models" / "humanoid"
        if mujoco_directory.is_dir():
            for model_input in sorted(mujoco_directory.rglob("*")):
                if model_input.is_file():
                    relative = model_input.relative_to(mujoco_directory).as_posix()
                    result[f"model/humanoid/{relative}"] = model_input

    source: object | None = None
    if isinstance(session_config, Mapping):
        source = session_config.get("source")
    elif session_config is not None:
        source = getattr(session_config, "source", None)
    if isinstance(source, Mapping):
        source_path = source.get("path")
    else:
        source_path = getattr(source, "path", None)
    if source_path is not None:
        candidate = Path(source_path).expanduser().resolve()
        if candidate.is_file():
            result["source"] = candidate

    if spec.checkpoint:
        checkpoint = Path(spec.checkpoint).expanduser()
        candidates = [checkpoint]
        if repo_root is not None and not checkpoint.is_absolute():
            candidates.append(repo_root / checkpoint)
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                result["checkpoint"] = resolved
                break
    return result


def _atomic_yaml(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(
            _serializable(payload),
            stream,
            allow_unicode=True,
            sort_keys=False,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_json_line(path: Path, payload: Mapping[str, object]) -> None:
    rendered = json.dumps(
        _serializable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def recover_interrupted_experiments(root: str | Path) -> tuple[Path, ...]:
    """Mark unfinished surviving ``.partial`` packages as interrupted.

    No package or sample is deleted.  Terminal ERROR/COMPLETE manifests remain
    truthful and untouched.  A malformed manifest is preserved and a separate
    ``recovery.yaml`` sidecar is written instead.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return ()
    recovered: list[Path] = []
    recovered_at = _utc_text(datetime.now(timezone.utc))
    for partial in sorted(root_path.glob("*/*.partial")):
        if not partial.is_dir():
            continue
        manifest_path = partial / "manifest.yaml"
        try:
            manifest: object = None
            if manifest_path.is_file():
                with manifest_path.open("r", encoding="utf-8") as stream:
                    manifest = yaml.safe_load(stream)
            if isinstance(manifest, Mapping):
                state = str(manifest.get("state", "")).strip().lower()
                if (
                    state in {RecorderState.ERROR.value, RecorderState.COMPLETE.value}
                    and manifest.get("ended_utc")
                    and manifest.get("stop_reason")
                ):
                    # A writer can intentionally leave a fully finalized ERROR
                    # package as ``.partial``.  Its recorded failure reason is
                    # more accurate than startup-time "interrupted" recovery.
                    continue
                updated = dict(manifest)
                updated["state"] = "interrupted"
                updated["complete"] = False
                updated["incomplete"] = True
                updated["stop_reason"] = "interrupted"
                updated["interrupted_at_utc"] = recovered_at
                _atomic_yaml(manifest_path, updated)
            else:
                _atomic_yaml(
                    partial / "recovery.yaml",
                    {
                        "schema_version": EXPERIMENT_SCHEMA_VERSION,
                        "run_id": partial.name.removesuffix(".partial"),
                        "state": "interrupted",
                        "complete": False,
                        "incomplete": True,
                        "stop_reason": "interrupted",
                        "interrupted_at_utc": recovered_at,
                        "manifest_error": "missing_or_invalid_manifest",
                    },
                )
            _append_json_line(
                partial / "events.jsonl",
                {
                    "run_id": partial.name.removesuffix(".partial"),
                    "recorded_utc": recovered_at,
                    "level": "warning",
                    "subsystem": "recorder",
                    "code": "RECORDER_INTERRUPTED_RECOVERED",
                    "message": "Незавершённый опыт обнаружен при запуске",
                    "sequence": None,
                },
            )
            recovered.append(partial)
        except OSError:
            # Recovery is best-effort: one inaccessible package must not stop
            # the application or hide the other recoverable packages.
            continue
    return tuple(recovered)


class ExperimentRecorder:
    """Asynchronous writer for one experiment at a time."""

    def __init__(
        self,
        root: str | Path,
        *,
        queue_capacity: int = DEFAULT_SAMPLE_QUEUE_CAPACITY,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        utc_now: Callable[[], datetime] | None = None,
        repo_root: str | Path | None = None,
        recover_partials: bool = True,
    ) -> None:
        if isinstance(queue_capacity, bool) or int(queue_capacity) <= 0:
            raise ValueError("queue_capacity must be a positive integer")
        if isinstance(chunk_size, bool) or int(chunk_size) <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self._root = Path(root).expanduser().resolve()
        self._queue_capacity = int(queue_capacity)
        self._chunk_size = int(chunk_size)
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        default_repo = Path(__file__).resolve().parents[3]
        self._repo_root = (
            default_repo
            if repo_root is None
            else Path(repo_root).expanduser().resolve()
        )
        self._lock = RLock()
        self._state = RecorderState.IDLE
        self._queue: Queue[_SampleItem | _EventItem | _StopItem] | None = None
        self._thread: Thread | None = None
        self._spec: ExperimentSpec | None = None
        self._run_id: str | None = None
        self._source_id: str | None = None
        self._joint_names: tuple[str, ...] = tuple(JOINT_NAMES)
        self._partial_path: Path | None = None
        self._final_path: Path | None = None
        self._manifest: dict[str, object] = {}
        self._started_utc: str | None = None
        self._ended_utc: str | None = None
        self._stop_reason: str | None = None
        self._accepted_samples = 0
        self._written_samples = 0
        self._dropped_samples = 0
        self._event_count = 0
        self._chunks: list[dict[str, object]] = []
        self._errors: list[str] = []
        self._incomplete = False
        self._last_summary: RecorderSummary | None = None
        self._recovered_paths = (
            recover_interrupted_experiments(self._root)
            if recover_partials
            else ()
        )

    @property
    def state(self) -> RecorderState:
        with self._lock:
            return self._state

    @property
    def recovered_paths(self) -> tuple[Path, ...]:
        return self._recovered_paths

    def recover_interrupted(self) -> tuple[Path, ...]:
        """Rescan the recorder root and mark surviving partial runs."""

        with self._lock:
            if self._state in {
                RecorderState.PREPARING,
                RecorderState.RECORDING,
                RecorderState.FINALIZING,
            }:
                raise RuntimeError("cannot recover partials while recording")
        recovered = recover_interrupted_experiments(self._root)
        with self._lock:
            known = dict.fromkeys((*self._recovered_paths, *recovered))
            self._recovered_paths = tuple(known)
        return recovered

    @property
    def run_id(self) -> str | None:
        with self._lock:
            return self._run_id

    @property
    def media_directory(self) -> Path | None:
        with self._lock:
            if self._spec is None or not self._spec.record_video:
                return None
            if self._partial_path is not None and self._partial_path.exists():
                return self._partial_path / "media"
            if self._final_path is not None and self._final_path.exists():
                return self._final_path / "media"
            return None

    @property
    def summary(self) -> RecorderSummary | None:
        with self._lock:
            if self._run_id is None or self._started_utc is None:
                return self._last_summary
            return self._make_summary_locked()

    def _now(self) -> datetime:
        value = self._utc_now()
        if not isinstance(value, datetime):
            raise TypeError("utc_now must return datetime")
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    def _new_run_id(self, spec: ExperimentSpec, started: datetime) -> str:
        return "_".join(
            (
                started.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
                _slug(spec.participant_code, "participant"),
                _slug(spec.movement, "movement"),
                f"t{spec.attempt:03d}",
                uuid4().hex[:8],
            )
        )

    def start(
        self,
        spec: ExperimentSpec,
        *,
        session_config: object | None = None,
        joint_names: Sequence[str] = JOINT_NAMES,
        source_id: str | None = None,
        artifact_files: Mapping[str, str | Path] | None = None,
        software_versions: Mapping[str, object] | None = None,
        git_metadata: Mapping[str, object] | None = None,
        run_id: str | None = None,
        media_source_file: str | Path | None = None,
    ) -> str:
        """Prepare and begin a run, returning its stable ID.

        ``artifact_files`` maps logical labels (``lock``, ``config``,
        ``model``, ``source``...) to files whose names, sizes and SHA-256
        digests are stored.  Their absolute paths never enter the manifest.
        """

        if not isinstance(spec, ExperimentSpec):
            raise TypeError("spec must be an ExperimentSpec")
        names = tuple(str(name).strip() for name in joint_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("joint_names must be a non-empty unique sequence")
        source = None if source_id is None else _required_text(source_id, "source_id")
        started = self._now()
        chosen_run_id = run_id or self._new_run_id(spec, started)
        if not _RUN_ID_PATTERN.fullmatch(chosen_run_id):
            raise ValueError("run_id contains unsafe characters")
        if media_source_file is not None and not spec.record_video:
            raise ValueError("media_source_file requires record_video=True")

        files_manifest: dict[str, object] = {}
        discovered_files: dict[str, str | Path] = _default_artifact_files(
            self._repo_root,
            session_config,
            spec,
        )
        discovered_files.update(artifact_files or {})
        for logical_name, path_value in discovered_files.items():
            label = _required_text(logical_name, "artifact label")
            candidate = Path(path_value).expanduser().resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"artifact file does not exist: {candidate}")
            files_manifest[label] = {
                "name": candidate.name,
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }

        with self._lock:
            if self._state in {
                RecorderState.PREPARING,
                RecorderState.RECORDING,
                RecorderState.FINALIZING,
            }:
                raise RuntimeError("an experiment is already active")
            self._state = RecorderState.PREPARING
            self._spec = spec
            self._run_id = chosen_run_id
            self._source_id = source
            self._joint_names = names
            self._started_utc = _utc_text(started)
            self._ended_utc = None
            self._stop_reason = None
            self._accepted_samples = 0
            self._written_samples = 0
            self._dropped_samples = 0
            self._event_count = 0
            self._chunks = []
            self._errors = []
            self._incomplete = False
            self._last_summary = None

        day_directory = self._root / started.astimezone(timezone.utc).strftime("%Y-%m-%d")
        partial = day_directory / f"{chosen_run_id}.partial"
        final = day_directory / chosen_run_id
        try:
            if partial.exists() or final.exists():
                raise FileExistsError(f"experiment run already exists: {chosen_run_id}")
            (partial / "chunks").mkdir(parents=True, exist_ok=False)
            if spec.record_video:
                (partial / "media").mkdir()
            if media_source_file is not None:
                media_source = Path(media_source_file).expanduser().resolve()
                if not media_source.is_file():
                    raise FileNotFoundError(f"media source does not exist: {media_source}")
                media_target = partial / "media" / f"source{media_source.suffix.lower()}"
                shutil.copy2(media_source, media_target)
                files_manifest["media/source"] = {
                    "name": media_target.name,
                    "size_bytes": media_target.stat().st_size,
                    "sha256": sha256_file(media_target),
                }

            manifest: dict[str, object] = {
                "schema_version": EXPERIMENT_SCHEMA_VERSION,
                "run_id": chosen_run_id,
                "state": RecorderState.RECORDING.value,
                "complete": False,
                "incomplete": False,
                "started_utc": self._started_utc,
                "ended_utc": None,
                "stop_reason": None,
                "experiment": _serializable(spec),
                "session_config": _serializable(session_config),
                "source_id": source,
                "joint_names": list(names),
                "units": {
                    "time": "s",
                    "angles": "rad",
                    "angular_velocity": "rad/s",
                    "position": "m",
                    "linear_velocity": "m/s",
                    "force": "N",
                },
                "software": _serializable(
                    software_versions if software_versions is not None else _package_versions()
                ),
                "git": _serializable(
                    git_metadata
                    if git_metadata is not None
                    else _git_metadata(self._repo_root)
                ),
                "files": files_manifest,
                "chunks": [],
                "counts": {
                    "accepted_samples": 0,
                    "written_samples": 0,
                    "dropped_samples": 0,
                    "unwritten_samples": 0,
                    "events": 0,
                },
                "errors": [],
            }
            _atomic_yaml(partial / "manifest.yaml", manifest)
        except Exception as error:
            with self._lock:
                self._partial_path = partial if partial.exists() else None
                self._final_path = final
                self._errors.append(f"preparation_failed: {error}")
                self._incomplete = True
                self._ended_utc = _utc_text(self._now())
                self._stop_reason = "preparation_failed"
                self._state = RecorderState.ERROR
                if self._run_id is not None and self._started_utc is not None:
                    self._last_summary = self._make_summary_locked()
            raise

        with self._lock:
            self._partial_path = partial
            self._final_path = final
            self._manifest = manifest
            self._queue = Queue(maxsize=self._queue_capacity)
            self._state = RecorderState.RECORDING
            self._thread = Thread(
                target=self._writer_main,
                name=f"experiment-recorder-{chosen_run_id[-8:]}",
                daemon=True,
            )
            self._thread.start()
        return chosen_run_id

    def _coerce_sample(self, value: object) -> ExperimentSample:
        if isinstance(value, ExperimentSample):
            sample = value
        elif isinstance(value, Mapping):
            sample = ExperimentSample(**dict(value))
        else:
            sample = ExperimentSample.from_snapshot(value)
        joint_count = len(self._joint_names)
        for name in (
            "raw_angles_rad",
            "safe_angles_rad",
            "actual_angles_rad",
            "joint_velocities_rad_s",
            "actuator_forces",
        ):
            value_array = getattr(sample, name)
            if value_array is not None and value_array.shape != (joint_count,):
                raise ValueError(
                    f"{name} must have shape ({joint_count},), got {value_array.shape}"
                )
        return sample

    def append(self, sample: ExperimentSample | Mapping[str, object] | object) -> bool:
        """Queue one sample without blocking the realtime pipeline.

        ``False`` means the bounded queue overflowed.  The run is marked
        incomplete but recording and the caller's pipeline remain active.
        """

        normalized = self._coerce_sample(sample)
        with self._lock:
            if self._state is not RecorderState.RECORDING or self._queue is None:
                return False
            queue = self._queue
            try:
                queue.put_nowait(_SampleItem(normalized))
            except Full:
                self._mark_error("sample_queue_overflow", dropped=1, fatal=False)
                return False
            # Hold the same state lock used by the writer's persisted counter:
            # a concurrently finalizing writer can never observe written >
            # accepted for an item that is already visible in the queue.
            self._accepted_samples += 1
        return True

    def record_event(
        self,
        event: object | str,
        *,
        timestamp_s: float | None = None,
        sequence: int | None = None,
        level: str | None = None,
        subsystem: str | None = None,
        message: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> bool:
        """Queue a structured event, accepting mappings, dataclasses or codes."""

        if isinstance(event, str):
            payload: dict[str, object] = {"code": _required_text(event, "event code")}
        else:
            converted = _serializable(event)
            payload = dict(converted) if isinstance(converted, Mapping) else {
                "code": type(event).__name__,
                "value": converted,
            }
        if timestamp_s is not None:
            payload["timestamp_s"] = _optional_scalar(timestamp_s, "timestamp_s")
        if sequence is not None:
            if isinstance(sequence, bool) or int(sequence) != sequence or int(sequence) < 0:
                raise ValueError("event sequence must be a non-negative integer")
            payload["sequence"] = int(sequence)
        else:
            payload.setdefault("sequence", None)
        if level is not None:
            payload["level"] = str(level)
        payload.setdefault("level", "info")
        if subsystem is not None:
            payload["subsystem"] = str(subsystem)
        payload.setdefault("subsystem", "recorder")
        if message is not None:
            payload["message"] = str(message)
        if details is not None:
            payload["details"] = _serializable(details)
        with self._lock:
            if self._state is not RecorderState.RECORDING or self._queue is None:
                return False
            payload["run_id"] = self._run_id
            payload["source_id"] = self._source_id
            payload["recorded_utc"] = _utc_text(self._now())
            queue = self._queue
            try:
                # Keep publication atomic with respect to ``stop()`` changing
                # state and enqueueing its sentinel.  ``put_nowait`` preserves
                # the realtime caller's non-blocking/overflow semantics.
                queue.put_nowait(_EventItem(MappingProxyType(payload)))
            except Full:
                self._mark_error("event_queue_overflow", fatal=False)
                return False
        return True

    def reserve_media_path(self, filename: str) -> Path:
        """Return a safe path for a camera encoder owned by the caller."""

        basename = Path(_required_text(filename, "filename")).name
        if basename != filename or basename in {".", ".."}:
            raise ValueError("filename must be a plain basename")
        directory = self.media_directory
        if directory is None:
            raise RuntimeError("video recording is not enabled")
        return directory / basename

    def mark_incomplete(self, reason: str) -> None:
        """Record a non-fatal companion-I/O failure without stopping samples."""

        rendered = _required_text(reason, "incomplete reason")
        self._mark_error(f"companion_io:{rendered}", fatal=False)

    def stop(self, reason: str = "manual", *, timeout_s: float = 30.0) -> RecorderSummary:
        """Flush the queue and finalize the package synchronously."""

        stop_reason = _required_text(reason, "stop reason")
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        join_error_thread = False
        with self._lock:
            if self._state is RecorderState.COMPLETE:
                if self._last_summary is not None:
                    return self._last_summary
                return self._make_summary_locked()
            if self._state is RecorderState.ERROR:
                if self._last_summary is not None:
                    return self._last_summary
                # The I/O thread publishes ERROR before its finally block has
                # finished the incomplete manifest.  Join it below instead of
                # returning a transient summary or enqueueing a useless stop.
                join_error_thread = True
            if self._state is RecorderState.IDLE:
                if self._last_summary is not None:
                    return self._last_summary
                raise RuntimeError("no experiment has been started")
            if not join_error_thread:
                self._state = RecorderState.FINALIZING
                self._stop_reason = stop_reason
            queue = self._queue
            thread = self._thread
        if (
            not join_error_thread
            and queue is not None
            and thread is not None
            and thread.is_alive()
        ):
            try:
                queue.put(_StopItem(stop_reason), timeout=timeout)
            except Full:
                self._mark_error("finalization_queue_timeout", fatal=True)
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                self._mark_error("finalization_thread_timeout", fatal=True)
        with self._lock:
            if self._last_summary is None:
                self._ended_utc = self._ended_utc or _utc_text(self._now())
                self._state = RecorderState.ERROR
                self._incomplete = True
                self._last_summary = self._make_summary_locked()
            return self._last_summary

    def close(self) -> RecorderSummary | None:
        """Finalize an active run; safe to call repeatedly."""

        with self._lock:
            active = self._state in {
                RecorderState.PREPARING,
                RecorderState.RECORDING,
                RecorderState.FINALIZING,
            }
        if active:
            return self.stop("recorder_closed")
        return self.summary

    def __enter__(self) -> "ExperimentRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _mark_error(self, message: str, *, dropped: int = 0, fatal: bool) -> None:
        with self._lock:
            self._incomplete = True
            rendered = str(message)
            if rendered not in self._errors:
                self._errors.append(rendered)
            self._dropped_samples += int(dropped)
            if fatal and self._state is RecorderState.RECORDING:
                self._state = RecorderState.ERROR

    def _write_event(self, payload: Mapping[str, object]) -> None:
        assert self._partial_path is not None
        _append_json_line(self._partial_path / "events.jsonl", payload)
        with self._lock:
            self._event_count += 1

    def _writer_main(self) -> None:
        assert self._queue is not None
        queue = self._queue
        pending: list[ExperimentSample] = []
        requested_reason = "writer_stopped"
        try:
            self._write_event(
                {
                    "run_id": self._run_id,
                    "source_id": self._source_id,
                    "recorded_utc": self._started_utc,
                    "sequence": None,
                    "level": "info",
                    "subsystem": "recorder",
                    "code": "RECORDER_STARTED",
                    "message": "Запись опыта начата",
                }
            )
            while True:
                item = queue.get()
                if isinstance(item, _SampleItem):
                    pending.append(item.sample)
                    if len(pending) >= self._chunk_size:
                        self._write_chunk(pending)
                        pending.clear()
                elif isinstance(item, _EventItem):
                    self._write_event(item.payload)
                elif isinstance(item, _StopItem):
                    requested_reason = item.reason
                    if pending:
                        self._write_chunk(pending)
                        pending.clear()
                    self._write_event(
                        {
                            "run_id": self._run_id,
                            "source_id": self._source_id,
                            "recorded_utc": _utc_text(self._now()),
                            "sequence": None,
                            "level": "info",
                            "subsystem": "recorder",
                            "code": "RECORDER_STOPPED",
                            "message": "Запись опыта завершена",
                            "reason": requested_reason,
                        }
                    )
                    break
        except Exception as error:  # isolated I/O boundary
            self._mark_error(
                f"io_error: {type(error).__name__}: {error}",
                fatal=True,
            )
            requested_reason = "recorder_error"
        finally:
            self._finalize_writer(requested_reason)

    def _write_chunk(self, samples: Sequence[ExperimentSample]) -> None:
        assert self._partial_path is not None
        chunk_index = len(self._chunks)
        filename = f"{chunk_index:06d}.npz"
        target = self._partial_path / "chunks" / filename
        temporary = target.with_name(f"{target.name}.tmp")
        arrays = self._samples_to_arrays(samples)
        if any(array.dtype.hasobject for array in arrays.values()):
            raise TypeError("object arrays are forbidden in experiment chunks")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        metadata = {
            "file": f"chunks/{filename}",
            "sha256": sha256_file(target),
            "samples": len(samples),
            "first_sequence": samples[0].sequence,
            "last_sequence": samples[-1].sequence,
        }
        with self._lock:
            self._chunks.append(metadata)
            self._written_samples += len(samples)
        self._sync_manifest(RecorderState.RECORDING.value)

    def _optional_arrays(
        self,
        samples: Sequence[ExperimentSample],
        name: str,
        shape: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.full((len(samples), *shape), np.nan, dtype=np.float64)
        for index, sample in enumerate(samples):
            value = getattr(sample, name)
            if value is not None:
                values[index] = value
        return values, np.isfinite(values)

    def _optional_scalars(
        self,
        samples: Sequence[ExperimentSample],
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.full(len(samples), np.nan, dtype=np.float64)
        for index, sample in enumerate(samples):
            value = getattr(sample, name)
            if value is not None:
                values[index] = float(value)
        return values, np.isfinite(values)

    def _optional_booleans(
        self,
        samples: Sequence[ExperimentSample],
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros(len(samples), dtype=np.bool_)
        mask = np.zeros(len(samples), dtype=np.bool_)
        for index, sample in enumerate(samples):
            value = getattr(sample, name)
            if value is not None:
                values[index] = value
                mask[index] = True
        return values, mask

    def _optional_strings(
        self,
        samples: Sequence[ExperimentSample],
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        rendered = [str(getattr(sample, name) or "") for sample in samples]
        width = max(1, *(len(value) for value in rendered))
        values = np.asarray(rendered, dtype=f"<U{width}")
        mask = np.asarray([getattr(sample, name) is not None for sample in samples])
        return values, mask

    def _samples_to_arrays(
        self, samples: Sequence[ExperimentSample]
    ) -> dict[str, np.ndarray]:
        joint_shape = (len(self._joint_names),)
        result: dict[str, np.ndarray] = {
            "timestamp_s": np.asarray(
                [sample.timestamp_s for sample in samples], dtype=np.float64
            ),
            "sequence": np.asarray(
                [sample.sequence for sample in samples], dtype=np.int64
            ),
        }
        shapes = {
            "landmarks_2d": (LANDMARK_COUNT, 2),
            "landmarks_3d": (LANDMARK_COUNT, 3),
            "visibility": (LANDMARK_COUNT,),
            "presence": (LANDMARK_COUNT,),
            "raw_angles_rad": joint_shape,
            "safe_angles_rad": joint_shape,
            "actual_angles_rad": joint_shape,
            "joint_velocities_rad_s": joint_shape,
            "base_position_m": (3,),
            "base_orientation_wxyz": (4,),
            "base_linear_velocity_m_s": (3,),
            "base_angular_velocity_rad_s": (3,),
            "center_of_mass_position_m": (3,),
            "right_foot_position_m": (3,),
            "left_foot_position_m": (3,),
            "right_foot_linear_velocity_m_s": (3,),
            "left_foot_linear_velocity_m_s": (3,),
            "actuator_forces": joint_shape,
        }
        for name, shape in shapes.items():
            result[name], result[f"{name}_mask"] = self._optional_arrays(
                samples, name, shape
            )
        result["landmark_valid_mask"] = (
            result["landmarks_2d_mask"].all(axis=2)
            & result["landmarks_3d_mask"].all(axis=2)
            & result["visibility_mask"]
            & result["presence_mask"]
        )
        for name in (
            "right_foot_normal_force_n",
            "left_foot_normal_force_n",
            "contact_count",
            "non_foot_ground_contact_count",
            "tracking_quality",
            "calibration_progress",
        ):
            result[name], result[f"{name}_mask"] = self._optional_scalars(
                samples, name
            )
        for name in (
            "calibrating",
            "command_stale",
            "safe_valid",
            "free_base_active",
            "balance_active",
        ):
            result[name], result[f"{name}_mask"] = self._optional_booleans(
                samples, name
            )
        for name in ("support_intent", "support_phase"):
            result[name], result[f"{name}_mask"] = self._optional_strings(samples, name)
        diagnostics = [
            ""
            if sample.diagnostics is None
            else json.dumps(
                _serializable(sample.diagnostics),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for sample in samples
        ]
        width = max(1, *(len(value) for value in diagnostics))
        result["diagnostics_json"] = np.asarray(diagnostics, dtype=f"<U{width}")
        result["diagnostics_json_mask"] = np.asarray(
            [sample.diagnostics is not None for sample in samples], dtype=np.bool_
        )
        return result

    def _sync_manifest(self, state: str) -> None:
        assert self._partial_path is not None
        with self._lock:
            self._manifest.update(
                {
                    "state": state,
                    "complete": False,
                    "incomplete": self._incomplete,
                    "ended_utc": self._ended_utc,
                    "stop_reason": self._stop_reason,
                    "chunks": list(self._chunks),
                    "counts": {
                        "accepted_samples": self._accepted_samples,
                        "written_samples": self._written_samples,
                        "dropped_samples": self._dropped_samples,
                        "unwritten_samples": max(
                            0, self._accepted_samples - self._written_samples
                        ),
                        "events": self._event_count,
                    },
                    "errors": list(self._errors),
                }
            )
            manifest = dict(self._manifest)
        _atomic_yaml(self._partial_path / "manifest.yaml", manifest)

    def _finalize_writer(self, requested_reason: str) -> None:
        media_manifest: dict[str, object] = {}
        partial_for_media = self._partial_path
        if partial_for_media is not None:
            media_root = partial_for_media / "media"
            try:
                if media_root.is_dir():
                    for media_file in sorted(media_root.rglob("*")):
                        if media_file.is_file():
                            relative = media_file.relative_to(media_root).as_posix()
                            media_manifest[f"media/{relative}"] = {
                                "name": media_file.name,
                                "size_bytes": media_file.stat().st_size,
                                "sha256": sha256_file(media_file),
                            }
            except OSError as error:
                with self._lock:
                    self._incomplete = True
                    self._errors.append(
                        f"media_hash_error: {type(error).__name__}: {error}"
                    )
        with self._lock:
            self._ended_utc = _utc_text(self._now())
            self._stop_reason = requested_reason
            success = not self._incomplete and not self._errors
            target_state = RecorderState.COMPLETE if success else RecorderState.ERROR
            self._state = target_state
            files = dict(self._manifest.get("files", {}))
            files.update(media_manifest)
            self._manifest.update(
                {
                    "state": target_state.value,
                    "complete": success,
                    "incomplete": not success,
                    "ended_utc": self._ended_utc,
                    "stop_reason": requested_reason,
                    "chunks": list(self._chunks),
                    "counts": {
                        "accepted_samples": self._accepted_samples,
                        "written_samples": self._written_samples,
                        "dropped_samples": self._dropped_samples,
                        "unwritten_samples": max(
                            0, self._accepted_samples - self._written_samples
                        ),
                        "events": self._event_count,
                    },
                    "errors": list(self._errors),
                    "files": files,
                }
            )
            manifest = dict(self._manifest)
            partial = self._partial_path
            final = self._final_path
        assert partial is not None and final is not None
        try:
            _atomic_yaml(partial / "manifest.yaml", manifest)
            if success:
                os.replace(partial, final)
        except Exception as error:
            error_message = f"finalization_error: {type(error).__name__}: {error}"
            with self._lock:
                self._incomplete = True
                self._errors.append(error_message)
                self._state = RecorderState.ERROR
                self._manifest.update(
                    {
                        "state": RecorderState.ERROR.value,
                        "complete": False,
                        "incomplete": True,
                        "errors": list(self._errors),
                    }
                )
                error_manifest = dict(self._manifest)
                success = False
            # A directory rename may fail even though the preceding manifest
            # write succeeded (for example an antivirus race on Windows).
            # Best-effort correction prevents a surviving .partial package
            # from falsely claiming to be complete.
            if partial.exists():
                try:
                    _atomic_yaml(partial / "manifest.yaml", error_manifest)
                except OSError:
                    pass
        with self._lock:
            self._state = RecorderState.COMPLETE if success else RecorderState.ERROR
            self._last_summary = self._make_summary_locked()

    def _make_summary_locked(self) -> RecorderSummary:
        assert self._run_id is not None
        assert self._started_utc is not None
        use_final = (
            self._state is RecorderState.COMPLETE
            and self._final_path is not None
            and self._final_path.exists()
        )
        path = self._final_path if use_final else self._partial_path
        if path is None:
            path = self._root
        return RecorderSummary(
            run_id=self._run_id,
            state=self._state,
            path=path,
            sample_count=self._written_samples,
            accepted_samples=self._accepted_samples,
            dropped_samples=self._dropped_samples,
            chunk_count=len(self._chunks),
            event_count=self._event_count,
            started_utc=self._started_utc,
            ended_utc=self._ended_utc,
            stop_reason=self._stop_reason,
            incomplete=self._incomplete or self._state is RecorderState.ERROR,
            error="; ".join(self._errors) if self._errors else None,
        )


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_SAMPLE_QUEUE_CAPACITY",
    "EXPERIMENT_SCHEMA_VERSION",
    "ExperimentRecorder",
    "ExperimentSample",
    "ExperimentSpec",
    "RecorderState",
    "RecorderSummary",
    "recover_interrupted_experiments",
    "sha256_file",
]
