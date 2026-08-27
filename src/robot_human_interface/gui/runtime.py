"""Immutable runtime and physical-output status contracts for the GUI.

These types deliberately keep safety decisions out of widgets.  A
``RobotReadiness`` instance is produced by the worker that owns the pipeline
and physical-output controller; the GUI may display it, but must not recreate
the decision from individual labels or cached frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class RuntimeMode(str, Enum):
    """Whether the active session is allowed to control physical hardware."""

    PRODUCTION = "production"
    DEMO = "demo"


class RobotUiState(str, Enum):
    """Stable and transitional states exposed to the operator UI."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED_DISARMED = "connected_disarmed"
    ARMING = "arming"
    ARMED = "armed"
    DISARMING = "disarming"
    DISCONNECTING = "disconnecting"
    DEGRADED = "degraded"


class ReadinessReason(str, Enum):
    """Stable machine-readable result of one authoritative readiness check."""

    READY = "ready"
    RUNTIME_DEMO = "runtime_demo"
    PIPELINE_NOT_RUNNING = "pipeline_not_running"
    ROBOT_NOT_CONNECTED = "robot_not_connected"
    NO_SNAPSHOT = "no_snapshot"
    SNAPSHOT_STALE = "snapshot_stale"
    SNAPSHOT_STATUS_INVALID = "snapshot_status_invalid"
    SAFE_VALID_FALSE = "safe_valid_false"
    SAFE_COMMAND_MISSING = "safe_command_missing"
    SAFE_COMMAND_INVALID = "safe_command_invalid"
    SAFETY_PROVENANCE_MISSING = "safety_provenance_missing"
    FREE_BASE_INACTIVE = "free_base_inactive"
    BALANCE_INACTIVE = "balance_inactive"
    CONTROLLER_NOT_READY = "controller_not_ready"


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """Authoritative runtime mode and an explicit fallback explanation."""

    mode: RuntimeMode
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", RuntimeMode(self.mode))
        reason = None if self.fallback_reason is None else str(self.fallback_reason).strip()
        if self.mode is RuntimeMode.PRODUCTION and reason:
            raise ValueError("production runtime cannot have a fallback reason")
        if self.mode is RuntimeMode.DEMO and not reason:
            raise ValueError("demo runtime requires a fallback reason")
        object.__setattr__(self, "fallback_reason", reason or None)

    @property
    def physical_output_allowed(self) -> bool:
        return self.mode is RuntimeMode.PRODUCTION

    @classmethod
    def production(cls) -> "RuntimeStatus":
        return cls(RuntimeMode.PRODUCTION)

    @classmethod
    def demo(cls, reason: str) -> "RuntimeStatus":
        return cls(RuntimeMode.DEMO, reason)


@dataclass(frozen=True, slots=True)
class RobotReadiness:
    """One worker-owned, fail-closed decision about whether arm may proceed."""

    ready: bool
    reason_code: ReadinessReason
    reason: str
    evaluated_at_s: float
    runtime: RuntimeStatus
    pipeline_state: str
    robot_state: RobotUiState
    source_id: str | None
    snapshot_sequence: int | None
    snapshot_age_s: float | None
    command_generation: int
    safe_command_valid: bool
    free_base_active: bool | None
    balance_active: bool | None

    def __post_init__(self) -> None:
        if type(self.ready) is not bool:
            raise ValueError("ready must be a boolean")
        object.__setattr__(self, "reason_code", ReadinessReason(self.reason_code))
        object.__setattr__(self, "robot_state", RobotUiState(self.robot_state))
        reason = str(self.reason).strip()
        if self.ready:
            if self.reason_code is not ReadinessReason.READY:
                raise ValueError("ready status must use ReadinessReason.READY")
            if reason:
                raise ValueError("ready status cannot have a blocking reason")
        elif self.reason_code is ReadinessReason.READY or not reason:
            raise ValueError("blocked status requires a non-ready code and reason")
        object.__setattr__(self, "reason", reason)

        evaluated = float(self.evaluated_at_s)
        if not math.isfinite(evaluated) or evaluated < 0.0:
            raise ValueError("evaluated_at_s must be finite and non-negative")
        object.__setattr__(self, "evaluated_at_s", evaluated)

        if self.snapshot_age_s is not None:
            age = float(self.snapshot_age_s)
            if not math.isfinite(age) or age < 0.0:
                raise ValueError("snapshot_age_s must be finite and non-negative")
            object.__setattr__(self, "snapshot_age_s", age)
        if self.snapshot_sequence is not None:
            sequence = int(self.snapshot_sequence)
            if sequence < 0:
                raise ValueError("snapshot_sequence must be non-negative")
            object.__setattr__(self, "snapshot_sequence", sequence)
        generation = int(self.command_generation)
        if generation < 0:
            raise ValueError("command_generation must be non-negative")
        object.__setattr__(self, "command_generation", generation)
        if type(self.safe_command_valid) is not bool:
            raise ValueError("safe_command_valid must be a boolean")

    @property
    def authoritative(self) -> bool:
        """Typed readiness objects are the only accepted GUI arm authority."""

        return True

    def semantic_key(self) -> tuple[object, ...]:
        """Key for change notifications; excludes continuously changing age."""

        return (
            self.ready,
            self.reason_code,
            self.reason,
            self.runtime,
            self.pipeline_state,
            self.robot_state,
            self.source_id,
            self.snapshot_sequence,
            self.command_generation,
            self.safe_command_valid,
            self.free_base_active,
            self.balance_active,
        )
