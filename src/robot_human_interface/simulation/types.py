"""Small, dependency-light types for the MuJoCo simulation boundary."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _readonly_vector(value: object, size: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class HumanoidState:
    """One immutable simulator state snapshot.

    Quaternion order is MuJoCo's native ``(w, x, y, z)``.  Joint arrays always
    follow ``config/joints.yaml``, not MuJoCo's body-traversal qpos order.
    """

    simulation_time_s: float
    joint_names: tuple[str, ...]
    joint_positions_rad: FloatArray
    joint_velocities_rad_s: FloatArray
    base_position_m: FloatArray
    base_orientation_wxyz: FloatArray
    base_linear_velocity_m_s: FloatArray
    base_angular_velocity_rad_s: FloatArray
    actuator_forces: FloatArray
    contact_count: int

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.joint_names)
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        time_s = float(self.simulation_time_s)
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("simulation_time_s must be finite and non-negative")
        object.__setattr__(self, "simulation_time_s", time_s)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(
            self,
            "joint_positions_rad",
            _readonly_vector(self.joint_positions_rad, len(names), "joint_positions_rad"),
        )
        object.__setattr__(
            self,
            "joint_velocities_rad_s",
            _readonly_vector(
                self.joint_velocities_rad_s,
                len(names),
                "joint_velocities_rad_s",
            ),
        )
        object.__setattr__(
            self,
            "base_position_m",
            _readonly_vector(self.base_position_m, 3, "base_position_m"),
        )
        object.__setattr__(
            self,
            "base_orientation_wxyz",
            _readonly_vector(self.base_orientation_wxyz, 4, "base_orientation_wxyz"),
        )
        object.__setattr__(
            self,
            "base_linear_velocity_m_s",
            _readonly_vector(
                self.base_linear_velocity_m_s,
                3,
                "base_linear_velocity_m_s",
            ),
        )
        object.__setattr__(
            self,
            "base_angular_velocity_rad_s",
            _readonly_vector(
                self.base_angular_velocity_rad_s,
                3,
                "base_angular_velocity_rad_s",
            ),
        )
        object.__setattr__(
            self,
            "actuator_forces",
            _readonly_vector(self.actuator_forces, len(names), "actuator_forces"),
        )
        if int(self.contact_count) < 0:
            raise ValueError("contact_count must be non-negative")
        object.__setattr__(self, "contact_count", int(self.contact_count))

    @property
    def is_finite(self) -> bool:
        """Whether every numeric value in the snapshot is finite."""

        return all(
            np.isfinite(values).all()
            for values in (
                self.joint_positions_rad,
                self.joint_velocities_rad_s,
                self.base_position_m,
                self.base_orientation_wxyz,
                self.base_linear_velocity_m_s,
                self.base_angular_velocity_rad_s,
                self.actuator_forces,
            )
        )


class LatestJointCommandBuffer:
    """Thread-safe latest-value buffer between perception and physics loops."""

    def __init__(self, initial_positions_rad: Sequence[float]) -> None:
        positions = np.asarray(initial_positions_rad, dtype=np.float64)
        if positions.ndim != 1 or not np.isfinite(positions).all():
            raise ValueError("initial_positions_rad must be a finite vector")
        self._lock = Lock()
        self._positions = positions.copy()
        self._sequence = 0

    def update(self, positions_rad: Sequence[float]) -> int:
        positions = np.asarray(positions_rad, dtype=np.float64)
        if positions.shape != self._positions.shape:
            raise ValueError(
                f"positions_rad must have shape {self._positions.shape}, got {positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("positions_rad must contain only finite values")
        with self._lock:
            self._positions[:] = positions
            self._sequence += 1
            return self._sequence

    def snapshot(self) -> tuple[FloatArray, int]:
        with self._lock:
            return self._positions.copy(), self._sequence
