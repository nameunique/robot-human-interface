"""Headless-first MuJoCo API for the 20-DOF humanoid proxy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypeAlias

import mujoco
import numpy as np
import yaml
from numpy.typing import NDArray

from .types import HumanoidState, LatestJointCommandBuffer


FloatArray = NDArray[np.float64]
PathLike: TypeAlias = str | Path
SceneName: TypeAlias = Literal["fixed", "free"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_joint_config() -> Path:
    return _project_root() / "config" / "joints.yaml"


def _default_scene(scene: SceneName) -> Path:
    return _project_root() / "models" / "humanoid" / f"scene_{scene}.xml"


class HumanoidSimulation:
    """Own a MuJoCo model/data pair and expose names in legacy motor order.

    Parameters
    ----------
    scene:
        ``"fixed"`` welds the free torso to world and is the camera-retargeting
        default. ``"free"`` allows the robot to fall. A path may also point at
        a custom compatible MJCF scene.
    joint_config_path:
        Canonical YAML schema. It supplies order, limits and neutral targets.

    Notes
    -----
    The class is deliberately synchronous. A camera thread may safely call
    :meth:`set_joint_targets`; the physics owner calls :meth:`step`.
    """

    def __init__(
        self,
        scene: SceneName | PathLike = "fixed",
        *,
        joint_config_path: PathLike | None = None,
    ) -> None:
        self.model_path = self._resolve_scene_path(scene)
        self.joint_config_path = Path(joint_config_path or _default_joint_config()).resolve()
        self._load_schema(self.joint_config_path)

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._viewer: Any | None = None
        self._closed = False

        self._joint_ids = np.asarray(
            [self._required_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names],
            dtype=np.int32,
        )
        self._actuator_ids = np.asarray(
            [self._required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.joint_names],
            dtype=np.int32,
        )
        self._joint_qpos_adr = self.model.jnt_qposadr[self._joint_ids].copy()
        self._joint_dof_adr = self.model.jnt_dofadr[self._joint_ids].copy()
        self._base_body_id = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._base_joint_id = self._required_id(mujoco.mjtObj.mjOBJ_JOINT, "base_free")
        self._base_qpos_adr = int(self.model.jnt_qposadr[self._base_joint_id])

        transmitted_joint_ids = self.model.actuator_trnid[self._actuator_ids, 0]
        if not np.array_equal(transmitted_joint_ids, self._joint_ids):
            raise ValueError("Actuator-to-joint mapping does not match config/joints.yaml")

        self.command_buffer = LatestJointCommandBuffer(self.home_positions_rad)
        self.reset()

    @staticmethod
    def _resolve_scene_path(scene: SceneName | PathLike) -> Path:
        if isinstance(scene, str) and scene in {"fixed", "free"}:
            path = _default_scene(scene)
        else:
            path = Path(scene)
            if not path.is_absolute():
                path = (_project_root() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo scene not found: {path}")
        return path.resolve()

    def _load_schema(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Joint configuration not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        records = sorted(document["joints"], key=lambda item: int(item["index"]))
        names = tuple(str(item["name"]) for item in records)
        declared_order = tuple(str(name) for name in document["joint_order"])
        if names != declared_order or len(names) != 20 or len(set(names)) != 20:
            raise ValueError("Joint schema must contain one ordered record for all 20 joints")
        indices = tuple(int(item["index"]) for item in records)
        if indices != tuple(range(20)):
            raise ValueError("Joint schema indices must be contiguous 0..19")

        self.joint_names = names
        self.lower_limits_rad = np.asarray(
            [float(item["limit_rad"][0]) for item in records], dtype=np.float64
        )
        self.upper_limits_rad = np.asarray(
            [float(item["limit_rad"][1]) for item in records], dtype=np.float64
        )
        self.home_positions_rad = np.asarray(
            [float(item["home_rad"]) for item in records], dtype=np.float64
        )
        if not (
            np.isfinite(self.lower_limits_rad).all()
            and np.isfinite(self.upper_limits_rad).all()
            and np.isfinite(self.home_positions_rad).all()
        ):
            raise ValueError("Joint schema contains non-finite values")
        if np.any(self.lower_limits_rad >= self.upper_limits_rad):
            raise ValueError("Every lower joint limit must be below its upper limit")
        if np.any(self.home_positions_rad < self.lower_limits_rad) or np.any(
            self.home_positions_rad > self.upper_limits_rad
        ):
            raise ValueError("Home pose must be inside all joint limits")

    def _required_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        identifier = int(mujoco.mj_name2id(self.model, object_type, name))
        if identifier < 0:
            raise ValueError(f"Required MuJoCo object is missing: {object_type.name} {name!r}")
        return identifier

    def _coerce_targets(self, targets: object) -> FloatArray:
        if hasattr(targets, "positions_rad"):
            positions = np.asarray(getattr(targets, "positions_rad"), dtype=np.float64)
            names = tuple(str(name) for name in getattr(targets, "joint_names", self.joint_names))
            return self._reorder_named_targets(names, positions)
        if isinstance(targets, Mapping):
            names = tuple(str(name) for name in targets.keys())
            positions = np.asarray([targets[name] for name in targets.keys()], dtype=np.float64)
            return self._reorder_named_targets(names, positions)
        positions = np.asarray(targets, dtype=np.float64)
        if positions.shape != (len(self.joint_names),):
            raise ValueError(
                f"joint targets must have shape ({len(self.joint_names)},), got {positions.shape}"
            )
        return positions

    def _reorder_named_targets(
        self,
        names: tuple[str, ...],
        positions: FloatArray,
    ) -> FloatArray:
        if positions.shape != (len(names),):
            raise ValueError("positions_rad must have one value per joint name")
        if len(set(names)) != len(names):
            raise ValueError("joint names must be unique")
        if set(names) != set(self.joint_names):
            missing = sorted(set(self.joint_names) - set(names))
            extra = sorted(set(names) - set(self.joint_names))
            raise ValueError(f"joint names do not match schema; missing={missing}, extra={extra}")
        by_name = dict(zip(names, positions, strict=True))
        return np.asarray([by_name[name] for name in self.joint_names], dtype=np.float64)

    def set_joint_targets(self, targets: object, *, clamp: bool = True) -> FloatArray:
        """Publish the latest 20 position targets and return the accepted copy.

        ``targets`` may be a sequence in canonical order, a complete name/value
        mapping, or a ``RobotJointCommand``-like object exposing ``joint_names``
        and ``positions_rad``.
        """

        self._ensure_open()
        positions = self._coerce_targets(targets)
        if not np.isfinite(positions).all():
            raise ValueError("joint targets must contain only finite values")
        if clamp:
            positions = np.clip(positions, self.lower_limits_rad, self.upper_limits_rad)
        elif np.any(positions < self.lower_limits_rad) or np.any(
            positions > self.upper_limits_rad
        ):
            raise ValueError("joint targets exceed configured limits")
        self.command_buffer.update(positions)
        return positions.copy()

    def apply_joint_command(self, command: object, *, clamp: bool = True) -> FloatArray:
        """Alias suited to the perception pipeline's ``RobotJointCommand``."""

        return self.set_joint_targets(command, clamp=clamp)

    def reset(
        self,
        joint_positions_rad: Sequence[float] | None = None,
        *,
        base_position_m: Sequence[float] | None = None,
        base_orientation_wxyz: Sequence[float] | None = None,
    ) -> HumanoidState:
        """Reset dynamics and place joints at the configured neutral pose."""

        self._ensure_open()
        mujoco.mj_resetData(self.model, self.data)

        positions = (
            self.home_positions_rad.copy()
            if joint_positions_rad is None
            else self._coerce_targets(joint_positions_rad)
        )
        if not np.isfinite(positions).all():
            raise ValueError("reset joint positions must be finite")
        positions = np.clip(positions, self.lower_limits_rad, self.upper_limits_rad)
        self.data.qpos[self._joint_qpos_adr] = positions

        if base_position_m is not None:
            base_position = np.asarray(base_position_m, dtype=np.float64)
            if base_position.shape != (3,) or not np.isfinite(base_position).all():
                raise ValueError("base_position_m must be a finite 3-vector")
            self.data.qpos[self._base_qpos_adr : self._base_qpos_adr + 3] = base_position
        if base_orientation_wxyz is not None:
            base_quaternion = np.asarray(base_orientation_wxyz, dtype=np.float64)
            if base_quaternion.shape != (4,) or not np.isfinite(base_quaternion).all():
                raise ValueError("base_orientation_wxyz must be a finite 4-vector")
            norm = float(np.linalg.norm(base_quaternion))
            if norm < 1e-12:
                raise ValueError("base_orientation_wxyz must have non-zero norm")
            self.data.qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7] = (
                base_quaternion / norm
            )

        self.data.qvel[:] = 0.0
        self.data.ctrl[self._actuator_ids] = positions
        self.command_buffer.update(positions)
        mujoco.mj_forward(self.model, self.data)
        self._assert_finite()
        self.sync_viewer()
        return self.get_state()

    def step(self, steps: int = 1) -> HumanoidState:
        """Advance a positive integer number of fixed MuJoCo timesteps."""

        self._ensure_open()
        if isinstance(steps, bool) or int(steps) != steps or int(steps) <= 0:
            raise ValueError("steps must be a positive integer")
        for _ in range(int(steps)):
            positions, _sequence = self.command_buffer.snapshot()
            self.data.ctrl[self._actuator_ids] = positions
            mujoco.mj_step(self.model, self.data)
        self._assert_finite()
        self.sync_viewer()
        return self.get_state()

    def get_state(self) -> HumanoidState:
        """Return a copy of the current state in canonical joint order."""

        self._ensure_open()
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._base_body_id,
            velocity,
            0,
        )
        return HumanoidState(
            simulation_time_s=float(self.data.time),
            joint_names=self.joint_names,
            joint_positions_rad=self.data.qpos[self._joint_qpos_adr],
            joint_velocities_rad_s=self.data.qvel[self._joint_dof_adr],
            base_position_m=self.data.xpos[self._base_body_id],
            base_orientation_wxyz=self.data.xquat[self._base_body_id],
            base_linear_velocity_m_s=velocity[3:6],
            base_angular_velocity_rad_s=velocity[0:3],
            actuator_forces=self.data.actuator_force[self._actuator_ids],
            contact_count=int(self.data.ncon),
        )

    def launch_viewer(self) -> Any:
        """Launch MuJoCo's passive viewer; physics remains caller-owned."""

        self._ensure_open()
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    @property
    def viewer_is_running(self) -> bool:
        return self._viewer is not None and bool(self._viewer.is_running())

    def sync_viewer(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def _assert_finite(self) -> None:
        arrays = (self.data.qpos, self.data.qvel, self.data.ctrl, self.data.actuator_force)
        if not all(np.isfinite(values).all() for values in arrays):
            raise FloatingPointError("MuJoCo state contains NaN or infinity")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("HumanoidSimulation is closed")

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._closed = True

    def __enter__(self) -> "HumanoidSimulation":
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
