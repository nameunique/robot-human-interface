"""Task-space pose-fidelity metrics shared by retargeters and evaluations.

Joint-angle error is not a meaningful cross-morphology metric: the human and
the robot have different link geometry, zero poses, and joint axes.  This
module instead expresses human and robot limb directions in the same
anatomical ``(forward, right, up)`` basis and measures their angular error.

The functions are deliberately independent of the teleoperation and balance
controllers.  They can therefore compare an unconstrained retargeting target,
the safety-projected motor command, and measured robot encoders with the same
metric without changing simulation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray

from robot_human_interface.skeleton import JOINT_NAMES, PoseLandmark as L, SkeletonFrame

from .geometry import body_basis


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DIRECTION_NAMES: tuple[str, ...] = (
    "right_upper_arm",
    "left_upper_arm",
    "right_forearm",
    "left_forearm",
    "right_hand",
    "left_hand",
    "right_arm",
    "left_arm",
    "right_thigh",
    "left_thigh",
    "right_shin",
    "left_shin",
    "right_foot",
    "left_foot",
    "right_leg",
    "left_leg",
    "head",
)

ARM_DIRECTION_NAMES: tuple[str, ...] = DIRECTION_NAMES[:8]
LEG_DIRECTION_NAMES: tuple[str, ...] = DIRECTION_NAMES[8:16]
HEAD_DIRECTION_NAMES: tuple[str, ...] = ("head",)
END_EFFECTOR_DIRECTION_NAMES: tuple[str, ...] = (
    "right_arm",
    "left_arm",
    "right_leg",
    "left_leg",
    "head",
)

_DIRECTION_INDEX = {name: index for index, name in enumerate(DIRECTION_NAMES)}


def _readonly_array(values: object, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    result = result.copy()
    result.setflags(write=False)
    return result


def _readonly_bool_array(values: object, shape: tuple[int, ...], name: str) -> BoolArray:
    result = np.asarray(values, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    result = result.copy()
    result.setflags(write=False)
    return result


def _unit(vector: object) -> FloatArray:
    result = np.asarray(vector, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"direction vector must have shape (3,), got {result.shape}")
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm < 1e-9:
        return np.full(3, np.nan, dtype=np.float64)
    return result / norm


@dataclass(frozen=True, slots=True)
class AnatomicalDirections:
    """Named unit vectors in an anatomical ``(forward, right, up)`` basis.

    Invalid or low-confidence human tasks are represented by a false entry in
    ``valid`` and a NaN vector.  Robot FK normally makes every entry valid.
    """

    vectors: FloatArray
    valid: BoolArray

    def __post_init__(self) -> None:
        shape = (len(DIRECTION_NAMES), 3)
        vectors = _readonly_array(self.vectors, shape, "vectors")
        valid = _readonly_bool_array(self.valid, (len(DIRECTION_NAMES),), "valid")
        finite_rows = np.isfinite(vectors).all(axis=1)
        if np.any(valid & ~finite_rows):
            raise ValueError("every valid anatomical direction must be finite")
        if np.any(valid):
            norms = np.linalg.norm(vectors[valid], axis=1)
            if not np.allclose(norms, 1.0, atol=1e-7):
                raise ValueError("every valid anatomical direction must be unit length")
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "valid", valid)

    def vector(self, name: str) -> FloatArray:
        """Return a read-only vector by its stable task name."""

        try:
            index = _DIRECTION_INDEX[name]
        except KeyError as error:
            raise KeyError(f"unknown anatomical direction: {name!r}") from error
        return self.vectors[index]

    def is_valid(self, name: str) -> bool:
        try:
            return bool(self.valid[_DIRECTION_INDEX[name]])
        except KeyError as error:
            raise KeyError(f"unknown anatomical direction: {name!r}") from error


@dataclass(frozen=True, slots=True)
class PoseFidelity:
    """Angular disagreement between two :class:`AnatomicalDirections` sets."""

    errors_deg: FloatArray
    valid: BoolArray

    def __post_init__(self) -> None:
        errors = _readonly_array(self.errors_deg, (len(DIRECTION_NAMES),), "errors_deg")
        valid = _readonly_bool_array(self.valid, (len(DIRECTION_NAMES),), "valid")
        if np.any(valid & (~np.isfinite(errors) | (errors < 0.0) | (errors > 180.0))):
            raise ValueError("valid direction errors must be finite and within [0, 180]")
        object.__setattr__(self, "errors_deg", errors)
        object.__setattr__(self, "valid", valid)

    def error_deg(self, name: str) -> float:
        """Return one named error, or NaN when that task was unavailable."""

        try:
            return float(self.errors_deg[_DIRECTION_INDEX[name]])
        except KeyError as error:
            raise KeyError(f"unknown anatomical direction: {name!r}") from error

    def mean_error_deg(self, names: Iterable[str] = DIRECTION_NAMES) -> float:
        """Mean of the available named errors; NaN if none are available."""

        indices: list[int] = []
        for name in names:
            try:
                indices.append(_DIRECTION_INDEX[name])
            except KeyError as error:
                raise KeyError(f"unknown anatomical direction: {name!r}") from error
        selected = np.asarray(indices, dtype=np.int64)
        accepted = selected[self.valid[selected]]
        if accepted.size == 0:
            return float("nan")
        return float(np.mean(self.errors_deg[accepted]))

    @property
    def arm_mean_error_deg(self) -> float:
        return self.mean_error_deg(ARM_DIRECTION_NAMES)

    @property
    def leg_mean_error_deg(self) -> float:
        return self.mean_error_deg(LEG_DIRECTION_NAMES)

    @property
    def end_effector_mean_error_deg(self) -> float:
        return self.mean_error_deg(END_EFFECTOR_DIRECTION_NAMES)


_CORE_LANDMARKS = (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP)
def human_anatomical_directions(
    frame: SkeletonFrame,
    *,
    confidence_threshold: float = 0.55,
) -> AnatomicalDirections:
    """Extract confidence-gated human limb and head directions.

    MediaPipe camera coordinates are converted with the per-frame torso basis,
    so camera placement and a person's yaw do not change the task convention.
    """

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be within [0, 1]")
    points = frame.landmarks_3d
    landmark_valid = frame.valid_mask(confidence_threshold)
    core_indices = np.asarray([int(index) for index in _CORE_LANDMARKS], dtype=np.int64)
    core_valid = bool(np.all(landmark_valid[core_indices]))
    vectors = np.full((len(DIRECTION_NAMES), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(DIRECTION_NAMES), dtype=bool)
    if not core_valid:
        return AnatomicalDirections(vectors, valid)

    basis = body_basis(points)
    if not all(
        np.isfinite(axis).all()
        for axis in (basis.forward, basis.lateral_right, basis.vertical_up)
    ):
        return AnatomicalDirections(vectors, valid)

    def anatomical(vector: object) -> FloatArray:
        value = np.asarray(vector, dtype=np.float64)
        return _unit(
            np.asarray(
                (
                    np.dot(value, basis.forward),
                    np.dot(value, basis.lateral_right),
                    np.dot(value, basis.vertical_up),
                )
            )
        )

    def install(name: str, vector: object, dependencies: Sequence[L]) -> None:
        indices = core_indices.tolist() + [int(item) for item in dependencies]
        if not bool(np.all(landmark_valid[np.asarray(indices, dtype=np.int64)])):
            return
        direction = anatomical(vector)
        if np.isfinite(direction).all():
            output_index = _DIRECTION_INDEX[name]
            vectors[output_index] = direction
            valid[output_index] = True

    for side in ("right", "left"):
        left = side == "left"
        shoulder = L.LEFT_SHOULDER if left else L.RIGHT_SHOULDER
        elbow = L.LEFT_ELBOW if left else L.RIGHT_ELBOW
        wrist = L.LEFT_WRIST if left else L.RIGHT_WRIST
        index = L.LEFT_INDEX if left else L.RIGHT_INDEX
        pinky = L.LEFT_PINKY if left else L.RIGHT_PINKY
        hip = L.LEFT_HIP if left else L.RIGHT_HIP
        knee = L.LEFT_KNEE if left else L.RIGHT_KNEE
        ankle = L.LEFT_ANKLE if left else L.RIGHT_ANKLE
        heel = L.LEFT_HEEL if left else L.RIGHT_HEEL
        toe = L.LEFT_FOOT_INDEX if left else L.RIGHT_FOOT_INDEX
        hand_end = 0.5 * (points[int(index)] + points[int(pinky)])
        install(
            f"{side}_upper_arm",
            points[int(elbow)] - points[int(shoulder)],
            (shoulder, elbow),
        )
        install(
            f"{side}_forearm",
            points[int(wrist)] - points[int(elbow)],
            (elbow, wrist),
        )
        install(
            f"{side}_hand",
            hand_end - points[int(wrist)],
            (wrist, index, pinky),
        )
        install(
            f"{side}_arm",
            points[int(wrist)] - points[int(shoulder)],
            (shoulder, wrist),
        )
        install(
            f"{side}_thigh",
            points[int(knee)] - points[int(hip)],
            (hip, knee),
        )
        install(
            f"{side}_shin",
            points[int(ankle)] - points[int(knee)],
            (knee, ankle),
        )
        install(
            f"{side}_foot",
            points[int(toe)] - points[int(heel)],
            (ankle, heel, toe),
        )
        install(
            f"{side}_leg",
            points[int(ankle)] - points[int(hip)],
            (hip, ankle),
        )

    face_dependencies = np.asarray(
        [*(int(index) for index in _CORE_LANDMARKS), int(L.NOSE), int(L.LEFT_EAR), int(L.RIGHT_EAR)],
        dtype=np.int64,
    )
    if bool(np.all(landmark_valid[face_dependencies])):
        ear_center = 0.5 * (points[int(L.LEFT_EAR)] + points[int(L.RIGHT_EAR)])
        direction = anatomical(points[int(L.NOSE)] - ear_center)
        if np.isfinite(direction).all():
            vectors[_DIRECTION_INDEX["head"]] = direction
            valid[_DIRECTION_INDEX["head"]] = True
    return AnatomicalDirections(vectors, valid)


def _required_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    identifier = int(mujoco.mj_name2id(model, object_type, name))
    if identifier < 0:
        raise ValueError(f"MuJoCo model is missing required {object_type.name} {name!r}")
    return identifier


def robot_anatomical_directions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> AnatomicalDirections:
    """Read robot FK directions without mutating ``model`` or ``data``.

    World vectors are transformed into the torso frame.  The inherited model's
    physical front/left/up axes are ``-X/+Y/+Z``, hence anatomical right is
    ``-Y``.  The result is invariant to free-base translation and rotation.
    """

    torso_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    torso_rotation = np.asarray(data.xmat[torso_id], dtype=np.float64).reshape(3, 3)
    world_to_base = torso_rotation.T
    vectors = np.full((len(DIRECTION_NAMES), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(DIRECTION_NAMES), dtype=bool)

    def anatomical(world_vector: object) -> FloatArray:
        local = world_to_base @ np.asarray(world_vector, dtype=np.float64)
        return _unit(np.asarray((-local[0], -local[1], local[2])))

    def install(name: str, world_vector: object) -> None:
        direction = anatomical(world_vector)
        if np.isfinite(direction).all():
            output_index = _DIRECTION_INDEX[name]
            vectors[output_index] = direction
            valid[output_index] = True

    body_names = {
        "shoulder_rh", "elbow_rh", "wrist_rh",
        "shoulder_lh", "elbow_lh", "wrist_lh",
        "knee_rl", "shin_rl", "motors_feet_rl",
        "knee_ll", "shin_ll", "motors_feet_ll",
    }
    body_ids = {
        name: _required_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in body_names
    }
    site_names = {
        "right_wrist_ik", "right_hand_ik", "left_wrist_ik", "left_hand_ik",
        "right_foot_contact", "right_toe_ik", "left_foot_contact", "left_toe_ik",
    }
    site_ids = {
        name: _required_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in site_names
    }

    def body_segment(name: str, proximal: str, distal: str) -> None:
        install(name, data.xpos[body_ids[distal]] - data.xpos[body_ids[proximal]])

    def site_segment(name: str, proximal: str, distal: str) -> None:
        install(name, data.site_xpos[site_ids[distal]] - data.site_xpos[site_ids[proximal]])

    for side, suffix in (("right", "rh"), ("left", "lh")):
        body_segment(f"{side}_upper_arm", f"shoulder_{suffix}", f"elbow_{suffix}")
        body_segment(f"{side}_forearm", f"elbow_{suffix}", f"wrist_{suffix}")
        site_segment(f"{side}_hand", f"{side}_wrist_ik", f"{side}_hand_ik")
        body_segment(f"{side}_arm", f"shoulder_{suffix}", f"wrist_{suffix}")

    # knee_* is the hip-pitch body; shin_* is the anatomical knee; and
    # motors_feet_* is the ankle-pitch body in this inherited Unity naming.
    for side, suffix in (("right", "rl"), ("left", "ll")):
        body_segment(f"{side}_thigh", f"knee_{suffix}", f"shin_{suffix}")
        body_segment(f"{side}_shin", f"shin_{suffix}", f"motors_feet_{suffix}")
        # Foot direction is longitudinal heel/sole-to-toe orientation, not the
        # ankle-to-toe displacement (which mixes shank length and ankle offset).
        site_segment(f"{side}_foot", f"{side}_foot_contact", f"{side}_toe_ik")
        body_segment(f"{side}_leg", f"knee_{suffix}", f"motors_feet_{suffix}")

    head_id = _required_id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    head_rotation = np.asarray(data.xmat[head_id], dtype=np.float64).reshape(3, 3)
    direction = anatomical(head_rotation @ np.asarray((-1.0, 0.0, 0.0)))
    if np.isfinite(direction).all():
        vectors[_DIRECTION_INDEX["head"]] = direction
        valid[_DIRECTION_INDEX["head"]] = True
    return AnatomicalDirections(vectors, valid)


def angular_pose_fidelity(
    reference: AnatomicalDirections,
    candidate: AnatomicalDirections,
) -> PoseFidelity:
    """Measure shortest 3-D angular error for all mutually valid tasks."""

    valid = reference.valid & candidate.valid
    errors = np.full(len(DIRECTION_NAMES), np.nan, dtype=np.float64)
    if np.any(valid):
        dots = np.einsum("ij,ij->i", reference.vectors[valid], candidate.vectors[valid])
        errors[valid] = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return PoseFidelity(errors, valid)


class MujocoPoseFidelityEvaluator:
    """Deterministically evaluate canonical motor positions with private FK data."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: Sequence[str] = JOINT_NAMES,
    ) -> None:
        names = tuple(str(name) for name in joint_names)
        if names != JOINT_NAMES:
            raise ValueError("joint_names must use the canonical 20-motor order")
        self.model = model
        self.joint_names = names
        joint_ids = np.asarray(
            [_required_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names],
            dtype=np.int32,
        )
        self._joint_qpos_addresses = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)

    def robot_directions(self, positions_rad: Sequence[float]) -> AnatomicalDirections:
        """Run FK in fresh data, leaving every caller-owned state untouched."""

        positions = np.asarray(positions_rad, dtype=np.float64)
        if positions.shape != (len(self.joint_names),) or not np.isfinite(positions).all():
            raise ValueError("positions_rad must be a finite canonical 20-vector")
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)
        data.qpos[self._joint_qpos_addresses] = positions
        mujoco.mj_forward(self.model, data)
        return robot_anatomical_directions(self.model, data)

    def evaluate(
        self,
        frame: SkeletonFrame,
        positions_rad: Sequence[float],
        *,
        confidence_threshold: float = 0.55,
    ) -> PoseFidelity:
        human = human_anatomical_directions(
            frame, confidence_threshold=confidence_threshold
        )
        robot = self.robot_directions(positions_rad)
        return angular_pose_fidelity(human, robot)
