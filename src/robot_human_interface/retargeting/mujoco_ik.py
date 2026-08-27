"""Bounded whole-body inverse-kinematics retargeting on the MuJoCo model."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi
from pathlib import Path
from time import monotonic
from typing import Callable, Mapping, Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from robot_human_interface.pose.calibration import (
    NeutralCalibrationError,
    NeutralCalibrationGate,
)
from robot_human_interface.resources import ResourceLocator
from robot_human_interface.skeleton import (
    JOINT_NAMES,
    PoseLandmark as L,
    RobotJointCommand,
    SkeletonFrame,
    canonicalize_mirrored_skeleton,
)

from .geometry import BodyBasis, UPPER_BODY_REQUIRED, WHOLE_BODY_REQUIRED, body_basis
from .retargeter import (
    DEFAULT_JOINT_SPECS,
    JointSpec,
    RetargetingConfig,
    load_joint_specs,
    load_retargeting_config,
)


FloatArray = NDArray[np.float64]


_SITE_WEIGHTS: Mapping[str, float] = {
    "right_elbow_ik": 1.5,
    "right_wrist_ik": 3.0,
    "right_hand_ik": 0.6,
    "left_elbow_ik": 1.5,
    "left_wrist_ik": 3.0,
    "left_hand_ik": 0.6,
    "right_knee_ik": 1.5,
    "right_ankle_ik": 3.0,
    "right_toe_ik": 0.8,
    "left_knee_ik": 1.5,
    "left_ankle_ik": 3.0,
    "left_toe_ik": 0.8,
    "head_nose_ik": 2.0,
}

_DIRECTION_WEIGHTS: Mapping[str, float] = {
    "right_thigh": 0.35,
    "right_shin": 0.35,
    "right_foot": 0.15,
    "right_leg": 0.50,
    "left_thigh": 0.35,
    "left_shin": 0.35,
    "left_foot": 0.15,
    "left_leg": 0.50,
    "face": 1.25,
}

_IMAGE_LIFT_DEADBAND = 0.04
_IMAGE_FLEX_GAIN = 0.50
_IMAGE_MAX_LIFT_FRACTION = 0.75
_IMAGE_LEG_DIRECTION_SCALE = 0.15
_IMAGE_ANKLE_HEIGHT_WEIGHT = 30.0
_IMAGE_TOE_HEIGHT_WEIGHT = 12.0

_CALIBRATION_LANDMARKS = UPPER_BODY_REQUIRED + (
    L.LEFT_KNEE,
    L.RIGHT_KNEE,
    L.LEFT_ANKLE,
    L.RIGHT_ANKLE,
)


@dataclass(frozen=True, slots=True)
class IKDiagnostics:
    marker_count: int
    cost: float
    optimality: float
    evaluations: int
    success: bool


_RESOURCES = ResourceLocator()


def _unit(vector: object) -> FloatArray | None:
    result = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if result.shape != (3,) or not np.isfinite(result).all() or norm < 1e-8:
        return None
    return result / norm


def _rotation_from_to(first: FloatArray, second: FloatArray) -> FloatArray:
    """Return a deterministic proper rotation taking unit ``first`` to ``second``."""

    a = _unit(first)
    b = _unit(second)
    if a is None or b is None:
        return np.eye(3)
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.eye(3)
        basis = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = _unit(np.cross(a, basis))
        assert axis is not None
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    x, y, z = cross
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


class MujocoIKRetargeter:
    """Fit the actual 20-DOF robot FK to calibrated human limb directions.

    The floating base is held fixed inside the kinematic solve; only the same
    twenty bounded motor angles transmitted to Unity/the real robot vary.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        joint_specs: Sequence[JointSpec] | None = None,
        config: RetargetingConfig | None = None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.joint_specs = tuple(joint_specs or DEFAULT_JOINT_SPECS)
        if tuple(item.name for item in self.joint_specs) != JOINT_NAMES:
            raise ValueError("joint_specs must use canonical joint order")
        self.config = config or RetargetingConfig()
        self._clock = clock
        path = Path(
            model_path or _RESOURCES.model("humanoid", "scene_fixed.xml")
        )
        self.model_path = path.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._lower = np.asarray([item.lower_rad for item in self.joint_specs])
        self._upper = np.asarray([item.upper_rad for item in self.joint_specs])
        self._home = np.asarray([item.start_rad + item.zero_offset_rad for item in self.joint_specs])
        self._home = np.clip(self._home, self._lower, self._upper)
        self._joint_ids = np.asarray([
            self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES
        ])
        self._qpos_adr = self.model.jnt_qposadr[self._joint_ids].copy()
        self._site_ids = {
            name: self._id(mujoco.mjtObj.mjOBJ_SITE, name) for name in _SITE_WEIGHTS
        }
        self._site_ids.update({
            name: self._id(mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("right_foot_contact", "left_foot_contact")
        })
        self._body_ids = {
            name: self._id(mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (
                "shoulder_rh", "shoulder_lh", "knee_rl", "knee_ll", "head"
            )
        }
        self._set_q(self._home)
        self._home_sites = {
            name: self.data.site_xpos[identifier].copy()
            for name, identifier in self._site_ids.items()
        }
        self._home_roots = {
            name: self.data.xpos[identifier].copy()
            for name, identifier in self._body_ids.items()
        }
        self._home_vectors = self._make_home_vectors()
        self._home_foot_offsets = {
            side: self._home_sites[f"{side}_toe_ik"] - self._home_sites[f"{side}_ankle_ik"]
            for side in ("right", "left")
        }
        self._home_leg_lengths = {
            "right": float(
                np.linalg.norm(
                    self._home_sites["right_ankle_ik"]
                    - self._home_roots["knee_rl"]
                )
            ),
            "left": float(
                np.linalg.norm(
                    self._home_sites["left_ankle_ik"]
                    - self._home_roots["knee_ll"]
                )
            ),
        }
        self.reset()

    @classmethod
    def from_yaml(
        cls,
        *,
        joints_path: str | Path | None = None,
        retargeting_path: str | Path | None = None,
        model_path: str | Path | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> "MujocoIKRetargeter":
        return cls(
            model_path,
            load_joint_specs(joints_path),
            load_retargeting_config(retargeting_path),
            clock=clock,
        )

    def _id(self, kind: mujoco.mjtObj, name: str) -> int:
        identifier = int(mujoco.mj_name2id(self.model, kind, name))
        if identifier < 0:
            raise ValueError(f"IK model is missing {kind.name} {name!r}")
        return identifier

    @property
    def neutral_positions_rad(self) -> FloatArray:
        return self._home.copy()

    @property
    def calibration_progress(self) -> float:
        if self._calibration_target == 0:
            return 1.0
        assert self._calibration_gate is not None
        return self._calibration_gate.progress

    @property
    def is_calibrating(self) -> bool:
        return self._calibration_target > 0

    @property
    def last_diagnostics(self) -> IKDiagnostics | None:
        return self._last_diagnostics

    def reset(self) -> None:
        self._references: dict[str, FloatArray] = {}
        self._alignments: dict[str, FloatArray] = {}
        self._fixed_image_up: FloatArray | None = None
        self._image_leg_scale: float | None = None
        self._neutral_ankle_difference_ratio = 0.0
        self._neutral_leg_flex = {"right": 0.0, "left": 0.0}
        self._calibration_target = self.config.auto_calibration_frames
        self._calibration_frames: list[SkeletonFrame] = []
        self._calibration_gate = self._new_calibration_gate(
            self._calibration_target
        )
        self._last_output: FloatArray | None = None
        self._last_output_timestamp: float | None = None
        self._last_valid_positions: FloatArray | None = None
        self._last_valid_timestamp: float | None = None
        self._last_diagnostics: IKDiagnostics | None = None

    def reset_temporal(self) -> None:
        """Clear solver history while retaining accepted neutral references."""

        if self._calibration_target > 0:
            self._calibration_frames = []
            self._calibration_gate = self._new_calibration_gate(
                self._calibration_target
            )
        self._last_output = None
        self._last_output_timestamp = None
        self._last_valid_positions = None
        self._last_valid_timestamp = None
        self._last_diagnostics = None

    def start_calibration(self, sample_count: int = 15) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self.reset()
        self._calibration_target = int(sample_count)
        self._calibration_gate = self._new_calibration_gate(
            self._calibration_target
        )

    def _new_calibration_gate(
        self, sample_count: int
    ) -> NeutralCalibrationGate | None:
        if sample_count == 0:
            return None
        landmarks = (
            UPPER_BODY_REQUIRED
            if self.config.mode == "upper_body"
            else _CALIBRATION_LANDMARKS
        )
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
            label="MuJoCo IK neutral-pose",
        )

    def _set_q(self, positions: FloatArray) -> None:
        self.data.qpos[self._qpos_adr] = positions
        mujoco.mj_forward(self.model, self.data)

    def _make_home_vectors(self) -> dict[str, FloatArray]:
        s = self._home_sites
        r = self._home_roots
        return {
            "right_upper": s["right_elbow_ik"] - r["shoulder_rh"],
            "right_forearm": s["right_wrist_ik"] - s["right_elbow_ik"],
            "right_hand": s["right_hand_ik"] - s["right_wrist_ik"],
            "left_upper": s["left_elbow_ik"] - r["shoulder_lh"],
            "left_forearm": s["left_wrist_ik"] - s["left_elbow_ik"],
            "left_hand": s["left_hand_ik"] - s["left_wrist_ik"],
            # In the inherited Unity naming knee_* is the hip-pitch body and
            # shin_* (the right/left_knee_ik site) is the anatomical knee.
            # Starting the morphology chain at rotat_axis_* incorrectly folded
            # the spatially separated yaw/roll mechanism into the thigh.
            "right_thigh": s["right_knee_ik"] - r["knee_rl"],
            "right_shin": s["right_ankle_ik"] - s["right_knee_ik"],
            "right_foot": s["right_toe_ik"] - s["right_foot_contact"],
            "left_thigh": s["left_knee_ik"] - r["knee_ll"],
            "left_shin": s["left_ankle_ik"] - s["left_knee_ik"],
            "left_foot": s["left_toe_ik"] - s["left_foot_contact"],
            # Pose fidelity measures the physical head forward axis.  The
            # visual nose marker is intentionally above that axis, so using
            # head->nose as the IK direction injects a false pitch offset.
            "face": np.array((-1.0, 0.0, 0.0)),
        }

    def _frame_is_valid(self, frame: SkeletonFrame) -> bool:
        required = UPPER_BODY_REQUIRED if self.config.mode == "upper_body" else WHOLE_BODY_REQUIRED
        return (
            frame.coverage(required, self.config.confidence_threshold) >= self.config.minimum_coverage
            and frame.mean_confidence(required) >= self.config.confidence_threshold
        )

    def _canonical_frame(self, frame: SkeletonFrame) -> SkeletonFrame:
        return canonicalize_mirrored_skeleton(frame) if self.config.mirrored_input else frame

    @staticmethod
    def _image_points(frame: SkeletonFrame) -> FloatArray:
        """Return normalized 2-D landmarks in image-height units."""

        points = frame.landmarks_2d.copy()
        if frame.image_size is not None:
            width, height = frame.image_size
            points[:, 0] *= float(width) / float(height)
        return points

    @staticmethod
    def _leg_flex_2d(points: FloatArray, *, left: bool) -> float:
        hip = int(L.LEFT_HIP if left else L.RIGHT_HIP)
        knee = int(L.LEFT_KNEE if left else L.RIGHT_KNEE)
        ankle = int(L.LEFT_ANKLE if left else L.RIGHT_ANKLE)
        upper = float(np.linalg.norm(points[knee] - points[hip]))
        lower = float(np.linalg.norm(points[ankle] - points[knee]))
        chain = upper + lower
        if not np.isfinite(chain) or chain < 1e-8:
            return 0.0
        chord = float(np.linalg.norm(points[ankle] - points[hip]))
        return float(np.clip(1.0 - chord / chain, 0.0, 1.0))

    def _install_image_reference(self, frames: Sequence[SkeletonFrame]) -> bool:
        canonical = [self._canonical_frame(frame) for frame in frames]
        required = np.asarray(
            [
                int(L.LEFT_SHOULDER),
                int(L.RIGHT_SHOULDER),
                int(L.LEFT_HIP),
                int(L.RIGHT_HIP),
                int(L.LEFT_KNEE),
                int(L.RIGHT_KNEE),
                int(L.LEFT_ANKLE),
                int(L.RIGHT_ANKLE),
            ],
            dtype=np.int64,
        )
        usable = [
            frame
            for frame in canonical
            if bool(np.all(frame.valid_mask(self.config.confidence_threshold)[required]))
        ]
        if not usable:
            return False

        image_points = [self._image_points(frame) for frame in usable]
        up_samples: list[FloatArray] = []
        scale_samples: list[float] = []
        for points in image_points:
            shoulders = 0.5 * (
                points[int(L.LEFT_SHOULDER)]
                + points[int(L.RIGHT_SHOULDER)]
            )
            hips = 0.5 * (
                points[int(L.LEFT_HIP)] + points[int(L.RIGHT_HIP)]
            )
            up = _unit(np.append(shoulders - hips, 0.0))
            if up is not None:
                up_samples.append(up[:2])
            leg_lengths = []
            for left in (False, True):
                hip = int(L.LEFT_HIP if left else L.RIGHT_HIP)
                knee = int(L.LEFT_KNEE if left else L.RIGHT_KNEE)
                ankle = int(L.LEFT_ANKLE if left else L.RIGHT_ANKLE)
                leg_lengths.append(
                    np.linalg.norm(points[knee] - points[hip])
                    + np.linalg.norm(points[ankle] - points[knee])
                )
            scale_samples.append(float(np.mean(leg_lengths)))

        image_up = _unit(
            np.append(np.median(np.asarray(up_samples), axis=0), 0.0)
        ) if up_samples else None
        image_scale = float(np.median(scale_samples)) if scale_samples else 0.0
        if image_up is None or not np.isfinite(image_scale) or image_scale < 1e-6:
            return False
        self._fixed_image_up = image_up[:2]
        self._image_leg_scale = image_scale
        ankle_differences = [
            float(
                np.dot(
                    points[int(L.RIGHT_ANKLE)] - points[int(L.LEFT_ANKLE)],
                    self._fixed_image_up,
                )
                / image_scale
            )
            for points in image_points
        ]
        self._neutral_ankle_difference_ratio = float(
            np.median(ankle_differences)
        )
        self._neutral_leg_flex = {
            side: float(
                np.median(
                    [
                        self._leg_flex_2d(points, left=side == "left")
                        for points in image_points
                    ]
                )
            )
            for side in ("right", "left")
        }
        return True

    def _image_leg_lifts(self, frame: SkeletonFrame) -> dict[str, float] | None:
        """Map an unambiguous 2-D swing side to bounded robot clearance."""

        if (
            self.config.mode != "whole_body"
            or self._fixed_image_up is None
            or self._image_leg_scale is None
        ):
            return None
        frame = self._canonical_frame(frame)
        required = np.asarray(
            [
                int(L.LEFT_HIP),
                int(L.RIGHT_HIP),
                int(L.LEFT_KNEE),
                int(L.RIGHT_KNEE),
                int(L.LEFT_ANKLE),
                int(L.RIGHT_ANKLE),
            ],
            dtype=np.int64,
        )
        if not bool(
            np.all(frame.valid_mask(self.config.confidence_threshold)[required])
        ):
            return None
        points = self._image_points(frame)
        signed_right_lift = float(
            np.dot(
                points[int(L.RIGHT_ANKLE)] - points[int(L.LEFT_ANKLE)],
                self._fixed_image_up,
            )
            / self._image_leg_scale
            - self._neutral_ankle_difference_ratio
        )
        flex_delta = (
            self._leg_flex_2d(points, left=False)
            - self._neutral_leg_flex["right"]
            - self._leg_flex_2d(points, left=True)
            + self._neutral_leg_flex["left"]
        )
        signed_right_lift += _IMAGE_FLEX_GAIN * flex_delta
        magnitude = max(0.0, abs(signed_right_lift) - _IMAGE_LIFT_DEADBAND)
        if magnitude <= 0.0:
            return None
        lifted_side = "right" if signed_right_lift > 0.0 else "left"
        lifts = {"right": 0.0, "left": 0.0}
        lifts[lifted_side] = min(
            magnitude * self._home_leg_lengths[lifted_side],
            _IMAGE_MAX_LIFT_FRACTION * self._home_leg_lengths[lifted_side],
        )
        return lifts

    def _observations(self, frame: SkeletonFrame) -> tuple[dict[str, FloatArray], dict[str, float]]:
        frame = self._canonical_frame(frame)
        points = frame.landmarks_3d
        valid = frame.valid_mask(self.config.confidence_threshold)
        scores = frame.confidence()
        torso_basis = body_basis(points)
        basis_vectors = (
            torso_basis.forward,
            torso_basis.lateral_right,
            torso_basis.vertical_up,
        )
        if not all(np.isfinite(vector).all() for vector in basis_vectors):
            return {}, {}

        def transform(vector: FloatArray, basis: BodyBasis) -> FloatArray | None:
            # Human anatomical forward/right/up -> robot -X/-Y/+Z.
            return _unit(np.array((
                -np.dot(vector, basis.forward),
                -np.dot(vector, basis.lateral_right),
                np.dot(vector, basis.vertical_up),
            )))

        core = (L.LEFT_SHOULDER, L.RIGHT_SHOULDER, L.LEFT_HIP, L.RIGHT_HIP)
        records: dict[str, tuple[FloatArray, tuple[L, ...]]] = {}

        def add(
            name: str,
            vector: FloatArray,
            dependencies: tuple[L, ...],
            basis: BodyBasis = torso_basis,
        ) -> None:
            indices = core + dependencies
            if not np.all(valid[[int(item) for item in indices]]):
                return
            direction = transform(vector, basis)
            if direction is not None:
                records[name] = (direction, indices)

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
            add(f"{side}_upper", points[int(elbow)] - points[int(shoulder)], (shoulder, elbow))
            add(f"{side}_forearm", points[int(wrist)] - points[int(elbow)], (elbow, wrist))
            hand_end = 0.5 * (points[int(index)] + points[int(pinky)])
            add(f"{side}_hand", hand_end - points[int(wrist)], (wrist, index, pinky))
            if self.config.mode == "whole_body":
                add(
                    f"{side}_thigh",
                    points[int(knee)] - points[int(hip)],
                    (hip, knee),
                    torso_basis,
                )
                add(
                    f"{side}_shin",
                    points[int(ankle)] - points[int(knee)],
                    (knee, ankle),
                    torso_basis,
                )
                add(
                    f"{side}_foot",
                    points[int(toe)] - points[int(heel)],
                    (ankle, heel, toe),
                    torso_basis,
                )
        face_center = 0.5 * (points[int(L.LEFT_EAR)] + points[int(L.RIGHT_EAR)])
        add("face", points[int(L.NOSE)] - face_center, (L.NOSE, L.LEFT_EAR, L.RIGHT_EAR))
        directions = {name: value[0] for name, value in records.items()}
        confidences = {
            name: float(np.min(scores[[int(item) for item in value[1]]]))
            for name, value in records.items()
        }
        return directions, confidences

    def _install_references(self, directions: Mapping[str, FloatArray]) -> None:
        for name, direction in directions.items():
            reference = _unit(direction)
            home = _unit(self._home_vectors[name])
            if reference is not None and home is not None:
                self._references[name] = reference
                self._alignments[name] = _rotation_from_to(reference, home)

    def calibrate(self, frame: SkeletonFrame) -> bool:
        """Explicitly override auto-admission with one caller-approved frame."""

        if not self._frame_is_valid(frame):
            return False
        gate = self._new_calibration_gate(1)
        assert gate is not None
        if not gate.accepts_explicit(frame):
            return False
        if not self._install_image_reference((frame,)):
            return False
        directions, _ = self._observations(frame)
        if not directions:
            return False
        # Explicit calibration replaces the complete task reference set.
        # Optional tasks absent from this approved frame must stay unavailable
        # instead of inheriting face/hand/foot alignments from an older epoch.
        self._references = {}
        self._alignments = {}
        self._install_references(directions)
        self._calibration_target = 0
        self._calibration_frames = []
        self._calibration_gate = None
        self._last_output = None
        self._last_output_timestamp = None
        # Do not let stale-pose fallback revive a command computed against the
        # previous calibration after the reference geometry has changed.
        self._last_valid_positions = None
        self._last_valid_timestamp = None
        self._last_diagnostics = None
        return True

    def _observe_calibration(self, frame: SkeletonFrame) -> bool:
        if self._calibration_target == 0:
            return False
        assert self._calibration_gate is not None
        accepted = self._calibration_gate.observe(frame)
        if accepted is None:
            return True
        if not self._install_image_reference(accepted):
            raise NeutralCalibrationError(
                "MuJoCo IK neutral-pose calibration produced an invalid image "
                "reference; hold a clearly visible, still, two-foot pose and "
                "explicitly restart calibration"
            )
        calibration_samples = [
            self._observations(sample)[0] for sample in accepted
        ]
        averaged: dict[str, FloatArray] = {}
        for name in self._home_vectors:
            values = [sample[name] for sample in calibration_samples if name in sample]
            # Optional tasks (face, hands, feet) are not part of the neutral
            # gate's mandatory landmark set.  A single intermittently visible
            # frame must not silently define their reference for the whole
            # session; admit a task only when every accepted neutral sample
            # observed it.
            if len(values) == len(calibration_samples):
                mean = _unit(np.mean(values, axis=0))
                if mean is not None:
                    averaged[name] = mean
        self._install_references(averaged)
        self._calibration_target = 0
        self._calibration_frames = []
        self._calibration_gate = None
        return True

    def _aligned(self, name: str, direction: FloatArray) -> FloatArray:
        if name not in self._alignments:
            raise KeyError(
                f"IK task {name!r} was unavailable during neutral calibration"
            )
        return self._alignments[name] @ direction * np.linalg.norm(self._home_vectors[name])

    def _targets(
        self,
        directions: Mapping[str, FloatArray],
        confidences: Mapping[str, float],
        image_lifts_m: Mapping[str, float] | None = None,
    ) -> list[tuple[str, FloatArray, float]]:
        targets: list[tuple[str, FloatArray, float]] = []
        for side, shoulder, hip in (
            ("right", "shoulder_rh", "knee_rl"),
            ("left", "shoulder_lh", "knee_ll"),
        ):
            upper, forearm, hand = (f"{side}_{part}" for part in ("upper", "forearm", "hand"))
            if upper in directions:
                elbow_target = self._home_roots[shoulder] + self._aligned(upper, directions[upper])
                targets.append((f"{side}_elbow_ik", elbow_target, confidences[upper]))
                if forearm in directions:
                    wrist_target = elbow_target + self._aligned(forearm, directions[forearm])
                    targets.append((f"{side}_wrist_ik", wrist_target, min(confidences[upper], confidences[forearm])))
                    if hand in directions:
                        hand_target = wrist_target + self._aligned(hand, directions[hand])
                        targets.append((f"{side}_hand_ik", hand_target, min(confidences[upper], confidences[forearm], confidences[hand])))
            thigh, shin, foot = (f"{side}_{part}" for part in ("thigh", "shin", "foot"))
            if self.config.mode == "whole_body" and thigh in directions:
                knee_target = self._home_roots[hip] + self._aligned(thigh, directions[thigh])
                targets.append((f"{side}_knee_ik", knee_target, confidences[thigh]))
                if shin in directions:
                    ankle_target = knee_target + self._aligned(shin, directions[shin])
                    if image_lifts_m is not None and side in image_lifts_m:
                        # MediaPipe world depth can invert the visibly lifted
                        # side.  Keep its horizontal 3-D target, but make sole
                        # clearance agree with confidence-gated image geometry.
                        ankle_target[2] = (
                            self._home_sites[f"{side}_ankle_ik"][2]
                            + float(image_lifts_m[side])
                        )
                    targets.append((f"{side}_ankle_ik", ankle_target, min(confidences[thigh], confidences[shin])))
                    if foot in directions:
                        sole_target = self._aligned(foot, directions[foot])
                        sole_unit = _unit(sole_target)
                        home_sole = _unit(self._home_vectors[foot])
                        home_offset = self._home_foot_offsets[side]
                        if sole_unit is None or home_sole is None:
                            continue
                        # The ankle motor origin is not the heel. Preserve the
                        # fixed neutral transform between sole heading and the
                        # ankle-to-toe offset while rotating the physical sole.
                        offset_alignment = _rotation_from_to(home_sole, home_offset)
                        toe_offset = (
                            offset_alignment @ sole_unit * np.linalg.norm(home_offset)
                        )
                        toe_target = ankle_target + toe_offset
                        if image_lifts_m is not None and side in image_lifts_m:
                            toe_target[2] = (
                                self._home_sites[f"{side}_toe_ik"][2]
                                + float(image_lifts_m[side])
                            )
                        targets.append((f"{side}_toe_ik", toe_target, min(confidences[thigh], confidences[shin], confidences[foot])))
        return targets

    def _solve(
        self,
        targets: Sequence[tuple[str, FloatArray, float]],
        directions: Mapping[str, FloatArray],
        confidences: Mapping[str, float],
        *,
        leg_direction_scale: float = 1.0,
        image_lifts_m: Mapping[str, float] | None = None,
    ) -> FloatArray:
        initial = self._last_valid_positions.copy() if self._last_valid_positions is not None else self._home.copy()

        direction_targets: list[tuple[str, FloatArray, float]] = []
        for side in ("right", "left"):
            chain: list[FloatArray] = []
            chain_confidence = 1.0
            for part in ("thigh", "shin", "foot"):
                name = f"{side}_{part}"
                if name not in directions:
                    continue
                target = self._aligned(name, directions[name])
                unit_target = _unit(target)
                if unit_target is not None:
                    direction_targets.append((name, unit_target, confidences[name]))
                    if part != "foot":
                        chain.append(target)
                        chain_confidence = min(chain_confidence, confidences[name])
            if len(chain) == 2:
                full_leg = _unit(chain[0] + chain[1])
                if full_leg is not None:
                    direction_targets.append((f"{side}_leg", full_leg, chain_confidence))
        if "face" in directions:
            face_target = _unit(self._aligned("face", directions["face"]))
            if face_target is not None:
                direction_targets.append(("face", face_target, confidences["face"]))

        def current_direction(name: str) -> FloatArray:
            if name == "face":
                rotation = self.data.xmat[self._body_ids["head"]].reshape(3, 3)
                return rotation @ np.array((-1.0, 0.0, 0.0))
            side, part = name.split("_", 1)
            suffix = "rl" if side == "right" else "ll"
            knee = self.data.site_xpos[self._site_ids[f"{side}_knee_ik"]]
            ankle = self.data.site_xpos[self._site_ids[f"{side}_ankle_ik"]]
            if part == "thigh":
                return knee - self.data.xpos[self._body_ids[f"knee_{suffix}"]]
            if part == "shin":
                return ankle - knee
            if part == "foot":
                toe = self.data.site_xpos[self._site_ids[f"{side}_toe_ik"]]
                sole = self.data.site_xpos[self._site_ids[f"{side}_foot_contact"]]
                return toe - sole
            if part == "leg":
                return ankle - self.data.xpos[self._body_ids[f"knee_{suffix}"]]
            raise AssertionError(name)

        def residual(positions: FloatArray) -> FloatArray:
            self._set_q(positions)
            values: list[FloatArray] = []
            for name, target, confidence in targets:
                weight = np.sqrt(_SITE_WEIGHTS[name] * confidence)
                values.append(weight * (self.data.site_xpos[self._site_ids[name]] - target))
            if image_lifts_m is not None:
                for side, lift_m in image_lifts_m.items():
                    leg_confidence = min(
                        confidences.get(f"{side}_thigh", 0.0),
                        confidences.get(f"{side}_shin", 0.0),
                    )
                    if leg_confidence <= 0.0:
                        continue
                    ankle_name = f"{side}_ankle_ik"
                    toe_name = f"{side}_toe_ik"
                    ankle_error = (
                        self.data.site_xpos[self._site_ids[ankle_name], 2]
                        - self._home_sites[ankle_name][2]
                        - float(lift_m)
                    )
                    toe_error = (
                        self.data.site_xpos[self._site_ids[toe_name], 2]
                        - self._home_sites[toe_name][2]
                        - float(lift_m)
                    )
                    values.append(
                        np.asarray(
                            [
                                np.sqrt(
                                    _IMAGE_ANKLE_HEIGHT_WEIGHT * leg_confidence
                                )
                                * ankle_error,
                                np.sqrt(
                                    _IMAGE_TOE_HEIGHT_WEIGHT * leg_confidence
                                )
                                * toe_error,
                            ]
                        )
                    )
            for name, target, confidence in direction_targets:
                current = _unit(current_direction(name))
                if current is not None:
                    direction_scale = (
                        1.0 if name == "face" else leg_direction_scale
                    )
                    weight = np.sqrt(
                        _DIRECTION_WEIGHTS[name]
                        * confidence
                        * direction_scale
                    )
                    values.append(weight * (current - target))
            values.append(np.sqrt(0.0015) * (positions - initial))
            values.append(np.sqrt(0.00015) * (positions - self._home))
            return np.concatenate(values)

        result = least_squares(
            residual,
            initial,
            bounds=(self._lower, self._upper),
            method="trf",
            max_nfev=12,
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-5,
        )
        positions = self._nearest_bounded_equivalent(
            np.asarray(result.x), initial
        )
        self._last_diagnostics = IKDiagnostics(
            marker_count=len(targets) + len(direction_targets),
            cost=float(result.cost),
            optimality=float(result.optimality),
            evaluations=int(result.nfev),
            success=bool(result.success),
        )
        return positions

    def _nearest_bounded_equivalent(
        self,
        positions: FloatArray,
        reference: FloatArray,
    ) -> FloatArray:
        """Keep full-turn joints on the nearest continuous angle branch.

        ``+179 deg`` and ``-179 deg`` are nearly the same orientation, but a
        limited hinge sees their raw difference as a 358-degree command.  For
        full-turn joints choose the equivalent angle nearest the previous
        command and clamp at the physical boundary instead of jumping across
        it.  Narrow-range joints remain ordinary non-periodic coordinates.
        """

        candidate = np.asarray(positions, dtype=np.float64).copy()
        previous = np.asarray(reference, dtype=np.float64)
        full_turn = (self._upper - self._lower) >= (2.0 * pi - 1e-6)
        delta = np.arctan2(
            np.sin(candidate[full_turn] - previous[full_turn]),
            np.cos(candidate[full_turn] - previous[full_turn]),
        )
        candidate[full_turn] = previous[full_turn] + delta
        return np.clip(candidate, self._lower, self._upper)

    def _smooth(self, desired: FloatArray, timestamp_s: float) -> FloatArray:
        if self._last_output is None or self._last_output_timestamp is None:
            result = desired.copy()
        elif timestamp_s < self._last_output_timestamp or self.config.smoothing_time_constant_s == 0.0:
            result = desired.copy()
        else:
            alpha = 1.0 - exp(-(timestamp_s - self._last_output_timestamp) / self.config.smoothing_time_constant_s)
            result = self._last_output + alpha * (desired - self._last_output)
        self._last_output = result.copy()
        self._last_output_timestamp = timestamp_s
        return result

    def _fallback(self, timestamp_s: float) -> RobotJointCommand:
        if self._last_valid_positions is None or self._last_valid_timestamp is None:
            positions = self._home.copy()
        else:
            stale_for = max(0.0, timestamp_s - self._last_valid_timestamp)
            if stale_for <= self.config.hold_seconds:
                positions = self._last_valid_positions.copy()
            else:
                progress = min(1.0, (stale_for - self.config.hold_seconds) / self.config.return_seconds)
                positions = (1.0 - progress) * self._last_valid_positions + progress * self._home
        self._last_output = positions.copy()
        self._last_output_timestamp = timestamp_s
        return RobotJointCommand.humanoid(timestamp_s, positions, 0.0, stale=True)

    def retarget(
        self,
        frame: SkeletonFrame | None,
        *,
        timestamp_s: float | None = None,
    ) -> RobotJointCommand:
        now = float(timestamp_s if timestamp_s is not None else (frame.timestamp_s if frame else self._clock()))
        if frame is None or not self._frame_is_valid(frame):
            return self._fallback(now)
        directions, confidences = self._observations(frame)
        if not directions:
            return self._fallback(now)
        if self._observe_calibration(frame):
            positions = self._home.copy()
        else:
            # A task missing from the explicit neutral window must not silently
            # treat its first later-seen (possibly moving) pose as neutral.
            # Keep it unavailable until the caller deliberately recalibrates.
            directions = {
                name: value
                for name, value in directions.items()
                if name in self._alignments
            }
            confidences = {
                name: value
                for name, value in confidences.items()
                if name in directions
            }
            image_lifts_m = self._image_leg_lifts(frame)
            targets = self._targets(directions, confidences, image_lifts_m)
            if not targets:
                return self._fallback(now)
            positions = self._smooth(
                self._solve(
                    targets,
                    directions,
                    confidences,
                    leg_direction_scale=(
                        _IMAGE_LEG_DIRECTION_SCALE
                        if image_lifts_m is not None
                        else 1.0
                    ),
                    image_lifts_m=image_lifts_m,
                ),
                now,
            )
        self._last_output = positions.copy()
        self._last_output_timestamp = now
        self._last_valid_positions = positions.copy()
        self._last_valid_timestamp = now
        confidence = float(np.mean(tuple(confidences.values()))) if confidences else 0.0
        return RobotJointCommand.humanoid(now, positions, np.clip(confidence, 0.0, 1.0))
