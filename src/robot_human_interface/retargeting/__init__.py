"""Geometry-based human-to-humanoid retargeting baseline."""

from .geometry import (
    HUMAN_TO_ROBOT_MAPPING,
    UPPER_BODY_REQUIRED,
    WHOLE_BODY_REQUIRED,
    canonicalize_mirrored_skeleton,
    compute_human_joint_angles,
    joint_landmark_validity,
)
from .retargeter import (
    DEFAULT_JOINT_SPECS,
    GeometricRetargeter,
    JointSpec,
    RetargetingConfig,
    load_joint_specs,
    load_retargeting_config,
)

__all__ = [
    "DEFAULT_JOINT_SPECS",
    "GeometricRetargeter",
    "HUMAN_TO_ROBOT_MAPPING",
    "JointSpec",
    "RetargetingConfig",
    "UPPER_BODY_REQUIRED",
    "WHOLE_BODY_REQUIRED",
    "canonicalize_mirrored_skeleton",
    "compute_human_joint_angles",
    "load_joint_specs",
    "load_retargeting_config",
    "joint_landmark_validity",
]
