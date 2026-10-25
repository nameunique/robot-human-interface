from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np

from robot_human_interface.pose import make_synthetic_skeleton
from robot_human_interface.retargeting import (
    MujocoIKRetargeter,
    MujocoPoseFidelityEvaluator,
    RetargetingConfig,
    load_joint_specs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = PROJECT_ROOT / "models" / "volodka" / "scene_fixed.xml"
JOINTS = PROJECT_ROOT / "config" / "joints_volodka.yaml"


def _retargeter() -> MujocoIKRetargeter:
    return MujocoIKRetargeter(
        MODEL,
        load_joint_specs(JOINTS),
        RetargetingConfig(
            mode="whole_body",
            auto_calibration_frames=0,
            smoothing_time_constant_s=0.0,
        ),
    )


def test_volodka_descriptor_and_arm_sites_define_native_anatomy() -> None:
    retargeter = _retargeter()
    evaluator = MujocoPoseFidelityEvaluator(retargeter.model)
    home = retargeter.neutral_positions_rad
    directions = evaluator.robot_directions(home)

    assert retargeter._arm_root_site_names == {
        "right": "right_shoulder_ik",
        "left": "left_shoulder_ik",
    }
    np.testing.assert_allclose(retargeter._robot_forward_axis, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(retargeter._head_forward_axis, (1.0, 0.0, 0.0))
    for name in (
        "right_upper_arm",
        "left_upper_arm",
        "right_forearm",
        "left_forearm",
    ):
        np.testing.assert_allclose(directions.vector(name), (0.0, 0.0, -1.0), atol=1e-9)
    for name in ("right_foot", "left_foot", "head"):
        np.testing.assert_allclose(directions.vector(name), (1.0, 0.0, 0.0), atol=2e-5)


def test_volodka_legacy_api_arm_signs_have_bilateral_meaning() -> None:
    retargeter = _retargeter()
    evaluator = MujocoPoseFidelityEvaluator(retargeter.model)
    home = retargeter.neutral_positions_rad

    def direction(index: int, name: str) -> np.ndarray:
        positions = home.copy()
        positions[index] += 0.2
        return evaluator.robot_directions(positions).vector(name)

    # Both legacy shoulder commands raise the matching upper arm forward.
    assert direction(0, "right_upper_arm")[0] > 0.15
    assert direction(1, "left_upper_arm")[0] > 0.15
    # Positive second-axis commands move each arm outwards.
    assert direction(2, "right_upper_arm")[1] > 0.15
    assert direction(3, "left_upper_arm")[1] < -0.15
    # Positive final-axis commands flex each forearm inwards.
    assert direction(4, "right_forearm")[1] < -0.15
    assert direction(5, "left_forearm")[1] > 0.15


def test_volodka_ik_tracks_forward_right_arm_in_positive_x_model() -> None:
    neutral = make_synthetic_skeleton(1.0)
    right_arm_forward = make_synthetic_skeleton(1.1, phase_rad=pi / 2.0)
    retargeter = _retargeter()
    assert retargeter.calibrate(neutral)

    command = retargeter.retarget(right_arm_forward)
    fidelity = MujocoPoseFidelityEvaluator(retargeter.model).evaluate(
        right_arm_forward, command.positions_rad
    )

    assert command.positions_rad[0] > retargeter.neutral_positions_rad[0] + 1.2
    assert fidelity.error_deg("right_arm") < 10.0
    assert fidelity.error_deg("left_arm") < 3.0
    assert retargeter.last_diagnostics is not None
    assert retargeter.last_diagnostics.success
