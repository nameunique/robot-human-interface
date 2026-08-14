from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
from typing import Callable, get_type_hints

import numpy as np
import pytest

from robot_human_interface.control import (
    HumanSupportIntentConfig,
    HumanSupportIntentEstimator,
    StandingBalanceConfig,
    SupportControlConfig,
    load_human_support_intent_config,
    load_standing_balance_config,
    load_support_control_config,
)
from robot_human_interface.pose import NeutralCalibrationError, make_synthetic_skeleton
from robot_human_interface.retargeting import (
    GeometricRetargeter,
    MujocoIKRetargeter,
    RetargetingConfig,
    load_retargeting_config,
)
from robot_human_interface.skeleton import PoseLandmark as L, SkeletonFrame


def _yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


_CONFIG_SCHEMAS = (
    (
        "retargeting",
        RetargetingConfig,
        load_retargeting_config,
    ),
    (
        "human_support_intent",
        HumanSupportIntentConfig,
        load_human_support_intent_config,
    ),
    (
        "support_control",
        SupportControlConfig,
        load_support_control_config,
    ),
    (
        "standing_balance",
        StandingBalanceConfig,
        load_standing_balance_config,
    ),
)


def _config_fields_of_type(config_type: type[object], expected: type[object]) -> tuple[str, ...]:
    annotations = get_type_hints(config_type)
    return tuple(
        item.name for item in fields(config_type) if annotations[item.name] is expected
    )


_NUMERIC_CONFIG_FIELDS = tuple(
    pytest.param(
        section,
        config_type,
        loader,
        field_name,
        id=f"{section}-{field_name}",
    )
    for section, config_type, loader in _CONFIG_SCHEMAS
    for field_name in (
        _config_fields_of_type(config_type, int)
        + _config_fields_of_type(config_type, float)
    )
)

_BOOLEAN_CONFIG_FIELDS = tuple(
    pytest.param(
        section,
        config_type,
        loader,
        field_name,
        id=f"{section}-{field_name}",
    )
    for section, config_type, loader in _CONFIG_SCHEMAS
    for field_name in _config_fields_of_type(config_type, bool)
)


@pytest.mark.parametrize(
    ("section", "config_type", "loader", "field_name"),
    _NUMERIC_CONFIG_FIELDS,
)
@pytest.mark.parametrize(
    "bad_value",
    (pytest.param(True, id="bool"), pytest.param("not-a-real", id="non-real")),
)
def test_direct_config_constructors_reject_non_real_numeric_fields(
    section: str,
    config_type: type[object],
    loader: Callable[[str | Path | None], object],
    field_name: str,
    bad_value: object,
) -> None:
    del loader
    with pytest.raises(ValueError, match=rf"{section}\.{field_name}.*real number"):
        config_type(**{field_name: bad_value})


@pytest.mark.parametrize(
    ("section", "config_type", "loader", "field_name"),
    _NUMERIC_CONFIG_FIELDS,
)
@pytest.mark.parametrize(
    ("bad_value", "yaml_value"),
    (
        pytest.param(True, "true", id="bool"),
        pytest.param("not-a-real", "'not-a-real'", id="non-real"),
    ),
)
def test_yaml_loaders_reject_non_real_numeric_fields(
    tmp_path: Path,
    section: str,
    config_type: type[object],
    loader: Callable[[str | Path | None], object],
    field_name: str,
    bad_value: object,
    yaml_value: str,
) -> None:
    del config_type, bad_value
    path = _yaml(tmp_path, f"{section}:\n  {field_name}: {yaml_value}\n")

    with pytest.raises(ValueError, match=rf"{section}\.{field_name}.*real number"):
        loader(path)


@pytest.mark.parametrize(
    ("section", "config_type", "loader", "field_name"),
    _BOOLEAN_CONFIG_FIELDS,
)
@pytest.mark.parametrize(
    "bad_value",
    (
        pytest.param(1, id="integer"),
        pytest.param("true", id="string"),
        pytest.param(np.bool_(True), id="numpy-bool"),
    ),
)
def test_direct_config_constructors_require_exact_booleans(
    section: str,
    config_type: type[object],
    loader: Callable[[str | Path | None], object],
    field_name: str,
    bad_value: object,
) -> None:
    del loader
    with pytest.raises(ValueError, match=rf"{section}\.{field_name}.*boolean"):
        config_type(**{field_name: bad_value})


@pytest.mark.parametrize(
    ("section", "config_type", "loader", "field_name"),
    _BOOLEAN_CONFIG_FIELDS,
)
@pytest.mark.parametrize(
    "yaml_value",
    (pytest.param("1", id="integer"), pytest.param("'true'", id="string")),
)
def test_yaml_loaders_require_exact_booleans(
    tmp_path: Path,
    section: str,
    config_type: type[object],
    loader: Callable[[str | Path | None], object],
    field_name: str,
    yaml_value: str,
) -> None:
    del config_type
    path = _yaml(tmp_path, f"{section}:\n  {field_name}: {yaml_value}\n")

    with pytest.raises(ValueError, match=rf"{section}\.{field_name}.*boolean"):
        loader(path)


@pytest.mark.parametrize("setting_name", ("joint_scales", "joint_signs"))
@pytest.mark.parametrize(
    "bad_value",
    (pytest.param(True, id="bool"), pytest.param("1", id="non-real")),
)
def test_retargeting_mapping_values_require_real_numbers(
    setting_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"retargeting\.{setting_name}\.shoulder_rh.*real number",
    ):
        RetargetingConfig(**{setting_name: {"shoulder_rh": bad_value}})


@pytest.mark.parametrize("setting_name", ("joint_scales", "joint_signs"))
@pytest.mark.parametrize(
    "yaml_value",
    (pytest.param("true", id="bool"), pytest.param("'1'", id="non-real")),
)
def test_retargeting_loader_requires_real_mapping_values(
    tmp_path: Path,
    setting_name: str,
    yaml_value: str,
) -> None:
    path = _yaml(
        tmp_path,
        "retargeting:\n"
        f"  {setting_name}:\n"
        f"    shoulder_rh: {yaml_value}\n",
    )

    with pytest.raises(
        ValueError,
        match=rf"retargeting\.{setting_name}\.shoulder_rh.*real number",
    ):
        load_retargeting_config(path)


@pytest.mark.parametrize(
    ("section", "bad_key", "loader"),
    (
        (
            "standing_balance",
            "pitch_feedback_gian",
            load_standing_balance_config,
        ),
        (
            "support_control",
            "touchdown_timout_s",
            load_support_control_config,
        ),
        (
            "human_support_intent",
            "activate_heigth_ratio",
            load_human_support_intent_config,
        ),
    ),
)
def test_every_balance_section_rejects_unknown_keys_with_section_and_key(
    tmp_path: Path,
    section: str,
    bad_key: str,
    loader: Callable[[str | Path | None], object],
) -> None:
    path = _yaml(tmp_path, f"{section}:\n  {bad_key}: 1.0\n")

    with pytest.raises(ValueError, match=rf"{section}.*{bad_key}"):
        loader(path)


@pytest.mark.parametrize(
    "loader",
    (
        load_standing_balance_config,
        load_support_control_config,
        load_human_support_intent_config,
    ),
)
def test_balance_loaders_reject_unknown_top_level_sections(
    tmp_path: Path,
    loader: Callable[[str | Path | None], object],
) -> None:
    path = _yaml(tmp_path, "standing_balnce:\n  enabled: true\n")

    with pytest.raises(ValueError, match="standing_balnce"):
        loader(path)


def test_retargeting_loader_rejects_unknown_setting(tmp_path: Path) -> None:
    path = _yaml(
        tmp_path,
        "retargeting:\n  auto_calibration_frames: 30\n"
        "  smoothing_time_constant: 0.1\n",
    )

    with pytest.raises(
        ValueError,
        match=r"retargeting settings.*smoothing_time_constant",
    ):
        load_retargeting_config(path)


def test_retargeting_loader_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    path = _yaml(
        tmp_path,
        "retargeting:\n  mode: whole_body\nretargting_metadata: {}\n",
    )

    with pytest.raises(ValueError, match="retargting_metadata"):
        load_retargeting_config(path)


def _moving_frame(sequence: int) -> SkeletonFrame:
    phase = 0.0 if sequence % 2 == 0 else 1.0
    return make_synthetic_skeleton(
        sequence / 30.0,
        phase_rad=phase,
        sequence=sequence,
    )


def _level_but_occluded_frame(sequence: int) -> SkeletonFrame:
    frame = make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
    visibility = frame.visibility.copy()
    presence = frame.presence.copy()
    visibility[int(L.LEFT_WRIST)] = 0.0
    presence[int(L.LEFT_WRIST)] = 0.0
    return replace(frame, visibility=visibility, presence=presence)


def _gate_invalid_owner_valid(
    frame: SkeletonFrame, *, sequence: int
) -> SkeletonFrame:
    visibility = frame.visibility.copy()
    presence = frame.presence.copy()
    # Retargeter coverage remains above 75%, and support-intent measurement
    # does not consume wrists. The stricter neutral posture gate does.
    visibility[int(L.LEFT_WRIST)] = 0.0
    presence[int(L.LEFT_WRIST)] = 0.0
    return replace(
        frame,
        timestamp_s=sequence / 30.0,
        sequence=sequence,
        visibility=visibility,
        presence=presence,
    )


def _one_leg_image_frame(sequence: int) -> SkeletonFrame:
    frame = make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
    points = frame.landmarks_2d.copy()
    # A static lifted ankle is still, but not a valid double-support neutral.
    points[int(L.RIGHT_ANKLE), 1] -= 0.22
    points[int(L.RIGHT_HEEL), 1] -= 0.22
    points[int(L.RIGHT_FOOT_INDEX), 1] -= 0.22
    return replace(frame, landmarks_2d=points)


def _moving_support_frame(sequence: int) -> SkeletonFrame:
    frame = make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
    points_2d = frame.landmarks_2d.copy()
    points_3d = frame.landmarks_3d.copy()
    side = (
        (L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX)
        if sequence % 2
        else (L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_HEEL, L.LEFT_FOOT_INDEX)
    )
    for landmark in side:
        weight = 0.5 if landmark in (L.LEFT_KNEE, L.RIGHT_KNEE) else 1.0
        points_2d[int(landmark), 1] -= 0.20 * weight
        points_3d[int(landmark), 1] -= 0.20 * weight
    return replace(frame, landmarks_2d=points_2d, landmarks_3d=points_3d)


def _arms_up_frame(sequence: int) -> SkeletonFrame:
    frame = make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
    points = frame.landmarks_3d.copy()
    for left in (False, True):
        shoulder = L.LEFT_SHOULDER if left else L.RIGHT_SHOULDER
        elbow = L.LEFT_ELBOW if left else L.RIGHT_ELBOW
        wrist = L.LEFT_WRIST if left else L.RIGHT_WRIST
        points[int(elbow)] = points[int(shoulder)] + np.asarray((0.0, -0.25, 0.0))
        points[int(wrist)] = points[int(shoulder)] + np.asarray((0.0, -0.50, 0.0))
    return replace(frame, landmarks_3d=points)


def _deep_squat_frame(sequence: int) -> SkeletonFrame:
    frame = make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
    points = frame.landmarks_3d.copy()
    for left in (False, True):
        hip = L.LEFT_HIP if left else L.RIGHT_HIP
        knee = L.LEFT_KNEE if left else L.RIGHT_KNEE
        ankle = L.LEFT_ANKLE if left else L.RIGHT_ANKLE
        points[int(knee)] = 0.5 * (
            points[int(hip)] + points[int(ankle)]
        ) + np.asarray((0.0, 0.0, 0.35))
    return replace(frame, landmarks_3d=points)


def _folded_arm_frame(sequence: int) -> SkeletonFrame:
    frame = make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
    points = frame.landmarks_3d.copy()
    # Both upper arms point across the torso, while wrists remain below their
    # shoulders. A shoulder->wrist-only gate could mistake this for arms-down.
    for left in (False, True):
        shoulder = L.LEFT_SHOULDER if left else L.RIGHT_SHOULDER
        elbow = L.LEFT_ELBOW if left else L.RIGHT_ELBOW
        wrist = L.LEFT_WRIST if left else L.RIGHT_WRIST
        lateral = -0.28 if left else 0.28
        points[int(elbow)] = points[int(shoulder)] + np.asarray((lateral, 0.0, 0.0))
        points[int(wrist)] = points[int(shoulder)] + np.asarray((0.0, 0.30, 0.0))
    return replace(frame, landmarks_3d=points)


def _camera_tilted_neutral(sequence: int, angle_rad: float = 0.55) -> SkeletonFrame:
    frame = make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    rotation_2d = np.asarray(((cosine, -sine), (sine, cosine)))
    rotation_3d = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))
    )
    hip_2d = 0.5 * (
        frame.landmarks_2d[int(L.LEFT_HIP)]
        + frame.landmarks_2d[int(L.RIGHT_HIP)]
    )
    hip_3d = 0.5 * (
        frame.landmarks_3d[int(L.LEFT_HIP)]
        + frame.landmarks_3d[int(L.RIGHT_HIP)]
    )
    points_2d = (frame.landmarks_2d - hip_2d) @ rotation_2d.T + hip_2d
    points_3d = (frame.landmarks_3d - hip_3d) @ rotation_3d.T + hip_3d
    return replace(frame, landmarks_2d=points_2d, landmarks_3d=points_3d)


def _strict_retargeting_config(*, samples: int, maximum: int) -> RetargetingConfig:
    return RetargetingConfig(
        auto_calibration_frames=samples,
        calibration_max_observations=maximum,
        calibration_max_pose_spread_ratio=0.02,
        calibration_max_ankle_offset_ratio=0.08,
        calibration_max_ankle_spread_ratio=0.02,
        smoothing_time_constant_s=0.0,
    )


def test_moving_first_frames_are_not_installed_as_geometric_neutral() -> None:
    retargeter = GeometricRetargeter(
        config=_strict_retargeting_config(samples=4, maximum=8)
    )
    home = retargeter.neutral_positions_rad

    for sequence in range(7):
        command = retargeter.retarget(_moving_frame(sequence))
        np.testing.assert_allclose(command.positions_rad, home)
        assert retargeter.is_calibrating
    with pytest.raises(
        NeutralCalibrationError,
        match=r"failed after 8 confident observations.*pose spread",
    ):
        retargeter.retarget(_moving_frame(7))

    # A failed automatic attempt cannot silently resume. The explicit
    # recalibration operation clears the timeout and admits a truly still hold.
    retargeter.start_calibration(4)
    for sequence in range(8, 12):
        retargeter.retarget(
            make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
        )
    assert not retargeter.is_calibrating


def test_mujoco_ik_also_rejects_a_moving_first_window() -> None:
    retargeter = MujocoIKRetargeter(
        config=_strict_retargeting_config(samples=3, maximum=5)
    )

    for sequence in range(4):
        retargeter.retarget(_moving_frame(sequence))
        assert retargeter.is_calibrating
    with pytest.raises(NeutralCalibrationError, match="MuJoCo IK.*failed after 5"):
        retargeter.retarget(_moving_frame(4))

    retargeter.start_calibration(3)
    for sequence in range(5, 8):
        retargeter.retarget(
            make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
        )
    assert not retargeter.is_calibrating


def test_sliding_gate_can_recover_from_motion_before_timeout() -> None:
    retargeter = GeometricRetargeter(
        config=_strict_retargeting_config(samples=4, maximum=10)
    )
    for sequence in range(3):
        retargeter.retarget(_moving_frame(sequence))
    for sequence in range(3, 7):
        retargeter.retarget(
            make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
        )

    assert not retargeter.is_calibrating


@pytest.mark.parametrize(
    "factory",
    (
        lambda config: GeometricRetargeter(config=config),
        lambda config: MujocoIKRetargeter(config=config),
    ),
    ids=("geometric", "mujoco_ik"),
)
def test_gate_invalid_owner_valid_frames_cannot_poison_retargeting_reference(
    factory: Callable[[RetargetingConfig], object],
) -> None:
    retargeter = factory(_strict_retargeting_config(samples=3, maximum=8))
    home = retargeter.neutral_positions_rad
    sequence = 0
    for _ in range(2):
        retargeter.retarget(
            make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
        )
        sequence += 1
    for _ in range(3):
        poison = _gate_invalid_owner_valid(
            _deep_squat_frame(sequence), sequence=sequence
        )
        command = retargeter.retarget(poison)
        np.testing.assert_allclose(command.positions_rad, home)
        sequence += 1
    retargeter.retarget(
        make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
    )
    assert not retargeter.is_calibrating

    command = retargeter.retarget(
        make_synthetic_skeleton((sequence + 1) / 30.0, sequence=sequence + 1)
    )
    np.testing.assert_allclose(command.positions_rad, home, atol=1e-7)


def test_occluded_frames_do_not_consume_confident_observation_timeout() -> None:
    retargeter = GeometricRetargeter(
        config=_strict_retargeting_config(samples=4, maximum=5)
    )

    for sequence in range(20):
        retargeter.retarget(_level_but_occluded_frame(sequence))
    assert retargeter.is_calibrating
    assert retargeter.calibration_progress == 0.0

    for sequence in range(20, 24):
        retargeter.retarget(
            make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
        )
    assert not retargeter.is_calibrating


@pytest.mark.parametrize(
    "factory",
    (
        lambda config: GeometricRetargeter(config=config),
        lambda config: MujocoIKRetargeter(config=config),
    ),
    ids=("geometric", "mujoco_ik"),
)
def test_static_one_leg_pose_is_not_a_neutral_reference(
    factory: Callable[[RetargetingConfig], object],
) -> None:
    retargeter = factory(_strict_retargeting_config(samples=3, maximum=4))
    for sequence in range(3):
        retargeter.retarget(_one_leg_image_frame(sequence))
        assert retargeter.is_calibrating
    with pytest.raises(
        NeutralCalibrationError,
        match=r"failed after 4 confident observations.*ankle offset",
    ):
        retargeter.retarget(_one_leg_image_frame(3))


@pytest.mark.parametrize(
    ("frame_factory", "metric"),
    (
        (_arms_up_frame, "arm deviation"),
        (_folded_arm_frame, "upper-arm deviation"),
        (_deep_squat_frame, "knee flexion"),
    ),
    ids=("arms_up", "folded_arms", "deep_squat"),
)
def test_static_non_neutral_posture_cannot_become_ik_reference(
    frame_factory: Callable[[int], SkeletonFrame],
    metric: str,
) -> None:
    retargeter = MujocoIKRetargeter(
        config=_strict_retargeting_config(samples=3, maximum=4)
    )
    for sequence in range(3):
        retargeter.retarget(frame_factory(sequence))
        assert retargeter.is_calibrating
    with pytest.raises(
        NeutralCalibrationError,
        match=rf"failed after 4 confident observations.*{metric}",
    ):
        retargeter.retarget(frame_factory(3))


def test_anatomical_posture_gate_is_invariant_to_camera_tilt() -> None:
    retargeter = MujocoIKRetargeter(
        config=_strict_retargeting_config(samples=3, maximum=4)
    )
    for sequence in range(3):
        retargeter.retarget(_camera_tilted_neutral(sequence))

    assert not retargeter.is_calibrating


@pytest.mark.parametrize(
    "frame_factory",
    (_arms_up_frame, _folded_arm_frame, _deep_squat_frame),
    ids=("arms_up", "folded_arms", "deep_squat"),
)
def test_support_intent_cannot_calibrate_on_non_neutral_posture(
    frame_factory: Callable[[int], SkeletonFrame],
) -> None:
    estimator = HumanSupportIntentEstimator(
        HumanSupportIntentConfig(
            calibration_frames=3,
            calibration_max_observations=4,
        )
    )
    for sequence in range(3):
        estimate = estimator.update(frame_factory(sequence))
        assert not estimate.calibrated
    with pytest.raises(NeutralCalibrationError, match="support-intent.*failed after 4"):
        estimator.update(frame_factory(3))


@pytest.mark.parametrize(
    "factory",
    (
        lambda config: GeometricRetargeter(config=config),
        lambda config: MujocoIKRetargeter(config=config),
    ),
    ids=("geometric", "mujoco_ik"),
)
def test_explicit_retargeter_calibration_keeps_posture_and_support_gates(
    factory: Callable[[RetargetingConfig], object],
) -> None:
    retargeter = factory(_strict_retargeting_config(samples=3, maximum=4))

    assert not retargeter.calibrate(_arms_up_frame(0))
    assert not retargeter.calibrate(_folded_arm_frame(0))
    assert not retargeter.calibrate(_deep_squat_frame(0))
    assert not retargeter.calibrate(_one_leg_image_frame(0))
    assert retargeter.calibrate(_camera_tilted_neutral(0))


def test_explicit_support_intent_calibration_uses_the_same_neutral_gate() -> None:
    estimator = HumanSupportIntentEstimator()

    assert not estimator.calibrate(_arms_up_frame(0))
    assert not estimator.calibrate(_folded_arm_frame(0))
    assert not estimator.calibrate(_deep_squat_frame(0))
    assert not estimator.calibrate(_one_leg_image_frame(0))
    assert estimator.calibrate(_camera_tilted_neutral(0))
    assert not estimator.is_calibrating


def test_support_intent_uses_the_same_stillness_timeout_and_explicit_reset() -> None:
    estimator = HumanSupportIntentEstimator(
        HumanSupportIntentConfig(
            calibration_frames=3,
            calibration_max_observations=5,
            calibration_max_pose_spread_ratio=0.02,
            calibration_max_ankle_offset_ratio=0.08,
            calibration_max_ankle_spread_ratio=0.02,
        )
    )

    for sequence in range(4):
        estimate = estimator.update(_moving_support_frame(sequence))
        assert not estimate.calibrated
    with pytest.raises(NeutralCalibrationError, match="support-intent.*failed after 5"):
        estimator.update(_moving_support_frame(4))

    estimator.start_calibration()
    for sequence in range(5, 8):
        estimate = estimator.update(
            make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
        )
    assert estimate.calibrated
    assert not estimator.is_calibrating


def test_gate_invalid_owner_valid_frames_cannot_poison_support_baseline() -> None:
    estimator = HumanSupportIntentEstimator(
        HumanSupportIntentConfig(
            calibration_frames=3,
            calibration_max_observations=8,
        )
    )
    sequence = 0
    for _ in range(2):
        estimator.update(
            make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
        )
        sequence += 1
    for _ in range(3):
        poison = _gate_invalid_owner_valid(
            _moving_support_frame(1), sequence=sequence
        )
        estimate = estimator.update(poison)
        assert not estimate.calibrated
        sequence += 1
    estimate = estimator.update(
        make_synthetic_skeleton(sequence / 30.0, sequence=sequence)
    )

    assert estimate.calibrated
    assert not estimator.is_calibrating
    assert estimator._baseline_ratio == pytest.approx(0.0, abs=1e-12)


def test_repository_yaml_files_pass_their_strict_schemas() -> None:
    retargeting = load_retargeting_config("config/retargeting.yaml")
    standing = load_standing_balance_config("config/balance.yaml")
    support = load_support_control_config("config/balance.yaml")
    intent = load_human_support_intent_config("config/balance.yaml")

    assert retargeting.calibration_max_observations == 150
    assert standing.enabled
    assert support.touchdown_timeout_s == 8.0
    assert intent.calibration_max_observations == 150


def test_legacy_configs_without_new_gate_fields_receive_safe_defaults(
    tmp_path: Path,
) -> None:
    retargeting_path = _yaml(
        tmp_path,
        "retargeting:\n  mode: whole_body\n  auto_calibration_frames: 12\n",
    )
    retargeting = load_retargeting_config(retargeting_path)
    assert retargeting.auto_calibration_frames == 12
    assert retargeting.calibration_max_observations == 150
    assert retargeting.calibration_max_arm_deviation_rad == pytest.approx(np.pi / 3.0)

    balance_path = _yaml(
        tmp_path,
        "standing_balance:\n  enabled: true\n"
        "human_support_intent:\n  calibration_frames: 12\n"
        "support_control:\n  shift_duration_s: 2.0\n",
    )
    intent = load_human_support_intent_config(balance_path)
    assert intent.calibration_frames == 12
    assert intent.calibration_max_observations == 150
