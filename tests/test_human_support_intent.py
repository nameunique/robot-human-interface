from __future__ import annotations

from dataclasses import replace

import numpy as np

from robot_human_interface.control import (
    HumanSupportIntentConfig,
    HumanSupportIntentEstimator,
    SupportIntent,
    load_human_support_intent_config,
)
from robot_human_interface.pose import make_synthetic_skeleton
from robot_human_interface.skeleton import PoseLandmark as L, SkeletonFrame


DT = 1.0 / 30.0


def _frame(
    sequence: int,
    *,
    lifted: str | None = None,
    height_ratio: float = 0.0,
    tilted_torso: bool = False,
    confidence: float = 0.99,
) -> SkeletonFrame:
    timestamp_s = sequence * DT
    base = make_synthetic_skeleton(
        timestamp_s,
        sequence=sequence,
        confidence=confidence,
    )
    points = base.landmarks_3d.copy()
    points_2d = base.landmarks_2d.copy()
    if tilted_torso:
        # Deliberately rotate only the torso-up estimate toward camera depth.
        # Equal feet must still read as equal after neutral calibration.
        points[int(L.LEFT_SHOULDER), 2] += 0.28
        points[int(L.RIGHT_SHOULDER), 2] += 0.28
    if lifted is not None:
        landmarks = (
            (L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX)
            if lifted == "right"
            else (L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_HEEL, L.LEFT_FOOT_INDEX)
        )
        hip = L.RIGHT_HIP if lifted == "right" else L.LEFT_HIP
        ankle = L.RIGHT_ANKLE if lifted == "right" else L.LEFT_ANKLE
        leg_length = float(np.linalg.norm(points[int(hip)] - points[int(ankle)]))
        leg_length_2d = float(
            np.linalg.norm(points_2d[int(hip)] - points_2d[int(ankle)])
        )
        camera_up = np.array((0.0, -1.0, 0.0))
        for index in landmarks:
            factor = 0.5 if index in (L.LEFT_KNEE, L.RIGHT_KNEE) else 1.0
            points[int(index)] += factor * height_ratio * leg_length * camera_up
            points_2d[int(index)] += (
                factor * height_ratio * leg_length_2d * camera_up[:2]
            )
    return replace(base, landmarks_2d=points_2d, landmarks_3d=points)


def _calibrated() -> tuple[HumanSupportIntentEstimator, int]:
    config = HumanSupportIntentConfig(
        calibration_frames=10,
        activation_hold_s=0.10,
        release_hold_s=0.10,
        filter_time_constant_s=0.03,
    )
    estimator = HumanSupportIntentEstimator(config)
    for sequence in range(config.calibration_frames):
        estimate = estimator.update(_frame(sequence))
        assert estimate.intent is SupportIntent.DOUBLE_SUPPORT
    assert not estimator.is_calibrating
    return estimator, config.calibration_frames


def _bilateral_frame(sequence: int, *, squat: bool = False) -> SkeletonFrame:
    """Synthetic camera-plane geometry with a controlled bilateral squat."""

    frame = make_synthetic_skeleton(
        sequence * DT,
        sequence=sequence,
        confidence=0.99,
    )
    points = frame.landmarks_2d.copy()
    aspect = (
        1.0
        if frame.image_size is None
        else frame.image_size[0] / frame.image_size[1]
    )

    def point(x: float, y: float) -> np.ndarray:
        return np.asarray((x / aspect, y), dtype=np.float64)

    for index, value in {
        L.LEFT_SHOULDER: point(-0.12, 0.00),
        L.RIGHT_SHOULDER: point(0.12, 0.00),
        L.LEFT_HIP: point(-0.10, 0.50 if squat else 0.30),
        L.RIGHT_HIP: point(0.10, 0.50 if squat else 0.30),
        L.LEFT_KNEE: point(-0.23 if squat else -0.10, 0.67 if squat else 0.60),
        L.RIGHT_KNEE: point(0.23 if squat else 0.10, 0.67 if squat else 0.60),
        L.LEFT_ANKLE: point(-0.10, 0.90),
        L.RIGHT_ANKLE: point(0.10, 0.90),
    }.items():
        points[int(index)] = value
    return replace(frame, landmarks_2d=points)


def _squat_calibrated() -> tuple[HumanSupportIntentEstimator, int]:
    config = HumanSupportIntentConfig(
        calibration_frames=5,
        squat_activation_hold_s=0.20,
        squat_release_hold_s=0.15,
        squat_filter_time_constant_s=0.01,
    )
    estimator = HumanSupportIntentEstimator(config)
    for sequence in range(config.calibration_frames):
        estimate = estimator.update(_bilateral_frame(sequence))
        assert not estimate.squat_active
    assert not estimator.is_calibrating
    return estimator, config.calibration_frames


def test_detects_each_lift_after_neutral_calibration_and_hysteresis() -> None:
    for side, expected in (
        ("right", SupportIntent.RIGHT_SWING),
        ("left", SupportIntent.LEFT_SWING),
    ):
        estimator, start = _calibrated()
        estimate = None
        for sequence in range(start, start + 12):
            estimate = estimator.update(
                _frame(sequence, lifted=side, height_ratio=0.30)
            )
        assert estimate is not None
        assert estimate.intent is expected
        assert estimate.calibrated and not estimate.stale
        assert max(estimate.right_lift_ratio, estimate.left_lift_ratio) > 0.2


def test_fixed_neutral_up_does_not_invert_when_the_torso_leans() -> None:
    estimator, start = _calibrated()
    for sequence in range(start, start + 20):
        estimate = estimator.update(_frame(sequence, tilted_torso=True))

    assert estimate.intent is SupportIntent.DOUBLE_SUPPORT
    assert abs(estimate.signed_height_ratio) < 0.02


def test_fixed_neutral_up_keeps_the_correct_lift_side_during_a_lean() -> None:
    estimator, start = _calibrated()
    for sequence in range(start, start + 15):
        estimate = estimator.update(
            _frame(
                sequence,
                lifted="right",
                height_ratio=0.22,
                tilted_torso=True,
            )
        )

    assert estimate.intent is SupportIntent.RIGHT_SWING


def test_low_confidence_gap_returns_to_double_support() -> None:
    estimator, start = _calibrated()
    for sequence in range(start, start + 12):
        estimate = estimator.update(
            _frame(sequence, lifted="right", height_ratio=0.30)
        )
    assert estimate.intent is SupportIntent.RIGHT_SWING

    last = start + 12
    estimate = estimator.update(None, timestamp_s=(last * DT) + 0.5)

    assert estimate.intent is SupportIntent.DOUBLE_SUPPORT
    assert estimate.stale


def test_manual_recalibration_clears_a_previous_lift_request() -> None:
    estimator, start = _calibrated()
    for sequence in range(start, start + 12):
        estimator.update(_frame(sequence, lifted="left", height_ratio=0.30))
    assert estimator.intent is SupportIntent.LEFT_SWING

    estimator.start_calibration()

    assert estimator.is_calibrating
    assert estimator.intent is SupportIntent.DOUBLE_SUPPORT
    assert estimator.calibration_progress == 0.0


def test_opposite_lift_must_pass_through_confirmed_double_support() -> None:
    estimator, sequence = _calibrated()
    for sequence in range(sequence, sequence + 15):
        estimate = estimator.update(
            _frame(sequence, lifted="right", height_ratio=0.30)
        )
    assert estimate.intent is SupportIntent.RIGHT_SWING

    observed: list[SupportIntent] = []
    for sequence in range(sequence + 1, sequence + 40):
        estimate = estimator.update(
            _frame(sequence, lifted="left", height_ratio=0.30)
        )
        observed.append(estimate.intent)
        if estimate.intent is SupportIntent.LEFT_SWING:
            break

    assert SupportIntent.DOUBLE_SUPPORT in observed
    assert observed.index(SupportIntent.DOUBLE_SUPPORT) < observed.index(
        SupportIntent.LEFT_SWING
    )


def test_stale_reset_cannot_reactivate_the_old_side_on_neutral_frames() -> None:
    estimator, sequence = _calibrated()
    for sequence in range(sequence, sequence + 15):
        estimator.update(_frame(sequence, lifted="right", height_ratio=0.30))
    assert estimator.intent is SupportIntent.RIGHT_SWING

    stale_time = (sequence * DT) + 0.5
    stale = estimator.update(None, timestamp_s=stale_time)
    assert stale.stale and stale.intent is SupportIntent.DOUBLE_SUPPORT

    for offset in range(1, 15):
        estimate = estimator.update(
            _frame(int(round(stale_time / DT)) + offset)
        )
    assert estimate.intent is SupportIntent.DOUBLE_SUPPORT


def test_low_ankle_confidence_cannot_start_a_single_support_transition() -> None:
    estimator, start = _calibrated()
    for sequence in range(start, start + 20):
        frame = _frame(sequence, lifted="right", height_ratio=0.30)
        visibility = frame.visibility.copy()
        visibility[int(L.RIGHT_ANKLE)] = 0.60
        estimate = estimator.update(replace(frame, visibility=visibility))

    assert estimate.intent is SupportIntent.DOUBLE_SUPPORT
    assert estimate.confidence == 0.60


def test_balance_yaml_is_the_runtime_source_for_intent_thresholds() -> None:
    config = load_human_support_intent_config("config/balance.yaml")
    assert config.calibration_frames == 30
    assert config.activate_height_ratio == 0.15
    assert config.release_height_ratio == 0.08
    assert config.squat_enter_pelvis_descent_ratio == 0.10
    assert np.degrees(config.squat_enter_hip_flexion_rad) == 20.0
    assert np.degrees(config.squat_enter_knee_flexion_rad) == 25.0


def test_default_squat_latency_keeps_verified_filter_and_dwell() -> None:
    config = HumanSupportIntentConfig()
    assert config.squat_activation_hold_s == 0.05
    assert config.squat_filter_time_constant_s == 0.05


def test_bilateral_squat_activates_only_after_dwell_and_reports_depth() -> None:
    estimator, start = _squat_calibrated()
    estimates = []
    for sequence in range(start, start + 12):
        estimates.append(estimator.update(_bilateral_frame(sequence, squat=True)))

    assert not any(item.squat_active for item in estimates[:6])
    assert estimates[-1].squat_active
    assert estimates[-1].intent is SupportIntent.DOUBLE_SUPPORT
    assert estimates[-1].squat_depth_ratio > 0.4
    assert estimates[-1].squat_bilateral_hip_flexion_rad > np.radians(20.0)
    assert estimates[-1].squat_bilateral_knee_flexion_rad > np.radians(25.0)


def test_squat_geometry_uses_fixed_neutral_leg_scale_and_exact_depth_map() -> None:
    estimator, start = _squat_calibrated()
    estimate = None
    for sequence in range(start, start + 20):
        estimate = estimator.update(_bilateral_frame(sequence, squat=True))

    assert estimate is not None and estimate.squat_active
    # Neutral hip-to-ankle height and bilateral leg-chain scale are both 0.6.
    # The squat lowers the hip by 0.2, so calibrated descent is exactly 1/3.
    expected_descent = 1.0 / 3.0
    expected_depth = (
        expected_descent - estimator.config.squat_exit_pelvis_descent_ratio
    ) / (
        estimator.config.squat_full_pelvis_descent_ratio
        - estimator.config.squat_exit_pelvis_descent_ratio
    )
    assert np.isclose(
        estimate.squat_pelvis_descent_ratio,
        expected_descent,
        atol=1e-9,
    )
    assert np.isclose(estimate.squat_depth_ratio, expected_depth, atol=1e-9)


def test_explicit_squat_calibration_requires_every_landmark_at_activation_confidence() -> None:
    estimator = HumanSupportIntentEstimator(
        HumanSupportIntentConfig(calibration_frames=1)
    )
    frame = _bilateral_frame(0)
    visibility = frame.visibility.copy()
    visibility[int(L.LEFT_KNEE)] = 0.60
    weak_knee = replace(frame, visibility=visibility)

    # The support detector's lower quantile remains high because seven of the
    # eight points are strong. Squat calibration is intentionally stricter.
    measurement = estimator._measurement(weak_knee)
    assert measurement is not None and measurement[3] > 0.65
    assert not estimator.calibrate(weak_knee)
    assert estimator.is_calibrating


def test_automatic_squat_calibration_ignores_a_single_weak_landmark() -> None:
    config = HumanSupportIntentConfig(
        calibration_frames=3,
        calibration_max_observations=10,
    )
    estimator = HumanSupportIntentEstimator(config)
    for sequence in range(3):
        frame = _bilateral_frame(sequence)
        visibility = frame.visibility.copy()
        visibility[int(L.RIGHT_KNEE)] = 0.60
        estimate = estimator.update(replace(frame, visibility=visibility))
        assert not estimate.calibrated
    assert estimator.is_calibrating
    assert estimator.calibration_progress == 0.0

    for sequence in range(3, 6):
        estimate = estimator.update(_bilateral_frame(sequence))
    assert estimate.calibrated
    assert not estimator.is_calibrating


def test_unilateral_lift_never_authorizes_bilateral_squat() -> None:
    estimator, start = _squat_calibrated()
    for sequence in range(start, start + 15):
        estimate = estimator.update(
            _frame(sequence, lifted="right", height_ratio=0.35)
        )

    assert not estimate.squat_active
    assert estimate.squat_depth_ratio == 0.0


def test_squat_release_is_debounced_and_stale_input_clears_request() -> None:
    estimator, sequence = _squat_calibrated()
    for sequence in range(sequence, sequence + 12):
        estimate = estimator.update(_bilateral_frame(sequence, squat=True))
    assert estimate.squat_active

    first_neutral = estimator.update(_bilateral_frame(sequence + 1))
    assert first_neutral.squat_active
    for sequence in range(sequence + 2, sequence + 10):
        estimate = estimator.update(_bilateral_frame(sequence))
    assert not estimate.squat_active

    for sequence in range(sequence + 1, sequence + 13):
        estimate = estimator.update(_bilateral_frame(sequence, squat=True))
    assert estimate.squat_active
    stale = estimator.update(None, timestamp_s=estimate.confidence + sequence * DT + 1.0)
    assert stale.stale
    assert not stale.squat_active
    assert stale.squat_depth_ratio == 0.0


def test_nonstale_gap_freezes_active_squat_but_resets_inactive_entry_dwell() -> None:
    estimator, sequence = _squat_calibrated()
    for sequence in range(sequence, sequence + 4):
        estimate = estimator.update(_bilateral_frame(sequence, squat=True))
    assert not estimate.squat_active

    gap_sequence = sequence + 1
    gap = estimator.update(None, timestamp_s=gap_sequence * DT)
    assert not gap.stale and not gap.squat_active
    assert not gap.squat_observation_fresh

    for sequence in range(gap_sequence + 1, gap_sequence + 6):
        estimate = estimator.update(_bilateral_frame(sequence, squat=True))
    assert not estimate.squat_active
    for sequence in range(sequence + 1, sequence + 5):
        estimate = estimator.update(_bilateral_frame(sequence, squat=True))
    assert estimate.squat_active

    held_depth = estimate.squat_depth_ratio
    short_gap = estimator.update(None, timestamp_s=(sequence * DT) + 0.10)
    assert not short_gap.stale and short_gap.squat_active
    assert not short_gap.squat_observation_fresh
    assert short_gap.squat_depth_ratio == held_depth
