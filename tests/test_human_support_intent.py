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
        camera_up = np.array((0.0, -1.0, 0.0))
        for index in landmarks:
            factor = 0.5 if index in (L.LEFT_KNEE, L.RIGHT_KNEE) else 1.0
            points[int(index)] += factor * height_ratio * leg_length * camera_up
    return replace(base, landmarks_3d=points)


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
