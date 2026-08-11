"""Benchmark geometric and MuJoCo-IK retargeting on replay videos.

The benchmark samples video frames for speed but keeps the initial calibration
window dense.  It compares FK directions in a common anatomical
``(forward, right, up)`` basis; no balance or support controller is involved.
This isolates retargeting quality from the later stability projection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from robot_human_interface.camera import OpenCVVideoSource
from robot_human_interface.pose import MediaPipePoseConfig, MediaPipePoseLandmarker
from robot_human_interface.retargeting import (
    ARM_DIRECTION_NAMES,
    END_EFFECTOR_DIRECTION_NAMES,
    LEG_DIRECTION_NAMES,
    GeometricRetargeter,
    MujocoIKRetargeter,
    MujocoPoseFidelityEvaluator,
    load_joint_specs,
    load_retargeting_config,
)
from robot_human_interface.skeleton import SkeletonEMAFilter, SkeletonFilterConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOINT_CONFIG = PROJECT_ROOT / "config" / "joints.yaml"
RETARGETING_CONFIG = PROJECT_ROOT / "config" / "retargeting.yaml"
POSE_MODEL = PROJECT_ROOT / "assets" / "models" / "pose_landmarker_full.task"
MUJOCO_MODEL = PROJECT_ROOT / "models" / "humanoid" / "scene_fixed.xml"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "pose-fidelity.json"

GROUPS: dict[str, tuple[str, ...]] = {
    "arms": ARM_DIRECTION_NAMES,
    "legs": LEG_DIRECTION_NAMES,
    "end_effectors": END_EFFECTOR_DIRECTION_NAMES,
    "head": ("head",),
    "right_arm": ("right_arm",),
    "left_arm": ("left_arm",),
    "right_leg": ("right_leg",),
    "left_leg": ("left_leg",),
}


def _default_videos() -> list[Path]:
    videos = [PROJECT_ROOT / "assets" / "videos" / "slow_balance_demo.mp4"]
    videos.extend(sorted((PROJECT_ROOT / "assets" / "videos" / "external").glob("*.mp4")))
    return videos


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean_deg": None, "p50_deg": None, "p90_deg": None}
    return {
        "count": int(finite.size),
        "mean_deg": float(np.mean(finite)),
        "p50_deg": float(np.percentile(finite, 50.0)),
        "p90_deg": float(np.percentile(finite, 90.0)),
    }


def _paired_improvement(before: Sequence[float], after: Sequence[float]) -> float | None:
    baseline = np.asarray(before, dtype=np.float64)
    candidate = np.asarray(after, dtype=np.float64)
    valid = np.isfinite(baseline) & np.isfinite(candidate)
    if not np.any(valid):
        return None
    baseline_mean = float(np.mean(baseline[valid]))
    if baseline_mean <= 1e-12:
        return None
    return float(100.0 * (1.0 - np.mean(candidate[valid]) / baseline_mean))


def evaluate_video(path: Path, *, stride: int) -> dict[str, object]:
    """Evaluate one video and return JSON-serializable evidence."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"video does not exist: {path}")
    config = load_retargeting_config(RETARGETING_CONFIG)
    specs = load_joint_specs(JOINT_CONFIG)
    geometric = GeometricRetargeter(specs, config)
    ik = MujocoIKRetargeter(MUJOCO_MODEL, specs, config)
    evaluator = MujocoPoseFidelityEvaluator(ik.model)
    source = OpenCVVideoSource(path, realtime=False)
    pose = MediaPipePoseLandmarker(
        MediaPipePoseConfig(model_asset_path=POSE_MODEL),
        landmark_filter=SkeletonEMAFilter(
            SkeletonFilterConfig(
                time_constant_s=0.08,
                confidence_threshold=0.5,
                max_gap_s=0.25,
            )
        ),
    )
    values: dict[str, dict[str, list[float]]] = {
        method: {group: [] for group in GROUPS}
        for method in ("geometric", "ik")
    }
    stale = {"geometric": 0, "ik": 0}
    source_frames = 0
    sampled_frames = 0
    detected_frames = 0
    evaluated_frames = 0
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            sequence = source_frames
            source_frames += 1
            # Keep the configured calibration window dense.  Sampling starts
            # only after both retargeters have observed the same first frames.
            use_frame = sequence < config.auto_calibration_frames or sequence % stride == 0
            if not use_frame:
                continue
            sampled_frames += 1
            skeleton = pose.estimate(frame)
            detected_frames += int(skeleton is not None)
            commands = {
                "geometric": geometric.retarget(skeleton, timestamp_s=frame.timestamp_s),
                "ik": ik.retarget(skeleton, timestamp_s=frame.timestamp_s),
            }
            if geometric.is_calibrating or ik.is_calibrating:
                continue
            evaluated_frames += 1
            for method, command in commands.items():
                stale[method] += int(command.stale)
                fidelity = (
                    None
                    if skeleton is None
                    else evaluator.evaluate(skeleton, command.positions_rad)
                )
                for group, direction_names in GROUPS.items():
                    values[method][group].append(
                        float("nan")
                        if fidelity is None
                        else fidelity.mean_error_deg(direction_names)
                    )
    finally:
        source.close()
        pose.close()

    metrics: dict[str, object] = {}
    for group in GROUPS:
        before = values["geometric"][group]
        after = values["ik"][group]
        metrics[group] = {
            "geometric": _summary(before),
            "ik": _summary(after),
            "ik_mean_improvement_percent": _paired_improvement(before, after),
        }
    return {
        "video": _project_path(path),
        "source_frames": source_frames,
        "sampled_frames": sampled_frames,
        "detected_frames": detected_frames,
        "evaluated_frames": evaluated_frames,
        "stale_commands": stale,
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=[],
        help="MP4 to evaluate; repeat for multiple files (default: bundled + assets/videos/external).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Evaluate every Nth source frame after the dense calibration window (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path (default: artifacts/pose-fidelity.json).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    videos = [path.expanduser() for path in args.video] if args.video else _default_videos()
    if not videos:
        raise FileNotFoundError("no videos selected for pose-fidelity evaluation")
    missing = [path for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"selected videos do not exist: {missing}")

    retargeting_config = load_retargeting_config(RETARGETING_CONFIG)
    report = {
        "schema_version": 1,
        "description": (
            "Raw retargeter FK direction error before standing-balance and support projection; "
            "angles use anatomical forward/right/up coordinates."
        ),
        "command": f"python tools/evaluate_pose_fidelity.py --stride {args.stride}",
        "configuration": {
            "stride": args.stride,
            "dense_calibration_frames": retargeting_config.auto_calibration_frames,
            "retargeting": asdict(retargeting_config),
            "skeleton_filter": {
                "time_constant_s": 0.08,
                "confidence_threshold": 0.5,
                "max_gap_s": 0.25,
            },
            "pose_model": _project_path(POSE_MODEL),
            "mujoco_model": _project_path(MUJOCO_MODEL),
            "versions": {
                "mediapipe": version("mediapipe"),
                "mujoco": version("mujoco"),
                "numpy": version("numpy"),
            },
        },
        "videos": [evaluate_video(path, stride=args.stride) for path in videos],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"pose-fidelity report: {output}")
    for video in report["videos"]:
        assert isinstance(video, dict)
        metrics = video["metrics"]
        assert isinstance(metrics, dict)
        arms = metrics["arms"]
        legs = metrics["legs"]
        end_effectors = metrics["end_effectors"]
        assert isinstance(arms, dict) and isinstance(legs, dict) and isinstance(end_effectors, dict)
        print(
            f"{video['video']}: IK mean improvement "
            f"arms={arms['ik_mean_improvement_percent']:+.1f}% "
            f"legs={legs['ik_mean_improvement_percent']:+.1f}% "
            f"end={end_effectors['ik_mean_improvement_percent']:+.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
