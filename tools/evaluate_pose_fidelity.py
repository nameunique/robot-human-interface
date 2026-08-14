"""Benchmark geometric and MuJoCo-IK retargeting on replay videos.

The benchmark samples video frames for speed but keeps the initial calibration
window dense.  It compares FK directions in a common anatomical
``(forward, right, up)`` basis; no balance or support controller is involved.
This isolates retargeting quality from the later stability projection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Mapping, Sequence

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
    RetargetingConfig,
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

# The raw diagnostic must identify every repository input that can change the
# generated measurements.  Video inputs and explicit calibration videos are
# added per invocation by ``_runtime_input_paths`` below.
RUNTIME_INPUTS = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "config" / "joints.yaml",
    PROJECT_ROOT / "config" / "retargeting.yaml",
    PROJECT_ROOT / "assets" / "models" / "pose_landmarker_full.task",
    PROJECT_ROOT / "models" / "humanoid" / "scene_fixed.xml",
    PROJECT_ROOT / "models" / "humanoid" / "robot.xml",
    PROJECT_ROOT / "src" / "robot_human_interface" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "camera" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "camera" / "sources.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "pose" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "pose" / "calibration.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "pose" / "mediapipe_tasks.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "pose" / "overlay.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "pose" / "synthetic.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "retargeting" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "retargeting" / "fidelity.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "retargeting" / "geometry.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "retargeting" / "mujoco_ik.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "retargeting" / "retargeter.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "filtering.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "transforms.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "types.py",
)

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
PRIMARY_GROUPS = ("arms", "legs", "end_effectors", "head")
MINIMUM_DETECTED_SAMPLE_FRACTION = 0.70
MINIMUM_EVALUATED_SAMPLE_FRACTION = 0.70
MINIMUM_VALID_EVALUATED_FRACTION = 0.70


@dataclass(frozen=True, slots=True)
class ClipSpec:
    """One raw-replay benchmark input and its neutral-reference contract."""

    name: str
    path: Path
    expected_frames: int | None = None
    calibration_video: Path | None = None
    calibration_frame: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not self.name.strip():
            raise ValueError("clip name must be non-empty")
        if self.expected_frames is not None and (
            isinstance(self.expected_frames, bool)
            or not isinstance(self.expected_frames, int)
            or self.expected_frames <= 0
        ):
            raise ValueError("expected_frames must be a positive integer")
        if (self.calibration_video is None) != (self.calibration_frame is None):
            raise ValueError("calibration_video and calibration_frame must be paired")
        if self.calibration_video is not None:
            object.__setattr__(
                self, "calibration_video", Path(self.calibration_video)
            )
        if self.calibration_frame is not None and (
            isinstance(self.calibration_frame, bool)
            or not isinstance(self.calibration_frame, int)
            or self.calibration_frame < 0
        ):
            raise ValueError("calibration_frame must be a non-negative integer")


def default_clips() -> tuple[ClipSpec, ...]:
    """Return the declared five-video matrix and controlled neutral sources."""

    external = PROJECT_ROOT / "assets" / "videos" / "external"
    neutral = external / "dvids_arm_circles.mp4"
    return (
        ClipSpec(
            "slow-balance",
            PROJECT_ROOT / "assets" / "videos" / "slow_balance_demo.mp4",
            1961,
        ),
        ClipSpec("arm-circles", neutral, 796),
        ClipSpec(
            "frontal-leg-swing",
            external / "dvids_frontal_leg_swing.mp4",
            836,
            calibration_video=neutral,
            calibration_frame=29,
        ),
        ClipSpec(
            "stationary-squat",
            external / "dvids_stationary_squat.mp4",
            817,
        ),
        ClipSpec(
            "trunk-circles",
            external / "dvids_trunk_circles.mp4",
            867,
            calibration_video=neutral,
            calibration_frame=29,
        ),
    )


def _default_videos() -> list[Path]:
    return [clip.path for clip in default_clips()]


def _clip_for_path(path: Path) -> ClipSpec:
    resolved = path.expanduser().resolve()
    for clip in default_clips():
        if clip.path.resolve() == resolved:
            return clip
    return ClipSpec(resolved.stem, resolved)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_input_paths(clips: Sequence[ClipSpec]) -> set[Path]:
    """Return the exact repository and replay inputs used by this run."""

    mesh_inputs = set(
        (PROJECT_ROOT / "models" / "humanoid" / "meshes").glob("*.obj")
    )
    return {
        *(path.expanduser().resolve() for path in RUNTIME_INPUTS),
        *(path.resolve() for path in mesh_inputs),
        *(clip.path.expanduser().resolve() for clip in clips),
        *(
            clip.calibration_video.expanduser().resolve()
            for clip in clips
            if clip.calibration_video is not None
        ),
    }


def _runtime_input_hashes(clips: Sequence[ClipSpec]) -> dict[str, str]:
    paths = _runtime_input_paths(clips)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"raw fidelity runtime inputs are missing: {missing}")
    return {
        _project_path(path): _sha256(path)
        for path in sorted(paths, key=lambda item: str(item).lower())
    }


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _runtime_snapshot_violations(
    snapshot: Mapping[str, object], clips: Sequence[ClipSpec]
) -> dict[str, list[str]]:
    expected_keys = {
        _project_path(path) for path in _runtime_input_paths(clips)
    }
    observed_keys = {str(key) for key in snapshot}
    return {
        "missing_keys": sorted(expected_keys - observed_keys),
        "unexpected_keys": sorted(observed_keys - expected_keys),
        "invalid_sha256_keys": sorted(
            str(key)
            for key, value in snapshot.items()
            if not _valid_sha256(value)
        ),
    }


def _git_output(*arguments: str, text: bool = True) -> str | bytes | None:
    try:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _git_revision() -> str | None:
    output = _git_output("rev-parse", "HEAD")
    if not isinstance(output, str):
        return None
    return output.strip() or None


def _git_worktree_provenance() -> dict[str, object]:
    status_output = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    status = status_output.strip() if isinstance(status_output, str) else None
    diff_output = _git_output("diff", "--binary", "--no-ext-diff", "HEAD", text=False)
    return {
        "dirty": None if status is None else bool(status),
        "status_porcelain": None if status is None else status.splitlines(),
        "tracked_diff_sha256": (
            hashlib.sha256(diff_output).hexdigest()
            if isinstance(diff_output, bytes)
            else None
        ),
        "note": (
            "Runtime input SHA-256 values identify the exact tested contents, "
            "including an uncommitted evaluator; git revision alone is not a "
            "complete identifier when dirty is true."
        ),
    }


def _capture_run_provenance(clips: Sequence[ClipSpec]) -> dict[str, object]:
    """Capture source identities before decoding the first replay frame."""

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "git_worktree": _git_worktree_provenance(),
        "runtime_input_sha256": _runtime_input_hashes(clips),
    }


def _git_provenance_valid(provenance: Mapping[str, object]) -> bool:
    revision = provenance.get("git_revision")
    worktree = provenance.get("git_worktree")
    if not (
        isinstance(revision, str)
        and len(revision) in (40, 64)
        and all(character in "0123456789abcdef" for character in revision.lower())
        and isinstance(worktree, Mapping)
    ):
        return False
    return bool(
        isinstance(worktree.get("dirty"), bool)
        and isinstance(worktree.get("status_porcelain"), list)
        and _valid_sha256(worktree.get("tracked_diff_sha256"))
    )


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


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _controlled_calibration(
    calibration_video: Path,
    calibration_frame: int,
    *,
    pose: MediaPipePoseLandmarker,
    geometric: GeometricRetargeter,
    ik: MujocoIKRetargeter,
) -> dict[str, object]:
    """Install one declared neutral frame without consuming main-source data."""

    path = calibration_video.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"calibration video does not exist: {path}")
    source = OpenCVVideoSource(path, realtime=False)
    selected = None
    try:
        for index in range(calibration_frame + 1):
            frame = source.read()
            if frame is None:
                raise ValueError(
                    f"calibration frame {calibration_frame} is outside video {path}"
                )
            skeleton = pose.estimate(frame)
            if index == calibration_frame:
                selected = skeleton
    finally:
        source.close()
        # Main-video inference needs an independent MediaPipe tracking/filter
        # epoch; calibration frames must not leak into source measurements.
        pose.close()

    geometric_accepted = bool(
        selected is not None and geometric.calibrate(selected)
    )
    ik_accepted = bool(selected is not None and ik.calibrate(selected))
    passed = geometric_accepted and ik_accepted
    failure_reason = None
    if selected is None:
        failure_reason = "selected_calibration_frame_has_no_skeleton"
    elif not geometric_accepted or not ik_accepted:
        failure_reason = "selected_calibration_frame_rejected"
    return {
        "mode": "explicit_replay_frame",
        "source": _project_path(path),
        "source_size_bytes": path.stat().st_size,
        "source_sha256": _sha256(path),
        "frame_index": calibration_frame,
        "geometric_accepted": geometric_accepted,
        "ik_accepted": ik_accepted,
        "status": "passed" if passed else "failed",
        "failure_reason": failure_reason,
    }


def _coverage_assessment(
    *,
    expected_frames: int | None,
    source_frames: int,
    sampled_frames: int,
    detected_frames: int,
    post_calibration_frames: int,
    evaluated_nonstale_frames: int,
    stale_frames: int,
    metrics: dict[str, object],
    calibration_passed: bool,
) -> dict[str, object]:
    """Return explicit fail-closed gates for raw diagnostic completeness."""

    detected_fraction = _fraction(detected_frames, sampled_frames)
    evaluated_fraction = _fraction(evaluated_nonstale_frames, sampled_frames)
    stale_fraction = _fraction(stale_frames, post_calibration_frames)
    group_valid_fractions: dict[str, float] = {}
    for group in PRIMARY_GROUPS:
        group_metrics = metrics[group]
        assert isinstance(group_metrics, dict)
        geometric = group_metrics["geometric"]
        ik = group_metrics["ik"]
        assert isinstance(geometric, dict) and isinstance(ik, dict)
        group_count = min(int(geometric["count"]), int(ik["count"]))
        group_valid_fractions[group] = _fraction(
            group_count, evaluated_nonstale_frames
        )

    gates: list[dict[str, object]] = [
        {
            "name": "calibration_completed",
            "value": calibration_passed,
            "threshold": True,
            "requirement": "==",
            "passed": calibration_passed,
        },
        {
            "name": "sampled_frames",
            "value": sampled_frames,
            "threshold": 1,
            "requirement": ">=",
            "passed": sampled_frames >= 1,
        },
        {
            "name": "detected_sample_fraction",
            "value": detected_fraction,
            "threshold": MINIMUM_DETECTED_SAMPLE_FRACTION,
            "requirement": ">=",
            "passed": detected_fraction >= MINIMUM_DETECTED_SAMPLE_FRACTION,
        },
        {
            "name": "evaluated_nonstale_sample_fraction",
            "value": evaluated_fraction,
            "threshold": MINIMUM_EVALUATED_SAMPLE_FRACTION,
            "requirement": ">=",
            "passed": evaluated_fraction >= MINIMUM_EVALUATED_SAMPLE_FRACTION,
        },
    ]
    if expected_frames is not None:
        gates.append(
            {
                "name": "source_frames",
                "value": source_frames,
                "threshold": expected_frames,
                "requirement": "==",
                "passed": source_frames == expected_frames,
            }
        )
    for group, valid_fraction in group_valid_fractions.items():
        gates.append(
            {
                "name": f"{group}_valid_evaluated_fraction",
                "value": valid_fraction,
                "threshold": MINIMUM_VALID_EVALUATED_FRACTION,
                "requirement": ">=",
                "passed": valid_fraction
                >= MINIMUM_VALID_EVALUATED_FRACTION,
            }
        )
    failed = [str(gate["name"]) for gate in gates if not gate["passed"]]
    return {
        "detected_sample_fraction": detected_fraction,
        "evaluated_nonstale_frames": evaluated_nonstale_frames,
        "evaluated_nonstale_sample_fraction": evaluated_fraction,
        "post_calibration_frames": post_calibration_frames,
        "stale_frames": stale_frames,
        "stale_post_calibration_fraction": stale_fraction,
        "evaluation_population": (
            "paired frames with a detected skeleton and non-stale geometric "
            "and IK commands"
        ),
        "group_valid_evaluated_fraction": group_valid_fractions,
        "gates": gates,
        "failure_reasons": failed,
        "coverage_passed": not failed,
    }


def evaluate_video(
    path: Path,
    *,
    stride: int,
    expected_frames: int | None = None,
    calibration_video: Path | None = None,
    calibration_frame: int | None = None,
) -> dict[str, object]:
    """Evaluate one video and return JSON-serializable evidence."""

    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if expected_frames is not None and (
        isinstance(expected_frames, bool)
        or not isinstance(expected_frames, int)
        or expected_frames <= 0
    ):
        raise ValueError("expected_frames must be a positive integer")
    if (calibration_video is None) != (calibration_frame is None):
        raise ValueError("calibration_video and calibration_frame must be paired")
    if calibration_frame is not None and (
        isinstance(calibration_frame, bool)
        or not isinstance(calibration_frame, int)
        or calibration_frame < 0
    ):
        raise ValueError("calibration_frame must be a non-negative integer")
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
    post_calibration_frames = 0
    stale_frames = 0
    calibration: dict[str, object] = {
        "mode": "automatic_window",
        "source": None,
        "source_size_bytes": None,
        "source_sha256": None,
        "frame_index": None,
        "geometric_accepted": None,
        "ik_accepted": None,
        "status": "pending",
        "failure_reason": None,
    }
    try:
        if calibration_video is not None and calibration_frame is not None:
            calibration = _controlled_calibration(
                calibration_video,
                calibration_frame,
                pose=pose,
                geometric=geometric,
                ik=ik,
            )
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
            post_calibration_frames += 1
            for method, command in commands.items():
                stale[method] += int(command.stale)
            frame_stale = any(command.stale for command in commands.values())
            stale_frames += int(frame_stale)
            # Raw fidelity is a paired comparison. A stale output from either
            # retargeter makes the complete frame ineligible; otherwise a held
            # fallback could make missing perception look like valid tracking.
            if skeleton is None or frame_stale:
                continue
            evaluated_frames += 1
            for method, command in commands.items():
                fidelity = evaluator.evaluate(skeleton, command.positions_rad)
                for group, direction_names in GROUPS.items():
                    values[method][group].append(
                        fidelity.mean_error_deg(direction_names)
                    )
    finally:
        source.close()
        pose.close()

    if calibration["mode"] == "automatic_window":
        geometric_accepted = not geometric.is_calibrating
        ik_accepted = not ik.is_calibrating
        calibration_passed = geometric_accepted and ik_accepted
        calibration.update(
            {
                "geometric_accepted": geometric_accepted,
                "ik_accepted": ik_accepted,
                "status": "passed" if calibration_passed else "failed",
                "failure_reason": (
                    None
                    if calibration_passed
                    else "automatic_calibration_incomplete"
                ),
            }
        )
    else:
        calibration_passed = bool(
            calibration["status"] == "passed"
            and not geometric.is_calibrating
            and not ik.is_calibrating
        )
        if not calibration_passed and calibration["status"] == "passed":
            calibration.update(
                {
                    "status": "failed",
                    "failure_reason": "explicit_calibration_state_incomplete",
                }
            )

    metrics: dict[str, object] = {}
    for group in GROUPS:
        before = values["geometric"][group]
        after = values["ik"][group]
        metrics[group] = {
            "geometric": _summary(before),
            "ik": _summary(after),
            "ik_mean_improvement_percent": _paired_improvement(before, after),
        }
    coverage = _coverage_assessment(
        expected_frames=expected_frames,
        source_frames=source_frames,
        sampled_frames=sampled_frames,
        detected_frames=detected_frames,
        post_calibration_frames=post_calibration_frames,
        evaluated_nonstale_frames=evaluated_frames,
        stale_frames=stale_frames,
        metrics=metrics,
        calibration_passed=bool(calibration_passed),
    )
    return {
        "video": _project_path(path),
        "video_size_bytes": path.stat().st_size,
        "video_sha256": _sha256(path),
        "expected_frames": expected_frames,
        "calibration": calibration,
        "source_frames": source_frames,
        "sampled_frames": sampled_frames,
        "detected_frames": detected_frames,
        "post_calibration_frames": post_calibration_frames,
        "evaluated_frames": evaluated_frames,
        "stale_commands": stale,
        "stale_frames": stale_frames,
        "coverage": coverage,
        "metrics": metrics,
        "measurement_status": (
            "complete" if coverage["coverage_passed"] else "incomplete"
        ),
        "measurement_complete": coverage["coverage_passed"],
        "coverage_passed": coverage["coverage_passed"],
        "fidelity_acceptance": "not_defined",
    }


def _gate(
    name: str,
    *,
    value: object,
    threshold: object,
    requirement: str,
    passed: bool,
) -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "requirement": requirement,
        "passed": bool(passed),
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_report(
    evaluations: Sequence[Mapping[str, object]],
    clips: Sequence[ClipSpec],
    *,
    stride: int,
    retargeting_config: RetargetingConfig,
    command_argv: Sequence[str],
    provenance: Mapping[str, object],
    runtime_hashes_at_completion: Mapping[str, object],
) -> dict[str, object]:
    """Build a raw-measurement report without implying fidelity acceptance."""

    captured = dict(provenance)
    initial_hashes_raw = captured.get("runtime_input_sha256")
    initial_hashes = (
        dict(initial_hashes_raw)
        if isinstance(initial_hashes_raw, Mapping)
        else {}
    )
    final_hashes = dict(runtime_hashes_at_completion)
    start_violations = _runtime_snapshot_violations(initial_hashes, clips)
    completion_violations = _runtime_snapshot_violations(final_hashes, clips)
    start_complete = not (
        start_violations["missing_keys"] or start_violations["unexpected_keys"]
    )
    completion_complete = not (
        completion_violations["missing_keys"]
        or completion_violations["unexpected_keys"]
    )
    start_valid = not start_violations["invalid_sha256_keys"]
    completion_valid = not completion_violations["invalid_sha256_keys"]
    runtime_inputs_unchanged = initial_hashes == final_hashes
    git_provenance_valid = _git_provenance_valid(captured)
    provenance_gates = [
        _gate(
            "git_provenance_complete_and_valid",
            value={
                "git_revision": captured.get("git_revision"),
                "git_worktree": captured.get("git_worktree"),
            },
            threshold="valid revision and worktree snapshot",
            requirement="==",
            passed=git_provenance_valid,
        ),
        _gate(
            "runtime_snapshot_at_start_complete",
            value={
                "missing_keys": start_violations["missing_keys"],
                "unexpected_keys": start_violations["unexpected_keys"],
            },
            threshold={"missing_keys": [], "unexpected_keys": []},
            requirement="==",
            passed=start_complete,
        ),
        _gate(
            "runtime_snapshot_at_start_valid",
            value=start_violations["invalid_sha256_keys"],
            threshold=[],
            requirement="==",
            passed=start_valid,
        ),
        _gate(
            "runtime_snapshot_at_completion_complete",
            value={
                "missing_keys": completion_violations["missing_keys"],
                "unexpected_keys": completion_violations["unexpected_keys"],
            },
            threshold={"missing_keys": [], "unexpected_keys": []},
            requirement="==",
            passed=completion_complete,
        ),
        _gate(
            "runtime_snapshot_at_completion_valid",
            value=completion_violations["invalid_sha256_keys"],
            threshold=[],
            requirement="==",
            passed=completion_valid,
        ),
        _gate(
            "runtime_inputs_unchanged_during_run",
            value=runtime_inputs_unchanged,
            threshold=True,
            requirement="==",
            passed=runtime_inputs_unchanged,
        ),
    ]
    coverage_passed = bool(evaluations) and all(
        evaluation.get("coverage_passed") is True for evaluation in evaluations
    )
    provenance_complete = all(gate["passed"] for gate in provenance_gates)
    measurement_complete = coverage_passed and provenance_complete
    command = [str(argument) for argument in command_argv]
    return {
        "schema_version": 3,
        "report_kind": "raw_pose_measurement_diagnostic",
        "description": (
            "Raw retargeter FK direction-error measurements before standing-balance "
            "and support projection. This is a diagnostic completeness report, not "
            "a pose-fidelity acceptance result."
        ),
        "measurement_contract": {
            "angles": "anatomical forward/right/up direction error in degrees",
            "sampling": (
                "the calibration window is dense, then every configured stride frame"
            ),
            "stale_policy": (
                "a frame is evaluated only when a skeleton is detected and both "
                "geometric and IK commands are non-stale"
            ),
        },
        "command": subprocess.list2cmdline(command),
        "command_argv": command,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_started_at_utc": captured.get("captured_at_utc"),
        "git_revision": captured.get("git_revision"),
        "git_worktree": captured.get("git_worktree"),
        "runtime_input_sha256": initial_hashes,
        "runtime_input_sha256_at_completion": final_hashes,
        "runtime_inputs_unchanged_during_run": runtime_inputs_unchanged,
        "provenance_gates": provenance_gates,
        "coverage_policy": {
            "minimum_detected_sample_fraction": MINIMUM_DETECTED_SAMPLE_FRACTION,
            "minimum_evaluated_nonstale_sample_fraction": (
                MINIMUM_EVALUATED_SAMPLE_FRACTION
            ),
            "minimum_valid_evaluated_fraction": MINIMUM_VALID_EVALUATED_FRACTION,
            "primary_groups": list(PRIMARY_GROUPS),
        },
        "configuration": {
            "stride": stride,
            "dense_calibration_frames": retargeting_config.auto_calibration_frames,
            "retargeting": asdict(retargeting_config),
            "skeleton_filter": {
                "time_constant_s": 0.08,
                "confidence_threshold": 0.5,
                "max_gap_s": 0.25,
            },
            "pose_model": _project_path(POSE_MODEL),
            "mujoco_model": _project_path(MUJOCO_MODEL),
            "clips": [clip.name for clip in clips],
            "canonical_complete_matrix": tuple(clips) == default_clips(),
            "versions": {
                "python": platform.python_version(),
                "mediapipe": _package_version("mediapipe"),
                "mujoco": _package_version("mujoco"),
                "numpy": _package_version("numpy"),
                "opencv-contrib-python": _package_version(
                    "opencv-contrib-python"
                ),
                "PyYAML": _package_version("PyYAML"),
                "scipy": _package_version("scipy"),
            },
        },
        "videos": [dict(evaluation) for evaluation in evaluations],
        "coverage_passed": coverage_passed,
        "provenance_complete": provenance_complete,
        "measurement_status": "complete" if measurement_complete else "incomplete",
        "measurement_complete": measurement_complete,
        "fidelity_acceptance": "not_defined",
        "fidelity_acceptance_reason": (
            "No pose-error thresholds are declared by this raw diagnostic; coverage "
            "alone cannot establish acceptable imitation fidelity."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=[],
        help=(
            "MP4 to evaluate; repeat for multiple files "
            "(default: bundled + assets/videos/external)."
        ),
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
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    command_argv = (
        [str(argument) for argument in getattr(sys, "orig_argv", sys.argv)]
        if argv is None
        else [
            sys.executable,
            _project_path(Path(__file__).resolve()),
            *effective_argv,
        ]
    )
    args = build_parser().parse_args(effective_argv)
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    clips = (
        [_clip_for_path(path) for path in args.video]
        if args.video
        else list(default_clips())
    )
    if not clips:
        raise FileNotFoundError("no videos selected for pose-fidelity evaluation")
    missing = [clip.path for clip in clips if not clip.path.is_file()]
    if missing:
        raise FileNotFoundError(f"selected videos do not exist: {missing}")

    provenance = _capture_run_provenance(clips)
    retargeting_config = load_retargeting_config(RETARGETING_CONFIG)
    evaluations = [
        evaluate_video(
            clip.path,
            stride=args.stride,
            expected_frames=clip.expected_frames,
            calibration_video=clip.calibration_video,
            calibration_frame=clip.calibration_frame,
        )
        for clip in clips
    ]
    runtime_hashes_at_completion = _runtime_input_hashes(clips)
    report = build_report(
        evaluations,
        clips,
        stride=args.stride,
        retargeting_config=retargeting_config,
        command_argv=command_argv,
        provenance=provenance,
        runtime_hashes_at_completion=runtime_hashes_at_completion,
    )
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
        assert (
            isinstance(arms, dict)
            and isinstance(legs, dict)
            and isinstance(end_effectors, dict)
        )

        def improvement_text(group: dict[str, object]) -> str:
            value = group.get("ik_mean_improvement_percent")
            return "n/a" if value is None else f"{float(value):+.1f}%"

        print(
            f"{video['video']}: measurement={video['measurement_status']} "
            f"evaluated={video['evaluated_frames']}/{video['sampled_frames']} "
            "IK mean improvement "
            f"arms={improvement_text(arms)} "
            f"legs={improvement_text(legs)} "
            f"end={improvement_text(end_effectors)}"
        )
    print(
        "raw diagnostic measurement: "
        f"{report['measurement_status']}; fidelity acceptance is not defined"
    )
    return 0 if report["measurement_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
