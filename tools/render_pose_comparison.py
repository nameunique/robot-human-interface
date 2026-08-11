"""Render synchronized human/robot snapshots from a replay video.

The robot column is the actual grounded-fixed MuJoCo state after applying the
default constrained-IK targets through the position actuators.  This is a
visual acceptance artifact, not a replacement for the numeric FK benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import mujoco
import numpy as np

from robot_human_interface.camera import OpenCVVideoSource
from robot_human_interface.pose import (
    MediaPipePoseConfig,
    MediaPipePoseLandmarker,
    draw_pose_overlay,
)
from robot_human_interface.retargeting import MujocoIKRetargeter
from robot_human_interface.simulation import HumanoidSimulation
from robot_human_interface.skeleton import (
    PoseLandmark,
    SkeletonEMAFilter,
    SkeletonFilterConfig,
    SkeletonFrame,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "assets" / "videos" / "slow_balance_demo.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "ik-pose-comparison.png"
DEFAULT_TIMES_S = (5.9, 29.7, 43.2, 50.5)


def _resize(image: np.ndarray, width: int = 400, height: int = 225) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _annotate_anatomical_sides(
    image_bgr: np.ndarray,
    skeleton: SkeletonFrame | None,
) -> np.ndarray:
    """Label human wrists/ankles with the same anatomical colors as the robot."""

    if skeleton is None:
        return image_bgr
    output = image_bgr.copy()
    height, width = output.shape[:2]
    valid = skeleton.valid_mask(0.5)
    labels = (
        (PoseLandmark.RIGHT_WRIST, "R", (50, 120, 235)),
        (PoseLandmark.RIGHT_ANKLE, "R", (50, 120, 235)),
        (PoseLandmark.LEFT_WRIST, "L", (80, 205, 105)),
        (PoseLandmark.LEFT_ANKLE, "L", (80, 205, 105)),
    )
    for landmark, label, color in labels:
        index = int(landmark)
        if not valid[index]:
            continue
        point = skeleton.landmarks_2d[index]
        x = int(round(float(point[0]) * (width - 1)))
        y = int(round(float(point[1]) * (height - 1)))
        cv2.putText(
            output,
            label,
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            color,
            4,
            cv2.LINE_AA,
        )
    return output


def _comparison_cell(
    human_bgr: np.ndarray,
    robot_front_bgr: np.ndarray,
    robot_rear_bgr: np.ndarray,
    *,
    media_time_s: float,
) -> np.ndarray:
    cell = np.full((300, 1200, 3), 20, dtype=np.uint8)
    cell[50:275, :400] = _resize(human_bgr)
    cell[50:275, 400:800] = _resize(robot_front_bgr)
    cell[50:275, 800:] = _resize(robot_rear_bgr)
    cv2.putText(
        cell,
        f"t={media_time_s:04.1f}s",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        cell,
        "human + MediaPipe",
        (150, 294),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (130, 235, 130),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        cell,
        "robot true front: R=orange, L=green",
        (430, 294),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (120, 210, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        cell,
        "robot rear: screen sides aligned",
        (872, 294),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (120, 210, 255),
        1,
        cv2.LINE_AA,
    )
    return cell


def render_comparison(
    video_path: Path,
    output_path: Path,
    sample_times_s: Sequence[float],
) -> Path:
    times = tuple(sorted(float(value) for value in sample_times_s))
    if not times or not np.isfinite(times).all() or times[0] < 0.0:
        raise ValueError("sample times must be finite, non-negative values")
    if not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")

    source = OpenCVVideoSource(video_path, realtime=False)
    pose = MediaPipePoseLandmarker(
        MediaPipePoseConfig(
            model_asset_path=PROJECT_ROOT
            / "assets"
            / "models"
            / "pose_landmarker_full.task"
        ),
        landmark_filter=SkeletonEMAFilter(
            SkeletonFilterConfig(
                time_constant_s=0.08,
                confidence_threshold=0.5,
                max_gap_s=0.25,
            )
        ),
    )
    retargeter = MujocoIKRetargeter.from_yaml(
        joints_path=PROJECT_ROOT / "config" / "joints.yaml",
        retargeting_path=PROJECT_ROOT / "config" / "retargeting.yaml",
    )
    simulation = HumanoidSimulation("fixed")
    renderer = mujoco.Renderer(simulation.model, height=360, width=640)
    render_options = mujoco.MjvOption()
    render_options.geomgroup[:] = 0
    render_options.geomgroup[0] = 1
    render_options.geomgroup[1] = 1
    render_options.sitegroup[:] = 0
    front_camera = mujoco.MjvCamera()
    front_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    front_camera.lookat[:] = (0.0, 0.0, 0.85)
    front_camera.distance = 2.8
    # Physical robot front is -X, so a camera located at -X (MuJoCo azimuth 0)
    # matches Unity's named FRONT view and the long toe edge.
    front_camera.azimuth = 0.0
    front_camera.elevation = -8.0
    rear_camera = mujoco.MjvCamera()
    rear_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    rear_camera.lookat[:] = front_camera.lookat
    rear_camera.distance = front_camera.distance
    rear_camera.azimuth = 180.0
    rear_camera.elevation = front_camera.elevation

    cells: list[np.ndarray] = []
    next_sample = 0
    first_timestamp_s: float | None = None
    previous_timestamp_s: float | None = None
    physics_accumulator_s = 0.0
    timestep_s = float(simulation.model.opt.timestep)
    try:
        while next_sample < len(times):
            frame = source.read()
            if frame is None:
                break
            if first_timestamp_s is None:
                first_timestamp_s = frame.timestamp_s
            skeleton = pose.estimate(frame)
            command = retargeter.retarget(skeleton, timestamp_s=frame.timestamp_s)
            simulation.apply_joint_command(command)

            if previous_timestamp_s is not None:
                physics_accumulator_s += max(
                    0.0, frame.timestamp_s - previous_timestamp_s
                )
            previous_timestamp_s = frame.timestamp_s
            physics_steps = int((physics_accumulator_s + 1e-12) // timestep_s)
            physics_accumulator_s -= physics_steps * timestep_s
            if physics_steps > 0:
                simulation.step(physics_steps)

            media_time_s = frame.timestamp_s - first_timestamp_s
            if media_time_s + 1e-9 < times[next_sample]:
                continue
            human = _annotate_anatomical_sides(
                draw_pose_overlay(frame.image_bgr, skeleton), skeleton
            )
            renderer.update_scene(
                simulation.data,
                camera=front_camera,
                scene_option=render_options,
            )
            robot_front = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            renderer.update_scene(
                simulation.data,
                camera=rear_camera,
                scene_option=render_options,
            )
            robot_rear = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cells.append(
                _comparison_cell(
                    human,
                    robot_front,
                    robot_rear,
                    media_time_s=media_time_s,
                )
            )
            next_sample += 1
    finally:
        renderer.close()
        simulation.close()
        pose.close()
        source.close()

    if len(cells) != len(times):
        raise RuntimeError(
            f"video ended after {len(cells)} of {len(times)} requested snapshots"
        )
    rows = []
    for start in range(0, len(cells), 2):
        row_cells = cells[start : start + 2]
        if len(row_cells) == 1:
            row_cells.append(np.full_like(row_cells[0], 20))
        rows.append(np.concatenate(row_cells, axis=1))
    contact_sheet = np.concatenate(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), contact_sheet):
        raise OSError(f"failed to write comparison image: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--time",
        type=float,
        action="append",
        default=[],
        help="Media second to render; repeat for multiple snapshots.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    times = args.time if args.time else DEFAULT_TIMES_S
    output = render_comparison(
        args.video.expanduser().resolve(),
        args.output.expanduser().resolve(),
        times,
    )
    print(f"pose comparison: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
