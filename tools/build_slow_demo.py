"""Build the bundled slow full-body tracking demo from two DVIDS clips.

The source clips are public-domain exercise demonstrations.  This script does
not download them; pass the local 1024x576 MP4 files downloaded from the URLs
documented in ``assets/README.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


OUTPUT_FPS = 29.97
OUTPUT_SIZE = (1024, 576)
SLOWDOWN_FACTOR = 2


def _read_segment(path: Path, start_s: float, end_s: float) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Source video has no usable FPS metadata: {path}")

    first = round(start_s * fps)
    last = round(end_s * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames: list[np.ndarray] = []
    for _ in range(first, last):
        ok, frame = capture.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != OUTPUT_SIZE:
            frame = cv2.resize(frame, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()

    if not frames:
        raise RuntimeError(f"Selected segment is empty: {path} [{start_s}, {end_s}]")
    return frames


def build_demo(arm_video: Path, balance_video: Path, output: Path) -> None:
    # Remove the long static introductions and fade-outs from the source clips.
    arms = _read_segment(arm_video, start_s=8.5, end_s=24.5)
    balance = _read_segment(balance_video, start_s=14.0, end_s=29.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        OUTPUT_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MP4: {output}")

    # A neutral lead-in gives the tracker time to acquire the full skeleton.
    for _ in range(round(2.0 * OUTPUT_FPS)):
        writer.write(arms[0])
    for frame in arms:
        for _ in range(SLOWDOWN_FACTOR):
            writer.write(frame)

    # A short neutral pause makes the boundary between arm and leg tests clear.
    for _ in range(round(1.5 * OUTPUT_FPS)):
        writer.write(balance[0])
    for frame in balance:
        for _ in range(SLOWDOWN_FACTOR):
            writer.write(frame)
    writer.release()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--balance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    build_demo(args.arms, args.balance, args.output)
