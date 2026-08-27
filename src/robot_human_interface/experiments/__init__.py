"""Reproducible, Qt-independent experiment recording primitives."""

from .recorder import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentRecorder,
    ExperimentSample,
    ExperimentSpec,
    RecorderState,
    RecorderSummary,
    recover_interrupted_experiments,
    sha256_file,
)
from .video import (
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_QUEUE_CAPACITY,
    CameraVideoSink,
    CameraVideoSinkStatus,
    CameraVideoStatus,
)

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "DEFAULT_VIDEO_CODEC",
    "DEFAULT_VIDEO_QUEUE_CAPACITY",
    "CameraVideoSink",
    "CameraVideoSinkStatus",
    "CameraVideoStatus",
    "ExperimentRecorder",
    "ExperimentSample",
    "ExperimentSpec",
    "RecorderState",
    "RecorderSummary",
    "recover_interrupted_experiments",
    "sha256_file",
]
