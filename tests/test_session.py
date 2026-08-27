from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest

from robot_human_interface.playback import PlaybackDiscontinuity, PlaybackState
from robot_human_interface.protocol import (
    OperatorSafetyAcknowledgement,
    SafeRobotController,
    finalize_safe_command,
)
from robot_human_interface.session import (
    PipelineSnapshot,
    SessionConfig,
    SessionStatus,
    SourceKind,
    SourceSpec,
    TeleopSession,
    create_default_session,
)
from robot_human_interface.skeleton import CameraFrame, JOINT_NAMES, RobotJointCommand


def _source(source_id: str = "synthetic") -> SourceSpec:
    return SourceSpec(SourceKind.SYNTHETIC, source_id, source_id)


@dataclass
class _Pipeline:
    config: SessionConfig
    clock: list[float]
    closed: int = 0
    reset_count: int = 0
    calibrations: list[int] = field(default_factory=list)
    viewer_open: bool = False
    steps: int = 0
    end_after: int | None = None
    fail: bool = False

    def step(self) -> PipelineSnapshot | None:
        if self.fail:
            raise RuntimeError("pipeline failed")
        if self.end_after is not None and self.steps >= self.end_after:
            return None
        image = np.full((2, 3, 3), self.steps, dtype=np.uint8)
        command = RobotJointCommand.humanoid(
            self.clock[0],
            np.zeros(len(JOINT_NAMES)),
            1.0,
        )
        snapshot = PipelineSnapshot(
            self.steps,
            self.clock[0],
            SessionStatus.RUNNING,
            self.config.source,
            frame=CameraFrame(image, self.clock[0], self.steps),
            raw_command=command,
            safe_command=finalize_safe_command(
                command,
                free_base_active=self.config.free_base,
                balance_active=self.config.balance_enabled,
            ),
            tracking_quality=0.9,
            telemetry={"mutable": [1, 2]},
        )
        image[:] = 255
        self.steps += 1
        return snapshot

    def reset(self) -> None:
        self.reset_count += 1

    def calibrate(self, sample_count: int) -> None:
        self.calibrations.append(sample_count)

    def open_viewer(self) -> None:
        self.viewer_open = True

    def close_viewer(self) -> None:
        self.viewer_open = False

    def close(self) -> None:
        self.closed += 1


@dataclass
class _SeekablePipeline(_Pipeline):
    frame_count: int = 4
    fps: float = 10.0
    current_frame: int = -1
    eof: bool = False
    rate: float = 1.0
    loop_enabled: bool = False
    loop_start_s: float = 0.0
    loop_end_s: float | None = None
    pending_discontinuity: PlaybackDiscontinuity | None = None

    @property
    def playback_state(self) -> PlaybackState:
        frame_index = self.current_frame if self.current_frame >= 0 else self.steps
        position = self.frame_count / self.fps if self.eof else frame_index / self.fps
        return PlaybackState(
            True,
            position,
            self.frame_count / self.fps,
            max(0, min(frame_index, self.frame_count - 1)),
            self.frame_count,
            self.fps,
            self.rate,
            self.loop_enabled,
            self.loop_start_s,
            self.loop_end_s,
            self.eof,
            self.pending_discontinuity,
        )

    def step(self) -> PipelineSnapshot | None:
        if self.steps >= self.frame_count:
            self.eof = True
            return None
        current = self.steps
        snapshot = super().step()
        assert snapshot is not None
        self.current_frame = current
        self.eof = False
        state = self.playback_state
        self.pending_discontinuity = None
        return replace(snapshot, playback=state)

    def seek(self, position_s: float) -> PlaybackState:
        target = min(
            self.frame_count - 1,
            max(0, int(np.floor(float(position_s) * self.fps + 1e-9))),
        )
        reason = (
            PlaybackDiscontinuity.RESTART
            if self.eof and target == 0
            else PlaybackDiscontinuity.SEEK
        )
        self.steps = target
        self.current_frame = target
        self.eof = False
        self.pending_discontinuity = reason
        return self.playback_state

    def step_relative(self, delta_frames: int) -> PlaybackState:
        target = min(
            self.frame_count - 1,
            max(0, max(self.current_frame, 0) + int(delta_frames)),
        )
        self.steps = target
        self.current_frame = target
        self.eof = False
        self.pending_discontinuity = PlaybackDiscontinuity.STEP
        return self.playback_state

    def set_rate(self, rate: float) -> PlaybackState:
        self.rate = float(rate)
        return self.playback_state

    def set_loop(
        self,
        enabled: bool,
        start_s: float,
        end_s: float | None,
    ) -> PlaybackState:
        self.loop_enabled = bool(enabled)
        self.loop_start_s = float(start_s)
        self.loop_end_s = end_s
        return self.playback_state


def test_source_spec_requires_paths_for_video_and_copies_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="video sources require"):
        SourceSpec(SourceKind.USER_VIDEO, "mine", "Моё видео")

    metadata = {"tags": ["one"]}
    spec = SourceSpec(
        SourceKind.USER_VIDEO,
        "mine",
        "Моё видео",
        path=tmp_path / "clip.mp4",
        metadata=metadata,
    )
    metadata["tags"].append("two")

    assert spec.path is not None and spec.path.is_absolute()
    assert spec.metadata["tags"] == ("one",)


def test_command_queue_runs_lifecycle_and_source_swap_closes_old_resources() -> None:
    clock = [1.0]
    pipelines: list[_Pipeline] = []

    def factory(config: SessionConfig) -> _Pipeline:
        pipeline = _Pipeline(config, clock)
        pipelines.append(pipeline)
        return pipeline

    session = TeleopSession(SessionConfig(_source("first")), factory, clock=lambda: clock[0])
    session.request_start()
    snapshot = session.step()

    assert session.status is SessionStatus.RUNNING
    assert snapshot.status is SessionStatus.RUNNING
    assert snapshot.source.source_id == "first"
    assert len(pipelines) == 1

    session.request_reset()
    session.request_calibrate(12)
    session.request_open_viewer()
    session.step()
    assert pipelines[0].reset_count == 1
    assert pipelines[0].calibrations == [12]
    assert pipelines[0].viewer_open

    session.request_change_source(_source("second"))
    changed = session.step()
    assert pipelines[0].closed == 1
    assert len(pipelines) == 2
    assert changed.source.source_id == "second"
    assert session.status is SessionStatus.RUNNING

    session.request_stop()
    stopped = session.step()
    assert stopped.status is SessionStatus.STOPPED
    assert pipelines[1].closed == 1


def test_snapshot_arrays_and_nested_telemetry_are_isolated_and_read_only() -> None:
    clock = [2.0]
    session = TeleopSession(
        SessionConfig(_source()),
        lambda config: _Pipeline(config, clock),
        clock=lambda: clock[0],
    )
    session.start()
    snapshot = session.step()

    assert snapshot.frame is not None
    assert np.all(snapshot.frame.image_bgr == 0)
    assert not snapshot.frame.image_bgr.flags.writeable
    with pytest.raises(ValueError):
        snapshot.frame.image_bgr[:] = 3
    assert snapshot.telemetry["mutable"] == (1, 2)

    second_copy = session.latest_snapshot
    assert second_copy is not snapshot
    assert second_copy.frame is not snapshot.frame
    assert second_copy.frame is not None
    assert not np.shares_memory(second_copy.frame.image_bgr, snapshot.frame.image_bgr)


def test_finite_source_and_pipeline_error_close_resources_and_emit_events() -> None:
    clock = [3.0]
    pipelines: list[_Pipeline] = []

    def finite_factory(config: SessionConfig) -> _Pipeline:
        pipeline = _Pipeline(config, clock, end_after=0)
        pipelines.append(pipeline)
        return pipeline

    session = TeleopSession(SessionConfig(_source()), finite_factory, clock=lambda: clock[0])
    session.start()
    ended = session.step()
    assert ended.status is SessionStatus.ENDED
    assert pipelines[0].closed == 1
    assert "SOURCE_ENDED" in {event.code for event in session.drain_events()}

    def failing_factory(config: SessionConfig) -> _Pipeline:
        pipeline = _Pipeline(config, clock, fail=True)
        pipelines.append(pipeline)
        return pipeline

    failed = TeleopSession(SessionConfig(_source()), failing_factory, clock=lambda: clock[0])
    failed.start()
    error_snapshot = failed.step()
    assert error_snapshot.status is SessionStatus.ERROR
    assert pipelines[1].closed == 1
    assert "PIPELINE_STEP_FAILED" in {event.code for event in failed.drain_events()}


def test_close_is_idempotent_and_rejects_new_queued_commands() -> None:
    clock = [4.0]
    pipelines: list[_Pipeline] = []

    def factory(config: SessionConfig) -> _Pipeline:
        pipeline = _Pipeline(config, clock)
        pipelines.append(pipeline)
        return pipeline

    session = TeleopSession(SessionConfig(_source()), factory, clock=lambda: clock[0])
    session.start()
    session.close()
    session.close()

    assert session.status is SessionStatus.CLOSED
    assert pipelines[0].closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.request_start()


def test_default_session_runs_the_real_synthetic_balance_pipeline() -> None:
    config = SessionConfig(
        SourceSpec(
            SourceKind.SYNTHETIC,
            "default-smoke",
            "Synthetic smoke",
            width=160,
            height=120,
            fps=120.0,
        ),
        physics_steps_per_frame=1,
    )
    session = create_default_session(config)

    session.start()
    snapshot = session.run_once()
    session.close()

    assert snapshot.frame is not None and snapshot.frame.image_bgr.shape == (120, 160, 3)
    assert snapshot.skeleton is not None
    assert snapshot.safe_command is not None
    assert snapshot.safe_command.joint_names == JOINT_NAMES
    assert snapshot.safe_command.stage == "balance_support_final"
    assert snapshot.telemetry["support_phase"] == "double_support"
    assert snapshot.telemetry["free_base_active"] is True
    assert snapshot.telemetry["balance_active"] is True
    assert session.status is SessionStatus.CLOSED

    payloads: list[str] = []

    class RecordingTransport:
        def send(self, payload: str) -> None:
            payloads.append(payload)

    output = SafeRobotController(RecordingTransport())
    assert output.connect()
    output.submit_safe_command(
        snapshot.safe_command,
        free_base=bool(snapshot.telemetry["free_base_active"]),
        balance_enabled=bool(snapshot.telemetry["balance_active"]),
    )
    assert output.arm(OperatorSafetyAcknowledgement(True, True, True))
    assert output.tick()
    assert len(payloads) == 1 and '"method":"setPositions"' in payloads[0]


@dataclass
class _RobotInvalidationSpy:
    reasons: list[str] = field(default_factory=list)
    closed: int = 0

    def invalidate(self, reason: str) -> None:
        self.reasons.append(reason)

    def close(self) -> None:
        self.closed += 1


class _WrongResultPipeline(_Pipeline):
    def step(self) -> object:
        return object()


def test_non_snapshot_pipeline_result_fails_closed_and_invalidates_robot() -> None:
    clock = [5.0]
    robot = _RobotInvalidationSpy()
    pipeline: _WrongResultPipeline | None = None

    def factory(config: SessionConfig) -> _WrongResultPipeline:
        nonlocal pipeline
        pipeline = _WrongResultPipeline(config, clock)
        return pipeline

    session = TeleopSession(
        SessionConfig(_source()),
        factory,
        robot_output=robot,  # type: ignore[arg-type]
        clock=lambda: clock[0],
    )
    session.start()

    snapshot = session.step()

    assert snapshot.status is SessionStatus.ERROR
    assert pipeline is not None and pipeline.closed == 1
    assert robot.reasons == ["pipeline_error"]
    events = session.drain_events()
    assert any(
        event.code == "PIPELINE_STEP_FAILED" and "PipelineSnapshot" in str(event.details)
        for event in events
    )


def test_seek_autopauses_builds_preview_and_never_exposes_safe_command() -> None:
    clock = [6.0]
    robot = _RobotInvalidationSpy()
    pipeline: _SeekablePipeline | None = None

    def factory(config: SessionConfig) -> _SeekablePipeline:
        nonlocal pipeline
        pipeline = _SeekablePipeline(config, clock)
        return pipeline

    session = TeleopSession(
        SessionConfig(_source()),
        factory,
        robot_output=robot,  # type: ignore[arg-type]
        clock=lambda: clock[0],
    )
    session.request_start()
    session.request_seek(0.2)

    preview = session.step()

    assert preview.status is SessionStatus.PAUSED
    assert preview.frame is not None and int(preview.frame.image_bgr[0, 0, 0]) == 2
    assert preview.safe_command is None
    assert preview.playback is not None
    assert preview.playback.frame_index == 2
    assert preview.playback.discontinuity_reason is PlaybackDiscontinuity.SEEK
    assert robot.reasons == ["playback_seek"]
    assert pipeline is not None and pipeline.closed == 0


def test_seekable_eof_retains_pipeline_and_step_back_recovers_preview() -> None:
    clock = [7.0]
    pipeline: _SeekablePipeline | None = None

    def factory(config: SessionConfig) -> _SeekablePipeline:
        nonlocal pipeline
        pipeline = _SeekablePipeline(config, clock, frame_count=2)
        return pipeline

    session = TeleopSession(SessionConfig(_source()), factory, clock=lambda: clock[0])
    session.start()
    assert session.step().status is SessionStatus.RUNNING
    assert session.step().status is SessionStatus.RUNNING
    ended = session.step()

    assert ended.status is SessionStatus.ENDED
    assert ended.playback is not None and ended.playback.eof
    assert pipeline is not None and pipeline.closed == 0

    session.request_step_frame(-1)
    recovered = session.step()
    assert recovered.status is SessionStatus.PAUSED
    assert recovered.playback is not None and recovered.playback.frame_index == 0
    assert recovered.safe_command is None
    assert pipeline.closed == 0


def test_playback_commands_on_nonseekable_source_only_warn() -> None:
    clock = [8.0]
    session = TeleopSession(
        SessionConfig(_source()),
        lambda config: _Pipeline(config, clock),
        clock=lambda: clock[0],
    )
    session.request_start()
    session.request_seek(1.0)

    snapshot = session.step()
    codes = [event.code for event in session.drain_events()]

    assert snapshot.status is SessionStatus.RUNNING
    assert "PLAYBACK_UNSUPPORTED" in codes
    assert "SESSION_COMMAND_FAILED" not in codes


def test_session_forwards_rate_and_loop_to_optional_pipeline() -> None:
    clock = [9.0]
    pipeline: _SeekablePipeline | None = None

    def factory(config: SessionConfig) -> _SeekablePipeline:
        nonlocal pipeline
        pipeline = _SeekablePipeline(config, clock)
        return pipeline

    session = TeleopSession(SessionConfig(_source()), factory, clock=lambda: clock[0])
    session.request_start()
    session.request_seek(0.0)
    session.step()
    session.request_set_playback_rate(1.5)
    session.request_set_loop(True, 0.1, 0.3)
    updated = session.step()

    assert pipeline is not None
    assert pipeline.rate == 1.5
    assert pipeline.loop_enabled
    assert pipeline.loop_start_s == pytest.approx(0.1)
    assert pipeline.loop_end_s == pytest.approx(0.3)
    assert updated.playback is not None and updated.playback.rate == 1.5
    assert updated.safe_command is None
