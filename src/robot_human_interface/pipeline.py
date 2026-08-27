"""Concrete resource-owning pipeline used by :class:`TeleopSession`.

Imports of OpenCV and MediaPipe remain lazy: ``QApplication`` can therefore be
created before a GUI worker calls :meth:`DefaultTeleopPipeline.start`, avoiding
cross-contamination between OpenCV's bundled Qt plugins and PyQt6.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from robot_human_interface.playback import PlaybackState
from robot_human_interface.protocol import finalize_safe_command
from robot_human_interface.resources import ResourceLocator
from robot_human_interface.session import (
    PipelineSnapshot,
    SessionConfig,
    SessionStatus,
    SourceKind,
)


class DefaultTeleopPipeline:
    """Synchronous video/camera -> pose -> safe command -> MuJoCo owner."""

    def __init__(
        self,
        config: SessionConfig,
        *,
        resources: ResourceLocator | None = None,
    ) -> None:
        if not isinstance(config, SessionConfig):
            raise TypeError("config must be a SessionConfig")
        self.config = config
        self._resources = resources or ResourceLocator()
        self._source: Any | None = None
        self._pose: Any | None = None
        self._retargeter: Any | None = None
        self._simulation: Any | None = None
        self._balance_controller: Any | None = None
        self._support_machine: Any | None = None
        self._support_latch: Any | None = None
        self._support_estimator: Any | None = None
        self._support_intent_type: Any | None = None
        self._support_phase_type: Any | None = None
        self._started = False
        self._closed = False
        self._sequence = 0
        self._previous_frame_timestamp_s: float | None = None
        self._physics_accumulator_s = 0.0
        self._last_safe_command: Any | None = None
        self._suppress_safe_command_once = False

    def _make_perception(self) -> tuple[Any, Any]:
        source_spec = self.config.source
        if source_spec.kind is SourceKind.SYNTHETIC:
            from robot_human_interface.camera import (
                SyntheticCameraConfig,
                SyntheticCameraSource,
            )
            from robot_human_interface.pose import SyntheticPoseEstimator

            source = SyntheticCameraSource(
                SyntheticCameraConfig(
                    width=source_spec.width,
                    height=source_spec.height,
                    fps=source_spec.fps,
                    realtime=True,
                )
            )
            return source, SyntheticPoseEstimator()

        from robot_human_interface.camera import (
            OpenCVCameraConfig,
            OpenCVCameraSource,
            OpenCVVideoSource,
        )
        from robot_human_interface.pose import (
            MediaPipePoseConfig,
            MediaPipePoseLandmarker,
        )
        from robot_human_interface.skeleton import SkeletonEMAFilter, SkeletonFilterConfig

        if source_spec.kind is SourceKind.CAMERA:
            source = OpenCVCameraSource(
                OpenCVCameraConfig(
                    index=source_spec.camera_index,
                    width=source_spec.width,
                    height=source_spec.height,
                    fps=source_spec.fps,
                    mirror=source_spec.mirror,
                    backend=source_spec.camera_backend,
                )
            )
        else:
            assert source_spec.path is not None
            if not source_spec.path.is_file():
                raise FileNotFoundError(f"video file does not exist: {source_spec.path}")
            source = OpenCVVideoSource(
                source_spec.path,
                mirror=source_spec.mirror,
                loop=source_spec.loop,
                realtime=True,
            )
        pose_model = self._resources.asset("models", "pose_landmarker_full.task")
        pose = MediaPipePoseLandmarker(
            MediaPipePoseConfig(model_asset_path=pose_model),
            landmark_filter=SkeletonEMAFilter(
                SkeletonFilterConfig(
                    time_constant_s=0.08,
                    confidence_threshold=0.5,
                    max_gap_s=0.25,
                )
            ),
        )
        return source, pose

    def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RuntimeError("pipeline is closed")
        try:
            self._source, self._pose = self._make_perception()
            from robot_human_interface.control import (
                HumanSupportIntentEstimator,
                StandingBalanceController,
                SupportIntent,
                SupportIntentLatch,
                SupportPhase,
                SupportStateMachine,
                load_human_support_intent_config,
                load_standing_balance_config,
                load_support_control_config,
            )
            from robot_human_interface.retargeting import (
                GeometricRetargeter,
                MujocoIKRetargeter,
            )
            from robot_human_interface.simulation import HumanoidSimulation

            retargeter_type = (
                MujocoIKRetargeter
                if self.config.retargeting == "ik"
                else GeometricRetargeter
            )
            self._retargeter = retargeter_type.from_yaml(
                joints_path=self._resources.config("joints.yaml"),
                retargeting_path=self._resources.config("retargeting.yaml"),
            )
            self._simulation = HumanoidSimulation(
                "free" if self.config.free_base else "fixed"
            )
            balance_enabled = bool(
                self.config.free_base and self.config.balance_enabled
            )
            balance_config = load_standing_balance_config(
                self._resources.config("balance.yaml")
            )
            self._balance_controller = StandingBalanceController.from_simulation(
                self._simulation,
                replace(balance_config, enabled=balance_enabled),
            )
            support_config = load_support_control_config(
                self._resources.config("balance.yaml")
            )
            self._support_machine = SupportStateMachine.from_simulation(
                self._simulation,
                support_config,
            )
            self._support_latch = SupportIntentLatch()
            self._support_estimator = HumanSupportIntentEstimator(
                load_human_support_intent_config(
                    self._resources.config("balance.yaml")
                )
            )
            self._support_intent_type = SupportIntent
            self._support_phase_type = SupportPhase
            self._started = True
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _require_started(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("pipeline is not running")

    @property
    def simulation(self) -> Any:
        self._require_started()
        return self._simulation

    @property
    def playback_state(self) -> PlaybackState | None:
        source = self._source
        if source is None:
            return None
        state = getattr(source, "playback_state", None)
        if state is None:
            return None
        if not isinstance(state, PlaybackState):
            raise TypeError("source playback_state must be a PlaybackState")
        return state

    def _playback_call(self, method_name: str, *args: object) -> PlaybackState:
        self._require_started()
        method = getattr(self._source, method_name, None)
        if not callable(method):
            raise NotImplementedError(
                f"source does not support playback operation {method_name}"
            )
        state = method(*args)
        if not isinstance(state, PlaybackState):
            state = self.playback_state
        if not isinstance(state, PlaybackState):
            raise TypeError(f"{method_name} must return a PlaybackState")
        return state

    def seek(self, position_s: float) -> PlaybackState:
        state = self._playback_call("seek", position_s)
        self._reset_temporal_state()
        return state

    def step_relative(self, delta_frames: int) -> PlaybackState:
        state = self._playback_call("step_relative", delta_frames)
        self._reset_temporal_state()
        return state

    def set_rate(self, rate: float) -> PlaybackState:
        return self._playback_call("set_rate", rate)

    def set_loop(
        self,
        enabled: bool,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> PlaybackState:
        state = self._playback_call("set_loop", enabled, start_s, end_s)
        self._reset_temporal_state()
        return state

    def _reset_temporal_state(self) -> None:
        """Reset time-dependent state without discarding accepted calibration."""

        pose_reset = getattr(self._pose, "reset_temporal", None)
        if callable(pose_reset):
            pose_reset()
        retargeting_reset = getattr(self._retargeter, "reset_temporal", None)
        if callable(retargeting_reset):
            retargeting_reset()
        if self._simulation is not None:
            self._simulation.reset()
        for resetter in (
            self._balance_controller,
            self._support_machine,
            self._support_latch,
        ):
            reset = getattr(resetter, "reset", None)
            if callable(reset):
                reset()
        intent_reset = getattr(self._support_estimator, "reset_temporal", None)
        if callable(intent_reset):
            intent_reset()
        self._previous_frame_timestamp_s = None
        self._physics_accumulator_s = 0.0
        self._last_safe_command = None
        self._suppress_safe_command_once = True

    def _tracking_quality(self, skeleton: Any | None) -> float:
        if skeleton is None:
            return 0.0
        confidence = skeleton.confidence()
        finite = confidence[np.isfinite(confidence)]
        return 0.0 if finite.size == 0 else float(np.clip(np.mean(finite), 0.0, 1.0))

    def _physics_step_count(self, frame_timestamp_s: float) -> int:
        assert self._simulation is not None
        timestep_s = float(self._simulation.model.opt.timestep)
        if self.config.physics_steps_per_frame > 0:
            return self.config.physics_steps_per_frame
        if self._previous_frame_timestamp_s is None:
            delta_s = 0.0
        else:
            delta_s = max(
                0.0,
                min(0.25, frame_timestamp_s - self._previous_frame_timestamp_s),
            )
        self._previous_frame_timestamp_s = frame_timestamp_s
        self._physics_accumulator_s += delta_s
        steps = int((self._physics_accumulator_s + 1e-12) // timestep_s)
        self._physics_accumulator_s -= steps * timestep_s
        return steps

    def step(self) -> PipelineSnapshot | None:
        self._require_started()
        assert self._source is not None
        assert self._pose is not None
        assert self._retargeter is not None
        assert self._simulation is not None
        assert self._balance_controller is not None
        assert self._support_machine is not None
        assert self._support_latch is not None
        assert self._support_estimator is not None
        assert self._support_intent_type is not None
        assert self._support_phase_type is not None

        frame = self._source.read()
        if frame is None:
            return None
        playback = self.playback_state
        if (
            playback is not None
            and playback.discontinuity_reason is not None
            and not self._suppress_safe_command_once
        ):
            # The source reports the discontinuity on the first decoded frame,
            # allowing externally initiated seeks and automatic loop wraps to
            # share the same fail-closed reset path.
            self._reset_temporal_state()
        skeleton = self._pose.estimate(frame)
        support_estimate = self._support_estimator.update(
            skeleton,
            timestamp_s=frame.timestamp_s,
        )
        raw_command = self._retargeter.retarget(
            skeleton,
            timestamp_s=frame.timestamp_s,
        )

        state = self._simulation.get_state()
        safe_command = None
        timestep_s = float(self._simulation.model.opt.timestep)
        balance_enabled = bool(self.config.free_base and self.config.balance_enabled)
        for _ in range(self._physics_step_count(frame.timestamp_s)):
            standing_command = self._balance_controller.update(
                raw_command,
                state,
                dt_s=timestep_s,
                squat_active=support_estimate.squat_active,
                squat_observation_fresh=support_estimate.squat_observation_fresh,
                squat_depth_ratio=support_estimate.squat_depth_ratio,
                allow_squat=(
                    self._support_machine.phase
                    is self._support_phase_type.DOUBLE_SUPPORT
                ),
            )
            if balance_enabled:
                prior = self._support_machine.last_diagnostics
                aborting = bool(
                    prior is not None
                    and prior.abort_reason is not None
                    and self._support_machine.phase
                    in {
                        self._support_phase_type.LOWER_SWING,
                        self._support_phase_type.CENTER_WEIGHT,
                    }
                )
                balance_diagnostics = self._balance_controller.last_diagnostics
                squat_interlocked = bool(
                    balance_diagnostics is not None
                    and not balance_diagnostics.squat_ready_for_support
                )
                observed = (
                    self._support_intent_type.DOUBLE_SUPPORT
                    if squat_interlocked or support_estimate.squat_active
                    else support_estimate.intent
                )
                requested = self._support_latch.update(
                    observed,
                    self._support_machine.phase,
                    stale=raw_command.stale or support_estimate.stale,
                    aborted=aborting,
                    timestamp_s=frame.timestamp_s,
                )
                safe_command = self._support_machine.update(
                    standing_command,
                    state,
                    dt_s=timestep_s,
                    intent=requested,
                )
            else:
                safe_command = standing_command
            self._simulation.apply_joint_command(safe_command)
            state = self._simulation.step()

        if safe_command is not None:
            self._last_safe_command = safe_command
        support_diagnostics = self._support_machine.last_diagnostics
        balance_diagnostics = self._balance_controller.last_diagnostics
        quaternion = state.base_orientation_wxyz
        upright = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
        tilt_rad = float(np.arccos(np.clip(upright, -1.0, 1.0)))
        telemetry = {
            # Preserve the complete immutable MuJoCo state for research
            # recording.  GUI cards may still consume the individual keys.
            "humanoid_state": state,
            "base_height_m": float(state.base_position_m[2]),
            "base_tilt_rad": tilt_rad,
            "simulation_time_s": float(state.simulation_time_s),
            "joint_positions_rad": state.joint_positions_rad,
            "joint_velocities_rad_s": state.joint_velocities_rad_s,
            "joint_lower_limits_rad": self._simulation.lower_limits_rad,
            "joint_upper_limits_rad": self._simulation.upper_limits_rad,
            "right_foot_force_n": float(state.right_foot_normal_force_n),
            "left_foot_force_n": float(state.left_foot_normal_force_n),
            # Contact-authoritative flags keep operator support visualization
            # from treating a swing foot as load-bearing.
            "right_foot_in_contact": bool(state.right_foot_in_contact),
            "left_foot_in_contact": bool(state.left_foot_in_contact),
            "support_intent": support_estimate.intent.value,
            "support_phase": self._support_machine.phase.value,
            "support_diagnostics": support_diagnostics,
            "balance_diagnostics": balance_diagnostics,
            "calibrating": bool(
                self._retargeter.is_calibrating
                or self._support_estimator.is_calibrating
            ),
            "calibration_progress": min(
                float(self._retargeter.calibration_progress),
                float(self._support_estimator.calibration_progress),
            ),
            "command_stale": bool(raw_command.stale),
            # These are measured pipeline mode facts, not GUI defaults.  Their
            # absence must be treated as unknown/fail-closed by robot output.
            "free_base_active": bool(self.config.free_base),
            "balance_active": balance_enabled,
        }
        finalized_safe_command = (
            None
            if safe_command is None or self._suppress_safe_command_once
            else finalize_safe_command(
                safe_command,
                free_base_active=bool(self.config.free_base),
                balance_active=balance_enabled,
            )
        )
        snapshot = PipelineSnapshot(
            self._sequence,
            frame.timestamp_s,
            SessionStatus.RUNNING,
            self.config.source,
            frame=frame,
            skeleton=skeleton,
            raw_command=raw_command,
            # Do not expose the unbalanced raw pose as a physical command on
            # the first frame before any servo tick has run.
            safe_command=finalized_safe_command,
            tracking_quality=self._tracking_quality(skeleton),
            telemetry=telemetry,
            playback=playback,
        )
        self._suppress_safe_command_once = False
        self._sequence += 1
        return snapshot

    def reset(self) -> None:
        self._require_started()
        self._simulation.reset()
        for resetter in (
            self._retargeter,
            self._balance_controller,
            self._support_machine,
            self._support_latch,
            self._support_estimator,
        ):
            resetter.reset()
        self._previous_frame_timestamp_s = None
        self._physics_accumulator_s = 0.0
        self._last_safe_command = None
        self._suppress_safe_command_once = False

    def calibrate(self, sample_count: int) -> None:
        self._require_started()
        self._retargeter.start_calibration(sample_count)
        # HumanSupportIntentEstimator owns its configured calibration window.
        self._support_estimator.start_calibration()
        self._support_latch.reset()

    def open_viewer(self) -> None:
        self._require_started()
        self._simulation.launch_viewer("visual")

    def close_viewer(self) -> None:
        self._require_started()
        self._simulation.close_viewer()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        for resource in (self._pose, self._source, self._simulation):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    errors.append(error)
        self._pose = None
        self._source = None
        self._simulation = None
        self._started = False
        if errors:
            raise errors[0]


def default_pipeline_factory(config: SessionConfig) -> DefaultTeleopPipeline:
    """Create an unopened concrete pipeline for ``TeleopSession``."""

    return DefaultTeleopPipeline(config)


__all__ = ["DefaultTeleopPipeline", "default_pipeline_factory"]
