"""Reproducible free-base robustness acceptance without camera/video inputs.

This evaluator deliberately exercises the controller and provisional dynamics
independently of the bundled replay videos.  Every trial loads the real
``scene_free.xml``, verifies that ``base_free`` is a free joint and that no
equality constraint exists, changes only in-memory physical parameters, and
then drives the robot exclusively through the production 20-motor target
interface.  Generalized coordinates and the floating base are never repaired
after the trial's one initialization reset.

The default acceptance contains nominal critical cases in both perturbation
directions plus a deterministic domain-randomized population.  Individual
failures remain in the JSON report.  Passing requires every critical case and
at least 95 percent of randomized trials; the randomized rate is accompanied
by a Wilson 95 percent confidence interval rather than presented as a bare
point estimate.

The model is explicitly provisional.  Passing this tool is evidence about a
bounded family around the current MuJoCo proxy, not evidence of physical-robot
safety or identification accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from numbers import Real
from pathlib import Path
import platform
import subprocess
from typing import Callable, Mapping, Sequence

import mujoco
import numpy as np

from robot_human_interface.control import (
    StandingBalanceController,
    SupportIntent,
    SupportPhase,
    SupportStateMachine,
    load_standing_balance_config,
    load_support_control_config,
)
from robot_human_interface.simulation import HumanoidSimulation
from robot_human_interface.skeleton import JOINT_NAMES, RobotJointCommand


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "freebase-robustness.json"
DEFAULT_RANDOMIZED_TRIALS = 20
DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE = 20
CANONICAL_SEED = 20260813
SINGLE_SUPPORT_HOLDOUT_SEED_OFFSET = 39_187
BALANCE_CONFIG = PROJECT_ROOT / "config" / "balance.yaml"
RUNTIME_INPUTS = (
    Path(__file__).resolve(),
    BALANCE_CONFIG,
    PROJECT_ROOT / "config" / "joints.yaml",
    PROJECT_ROOT / "models" / "humanoid" / "scene_free.xml",
    PROJECT_ROOT / "models" / "humanoid" / "robot.xml",
    PROJECT_ROOT / "src" / "robot_human_interface" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "control" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "control" / "standing.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "control" / "support.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "simulation" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "simulation" / "humanoid.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "simulation" / "types.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "__init__.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "filtering.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "transforms.py",
    PROJECT_ROOT / "src" / "robot_human_interface" / "skeleton" / "types.py",
)

ARM_INDICES = np.arange(0, 6, dtype=np.int64)
LEG_INDICES = np.arange(6, 18, dtype=np.int64)
RIGHT_LEG_INDICES = np.asarray((6, 8, 10, 12, 14, 16), dtype=np.int64)
LEFT_LEG_INDICES = np.asarray((7, 9, 11, 13, 15, 17), dtype=np.int64)
HEAD_INDICES = np.arange(18, 20, dtype=np.int64)
MOTION_GROUPS = {
    "arms": ARM_INDICES,
    "legs": LEG_INDICES,
    "right_leg": RIGHT_LEG_INDICES,
    "left_leg": LEFT_LEG_INDICES,
    "head": HEAD_INDICES,
}
LOADED_FOOT_THRESHOLD_N = 4.0
PUSH_DURATION_S = 0.15
PUSH_FORCE_N = 2.0
PUSH_TORQUE_N_M = 0.12
RANDOMIZATION_BOUNDS: dict[str, tuple[float, float]] = {
    "body_mass_and_inertia": (0.90, 1.10),
    "foot_ground_friction": (0.85, 1.15),
    "actuator_strength": (0.85, 1.15),
    "actuator_kp": (0.90, 1.10),
    "actuator_kv": (0.90, 1.10),
    "joint_damping": (0.85, 1.15),
}
CANONICAL_SCENE_PATH = "models/humanoid/scene_free.xml"
CANONICAL_ACTUATOR_COUNT = 20
CANONICAL_JOINT_COUNT = 21
CANONICAL_BASE_JOINT_ID = 0
CANONICAL_POSITIVE_MASS_BODY_NAMES = (
    "torso",
    "shoulder_rh",
    "elbow_rh",
    "wrist_rh",
    "shoulder_lh",
    "elbow_lh",
    "wrist_lh",
    "rotat_axis_rl",
    "motors_thigh_rl",
    "knee_rl",
    "shin_rl",
    "motors_feet_rl",
    "foot_rl",
    "rotat_axis_ll",
    "motors_thigh_ll",
    "knee_ll",
    "shin_ll",
    "motors_feet_ll",
    "foot_ll",
    "neck",
    "head",
)
CANONICAL_STATE_MUTATION_POLICY = (
    "one initialization reset after mj_setConst; thereafter motor targets only; "
    "finite torso xfrc_applied only in push scenarios; no qpos/base repair"
)


@dataclass(frozen=True, slots=True)
class RobustnessThresholds:
    """Predeclared per-trial and aggregate acceptance thresholds."""

    minimum_base_height_m: float = 0.80
    maximum_tilt_deg: float = 20.0
    maximum_horizontal_drift_m: float = 0.12
    # A single-support transfer must move the CoM from the bilateral midpoint
    # to a sole whose center is ~0.19 m away; absolute 0.12 m world drift is
    # therefore not a physically valid one-leg gate for this geometry.  Gate
    # the support-frame error and return-to-origin instead.
    # The contact sole half-width is 0.066 m.  Keep the projected CoM inside
    # that physical support polygon with a 1 mm geometric margin.
    maximum_single_support_com_to_stance_y_error_m: float = 0.065
    # The contact sole half-length is 0.105 m.  Keep 15 mm of fore-aft margin.
    maximum_single_support_com_to_stance_x_error_m: float = 0.090
    maximum_single_support_stance_foot_travel_m: float = 0.02
    maximum_single_support_final_base_drift_m: float = 0.06
    maximum_loaded_foot_slip_speed_m_s: float = 0.15
    maximum_loaded_foot_slip_distance_m: float = 0.075
    maximum_non_foot_ground_contacts: int = 0
    minimum_final_base_height_m: float = 0.88
    maximum_recovery_tilt_deg: float = 8.0
    maximum_recovery_base_speed_m_s: float = 0.06
    maximum_recovery_base_angular_speed_rad_s: float = 0.40
    maximum_recovery_joint_speed_rad_s: float = 0.40
    maximum_recovery_tracking_error_rad: float = 0.12
    maximum_recovery_loaded_foot_slip_speed_m_s: float = 0.03
    maximum_recovery_capture_point_error_m: float = 0.07
    minimum_recovery_foot_force_n: float = 4.0
    minimum_recovery_total_support_force_n: float = 20.0
    minimum_single_support_stance_force_n: float = 12.0
    minimum_single_support_stance_load_fraction: float = 0.65
    maximum_single_support_swing_load_fraction: float = 0.35
    maximum_swing_precontact_vertical_speed_m_s: float = 0.50
    maximum_swing_impact_force_n: float = 60.0
    maximum_swing_contact_impulse_n_s: float = 9.0
    recovery_observation_s: float = 1.0
    fall_height_m: float = 0.65
    fall_tilt_deg: float = 45.0
    required_randomized_pass_rate: float = 0.95
    minimum_randomized_wilson_lower_bound: float = 0.80


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """One controller-level motion or finite disturbance contract."""

    name: str
    kind: str
    active_duration_s: float
    recovery_duration_s: float
    critical: bool = True
    direction: int = 0
    support_intent: SupportIntent | None = None
    minimum_command_arm_span_deg: float = 0.0
    minimum_command_leg_span_deg: float = 0.0
    minimum_actual_arm_span_deg: float = 0.0
    minimum_actual_leg_span_deg: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.active_duration_s + self.recovery_duration_s


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("neutral_settle", "neutral", 4.0, 4.0),
    ScenarioSpec(
        "combined_upper_body_slow",
        "upper_body",
        10.0,
        5.0,
        minimum_command_arm_span_deg=20.0,
        minimum_actual_arm_span_deg=15.0,
    ),
    ScenarioSpec(
        "crouch_positive",
        "crouch",
        10.0,
        5.0,
        direction=1,
        minimum_command_leg_span_deg=2.0,
        minimum_actual_leg_span_deg=1.0,
    ),
    ScenarioSpec(
        "crouch_negative",
        "crouch",
        10.0,
        5.0,
        direction=-1,
        minimum_command_leg_span_deg=1.0,
        minimum_actual_leg_span_deg=0.5,
    ),
    ScenarioSpec("push_sagittal_positive", "push_sagittal", 6.0, 5.0, direction=1),
    ScenarioSpec("push_sagittal_negative", "push_sagittal", 6.0, 5.0, direction=-1),
    ScenarioSpec("push_lateral_positive", "push_lateral", 6.0, 5.0, direction=1),
    ScenarioSpec("push_lateral_negative", "push_lateral", 6.0, 5.0, direction=-1),
    ScenarioSpec(
        "right_single_support",
        "single_support",
        15.0,
        5.0,
        support_intent=SupportIntent.RIGHT_SWING,
        minimum_command_leg_span_deg=12.0,
        minimum_actual_leg_span_deg=8.0,
    ),
    ScenarioSpec(
        "left_single_support",
        "single_support",
        15.0,
        5.0,
        support_intent=SupportIntent.LEFT_SWING,
        minimum_command_leg_span_deg=12.0,
        minimum_actual_leg_span_deg=8.0,
    ),
)

SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """A scenario/seed pair and whether uncertainty is enabled."""

    scenario: ScenarioSpec
    seed: int
    randomized: bool
    critical: bool
    ordinal: int
    cohort: str = "diagnostic"

    @property
    def trial_id(self) -> str:
        mode = (
            "holdout"
            if self.cohort == "single_support_holdout"
            else "random"
            if self.randomized
            else "nominal"
        )
        return f"{mode}-{self.ordinal:03d}-{self.scenario.name}-seed-{self.seed}"


@dataclass(slots=True)
class _SwingContactEpisode:
    airborne_sample_available: bool = False
    last_airborne_vertical_velocity_m_s: float = 0.0
    contact_active: bool = False
    contact_impulse_n_s: float = 0.0


@dataclass(slots=True)
class _SwingContactTelemetry:
    maximum_impact_speed_m_s: float = 0.0
    maximum_precontact_vertical_speed_m_s: float = 0.0
    maximum_impact_force_n: float = 0.0
    maximum_contact_impulse_n_s: float = 0.0
    episodes: int = 0
    right: _SwingContactEpisode = field(default_factory=_SwingContactEpisode)
    left: _SwingContactEpisode = field(default_factory=_SwingContactEpisode)

    def update(
        self,
        *,
        phase: SupportPhase | None,
        active_intent: SupportIntent,
        right_force_n: float,
        left_force_n: float,
        right_velocity_m_s: np.ndarray,
        left_velocity_m_s: np.ndarray,
        dt_s: float,
    ) -> None:
        landing_admissible = phase in {
            SupportPhase.LIFT_SWING,
            SupportPhase.HOLD_SWING,
            SupportPhase.LOWER_SWING,
            SupportPhase.VERIFY_TOUCHDOWN,
        }
        # CENTER_WEIGHT is entered only after the controller's timed
        # touchdown confirmation.  From that phase onward both feet are
        # stance feet; keeping the former swing episode open would integrate
        # ordinary support load for seconds and report a fictitious impact.
        if not landing_admissible:
            for episode in (self.right, self.left):
                if episode.contact_active:
                    self._finish_episode(episode)
                episode.airborne_sample_available = False
            return
        swing_index = (
            0
            if active_intent is SupportIntent.RIGHT_SWING
            else 1
            if active_intent is SupportIntent.LEFT_SWING
            else None
        )
        samples = (
            (right_force_n, right_velocity_m_s, self.right),
            (left_force_n, left_velocity_m_s, self.left),
        )
        for index, (force_n, velocity, episode) in enumerate(samples):
            if index != swing_index:
                if episode.contact_active:
                    self._finish_episode(episode)
                episode.airborne_sample_available = False
                continue
            contacting = force_n >= 1.0
            if not contacting:
                # Keep a confirmed episode open across solver chatter, but
                # retain the final zero-force velocity for a later re-impact.
                # Otherwise a small early contact can mask the true landing
                # speed while its impulse continues accumulating.
                episode.airborne_sample_available = True
                episode.last_airborne_vertical_velocity_m_s = float(velocity[2])
                continue
            if not episode.contact_active:
                if not landing_admissible or not episode.airborne_sample_available:
                    continue
                episode.contact_active = True
                episode.contact_impulse_n_s = 0.0
                self.episodes += 1
                self.maximum_precontact_vertical_speed_m_s = max(
                    self.maximum_precontact_vertical_speed_m_s,
                    max(0.0, -episode.last_airborne_vertical_velocity_m_s),
                )
                episode.airborne_sample_available = False
            elif episode.airborne_sample_available:
                self.maximum_precontact_vertical_speed_m_s = max(
                    self.maximum_precontact_vertical_speed_m_s,
                    max(0.0, -episode.last_airborne_vertical_velocity_m_s),
                )
                episode.airborne_sample_available = False
            self.maximum_impact_speed_m_s = max(
                self.maximum_impact_speed_m_s,
                float(np.linalg.norm(velocity[:2])),
            )
            self.maximum_impact_force_n = max(
                self.maximum_impact_force_n, force_n
            )
            episode.contact_impulse_n_s += force_n * dt_s
            self.maximum_contact_impulse_n_s = max(
                self.maximum_contact_impulse_n_s, episode.contact_impulse_n_s
            )

    def _finish_episode(self, episode: _SwingContactEpisode) -> None:
        self.maximum_contact_impulse_n_s = max(
            self.maximum_contact_impulse_n_s, episode.contact_impulse_n_s
        )
        episode.contact_active = False
        episode.contact_impulse_n_s = 0.0


def _phase_aware_loaded_feet(
    phase: SupportPhase | None,
    active_intent: SupportIntent,
) -> tuple[bool, bool]:
    """Return feet that are semantically load-bearing for slip telemetry.

    Solver contact on a moving swing sole is landing/collision telemetry, not
    stance-foot slip.  The swing foot remains excluded through touchdown
    verification and is admitted only after the FSM enters CENTER_WEIGHT.
    """

    if phase in {
        SupportPhase.DOUBLE_SUPPORT,
        SupportPhase.SHIFT_WEIGHT,
        SupportPhase.VERIFY_STANCE,
        SupportPhase.CENTER_WEIGHT,
        None,
    }:
        return True, True
    if active_intent is SupportIntent.RIGHT_SWING:
        return False, True
    if active_intent is SupportIntent.LEFT_SWING:
        return True, False
    return True, True


def _phase_aware_loaded_mask(
    foot_forces_n: np.ndarray,
    phase: SupportPhase | None,
    active_intent: SupportIntent,
) -> np.ndarray:
    """Return the force-loaded feet that currently serve as stance feet."""

    forces = np.asarray(foot_forces_n, dtype=np.float64)
    if forces.shape != (2,):
        raise ValueError("foot_forces_n must contain right and left force")
    semantic_stance = np.asarray(
        _phase_aware_loaded_feet(phase, active_intent), dtype=np.bool_
    )
    return (forces >= LOADED_FOOT_THRESHOLD_N) & semantic_stance


def default_scenarios(*, include_one_leg: bool = True) -> tuple[ScenarioSpec, ...]:
    """Return the immutable default matrix, optionally omitting long support cases."""

    if include_one_leg:
        return SCENARIOS
    return tuple(scenario for scenario in SCENARIOS if scenario.kind != "single_support")


def build_trial_specs(
    *,
    randomized_trials: int = 20,
    single_support_holdout_trials_per_side: int = (
        DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
    ),
    seed: int = 20260813,
    scenarios: Sequence[ScenarioSpec] | None = None,
) -> list[TrialSpec]:
    """Build critical, broad-randomized, then single-support holdout trials."""

    if (
        isinstance(randomized_trials, bool)
        or not isinstance(randomized_trials, (int, np.integer))
        or randomized_trials < 0
    ):
        raise ValueError("randomized_trials must be a non-negative integer")
    if (
        isinstance(single_support_holdout_trials_per_side, bool)
        or not isinstance(
            single_support_holdout_trials_per_side, (int, np.integer)
        )
        or single_support_holdout_trials_per_side < 0
    ):
        raise ValueError(
            "single_support_holdout_trials_per_side must be a non-negative integer"
        )
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    selected = tuple(scenarios or default_scenarios())
    if not selected:
        raise ValueError("at least one scenario is required")
    if len({scenario.name for scenario in selected}) != len(selected):
        raise ValueError("scenario names must be unique")

    trials: list[TrialSpec] = []
    ordinal = 0
    for scenario_index, scenario in enumerate(selected):
        if not scenario.critical:
            continue
        trials.append(
            TrialSpec(
                scenario=scenario,
                seed=int(seed) + scenario_index,
                randomized=False,
                critical=True,
                ordinal=ordinal,
                cohort="critical",
            )
        )
        ordinal += 1
    for random_index in range(int(randomized_trials)):
        scenario = selected[random_index % len(selected)]
        trials.append(
            TrialSpec(
                scenario=scenario,
                seed=int(seed) + 10_000 + random_index,
                randomized=True,
                critical=False,
                ordinal=ordinal,
                cohort="broad_randomized",
            )
        )
        ordinal += 1
    holdout_scenarios = tuple(
        scenario
        for name in ("right_single_support", "left_single_support")
        for scenario in selected
        if scenario.name == name
    )
    for holdout_index in range(
        int(single_support_holdout_trials_per_side) * len(holdout_scenarios)
    ):
        scenario = holdout_scenarios[holdout_index % len(holdout_scenarios)]
        trials.append(
            TrialSpec(
                scenario=scenario,
                seed=int(seed) + SINGLE_SUPPORT_HOLDOUT_SEED_OFFSET + holdout_index,
                randomized=True,
                critical=False,
                ordinal=ordinal,
                cohort="single_support_holdout",
            )
        )
        ordinal += 1
    return trials


def _named_ids(model: mujoco.MjModel, object_type: mujoco.mjtObj) -> list[tuple[int, str]]:
    count_by_type = {
        mujoco.mjtObj.mjOBJ_BODY: model.nbody,
        mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
        mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
        mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
    }
    result: list[tuple[int, str]] = []
    for identifier in range(int(count_by_type[object_type])):
        name = mujoco.mj_id2name(model, object_type, identifier)
        if name is not None:
            result.append((identifier, name))
    return result


def _uniform_map(
    rng: np.random.Generator,
    names: Sequence[str],
    low: float,
    high: float,
    *,
    randomized: bool,
) -> dict[str, float]:
    if not randomized:
        return {name: 1.0 for name in names}
    values = rng.uniform(low, high, len(names))
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _randomization_bounds_payload() -> dict[str, list[float]]:
    return {
        name: [float(bounds[0]), float(bounds[1])]
        for name, bounds in RANDOMIZATION_BOUNDS.items()
    }


def _variation_realization_sha256(parameters: Mapping[str, object]) -> str:
    payload = {
        str(key): value
        for key, value in parameters.items()
        if key != "realization_sha256"
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_variation_from_names(
    body_names: Sequence[str],
    actuator_names: Sequence[str],
    *,
    seed: int,
    randomized: bool,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    result: dict[str, object] = {
        "seed": int(seed),
        "randomized": bool(randomized),
        "body_mass_inertia_factor": _uniform_map(
            rng,
            body_names,
            *RANDOMIZATION_BOUNDS["body_mass_and_inertia"],
            randomized=randomized,
        ),
        "foot_ground_friction_factor": (
            float(rng.uniform(*RANDOMIZATION_BOUNDS["foot_ground_friction"]))
            if randomized
            else 1.0
        ),
        "actuator_strength_factor": _uniform_map(
            rng,
            actuator_names,
            *RANDOMIZATION_BOUNDS["actuator_strength"],
            randomized=randomized,
        ),
        "actuator_kp_factor": _uniform_map(
            rng,
            actuator_names,
            *RANDOMIZATION_BOUNDS["actuator_kp"],
            randomized=randomized,
        ),
        "actuator_kv_factor": _uniform_map(
            rng,
            actuator_names,
            *RANDOMIZATION_BOUNDS["actuator_kv"],
            randomized=randomized,
        ),
        "joint_damping_factor": _uniform_map(
            rng,
            actuator_names,
            *RANDOMIZATION_BOUNDS["joint_damping"],
            randomized=randomized,
        ),
        "bounds": _randomization_bounds_payload(),
    }
    result["realization_sha256"] = _variation_realization_sha256(result)
    return result


def sample_model_variation(
    simulation: HumanoidSimulation,
    *,
    seed: int,
    randomized: bool,
) -> dict[str, object]:
    """Sample conservative, physically consistent provisional-model factors."""

    model = simulation.model
    bodies = [
        name
        for body_id, name in _named_ids(model, mujoco.mjtObj.mjOBJ_BODY)
        if body_id != 0 and float(model.body_mass[body_id]) > 0.0
    ]
    actuators = [name for _, name in _named_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR)]
    # Mass and diagonal inertia use the same factor for each body.  This
    # preserves every inertia triangle inequality and radius of gyration.
    return _sample_variation_from_names(
        bodies, actuators, seed=seed, randomized=randomized
    )


def _apply_model_variation(
    simulation: HumanoidSimulation,
    parameters: Mapping[str, object],
) -> None:
    """Apply one sampled realization and recompute MuJoCo model constants."""

    model = simulation.model
    data = simulation.data
    body_factors = parameters["body_mass_inertia_factor"]
    strength_factors = parameters["actuator_strength_factor"]
    kp_factors = parameters["actuator_kp_factor"]
    kv_factors = parameters["actuator_kv_factor"]
    damping_factors = parameters["joint_damping_factor"]
    if not all(
        isinstance(item, Mapping)
        for item in (
            body_factors,
            strength_factors,
            kp_factors,
            kv_factors,
            damping_factors,
        )
    ):
        raise ValueError("model variation factor groups must be mappings")

    for body_id, body_name in _named_ids(model, mujoco.mjtObj.mjOBJ_BODY):
        if body_id == 0 or float(model.body_mass[body_id]) <= 0.0:
            continue
        factor = float(body_factors[body_name])
        model.body_mass[body_id] *= factor
        model.body_inertia[body_id] *= factor
        inertia = np.asarray(model.body_inertia[body_id], dtype=np.float64)
        if np.any(inertia <= 0.0) or 2.0 * float(np.max(inertia)) > float(np.sum(inertia)) + 1e-12:
            raise ValueError(f"invalid inertia after variation for {body_name}")

    friction_factor = float(parameters["foot_ground_friction_factor"])
    for geom_name in ("ground", "foot_rl_geom", "foot_ll_geom"):
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
        if geom_id < 0:
            raise ValueError(f"required friction geom is missing: {geom_name}")
        model.geom_friction[geom_id] *= friction_factor

    for actuator_id, actuator_name in _named_ids(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR
    ):
        strength = float(strength_factors[actuator_name])
        kp = float(kp_factors[actuator_name])
        kv = float(kv_factors[actuator_name])
        damping = float(damping_factors[actuator_name])
        model.actuator_forcerange[actuator_id] *= strength
        # MuJoCo's position shortcut compiles to gain*ctrl - kp*q - kv*qdot.
        # Scale gain and the position bias together to retain that semantics.
        model.actuator_gainprm[actuator_id, 0] *= kp
        model.actuator_biasprm[actuator_id, 1] *= kp
        model.actuator_biasprm[actuator_id, 2] *= kv
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        dof_id = int(model.jnt_dofadr[joint_id])
        model.dof_damping[dof_id] *= damping

    mujoco.mj_setConst(model, data)
    # This is the trial's sole post-construction initialization.  No qpos or
    # floating-base write occurs after this reset; only motor targets and the
    # declared finite disturbance are used below.
    simulation.reset()


def _base_tilt_deg(quaternion_wxyz: Sequence[float]) -> float:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).copy()
    quaternion /= np.linalg.norm(quaternion)
    _, x, y, _ = quaternion
    upright = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(float(np.clip(upright, -1.0, 1.0))))


def _reference_positions(
    scenario: ScenarioSpec,
    home: np.ndarray,
    elapsed_s: float,
) -> np.ndarray:
    target = home.copy()
    if elapsed_s >= scenario.active_duration_s:
        return target
    if scenario.kind == "upper_body":
        phase = 2.0 * math.pi * 0.08 * elapsed_s
        target[0:2] += math.radians(38.0) * (0.5 + 0.5 * math.sin(phase))
        target[2] += math.radians(22.0) * math.sin(phase - 0.4)
        target[3] += math.radians(22.0) * math.sin(phase + 0.4)
        target[4:6] += math.radians(12.0) * math.sin(phase * 0.75)
        target[18] += math.radians(18.0) * math.sin(phase * 0.5)
        target[19] += math.radians(12.0) * math.sin(phase * 0.5 + 0.5)
    elif scenario.kind == "crouch":
        # A signed half-cosine excursion returns to neutral before recovery.
        phase = 2.0 * math.pi * min(elapsed_s, 8.0) / 8.0
        amplitude = scenario.direction * math.radians(12.0) * 0.5 * (1.0 - math.cos(phase))
        crouch = amplitude * np.asarray((0.7, 1.0, 0.3), dtype=np.float64)
        target[np.asarray((10, 12, 14))] += crouch
        target[np.asarray((11, 13, 15))] += crouch
    return target


def _support_intent(scenario: ScenarioSpec, elapsed_s: float) -> SupportIntent:
    if scenario.support_intent is None:
        return SupportIntent.DOUBLE_SUPPORT
    # Let the nominal robot establish bilateral contact before admission and
    # request return early enough to observe physical touchdown and centering.
    if 2.0 <= elapsed_s < 7.0:
        return scenario.support_intent
    return SupportIntent.DOUBLE_SUPPORT


def _apply_finite_perturbation(
    simulation: HumanoidSimulation,
    scenario: ScenarioSpec,
    elapsed_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    force = np.zeros(3, dtype=np.float64)
    torque = np.zeros(3, dtype=np.float64)
    pulse_active = 2.50 <= elapsed_s < 2.50 + PUSH_DURATION_S
    if pulse_active and scenario.kind == "push_sagittal":
        force[0] = PUSH_FORCE_N * scenario.direction
        torque[1] = -PUSH_TORQUE_N_M * scenario.direction
    elif pulse_active and scenario.kind == "push_lateral":
        force[1] = PUSH_FORCE_N * scenario.direction
        torque[0] = PUSH_TORQUE_N_M * scenario.direction
    simulation.data.xfrc_applied[:] = 0.0
    if np.any(force) or np.any(torque):
        torso_id = int(
            mujoco.mj_name2id(
                simulation.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )
        )
        simulation.data.xfrc_applied[torso_id, :3] = force
        simulation.data.xfrc_applied[torso_id, 3:] = torque
    return force, torque


def _motion_summary(samples: Sequence[np.ndarray]) -> dict[str, dict[str, object]]:
    values = np.asarray(samples, dtype=np.float64)
    result: dict[str, dict[str, object]] = {}
    for name, indices in MOTION_GROUPS.items():
        if values.size == 0:
            result[name] = {
                "maximum_span_deg": None,
                "joint_span_deg": [],
                "minimum_excursion_deg": [],
                "maximum_excursion_deg": [],
            }
            continue
        spans = np.degrees(np.ptp(values[:, indices], axis=0))
        excursions = np.degrees(values[:, indices] - values[0, indices])
        result[name] = {
            "maximum_span_deg": float(np.max(spans)),
            "joint_span_deg": [float(value) for value in spans],
            "minimum_excursion_deg": [
                float(value) for value in np.min(excursions, axis=0)
            ],
            "maximum_excursion_deg": [
                float(value) for value in np.max(excursions, axis=0)
            ],
        }
    return result


def _model_evidence(simulation: HumanoidSimulation) -> dict[str, object]:
    model = simulation.model
    base_joint_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    )
    joint_type = (
        int(model.jnt_type[base_joint_id]) if base_joint_id >= 0 else None
    )
    base_joint_type = (
        "free"
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE)
        else "missing" if joint_type is None else str(mujoco.mjtJoint(joint_type).name)
    )
    evidence = {
        "scene": "free",
        "scene_path": _project_path(simulation.model_path),
        "equality_constraint_count": int(model.neq),
        "base_joint_type": base_joint_type,
        "base_joint_id": base_joint_id,
        "total_mass_kg": float(np.sum(model.body_mass)),
        "timestep_s": float(model.opt.timestep),
        "actuator_count": int(model.nu),
        "joint_count": int(model.njnt),
        "positive_mass_body_names": [
            name
            for body_id, name in _named_ids(model, mujoco.mjtObj.mjOBJ_BODY)
            if body_id != 0 and float(model.body_mass[body_id]) > 0.0
        ],
        "actuator_names": [
            name for _, name in _named_ids(model, mujoco.mjtObj.mjOBJ_ACTUATOR)
        ],
        "state_mutation_policy": CANONICAL_STATE_MUTATION_POLICY,
    }
    if evidence["equality_constraint_count"] != 0:
        raise RuntimeError("robustness evaluation requires model.neq == 0")
    if evidence["base_joint_type"] != "free":
        raise RuntimeError("robustness evaluation requires a free base_free joint")
    if int(model.nu) != CANONICAL_ACTUATOR_COUNT:
        raise RuntimeError("robustness evaluation requires the canonical 20 actuators")
    return evidence


def run_trial(
    trial: TrialSpec,
    thresholds: RobustnessThresholds,
    *,
    duration_scale: float = 1.0,
) -> dict[str, object]:
    """Execute one real free-base MuJoCo trial and return raw evidence."""

    if not math.isfinite(duration_scale) or not 0.0 < duration_scale <= 1.0:
        raise ValueError("duration_scale must be within (0, 1]")
    with HumanoidSimulation("free") as simulation:
        parameters = sample_model_variation(
            simulation, seed=trial.seed, randomized=trial.randomized
        )
        _apply_model_variation(simulation, parameters)
        model = _model_evidence(simulation)
        balance = StandingBalanceController.from_simulation(
            simulation, load_standing_balance_config(BALANCE_CONFIG)
        )
        support = (
            SupportStateMachine.from_simulation(
                simulation, load_support_control_config(BALANCE_CONFIG)
            )
            if trial.scenario.kind == "single_support"
            else None
        )
        dt_s = float(simulation.model.opt.timestep)
        requested_duration_s = trial.scenario.duration_s * duration_scale
        step_count = max(1, int(math.ceil(requested_duration_s / dt_s)))
        # Scaling is used only by the explicit unit-test smoke.  It is recorded
        # and cannot satisfy the complete default suite's coverage contract.
        scenario_clock_scale = 1.0 / duration_scale

        initial = simulation.get_state()
        initial_base_xy = np.asarray(initial.base_position_m[:2], dtype=np.float64).copy()
        initial_right_foot_xy = np.asarray(
            initial.right_foot_position_m[:2], dtype=np.float64
        ).copy()
        initial_left_foot_xy = np.asarray(
            initial.left_foot_position_m[:2], dtype=np.float64
        ).copy()
        initial_right_foot_z = float(initial.right_foot_position_m[2])
        initial_left_foot_z = float(initial.left_foot_position_m[2])
        initial_right_over_left_z = initial_right_foot_z - initial_left_foot_z
        initial_left_over_right_z = -initial_right_over_left_z
        minimum_height = float(initial.base_position_m[2])
        maximum_tilt = _base_tilt_deg(initial.base_orientation_wxyz)
        maximum_drift = 0.0
        maximum_loaded_slip = 0.0
        slip_distances = np.zeros(2, dtype=np.float64)
        maximum_nonfoot = int(initial.non_foot_ground_contact_count)
        maximum_right_clearance = 0.0
        maximum_left_clearance = 0.0
        maximum_right_clearance_over_left = 0.0
        maximum_left_clearance_over_right = 0.0
        peak_force = np.zeros(3, dtype=np.float64)
        peak_torque = np.zeros(3, dtype=np.float64)
        applied_force_impulse = np.zeros(3, dtype=np.float64)
        applied_torque_impulse = np.zeros(3, dtype=np.float64)
        commanded_samples: list[np.ndarray] = []
        actual_samples: list[np.ndarray] = []
        support_phases: list[str] = []
        support_phase_events: list[dict[str, str]] = []
        support_abort_reasons: set[str] = set()
        support_frame_observations = 0
        maximum_support_frame_com_x_error = 0.0
        maximum_support_frame_com_y_error = 0.0
        maximum_support_frame_stance_foot_travel = 0.0
        swing_contact_telemetry = _SwingContactTelemetry()
        hold_stance_forces: list[float] = []
        hold_swing_forces: list[float] = []
        hold_stance_load_fractions: list[float] = []
        fell = False
        stopped_early = False

        recovery_start_s = max(
            0.0, requested_duration_s - thresholds.recovery_observation_s
        )
        recovery_heights: list[float] = []
        recovery_tilts: list[float] = []
        recovery_base_drifts: list[float] = []
        recovery_base_speeds: list[float] = []
        recovery_base_angular_speeds: list[float] = []
        recovery_joint_speeds: list[float] = []
        recovery_tracking_errors: list[float] = []
        recovery_slips: list[float] = []
        recovery_capture_point_errors: list[float] = []
        recovery_right_foot_forces: list[float] = []
        recovery_left_foot_forces: list[float] = []
        recovery_sample_times_s: list[float] = []

        state = initial
        final_command = simulation.home_positions_rad.copy()
        for step in range(step_count):
            elapsed_s = step * dt_s
            scenario_elapsed_s = elapsed_s * scenario_clock_scale
            target = _reference_positions(
                trial.scenario,
                simulation.home_positions_rad,
                scenario_elapsed_s,
            )
            reference = RobotJointCommand.humanoid(
                elapsed_s, target, confidence=1.0
            )
            balanced = balance.update(reference, state, dt_s=dt_s)
            if support is not None:
                final_command_object = support.update(
                    balanced,
                    state,
                    dt_s=dt_s,
                    intent=_support_intent(trial.scenario, scenario_elapsed_s),
                )
                diagnostics = support.last_diagnostics
                if diagnostics is not None:
                    phase = diagnostics.phase.value
                    active_intent = diagnostics.active_intent.value
                    if (
                        not support_phase_events
                        or support_phase_events[-1]["phase"] != phase
                        or support_phase_events[-1]["active_intent"]
                        != active_intent
                    ):
                        support_phase_events.append(
                            {
                                "phase": phase,
                                "active_intent": active_intent,
                                "requested_intent": diagnostics.requested_intent.value,
                            }
                        )
                    if not support_phases or support_phases[-1] != phase:
                        support_phases.append(phase)
                    if diagnostics.abort_reason:
                        support_abort_reasons.add(diagnostics.abort_reason)
                    if diagnostics.phase is SupportPhase.HOLD_SWING:
                        hold_stance_forces.append(float(diagnostics.stance_force_n))
                        hold_swing_forces.append(float(diagnostics.swing_force_n))
                        hold_stance_load_fractions.append(
                            float(diagnostics.stance_load_fraction)
                        )
            else:
                final_command_object = balanced
            final_command = np.asarray(
                final_command_object.positions_rad, dtype=np.float64
            ).copy()
            simulation.apply_joint_command(final_command_object)
            force, torque = _apply_finite_perturbation(
                simulation, trial.scenario, scenario_elapsed_s
            )
            peak_force = np.maximum(peak_force, np.abs(force))
            peak_torque = np.maximum(peak_torque, np.abs(torque))
            applied_force_impulse += force * dt_s
            applied_torque_impulse += torque * dt_s
            state = simulation.step()

            commanded_samples.append(final_command.copy())
            actual_samples.append(np.asarray(state.joint_positions_rad).copy())
            height = float(state.base_position_m[2])
            tilt = _base_tilt_deg(state.base_orientation_wxyz)
            drift = float(
                np.linalg.norm(np.asarray(state.base_position_m[:2]) - initial_base_xy)
            )
            foot_speeds = np.asarray(
                (
                    np.linalg.norm(state.right_foot_linear_velocity_m_s[:2]),
                    np.linalg.norm(state.left_foot_linear_velocity_m_s[:2]),
                ),
                dtype=np.float64,
            )
            foot_forces = np.asarray(
                (state.right_foot_normal_force_n, state.left_foot_normal_force_n),
                dtype=np.float64,
            )
            phase = diagnostics.phase if support is not None and diagnostics is not None else None
            active_intent = (
                diagnostics.active_intent
                if support is not None and diagnostics is not None
                else SupportIntent.DOUBLE_SUPPORT
            )
            stance_mask = _phase_aware_loaded_mask(
                foot_forces, phase, active_intent
            )
            loaded_slip = (
                float(np.max(foot_speeds[stance_mask]))
                if np.any(stance_mask)
                else 0.0
            )
            slip_distances[stance_mask] += foot_speeds[stance_mask] * dt_s
            swing_contact_telemetry.update(
                phase=phase,
                active_intent=active_intent,
                right_force_n=float(foot_forces[0]),
                left_force_n=float(foot_forces[1]),
                right_velocity_m_s=np.asarray(
                    state.right_foot_linear_velocity_m_s, dtype=np.float64
                ),
                left_velocity_m_s=np.asarray(
                    state.left_foot_linear_velocity_m_s, dtype=np.float64
                ),
                dt_s=dt_s,
            )

            verified_single_support = bool(
                support is not None
                and diagnostics is not None
                and phase is SupportPhase.HOLD_SWING
                and diagnostics.stance_force_n
                >= thresholds.minimum_single_support_stance_force_n
                and diagnostics.stance_load_fraction
                >= thresholds.minimum_single_support_stance_load_fraction
                and 1.0 - diagnostics.stance_load_fraction
                <= thresholds.maximum_single_support_swing_load_fraction
            )
            if verified_single_support:
                if active_intent is SupportIntent.RIGHT_SWING:
                    stance_position = np.asarray(state.left_foot_position_m)
                    stance_initial_xy = initial_left_foot_xy
                else:
                    stance_position = np.asarray(state.right_foot_position_m)
                    stance_initial_xy = initial_right_foot_xy
                support_frame_observations += 1
                maximum_support_frame_com_x_error = max(
                    maximum_support_frame_com_x_error,
                    abs(
                        float(
                            state.center_of_mass_position_m[0]
                            - stance_position[0]
                        )
                    ),
                )
                maximum_support_frame_com_y_error = max(
                    maximum_support_frame_com_y_error,
                    abs(
                        float(
                            state.center_of_mass_position_m[1]
                            - stance_position[1]
                        )
                    ),
                )
                maximum_support_frame_stance_foot_travel = max(
                    maximum_support_frame_stance_foot_travel,
                    float(
                        np.linalg.norm(stance_position[:2] - stance_initial_xy)
                    ),
                )

            minimum_height = min(minimum_height, height)
            maximum_tilt = max(maximum_tilt, tilt)
            maximum_drift = max(maximum_drift, drift)
            maximum_loaded_slip = max(maximum_loaded_slip, loaded_slip)
            maximum_nonfoot = max(
                maximum_nonfoot, int(state.non_foot_ground_contact_count)
            )
            maximum_right_clearance = max(
                maximum_right_clearance,
                float(state.right_foot_position_m[2]) - initial_right_foot_z,
            )
            maximum_left_clearance = max(
                maximum_left_clearance,
                float(state.left_foot_position_m[2]) - initial_left_foot_z,
            )
            maximum_right_clearance_over_left = max(
                maximum_right_clearance_over_left,
                float(state.right_foot_position_m[2] - state.left_foot_position_m[2])
                - initial_right_over_left_z,
            )
            maximum_left_clearance_over_right = max(
                maximum_left_clearance_over_right,
                float(state.left_foot_position_m[2] - state.right_foot_position_m[2])
                - initial_left_over_right_z,
            )

            sample_time_s = float(state.simulation_time_s)
            if (
                sample_time_s
                > recovery_start_s + 0.5 * dt_s
            ):
                recovery_sample_times_s.append(sample_time_s)
                recovery_heights.append(height)
                recovery_tilts.append(tilt)
                recovery_base_drifts.append(drift)
                recovery_base_speeds.append(
                    float(np.linalg.norm(state.base_linear_velocity_m_s))
                )
                recovery_base_angular_speeds.append(
                    float(np.linalg.norm(state.base_angular_velocity_rad_s))
                )
                recovery_joint_speeds.append(
                    float(np.max(np.abs(state.joint_velocities_rad_s)))
                )
                recovery_tracking_errors.append(
                    float(np.max(np.abs(state.joint_positions_rad - final_command)))
                )
                recovery_slips.append(loaded_slip)
                balance_diagnostics = balance.last_diagnostics
                recovery_capture_point_errors.append(
                    abs(float(balance_diagnostics.capture_point_error_x_m))
                    if balance_diagnostics is not None
                    else math.nan
                )
                recovery_right_foot_forces.append(
                    float(state.right_foot_normal_force_n)
                )
                recovery_left_foot_forces.append(
                    float(state.left_foot_normal_force_n)
                )

            fell = bool(
                height < thresholds.fall_height_m
                or tilt > thresholds.fall_tilt_deg
                or state.non_foot_ground_contact_count > 0
            )
            if fell:
                stopped_early = step + 1 < step_count
                break

        simulation.data.xfrc_applied[:] = 0.0
        completed_duration = bool(not stopped_early and len(actual_samples) == step_count)
        final_base_drift_m = float(
            np.linalg.norm(np.asarray(state.base_position_m[:2]) - initial_base_xy)
        )
        support_completed = True
        support_phase = None
        if support is not None:
            support_phase = support.phase.value
            support_completed = bool(
                support.phase is SupportPhase.DOUBLE_SUPPORT
                and support.active_intent is SupportIntent.DOUBLE_SUPPORT
                and not support_abort_reasons
            )

        expected_recovery_samples = int(
            math.ceil(thresholds.recovery_observation_s / dt_s - 1e-12)
        )
        recovery = {
            "observation_s": thresholds.recovery_observation_s,
            "window_start_time_s": recovery_start_s,
            "window_end_time_s": requested_duration_s,
            "sample_period_s": dt_s,
            "expected_samples": expected_recovery_samples,
            "samples": len(recovery_heights),
            "first_sample_time_s": (
                recovery_sample_times_s[0] if recovery_sample_times_s else None
            ),
            "last_sample_time_s": (
                recovery_sample_times_s[-1] if recovery_sample_times_s else None
            ),
            "maximum_sample_gap_s": (
                float(np.max(np.diff(recovery_sample_times_s)))
                if len(recovery_sample_times_s) > 1
                else None
            ),
            "minimum_base_height_m": (
                float(min(recovery_heights)) if recovery_heights else None
            ),
            "maximum_tilt_deg": (
                float(max(recovery_tilts)) if recovery_tilts else None
            ),
            "maximum_horizontal_drift_m": (
                float(max(recovery_base_drifts))
                if recovery_base_drifts
                else None
            ),
            "maximum_base_speed_m_s": (
                float(max(recovery_base_speeds)) if recovery_base_speeds else None
            ),
            "maximum_base_angular_speed_rad_s": (
                float(max(recovery_base_angular_speeds))
                if recovery_base_angular_speeds
                else None
            ),
            "maximum_joint_speed_rad_s": (
                float(max(recovery_joint_speeds)) if recovery_joint_speeds else None
            ),
            "maximum_tracking_error_rad": (
                float(max(recovery_tracking_errors))
                if recovery_tracking_errors
                else None
            ),
            "maximum_loaded_foot_slip_speed_m_s": (
                float(max(recovery_slips)) if recovery_slips else None
            ),
            "maximum_capture_point_error_m": (
                float(max(recovery_capture_point_errors))
                if recovery_capture_point_errors
                else None
            ),
            "minimum_right_foot_force_n": (
                float(min(recovery_right_foot_forces))
                if recovery_right_foot_forces
                else None
            ),
            "minimum_left_foot_force_n": (
                float(min(recovery_left_foot_forces))
                if recovery_left_foot_forces
                else None
            ),
            "minimum_total_support_force_n": (
                float(
                    min(
                        right + left
                        for right, left in zip(
                            recovery_right_foot_forces,
                            recovery_left_foot_forces,
                            strict=True,
                        )
                    )
                )
                if recovery_right_foot_forces
                else None
            ),
        }
        metrics = {
            "requested_duration_s": requested_duration_s,
            "simulated_duration_s": float(state.simulation_time_s),
            "duration_scale": duration_scale,
            "completed_duration": completed_duration,
            "fell": fell,
            "minimum_base_height_m": minimum_height,
            "maximum_tilt_deg": maximum_tilt,
            "maximum_horizontal_drift_m": maximum_drift,
            "final_horizontal_drift_m": final_base_drift_m,
            "maximum_loaded_foot_slip_speed_m_s": maximum_loaded_slip,
            "right_loaded_foot_slip_distance_m": float(slip_distances[0]),
            "left_loaded_foot_slip_distance_m": float(slip_distances[1]),
            "maximum_non_foot_ground_contacts": maximum_nonfoot,
            "final_base_height_m": float(state.base_position_m[2]),
            "final_tilt_deg": _base_tilt_deg(state.base_orientation_wxyz),
            "motion": {
                "motor_command": _motion_summary(commanded_samples),
                "actual_joint": _motion_summary(actual_samples),
            },
            "perturbation": {
                "kind": trial.scenario.kind if "push" in trial.scenario.kind else None,
                "direction": trial.scenario.direction,
                "peak_force_abs_n": [float(value) for value in peak_force],
                "peak_torque_abs_n_m": [float(value) for value in peak_torque],
                "signed_force_impulse_n_s": [
                    float(value) for value in applied_force_impulse
                ],
                "signed_torque_impulse_n_m_s": [
                    float(value) for value in applied_torque_impulse
                ],
            },
            "support": {
                "requested": (
                    trial.scenario.support_intent.value
                    if trial.scenario.support_intent is not None
                    else None
                ),
                "completed": support_completed,
                "final_phase": support_phase,
                "phase_sequence": support_phases,
                "phase_events": support_phase_events,
                "abort_reasons": sorted(support_abort_reasons),
                "maximum_right_foot_clearance_m": maximum_right_clearance,
                "maximum_left_foot_clearance_m": maximum_left_clearance,
                "maximum_right_foot_clearance_over_left_m": (
                    maximum_right_clearance_over_left
                ),
                "maximum_left_foot_clearance_over_right_m": (
                    maximum_left_clearance_over_right
                ),
                "hold_observations": len(hold_stance_forces),
                "minimum_hold_stance_force_n": (
                    float(min(hold_stance_forces)) if hold_stance_forces else None
                ),
                "maximum_hold_swing_force_n": (
                    float(max(hold_swing_forces)) if hold_swing_forces else None
                ),
                "minimum_hold_stance_load_fraction": (
                    float(min(hold_stance_load_fractions))
                    if hold_stance_load_fractions
                    else None
                ),
                "maximum_hold_swing_load_fraction": (
                    float(max(1.0 - value for value in hold_stance_load_fractions))
                    if hold_stance_load_fractions
                    else None
                ),
                "support_frame_observations": support_frame_observations,
                "maximum_support_frame_com_to_stance_x_error_m": (
                    maximum_support_frame_com_x_error
                ),
                "maximum_support_frame_com_to_stance_y_error_m": (
                    maximum_support_frame_com_y_error
                ),
                "maximum_support_frame_stance_foot_travel_m": (
                    maximum_support_frame_stance_foot_travel
                ),
                "swing_contact_episodes": swing_contact_telemetry.episodes,
                "maximum_swing_foot_impact_speed_m_s": (
                    swing_contact_telemetry.maximum_impact_speed_m_s
                ),
                "maximum_swing_foot_precontact_vertical_speed_m_s": (
                    swing_contact_telemetry.maximum_precontact_vertical_speed_m_s
                ),
                "maximum_swing_foot_impact_force_n": (
                    swing_contact_telemetry.maximum_impact_force_n
                ),
                "maximum_swing_foot_contact_impulse_n_s": (
                    swing_contact_telemetry.maximum_contact_impulse_n_s
                ),
            },
            "recovery": recovery,
        }
        return {
            "trial_id": trial.trial_id,
            "scenario": trial.scenario.name,
            "seed": trial.seed,
            "randomized": trial.randomized,
            "critical": trial.critical,
            "cohort": trial.cohort,
            "parameters": parameters,
            "model": model,
            "metrics": metrics,
        }


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _gate(name: str, passed: bool, observed: object, required: object) -> dict[str, object]:
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "required": required,
    }


def _nested_motion_span(
    metrics: Mapping[str, object], source: str, group: str
) -> object:
    motion = metrics.get("motion")
    if not isinstance(motion, Mapping):
        return None
    source_metrics = motion.get(source)
    if not isinstance(source_metrics, Mapping):
        return None
    group_metrics = source_metrics.get(group)
    if not isinstance(group_metrics, Mapping):
        return None
    return group_metrics.get("maximum_span_deg")


def _nested_motion_vector(
    metrics: Mapping[str, object], source: str, group: str, field: str
) -> list[float] | None:
    motion = metrics.get("motion")
    if not isinstance(motion, Mapping):
        return None
    source_metrics = motion.get(source)
    if not isinstance(source_metrics, Mapping):
        return None
    group_metrics = source_metrics.get(group)
    if not isinstance(group_metrics, Mapping):
        return None
    raw = group_metrics.get(field)
    if not isinstance(raw, (list, tuple, np.ndarray)):
        return None
    values = [float(value) for value in raw if _finite_number(value)]
    if len(values) != len(raw) or len(values) != len(MOTION_GROUPS[group]):
        return None
    return values


def _valid_support_phase_evidence(
    support: Mapping[str, object], expected_intent: SupportIntent
) -> tuple[bool, dict[str, object]]:
    raw_sequence = support.get("phase_sequence")
    raw_events = support.get("phase_events")
    sequence = (
        [str(value) for value in raw_sequence]
        if isinstance(raw_sequence, (list, tuple))
        else []
    )
    prefix = [
        SupportPhase.DOUBLE_SUPPORT.value,
        SupportPhase.SHIFT_WEIGHT.value,
        SupportPhase.VERIFY_STANCE.value,
        SupportPhase.LIFT_SWING.value,
        SupportPhase.HOLD_SWING.value,
        SupportPhase.LOWER_SWING.value,
    ]
    allowed_sequences = (
        prefix
        + [
            SupportPhase.CENTER_WEIGHT.value,
            SupportPhase.DOUBLE_SUPPORT.value,
        ],
        prefix
        + [
            SupportPhase.VERIFY_TOUCHDOWN.value,
            SupportPhase.CENTER_WEIGHT.value,
            SupportPhase.DOUBLE_SUPPORT.value,
        ],
    )
    events_valid = isinstance(raw_events, (list, tuple)) and len(raw_events) == len(
        sequence
    )
    event_phases: list[str] = []
    event_active_intents: list[str] = []
    event_requested_intents: list[str] = []
    if events_valid:
        for event in raw_events:
            if not isinstance(event, Mapping):
                events_valid = False
                break
            phase = event.get("phase")
            active_intent = event.get("active_intent")
            requested_intent = event.get("requested_intent")
            if not all(
                isinstance(value, str)
                for value in (phase, active_intent, requested_intent)
            ):
                events_valid = False
                break
            event_phases.append(str(phase))
            event_active_intents.append(str(active_intent))
            event_requested_intents.append(str(requested_intent))
    if events_valid:
        events_valid = event_phases == sequence
    if events_valid and sequence:
        expected_active = [
            (
                SupportIntent.DOUBLE_SUPPORT.value
                if index in (0, len(sequence) - 1)
                else expected_intent.value
            )
            for index in range(len(sequence))
        ]
        events_valid = event_active_intents == expected_active
    if events_valid and SupportPhase.HOLD_SWING.value in sequence:
        hold_index = sequence.index(SupportPhase.HOLD_SWING.value)
        events_valid = bool(
            event_requested_intents[0] == SupportIntent.DOUBLE_SUPPORT.value
            and all(
                value == expected_intent.value
                for value in event_requested_intents[1 : hold_index + 1]
            )
            and all(
                value
                in {expected_intent.value, SupportIntent.DOUBLE_SUPPORT.value}
                for value in event_requested_intents[hold_index + 1 :]
            )
        )
    elif events_valid:
        events_valid = False
    valid = sequence in allowed_sequences and events_valid
    return valid, {
        "phase_sequence": sequence,
        "active_intent_sequence": event_active_intents,
        "requested_intent_sequence": event_requested_intents,
    }


def _vector3(value: object) -> np.ndarray | None:
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if result.shape != (3,) or not np.isfinite(result).all():
        return None
    return result


def assess_trial(
    trial: TrialSpec,
    evaluation: Mapping[str, object],
    thresholds: RobustnessThresholds,
) -> dict[str, object]:
    """Attach explicit safety, recovery, excitation, and model gates."""

    metrics = evaluation.get("metrics")
    model = evaluation.get("model")
    if not isinstance(metrics, Mapping) or not isinstance(model, Mapping):
        return {
            **dict(evaluation),
            "trial_id": evaluation.get("trial_id", trial.trial_id),
            "scenario": trial.scenario.name,
            "seed": trial.seed,
            "randomized": trial.randomized,
            "critical": trial.critical,
            "cohort": trial.cohort,
            "gates": [_gate("evaluation_completed", False, False, True)],
            "passed": False,
        }

    gates: list[dict[str, object]] = []
    scenario = trial.scenario
    model_timestep = model.get("timestep_s")
    requested_duration = metrics.get("requested_duration_s")
    simulated_duration = metrics.get("simulated_duration_s")

    def maximum(name: str, limit: float | int) -> None:
        observed = metrics.get(name)
        gates.append(
            _gate(
                name,
                _finite_number(observed) and float(observed) <= float(limit),
                observed,
                {"maximum": limit},
            )
        )

    def minimum(name: str, limit: float) -> None:
        observed = metrics.get(name)
        gates.append(
            _gate(
                name,
                _finite_number(observed) and float(observed) >= limit,
                observed,
                {"minimum": limit},
            )
        )

    gates.extend(
        (
            _gate("scene", model.get("scene") == "free", model.get("scene"), "free"),
            _gate(
                "canonical_scene_path",
                model.get("scene_path") == CANONICAL_SCENE_PATH,
                model.get("scene_path"),
                CANONICAL_SCENE_PATH,
            ),
            _gate(
                "equality_constraint_count",
                model.get("equality_constraint_count") == 0,
                model.get("equality_constraint_count"),
                0,
            ),
            _gate(
                "base_joint_type",
                model.get("base_joint_type") == "free",
                model.get("base_joint_type"),
                "free",
            ),
            _gate(
                "canonical_base_joint_id",
                model.get("base_joint_id") == CANONICAL_BASE_JOINT_ID,
                model.get("base_joint_id"),
                CANONICAL_BASE_JOINT_ID,
            ),
            _gate(
                "canonical_actuator_count",
                model.get("actuator_count") == CANONICAL_ACTUATOR_COUNT,
                model.get("actuator_count"),
                CANONICAL_ACTUATOR_COUNT,
            ),
            _gate(
                "canonical_joint_count",
                model.get("joint_count") == CANONICAL_JOINT_COUNT,
                model.get("joint_count"),
                CANONICAL_JOINT_COUNT,
            ),
            _gate(
                "state_mutation_policy",
                model.get("state_mutation_policy")
                == CANONICAL_STATE_MUTATION_POLICY,
                model.get("state_mutation_policy"),
                CANONICAL_STATE_MUTATION_POLICY,
            ),
            _gate(
                "duration_scale",
                metrics.get("duration_scale") == 1.0,
                metrics.get("duration_scale"),
                1.0,
            ),
            _gate(
                "requested_duration_s",
                _finite_number(requested_duration)
                and abs(float(requested_duration) - scenario.duration_s) <= 1e-9,
                requested_duration,
                scenario.duration_s,
            ),
            _gate(
                "simulated_duration_s",
                _finite_number(simulated_duration)
                and _finite_number(model_timestep)
                and float(model_timestep) > 0.0
                and abs(float(simulated_duration) - scenario.duration_s)
                <= float(model_timestep) + 1e-9,
                simulated_duration,
                {"target": scenario.duration_s, "tolerance_timestep_s": model_timestep},
            ),
            _gate(
                "completed_duration",
                metrics.get("completed_duration") is True,
                metrics.get("completed_duration"),
                True,
            ),
            _gate("fell", metrics.get("fell") is False, metrics.get("fell"), False),
        )
    )

    parameters = evaluation.get("parameters")
    parameters_present = isinstance(parameters, Mapping)
    gates.append(
        _gate(
            "variation_parameters_present", parameters_present, parameters_present, True
        )
    )
    body_names_raw = model.get("positive_mass_body_names")
    actuator_names_raw = model.get("actuator_names")
    body_names = (
        [str(name) for name in body_names_raw]
        if isinstance(body_names_raw, (list, tuple))
        else []
    )
    actuator_names = (
        [str(name) for name in actuator_names_raw]
        if isinstance(actuator_names_raw, (list, tuple))
        else []
    )
    gates.extend(
        (
            _gate(
                "canonical_actuator_names",
                actuator_names == list(JOINT_NAMES),
                actuator_names,
                list(JOINT_NAMES),
            ),
            _gate(
                "canonical_positive_mass_bodies",
                body_names == list(CANONICAL_POSITIVE_MASS_BODY_NAMES),
                body_names,
                list(CANONICAL_POSITIVE_MASS_BODY_NAMES),
            ),
        )
    )
    if parameters_present:
        assert isinstance(parameters, Mapping)
        gates.extend(
            (
                _gate(
                    "variation_seed",
                    parameters.get("seed") == trial.seed,
                    parameters.get("seed"),
                    trial.seed,
                ),
                _gate(
                    "variation_randomized_flag",
                    parameters.get("randomized") is trial.randomized,
                    parameters.get("randomized"),
                    trial.randomized,
                ),
                _gate(
                    "variation_bounds",
                    parameters.get("bounds") == _randomization_bounds_payload(),
                    parameters.get("bounds"),
                    _randomization_bounds_payload(),
                ),
            )
        )
        factor_specs = (
            (
                "body_mass_inertia_factor",
                "body_mass_and_inertia",
                body_names,
            ),
            ("actuator_strength_factor", "actuator_strength", actuator_names),
            ("actuator_kp_factor", "actuator_kp", actuator_names),
            ("actuator_kv_factor", "actuator_kv", actuator_names),
            ("joint_damping_factor", "joint_damping", actuator_names),
        )
        factor_maps_valid = True
        factor_bounds_valid = True
        all_factors: list[float] = []
        factor_observations: dict[str, object] = {}
        for factor_name, bounds_name, expected_names in factor_specs:
            raw_factors = parameters.get(factor_name)
            factor_observations[factor_name] = raw_factors
            if not isinstance(raw_factors, Mapping) or set(raw_factors) != set(
                expected_names
            ):
                factor_maps_valid = False
                factor_bounds_valid = False
                continue
            values = list(raw_factors.values())
            if not values or not all(_finite_number(value) for value in values):
                factor_bounds_valid = False
                continue
            numeric = [float(value) for value in values]
            all_factors.extend(numeric)
            low, high = RANDOMIZATION_BOUNDS[bounds_name]
            if trial.randomized:
                factor_bounds_valid &= all(low <= value <= high for value in numeric)
            else:
                factor_bounds_valid &= all(abs(value - 1.0) <= 1e-12 for value in numeric)
        friction_factor = parameters.get("foot_ground_friction_factor")
        all_factors.append(float(friction_factor)) if _finite_number(
            friction_factor
        ) else None
        friction_low, friction_high = RANDOMIZATION_BOUNDS[
            "foot_ground_friction"
        ]
        friction_valid = bool(
            _finite_number(friction_factor)
            and (
                friction_low <= float(friction_factor) <= friction_high
                if trial.randomized
                else abs(float(friction_factor) - 1.0) <= 1e-12
            )
        )
        factor_bounds_valid &= friction_valid
        randomized_realization = bool(
            not trial.randomized
            or any(abs(value - 1.0) > 1e-12 for value in all_factors)
        )
        try:
            expected_realization_hash = _variation_realization_sha256(parameters)
        except (TypeError, ValueError):
            expected_realization_hash = None
        observed_realization_hash = parameters.get("realization_sha256")
        realization_hash_valid = bool(
            isinstance(observed_realization_hash, str)
            and len(observed_realization_hash) == 64
            and observed_realization_hash == expected_realization_hash
        )
        expected_seed_replay = _sample_variation_from_names(
            body_names,
            actuator_names,
            seed=trial.seed,
            randomized=trial.randomized,
        )
        seed_replay_valid = dict(parameters) == expected_seed_replay
        gates.extend(
            (
                _gate(
                    "variation_factor_maps",
                    factor_maps_valid,
                    factor_observations,
                    "exact sampled body/actuator name maps",
                ),
                _gate(
                    "variation_factor_bounds",
                    factor_bounds_valid,
                    all_factors,
                    (
                        _randomization_bounds_payload()
                        if trial.randomized
                        else "all factors exactly 1.0"
                    ),
                ),
                _gate(
                    "variation_randomized_realization",
                    randomized_realization,
                    randomized_realization,
                    True,
                ),
                _gate(
                    "variation_realization_sha256",
                    realization_hash_valid,
                    observed_realization_hash,
                    expected_realization_hash,
                ),
                _gate(
                    "variation_seed_replay",
                    seed_replay_valid,
                    parameters,
                    expected_seed_replay,
                ),
            )
        )

    minimum("minimum_base_height_m", thresholds.minimum_base_height_m)
    maximum("maximum_tilt_deg", thresholds.maximum_tilt_deg)
    if scenario.kind != "single_support":
        maximum("maximum_horizontal_drift_m", thresholds.maximum_horizontal_drift_m)
    else:
        gates.append(
            _gate(
                "maximum_horizontal_drift_m_descriptive",
                _finite_number(metrics.get("maximum_horizontal_drift_m")),
                metrics.get("maximum_horizontal_drift_m"),
                "finite descriptive value; gated in the stance-foot support frame",
            )
        )
    maximum(
        "maximum_loaded_foot_slip_speed_m_s",
        thresholds.maximum_loaded_foot_slip_speed_m_s,
    )
    maximum(
        "right_loaded_foot_slip_distance_m",
        thresholds.maximum_loaded_foot_slip_distance_m,
    )
    maximum(
        "left_loaded_foot_slip_distance_m",
        thresholds.maximum_loaded_foot_slip_distance_m,
    )
    maximum(
        "maximum_non_foot_ground_contacts",
        thresholds.maximum_non_foot_ground_contacts,
    )
    minimum("final_base_height_m", thresholds.minimum_final_base_height_m)

    recovery = metrics.get("recovery")
    if not isinstance(recovery, Mapping):
        gates.append(_gate("recovery_present", False, recovery, "mapping"))
    else:
        expected_recovery_samples = (
            int(
                math.ceil(
                    thresholds.recovery_observation_s / float(model_timestep)
                    - 1e-12
                )
            )
            if _finite_number(model_timestep) and float(model_timestep) > 0.0
            else None
        )
        expected_recovery_start = (
            scenario.duration_s - thresholds.recovery_observation_s
        )
        recovery_samples = recovery.get("samples")
        first_sample_time = recovery.get("first_sample_time_s")
        last_sample_time = recovery.get("last_sample_time_s")
        maximum_sample_gap = recovery.get("maximum_sample_gap_s")
        continuous_final_interval = bool(
            expected_recovery_samples is not None
            and isinstance(recovery_samples, int)
            and not isinstance(recovery_samples, bool)
            and recovery_samples == expected_recovery_samples
            and recovery.get("expected_samples") == expected_recovery_samples
            and _finite_number(recovery.get("observation_s"))
            and abs(
                float(recovery.get("observation_s"))
                - thresholds.recovery_observation_s
            )
            <= 1e-12
            and _finite_number(recovery.get("window_start_time_s"))
            and abs(
                float(recovery.get("window_start_time_s"))
                - expected_recovery_start
            )
            <= 1e-9
            and _finite_number(recovery.get("window_end_time_s"))
            and abs(
                float(recovery.get("window_end_time_s")) - scenario.duration_s
            )
            <= 1e-9
            and _finite_number(recovery.get("sample_period_s"))
            and _finite_number(model_timestep)
            and abs(
                float(recovery.get("sample_period_s")) - float(model_timestep)
            )
            <= 1e-12
            and _finite_number(first_sample_time)
            and float(first_sample_time) > expected_recovery_start
            and float(first_sample_time)
            <= expected_recovery_start + float(model_timestep) + 1e-9
            and _finite_number(last_sample_time)
            and abs(float(last_sample_time) - scenario.duration_s)
            <= float(model_timestep) + 1e-9
            and _finite_number(simulated_duration)
            and abs(float(last_sample_time) - float(simulated_duration)) <= 1e-9
            and _finite_number(maximum_sample_gap)
            and 0.0 < float(maximum_sample_gap) <= float(model_timestep) + 1e-9
        )
        gates.append(
            _gate(
                "recovery_continuous_final_interval",
                continuous_final_interval,
                {
                    "observation_s": recovery.get("observation_s"),
                    "window_start_time_s": recovery.get("window_start_time_s"),
                    "window_end_time_s": recovery.get("window_end_time_s"),
                    "sample_period_s": recovery.get("sample_period_s"),
                    "expected_samples": recovery.get("expected_samples"),
                    "samples": recovery_samples,
                    "first_sample_time_s": first_sample_time,
                    "last_sample_time_s": last_sample_time,
                    "maximum_sample_gap_s": maximum_sample_gap,
                },
                {
                    "window": [expected_recovery_start, scenario.duration_s],
                    "sample_period_s": model_timestep,
                    "samples": expected_recovery_samples,
                    "continuous_and_final": True,
                },
            )
        )
        recovery_contract = (
            (
                "samples",
                lambda value: isinstance(value, int)
                and not isinstance(value, bool)
                and value == expected_recovery_samples,
                {"exact": expected_recovery_samples},
            ),
            (
                "minimum_base_height_m",
                lambda value: _finite_number(value)
                and float(value) >= thresholds.minimum_final_base_height_m,
                {"minimum": thresholds.minimum_final_base_height_m},
            ),
            (
                "maximum_tilt_deg",
                lambda value: _finite_number(value)
                and float(value) <= thresholds.maximum_recovery_tilt_deg,
                {"maximum": thresholds.maximum_recovery_tilt_deg},
            ),
            (
                "maximum_base_speed_m_s",
                lambda value: _finite_number(value)
                and float(value) <= thresholds.maximum_recovery_base_speed_m_s,
                {"maximum": thresholds.maximum_recovery_base_speed_m_s},
            ),
            (
                "maximum_base_angular_speed_rad_s",
                lambda value: _finite_number(value)
                and float(value)
                <= thresholds.maximum_recovery_base_angular_speed_rad_s,
                {
                    "maximum": (
                        thresholds.maximum_recovery_base_angular_speed_rad_s
                    )
                },
            ),
            (
                "maximum_joint_speed_rad_s",
                lambda value: _finite_number(value)
                and float(value) <= thresholds.maximum_recovery_joint_speed_rad_s,
                {"maximum": thresholds.maximum_recovery_joint_speed_rad_s},
            ),
            (
                "maximum_tracking_error_rad",
                lambda value: _finite_number(value)
                and float(value) <= thresholds.maximum_recovery_tracking_error_rad,
                {"maximum": thresholds.maximum_recovery_tracking_error_rad},
            ),
            (
                "maximum_loaded_foot_slip_speed_m_s",
                lambda value: _finite_number(value)
                and float(value)
                <= thresholds.maximum_recovery_loaded_foot_slip_speed_m_s,
                {
                    "maximum": (
                        thresholds.maximum_recovery_loaded_foot_slip_speed_m_s
                    )
                },
            ),
            (
                "maximum_capture_point_error_m",
                lambda value: _finite_number(value)
                and float(value)
                <= thresholds.maximum_recovery_capture_point_error_m,
                {
                    "maximum": thresholds.maximum_recovery_capture_point_error_m
                },
            ),
            (
                "minimum_right_foot_force_n",
                lambda value: _finite_number(value)
                and float(value) >= thresholds.minimum_recovery_foot_force_n,
                {"minimum": thresholds.minimum_recovery_foot_force_n},
            ),
            (
                "minimum_left_foot_force_n",
                lambda value: _finite_number(value)
                and float(value) >= thresholds.minimum_recovery_foot_force_n,
                {"minimum": thresholds.minimum_recovery_foot_force_n},
            ),
            (
                "minimum_total_support_force_n",
                lambda value: _finite_number(value)
                and float(value)
                >= thresholds.minimum_recovery_total_support_force_n,
                {
                    "minimum": thresholds.minimum_recovery_total_support_force_n
                },
            ),
        )
        for name, predicate, required in recovery_contract:
            observed = recovery.get(name)
            gates.append(
                _gate(f"recovery_{name}", bool(predicate(observed)), observed, required)
            )

    scenario_leg_group = (
        "right_leg"
        if scenario.support_intent is SupportIntent.RIGHT_SWING
        else "left_leg"
        if scenario.support_intent is SupportIntent.LEFT_SWING
        else "legs"
    )
    for source, group, minimum_span in (
        ("motor_command", "arms", scenario.minimum_command_arm_span_deg),
        ("motor_command", scenario_leg_group, scenario.minimum_command_leg_span_deg),
        ("actual_joint", "arms", scenario.minimum_actual_arm_span_deg),
        ("actual_joint", scenario_leg_group, scenario.minimum_actual_leg_span_deg),
    ):
        if minimum_span <= 0.0:
            continue
        observed = _nested_motion_span(metrics, source, group)
        gates.append(
            _gate(
                f"{source}_{group}_motion",
                _finite_number(observed) and float(observed) >= minimum_span,
                observed,
                {"minimum_span_deg": minimum_span},
            )
        )

    def joint_coverage(
        source: str,
        group: str,
        local_indices: Sequence[int],
        minimum_span_deg: float,
        label: str,
    ) -> None:
        spans = _nested_motion_vector(metrics, source, group, "joint_span_deg")
        selected = (
            [spans[index] for index in local_indices]
            if spans is not None
            and all(0 <= index < len(spans) for index in local_indices)
            else None
        )
        gates.append(
            _gate(
                f"{source}_{label}_joint_coverage",
                selected is not None
                and all(value >= minimum_span_deg for value in selected),
                selected,
                {"each_joint_minimum_span_deg": minimum_span_deg},
            )
        )

    def directional_coverage(
        source: str,
        group: str,
        local_indices: Sequence[int],
        direction: int,
        minimum_excursion_deg: float,
        label: str,
    ) -> None:
        field = (
            "maximum_excursion_deg" if direction > 0 else "minimum_excursion_deg"
        )
        excursions = _nested_motion_vector(metrics, source, group, field)
        selected = (
            [excursions[index] for index in local_indices]
            if excursions is not None
            and all(0 <= index < len(excursions) for index in local_indices)
            else None
        )
        passed = bool(
            selected is not None
            and (
                all(value >= minimum_excursion_deg for value in selected)
                if direction > 0
                else all(value <= -minimum_excursion_deg for value in selected)
            )
        )
        gates.append(
            _gate(
                f"{source}_{label}_signed_excursion",
                passed,
                selected,
                {
                    "direction": direction,
                    "each_joint_minimum_abs_excursion_deg": minimum_excursion_deg,
                },
            )
        )

    if scenario.kind == "upper_body":
        joint_coverage("motor_command", "arms", range(6), 12.0, "arms")
        joint_coverage("actual_joint", "arms", range(6), 10.0, "arms")
        joint_coverage("motor_command", "head", range(2), 8.0, "head")
        joint_coverage("actual_joint", "head", range(2), 6.0, "head")
    elif scenario.kind == "crouch":
        # Canonical leg group 6..17: bilateral pitch joints 10..15 occupy 4..9.
        joint_coverage("motor_command", "legs", range(4, 10), 0.40, "crouch")
        joint_coverage("actual_joint", "legs", range(4, 10), 0.20, "crouch")
        # Knee/shin pitch joints 10..13 retain the signed crouch command.  The
        # ankle pitch channels also carry balance bias and are therefore span-
        # checked above but deliberately excluded from this sign assertion.
        directional_coverage(
            "motor_command", "legs", range(4, 8), scenario.direction, 0.75, "crouch"
        )
        directional_coverage(
            "actual_joint", "legs", range(4, 8), scenario.direction, 0.50, "crouch"
        )
    elif scenario.kind == "single_support":
        # In a side-specific six-joint group, lift pitch channels are 2..4.
        joint_coverage(
            "motor_command", scenario_leg_group, range(2, 5), 8.0, "swing_leg"
        )
        joint_coverage(
            "actual_joint", scenario_leg_group, range(2, 5), 6.0, "swing_leg"
        )

    perturbation = metrics.get("perturbation")
    if scenario.kind.startswith("push"):
        force_axis = 0 if scenario.kind == "push_sagittal" else 1
        torque_axis = 1 if scenario.kind == "push_sagittal" else 0
        torque_sign = (
            -scenario.direction
            if scenario.kind == "push_sagittal"
            else scenario.direction
        )
        expected_peak_force = np.zeros(3, dtype=np.float64)
        expected_peak_torque = np.zeros(3, dtype=np.float64)
        expected_force_impulse = np.zeros(3, dtype=np.float64)
        expected_torque_impulse = np.zeros(3, dtype=np.float64)
        expected_peak_force[force_axis] = PUSH_FORCE_N
        expected_peak_torque[torque_axis] = PUSH_TORQUE_N_M
        expected_force_impulse[force_axis] = (
            PUSH_FORCE_N * PUSH_DURATION_S * scenario.direction
        )
        expected_torque_impulse[torque_axis] = (
            PUSH_TORQUE_N_M * PUSH_DURATION_S * torque_sign
        )
        peak_force = (
            _vector3(perturbation.get("peak_force_abs_n"))
            if isinstance(perturbation, Mapping)
            else None
        )
        peak_torque = (
            _vector3(perturbation.get("peak_torque_abs_n_m"))
            if isinstance(perturbation, Mapping)
            else None
        )
        force_impulse = (
            _vector3(perturbation.get("signed_force_impulse_n_s"))
            if isinstance(perturbation, Mapping)
            else None
        )
        torque_impulse = (
            _vector3(perturbation.get("signed_torque_impulse_n_m_s"))
            if isinstance(perturbation, Mapping)
            else None
        )
        perturbation_ok = bool(
            isinstance(perturbation, Mapping)
            and perturbation.get("kind") == scenario.kind
            and perturbation.get("direction") == scenario.direction
            and peak_force is not None
            and peak_torque is not None
            and force_impulse is not None
            and torque_impulse is not None
            and np.allclose(peak_force, expected_peak_force, rtol=0.0, atol=1e-9)
            and np.allclose(peak_torque, expected_peak_torque, rtol=0.0, atol=1e-9)
            and np.allclose(
                force_impulse, expected_force_impulse, rtol=0.0, atol=1e-9
            )
            and np.allclose(
                torque_impulse, expected_torque_impulse, rtol=0.0, atol=1e-9
            )
        )
        gates.append(
            _gate(
                "finite_signed_perturbation",
                perturbation_ok,
                perturbation,
                {
                    "kind": scenario.kind,
                    "direction": scenario.direction,
                    "peak_force_abs_n": expected_peak_force.tolist(),
                    "peak_torque_abs_n_m": expected_peak_torque.tolist(),
                    "signed_force_impulse_n_s": expected_force_impulse.tolist(),
                    "signed_torque_impulse_n_m_s": expected_torque_impulse.tolist(),
                },
            )
        )
    else:
        zero_vectors = (
            _vector3(perturbation.get(name))
            if isinstance(perturbation, Mapping)
            else None
            for name in (
                "peak_force_abs_n",
                "peak_torque_abs_n_m",
                "signed_force_impulse_n_s",
                "signed_torque_impulse_n_m_s",
            )
        )
        vectors = list(zero_vectors)
        no_perturbation = bool(
            isinstance(perturbation, Mapping)
            and perturbation.get("kind") is None
            and all(
                vector is not None
                and np.allclose(vector, np.zeros(3), rtol=0.0, atol=1e-12)
                for vector in vectors
            )
        )
        gates.append(
            _gate(
                "no_undeclared_perturbation",
                no_perturbation,
                perturbation,
                "all finite force/torque peaks and impulses exactly zero",
            )
        )

    support = metrics.get("support")
    if scenario.kind == "single_support":
        expected_intent = scenario.support_intent
        assert expected_intent is not None
        support_values: Mapping[str, object] = (
            support if isinstance(support, Mapping) else {}
        )
        recovery_values: Mapping[str, object] = (
            recovery if isinstance(recovery, Mapping) else {}
        )
        support_completed = support_values.get("completed")
        support_abort_reasons = support_values.get("abort_reasons")
        phase_valid, phase_evidence = _valid_support_phase_evidence(
            support_values, expected_intent
        )
        gates.extend(
            (
                _gate(
                    "support_requested_side",
                    support_values.get("requested") == expected_intent.value,
                    support_values.get("requested"),
                    expected_intent.value,
                ),
                _gate(
                    "support_cycle_completed",
                    support_completed is True
                    and support_values.get("final_phase")
                    == SupportPhase.DOUBLE_SUPPORT.value,
                    {
                        "completed": support_completed,
                        "final_phase": (
                            support_values.get("final_phase")
                        ),
                    },
                    {
                        "completed": True,
                        "final_phase": SupportPhase.DOUBLE_SUPPORT.value,
                    },
                ),
                _gate(
                    "support_phase_sequence_and_side",
                    phase_valid,
                    phase_evidence,
                    (
                        "DOUBLE -> SHIFT -> VERIFY_STANCE -> LIFT -> HOLD -> "
                        "LOWER -> [VERIFY_TOUCHDOWN] -> CENTER -> DOUBLE, with "
                        f"active side {expected_intent.value}"
                    ),
                ),
                _gate(
                    "support_abort_reasons",
                    support_abort_reasons == [],
                    support_abort_reasons,
                    [],
                ),
            )
        )
        hold_observations = support_values.get("hold_observations")
        hold_stance_force = support_values.get("minimum_hold_stance_force_n")
        hold_stance_fraction = support_values.get(
            "minimum_hold_stance_load_fraction"
        )
        hold_swing_fraction = support_values.get(
            "maximum_hold_swing_load_fraction"
        )
        gates.extend(
            (
                _gate(
                    "single_support_hold_observations",
                    isinstance(hold_observations, int)
                    and not isinstance(hold_observations, bool)
                    and hold_observations > 0,
                    hold_observations,
                    {"minimum": 1},
                ),
                _gate(
                    "single_support_hold_stance_force",
                    _finite_number(hold_stance_force)
                    and float(hold_stance_force)
                    >= thresholds.minimum_single_support_stance_force_n,
                    hold_stance_force,
                    {
                        "minimum": thresholds.minimum_single_support_stance_force_n
                    },
                ),
                _gate(
                    "single_support_hold_stance_load_fraction",
                    _finite_number(hold_stance_fraction)
                    and float(hold_stance_fraction)
                    >= thresholds.minimum_single_support_stance_load_fraction,
                    hold_stance_fraction,
                    {
                        "minimum": (
                            thresholds.minimum_single_support_stance_load_fraction
                        )
                    },
                ),
                _gate(
                    "single_support_hold_swing_load_fraction",
                    _finite_number(hold_swing_fraction)
                    and float(hold_swing_fraction)
                    <= thresholds.maximum_single_support_swing_load_fraction,
                    hold_swing_fraction,
                    {
                        "maximum": (
                            thresholds.maximum_single_support_swing_load_fraction
                        )
                    },
                ),
                _gate(
                    "single_support_support_frame_observations",
                    isinstance(
                        support_values.get("support_frame_observations"), int
                    )
                    and not isinstance(
                        support_values.get("support_frame_observations"), bool
                    )
                    and int(support_values.get("support_frame_observations")) > 0,
                    support_values.get("support_frame_observations"),
                    {
                        "minimum": 1,
                        "phase": SupportPhase.HOLD_SWING.value,
                        "verified_stance_load": True,
                    },
                ),
                _gate(
                    "single_support_com_to_stance_x_error_m",
                    _finite_number(
                        support_values.get(
                            "maximum_support_frame_com_to_stance_x_error_m"
                        )
                    )
                    and float(
                        support_values.get(
                            "maximum_support_frame_com_to_stance_x_error_m"
                        )
                    )
                    <= thresholds.maximum_single_support_com_to_stance_x_error_m,
                    support_values.get(
                        "maximum_support_frame_com_to_stance_x_error_m"
                    ),
                    {
                        "maximum": (
                            thresholds.maximum_single_support_com_to_stance_x_error_m
                        ),
                        "phase": SupportPhase.HOLD_SWING.value,
                    },
                ),
                _gate(
                    "single_support_com_to_stance_y_error_m",
                    _finite_number(
                        support_values.get(
                            "maximum_support_frame_com_to_stance_y_error_m"
                        )
                    )
                    and float(
                        support_values.get(
                            "maximum_support_frame_com_to_stance_y_error_m"
                        )
                    )
                    <= thresholds.maximum_single_support_com_to_stance_y_error_m,
                    support_values.get(
                        "maximum_support_frame_com_to_stance_y_error_m"
                    ),
                    {
                        "maximum": (
                            thresholds.maximum_single_support_com_to_stance_y_error_m
                        ),
                        "phase": SupportPhase.HOLD_SWING.value,
                    },
                ),
                _gate(
                    "single_support_stance_foot_travel_m",
                    _finite_number(
                        support_values.get(
                            "maximum_support_frame_stance_foot_travel_m"
                        )
                    )
                    and float(
                        support_values.get(
                            "maximum_support_frame_stance_foot_travel_m"
                        )
                    )
                    <= thresholds.maximum_single_support_stance_foot_travel_m,
                    support_values.get(
                        "maximum_support_frame_stance_foot_travel_m"
                    ),
                    {
                        "maximum": (
                            thresholds.maximum_single_support_stance_foot_travel_m
                        ),
                        "phase": SupportPhase.HOLD_SWING.value,
                    },
                ),
                _gate(
                    "single_support_final_base_drift_m",
                    _finite_number(metrics.get("final_horizontal_drift_m"))
                    and float(metrics.get("final_horizontal_drift_m"))
                    <= thresholds.maximum_single_support_final_base_drift_m,
                    metrics.get("final_horizontal_drift_m"),
                    {
                        "maximum": (
                            thresholds.maximum_single_support_final_base_drift_m
                        )
                    },
                ),
                _gate(
                    "single_support_recovery_base_drift_m",
                    _finite_number(
                        recovery_values.get("maximum_horizontal_drift_m")
                    )
                    and float(recovery_values.get("maximum_horizontal_drift_m"))
                    <= thresholds.maximum_single_support_final_base_drift_m,
                    recovery_values.get("maximum_horizontal_drift_m"),
                    {
                        "maximum": (
                            thresholds.maximum_single_support_final_base_drift_m
                        ),
                        "window": "continuous final recovery interval",
                    },
                ),
            )
        )
        swing_contact_episodes = support_values.get("swing_contact_episodes")
        swing_precontact_vertical_speed = support_values.get(
            "maximum_swing_foot_precontact_vertical_speed_m_s"
        )
        swing_impact_force = support_values.get(
            "maximum_swing_foot_impact_force_n"
        )
        swing_contact_impulse = support_values.get(
            "maximum_swing_foot_contact_impulse_n_s"
        )
        gates.extend(
            (
                _gate(
                    "swing_contact_episode_observed",
                    isinstance(swing_contact_episodes, int)
                    and not isinstance(swing_contact_episodes, bool)
                    and swing_contact_episodes > 0,
                    swing_contact_episodes,
                    {"minimum": 1},
                ),
                _gate(
                    "swing_precontact_vertical_speed",
                    _finite_number(swing_precontact_vertical_speed)
                    and float(swing_precontact_vertical_speed)
                    <= thresholds.maximum_swing_precontact_vertical_speed_m_s,
                    swing_precontact_vertical_speed,
                    {
                        "maximum": (
                            thresholds.maximum_swing_precontact_vertical_speed_m_s
                        )
                    },
                ),
                _gate(
                    "swing_impact_force",
                    _finite_number(swing_impact_force)
                    and 1.0 <= float(swing_impact_force)
                    <= thresholds.maximum_swing_impact_force_n,
                    swing_impact_force,
                    {
                        "minimum": 1.0,
                        "maximum": thresholds.maximum_swing_impact_force_n,
                    },
                ),
                _gate(
                    "swing_contact_impulse",
                    _finite_number(swing_contact_impulse)
                    and 0.0 < float(swing_contact_impulse)
                    <= thresholds.maximum_swing_contact_impulse_n_s,
                    swing_contact_impulse,
                    {
                        "exclusive_minimum": 0.0,
                        "maximum": thresholds.maximum_swing_contact_impulse_n_s,
                    },
                ),
            )
        )
        intended_clearance_name = (
            "maximum_right_foot_clearance_over_left_m"
            if expected_intent is SupportIntent.RIGHT_SWING
            else "maximum_left_foot_clearance_over_right_m"
        )
        opposite_clearance_name = (
            "maximum_left_foot_clearance_over_right_m"
            if expected_intent is SupportIntent.RIGHT_SWING
            else "maximum_right_foot_clearance_over_left_m"
        )
        intended_clearance = (
            support_values.get(intended_clearance_name)
        )
        opposite_clearance = (
            support_values.get(opposite_clearance_name)
        )
        gates.extend(
            (
                _gate(
                    "swing_foot_clearance",
                    _finite_number(intended_clearance)
                    and float(intended_clearance) >= 0.02,
                    intended_clearance,
                    {
                        "minimum_relative_to_stance_foot_m": 0.02,
                        "field": intended_clearance_name,
                    },
                ),
                _gate(
                    "swing_foot_side_dominance",
                    _finite_number(intended_clearance)
                    and _finite_number(opposite_clearance)
                    and float(intended_clearance)
                    >= float(opposite_clearance) + 0.01,
                    {
                        "intended_m": intended_clearance,
                        "opposite_m": opposite_clearance,
                    },
                    {"minimum_intended_advantage_m": 0.01},
                ),
            )
        )

    numeric_payload = _numeric_leaves({"model": model, "metrics": metrics})
    gates.append(
        _gate(
            "finite_numeric_metrics",
            bool(numeric_payload) and all(math.isfinite(value) for value in numeric_payload),
            all(math.isfinite(value) for value in numeric_payload),
            True,
        )
    )
    return {
        **dict(evaluation),
        "trial_id": evaluation.get("trial_id", trial.trial_id),
        "scenario": trial.scenario.name,
        "seed": trial.seed,
        "randomized": trial.randomized,
        "critical": trial.critical,
        "cohort": trial.cohort,
        "scenario_contract": asdict(trial.scenario),
        "gates": gates,
        "passed": all(bool(gate["passed"]) for gate in gates),
    }


def _numeric_leaves(value: object) -> list[float]:
    if isinstance(value, Mapping):
        result: list[float] = []
        for child in value.values():
            result.extend(_numeric_leaves(child))
        return result
    if isinstance(value, (list, tuple, np.ndarray)):
        result = []
        for child in value:
            result.extend(_numeric_leaves(child))
        return result
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return [float(value)]
    return []


TrialEvaluator = Callable[[TrialSpec, RobustnessThresholds], dict[str, object]]


def evaluate_suite(
    trials: Sequence[TrialSpec],
    *,
    thresholds: RobustnessThresholds,
    evaluator: TrialEvaluator = run_trial,
) -> list[dict[str, object]]:
    """Run every trial while retaining exceptions as machine-readable failures."""

    results: list[dict[str, object]] = []
    for trial in trials:
        try:
            evaluation = evaluator(trial, thresholds)
        except Exception as error:
            evaluation = {
                "trial_id": trial.trial_id,
                "scenario": trial.scenario.name,
                "seed": trial.seed,
                "randomized": trial.randomized,
                "critical": trial.critical,
                "cohort": trial.cohort,
                "parameters": None,
                "model": None,
                "metrics": None,
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(assess_trial(trial, evaluation, thresholds))
    return results


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    """Return the two-sided Wilson binomial interval (95 percent by default)."""

    if isinstance(successes, bool) or isinstance(trials, bool):
        raise ValueError("successes and trials must be integers")
    if successes < 0 or trials < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if trials == 0:
        return None, None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"robustness runtime input is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_input_hashes() -> dict[str, str]:
    return {_project_path(path): _sha256(path) for path in RUNTIME_INPUTS}


_MINIMUM_ROBUSTNESS_THRESHOLD_FIELDS = (
    "minimum_base_height_m",
    "minimum_final_base_height_m",
    "minimum_recovery_foot_force_n",
    "minimum_recovery_total_support_force_n",
    "minimum_single_support_stance_force_n",
    "minimum_single_support_stance_load_fraction",
    "recovery_observation_s",
    "fall_height_m",
    "required_randomized_pass_rate",
    "minimum_randomized_wilson_lower_bound",
)
_MAXIMUM_ROBUSTNESS_THRESHOLD_FIELDS = (
    "maximum_tilt_deg",
    "maximum_horizontal_drift_m",
    "maximum_single_support_com_to_stance_x_error_m",
    "maximum_single_support_com_to_stance_y_error_m",
    "maximum_single_support_stance_foot_travel_m",
    "maximum_single_support_final_base_drift_m",
    "maximum_loaded_foot_slip_speed_m_s",
    "maximum_loaded_foot_slip_distance_m",
    "maximum_non_foot_ground_contacts",
    "maximum_recovery_tilt_deg",
    "maximum_recovery_base_speed_m_s",
    "maximum_recovery_base_angular_speed_rad_s",
    "maximum_recovery_joint_speed_rad_s",
    "maximum_recovery_tracking_error_rad",
    "maximum_recovery_loaded_foot_slip_speed_m_s",
    "maximum_recovery_capture_point_error_m",
    "maximum_single_support_swing_load_fraction",
    "maximum_swing_precontact_vertical_speed_m_s",
    "maximum_swing_impact_force_n",
    "maximum_swing_contact_impulse_n_s",
    "fall_tilt_deg",
)


def _threshold_policy_violations(
    thresholds: RobustnessThresholds,
) -> dict[str, dict[str, object]]:
    canonical = RobustnessThresholds()
    violations: dict[str, dict[str, object]] = {}
    for name in _MINIMUM_ROBUSTNESS_THRESHOLD_FIELDS:
        observed = getattr(thresholds, name)
        required = getattr(canonical, name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, Real)
            or not math.isfinite(float(observed))
        ):
            violations[name] = {
                "observed": observed,
                "canonical": required,
                "required_relation": "finite real >=",
            }
            continue
        if observed < required:
            violations[name] = {
                "observed": observed,
                "canonical": required,
                "required_relation": ">=",
            }
    for name in _MAXIMUM_ROBUSTNESS_THRESHOLD_FIELDS:
        observed = getattr(thresholds, name)
        required = getattr(canonical, name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, Real)
            or not math.isfinite(float(observed))
        ):
            violations[name] = {
                "observed": observed,
                "canonical": required,
                "required_relation": "finite real <=",
            }
            continue
        if observed > required:
            violations[name] = {
                "observed": observed,
                "canonical": required,
                "required_relation": "<=",
            }
    return violations


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


def _provenance() -> dict[str, object]:
    revision = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    diff = _git_output("diff", "--binary", "--no-ext-diff", "HEAD", text=False)
    versions: dict[str, str | None] = {}
    versions["python"] = platform.python_version()
    for package in ("mujoco", "numpy", "PyYAML"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision.strip() if isinstance(revision, str) else None,
        "git_worktree": {
            "dirty": None if not isinstance(status, str) else bool(status.strip()),
            "status_porcelain": (
                None if not isinstance(status, str) else status.strip().splitlines()
            ),
            "tracked_diff_sha256": (
                hashlib.sha256(diff).hexdigest() if isinstance(diff, bytes) else None
            ),
        },
        "versions": versions,
        "runtime_input_sha256": _runtime_input_hashes(),
    }


def _trial_assessment_passed(result: Mapping[str, object]) -> tuple[bool, bool]:
    """Return conservative pass and whether the summary matches its gates."""

    gates = result.get("gates")
    if not isinstance(gates, list) or not gates:
        return False, False
    gate_values: list[bool] = []
    for gate in gates:
        if not isinstance(gate, Mapping) or type(gate.get("passed")) is not bool:
            return False, False
        gate_values.append(bool(gate["passed"]))
    gates_passed = all(gate_values)
    reported = result.get("passed")
    consistent = type(reported) is bool and reported is gates_passed
    return bool(consistent and gates_passed), consistent


def build_report(
    results: Sequence[Mapping[str, object]],
    *,
    thresholds: RobustnessThresholds,
    selected_scenarios: Sequence[ScenarioSpec],
    randomized_trials_requested: int,
    seed: int,
    single_support_holdout_trials_per_side_requested: int = (
        DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
    ),
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build aggregate coverage/rate gates without hiding failed trials."""

    for name, value in (
        ("randomized_trials_requested", randomized_trials_requested),
        (
            "single_support_holdout_trials_per_side_requested",
            single_support_holdout_trials_per_side_requested,
        ),
        ("seed", seed),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise ValueError(f"{name} must be a non-negative integer")

    captured = dict(provenance or _provenance())
    final_hashes = _runtime_input_hashes()
    initial_hashes = captured.get("runtime_input_sha256")
    inputs_unchanged = initial_hashes == final_hashes
    selected_scenario_contracts_exact = tuple(selected_scenarios) == SCENARIOS
    threshold_violations = _threshold_policy_violations(thresholds)
    assessment_status = {
        str(result.get("trial_id")): _trial_assessment_passed(result)
        for result in results
    }
    inconsistent_assessments = sorted(
        trial_id
        for trial_id, (_, consistent) in assessment_status.items()
        if not consistent
    )

    def trial_passed(result: Mapping[str, object]) -> bool:
        return assessment_status.get(str(result.get("trial_id")), (False, False))[0]

    critical_results = [result for result in results if bool(result.get("critical"))]
    randomized_results = [
        result for result in results if result.get("cohort") == "broad_randomized"
    ]
    holdout_results = [
        result
        for result in results
        if result.get("cohort") == "single_support_holdout"
    ]
    randomized_successes = sum(trial_passed(result) for result in randomized_results)
    randomized_count = len(randomized_results)
    randomized_rate = (
        randomized_successes / randomized_count if randomized_count else None
    )
    ci_low, ci_high = wilson_interval(randomized_successes, randomized_count)
    # The acceptance contract is global and predeclared.  ``--scenario`` and
    # ``--no-one-leg`` are useful diagnostic selectors, but cannot turn an
    # incomplete matrix into a passing artifact.
    expected_critical = {scenario.name for scenario in SCENARIOS if scenario.critical}
    observed_critical = {
        str(result.get("scenario")) for result in critical_results
    }
    missing_critical = sorted(expected_critical - observed_critical)
    failed_critical = sorted(
        str(result.get("scenario"))
        for result in critical_results
        if not trial_passed(result)
    )
    expected_randomized_specs = [
        trial
        for trial in build_trial_specs(
            randomized_trials=DEFAULT_RANDOMIZED_TRIALS,
            single_support_holdout_trials_per_side=(
                DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
            ),
            seed=CANONICAL_SEED,
            scenarios=SCENARIOS,
        )
        if trial.cohort == "broad_randomized"
    ]
    expected_holdout_specs = [
        trial
        for trial in build_trial_specs(
            randomized_trials=DEFAULT_RANDOMIZED_TRIALS,
            single_support_holdout_trials_per_side=(
                DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
            ),
            seed=CANONICAL_SEED,
            scenarios=SCENARIOS,
        )
        if trial.cohort == "single_support_holdout"
    ]
    expected_all_specs = build_trial_specs(
        randomized_trials=DEFAULT_RANDOMIZED_TRIALS,
        single_support_holdout_trials_per_side=(
            DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
        ),
        seed=CANONICAL_SEED,
        scenarios=SCENARIOS,
    )
    expected_trial_signature = [
        {
            "trial_id": trial.trial_id,
            "scenario": trial.scenario.name,
            "seed": trial.seed,
            "randomized": trial.randomized,
            "critical": trial.critical,
            "cohort": trial.cohort,
            "scenario_contract": asdict(trial.scenario),
        }
        for trial in expected_all_specs
    ]
    observed_trial_signature = [
        {
            "trial_id": result.get("trial_id"),
            "scenario": result.get("scenario"),
            "seed": result.get("seed"),
            "randomized": result.get("randomized"),
            "critical": result.get("critical"),
            "cohort": result.get("cohort"),
            "scenario_contract": result.get("scenario_contract"),
        }
        for result in results
    ]
    expected_randomized_signature = [
        (trial.scenario.name, trial.seed) for trial in expected_randomized_specs
    ]
    observed_randomized_signature = [
        (str(result.get("scenario")), result.get("seed"))
        for result in randomized_results
    ]
    expected_holdout_signature = [
        (trial.scenario.name, trial.seed) for trial in expected_holdout_specs
    ]
    observed_holdout_signature = [
        (str(result.get("scenario")), result.get("seed"))
        for result in holdout_results
    ]
    expected_randomized_counts = Counter(
        scenario.name
        for scenario in SCENARIOS
        for _ in range(DEFAULT_RANDOMIZED_TRIALS // len(SCENARIOS))
    )
    observed_randomized_counts = Counter(
        str(result.get("scenario")) for result in randomized_results
    )
    expected_holdout_counts = Counter(
        trial.scenario.name for trial in expected_holdout_specs
    )
    observed_holdout_counts = Counter(
        str(result.get("scenario")) for result in holdout_results
    )
    canonical_randomized_count = bool(
        randomized_trials_requested == DEFAULT_RANDOMIZED_TRIALS
        and randomized_count == DEFAULT_RANDOMIZED_TRIALS
    )
    randomized_matrix_exact = bool(
        canonical_randomized_count
        and observed_randomized_signature == expected_randomized_signature
    )
    canonical_holdout_count = bool(
        single_support_holdout_trials_per_side_requested
        == DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
        and len(holdout_results)
        == 2 * DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
    )
    holdout_matrix_exact = bool(
        canonical_holdout_count
        and observed_holdout_signature == expected_holdout_signature
    )
    randomized_rate_passed = bool(
        canonical_randomized_count
        and randomized_rate is not None
        and randomized_rate + 1e-12 >= thresholds.required_randomized_pass_rate
    )
    randomized_wilson_passed = bool(
        canonical_randomized_count
        and ci_low is not None
        and ci_low + 1e-12
        >= thresholds.minimum_randomized_wilson_lower_bound
    )
    holdout_by_side: dict[str, dict[str, object]] = {}
    holdout_side_wilson_passed: dict[str, bool] = {}
    for scenario_name in ("right_single_support", "left_single_support"):
        side_results = [
            result
            for result in holdout_results
            if result.get("scenario") == scenario_name
        ]
        side_successes = sum(trial_passed(result) for result in side_results)
        side_count = len(side_results)
        side_ci_low, side_ci_high = wilson_interval(side_successes, side_count)
        holdout_by_side[scenario_name] = {
            "trials": side_count,
            "successes": side_successes,
            "pass_rate": side_successes / side_count if side_count else None,
            "wilson_95_ci": [side_ci_low, side_ci_high],
        }
        holdout_side_wilson_passed[scenario_name] = bool(
            canonical_holdout_count
            and side_count == DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
            and side_ci_low is not None
            and side_ci_low + 1e-12
            >= thresholds.minimum_randomized_wilson_lower_bound
        )
    critical_passed = bool(
        expected_critical
        and not missing_critical
        and not failed_critical
        and len(critical_results) == len(expected_critical)
    )

    aggregate_gates = [
        _gate(
            "canonical_seed",
            seed == CANONICAL_SEED,
            seed,
            CANONICAL_SEED,
        ),
        _gate(
            "exact_canonical_scenario_contracts",
            selected_scenario_contracts_exact,
            [asdict(scenario) for scenario in selected_scenarios],
            [asdict(scenario) for scenario in SCENARIOS],
        ),
        _gate(
            "exact_ordered_trial_contracts",
            observed_trial_signature == expected_trial_signature,
            observed_trial_signature,
            expected_trial_signature,
        ),
        _gate(
            "trial_assessments_consistent",
            not inconsistent_assessments,
            inconsistent_assessments,
            [],
        ),
        _gate(
            "canonical_or_stricter_thresholds",
            not threshold_violations,
            threshold_violations,
            {
                "minimum_thresholds": ">= canonical defaults",
                "maximum_thresholds": "<= canonical defaults",
            },
        ),
        _gate(
            "critical_scenarios_complete",
            not missing_critical and len(critical_results) == len(expected_critical),
            {"observed": sorted(observed_critical), "missing": missing_critical},
            sorted(expected_critical),
        ),
        _gate(
            "all_critical_trials_passed",
            critical_passed,
            {"failed": failed_critical},
            {"failed": []},
        ),
        _gate(
            "randomized_trial_count",
            canonical_randomized_count,
            {
                "requested": randomized_trials_requested,
                "evaluated": randomized_count,
            },
            {"requested": DEFAULT_RANDOMIZED_TRIALS, "evaluated": DEFAULT_RANDOMIZED_TRIALS},
        ),
        _gate(
            "randomized_scenario_coverage",
            observed_randomized_counts == expected_randomized_counts,
            dict(sorted(observed_randomized_counts.items())),
            dict(sorted(expected_randomized_counts.items())),
        ),
        _gate(
            "randomized_matrix_exact",
            randomized_matrix_exact,
            observed_randomized_signature,
            expected_randomized_signature,
        ),
        _gate(
            "randomized_pass_rate",
            randomized_rate_passed,
            randomized_rate,
            {"minimum": thresholds.required_randomized_pass_rate},
        ),
        _gate(
            "randomized_wilson_95_lower_bound",
            randomized_wilson_passed,
            ci_low,
            {"minimum": thresholds.minimum_randomized_wilson_lower_bound},
        ),
        _gate(
            "single_support_holdout_trial_count",
            canonical_holdout_count,
            {
                "requested_per_side": (
                    single_support_holdout_trials_per_side_requested
                ),
                "evaluated": len(holdout_results),
            },
            {
                "requested_per_side": (
                    DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
                ),
                "evaluated": 2 * DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE,
            },
        ),
        _gate(
            "single_support_holdout_side_coverage",
            observed_holdout_counts == expected_holdout_counts,
            dict(sorted(observed_holdout_counts.items())),
            dict(sorted(expected_holdout_counts.items())),
        ),
        _gate(
            "single_support_holdout_matrix_exact",
            holdout_matrix_exact,
            observed_holdout_signature,
            expected_holdout_signature,
        ),
        *(
            _gate(
                f"{scenario_name}_holdout_wilson_95_lower_bound",
                holdout_side_wilson_passed[scenario_name],
                holdout_by_side[scenario_name]["wilson_95_ci"][0],
                {"minimum": thresholds.minimum_randomized_wilson_lower_bound},
            )
            for scenario_name in (
                "right_single_support",
                "left_single_support",
            )
        ),
        _gate(
            "runtime_inputs_unchanged",
            inputs_unchanged,
            inputs_unchanged,
            True,
        ),
    ]
    return {
        "schema_version": 3,
        "description": (
            "Controller-level free-base domain randomization and finite-perturbation "
            "acceptance on the provisional MuJoCo proxy; no camera/video input."
        ),
        "scientific_scope": (
            "Simulation robustness around provisional parameters only; not physical "
            "robot validation, hardware closed-loop evidence, or a safety certificate."
        ),
        "command": "python tools/evaluate_freebase_robustness.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_started_at_utc": captured.get("captured_at_utc"),
        "git_revision": captured.get("git_revision"),
        "git_worktree": captured.get("git_worktree"),
        "versions": captured.get("versions"),
        "runtime_input_sha256": initial_hashes,
        "runtime_input_sha256_at_completion": final_hashes,
        "runtime_inputs_unchanged_during_run": inputs_unchanged,
        "configuration": {
            "scene": "free",
            "headless": True,
            "motor_targets_only": True,
            "qpos_or_base_repair_after_initial_reset": False,
            "seed": seed,
            "randomized_trials_requested": randomized_trials_requested,
            "canonical_randomized_trials_required": DEFAULT_RANDOMIZED_TRIALS,
            "single_support_holdout_trials_per_side_requested": (
                single_support_holdout_trials_per_side_requested
            ),
            "canonical_single_support_holdout_trials_per_side_required": (
                DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE
            ),
            "single_support_holdout_seed_offset": (
                SINGLE_SUPPORT_HOLDOUT_SEED_OFFSET
            ),
            "selected_scenarios": [scenario.name for scenario in selected_scenarios],
            "thresholds": asdict(thresholds),
            "randomization_bounds": {
                "body_mass_and_inertia": "per body, shared factor, +/-10%",
                "foot_and_ground_friction": "+/-15%",
                "actuator_strength": "per actuator, +/-15%",
                "actuator_kp_and_kv": "per actuator, +/-10%",
                "joint_damping": "per actuated dof, +/-15%",
            },
        },
        "aggregate": {
            "critical": {
                "expected": len(expected_critical),
                "passed": len(expected_critical) - len(failed_critical) - len(missing_critical),
                "failed_scenarios": failed_critical,
                "missing_scenarios": missing_critical,
            },
            "randomized": {
                "trials": randomized_count,
                "successes": randomized_successes,
                "pass_rate": randomized_rate,
                "wilson_95_ci": [ci_low, ci_high],
                "scenario_counts": dict(sorted(observed_randomized_counts.items())),
            },
            "single_support_holdout": {
                "trials": len(holdout_results),
                "successes": sum(
                    trial_passed(result) for result in holdout_results
                ),
                "scenario_counts": dict(sorted(observed_holdout_counts.items())),
                "by_side": holdout_by_side,
            },
            "gates": aggregate_gates,
        },
        "overall_passed": all(bool(gate["passed"]) for gate in aggregate_gates),
        "trials": list(results),
    }


def _sanitize_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_sanitize_json(child) for child in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, SupportIntent):
        return value.value
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    parser.add_argument(
        "--randomized-trials", type=int, default=DEFAULT_RANDOMIZED_TRIALS
    )
    parser.add_argument(
        "--single-support-holdout-trials-per-side",
        type=int,
        default=DEFAULT_SINGLE_SUPPORT_HOLDOUT_TRIALS_PER_SIDE,
        help=(
            "Independent randomized one-leg trials per side. Any noncanonical "
            "count remains diagnostic and cannot pass acceptance."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(SCENARIO_BY_NAME),
        default=[],
        help="Run only a named scenario; repeat to select several. Incomplete matrices fail.",
    )
    parser.add_argument(
        "--one-leg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include both long contact-gated one-leg cases (default: enabled).",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    evaluator: TrialEvaluator = run_trial,
) -> int:
    args = build_parser().parse_args(argv)
    if args.randomized_trials < 0:
        raise ValueError("--randomized-trials must be non-negative")
    if args.single_support_holdout_trials_per_side < 0:
        raise ValueError(
            "--single-support-holdout-trials-per-side must be non-negative"
        )
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    available = default_scenarios(include_one_leg=args.one_leg)
    selected = (
        tuple(SCENARIO_BY_NAME[name] for name in args.scenario)
        if args.scenario
        else available
    )
    unavailable = [scenario.name for scenario in selected if scenario not in available]
    if unavailable:
        raise ValueError(
            "one-leg scenarios selected while --no-one-leg is active: "
            f"{unavailable}"
        )
    thresholds = RobustnessThresholds()
    provenance = _provenance()
    trials = build_trial_specs(
        randomized_trials=args.randomized_trials,
        single_support_holdout_trials_per_side=(
            args.single_support_holdout_trials_per_side
        ),
        seed=args.seed,
        scenarios=selected,
    )
    results = evaluate_suite(trials, thresholds=thresholds, evaluator=evaluator)
    report = build_report(
        results,
        thresholds=thresholds,
        selected_scenarios=selected,
        randomized_trials_requested=args.randomized_trials,
        single_support_holdout_trials_per_side_requested=(
            args.single_support_holdout_trials_per_side
        ),
        seed=args.seed,
        provenance=provenance,
    )
    sanitized = _sanitize_json(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(sanitized, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sanitized, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["overall_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
