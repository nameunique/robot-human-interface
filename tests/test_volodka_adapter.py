"""The teleoperation adapter must not silently alter the imported physics."""

from pathlib import Path
import json
import runpy
import shutil

import mujoco
import numpy as np
import pytest
import yaml

from robot_human_interface.control import StandingBalanceController, load_standing_balance_config
from robot_human_interface.simulation import HumanoidSimulation
from robot_human_interface.skeleton import JOINT_NAMES, RobotJointCommand


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "volodka"
JOINTS = ROOT / "config" / "joints_volodka.yaml"
MAPPING = json.loads((MODEL / "joint_mapping.json").read_text(encoding="utf-8"))["joints"]


def simulation(scene="free"):
    return HumanoidSimulation(MODEL / f"scene_{scene}.xml", joint_config_path=JOINTS)


def test_runtime_adapter_rebuilds_from_committed_inputs_without_upstream(tmp_path):
    namespace = runpy.run_path(str(ROOT / "tools" / "build_volodka_adapter.py"))
    build = namespace["build"]
    model_dir = tmp_path / "models" / "volodka"
    model_dir.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    shutil.copy2(MODEL / "source_scene.xml", model_dir)
    shutil.copy2(ROOT / "config" / "joints.yaml", tmp_path / "config")
    build.__globals__["ROOT"] = tmp_path
    build.__globals__["MODEL"] = model_dir
    build()
    for relative in (
        "models/volodka/scene_free.xml", "models/volodka/scene_fixed.xml",
        "models/volodka/joint_mapping.json", "config/joints_volodka.yaml",
    ):
        assert (tmp_path / relative).read_text(encoding="utf-8") == (ROOT / relative).read_text(encoding="utf-8")


def test_adapter_preserves_full_body_physics_and_forward_kinematics():
    native = mujoco.MjModel.from_xml_path(str(MODEL / "source_scene.xml"))
    native_data = mujoco.MjData(native)
    with simulation() as adapted:
        assert adapted.joint_names == JOINT_NAMES
        assert adapted.model.nu == 20
        assert sum(adapted.model.body_mass) == pytest.approx(3.0942)
        assert adapted.model.npair == native.npair == 23
        assert adapted.model.opt.timestep == native.opt.timestep
        for attribute in (
            "geom_size", "geom_pos", "geom_quat", "geom_friction",
            "geom_contype", "geom_conaffinity", "pair_solref", "pair_solimp",
            "pair_friction", "pair_dim", "mesh_vert",
        ):
            np.testing.assert_array_equal(
                getattr(adapted.model, attribute), getattr(native, attribute)
            )
        assert np.max(np.abs(adapted.model.actuator_forcerange)) == pytest.approx(1.5)
        np.testing.assert_allclose(adapted.model.actuator_gear[:, 0], 1)
        pairs = [("base_link", "torso")] + [(row["native_body"], row["api_name"]) for row in MAPPING]
        for original, api in pairs:
            a, b = native.body(original).id, adapted.model.body(api).id
            for attribute in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
                np.testing.assert_allclose(getattr(native, attribute)[a], getattr(adapted.model, attribute)[b], atol=1e-12)
        for offset in (np.zeros(20), np.random.default_rng(13).uniform(-.05, .05, 20)):
            adapted.reset(adapted.home_positions_rad + offset)
            for i, row in enumerate(MAPPING):
                native_data.qpos[native.jnt_qposadr[native.joint(row["native_joint"]).id]] = row["sign"] * offset[i]
            mujoco.mj_forward(native, native_data)
            for original, api in pairs:
                np.testing.assert_allclose(native_data.body(original).xpos, adapted.data.body(api).xpos, atol=1e-12)
                np.testing.assert_allclose(native_data.body(original).xmat, adapted.data.body(api).xmat, atol=1e-12)


def test_adapter_preserves_actuator_dynamics_not_only_visual_pose():
    native = mujoco.MjModel.from_xml_path(str(MODEL / "source_scene.xml"))
    native_data = mujoco.MjData(native)
    with simulation() as adapted:
        offset = np.zeros(20)
        offset[:2] = .12
        adapted.set_joint_targets(adapted.home_positions_rad + offset)
        for index, row in enumerate(MAPPING):
            actuator = native.actuator(f"reference_position__{row['native_joint']}").id
            native_data.ctrl[actuator] = row["sign"] * offset[index]
        for _ in range(250):
            mujoco.mj_step(native, native_data)
            adapted.step()
        np.testing.assert_allclose(native_data.body("base_link").xpos, adapted.data.body("torso").xpos, atol=1e-8)
        for index, row in enumerate(MAPPING):
            q = native_data.qpos[native.jnt_qposadr[native.joint(row["native_joint"]).id]]
            api_q = adapted.get_state().joint_positions_rad[index]
            assert q == pytest.approx(row["sign"] * (api_q - row["api_home_rad"]), abs=1e-8)


@pytest.mark.parametrize("scene", ["fixed", "free"])
def test_imported_model_is_metric_and_has_compatible_schema(scene):
    config = yaml.safe_load(JOINTS.read_text(encoding="utf-8"))
    with simulation(scene) as adapted:
        for record in config["joints"]:
            joint = adapted.model.joint(record["name"])
            np.testing.assert_allclose(joint.range, record["limit_rad"])
            np.testing.assert_allclose(joint.axis, record["mujoco_axis"])
            np.testing.assert_allclose(adapted.model.actuator(record["name"]).ctrlrange, record["limit_rad"])
        for suffix in ("rl", "ll"):
            assert np.linalg.norm(adapted.model.body(f"shin_{suffix}").pos) == pytest.approx(.08)
            assert np.linalg.norm(adapted.model.body(f"motors_feet_{suffix}").pos) == pytest.approx(.08)
        assert adapted.get_state().base_position_m[2] == pytest.approx(.21425)
        assert adapted.model.neq == (1 if scene == "fixed" else 0)


def test_new_balance_profile_tracks_arms_and_supports_weight_without_base_lock():
    config = load_standing_balance_config(ROOT / "config" / "balance_volodka.yaml")
    with simulation() as adapted:
        controller = StandingBalanceController.from_simulation(adapted, config)
        max_arm_excursion = 0.0
        minimum_height = float("inf")
        for tick in range(180):
            raw = adapted.home_positions_rad.copy()
            raw[:2] += .6 * np.sin(np.pi * tick / 179) ** 2
            command = RobotJointCommand.humanoid(timestamp_s=tick / 30, positions_rad=raw, confidence=1.0)
            safe = controller.update(command, adapted.get_state(), dt_s=1 / 30)
            adapted.set_joint_targets(safe)
            state = adapted.step(17)
            assert state.is_finite
            assert state.non_foot_ground_contact_count == 0
            minimum_height = min(minimum_height, state.base_position_m[2])
            max_arm_excursion = max(max_arm_excursion, abs(state.joint_positions_rad[0] - adapted.home_positions_rad[0]))
        assert minimum_height > .18
        assert max_arm_excursion > .3
        assert state.right_foot_normal_force_n > 5
        assert state.left_foot_normal_force_n > 5
        assert state.right_foot_normal_force_n + state.left_foot_normal_force_n == pytest.approx(3.0942 * 9.81, rel=.1)
        assert adapted.model.neq == 0
