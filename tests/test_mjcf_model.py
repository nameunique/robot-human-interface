from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "humanoid"
CONFIG_PATH = PROJECT_ROOT / "config" / "joints.yaml"

JOINT_NAMES = (
    "shoulder_rh",
    "shoulder_lh",
    "elbow_rh",
    "elbow_lh",
    "wrist_rh",
    "wrist_lh",
    "rotat_axis_rl",
    "rotat_axis_ll",
    "motors_thigh_rl",
    "motors_thigh_ll",
    "knee_rl",
    "knee_ll",
    "shin_rl",
    "shin_ll",
    "motors_feet_rl",
    "motors_feet_ll",
    "foot_rl",
    "foot_ll",
    "neck",
    "head",
)


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _load_model(scene: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MODEL_DIR / f"scene_{scene}.xml"))


def _object_names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> tuple[str, ...]:
    return tuple(mujoco.mj_id2name(model, object_type, index) for index in range(count))


@pytest.mark.parametrize("scene", ["fixed", "free"])
def test_model_has_exact_joint_and_actuator_schema(scene: str) -> None:
    model = _load_model(scene)
    assert model.nu == 20

    actuator_names = _object_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    assert actuator_names == JOINT_NAMES

    for actuator_id, joint_name in enumerate(JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert joint_id >= 0
        assert int(model.actuator_trnid[actuator_id, 0]) == joint_id


@pytest.mark.parametrize("scene", ["fixed", "free"])
def test_mjcf_matches_limits_axes_and_provisional_total_mass(scene: str) -> None:
    model = _load_model(scene)
    config = _load_config()
    assert config["parameter_status"] == "provisional"
    assert tuple(config["joint_order"]) == JOINT_NAMES

    records = sorted(config["joints"], key=lambda item: item["index"])
    for record in records:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, record["name"])
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            record["name"],
        )
        np.testing.assert_allclose(model.jnt_axis[joint_id], record["mujoco_axis"], atol=1e-12)
        np.testing.assert_allclose(model.jnt_range[joint_id], record["limit_rad"], atol=1e-9)
        np.testing.assert_allclose(
            model.actuator_ctrlrange[actuator_id],
            record["limit_rad"],
            atol=1e-9,
        )

    robot_mass = float(np.sum(model.body_mass))
    carriage_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "standing_carriage")
    if carriage_id >= 0:
        robot_mass -= float(model.body_mass[carriage_id])
    assert robot_mass == pytest.approx(2.933134, abs=1e-8)


def test_fixed_scene_uses_vertical_carriage_while_free_scene_has_no_constraint() -> None:
    fixed = _load_model("fixed")
    free = _load_model("free")
    assert fixed.neq == 1
    assert free.neq == 0
    assert mujoco.mj_id2name(fixed, mujoco.mjtObj.mjOBJ_EQUALITY, 0) == "fixed_base"
    assert mujoco.mj_name2id(fixed, mujoco.mjtObj.mjOBJ_JOINT, "standing_vertical") >= 0
    assert mujoco.mj_name2id(free, mujoco.mjtObj.mjOBJ_JOINT, "standing_vertical") == -1
    assert fixed.nq == 28
    assert fixed.nv == 27
    assert free.nq == 27
    assert free.nv == 26


@pytest.mark.parametrize("scene", ["fixed", "free"])
def test_anatomical_sites_point_toward_imported_mesh_front(scene: str) -> None:
    model = _load_model(scene)

    head_nose = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "head_nose_ik"
    )
    assert model.site_pos[head_nose, 0] == pytest.approx(-0.15)

    for side in ("right", "left"):
        toe = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_toe_ik"
        )
        sole = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot_contact"
        )
        # Unity's named front is +Z and the converter maps it to MuJoCo -X.
        # The -0.045 edge is therefore the toe of the 0.21 m-long sole proxy.
        assert model.site_pos[toe, 0] == pytest.approx(-0.045)
        assert model.site_pos[toe, 0] < model.site_pos[sole, 0]


@pytest.mark.parametrize("scene", ["fixed", "free"])
def test_visual_meshes_and_collision_proxies_are_separate_layers(scene: str) -> None:
    model = _load_model(scene)
    assert model.nmesh == 21
    assert len(tuple((MODEL_DIR / "meshes").glob("*.obj"))) == 21

    groups, counts = np.unique(model.geom_group, return_counts=True)
    assert dict(zip(groups.tolist(), counts.tolist(), strict=True)) == {0: 1, 1: 21, 2: 21}
    visual_ids = np.flatnonzero(model.geom_group == 1)
    assert np.all(model.geom_contype[visual_ids] == 0)
    assert np.all(model.geom_conaffinity[visual_ids] == 0)
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, 0) == "ground"


@pytest.mark.parametrize("scene", ["fixed", "free"])
def test_one_thousand_headless_steps_remain_finite(scene: str) -> None:
    model = _load_model(scene)
    data = mujoco.MjData(model)
    config = _load_config()
    records = sorted(config["joints"], key=lambda item: item["index"])
    home = np.asarray([record["home_rad"] for record in records], dtype=np.float64)

    for actuator_id, (joint_name, target) in enumerate(zip(JOINT_NAMES, home, strict=True)):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = target
        data.ctrl[actuator_id] = target
    mujoco.mj_forward(model, data)
    for _ in range(1000):
        mujoco.mj_step(model, data)

    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert np.isfinite(data.actuator_force).all()


def test_every_actuator_changes_its_transmitted_joint() -> None:
    model = _load_model("fixed")
    config = _load_config()
    records = sorted(config["joints"], key=lambda item: item["index"])
    home = np.asarray([record["home_rad"] for record in records], dtype=np.float64)

    for actuator_id, record in enumerate(records):
        data = mujoco.MjData(model)
        for baseline_id, baseline_record in enumerate(records):
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                baseline_record["name"],
            )
            data.qpos[model.jnt_qposadr[joint_id]] = home[baseline_id]
        data.ctrl[:] = home
        mujoco.mj_forward(model, data)

        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, record["name"])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        initial = float(data.qpos[qpos_adr])
        target = float(
            np.clip(
                initial + 0.08,
                record["limit_rad"][0] + 0.01,
                record["limit_rad"][1] - 0.01,
            )
        )
        data.ctrl[actuator_id] = target
        for _ in range(150):
            mujoco.mj_step(model, data)
        assert abs(float(data.qpos[qpos_adr]) - initial) > 1e-4, record["name"]
