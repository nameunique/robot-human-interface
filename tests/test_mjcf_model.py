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

    assert float(np.sum(model.body_mass)) == pytest.approx(2.933134, abs=1e-8)


def test_fixed_and_free_scenes_differ_only_by_base_constraint() -> None:
    fixed = _load_model("fixed")
    free = _load_model("free")
    assert fixed.neq == 1
    assert free.neq == 0
    assert mujoco.mj_id2name(fixed, mujoco.mjtObj.mjOBJ_EQUALITY, 0) == "fixed_base"
    assert fixed.nq == free.nq == 27
    assert fixed.nv == free.nv == 26


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
