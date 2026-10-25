"""Portable source-physics snapshot, independent of the upstream archive."""

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest


MODEL_ROOT = Path(__file__).resolve().parents[1] / "models" / "volodka"


@pytest.fixture(scope="module")
def source_model():
    return mujoco.MjModel.from_xml_path(str(MODEL_ROOT / "source_scene.xml"))


def test_source_snapshot_compiles_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(MODEL_ROOT / "source_scene.xml"))
    assert model.nbody == 22  # world plus 21 physical bodies
    assert model.nq == 27  # free base plus 20 hinges
    assert model.nu == 20
    assert float(model.body_mass.sum()) == pytest.approx(3.0942, abs=1e-12)


def test_cad_meshes_match_original_manifest():
    manifest = json.loads((MODEL_ROOT / "meshes" / "manifest.json").read_text("utf-8"))
    assert len(manifest["assets"]) == 21
    for asset in manifest["assets"]:
        filename = asset["file"]
        assert Path(filename).name == filename
        data = (MODEL_ROOT / "meshes" / filename).read_bytes()
        assert len(data) == asset["size_bytes"]
        assert hashlib.sha256(data).hexdigest().upper() == asset["sha256"].upper()


def test_source_mechanics_and_reference_limits_are_preserved(source_model):
    snapshot = json.loads((MODEL_ROOT / "physical_snapshot.json").read_text("utf-8"))
    names = snapshot["joint_order"]
    assert len(names) == 20 and len(set(names)) == 20
    for index, name in enumerate(names):
        joint = source_model.joint(name)
        assert source_model.actuator_trnid[index, 0] == joint.id
        assert np.allclose(source_model.jnt_axis[joint.id], [0.0, 0.0, 1.0])
    assert np.all(source_model.actuator_forcelimited)
    assert np.allclose(source_model.actuator_forcerange, [-1.5, 1.5])
    assert np.all(source_model.body_inertia[1:] > 0)
    assert np.allclose(source_model.actuator_gear[:, 0], 1.0)
    assert np.allclose(source_model.actuator_gear[:, 1:], 0.0)
    assert np.allclose(source_model.qpos0[7:], 0.0)
    assert source_model.qpos0[2] == pytest.approx(0.21425)


def test_explicit_source_contacts_and_visual_only_meshes_are_preserved():
    root = ET.parse(MODEL_ROOT / "source_scene.xml").getroot()
    assert root.find("compiler").get("meshdir") == "meshes"
    meshes = root.findall("./asset/mesh")
    assert len(meshes) == 21
    for mesh in meshes:
        assert Path(mesh.get("file")).name == mesh.get("file")
        assert np.allclose(np.fromstring(mesh.get("scale"), sep=" "), 0.001)
    pairs = root.findall("./contact/pair")
    assert len(pairs) == 23  # 21 external envelopes plus two sole patches
    assert all("floor" in (pair.get("geom1"), pair.get("geom2")) for pair in pairs)
    for geom in root.findall(".//geom"):
        if geom.get("type") == "mesh":
            assert geom.get("contype") == geom.get("conaffinity") == "0"


def test_export_hashes_and_passport_provenance():
    provenance = json.loads((MODEL_ROOT / "import_provenance.json").read_text("utf-8"))
    for key, filename in (
        ("source_scene_sha256", "source_scene.xml"),
        ("snapshot_sha256", "physical_snapshot.json"),
        ("cad_manifest_sha256", "meshes/manifest.json"),
    ):
        assert hashlib.sha256((MODEL_ROOT / filename).read_bytes()).hexdigest() == provenance[key]
    assert provenance["self_collision"] is False
    assert provenance["source_files_sha256"]
    assert all(not Path(name).is_absolute() for name in provenance["source_files_sha256"])
