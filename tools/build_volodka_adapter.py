"""Build the application-facing CAD model from the committed source snapshot.

No upstream checkout is needed. This is a coordinate/name adapter, not a new
mass, inertia, contact or motor identification. Run from any directory with
the project's Python environment.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "volodka"

# API name -> upstream body, upstream joint, native radians per API radian.
# The retained names/order are compatibility labels, not hardware calibration.
MAPPING = {
    "shoulder_rh": ("right_arm_link1", "right_shoulder_roll", 1),
    "shoulder_lh": ("left_arm_link1", "left_shoulder_roll", 1),
    "elbow_rh": ("right_arm_link2", "right_shoulder_yaw", -1),
    "elbow_lh": ("left_arm_link2", "left_shoulder_yaw", 1),
    "wrist_rh": ("right_arm_link3", "right_elbow", -1),
    "wrist_lh": ("left_arm_link3", "left_elbow", 1),
    "rotat_axis_rl": ("right_leg_link1", "right_hip_yaw", -1),
    "rotat_axis_ll": ("left_leg_link1", "left_hip_yaw", 1),
    "motors_thigh_rl": ("right_leg_link2", "right_hip_roll", -1),
    "motors_thigh_ll": ("left_leg_link2", "left_hip_roll", 1),
    "knee_rl": ("right_leg_link3", "right_hip_pitch", 1),
    "knee_ll": ("left_leg_link3", "left_hip_pitch", 1),
    "shin_rl": ("right_leg_link4", "right_knee", 1),
    "shin_ll": ("left_leg_link4", "left_knee", 1),
    "motors_feet_rl": ("right_leg_link5", "right_ankle_pitch", 1),
    "motors_feet_ll": ("left_leg_link5", "left_ankle_pitch", 1),
    "foot_rl": ("right_leg_link6", "right_ankle_roll", -1),
    "foot_ll": ("left_leg_link6", "left_ankle_roll", 1),
    "neck": ("head_link1", "head_neck_yaw", 1),
    "head": ("head_link2", "head_neck_pitch", 1),
}


def numbers(values) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def site(body: ET.Element, name: str, position) -> None:
    ET.SubElement(body, "site", name=name, pos=numbers(position), size="0.002", group="5")


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build() -> None:
    source = ET.parse(MODEL / "source_scene.xml").getroot()
    legacy = yaml.safe_load((ROOT / "config" / "joints.yaml").read_text(encoding="utf-8"))
    old_records = {record["name"]: record for record in legacy["joints"]}
    body_names = {native_body: api for api, (native_body, _, _) in MAPPING.items()}
    body_names["base_link"] = "torso"
    bodies = {body.get("name"): body for body in source.iter("body")}
    joints = {joint.get("name"): joint for joint in source.iter("joint")}
    actuator_element = source.find("actuator")
    assert actuator_element is not None
    actuators = {item.get("joint"): item for item in actuator_element}
    actuator_element.clear()
    records = []
    mapping_records = []
    for index, (api, (native_body, native_joint, sign)) in enumerate(MAPPING.items()):
        home = float(old_records[api]["home_rad"])
        joint = joints[native_joint]
        native_limits = [float(value) for value in joint.attrib["range"].split()]
        limits = sorted(home + sign * value for value in native_limits)
        axis = [sign * float(value) for value in joint.attrib["axis"].split()]
        joint.set("name", api)
        joint.set("axis", numbers(axis))
        joint.set("ref", numbers([home]))
        joint.set("range", numbers(limits))
        actuator = actuators[native_joint]
        actuator.set("name", api)
        actuator.set("joint", api)
        actuator.set("group", "0")
        actuator.set("ctrllimited", "true")
        actuator.set("ctrlrange", numbers(limits))
        actuator_element.append(actuator)
        body = bodies[native_body]
        parent_native = next(parent.get("name") for parent in source.iter("body") if body in list(parent))
        records.append({
            "index": index, "name": api, "parent": body_names[parent_native],
            "axis": axis, "mujoco_axis": axis, "limit_rad": limits,
            "limit_deg": [math.degrees(value) for value in limits],
            "home_rad": home, "home_deg": math.degrees(home),
            "mass_kg": float(body.find("inertial").attrib["mass"]),
        })
        mapping_records.append({
            "api_name": api, "native_body": native_body, "native_joint": native_joint,
            "sign": sign, "api_home_rad": home, "native_limits_rad": native_limits,
        })

    # Sites describe anatomical landmarks, not extra rigid bodies/masses.
    # In the CAD arm chain the second motor is shoulder abduction, and the
    # third motor is the elbow, despite the inherited API labels elbow/wrist.
    for side in ("right", "left"):
        site(bodies[f"{side}_arm_link2"], f"{side}_shoulder_ik", (0, 0, 0))
        site(bodies[f"{side}_arm_link3"], f"{side}_elbow_ik", (0, 0, 0))
        site(bodies[f"{side}_arm_link3"], f"{side}_wrist_ik", (.100, 0, 0))
        site(bodies[f"{side}_arm_link3"], f"{side}_hand_ik", (.112, 0, 0))
        site(bodies[f"{side}_leg_link4"], f"{side}_knee_ik", (0, 0, 0))
        site(bodies[f"{side}_leg_link5"], f"{side}_ankle_ik", (0, 0, 0))
        foot = bodies[f"{side}_leg_link6"]
        lateral_center = .00725 if side == "left" else -.00725
        site(foot, f"{side}_foot_contact", (.02925, lateral_center, -.002))
        site(foot, f"{side}_toe_ik", (.02925, lateral_center, .063))
    site(bodies["head_link2"], "head_nose_ik", (.045, .025, 0))
    site(bodies["base_link"], "imu_site", (0, 0, .07))
    for native, body in bodies.items():
        body.set("name", body_names[native])
    source.find(".//freejoint").set("name", "base_free")
    geom_names = {"floor": "ground", "right_sole_contact": "foot_rl_geom", "left_sole_contact": "foot_ll_geom"}
    for geom in source.iter("geom"):
        name = geom.get("name", "")
        geom.set("name", geom_names.get(name, name))
        geom.set("group", "1" if name.startswith("cad_visual_") else "0" if name == "floor" else "2")
    for pair in source.findall("contact/pair"):
        for attribute in ("geom1", "geom2"):
            pair.set(attribute, geom_names.get(pair.get(attribute), pair.get(attribute)))
    custom = source.find("custom")
    if custom is None:
        custom = ET.SubElement(source, "custom")
    ET.SubElement(custom, "numeric", name="robot_forward_axis", data="1 0 0")
    ET.SubElement(custom, "numeric", name="head_forward_axis", data="1 0 0")
    ET.SubElement(custom, "text", name="physical_model_profile", data="volodka")
    ET.SubElement(custom, "text", name="hardware_mapping_status", data="unverified_simulation_only")
    ET.SubElement(source, "statistic", center="0 0 0.21", extent="0.65")
    source.set("model", "volodka_teleop_free")
    write_xml(source, MODEL / "scene_free.xml")
    fixed = deepcopy(source)
    fixed.set("model", "volodka_teleop_assisted")
    carriage = ET.SubElement(fixed.find("worldbody"), "body", name="standing_carriage", pos="0 0 0.21425")
    ET.SubElement(carriage, "joint", name="standing_vertical", type="slide", axis="0 0 1", limited="false", damping="0.1")
    ET.SubElement(carriage, "inertial", pos="0 0 0", mass="0.000001", diaginertia="1e-9 1e-9 1e-9")
    equality = ET.SubElement(fixed, "equality")
    ET.SubElement(equality, "weld", name="fixed_base", body1="torso", body2="standing_carriage", solref="0.002 1", solimp="0.99 0.999 0.0001")
    write_xml(fixed, MODEL / "scene_fixed.xml")
    config = {
        "schema_version": 1, "robot_model": "volodka_cad_teleop", "parameter_status": "provisional",
        "source": {"snapshot": "models/volodka/physical_snapshot.json", "adapter": "models/volodka/joint_mapping.json", "hardware_mapping_verified": False},
        "units": {"internal_angle": "rad", "length": "m", "mass": "kg"},
        "coordinates": {"robot_front_mujoco": [1, 0, 0], "robot_left_mujoco": [0, 1, 0], "robot_up_mujoco": [0, 0, 1], "model_space_to_metre_scale": 1.0},
        "physical_parameters": {"total_mass_kg": 3.0942, "torso_mass_kg": .7093, "inertia": "upstream_rod_box_approximation", "center_of_mass": "upstream_CAD_passport", "actuator_gains": "unidentified_legacy_numerical_reference", "self_collisions": False},
        "joint_order": list(MAPPING), "joints": records,
    }
    (ROOT / "config" / "joints_volodka.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (MODEL / "joint_mapping.json").write_text(json.dumps({
        "schema_version": 1, "formula": "q_native = sign * (q_api - api_home_rad)",
        "hardware_mapping_verified": False, "robot_forward_axis": [1, 0, 0],
        "landmark_status": "wrist/hand/nose are retargeting proxies, not CAD mass properties",
        "joints": mapping_records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Built portable Volodka free/assisted scenes and compatible joint schema.")


if __name__ == "__main__":
    build()
