"""Align a closed jaw reference to existing contact surfaces, entirely offline."""

import itertools
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _mujoco_available, _scipy_available, require_package

from .config import digest, fingerprint, write_json
from .gripper_fit import select_measurements

if TYPE_CHECKING or _mujoco_available:
    import mujoco
if TYPE_CHECKING or _scipy_available:
    from scipy.optimize import brentq


FIXED = [f"fixed_jaw_sph_tip{i}" for i in (1, 2, 3)] + ["fixed_jaw_box3", "fixed_jaw_box4"]
MOVING = [f"moving_jaw_sph_tip{i}" for i in (1, 2, 3)] + ["moving_jaw_box2"]
REFERENCE_PAIR = ("fixed_jaw_sph_tip3", "moving_jaw_box2")


def measured_gap_m(model, data):
    """Use explicit material sites when present; keep legacy assets identifiable."""
    require_package("mujoco", extra="sim2real")
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ("fixed_jaw_measurement", "moving_jaw_measurement")
    ]
    if all(i >= 0 for i in ids):
        return float(np.linalg.norm(data.site_xpos[ids[1]] - data.site_xpos[ids[0]]))
    if any(i >= 0 for i in ids):
        raise ValueError("A jaw measurement requires both material sites")
    fixed, moving = (model.geom(n).id for n in ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1"))
    return float(
        max(
            0,
            np.linalg.norm(data.geom_xpos[fixed] - data.geom_xpos[moving])
            - model.geom_size[fixed, 0]
            - model.geom_size[moving, 0],
        )
    )


def numbers(values):
    """Serialize full precision numeric XML attributes."""
    return " ".join(map(str, np.asarray(values).ravel()))


class ContactReference:
    """Two material points at first sphere/box contact, not sphere-centre distance."""

    def __init__(self, asset):
        """Find the first contact while closing; leave the source asset unchanged."""
        require_package("mujoco", extra="sim2real")
        require_package("scipy", extra="scipy-dep")
        self.model = mujoco.MjModel.from_xml_path(str(asset))
        self.data = mujoco.MjData(self.model)
        self.qidx = self.model.joint("gripper").qposadr[0]
        self.geom_ids = [self.model.geom(n).id for n in REFERENCE_PAIR]
        self.body_ids = [int(self.model.geom_bodyid[g]) for g in self.geom_ids]
        # Scan from an open jaw so that a second crossing after penetration cannot
        # be mistaken for first contact. Kinematics may exceed the old XML limit.
        angles = np.linspace(0, -0.4, 161)
        bracket = next(
            (
                (a, b)
                for a, b in zip(angles[:-1], angles[1:], strict=True)
                if self.signed_gap(a) > 0 and self.signed_gap(b) <= 0
            ),
            None,
        )
        if bracket is None:
            raise ValueError("No first contact in the inspected jaw range")
        self.closed_rad = float(brentq(self.signed_gap, *bracket, xtol=1e-12))
        self.signed_gap(self.closed_rad)
        self.local_points = [
            (point - self.data.xpos[b]) @ self.data.xmat[b].reshape(3, 3)
            for point, b in zip(self.segment.reshape(2, 3), self.body_ids, strict=True)
        ]

    def signed_gap(self, angle):
        """Signed surface distance with a consistency check on the closest segment."""
        self.data.qpos[self.qidx] = angle
        mujoco.mj_forward(self.model, self.data)
        self.segment = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(self.model, self.data, *self.geom_ids, 0.2, self.segment))
        length = np.linalg.norm(self.segment[:3] - self.segment[3:])
        if not np.isfinite(distance) or abs(abs(distance) - length) > 1e-7:
            raise ValueError("Collision distance and closest segment disagree")
        return distance

    def landmark_gap_mm(self, angle):
        """Distance between fixed material landmarks; correspondence needs a ruler photo."""
        self.data.qpos[self.qidx] = angle
        mujoco.mj_forward(self.model, self.data)
        points = [
            local @ self.data.xmat[b].reshape(3, 3).T + self.data.xpos[b]
            for local, b in zip(self.local_points, self.body_ids, strict=True)
        ]
        return float(np.linalg.norm(points[1] - points[0]) * 1000)


def write_contact_asset(asset, out, reference, angle_range):
    """Keep all shapes/inertia; explicitly enable jaw pairs and document changed limits."""
    tree = ET.parse(asset)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(Path(asset).parent / "assets"))
    root.find("option").set("timestep", str(1 / 300))
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for fixed, moving in itertools.product(FIXED, MOVING):
        ET.SubElement(contact, "pair", geom1=fixed, geom2=moving, condim="3", solref="0.01 1")
    for name, b, point in zip(
        ("fixed_jaw_measurement", "moving_jaw_measurement"),
        reference.body_ids,
        reference.local_points,
        strict=True,
    ):
        body = root.find(f".//body[@name='{reference.model.body(b).name}']")
        ET.SubElement(body, "site", name=name, pos=numbers(point), size="0.001", rgba="0 0.8 0.8 1")
    root.find(".//joint[@name='gripper']").set("range", numbers(angle_range))
    root.find("./actuator/position[@name='gripper']").set("ctrlrange", numbers(angle_range))
    tree.write(out, encoding="utf-8", xml_declaration=True)


def contact_bench(asset, closed_rad, enabled):
    """Counterfactual empty-close/open test; one free jaw, no pose resets after start."""
    tree = ET.parse(asset)
    root = tree.getroot()
    root.find("option").set("gravity", "0 0 0")
    # Isolate the jaw from arm motion, objects and joint limits. This diagnostic
    # deliberately broadens only its in-memory limits to prove contact stops it.
    for body in root.findall(".//body"):
        for joint in list(body.findall("joint")):
            if joint.get("name") != "gripper":
                body.remove(joint)
            else:
                joint.attrib.update(range="-0.5 1.5", damping="0.2", frictionloss="0", armature="0.001")
    for geom in root.findall(".//geom"):
        geom.attrib.update(contype="0", conaffinity="0")
    actuator = root.find("actuator")
    for item in list(actuator):
        if item.get("name") != "gripper":
            actuator.remove(item)
        else:
            item.attrib.update(kp="20", kv="0.2", forcerange="-0.1 0.1", ctrlrange="-0.5 1.5")
    if not enabled:
        root.remove(root.find("contact"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qpos[0] = closed_rad + 0.03
    data.ctrl[0] = closed_rad - 0.02
    peak_force = 0.0
    for _ in range(900):
        mujoco.mj_step(model, data)
        for i in range(data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            peak_force = max(peak_force, abs(float(force[0])))
    closed = float(data.qpos[0])
    contact_count = int(data.ncon)
    data.ctrl[0] = closed_rad + 0.03
    for _ in range(600):
        mujoco.mj_step(model, data)
    return {
        "contact_enabled": enabled,
        "closed_error_deg": float(np.rad2deg(closed - closed_rad)),
        "closed_contact_count": contact_count,
        "peak_normal_force_n": peak_force,
        "opened_error_deg": float(np.rad2deg(data.qpos[0] - (closed_rad + 0.03))),
        "opened_contact_count": int(data.ncon),
        "scope": "Isolated jaw, gravity off, 0.1 Nm cap; not real force calibration or grasp success",
    }


def align_gripper_contact(config, measurements, out):
    """Freeze a closed-reference candidate; old check data is diagnostic, not acceptance."""
    groups = select_measurements(measurements)
    anchors = [r for r in measurements["measurements"] if r["id"] == "gap_009"]
    if len(anchors) != 1 or anchors[0]["gap_mm"] != 0 or anchors[0].get("raw_position") != 2054:
        raise ValueError("Expected the verified first-contact gap_009 reference")
    # The reference was independently read raw=2054. Verify the saved percentage
    # against the unchanged motor range before using it as the angular zero.
    calibration = measurements["calibration"]
    if calibration["drive_mode"] != 0 or (calibration["range_min"], calibration["range_max"]) != (2030, 2700):
        raise ValueError("This evidence applies only to the inspected SO-101 calibration")
    anchor_percent = float(anchors[0]["actual_opening_percent"])
    if not np.isclose(anchor_percent, 100 * (2054 - 2030) / 670):
        raise ValueError("Closed reference no longer matches its raw readback")
    reference = ContactReference(config["asset"])
    scale = 2 * np.pi * 670 / 4096 / 100
    offset = reference.closed_rad - scale * anchor_percent
    angle_range = [offset, offset + 100 * scale]
    path = out / "so101.contact_candidate.xml"
    write_contact_asset(config["asset"], path, reference, angle_range)
    dynamics = [contact_bench(path, reference.closed_rad, enabled) for enabled in (True, False)]
    yes, no = dynamics
    contact_pass = (
        abs(yes["closed_error_deg"]) < 0.3
        and yes["closed_contact_count"] > 0
        and yes["peak_normal_force_n"] > 0
        and yes["opened_contact_count"] == 0
        and abs(yes["opened_error_deg"]) < 0.1
        and no["closed_error_deg"] < -0.8
        and no["closed_contact_count"] == 0
    )
    predictions = [
        dict(r, predicted_mm=reference.landmark_gap_mm(scale * r["percent"] + offset))
        for split in ("fit", "check")
        for r in groups[split]
    ]
    for row in predictions:
        row["error_mm"] = row["predicted_mm"] - row["gap_mm"]
    report = {
        "passed": False,
        "status": "contact_implementation_checked_pending_independent_ruler_correspondence",
        "source_config_sha256": fingerprint(config),
        "source_asset_sha256": digest(config["asset"]),
        "measurements_sha256": fingerprint(measurements),
        "candidate_asset_sha256": digest(path),
        "closed_reference_id": "gap_009",
        "closed_raw": 2054,
        "closed_percent": anchor_percent,
        "closed_sim_rad": reference.closed_rad,
        "scale_rad_per_percent": scale,
        "offset_rad": offset,
        "gripper_knots": [[0, angle_range[0]], [100, angle_range[1]]],
        "assumption": "Rigid direct drive, STS3215 4096 counts/revolution; endpoints inferred from motor range",
        "reference_geoms": list(REFERENCE_PAIR),
        "reference_local_points_m": [p.tolist() for p in reference.local_points],
        "gap_definition": "Distance between two material sites that coincide at first contact; ruler correspondence unconfirmed",
        "original_joint_range_rad": reference.model.jnt_range[reference.model.joint("gripper").id].tolist(),
        "candidate_joint_and_control_range_rad": angle_range,
        "geometry_resized_or_translated": False,
        "collision_pairs_added": len(FIXED) * len(MOVING),
        "contact_bench_passed": bool(contact_pass),
        "contact_bench": dynamics,
        "historical_diagnostic_predictions": predictions,
        "historical_max_error_mm": max(abs(r["error_mm"]) for r in predictions),
        "excluded_measurements": groups["excluded"],
        "independent_checks": [],
        "independent_check_target": {"minimum_count": 3, "max_abs_error_mm": 3},
        "fresh_check_protocol": "35%, 60%, 85% approximately; same inner-tip ruler locations, raw readback and ruler photo, do not refit on these",
        "full_scene_calibration_passed": False,
    }
    write_json(out / "contact_alignment.json", report)
    write_json(out / "measurements.snapshot.json", measurements)
    return report
