"""Offline contact regression: real surfaces, frozen mapping, no hardware."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np

from lerobot.sim2real.config import digest, load_config
from lerobot.sim2real.gripper_contact import ContactReference, align_gripper_contact, measured_gap_m

ROOT = Path(__file__).resolve().parents[2]


class GripperContactTest(unittest.TestCase):
    """Exercise the inspected pinned SO-101 asset and recorded raw reference."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs/so101_r2s2r_20261017_measured_board.json")
        cls.document = json.loads(
            (ROOT / "outputs/calibration/20261018/gripper_measurements_001/measurements.json").read_text(
                encoding="utf-8"
            )
        )

    def test_closed_surfaces_and_landmarks_are_not_legacy_sphere_gap(self):
        geometry = ContactReference(self.config["asset"])
        self.assertAlmostEqual(geometry.landmark_gap_mm(geometry.closed_rad), 0, places=6)
        self.assertGreater(measured_gap_m(geometry.model, geometry.data), 0.002)
        self.assertGreater(geometry.signed_gap(geometry.closed_rad + 0.01), 0)
        self.assertLess(geometry.signed_gap(geometry.closed_rad - 0.01), 0)
        gaps = [geometry.landmark_gap_mm(q) for q in np.linspace(geometry.closed_rad, 0.8, 51)]
        self.assertTrue(np.all(np.diff(gaps) > 0))

    def test_contact_stops_motion_and_check_values_cannot_refit_or_accept(self):
        before = digest(self.config["asset"])
        original = deepcopy(self.document)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = align_gripper_contact(self.config, self.document, out)
            self.assertTrue(report["contact_bench_passed"])
            self.assertFalse(report["passed"])
            source = ET.parse(self.config["asset"])
            candidate = ET.parse(out / "so101.contact_candidate.xml")
            self.assertEqual(
                [g.attrib for g in source.findall(".//geom")],
                [g.attrib for g in candidate.findall(".//geom")],
            )
            model = mujoco.MjModel.from_xml_path(str(out / "so101.contact_candidate.xml"))
            data = mujoco.MjData(model)
            data.qpos[model.joint("gripper").qposadr[0]] = report["closed_sim_rad"]
            mujoco.mj_forward(model, data)
            self.assertLess(measured_gap_m(model, data), 1e-8)
            changed = deepcopy(self.document)
            for row in changed["measurements"]:
                if row["split"] == "check" and row["gap_mm"] > 0:
                    row["gap_mm"] += 20
            other = align_gripper_contact(self.config, changed, out)
            self.assertEqual(report["candidate_asset_sha256"], other["candidate_asset_sha256"])
            self.assertEqual(report["gripper_knots"], other["gripper_knots"])
            self.assertGreater(other["historical_max_error_mm"], 15)
            self.assertFalse(other["passed"])
        self.assertEqual(digest(self.config["asset"]), before)
        self.assertEqual(self.document, original)

    def test_corrupt_closed_raw_reference_is_rejected(self):
        document = deepcopy(self.document)
        next(r for r in document["measurements"] if r["id"] == "gap_009")["raw_position"] += 1
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            align_gripper_contact(self.config, document, Path(tmp))


if __name__ == "__main__":
    unittest.main()
