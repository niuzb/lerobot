"""Offline checks for clipped measurements, held-out isolation and asset limits."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from lerobot.sim2real.config import load_config
from lerobot.sim2real.gripper_fit import fit_gripper, select_measurements

ROOT = Path(__file__).resolve().parents[2]


def measurements():
    return {
        "measurements": [
            {"id": n, "split": split, "actual_opening_percent": p, "gap_mm": g}
            for n, split, p, g in [
                ("closed", "fit", 3.134, 0),
                ("low", "fit", 23.7, 16),
                ("bad", "fit", 49.3, 47),
                ("high", "fit", 76, 56),
                ("saturated", "fit", 100, 82),
                ("check1", "check", 50.7, 38),
                ("check2", "check", 49.7, 36),
                ("check3", "check", 93.9, 70),
            ]
        ],
        "quality_flags": [{"status": "provisionally_excluded_from_fit", "measurement_ids": ["bad"]}],
    }


class MeasurementTests(unittest.TestCase):
    def test_saturation_outlier_and_contact_are_not_fit_coordinates(self):
        doc = measurements()
        before = deepcopy(doc)
        groups = select_measurements(doc)
        self.assertEqual([r["id"] for r in groups["fit"]], ["low", "high"])
        self.assertEqual([r["id"] for r in groups["closed"]], ["closed"])
        self.assertEqual(len(groups["excluded"]), 2)
        self.assertEqual(doc, before)

    def test_duplicate_and_nonfinite_rejected(self):
        for corrupt in ("duplicate", "nan"):
            doc = measurements()
            if corrupt == "duplicate":
                doc["measurements"].append(deepcopy(doc["measurements"][0]))
            else:
                doc["measurements"][0]["gap_mm"] = float("nan")
            with self.assertRaises(ValueError):
                select_measurements(doc)

    def test_heldout_values_cannot_improve_fitted_mapping_or_accept_bad_geometry(self):
        config = load_config(ROOT / "configs/so101_r2s2r_20261017_measured_board.json")
        original = deepcopy(config)
        doc = measurements()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            first = fit_gripper(config, doc, out)
            self.assertLess(first["heldout_max_error_mm"], 3)
            self.assertFalse(first["passed"])
            self.assertFalse(first["candidate_within_asset_limits"])
            for row in doc["measurements"]:
                if row["split"] == "check":
                    row["gap_mm"] += 20
            second = fit_gripper(config, doc, out)
            for key in ("scale_rad_per_percent", "offset_rad", "gripper_knots"):
                self.assertEqual(first["fit_model"][key], second["fit_model"][key])
            self.assertFalse(second["heldout_gaps_within_3mm"])
            self.assertFalse(json.loads((out / "gripper_report.json").read_text())["passed"])
        self.assertEqual(config, original)


if __name__ == "__main__":
    unittest.main()
