"""Verify physical holes and support, not just a perforated-looking texture."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
from test_core import scene

from lerobot.sim2real.lattice import validate_lattice
from lerobot.sim2real.simulation import make_scene


class LatticeTest(unittest.TestCase):
    """Exercise the assembled mesh holder with rays and a freely falling tape."""

    @classmethod
    def setUpClass(cls):
        cls.config = scene()
        measured = json.loads(
            (Path(__file__).resolve().parents[2] / "configs/so101_r2s2r_20261018_mesh_holder.json").read_text(
                encoding="utf-8"
            )
        )
        cls.config["holder"] = measured["holder"]
        cls.config["holder"]["position_m"] = [0, 0, 0]
        cls.config["tape"]["position_m"] = [0, 0, 0.14]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.xml"
            make_scene(cls.config, path)
            tree = ET.parse(path)
            world = tree.getroot().find("worldbody")
            world.remove(world.find("body[@name='base']"))
            tree.getroot().remove(tree.getroot().find("actuator"))
            cls.model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))

    def test_open_wall_and_open_mouth_with_correct_total_mass(self):
        model = self.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        holder = model.body("holder").id
        self.assertAlmostEqual(model.body_subtreemass[holder], 0.005, places=9)
        hit, miss = 0, 0
        geom = np.zeros(1, dtype=np.int32)
        for angle in np.linspace(0, 2 * np.pi, 128, endpoint=False):
            distance = mujoco.mj_ray(
                model,
                data,
                np.array([0.0, 0.0, 0.052]),
                np.array([np.cos(angle), np.sin(angle), 0.0]),
                None,
                1,
                -1,
                geom,
            )
            if distance < 0:
                miss += 1
            else:
                hit += 1
                self.assertTrue((model.geom(geom[0]).name or "").startswith("holder_wall"))
                self.assertLess(abs(distance - 0.0385), 0.0012)
        self.assertGreater(hit, 4)
        self.assertGreater(miss, 40)
        # Avoid the freely suspended tape by probing 30 mm off-centre.
        distance = mujoco.mj_ray(
            model, data, np.array([0.03, 0.0, 0.13]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geom
        )
        self.assertEqual(model.geom(geom[0]).name, "holder_bottom")
        self.assertAlmostEqual(distance, 0.126, places=6)

    def test_tape_falls_through_open_mouth_and_is_supported(self):
        model = self.model
        data = mujoco.MjData(model)
        for _ in range(900):
            mujoco.mj_step(model, data)
        tape, holder = model.body("tape").id, model.body("holder").id
        relative = data.xpos[tape] - data.xpos[holder]
        self.assertLess(np.linalg.norm(relative[:2]), 0.003)
        self.assertLess(abs(relative[2] - 0.0135), 0.002)
        self.assertLess(np.linalg.norm(data.qvel), 0.02)
        self.assertTrue(
            any(
                "holder_bottom" in [model.geom(c.geom1).name, model.geom(c.geom2).name]
                and any((model.geom(g).name or "").startswith("tape_") for g in (c.geom1, c.geom2))
                for c in data.contact
            )
        )

    def test_invalid_or_closed_mesh_is_rejected(self):
        for key, value in (
            ("strand_width_m", 0.01),
            ("segments_per_strand", 1000),
            ("circumferential_slope", float("nan")),
            ("rim_height_m", 0.05),
        ):
            holder = deepcopy(self.config["holder"])
            holder["mesh"][key] = value
            with self.assertRaises(ValueError):
                validate_lattice(holder)


if __name__ == "__main__":
    unittest.main()
