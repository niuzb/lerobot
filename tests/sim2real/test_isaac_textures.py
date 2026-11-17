"""Framewise appearance coverage, reproducibility and real PNG-pattern checks."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

EXAMPLE = Path(__file__).resolve().parents[2] / "scripts/simulation/so101_tape_leisaac"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load("_tape_texture_common", "common.py")
with patch.dict(sys.modules, {"common": common}):
    texture = load("_tape_frame_textures", "texture_randomization.py")


class FrameTextureTest(unittest.TestCase):
    def test_each_group_changes_every_frame_and_reproduces(self):
        a = texture.TextureSchedule(81000, 6, 16)
        b = texture.TextureSchedule(81000, 6, 16)
        rows = [a.next(i) for i in range(545)]
        self.assertEqual(rows, [b.next(i) for i in range(545)])
        bank = {"groups": {group: list(range(16)) for group in texture.GROUPS}}
        texture.validate_schedule(rows, 545, bank)
        self.assertTrue(all(len({row["texture_ids"][g] for row in rows}) == 16 for g in texture.GROUPS))
        self.assertNotEqual(
            [row["texture_ids"]["table"] for row in rows], [row["texture_ids"]["background"] for row in rows]
        )

    def test_episode_seed_changes_schedule(self):
        a = texture.TextureSchedule(81000, 6, 16)
        b = texture.TextureSchedule(81000, 7, 16)
        self.assertNotEqual([a.next(i) for i in range(20)], [b.next(i) for i in range(20)])

    def test_detect_repeated_missing_and_misaligned_texture_frames(self):
        schedule = texture.TextureSchedule(8, 0, 4)
        rows = [schedule.next(i) for i in range(8)]
        bank = {"groups": {group: list(range(4)) for group in texture.GROUPS}}
        changed = copy.deepcopy(rows)
        changed[3]["texture_ids"]["table"] = changed[2]["texture_ids"]["table"]
        with self.assertRaisesRegex(ValueError, "did not change"):
            texture.validate_schedule(changed, 8, bank)
        with self.assertRaisesRegex(ValueError, "missing frames"):
            texture.validate_schedule(rows[:-1], 8, bank)
        changed = copy.deepcopy(rows)
        changed[1]["source_frame_index"] = 2
        with self.assertRaisesRegex(ValueError, "indices"):
            texture.validate_schedule(changed, 8, bank)

    def test_pngs_are_lossless_spatial_patterns(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "pattern.png"
            for index in range(len(texture.PATTERNS)):
                pixels = texture.texture_pixels(81, index, 64)
                texture.write_png(path, pixels)
                with Image.open(path) as source:
                    np.testing.assert_array_equal(np.asarray(source), pixels)
                self.assertGreater(float(pixels.std(axis=(0, 1)).max()), 5)

    def test_frozen_camera_is_rejected_even_if_texture_ids_change(self):
        schedule = texture.TextureSchedule(8, 0, 4)
        rows = [schedule.next(i) for i in range(3)]
        bank = {"groups": {group: list(range(4)) for group in texture.GROUPS}}
        for i, row in enumerate(rows):
            row["capture_rgb_sha256"] = {"front": f"{i:064x}", "side": "a" * 64}
        with self.assertRaisesRegex(ValueError, "identical image"):
            texture.validate_schedule(rows, 3, bank)

    def test_bank_integrity_and_seed_reproducibility(self):
        with TemporaryDirectory() as temporary:
            a = texture.create_bank(Path(temporary) / "a", 81, 4, 32)
            b = texture.create_bank(Path(temporary) / "b", 81, 4, 32)
            self.assertEqual(a, b)
            for group in texture.GROUPS:
                self.assertEqual(len({item["sha256"] for item in a["groups"][group]}), 4)
                for item in a["groups"][group]:
                    self.assertEqual(common.sha256(Path(temporary) / "a" / item["file"]), item["sha256"])


if __name__ == "__main__":
    unittest.main()
