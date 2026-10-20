"""Check video/data export alignment using an isolated synthetic fixture, never training data."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from lerobot.sim2real.data import export_episode, new_dataset


class ExportTests(unittest.TestCase):
    def test_pre_action_state_and_applied_target_survive_lerobot_roundtrip(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "synthetic_test_only"
            dataset = new_dataset(root, "local/r2s2r_test_fixture")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[..., 0] = 180
            trajectory = {"rows": [], "frames": []}
            for i in range(6):
                trajectory["rows"].append(
                    {
                        "state": [float(i)] * 6,
                        "sent_action": [float(i + 10)] * 6,
                        "post_state": [float(i + 1)] * 6,
                    }
                )
                trajectory["frames"].append({"front": frame.copy(), "side": frame.copy()})
            try:
                export_episode(dataset, trajectory)
            finally:
                dataset.finalize()
            reader = LeRobotDataset("local/r2s2r_test_fixture", root=root, video_backend="pyav")
            self.assertEqual(len(reader), 6)
            for i in (0, 3, 5):
                row = reader[i]
                np.testing.assert_allclose(row["observation.state"].numpy(), [i] * 6)
                np.testing.assert_allclose(row["action"].numpy(), [i + 10] * 6)
                self.assertEqual(tuple(row["observation.images.front"].shape), (3, 480, 640))
                self.assertGreater(float(row["observation.images.front"][0].mean()), 0.6)


if __name__ == "__main__":
    unittest.main()
