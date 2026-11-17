"""Automatic training requires exactly 100 independent physical successes."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

EXAMPLE = Path(__file__).resolve().parents[2] / "scripts/simulation/so101_tape_leisaac"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load("_batch_common", "common.py")
with patch.dict(sys.modules, {"common": common}):
    merge = load("_batch_merge", "merge_batch.py")
training = load("_batch_training", "train_batch.py")


class BatchTrainingGateTest(unittest.TestCase):
    def setUp(self):
        self.entries = [
            {
                "index": i,
                "physical_seed": 1000000 + 100 * i,
                "physics_npz_sha256": f"{i:064x}",
                "height_delta_m": (0.02, 0.0, -0.02)[i % 3],
            }
            for i in range(100)
        ]

    def test_100_distinct_successes_and_balanced_training_heights(self):
        merge.validate_sources(self.entries, 100)
        for height in (0.02, 0.0, -0.02):
            self.assertEqual(sum(entry["height_delta_m"] == height for entry in self.entries[:90]), 30)

    def test_partial_or_extra_count_cannot_start_training(self):
        for entries in (self.entries[:99], self.entries + [self.entries[0]]):
            with self.assertRaisesRegex(ValueError, "exactly 100"):
                merge.validate_sources(entries, 100)

    def test_rerendered_trajectory_does_not_count_twice(self):
        for key in ("physical_seed", "physics_npz_sha256"):
            entries = copy.deepcopy(self.entries)
            entries[99][key] = entries[0][key]
            with self.assertRaisesRegex(ValueError, "Repeated physical trajectory"):
                merge.validate_sources(entries, 100)

    def test_episode_order_and_height_balance_are_checked(self):
        entries = copy.deepcopy(self.entries)
        entries[90]["height_delta_m"] = -0.02
        with self.assertRaisesRegex(ValueError, "balance"):
            merge.validate_sources(entries, 100)

    def test_normalization_excludes_validation_episodes(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [0, 0, 1, 1],
                        "observation.state": [[0.0] * 25, [2.0] * 25, [1000.0] * 25, [1002.0] * 25],
                        "action": [[0.0] * 6, [2.0] * 6, [1000.0] * 6, [1002.0] * 6],
                    }
                ),
                root / "data/rows.parquet",
            )
            dataset = SimpleNamespace(
                root=root,
                episodes=[0],
                num_episodes=2,
                meta=SimpleNamespace(
                    features={key: {"dtype": "float32"} for key in ("observation.state", "action")}, stats={}
                ),
            )
            report = training.train_only_stats(dataset, root / "stats.json")
            self.assertEqual(report["episodes"], [0])
            self.assertEqual(report["frames"], 2)
            np.testing.assert_array_equal(dataset.meta.stats["observation.state"]["mean"], np.ones(25))


if __name__ == "__main__":
    unittest.main()
