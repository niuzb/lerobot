"""Strict domain balance, deterministic resume and validation exclusion."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa

EXAMPLE = Path(__file__).resolve().parents[2] / "scripts/simulation/so101_tape_leisaac"
with patch.object(sys, "path", [str(EXAMPLE), *sys.path]):
    spec = importlib.util.spec_from_file_location("_isaac_mixed", EXAMPLE / "mixed_data.py")
    mixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mixed)


def test_every_batch_is_exactly_four_from_each_domain():
    for nreal, nsim in [(7, 19), (40, 9), (4, 4)]:
        sampler = mixed.BalancedBatchSampler(nreal, nsim, seed=41)
        for _ in range(3):
            for batch in sampler:
                assert len(batch) == 8
                assert sum(i < nreal for i in batch) == 4
                assert sum(nreal <= i < nreal + nsim for i in batch) == 4


def test_resume_reproduces_original_domain_sample_sequence():
    source = mixed.BalancedBatchSampler(11, 31, seed=91)
    full = list(source) + list(source) + list(source)
    for offset in [1, 7, 8, 9, 17]:
        resumed = mixed.BalancedBatchSampler(11, 31, seed=91, start_step=offset)
        continuation = list(resumed) + list(resumed) + list(resumed)
        assert continuation[: len(full) - offset] == full[offset:]


def test_real_holdout_is_fixed_and_completely_excluded():
    training, validation = mixed.real_split()
    assert validation == [22, 23, 38, 40, 47, 60, 71, 72, 82, 84]
    assert len(training) == 78 and not set(training) & set(validation)
    table = pa.table(
        {
            "episode_index": list(range(88)),
            "value": [1000000.0 if i in validation else 2.0 for i in range(88)],
        }
    )
    train = mixed.select_episodes(table, training)
    moments = mixed.balanced_moments(np.array(train["value"].to_pylist())[:, None], np.array([[4.0]]))
    np.testing.assert_allclose(moments["mean"], [3.0])
    np.testing.assert_allclose(moments["std"], [1.0])


def test_statistics_weight_domains_equally_instead_of_by_frame_count():
    stats = mixed.balanced_moments(np.zeros((2, 1)), np.full((100, 1), 10.0))
    np.testing.assert_allclose(stats["mean"], [5.0])
    np.testing.assert_allclose(stats["std"], [5.0])
