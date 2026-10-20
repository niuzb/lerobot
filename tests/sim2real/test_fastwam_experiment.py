"""Guard the SO-101 FastWAM temporal split and effective domain balance."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "train_fastwam_so101_20261020.py"
if not SCRIPT.exists():
    SCRIPT = Path(__file__).resolve().parents[2].parent / "train_fastwam_so101_20261020.py"
spec = importlib.util.spec_from_file_location("fastwam_experiment", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.mark.parametrize(
    "ids,horizon,expected",
    [
        ([0] * 33, 32, [0]),
        ([0] * 32, 32, []),
        ([79] * 34 + [80] * 33, 32, [0, 1, 34]),
        ([0, 1, 0], 2, []),
        ([], 32, []),
    ],
)
def test_future_video_window_does_not_cross_episodes(ids, horizon, expected):
    """An equal endpoint alone must not conceal an intermediate episode boundary."""
    assert experiment.valid_starts(ids, horizon).tolist() == expected


def test_each_optimizer_update_has_exact_domain_balance():
    """Independent reproducible draws retain four valid examples per domain."""
    real, sim = np.array([4, 8, 12]), np.array([16, 20, 24, 28])
    first = experiment.sample_update(real, sim, 1000, 1)
    assert first == experiment.sample_update(real, sim, 1000, 1)
    assert first != experiment.sample_update(real, sim, 1000, 2)
    assert len(first) == 8
    assert sum(domain == "real" for domain, _ in first) == 4
    assert sum(domain == "sim" for domain, _ in first) == 4
    assert all(index in (real if domain == "real" else sim) for domain, index in first)


def test_partial_base_requires_shared_weights_and_full_reload():
    """Only task heads may be new; a missing video block must never load silently."""
    block = "model.mot.layers.0.blocks.video.weight"
    legacy = "model.mot.mixtures.video.blocks.0.weight"
    head = "model.mot.mixtures.action.head.weight"
    model = {block: (4, 4), head: (6, 4)}
    assert experiment.audit_checkpoint(model, {legacy: (4, 4)}) == [head]
    with pytest.raises(ValueError, match="coverage"):
        experiment.audit_checkpoint(model, {head: (6, 4)})
    with pytest.raises(ValueError, match="coverage"):
        experiment.audit_checkpoint(model, {legacy: (4, 4)}, full=True)
    assert experiment.audit_checkpoint(model, model, full=True) == []
