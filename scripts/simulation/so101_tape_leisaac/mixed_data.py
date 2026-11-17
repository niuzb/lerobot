"""Leakage-free real/simulation datasets and exact 4+4 minibatches."""

import copy
import math
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from common import TASK
from eval_real_loss import SpatialRealDataset, build_spatial_rows, save
from torch.utils.data import BatchSampler, Dataset, SequentialSampler

REAL_REPO = "zhniu/smolvla_tape_clean_20261002_142151_curated_v1"
SIM_REPO = "local/so101_tape100_framewise_20261109"
RENAME = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.side": "observation.images.camera2",
}


def real_split(seed=20261110):
    validation = sorted(random.Random(seed).sample(range(88), 10))
    training = sorted(set(range(88)) - set(validation))
    return training, validation


def select_episodes(table, episodes):
    return table.filter(pc.is_in(table["episode_index"], value_set=pa.array(episodes)))


def balanced_moments(real, simulated):
    """Moments of the 50/50 source mixture actually sampled by the optimizer."""
    real, simulated = np.asarray(real, dtype=np.float64), np.asarray(simulated, dtype=np.float64)
    if real.ndim != 2 or simulated.ndim != 2 or real.shape[1] != simulated.shape[1]:
        raise ValueError("Training moment arrays must have matching feature dimensions")
    if not len(real) or not len(simulated) or not np.isfinite(real).all() or not np.isfinite(simulated).all():
        raise ValueError("Training statistics require finite nonempty domains")
    rmean, smean = real.mean(0), simulated.mean(0)
    mean = (rmean + smean) / 2
    variance = (real.var(0) + simulated.var(0)) / 2 + (rmean - smean) ** 2 / 4
    return {
        "mean": mean,
        "std": np.maximum(np.sqrt(variance), 1e-6),
        "min": np.minimum(real.min(0), simulated.min(0)),
        "max": np.maximum(real.max(0), simulated.max(0)),
        "count": np.array([len(real) + len(simulated)]),
    }


class BalancedBatchSampler(BatchSampler):
    """Four shuffled real and four shuffled simulated frame anchors in every batch.

    The smaller domain is reshuffled/repeated when exhausted. Epoch and resume
    order are deterministic, independent of torch/flow-matching RNG consumption.
    """

    def __init__(self, nreal, nsim, seed=20261110, start_step=0):
        if min(nreal, nsim) < 4:
            raise ValueError("Each source needs at least four samples")
        super().__init__(SequentialSampler(range(nreal + nsim)), batch_size=8, drop_last=True)
        self.nreal, self.nsim, self.seed = nreal, nsim, seed
        self.batches_per_epoch = math.ceil(max(nreal, nsim) / 4)
        self.epoch, self.offset = divmod(start_step, self.batches_per_epoch)

    def __len__(self):
        return self.batches_per_epoch - self.offset

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        real, sim = [], []

        def take(pool, size):
            if len(pool) < 4:
                pool[:0] = rng.sample(range(size), size)
            return [pool.pop() for _ in range(4)]

        offset = self.offset
        self.epoch += 1
        self.offset = 0
        for step in range(self.batches_per_epoch):
            batch = take(real, self.nreal) + [self.nreal + i for i in take(sim, self.nsim)]
            rng.shuffle(batch)
            if step >= offset:
                yield batch


class PolicyRows(Dataset):
    """Return a uniform schema for both domains, including an audit-only flag."""

    def __init__(self, source, domain, training):
        self.source, self.domain, self.training = source, domain, training

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        item = self.source[index]
        keys = ["observation.state", *RENAME, "action", "action_is_pad", "episode_index", "index"]
        out = {k: item[k] for k in keys}
        out["task"] = TASK
        out["_domain"] = self.domain  # 0 real, 1 simulation; never a policy input.
        out["_mixed_training"] = self.training
        return out


class MixedDataset(Dataset):
    def __init__(self, real, simulated, metadata):
        self.real, self.simulated, self.meta = real, simulated, metadata
        self.num_frames = len(self)
        self.num_episodes = 178
        self.episodes = list(range(178))
        self.repo_id = "local/so101_real78_sim100_balanced"

    def __len__(self):
        return len(self.real) + len(self.simulated)

    def __getitem__(self, index):
        return self.real[index] if index < len(self.real) else self.simulated[index - len(self.real)]


def make_mixed_datasets(cfg, paths, output):
    import json

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    train_ids, validation_ids = real_split()
    assert not set(train_ids) & set(validation_ids) and len(train_ids) == 78
    real_root, sim_root = Path(paths["real_root"]), Path(paths["sim_root"])
    real_table = pq.read_table(sorted((real_root / "data").rglob("*.parquet")))
    sim_table = pq.read_table(sorted((sim_root / "data").rglob("*.parquet")))
    if set(real_table["episode_index"].to_pylist()) != set(range(88)):
        raise ValueError("Real source must be the original 88 episodes")
    if set(sim_table["episode_index"].to_pylist()) != set(range(100)):
        raise ValueError("Simulation source must contain all 100 verified episodes")
    scene = json.loads(Path(paths["scene"]).read_text())
    auxiliary, raw_states = build_spatial_rows(scene, real_table)
    real_train = select_episodes(real_table, train_ids)
    real_eval = select_episodes(real_table, validation_ids)
    real_state = np.asarray(real_train["observation.state"].to_pylist(), dtype=np.float32)
    extras = np.asarray([auxiliary[i] for i in real_train["index"].to_pylist()])
    real_state = np.concatenate([real_state, extras], axis=1)
    assert real_state.shape == (24528, 25) and real_eval.num_rows == 3104
    stats = {
        "observation.state": balanced_moments(real_state, sim_table["observation.state"].to_pylist()),
        "action": balanced_moments(real_train["action"].to_pylist(), sim_table["action"].to_pylist()),
    }
    report = {
        "scope": "training episodes only; 50/50 real/simulation domain mixture",
        "real_training_episodes": train_ids,
        "real_validation_episodes": validation_ids,
        "sim_training_episodes": list(range(100)),
        "real_training_frames": real_train.num_rows,
        "real_validation_frames": real_eval.num_rows,
        "sim_training_frames": sim_table.num_rows,
        "validation_excluded": True,
        "calibration_passed": False,
        "spatial_method": "same archived camera estimates / FK / table baseline as prior real10 evaluation",
        "stats": {
            k: {name: value.tolist() for name, value in entries.items()} for k, entries in stats.items()
        },
    }
    save(output / "training_stats.json", report)
    real_meta = LeRobotDatasetMetadata(REAL_REPO, root=real_root)
    real_delta = resolve_delta_timestamps(cfg.policy, real_meta, RENAME)
    sim_meta = LeRobotDatasetMetadata(SIM_REPO, root=sim_root)
    sim_delta = resolve_delta_timestamps(cfg.policy, sim_meta, RENAME)
    common = {"video_backend": "pyav", "return_uint8": True}
    real_source = LeRobotDataset(
        REAL_REPO, root=real_root, episodes=train_ids, delta_timestamps=real_delta, **common
    )
    eval_source = LeRobotDataset(
        REAL_REPO, root=real_root, episodes=validation_ids, delta_timestamps=real_delta, **common
    )
    sim_source = LeRobotDataset(
        SIM_REPO, root=sim_root, episodes=list(range(100)), delta_timestamps=sim_delta, **common
    )
    real = PolicyRows(SpatialRealDataset(real_source, auxiliary, raw_states), 0, True)
    evaluation = PolicyRows(SpatialRealDataset(eval_source, auxiliary, raw_states), 0, False)
    simulated = PolicyRows(sim_source, 1, True)
    metadata = copy.deepcopy(sim_source.meta)
    metadata.stats = stats
    mixed = MixedDataset(real, simulated, metadata)
    assert len(real) == real_train.num_rows and len(evaluation) == real_eval.num_rows
    assert len(simulated) == sim_table.num_rows
    return mixed, evaluation
