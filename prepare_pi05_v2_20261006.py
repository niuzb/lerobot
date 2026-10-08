"""Build an isolated dataset view with train-only joint statistics and audit the split."""
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

BASE = Path('/root/code/lerobot')
source = BASE / 'datasets/smolvla_tape_clean_20261002_142151_curated_v1'
target = BASE / 'datasets/pi05_tape_v2_20261006'
assert not target.exists(), f'Refusing to overwrite {target}'
table = pq.read_table(sorted((source / 'data').rglob('*.parquet')))
episodes = np.asarray(table['episode_index'])
assert sorted(set(episodes.tolist())) == list(range(88))
assert len(set(table['task_index'].to_pylist())) == 1, 'Review per-task split before training'
train_ids, eval_ids = list(range(79)), list(range(79, 88))
mask = np.isin(episodes, train_ids)
assert mask.any() and (~mask).any()
stats = json.loads((source / 'meta/stats.json').read_text())
audit = {'train_episodes': train_ids, 'eval_episodes': eval_ids,
         'train_frames': int(mask.sum()), 'eval_frames': int((~mask).sum()),
         'source': str(source), 'initialization': 'original pinned pi05_base, not v1',
         'eval_split': 0.1, 'normalization': 'train-only exact joint quantiles'}
for key in ('action', 'observation.state'):
    values = np.asarray(table[key].to_pylist(), dtype=np.float64)
    assert values.shape == (27632, 6) and np.isfinite(values).all()
    train = values[mask]
    stats[key] = {name: value.tolist() for name, value in {
        'min': train.min(0), 'max': train.max(0), 'mean': train.mean(0),
        'std': train.std(0), 'count': np.array([len(train)]),
        **{f'q{int(q*100):02d}': np.quantile(train, q, axis=0)
           for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
    }.items()}
    same_episode = episodes[1:] == episodes[:-1]
    speeds = np.abs(np.diff(values, axis=0)[same_episode]) * 30
    audit[key] = {'speed_p50': np.quantile(speeds, .5, axis=0).tolist(),
                  'speed_p95': np.quantile(speeds, .95, axis=0).tolist(),
                  'fraction_above_5_units_per_second': (speeds > 5).mean(0).tolist()}
target.mkdir()
shutil.copytree(source / 'meta', target / 'meta')
for name in ('data', 'videos'):
    (target / name).symlink_to(source / name, target_is_directory=True)
(target / 'meta/stats.json').write_text(json.dumps(stats, indent=2))
(BASE / 'pi05_v2_20261006.audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2), flush=True)
