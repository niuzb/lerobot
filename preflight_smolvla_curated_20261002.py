import json
import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

root = Path('/root/code/lerobot/datasets/smolvla_tape_clean_20261002_142151_curated_v1')
print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available()
print('GPU', torch.cuda.get_device_name(0), flush=True)
ds = LeRobotDataset('zhniu/smolvla_tape_clean_20261002_142151_curated_v1', root=root, video_backend='pyav')
assert len(ds) == 27632
for i in [0, len(ds)//2, len(ds)-1]:
    row = ds[i]
    assert row['observation.state'].shape == (6,)
    assert row['action'].shape == (6,)
    assert torch.isfinite(row['observation.state']).all() and torch.isfinite(row['action']).all()
    for key in ['observation.images.front', 'observation.images.side']:
        assert row[key].shape == (3,480,640)
print('DATASET_OK', len(ds), 'frames', ds.meta.total_episodes, 'episodes', flush=True)
