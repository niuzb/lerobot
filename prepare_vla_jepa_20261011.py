"""Prepare pinned VLA-JEPA assets and SO101 processors without touching robot hardware."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.utils.feature_utils import dataset_to_policy_features

BASE = Path('/root/code/lerobot')
RUN = 'vla_jepa_tape_20261011'
launch = BASE / 'outputs/vla_jepa_launch_20261011'
assets = json.loads((launch / 'asset_revisions.json').read_text())
def fetch_asset(asset):
    print('FETCH', asset['repo_id'], asset['revision'], flush=True)
    path = snapshot_download(
        asset['repo_id'], revision=asset['revision'],
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'],
    )
    return asset['repo_id'], path

with ThreadPoolExecutor(max_workers=3) as pool:
    paths = dict(pool.map(fetch_asset, assets))

# Reuse v2's isolated metadata with train-only state/action statistics.
root = BASE / 'datasets/pi05_tape_v2_20261006'
dataset = LeRobotDataset('zhniu/smolvla_tape_clean_20261002_142151_curated_v1',
                         root=root, video_backend='pyav')
assert len(dataset) == 27632 and dataset.meta.total_episodes == 88
features = dataset_to_policy_features(dataset.meta.features)
source = Path(paths['lerobot/VLA-JEPA-Pretrain'])
config = VLAJEPAConfig.from_pretrained(source)
config.input_features = {k: v for k, v in features.items() if v.type is not FeatureType.ACTION}
config.output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
config.action_feature_names = dataset.meta.features['action']['names']
config.qwen_model_name = paths['Qwen/Qwen3-VL-2B-Instruct']
config.jepa_encoder_name = paths['facebook/vjepa2-vitl-fpc64-256']
config.freeze_qwen = False
config.enable_world_model = True
config.causal_world_model_context = True
config.world_model_num_views = 2
config.binarize_gripper_action = False
config.pre_snap_gripper_action = False
config.gripper_dim = 5
config.use_relative_actions = False
config.resize_images_to = (224, 224)
config.reinit_modules = ['model.action_model.action_encoder',
                         'model.action_model.action_decoder',
                         'model.action_model.state_encoder']
config.device = 'cuda'
config.push_to_hub = False
config.repo_id = 'zhniu/' + RUN
config.optimizer_lr = 1e-5
config.scheduler_warmup_steps = 2000
config.scheduler_decay_steps = 40000
config.validate_features()
assert config.action_dim == config.state_dim == 6
assert config.resolved_gripper_dim == 5 and config.enable_world_model
assert list(config.image_features) == ['observation.images.front', 'observation.images.side']
target = BASE / 'outputs' / (RUN + '_base_assets')
target.mkdir(exist_ok=True)
config.save_pretrained(target)
weight = source / 'model.safetensors'
assert weight.is_file() and weight.stat().st_size > 1_000_000_000
link = target / weight.name
if not link.exists():
    link.symlink_to(weight)
pre, post = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
pre.save_pretrained(target)
post.save_pretrained(target)
sample = dataset[0]
assert torch.isfinite(sample['action']).all()
(launch / 'prepared.json').write_text(json.dumps({
    'assets': assets, 'paths': paths, 'model_path': str(target),
    'dataset_root': str(root), 'train_episodes': list(range(79)),
    'eval_episodes': list(range(79, 88)), 'world_model_enabled': True,
    'causal_world_model_context': True, 'action_dim': 6, 'state_dim': 6,
    'gripper_dim': 5, 'physical_gripper_range': [0, 100],
}, indent=2))
print('JEPA_ASSETS_READY', str(target), flush=True)
