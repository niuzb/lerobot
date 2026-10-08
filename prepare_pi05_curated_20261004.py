"""Check the curated dataset and fetch pinned Pi05 assets before training."""
import hashlib
import json
import shutil
from pathlib import Path

import torch
import transformers
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType
from lerobot.utils.feature_utils import dataset_to_policy_features
from transformers import AutoTokenizer

base = Path('/root/code/lerobot')
root = base / 'datasets/smolvla_tape_clean_20261002_142151_curated_v1'
print('ENV', torch.__version__, transformers.__version__, flush=True)
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
print('GPU', torch.cuda.get_device_name(0), flush=True)
archive = base / 'incoming/smolvla_tape_clean_20261002_142151_curated_v1.tar'
with archive.open('rb') as handle:
    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
assert digest == '6ca6809526d514499a83c602949f96c52a7176df2663efc8d89c48ce20bf6ab5'
stats = json.loads((root / 'meta/stats.json').read_text())
for key in ['observation.state', 'action']:
    assert 'q01' in stats[key] and 'q99' in stats[key]
dataset = LeRobotDataset('zhniu/smolvla_tape_clean_20261002_142151_curated_v1', root=root, video_backend='pyav')
assert len(dataset) == 27632 and dataset.meta.total_episodes == 88
for index in [0, len(dataset) // 2, len(dataset) - 1]:
    row = dataset[index]
    for key in ['observation.state', 'action']:
        assert row[key].shape == (6,) and torch.isfinite(row[key]).all()
    for key in ['observation.images.front', 'observation.images.side']:
        assert row[key].shape == (3, 480, 640)
print('DATASET_OK 88 episodes 27632 frames; quantiles present', flush=True)
api = HfApi()
model_revision = api.model_info('lerobot/pi05_base').sha
tokenizer_revision = api.model_info('google/paligemma-3b-pt-224').sha
# Check gated tokenizer access before downloading the large weights.
hf_hub_download('google/paligemma-3b-pt-224', 'tokenizer_config.json', revision=tokenizer_revision)
model_path = snapshot_download('lerobot/pi05_base', revision=model_revision, allow_patterns=['config.json', 'model.safetensors'])
tokenizer_path = snapshot_download('google/paligemma-3b-pt-224', revision=tokenizer_revision, allow_patterns=['tokenizer*', 'special_tokens_map.json', 'added_tokens.json'])
AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
assert (Path(model_path) / 'model.safetensors').stat().st_size > 1_000_000_000
prepared_path = base / 'outputs/pi05_tape_curated_20261004_base_assets'
prepared_path.mkdir(parents=True, exist_ok=True)
weight_link = prepared_path / 'model.safetensors'
if not weight_link.exists():
    weight_link.symlink_to(Path(model_path) / 'model.safetensors')
shutil.copyfile(Path(model_path) / 'config.json', prepared_path / 'config.json')
features = dataset_to_policy_features(dataset.meta.features)
config = PI05Config(device='cuda', dtype='bfloat16', empty_cameras=1, text_tokenizer_name=tokenizer_path)
config.output_features = {key: value for key, value in features.items() if value.type is FeatureType.ACTION}
config.input_features = {key: value for key, value in features.items() if key not in config.output_features}
config.validate_features()
preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
preprocessor.save_pretrained(prepared_path)
postprocessor.save_pretrained(prepared_path)
model_path = str(prepared_path)
assets = dict(model_path=model_path, model_revision=model_revision, tokenizer_path=tokenizer_path, tokenizer_revision=tokenizer_revision, dataset_archive_sha256=digest)
(base / 'pi05_tape_curated_20261004.assets.json').write_text(json.dumps(assets, indent=2))
print('ASSETS_READY', json.dumps(assets), flush=True)
