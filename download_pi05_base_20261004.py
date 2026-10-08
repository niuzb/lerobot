"""Prefetch public Pi05 weights while tokenizer access is being enabled."""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

revision = HfApi().model_info('lerobot/pi05_base').sha
print('DOWNLOADING lerobot/pi05_base', revision, flush=True)
path = snapshot_download('lerobot/pi05_base', revision=revision, allow_patterns=['config.json', 'model.safetensors'])
assert (Path(path) / 'model.safetensors').stat().st_size > 1_000_000_000
Path('/root/code/lerobot/pi05_base_20261004.download.json').write_text(json.dumps(dict(revision=revision, path=path), indent=2))
print('BASE_WEIGHTS_READY', path, flush=True)
