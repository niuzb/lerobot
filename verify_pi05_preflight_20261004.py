"""Require a real finite optimizer update and a complete preflight checkpoint."""
import json
import math
import re
from pathlib import Path

base = Path('/root/code/lerobot')
log = (base / 'pi05_tape_curated_20261004.preflight.log').read_text()
assert 'All keys loaded successfully!' in log
assert 'End of training' in log
losses = re.findall(r'loss:([\w.+-]+)', log)
assert losses and all(math.isfinite(float(loss)) for loss in losses), losses
checkpoint = base / 'outputs/train/pi05_tape_curated_20261004_preflight_v3/checkpoints/000004'
model = checkpoint / 'pretrained_model'
for name in ['model.safetensors', 'config.json', 'train_config.json', 'policy_preprocessor.json', 'policy_postprocessor.json']:
    assert (model / name).is_file() and (model / name).stat().st_size > 0, name
config = json.loads((model / 'config.json').read_text())
assert config['type'] == 'pi05'
assert config['input_features']['observation.state']['shape'] == [6]
assert config['output_features']['action']['shape'] == [6]
state = checkpoint / 'training_state'
assert (state / 'optimizer_state.safetensors').is_file()
assert (state / 'optimizer_state.safetensors').stat().st_size > 1_000_000
print('PREFLIGHT_CHECKPOINT_OK', checkpoint, 'finite losses', losses, flush=True)
