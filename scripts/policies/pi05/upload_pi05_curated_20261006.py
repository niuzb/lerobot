"""Upload the final Pi05 artifact and verify every file at the resulting commit."""
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

os.environ['HTTPS_PROXY'] = 'http://sys-proxy-rd-relay.byted.org:8118'
os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY']
base = Path('/root/code/lerobot')
root = base / 'outputs/train/pi05_tape_curated_20261004/checkpoints/080000/pretrained_model'
repo = 'zhniu/pi05_tape_curated_20261004'
assert (base / 'pi05_tape_curated_20261004.formal.exitcode').read_text().strip() == '0'
config = json.loads((root / 'config.json').read_text())
assert config['type'] == 'pi05'
assert config['output_features']['action']['shape'] == [6]
api = HfApi()
assert api.whoami()['name'] == 'zhniu'
files = sorted(path for path in root.rglob('*') if path.is_file())
assert any(path.name == 'tokenizer.json' for path in files)
manifest = {}
for path in files:
    name = path.relative_to(root).as_posix()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest[name] = dict(size=path.stat().st_size, sha256=digest)
    print('HASHED', name, path.stat().st_size, digest, flush=True)
card = '''---
library_name: lerobot
pipeline_tag: robotics
base_model: lerobot/pi05_base
tags:
- pi05
- so101
- robotics
---

# Pi05 trained on curated SO-101 demonstrations

Fine-tuned from `lerobot/pi05_base` on the same newly recorded, cleaned local
dataset used for `zhniu/smolvla_tape_curated_20261002`.

- Dataset ID: `zhniu/smolvla_tape_clean_20261002_142151_curated_v1` (local dataset;
  this ID does not imply that the dataset is published on the Hub).
- 88 episodes, 27,632 frames, 30 FPS; two 640x480 RGB views.
- `observation.images.front`: wrist camera; `observation.images.side`: external camera.
- Six-dimensional SO-101 state/action; absolute actions, recorded units retained.
- STATE and ACTION use QUANTILES normalization; VISUAL uses IDENTITY.
- One masked empty camera, chunk size 50, n_action_steps 30.
- Frozen vision-language backbone; action expert fine-tuning, BF16,
  gradient checkpointing; batch 4 x accumulation 4 = effective batch 16.
- 80,000 microbatch steps = 20,000 optimizer updates, seed 1000.
- Training completed on 2026-10-05 at 20:51:37 Asia/Shanghai; exit code 0.
- Final training loss: 0.029. No held-out validation or real-robot success-rate
  evaluation has been performed.
- Base revision: `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`.
- Tokenizer revision: `35e4f46485b4d07967e7e9935bc3786aad50687c`.

This repository contains the final pretrained_model artifact, including weights,
configuration, preprocessing/postprocessing pipelines, normalization statistics,
and bundled tokenizer. Optimizer/RNG state is retained on the training host and
is not included here. The training config records original host-specific paths.
Load the model and saved processor pipelines from this Hub repository for inference.
'''
# Match the earlier SmolVLA publication under the same account.
api.create_repo(repo, repo_type='model', exist_ok=True)
card_path = base / 'outputs/pi05_model_card_20261006.md'
card_path.write_text(card)
operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(root / name)) for name in manifest]
operations.append(CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=str(card_path)))
commit = api.create_commit(repo_id=repo, operations=operations, commit_message='Upload final Pi05: 20000 optimizer updates on 88 curated SO-101 demonstrations')
print('UPLOADED', repo, commit.oid, flush=True)
remote = {f.rfilename: f for f in api.model_info(repo, revision=commit.oid, files_metadata=True).siblings}
for name, expected in manifest.items():
    assert name in remote and remote[name].size == expected['size'], name
    if remote[name].lfs:
        assert remote[name].lfs.sha256 == expected['sha256'], name
    else:
        content = hf_hub_download(repo, filename=name, revision=commit.oid)
        with open(content, 'rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256'], name
result = dict(repo_id=repo, commit=commit.oid, url=f'https://huggingface.co/{repo}', files=manifest, verified=True)
(base / 'pi05_tape_curated_20261004.upload_verified.json').write_text(json.dumps(result, indent=2))
print('HUB_VERIFIED', repo, commit.oid, len(manifest), 'model files', flush=True)
