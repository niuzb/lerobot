"""Verify the published Pi05 commit against the final local checkpoint."""
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

os.environ['HTTPS_PROXY'] = 'http://sys-proxy-rd-relay.byted.org:8118'
os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY']
base = Path('/root/code/lerobot')
root = base / 'outputs/train/pi05_tape_curated_20261004/checkpoints/080000/pretrained_model'
repo = 'zhniu/pi05_tape_curated_20261004'
api = HfApi()
info = api.model_info(repo, files_metadata=True)
revision = info.sha
remote = {file.rfilename: file for file in info.siblings}
manifest = {}
for path in sorted(root.rglob('*')):
    if not path.is_file():
        continue
    name = path.relative_to(root).as_posix()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert name in remote and remote[name].size == path.stat().st_size, name
    if remote[name].lfs:
        assert remote[name].lfs.sha256 == digest, name
    else:
        downloaded = hf_hub_download(repo, filename=name, revision=revision)
        with open(downloaded, 'rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
    manifest[name] = dict(size=path.stat().st_size, sha256=digest)
    print('VERIFIED', name, flush=True)
assert 'README.md' in remote
result = dict(repo_id=repo, commit=revision, url=f'https://huggingface.co/{repo}', files=manifest, verified=True)
(base / 'pi05_tape_curated_20261004.upload_verified.json').write_text(json.dumps(result, indent=2))
print('HUB_VERIFIED', repo, revision, len(manifest), 'model files', flush=True)
