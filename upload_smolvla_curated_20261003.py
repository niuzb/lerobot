import os
from pathlib import Path
import hashlib
from huggingface_hub import HfApi

os.environ['HTTPS_PROXY'] = 'http://sys-proxy-rd-relay.byted.org:8118'
os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY']
root = Path('/root/code/lerobot/outputs/train/smolvla_tape_curated_20261002/checkpoints/020000/pretrained_model')
repo = 'zhniu/smolvla_tape_curated_20261002'
api = HfApi()
assert api.whoami()['name'] == 'zhniu'
api.create_repo(repo, repo_type='model', exist_ok=True)
commit = api.upload_folder(repo_id=repo, folder_path=root, commit_message='Upload SmolVLA trained for 20000 steps on 88 curated demonstrations')
print('UPLOADED', commit.oid, flush=True)
remote = {f.rfilename: f for f in api.model_info(repo, revision=commit.oid, files_metadata=True).siblings}
for path in root.rglob('*'):
    if not path.is_file():
        continue
    name = path.relative_to(root).as_posix()
    assert name in remote, name
    assert remote[name].size == path.stat().st_size, name
    if remote[name].lfs:
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == remote[name].lfs.sha256, name
print('HUB_VERIFIED', repo, commit.oid, flush=True)
