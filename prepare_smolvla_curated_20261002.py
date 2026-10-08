"""Verify the local-to-Merlin archive before extracting into a new dataset root."""
import hashlib
import json
import tarfile
from pathlib import Path

name = 'smolvla_tape_clean_20261002_142151_curated_v1'
archive = Path('/root/code/lerobot/incoming') / f'{name}.tar'
parent = Path('/root/code/lerobot/datasets')
dest = parent / name
expected = '6ca6809526d514499a83c602949f96c52a7176df2663efc8d89c48ce20bf6ab5'
with archive.open('rb') as f:
    actual = hashlib.file_digest(f, 'sha256').hexdigest()
assert actual == expected, f'Archive checksum mismatch: {actual}'
print('ARCHIVE_SHA256_OK', actual, flush=True)
assert not dest.exists(), f'Refusing to overwrite existing dataset: {dest}'
parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        parts = Path(member.name).parts
        assert parts and parts[0] == name and '..' not in parts and not Path(member.name).is_absolute()
        assert member.isfile() or member.isdir()
    tar.extractall(parent, filter='data')
info = json.loads((dest / 'meta/info.json').read_text())
assert info['total_episodes'] == 88 and info['total_frames'] == 27632
print('EXTRACTED', dest, info['total_episodes'], info['total_frames'], flush=True)
