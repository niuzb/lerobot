"""Inventory only regular audio files under the user-authorized train_data tree."""
import collections
import json
import os
import stat
import sys
import time

root = os.path.realpath('/root/train_data')
assert root == '/mnt/rootfs', root
extensions = {'.wav', '.flac', '.mp3', '.ogg', '.opus', '.m4a', '.aac', '.wma', '.aif', '.aiff', '.au'}
counts = collections.defaultdict(lambda: [0, 0])
groups = collections.defaultdict(lambda: [0, 0])
samples = collections.defaultdict(list)
errors = []
total = 0
start = time.monotonic()
for folder, dirs, files in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e))):
    dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(folder, d))]
    for name in files:
        ext = os.path.splitext(name)[1].lower()
        if ext not in extensions:
            continue
        path = os.path.join(folder, name)
        try:
            st = os.lstat(path)
            if not stat.S_ISREG(st.st_mode):
                continue
            counts[ext][0] += 1
            counts[ext][1] += st.st_size
            group = os.path.relpath(path, root).split(os.sep)[0]
            groups[group][0] += 1
            groups[group][1] += st.st_size
            total += 1
            if len(samples[ext]) < 3:
                with open(path, 'rb') as f:
                    header = f.read(16).hex()
                samples[ext].append({'path': path, 'header': header})
        except OSError as e:
            errors.append(str(e))
    if time.monotonic() - start > 30:
        print(json.dumps({'progress_audio_files': total}), flush=True)
        start = time.monotonic()
print(json.dumps({'root': root, 'device': os.stat(root).st_dev,
                  'training_device': os.stat('/root/code/lerobot').st_dev,
                  'extensions': counts, 'directories': groups, 'samples': samples,
                  'errors': errors[:20]}, ensure_ascii=False), flush=True)
