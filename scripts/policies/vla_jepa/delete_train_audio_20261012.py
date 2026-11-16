"""Delete the WAV/FLAC files explicitly authorized under /root/train_data."""
import concurrent.futures
import json
import os
import stat
import time

ROOT = os.path.realpath('/root/train_data')
assert ROOT == '/mnt/rootfs' and os.path.isdir(ROOT), ROOT


def candidates():
    for folder, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(folder, d))]
        for name in files:
            if os.path.splitext(name)[1].lower() in {'.wav', '.flac'}:
                yield os.path.join(folder, name)


def remove_audio(path):
    try:
        resolved = os.path.realpath(path)
        assert os.path.commonpath([ROOT, resolved]) == ROOT and resolved != ROOT
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode):
            return (0, 0, 'skipped non-regular: ' + path)
        os.unlink(path)
        return (1, st.st_size, None)
    except Exception as exc:
        return (0, 0, str(exc) + ': ' + path)


if __name__ == '__main__':
    count = size = 0
    errors = []
    last = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for n, b, error in pool.map(remove_audio, candidates()):
            count += n
            size += b
            if error:
                errors.append(error)
            if time.monotonic() - last > 20:
                print(json.dumps({'deleted_files': count, 'deleted_bytes': size}), flush=True)
                last = time.monotonic()
    remaining = sum(1 for _ in candidates())
    print(json.dumps({'root': ROOT, 'deleted_files': count, 'deleted_bytes': size,
                      'remaining_audio_files': remaining, 'errors': errors[:30],
                      'error_count': len(errors)}), flush=True)
