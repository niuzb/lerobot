"""Fetch pinned public assets and prepare a reproducible FastWAM job without a GPU."""

import argparse
import fnmatch
import hashlib
import json
import os
import time
from pathlib import Path


def write_json(path, value):
    """Replace a status file atomically."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def main():
    """Download public files and verify hashes from the official repository metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["job_root"])
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    from huggingface_hub import snapshot_download

    # Expected file hashes come from the official HF endpoint, separately retrieved locally.
    locked = json.loads((root / "model_files.lock.json").read_text())
    status = {"completed": False, "started_unix": time.time(), "models": {}}
    write_json(root / "preparation_status.json", status)
    try:
        for spec in config["models"]:
            status["stage"] = "download_" + spec["role"]
            write_json(root / "preparation_status.json", status)
            target = root / "assets" / spec["role"]
            snapshot_download(
                spec["repo"],
                revision=spec["revision"],
                local_dir=target,
                allow_patterns=spec["patterns"],
                endpoint=args.endpoint,
                token=False,
                max_workers=2,
            )
            verified = {}
            for item in locked[spec["repo"]]["files"]:
                name = item["path"]
                if not any(fnmatch.fnmatch(name, pat) for pat in spec["patterns"]):
                    continue
                path = target / name
                if not path.is_file() or path.stat().st_size != item["size"]:
                    raise ValueError(f"Missing or incomplete model file: {name}")
                algorithm = "sha256" if item["lfs_sha256"] else "sha1"
                h = hashlib.new(algorithm)
                if algorithm == "sha1":
                    h.update(f"blob {path.stat().st_size}\0".encode())
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        h.update(block)
                expected = item["lfs_sha256"] or item["git_blob_sha1"]
                if h.hexdigest() != expected:
                    raise ValueError(f"Official model hash mismatch: {name}")
                verified[name] = expected
            status["models"][spec["role"]] = {
                "repo": spec["repo"],
                "revision": spec["revision"],
                "path": str(target),
                "verified_files": verified,
            }
            write_json(root / "preparation_status.json", status)
        status.update(completed=True, stage="assets_verified", finished_unix=time.time())
    except Exception as exc:
        status.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(root / "preparation_status.json", status)


if __name__ == "__main__":
    main()
