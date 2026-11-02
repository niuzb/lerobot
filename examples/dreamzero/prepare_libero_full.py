"""Verify immutable LIBERO/Wan inputs and prepare a leakage-free full-training experiment."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.dreamzero.checkpoint import weight_files
from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig

from .libero_full_utils import file_hash, split_episodes, validation_indices, write_json


def _task(suite_name, task_id):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    states = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    return task, torch.load(states, map_location="cpu", weights_only=False), bddl


def make_config(job, camera_keys, image_shapes):
    """Construct a fresh full policy, never loading an earlier robot checkpoint."""
    original = {
        "backbone": "wan22_5b",
        "train_mode": "full",
        "init_kind": "wan",
        "init_from": job["wan_dit_path"],
        "init_revision": job["wan_dit_revision"],
        **job["component_paths"],
        "camera_keys": camera_keys,
        "camera_layout": "horizontal",
        "image_height": 160,
        "image_width": 320,
        "action_dim": 7,
        "state_dim": 8,
        "relative_action_indices": [],
        "fps": 20,
        "video_stride": 2,
        "num_frame_per_block": 2,
        "num_action_per_block": 16,
        "training_chunks": 4,
        "max_cache_chunks": 4,
        "n_action_steps": 16,
        "use_gradient_checkpointing": True,
        "optimizer_lr": 1e-5,
        "optimizer_weight_decay": 1e-5,
        "scheduler_warmup_steps": 1000,
        "text_encoder_device": "cuda",
        "pretrained_path": None,
        "device": "cuda",
        "push_to_hub": False,
    }
    cfg = DreamZeroConfig(**original)
    # Preparation is CPU-only; the sealed config targets the retained CUDA worker.
    cfg.device = "cuda"
    cfg.input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=image_shapes[key]) for key in cfg.camera_keys
    }
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    return cfg


def prepare(root, previous_root, protocol_manifest, reference_root):
    """Verify assets, fix an episode split, and compute training-only statistics."""
    root, previous = Path(root).resolve(), Path(previous_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "READY").exists():
        raise FileExistsError("Preparation is already sealed; do not silently change an active run")
    write_json(root / "preparation_status.json", {"stage": "verify_assets", "started_unix": time.time()})
    reference = Path(reference_root).resolve()
    specs = json.loads((previous / "assets.lock.json").read_text())
    verified = []
    # The earlier DreamZero preparation pinned raw Wan/CLIP and all four LIBERO
    # suites; FastWAM's lock owns the shared VAE/T5/tokenizer assets.
    shared_specs = [s for s in json.loads((reference / "assets.lock.json").read_text())
                    if s["repo"].startswith("Wan-AI/")]
    for base, spec in [(previous, s) for s in specs] + [(reference, s) for s in shared_specs]:
        path = base / "assets" / spec["repo"] / spec["filename"]
        assert path.is_file() and path.stat().st_size == spec["size"], str(path)
        actual = file_hash(path, "sha256" if spec["sha256"] else "sha1", not bool(spec["sha256"]))
        assert actual == (spec["sha256"] or spec["git_blob_sha1"]), str(path)
        verified.append({"path": str(path), "revision": spec["revision"], "digest": actual})
    write_json(root / "assets_verified.json", verified)
    wan_revision = DreamZeroConfig().wan_revision
    wan_dit_path = previous / "assets/Wan-AI/Wan2.2-TI2V-5B"
    dit_files, _ = weight_files(wan_dit_path)
    locked_dit = {s["filename"] for s in specs if s["repo"] == "Wan-AI/Wan2.2-TI2V-5B"
                  and s["revision"] == wan_revision}
    if not {p.name for p in dit_files}.issubset(locked_dit):
        raise ValueError("Original Wan DiT shards are not all pinned in the previous asset lock")
    if not any(wan_dit_path.glob("*.safetensors.index.json")) and len(dit_files) != 1:
        raise ValueError("Wan 5B DiT must have a single weight file or a shard index")
    dit_lock = [{"path": str(path), "sha256": file_hash(path), "size": path.stat().st_size}
                for path in sorted(set(dit_files + list(wan_dit_path.glob("*.safetensors.index.json"))))]
    if not dit_lock:
        raise ValueError("Pinned Wan 5B snapshot contains no DiT weights")
    write_json(root / "wan_dit_verified.json", {"repo": "Wan-AI/Wan2.2-TI2V-5B",
        "revision": wan_revision, "directory": str(wan_dit_path), "files": dit_lock})
    protocol_path = Path(protocol_manifest).resolve()
    protocol = json.loads(protocol_path.read_text())
    if (protocol.get("protocol_version") != 1 or len(protocol.get("tasks", [])) != 40
            or protocol.get("episodes") != 2000 or protocol.get("trials_per_task") != 50):
        raise ValueError("Expected the audited 2,000-episode LIBERO v1 evaluation manifest")
    for name, digest in protocol["libero_source_hashes"].items():
        if file_hash(reference / name) != digest:
            raise ValueError(f"LIBERO source changed since FastWAM: {name}")
    for record in protocol["tasks"]:
        suite, task_id = record["suite"], record["task_id"]
        task, states, bddl = _task(suite, task_id)
        if len(states) != record["initial_states"] or file_hash(bddl) != record["bddl_sha256"]:
            raise ValueError(f"LIBERO BDDL or initial states changed for {suite}/{task_id}")
        state_file = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        state_digest = hashlib.sha256(state_file.read_bytes()).hexdigest()
        if state_digest != record["initial_states_sha256"]:
            raise ValueError(f"LIBERO initial-state hash changed for {suite}/{task_id}")
    protocol_sha256 = file_hash(protocol_path)
    write_json(root / "protocol_verified.json", {"path": str(protocol_path), "sha256": protocol_sha256})
    job = {"root": str(root), "previous_root": str(previous), "dataset_root": str(previous / "dataset"),
           "repo_id": "local/dreamzero_libero_40", "seed": 1000,
           "dataset_revision": None,
           "dataset_source_revision": None,
           "protocol_manifest": str(protocol_path), "protocol_sha256": protocol_sha256,
           "optimizer_updates": 20000, "accumulation": 16, "micro_batch": 1,
           "warmup_updates": 1000, "lr": 1e-5, "weight_decay": 1e-5, "grad_clip": 1.0,
           "save_every": 1000, "eval_updates": [2000, 5000, 10000, 20000],
           "worker_id": "4434805", "gpu_memory_limit_gib": 72, "host_memory_limit_gib": 90,
           "deepspeed_version": None, "release_gpu": False, "push_to_hub": False}
    job["wan_dit_path"] = str(wan_dit_path)
    job["wan_dit_revision"] = wan_revision
    job["component_paths"] = {
        "wan_path": str(reference / "assets/Wan-AI/Wan2.2-TI2V-5B"),
        "image_encoder_path": str(previous / "assets/Wan-AI/Wan2.1-I2V-14B-480P"),
        "tokenizer_path": str(reference / "assets/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
    }
    revisions = {s["revision"] for s in specs if s["repo"] == "yuanty/LIBERO-fastwam"}
    if revisions != {"ee018b997c430bb12b5bf3c892d744798c5a2f91"}:
        raise ValueError(f"Unexpected LIBERO source revisions: {revisions}")
    job["dataset_source_revision"] = next(iter(revisions))
    meta = LeRobotDatasetMetadata(job["repo_id"], root=job["dataset_root"])
    reported_revision = getattr(meta.info, "revision", None)
    if reported_revision is not None and reported_revision != "ee018b997c430bb12b5bf3c892d744798c5a2f91":
        raise ValueError(f"LIBERO dataset revision changed: {reported_revision}")
    assert meta.fps == 20 and meta.total_episodes == 1712 and meta.total_tasks == 40
    camera_keys = ["observation.images.image", "observation.images.wrist_image"]
    assert set(meta.camera_keys) == set(camera_keys)
    assert len(camera_keys) == 2, f"Expected the native two-camera LIBERO dataset, got {camera_keys}"
    image_shapes = {}
    for key in camera_keys:
        shape = tuple(meta.features[key]["shape"])
        if len(shape) != 3:
            raise ValueError(f"Unexpected image feature shape for {key}: {shape}")
        # LeRobot metadata stores image shape as HWC; policy feature descriptors use CHW.
        image_shapes[key] = (shape[2], shape[0], shape[1])
    cfg = make_config(job, camera_keys, image_shapes)
    split = split_episodes(list(meta.episodes), tail=cfg.drop_n_last_frames)
    assert len(split["train"]) == len(split["validation"]) == 40
    write_json(root / "split.json", split)
    validation = validation_indices(split["validation"])
    write_json(root / "validation_indices.json", validation)
    train_episodes = sorted(row["episode"] for rows in split["train"].values() for row in rows)
    validation_episodes = sorted(row["episode"] for rows in split["validation"].values() for row in rows)
    train_dataset = LeRobotDataset(job["repo_id"], root=job["dataset_root"], episodes=train_episodes,
                                   delta_timestamps=resolve_delta_timestamps(cfg, meta),
                                   video_backend="pyav", return_uint8=True, download_videos=False)
    validation_dataset = LeRobotDataset(job["repo_id"], root=job["dataset_root"],
                                        episodes=validation_episodes,
                                        delta_timestamps=resolve_delta_timestamps(cfg, meta),
                                        video_backend="pyav", return_uint8=True, download_videos=False)
    stats = {}
    for key in ("action", "observation.state"):
        values = np.asarray(train_dataset.hf_dataset.data.column(key).to_pylist(), dtype=np.float64)
        assert values.shape[0] == train_dataset.num_frames and np.isfinite(values).all()
        stats[key] = {"min": values.min(0).tolist(), "max": values.max(0).tolist(),
                      "mean": values.mean(0).tolist(), "std": values.std(0).tolist()}
    cfg.normalization_style = "min_max"
    cfg.action_stats = {k: stats["action"][k] for k in ("min", "max")}
    cfg.state_stats = {k: stats["observation.state"][k] for k in ("min", "max")}
    write_json(root / "train_stats.json", stats)
    cfg.save_pretrained(root / "policy_config")
    # Read a complete video window for every task and held-out audit window.
    train_mapping = train_dataset.absolute_to_relative_idx
    validation_mapping = validation_dataset.absolute_to_relative_idx
    assert train_mapping is not None and validation_mapping is not None
    for row in validation:
        sample = validation_dataset[validation_mapping[row["index"]]]
        assert sample["action"].shape == (64, 7) and sample["observation.state"].shape == (4, 8)
        assert all(not torch.as_tensor(v).any() for k, v in sample.items() if k.endswith("_is_pad"))
        assert all(sample[key].shape[:2] == (33, 3) for key in cfg.camera_keys)
    indices = [rows[0]["start"] for rows in split["train"].values()]
    for index in indices:
        sample = train_dataset[train_mapping[index]]
        assert sample["action"].shape == (64, 7) and sample["observation.state"].shape == (4, 8)
        assert all(not torch.as_tensor(v).any() for k, v in sample.items() if k.endswith("_is_pad"))
        assert all(sample[key].shape[:2] == (33, 3) for key in cfg.camera_keys)
    # The merged view has reindexed tables. Preserve its exact identity as well as the official assets.
    merged = {str(p.relative_to(Path(job["dataset_root"]))): file_hash(p)
              for p in sorted(Path(job["dataset_root"]).rglob("*")) if p.is_file()}
    write_json(root / "dataset_hashes.json", merged)
    job["dataset_revision"] = "sha256:" + file_hash(root / "dataset_hashes.json")
    write_json(root / "job.json", job)
    fingerprint = (file_hash(root / "job.json") + file_hash(root / "split.json") +
                   file_hash(root / "train_stats.json") + file_hash(root / "wan_dit_verified.json"))
    write_json(root / "data_audit.json", {
        "passed": True, "train_episodes": sum(map(len, split["train"].values())),
        "validation_episodes": sum(map(len, split["validation"].values())),
        "excluded_short_episodes": split["excluded_short_episodes"],
        "training_frames_for_stats": train_dataset.num_frames, "validated_windows": len(indices) + len(validation),
        "protocol_sha256": protocol_sha256,
        "fingerprint": fingerprint, "camera_order": cfg.camera_keys,
        "action": "eef delta xyz + axis-angle delta xyz + gripper (0 closed, 1 open)",
        "state": "eef xyz + axis-angle xyz + two gripper joint positions",
        "original_data_modified": False,
    })
    # Confirm quota permits actual writes; free-space estimates alone were unreliable on this filesystem.
    probe = root / "write_probe.tmp"
    with probe.open("wb") as stream:
        stream.write(os.urandom(4 * 1024**2))
        stream.flush()
        os.fsync(stream.fileno())
    probe.unlink()
    write_json(root / "preparation_status.json", {"stage": "ready", "finished_unix": time.time()})
    (root / "READY").write_text("Verified data and immutable assets; GPU preflight is still required.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--previous-root", required=True)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--reference-root", required=True, help="Previous FastWAM asset/environment root")
    args = parser.parse_args()
    prepare(args.root, args.previous_root, args.protocol_manifest, args.reference_root)
