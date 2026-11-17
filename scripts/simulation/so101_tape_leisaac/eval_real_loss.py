"""Evaluate the frozen 25D tape policy on 10 randomly selected real episodes.

Missing spatial measurements are reconstructed from the archived scene estimates,
not fitted to validation actions. The source data and checkpoint remain read-only.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from common import TASK, JointMap, Kinematics
from torch.utils.data import DataLoader, Dataset


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def build_spatial_rows(scene, table):
    mapping, fk = JointMap(scene["joint_map"]), Kinematics(scene["asset"])
    rows, states = {}, {}
    for index, state in zip(table["index"].to_pylist(), table["observation.state"].to_pylist(), strict=True):
        state = np.asarray(state, dtype=np.float32)
        frames = fk.frames(mapping.to_sim(state))
        base = frames["base"]
        auxiliary = [0.0]  # Original baseline tabletop, not measured floor height.
        for name in ["front", "side"]:
            cam = scene["cameras"][name]
            parent = np.eye(4) if cam["parent"] == "world" else frames[cam["parent"]]
            camera_base = np.linalg.solve(base, parent @ np.asarray(cam["T_parent_cv"]))
            rotation = camera_base[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
                raise ValueError("Invalid archived camera rotation")
            auxiliary.extend(camera_base[:3, 3])
            auxiliary.extend(rotation[:, :2].T.ravel())
        rows[index] = np.asarray(auxiliary, dtype=np.float32)
        states[index] = state
    return rows, states


class SpatialRealDataset(Dataset):
    def __init__(self, source, auxiliary, raw_states):
        self.source, self.auxiliary, self.raw_states = source, auxiliary, raw_states

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        frame = dict(self.source[index])
        global_index = int(frame["index"])
        state = frame["observation.state"]
        if state.shape[-1] != 6 or state.numel() != 6:
            raise ValueError("Expected one 6D real state per observation")
        if not np.array_equal(state.reshape(-1).numpy(), self.raw_states[global_index]):
            raise ValueError("Spatial-feature and real-video row alignment differs")
        extra = torch.from_numpy(self.auxiliary[global_index]).reshape(*state.shape[:-1], 19)
        frame["observation.state"] = torch.cat([state, extra], dim=-1)
        frame["task"] = TASK
        return frame


def main():
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--real-repo-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        raise FileExistsError("Evaluation report already exists; refusing to overwrite")
    info = json.loads((args.real_root / "meta/info.json").read_text())
    if info["total_episodes"] != 88 or info["features"]["observation.state"]["shape"] != [6]:
        raise ValueError("Require the identified 88-episode real dataset")
    episodes = sorted(random.Random(args.seed).sample(range(88), 10))
    scene = json.loads(args.scene.read_text())
    table = pq.read_table(sorted((args.real_root / "data").rglob("*.parquet")))
    table = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(episodes)))
    expected_counts = {ep: table["episode_index"].to_pylist().count(ep) for ep in episodes}
    checkpoint_hash = sha256(args.checkpoint / "model.safetensors")
    if checkpoint_hash != "d95a0b2ae9949d1fa0f60768ea10c2fa8623e550bcbac79f37c2ec2f4c9174b9":
        raise ValueError("Checkpoint differs from the published step-20000 policy")
    plan = {
        "real_dataset": args.real_repo_id,
        "real_dataset_total_episodes": 88,
        "real_dataset_total_frames": info["total_frames"],
        "sample_seed": args.seed,
        "episode_indices_zero_based": episodes,
        "episode_counts": expected_counts,
        "expected_frames": table.num_rows,
        "checkpoint": str(args.checkpoint),
        "model_sha256": checkpoint_hash,
        "scene_sha256": sha256(args.scene),
        "robot_kinematics_sha256": sha256(scene["asset"]),
        "real_info_sha256": sha256(args.real_root / "meta/info.json"),
        "spatial_input_method": "archived real-scene camera extrinsics + real joint-state forward kinematics; table relative baseline 0 m",
        "camera_parents": {k: v["parent"] for k, v in scene["cameras"].items()},
        "calibration_passed": False,
        "spatial_measurements_recorded_in_real_dataset": False,
        "scene_estimates_not_refitted": True,
        "normalization": "unchanged saved processors fitted on 90 synthetic training episodes; no real-data refit",
        "task_prompt": TASK,
        "batch_size": args.batch_size,
        "precision": "bf16 autocast",
        "action_chunk_size": 50,
        "shuffle": False,
        "gradients": False,
        "optimizer_steps": 0,
        "metric": "offline supervised flow-matching loss on real observations/actions; not task success rate",
    }
    save(args.output / "plan.json", plan)
    print(json.dumps({"stage": "prepared", "episodes": episodes, "frames": table.num_rows}), flush=True)
    auxiliary, raw_states = build_spatial_rows(scene, table)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
    cfg.device, cfg.load_vlm_weights = "cuda", False
    assert cfg.n_obs_steps == 1 and cfg.chunk_size == 50
    meta = LeRobotDatasetMetadata(args.real_repo_id, root=args.real_root)
    rename_map = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.side": "observation.images.camera2",
    }
    delta = resolve_delta_timestamps(cfg, meta, rename_map=rename_map)
    source = LeRobotDataset(
        args.real_repo_id,
        root=args.real_root,
        episodes=episodes,
        delta_timestamps=delta,
        video_backend="pyav",
        return_uint8=False,
    )
    assert len(source) == table.num_rows
    sample = source[0]
    plan["original_real_task_prompt"] = sample["task"]
    plan["delta_timestamps"] = delta
    save(args.output / "plan.json", plan)
    data = SpatialRealDataset(source, auxiliary, raw_states)
    torch.set_num_threads(4)
    policy = SmolVLAPolicy.from_pretrained(
        args.checkpoint, config=cfg, strict=True, local_files_only=True
    ).eval()
    policy.requires_grad_(False)
    pre, _ = make_pre_post_processors(
        cfg, args.checkpoint, preprocessor_overrides={"device_processor": {"device": "cuda"}}
    )
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        multiprocessing_context="spawn" if args.workers else None,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    episode_stats = {
        ep: {"frames": 0, "frame_loss_sum": 0.0, "valid_loss_sum": 0.0, "valid_action_values": 0}
        for ep in episodes
    }
    batch_losses, total_frames, reduction_checked = [], 0, False
    start = time.monotonic()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            ep_ids = batch["episode_index"].reshape(-1).tolist()
            batch = _preprocess_dataset_batch(batch, meta.camera_keys, rename_map, pre)
            assert batch["observation.state"].shape[-1] == 25
            mask = batch.get("action_is_pad")
            counts = (
                (~mask).sum(dim=1) * 6 if mask is not None else torch.full((len(ep_ids),), 300, device="cuda")
            )
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses, _ = policy(batch, reduction="none")
            if losses.shape != (len(ep_ids),) or not torch.isfinite(losses).all():
                raise ValueError("Invalid per-frame evaluation loss")
            values, weights = losses.double().cpu().tolist(), counts.cpu().tolist()
            native_loss = sum(v * w for v, w in zip(values, weights, strict=True)) / sum(weights)
            if not reduction_checked and min(weights) < 300:
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    scalar, _ = policy(batch)
                if not np.isclose(float(scalar), native_loss, rtol=1e-4, atol=1e-6):
                    raise ValueError("Native scalar and reconstructed masked reduction disagree")
                reduction_checked = True
            batch_losses.append(native_loss)
            for ep, value, count in zip(ep_ids, values, weights, strict=True):
                e = episode_stats[ep]
                e["frames"] += 1
                e["frame_loss_sum"] += value
                e["valid_loss_sum"] += value * count
                e["valid_action_values"] += count
            total_frames += len(ep_ids)
            if batch_number % 25 == 0:
                progress = {
                    "stage": "evaluating",
                    "batches": batch_number,
                    "frames": total_frames,
                    "expected_frames": table.num_rows,
                    "native_eval_loss_so_far": sum(batch_losses) / len(batch_losses),
                    "elapsed_seconds": time.monotonic() - start,
                }
                save(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    if total_frames != table.num_rows or any(
        episode_stats[ep]["frames"] != n for ep, n in expected_counts.items()
    ):
        raise ValueError("Evaluation did not cover every selected frame")
    assert reduction_checked
    per_episode = {
        ep: {
            "frames": e["frames"],
            "frame_mean_loss": e["frame_loss_sum"] / e["frames"],
            "valid_action_weighted_loss": e["valid_loss_sum"] / e["valid_action_values"],
        }
        for ep, e in episode_stats.items()
    }
    report = {
        **plan,
        "passed": True,
        "frames_evaluated": total_frames,
        "batches": len(batch_losses),
        "native_eval_loss": sum(batch_losses) / len(batch_losses),
        "frame_mean_loss": sum(e["frame_loss_sum"] for e in episode_stats.values()) / total_frames,
        "valid_action_weighted_loss": sum(e["valid_loss_sum"] for e in episode_stats.values())
        / sum(e["valid_action_values"] for e in episode_stats.values()),
        "per_episode": per_episode,
        "native_masked_reduction_verified": True,
        "runtime_seconds": time.monotonic() - start,
        "checkpoint_unchanged": sha256(args.checkpoint / "model.safetensors") == checkpoint_hash,
    }
    assert report["checkpoint_unchanged"]
    save(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "stage": "complete",
                "native_eval_loss": report["native_eval_loss"],
                "frames": total_frames,
                "per_episode": per_episode,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
