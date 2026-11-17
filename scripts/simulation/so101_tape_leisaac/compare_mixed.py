"""Compare frozen policies on the same real holdout in physical action units."""

import argparse
import json
import time
from pathlib import Path

import torch
from eval_real_loss import SpatialRealDataset, build_spatial_rows, save, sha256
from mixed_data import REAL_REPO, RENAME, PolicyRows, real_split
from torch.utils.data import DataLoader


def evaluate(checkpoint, label, root, request):
    import pyarrow.parquet as pq

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch

    _, validation_ids = real_split()
    paths = request["paths"]
    real_root = Path(paths["real_root"])
    cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.device, cfg.load_vlm_weights = "cuda", False
    assert cfg.chunk_size == 50 and cfg.max_action_dim == 32
    table = pq.read_table(sorted((real_root / "data").rglob("*.parquet")))
    auxiliary, states = build_spatial_rows(json.loads(Path(paths["scene"]).read_text()), table)
    meta = LeRobotDatasetMetadata(REAL_REPO, root=real_root)
    source = LeRobotDataset(
        REAL_REPO,
        root=real_root,
        episodes=validation_ids,
        delta_timestamps=resolve_delta_timestamps(cfg, meta, RENAME),
        video_backend="pyav",
        return_uint8=True,
    )
    data = PolicyRows(SpatialRealDataset(source, auxiliary, states), 0, False)
    loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=2, multiprocessing_context="spawn")
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=cfg, strict=True, local_files_only=True).eval()
    policy.requires_grad_(False)
    pre, post = make_pre_post_processors(
        cfg, checkpoint, preprocessor_overrides={"device_processor": {"device": "cuda"}}
    )
    std = torch.tensor(
        json.loads((root / "training_stats.json").read_text())["stats"]["action"]["std"], dtype=torch.float64
    )
    generator = torch.Generator(device="cuda").manual_seed(20261110)
    absolute, squared, common_squared = (torch.zeros(6, dtype=torch.float64) for _ in range(3))
    count, frames = 0, 0
    start = time.monotonic()
    per_episode = {
        ep: {"frames": 0, "targets": 0, "absolute": torch.zeros(6, dtype=torch.float64)}
        for ep in validation_ids
    }
    with torch.inference_mode():
        for i, batch in enumerate(loader, 1):
            targets = batch["action"].double().clone()
            valid = (~batch["action_is_pad"]).clone()
            episode_ids = batch["episode_index"].reshape(-1).tolist()
            batch.pop("_domain")
            batch.pop("_mixed_training")
            prepared = _preprocess_dataset_batch(batch, meta.camera_keys, RENAME, pre)
            noise = torch.randn((len(targets), 50, 32), generator=generator, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = post(policy.predict_action_chunk(prepared, noise=noise)).double().cpu()
            if predictions.shape != targets.shape or not torch.isfinite(predictions).all():
                raise ValueError("Invalid denormalized prediction chunk")
            residual = predictions - targets
            masked = valid.unsqueeze(-1)
            absolute += (residual.abs() * masked).sum(dim=(0, 1))
            squared += (residual.square() * masked).sum(dim=(0, 1))
            common_squared += ((residual / std).square() * masked).sum(dim=(0, 1))
            count += int(valid.sum())
            frames += len(targets)
            for j, ep in enumerate(episode_ids):
                e = per_episode[ep]
                e["frames"] += 1
                e["targets"] += int(valid[j].sum())
                e["absolute"] += (residual[j].abs() * masked[j]).sum(0)
            if i % 25 == 0:
                print(
                    json.dumps(
                        {
                            "comparison": label,
                            "frames": frames,
                            "expected": len(data),
                            "seconds": time.monotonic() - start,
                        }
                    ),
                    flush=True,
                )
    assert frames == len(data) == 3104
    mae, mse = absolute / count, squared / count
    report = {
        "label": label,
        "checkpoint": str(checkpoint),
        "model_sha256": sha256(checkpoint / "model.safetensors"),
        "real_validation_episodes": validation_ids,
        "frames": frames,
        "valid_action_target_steps": count,
        "noise_seed": 20261110,
        "chunk_size": 50,
        "flow_sampling_steps": cfg.num_steps,
        "per_joint_mae": mae.tolist(),
        "per_joint_mse": mse.tolist(),
        "arm_mean_mae_degrees": float(mae[:5].mean()),
        "gripper_mae_percentage_points": float(mae[5]),
        "common_training_scale_action_mse": float((common_squared / count).mean()),
        "common_scale_source": "same 50/50 training-only action standard deviations for both models",
        "per_episode": {
            ep: {"frames": e["frames"], "per_joint_mae": (e["absolute"] / e["targets"]).tolist()}
            for ep, e in per_episode.items()
        },
        "calibration_passed": False,
        "spatial_conditioning": "same archived scene estimates and FK for both policies",
        "runtime_seconds": time.monotonic() - start,
        "passed": True,
    }
    save(root / f"{label}_physical_metrics.json", report)
    print(
        json.dumps(
            {
                "comparison_complete": label,
                "arm_mae": report["arm_mean_mae_degrees"],
                "gripper_mae": report["gripper_mae_percentage_points"],
                "common_mse": report["common_training_scale_action_mse"],
            }
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", choices=["old_sim_only", "new_mixed"], required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    evaluate(args.checkpoint, args.label, args.root, json.loads((args.root / "request.json").read_text()))


if __name__ == "__main__":
    main()
