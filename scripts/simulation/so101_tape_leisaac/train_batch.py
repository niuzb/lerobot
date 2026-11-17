"""SmolVLA preflight and native training with train-episode-only numeric statistics."""

import argparse
import json
from pathlib import Path


def train_only_stats(dataset, destination):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(sorted((dataset.root / "data").rglob("*.parquet")))
    episodes = dataset.episodes if dataset.episodes is not None else list(range(dataset.num_episodes))
    table = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(episodes)))
    if not table.num_rows:
        raise ValueError("No training rows for normalization")
    numeric = {}
    for key, feature in dataset.meta.features.items():
        if feature["dtype"] != "float32" or not (key.startswith("observation.") or key == "action"):
            continue
        values = np.asarray(table[key].to_pylist(), dtype=np.float64).reshape(table.num_rows, -1)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite training statistics: {key}")
        numeric[key] = {
            "mean": values.mean(0),
            "std": np.maximum(values.std(0), 1e-6),
            "min": values.min(0),
            "max": values.max(0),
            "count": np.array([len(values)]),
        }
    dataset.meta.stats.update(numeric)
    report = {
        "episodes": list(episodes),
        "frames": table.num_rows,
        "scope": "training episodes only; validation excluded",
        "stats": {
            key: {name: value.tolist() for name, value in stats.items()} for key, stats in numeric.items()
        },
    }
    Path(destination).write_text(json.dumps(report, indent=2) + "\n")
    return report


def preflight(config, dataset_root, repo_id, output):
    import draccus
    import torch
    from torch.utils.data import DataLoader

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.processor.rename_processor import rename_stats

    cfg = draccus.parse(TrainPipelineConfig, config_path=config, args=[])
    if not isinstance(cfg.policy, SmolVLAConfig):
        raise TypeError("This preflight is specific to SmolVLA")
    cfg.validate()
    cfg.dataset.root, cfg.dataset.repo_id = Path(dataset_root), repo_id
    cfg.dataset.eval_split = 0.0
    cfg.dataset.episodes = None
    dataset = make_dataset(cfg)
    train_only_stats(dataset, output.parent / "preflight_stats.json")
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    stats = rename_stats(dataset.meta.stats, cfg.rename_map)
    pre, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
            "normalizer_processor": {
                "stats": stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        },
    )
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    batch = next(iter(DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0)))
    for key, value in batch.items():
        if key.startswith("observation.images.") and value.dtype == torch.uint8:
            batch[key] = value.float().div(255)
    batch = pre(batch)
    policy.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy(batch)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite SmolVLA preflight loss")
    loss.backward()
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
    if not torch.isfinite(gradient_norm):
        raise ValueError("Non-finite SmolVLA gradients")
    free, total = torch.cuda.mem_get_info()
    optimizer_bytes = sum(parameter.numel() * 8 for parameter in trainable)
    if free < optimizer_bytes + 512 * 1024**2:
        raise ValueError("Insufficient GPU headroom for AdamW optimizer states")
    report = {
        "passed": True,
        "batch_size": cfg.batch_size,
        "state_dimension": 25,
        "action_dimension": 6,
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_total_bytes": total,
        "estimated_optimizer_state_bytes": optimizer_bytes,
        "optimizer_steps": 0,
        "weights_saved": False,
        "mode": "forward/backward validation only",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-dataset", type=Path)
    parser.add_argument("--preflight-repo-id")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stats-output", type=Path)
    args = parser.parse_args()
    if args.preflight_dataset is not None:
        preflight(args.config, args.preflight_dataset, args.preflight_repo_id, args.report)
        return

    # Patch only this process's dataset factory, not the repository or dataset.
    # The native training entrypoint still owns optimization/checkpoint/eval.
    import sys

    from lerobot.scripts import lerobot_train

    original_factory = lerobot_train.make_train_eval_datasets

    def datasets_with_train_stats(cfg):
        training, validation = original_factory(cfg)
        if (
            list(training.episodes or []) != list(range(90))
            or validation is None
            or list(validation.episodes or []) != list(range(90, 100))
        ):
            raise ValueError("Require the declared 90/10 physical-episode split")
        train_only_stats(training, args.stats_output or args.config.parent / "training_stats.json")
        return training, validation

    lerobot_train.make_train_eval_datasets = datasets_with_train_stats
    sys.argv = [sys.argv[0], f"--config_path={args.config}"]
    if args.resume:
        sys.argv.append("--resume=true")
    lerobot_train.train()


if __name__ == "__main__":
    main()
