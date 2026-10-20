"""Single-GPU paired SmolVLA experiment with an exact per-batch domain ratio."""

import json
from pathlib import Path

import numpy as np

from .config import TRAIN, VALIDATION, digest, fingerprint, write_json
from .data import SourceData

RENAME = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.side": "observation.images.camera2",
}


def batch_indices(real_size, sim_size, steps, seed, mixed):
    """Yield reproducible eight-sample batches with exact four/four domain balance."""
    if real_size <= 0 or (mixed and sim_size <= 0):
        raise ValueError("Cannot sample an empty domain")
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if mixed:
            yield np.r_[rng.integers(real_size, size=4), real_size + rng.integers(sim_size, size=4)].tolist()
        else:
            yield rng.integers(real_size, size=8).tolist()


def train(config, out, base, variant, generated, device, smoke=False):
    """Run a fresh fixed-budget single-device experiment and save native checkpoints."""
    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    from lerobot.common.train_utils import save_checkpoint
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.utils.random_utils import set_seed

    if not Path(base).is_dir():
        raise ValueError("--base must be a local snapshot of fixed lerobot/smolvla_base weights")
    source = SourceData(config)
    set_seed(1000)
    torch.set_num_threads(4)
    cfg_policy = SmolVLAConfig.from_pretrained(base)
    cfg_policy.pretrained_path = Path(base)
    cfg_policy.device = device
    cfg_policy.push_to_hub = False
    cfg_policy.scheduler_decay_steps = 20000
    cfg_policy.n_action_steps = 30
    # The base snapshot's camera list is not our two-camera embodiment.
    cfg_policy.input_features = {}
    real = LeRobotDataset(
        config["dataset_repo_id"], root=config["dataset_root"], episodes=TRAIN, video_backend="pyav"
    )
    delta = resolve_delta_timestamps(cfg_policy, real.meta, RENAME)
    real = LeRobotDataset(
        config["dataset_repo_id"],
        root=config["dataset_root"],
        episodes=TRAIN,
        delta_timestamps=delta,
        video_backend="pyav",
    )
    validation = LeRobotDataset(
        config["dataset_repo_id"],
        root=config["dataset_root"],
        episodes=VALIDATION,
        delta_timestamps=delta,
        video_backend="pyav",
    )
    # Use the same train-only real statistics for both variants: isolate the data augmentation.
    stats = {}
    for key, array in [("observation.state", source.state), ("action", source.action)]:
        values = array[np.isin(source.ep, TRAIN)].astype(np.float64)
        stats[key] = {
            "mean": values.mean(0),
            "std": np.maximum(values.std(0), 1e-6),
            "min": values.min(0),
            "max": values.max(0),
            "count": np.array([len(values)]),
        }
    real.meta.stats = stats
    mixed = variant == "mixed"
    sim = None
    if mixed:
        manifest = json.loads((Path(generated) / "generation_manifest.json").read_text())
        accepted = manifest.get("accepted")
        if not manifest.get("passed") or accepted not in ((20, 200) if smoke else (200,)):
            raise ValueError(
                "Mixed training needs 200 successes; a two-step smoke test may use the 20-success pilot"
            )
        if manifest.get("physics_only"):
            raise ValueError("Physical trajectories must be rendered and exported before training")
        if manifest["config_sha256"] != fingerprint(config):
            raise ValueError("Generated dataset belongs to another scene")
        sim_root = Path(generated) / "dataset"
        files = manifest.get("dataset_sha256", {})
        if not files:
            raise ValueError("Generated data has no finalized integrity manifest")
        for name, expected in files.items():
            path = (sim_root / name).resolve()
            if not path.is_relative_to(sim_root.resolve()) or digest(path) != expected:
                raise ValueError(f"Generated data changed after acceptance: {name}")
        sim = LeRobotDataset(
            "local/so101_r2s2r_sim", root=sim_root, delta_timestamps=delta, video_backend="pyav"
        )
    dataset = ConcatDataset([real, sim]) if mixed else real
    steps = 2 if smoke else 20000
    sampler = list(batch_indices(len(real), len(sim) if sim is not None else 0, steps, 1000, mixed))
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    policy = make_policy(cfg_policy, ds_meta=real.meta, rename_map=RENAME)
    mapped_stats = rename_stats(stats, RENAME)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=base,
        dataset_stats=mapped_stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": RENAME},
            "normalizer_processor": {
                "stats": mapped_stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": mapped_stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        },
    )
    optimizer_cfg = cfg_policy.get_optimizer_preset()
    scheduler_cfg = cfg_policy.get_scheduler_preset()
    optimizer = optimizer_cfg.build(policy.get_optim_params())
    # Smoke testing uses the full schedule, never a two-step decay schedule for production.
    scheduler = scheduler_cfg.build(optimizer, 20000)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=config["dataset_repo_id"], root=config["dataset_root"], episodes=TRAIN),
        policy=cfg_policy,
        steps=steps,
        batch_size=8,
        num_workers=0,
        output_dir=out,
        seed=1000,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        rename_map=RENAME,
    )
    manifest = {
        "kind": "paired_smolvla_training",
        "variant": variant,
        "smoke": smoke,
        "steps": steps,
        "seed": 1000,
        "base": str(base),
        "source": source.manifest(),
        "base_sha256": {p.name: digest(p) for p in Path(base).glob("*") if p.is_file()},
        "domain_batch": {"real": 4 if mixed else 8, "sim": 4 if mixed else 0},
        "generated": str(generated) if mixed else None,
        "resume_supported": False,
        "scene_status": config.get("experiment", {}).get("status", "baseline_scene_not_used"),
        "calibration_passed": config.get("experiment", {}).get("calibration_passed", False),
        "generated_manifest_sha256": digest(Path(generated) / "generation_manifest.json") if mixed else None,
        "completed": False,
    }
    write_json(out / "training_manifest.json", manifest)
    val_loader = DataLoader(validation, batch_size=8, shuffle=False, num_workers=0)
    for step, batch in enumerate(dataloader, 1):
        policy.train()
        loss, _ = policy(pre(batch))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), optimizer_cfg.grad_clip_norm)
        if not torch.isfinite(norm):
            raise RuntimeError("Non-finite gradients")
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        if step % 100 == 0 or smoke:
            print(json.dumps({"step": step, "loss": loss.item(), "grad_norm": norm.item()}), flush=True)
        if step % 5000 == 0 or step == steps:
            policy.eval()
            validation_loss, count = 0.0, 0
            with torch.inference_mode():
                for batch in val_loader:
                    value, _ = policy(pre(batch))
                    n = len(batch["observation.state"])
                    validation_loss += value.item() * n
                    count += n
                    if smoke:
                        break
            if not np.isfinite(validation_loss):
                raise RuntimeError("Non-finite validation loss")
            metrics = {
                "step": step,
                "validation_loss": validation_loss / count,
                "validation_frames": count,
                "validation_scope": "first_batch" if smoke else "all_9_episodes",
            }
            checkpoint = out / "checkpoints" / f"{step:06d}"
            save_checkpoint(checkpoint, step, cfg, policy, optimizer, scheduler, pre, post)
            write_json(checkpoint / "validation.json", metrics)
            print(json.dumps(metrics), flush=True)
    manifest["completed"] = True
    write_json(out / "training_manifest.json", manifest)
