"""Native SmolVLA training with audited four-real/four-simulated minibatches."""

import argparse
import json
import sys
from pathlib import Path

import torch
from eval_real_loss import save
from mixed_data import BalancedBatchSampler, make_mixed_datasets, real_split


def main():
    from lerobot.scripts import lerobot_train

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    request = json.loads((args.root / "request.json").read_text())
    train_ids, validation_ids = real_split()
    if request["real_validation_episodes"] != validation_ids:
        raise ValueError("Fixed real holdout changed")
    audit = {"completed_optimizer_steps": 0, "real_samples": 0, "sim_samples": 0, "violations": 0}
    pending = False
    original_preprocess = lerobot_train._preprocess_dataset_batch
    original_update = lerobot_train.update_policy

    def datasets(cfg):
        if cfg.batch_size != 8 or cfg.accelerator.gradient_accumulation.steps != 1:
            raise ValueError("Require batch8 with one optimizer update per balanced batch")
        return make_mixed_datasets(cfg, request["paths"], args.root)

    def dataloaders(cfg, training, validation, step, parallel_dims):
        if parallel_dims.dp_world_size != 1:
            raise ValueError("Strict 4+4 recipe is verified for one GPU only")
        audit.update(completed_optimizer_steps=step, real_samples=step * 4, sim_samples=step * 4)
        shared = {
            "num_workers": cfg.num_workers,
            "pin_memory": True,
            "prefetch_factor": cfg.prefetch_factor if cfg.num_workers else None,
            "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
            "multiprocessing_context": cfg.dataloader_multiprocessing_context if cfg.num_workers else None,
        }
        train = torch.utils.data.DataLoader(
            training,
            batch_sampler=BalancedBatchSampler(len(training.real), len(training.simulated), cfg.seed, step),
            **shared,
        )
        evaluation = torch.utils.data.DataLoader(validation, batch_size=8, shuffle=False, **shared)
        return train, evaluation

    def preprocess(batch, camera_keys, rename_map, preprocessor):
        nonlocal pending
        domains = batch.pop("_domain")
        training = batch.pop("_mixed_training")
        if bool(training.any()):
            if pending or not bool(training.all()) or len(domains) != 8:
                raise ValueError("Invalid training batch boundary")
            if int((domains == 0).sum()) != 4 or int((domains == 1).sum()) != 4:
                raise ValueError("Batch violated exact four-real/four-simulation composition")
            real_episodes = batch["episode_index"].reshape(-1)[domains == 0].cpu().tolist()
            if not set(real_episodes).issubset(train_ids) or set(real_episodes) & set(validation_ids):
                raise ValueError("Held-out real episode appeared in a training batch")
            pending = True
        elif set(batch["episode_index"].reshape(-1).cpu().tolist()) - set(validation_ids):
            raise ValueError("Evaluation batch is outside the fixed real holdout")
        return original_preprocess(batch, camera_keys, rename_map, preprocessor)

    def update(*values, **kwargs):
        nonlocal pending
        if not pending:
            raise ValueError("Optimizer update without an audited balanced batch")
        result = original_update(*values, **kwargs)
        audit["completed_optimizer_steps"] += 1
        audit["real_samples"] += 4
        audit["sim_samples"] += 4
        pending = False
        if audit["completed_optimizer_steps"] % 50 == 0:
            save(args.root / "batch_balance.json", audit)
        return result

    lerobot_train.make_train_eval_datasets = datasets
    lerobot_train.make_dataloaders = dataloaders
    lerobot_train._preprocess_dataset_batch = preprocess
    lerobot_train.update_policy = update
    torch.set_num_threads(4)
    sys.argv = [sys.argv[0], f"--config_path={args.config}"]
    if args.resume:
        sys.argv.append("--resume=true")
    lerobot_train.train()
    if audit["completed_optimizer_steps"] != 20000 or audit["real_samples"] != audit["sim_samples"]:
        raise ValueError("Training ended without 20,000 balanced optimizer updates")
    save(args.root / "batch_balance.json", audit)


if __name__ == "__main__":
    main()
