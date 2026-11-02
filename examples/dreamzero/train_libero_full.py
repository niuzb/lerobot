"""Single-GPU DreamZero full fine-tuning on the fixed LIBERO episode split.

Launch on the retained Merlin A100 after reviewing the preparation audit:
    uv run python -m examples.dreamzero.train_libero_full --root outputs/dreamzero_libero_full

The script refuses to initialize unless preparation produced READY. Run
--max-updates 4 --smoke for the required save/reload/continue test; production uses
exactly 20,000 optimizer updates.
"""

import argparse
import gc
import json
import logging
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig
from lerobot.policies.dreamzero.modeling_dreamzero import DreamZeroPolicy
from lerobot.policies.dreamzero.processor_dreamzero import make_dreamzero_pre_post_processors

from .libero_full_runtime import (
    MODES,
    check_first_allocation,
    deepspeed_config,
    external_optimizer,
    optimizer_audit,
    optimizer_layout,
    resolve_runtime,
    runtime_versions,
)
from .libero_full_utils import (
    BalancedWindows,
    check_memory,
    export_policy,
    file_hash,
    isolated_rng,
    memory_snapshot,
    write_json,
)


class RelativeWindowSampler(Sampler):
    """Map persisted absolute LeRobot indices into a training-episode view."""

    def __init__(self, balanced, absolute_to_relative):
        if not absolute_to_relative:
            raise ValueError("The episode-filtered dataset did not expose an index mapping")
        self.balanced = balanced
        self.absolute_to_relative = absolute_to_relative

    def __iter__(self):
        while True:
            absolute = self.balanced.next()
            try:
                yield self.absolute_to_relative[absolute]
            except KeyError as exc:
                raise IndexError(f"Sample {absolute} is outside the training episode view") from exc

    def __len__(self):
        return 2**31


def _write_deepspeed_config(root, job, runtime):
    config = deepspeed_config(job, runtime)
    path = Path(root) / "deepspeed.json"
    if (
        path.exists()
        and json.loads(path.read_text()) != config
        and any((Path(root) / "checkpoints").glob("*/COMPLETE.json"))
    ):
        raise ValueError("DeepSpeed configuration changed after a checkpoint; use a separate run")
    write_json(path, config)
    return config


def _precompute_text(policy, tasks):
    """Cache fixed LIBERO and negative prompt embeddings, then release T5/tokenizer."""
    frozen = policy._frozen()
    phrases = [policy.config.prompt_template.format(task=task) for task in tasks]
    phrases.append(policy.config.negative_prompt)
    cache = {phrase: frozen.encode_text([phrase]).detach() for phrase in phrases}

    def encode_text(batch_phrases):
        try:
            return torch.cat([cache[phrase] for phrase in batch_phrases], dim=0)
        except KeyError as exc:
            raise ValueError(f"Text was not part of the fixed LIBERO task set: {exc.args[0]}") from exc

    frozen.encode_text = encode_text
    del frozen.text, frozen.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"tasks": len(tasks), "negative_prompt_cached": True, "text_encoder_released": True}


def _save_resume(engine, balanced, update, root, rng):
    from .libero_full_checkpoint import save_resume

    return save_resume(engine, balanced, update, root, rng)


def _resume(engine, balanced, checkpoint_dir, root):
    from .libero_full_checkpoint import resume

    return resume(engine, balanced, checkpoint_dir, root)


def _master_probes(policy):
    probes = {}
    for index in (0, len(policy.model.blocks) // 2, len(policy.model.blocks) - 1):
        parameter = policy.model.blocks[index].self_attn.q.weight
        if hasattr(parameter, "ds_id"):
            from deepspeed.utils import safe_get_local_fp32_param

            value = safe_get_local_fp32_param(parameter)
            if value is None:
                raise RuntimeError("ZeRO-3 did not expose the DiT FP32 master")
            probes[str(index)] = value.detach().flatten()[:1024].cpu().clone()
            continue
        mapping = getattr(parameter, "_hp_mapping", None)
        if mapping is None:
            raise RuntimeError("DeepSpeed did not expose the DiT FP32 master mapping")
        probes[str(index)] = mapping.get_hp_fragment().detach().flatten()[:1024].cpu().clone()
    return probes


def _check_master_changes(before, policy):
    after = _master_probes(policy)
    changes = {name: float((after[name] - value).abs().max()) for name, value in before.items()}
    if not all(value > 0 for value in changes.values()):
        raise RuntimeError(f"Representative FP32 DiT master weights did not change: {changes}")
    return changes


def _export_candidate(policy, root, update, preprocessor, postprocessor):
    candidates = Path(root) / "candidates"
    candidates.mkdir(exist_ok=True)
    target = candidates / f"update_{update:06d}"
    if target.exists():
        shutil.rmtree(target)
    export_policy(policy, target, preprocessor, postprocessor)
    return target


def _validate(policy, validation_dataset, indices, absolute_to_relative, tasks, root, update, preprocessor):
    """Report overall and per-task fixed-window metrics without changing training RNG."""
    names = ("loss", "loss_video", "loss_action", "action_mse_7d", "gripper_accuracy")
    sums = dict.fromkeys(names, 0.0)
    task_sums = {task: dict.fromkeys(names, 0.0) for task in tasks}
    policy.eval()
    with isolated_rng(9000), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in indices:
            task, idx = row["task"], row["index"]
            sample = validation_dataset[absolute_to_relative[idx]]
            batch = DataLoader([sample], batch_size=1).__iter__().__next__()
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(policy.device)
            for key in policy.config.camera_keys:
                if batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].float() / 255.0
            batch = preprocessor(batch)
            # Validation uses fixed noise while isolated_rng restores the training stream.
            torch.manual_seed(9000 + idx)
            torch.cuda.manual_seed_all(9000 + idx)
            loss, metrics = policy(batch)
            values = {
                "loss": float(loss),
                "loss_video": float(metrics["loss_video"]),
                "loss_action": float(metrics["loss_action"]),
                "action_mse_7d": float(metrics.get("action_mse_7d", 0.0)),
                "gripper_accuracy": float(metrics.get("gripper_accuracy", 0.0)),
            }
            for name, value in values.items():
                sums[name] += value
                task_sums[task][name] += value
    counts = {task: sum(row["task"] == task for row in indices) for task in tasks}
    policy.train()
    result = {name: value / max(len(indices), 1) for name, value in sums.items()}
    result["by_task"] = {
        task: {name: value / max(counts[task], 1) for name, value in values.items()}
        for task, values in task_sums.items()
    }
    result.update({"update": update, "windows": len(indices), "tasks": len(tasks)})
    path = Path(root) / "validation" / f"update_{update:06d}.json"
    write_json(path, result)
    return result


def _record_best(root, candidate, validation, closed_loop):
    from .libero_full_utils import candidate_key

    best_path = Path(root) / "best_candidate.json"
    key = candidate_key(closed_loop, validation["action_mse_7d"])
    previous = json.loads(best_path.read_text()) if best_path.exists() else None
    if previous is None or tuple(key) > tuple(previous["key"]):
        write_json(
            best_path,
            {
                "update": closed_loop["update"],
                "path": str(candidate),
                "key": key,
                "validation": validation,
                "closed_loop": closed_loop,
            },
        )
        for path in (Path(root) / "candidates").glob("update_*"):
            if path != candidate:
                shutil.rmtree(path)
        return True
    shutil.rmtree(candidate)
    return False


def _run_final_eval(root, job, tasks, meta, best):
    from .evaluate_libero_full import evaluate_full

    candidate = Path(best["path"])
    policy = DreamZeroPolicy.from_pretrained(candidate).to(device="cuda", dtype=torch.bfloat16)
    policy._frozen()
    _precompute_text(policy, tasks)
    preprocessor, postprocessor = make_dreamzero_pre_post_processors(policy.config, dataset_meta=meta)
    policy.eval()
    try:
        selection_path = Path(root) / "closed_loop" / f"selection_update_{best['update']:06d}.json"
        selection = json.loads(selection_path.read_text())
        result = evaluate_full(
            policy, preprocessor, postprocessor, root, best["update"], selection, job["protocol_manifest"]
        )
    finally:
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    return result


def run(
    root,
    max_updates=None,
    resume_from=None,
    smoke=False,
    verify_resume=False,
    optimizer_mode=None,
    offload_ratio=None,
):
    root = Path(root).resolve()
    if not (root / "READY").is_file():
        raise FileNotFoundError(f"Preparation is incomplete: {root / 'READY'}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This job requires the retained single Merlin A100 worker")
    job = json.loads((root / "job.json").read_text())
    if job["optimizer_updates"] != 20000 or job["accumulation"] != 16:
        raise ValueError("The approved production budget is fixed at 20,000 updates × 16 microsteps")
    if max_updates is not None and max_updates <= 0:
        raise ValueError("--max-updates must be positive")
    import deepspeed

    runtime = resolve_runtime(root, optimizer_mode, offload_ratio)
    versions = runtime_versions(runtime)
    job["deepspeed_version"] = versions["deepspeed"]
    write_json(root / "job.json", job)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("DeepSpeed %s; device %s", job["deepspeed_version"], torch.cuda.get_device_name(0))
    ds_cfg = _write_deepspeed_config(root, job, runtime)
    if not smoke and not verify_resume:
        passed = root / "SMOKE_PASSED.json"
        if not passed.exists():
            raise RuntimeError("Four-update save/reload/continuation preflight has not passed")
        hashes = json.loads(passed.read_text()).get("runtime_hashes", {})
        if hashes != {name: file_hash(root / name) for name in ("runtime.json", "deepspeed.json")}:
            raise RuntimeError("Preflight belongs to a different optimizer configuration")

    split = json.loads((root / "split.json").read_text())
    tasks = sorted(split["train"])
    train_ids = sorted(row["episode"] for rows in split["train"].values() for row in rows)
    validation_ids = sorted(row["episode"] for rows in split["validation"].values() for row in rows)
    policy_cfg = DreamZeroConfig.from_pretrained(root / "policy_config")
    meta = LeRobotDatasetMetadata("local/dreamzero_libero_40", root=job["dataset_root"])
    train_ds = LeRobotDataset(
        meta.repo_id,
        root=job["dataset_root"],
        episodes=train_ids,
        delta_timestamps=resolve_delta_timestamps(policy_cfg, meta),
        video_backend="pyav",
        return_uint8=True,
        download_videos=False,
    )
    val_ds = LeRobotDataset(
        meta.repo_id,
        root=job["dataset_root"],
        episodes=validation_ids,
        delta_timestamps=resolve_delta_timestamps(policy_cfg, meta),
        video_backend="pyav",
        return_uint8=True,
        download_videos=False,
    )
    train_map, val_map = train_ds.absolute_to_relative_idx, val_ds.absolute_to_relative_idx
    if train_map is None or val_map is None:
        raise RuntimeError("LeRobot did not build the expected episode index mapping")
    sampler = BalancedWindows(split["train"], seed=job["seed"])
    initial_update = 0
    if resume_from:
        sampler_state_path = Path(resume_from) / "training_state" / "state.pt"
        sampler_state = torch.load(sampler_state_path, map_location="cpu", weights_only=False)
        sampler.load_state_dict(sampler_state["balanced"])
        initial_update = int(sampler_state["update"])
    # No worker prefetch: the sampler RNG state exactly follows consumed microsteps.
    loader = DataLoader(
        train_ds,
        batch_size=1,
        sampler=RelativeWindowSampler(sampler, train_map),
        num_workers=0,
        pin_memory=True,
    )
    iterator = iter(loader)

    import random

    import numpy as np

    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    policy = DreamZeroPolicy(policy_cfg).to(device="cuda", dtype=torch.bfloat16)
    policy.train()
    components = _precompute_text(policy, tasks)
    preprocessor, postprocessor = make_dreamzero_pre_post_processors(policy_cfg, dataset_meta=meta)
    trainable = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    expected_ids = {id(p) for p in policy.model.parameters() if p.requires_grad}
    optimizer_ids = {id(p) for p in policy.get_optim_params()}
    if not trainable or optimizer_ids != expected_ids:
        raise RuntimeError("Optimizer does not contain every and only trainable DreamZero parameter")
    trainable_count = sum(p.numel() for _, p in trainable)
    if not all(p.requires_grad for p in policy.model.blocks.parameters()):
        raise RuntimeError("At least one Wan DiT block is frozen")
    frozen_ids = {
        id(p) for module in (policy._components.vae, policy._components.image) for p in module.parameters()
    }
    if any(id(p) in frozen_ids for _, p in trainable):
        raise RuntimeError("A frozen VAE/CLIP parameter entered the optimizer")
    probe_param = next((p for name, p in trainable if "blocks" in name), None)
    if probe_param is None:
        raise RuntimeError("No Wan DiT block parameter available for update audit")
    frozen_probes = {
        name: next(module.parameters()).detach().float().flatten()[:16].cpu().clone()
        for name, module in (("vae", policy._components.vae), ("clip", policy._components.image))
    }

    memory_check = check_memory
    memory_check()
    torch.cuda.reset_peak_memory_stats()
    engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=policy,
        model_parameters=policy.get_optim_params(),
        config=ds_cfg,
        optimizer=external_optimizer(policy.get_optim_params(), job, runtime),
    )
    init_memory = memory_check()
    write_json(
        root / "memory_preflight.json",
        {
            "stage": "optimizer_initialized",
            **init_memory,
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(),
        },
    )
    audit = optimizer_audit(optimizer, runtime)
    write_json(root / "optimizer_audit.json", {**audit, "stage": "initialized", "versions": versions})
    if resume_from:
        initial_update, rng = _resume(engine, sampler, resume_from, root)
        from .libero_full_utils import restore_rng

        restore_rng(rng)
    master_before = _master_probes(engine.module)

    target_updates = min(job["optimizer_updates"], max_updates or job["optimizer_updates"])
    if smoke and max_updates is None:
        target_updates = 4
    log_path = root / "train_metrics.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Beginning at optimizer update %d; target %d", initial_update, target_updates)
    microstep = initial_update * job["accumulation"]
    latest_validation = None
    try:
        for microstep in range(initial_update * job["accumulation"], target_updates * job["accumulation"]):
            batch = next(iterator)
            for key, value in list(batch.items()):
                if key.endswith("_is_pad") and torch.as_tensor(value).any():
                    raise RuntimeError(f"Padding found in sampled batch {key}")
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to("cuda", non_blocking=True)
            for key in policy_cfg.camera_keys:
                if batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].float() / 255.0
            batch = preprocessor(batch)
            loss, losses = engine(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at microstep {microstep + 1}")
            engine.backward(loss)
            boundary = engine.is_gradient_accumulation_boundary()
            if boundary:
                finite = torch.ones((), dtype=torch.bool, device="cuda")
                for parameter in engine.module.parameters():
                    if parameter.grad is not None:
                        finite &= torch.isfinite(parameter.grad).all()
                if not bool(finite.item()):
                    raise FloatingPointError(f"Nonfinite gradient at microstep {microstep + 1}")
            if boundary and engine.global_steps == 0:
                check_first_allocation(optimizer, runtime, root)
            if boundary and hasattr(optimizer, "backup_optimizer"):
                # WarmupLR changes the primary optimizer after each engine step.
                for primary, backup in zip(
                    optimizer.optimizer.param_groups, optimizer.backup_optimizer.param_groups, strict=True
                ):
                    backup["lr"] = primary["lr"]
            engine.step()
            if boundary and not torch.isfinite(torch.as_tensor(engine.get_global_grad_norm())):
                raise FloatingPointError("DeepSpeed reported a nonfinite global gradient norm")
            if boundary:
                if engine.global_steps == 1:
                    audit = optimizer_audit(optimizer, runtime, initialized=True)
                    write_json(
                        root / "optimizer_audit.json",
                        {**audit, "stage": "after_first_update", "versions": versions},
                    )
                update = engine.global_steps
                metrics = {
                    "update": update,
                    "microstep": update * job["accumulation"],
                    "loss": float(loss.detach().float().item()),
                    **{
                        key: float(value.detach().float().item())
                        if isinstance(value, torch.Tensor)
                        else float(value)
                        for key, value in losses.items()
                    },
                    "memory": check_memory(),
                }
                with log_path.open("a") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                should_validate = (
                    update % job["save_every"] == 0
                    or update in job["eval_updates"]
                    or update == target_updates
                ) and not (smoke or verify_resume)
                should_save = update == 4 or update % job["save_every"] == 0 or should_validate
                if should_save:
                    from .libero_full_utils import rng_state

                    checkpoint = _save_resume(engine, sampler, update, root, rng_state())
                if update == 4 and smoke:
                    _export_candidate(engine.module, root, update, preprocessor, postprocessor)
                    break
                if should_validate:
                    fixed = json.loads((root / "validation_indices.json").read_text())
                    latest_validation = _validate(
                        engine.module, val_ds, fixed, val_map, tasks, root, update, preprocessor
                    )
                    if update in job["eval_updates"]:
                        from .evaluate_libero_full import evaluate_selection

                        candidate = _export_candidate(
                            engine.module, root, update, preprocessor, postprocessor
                        )
                        # Optimizer state remains on the selected profile's devices;
                        # inference runs sequentially after a completed checkpoint.
                        engine.module.eval()
                        with isolated_rng(99000 + update):
                            selection = evaluate_selection(
                                engine.module,
                                preprocessor,
                                postprocessor,
                                root,
                                update,
                                job["protocol_manifest"],
                            )
                        engine.module.train()
                        is_best = _record_best(root, candidate, latest_validation, selection)
                        write_json(
                            root / "candidate_audit.json",
                            {
                                "update": update,
                                "all_dit_blocks_trainable": True,
                                "trainable_parameter_count": trainable_count,
                                "optimizer_parameter_count": sum(
                                    p.numel() for p in optimizer_layout(optimizer)[0]
                                ),
                                "fp32_dit_max_changes": _check_master_changes(master_before, engine.module),
                                "frozen_vae_clip_in_optimizer": False,
                                "frozen_vae_clip_unchanged": all(
                                    torch.equal(
                                        frozen_probes[name],
                                        next(module.parameters()).detach().float().flatten()[:16].cpu(),
                                    )
                                    for name, module in (
                                        ("vae", engine.module._components.vae),
                                        ("clip", engine.module._components.image),
                                    )
                                ),
                                "components": components,
                                "is_best": is_best,
                                "microsteps": update * job["accumulation"],
                                "memory": check_memory(),
                            },
                        )
                        if not json.loads((root / "candidate_audit.json").read_text())[
                            "frozen_vae_clip_unchanged"
                        ]:
                            raise RuntimeError("A frozen encoder parameter changed during training")
                if update % 10 == 0:
                    logging.info(
                        "%d updates / %d microsteps: %s", update, update * job["accumulation"], metrics
                    )
        if verify_resume:
            changes = _check_master_changes(master_before, engine.module)
            if engine.global_steps != target_updates:
                raise RuntimeError("Independent resume did not advance and update the audited DiT parameter")
            write_json(
                root / "resume_verification.json",
                {
                    "loaded_update": initial_update,
                    "continued_update": engine.global_steps,
                    "fp32_dit_max_changes": changes,
                    "passed": True,
                },
            )
        if smoke:
            if engine.global_steps != 4:
                raise RuntimeError(f"Smoke run stopped at {engine.global_steps}, expected four updates")
            # Independent restoration is performed by a fresh subprocess so both model
            # weights and optimizer/sampler/RNG state are read through the production path.
            changes = _check_master_changes(master_before, engine.module)
            frozen_unchanged = all(
                torch.equal(
                    frozen_probes[name], next(module.parameters()).detach().float().flatten()[:16].cpu()
                )
                for name, module in (("vae", policy._components.vae), ("clip", policy._components.image))
            )
            if not frozen_unchanged:
                raise RuntimeError("Frozen encoder changed during preflight")
            write_json(
                root / "smoke_training.json",
                {
                    "updates": 4,
                    "microsteps": 64,
                    "checkpoint": str(checkpoint),
                    "fp32_dit_max_changes": changes,
                    "all_dit_blocks_trainable": True,
                    "frozen_unchanged": True,
                    "memory": check_memory(),
                    "requires_independent_reload": True,
                },
            )
            logging.info("Four updates saved; waiting for independent process reload and continuation")
        elif engine.global_steps >= job["optimizer_updates"]:
            _check_master_changes(master_before, engine.module)
            write_json(
                root / "training_complete.json",
                {
                    "updates": engine.global_steps,
                    "microsteps": engine.global_steps * job["accumulation"],
                    "all_dit_blocks_trainable": True,
                    "probe_parameter_changed": True,
                    "completed_unix": time.time(),
                    "memory": check_memory(),
                },
            )
    except BaseException as exc:
        try:
            memory = check_memory()
        except BaseException as mem_exc:
            memory = {"memory_check_error": repr(mem_exc)}
        write_json(
            root / "failure.json",
            {
                "update": getattr(engine, "global_steps", initial_update),
                "microstep": microstep,
                "error": repr(exc),
                "memory": memory,
            },
        )
        raise
    finally:
        try:
            final_memory = check_memory()
        except BaseException as mem_exc:
            final_memory = {"memory_check_error": repr(mem_exc)}
        logging.info("Worker retained by request; final memory=%s", final_memory)


def evaluate_best(root):
    root = Path(root).resolve()
    job = json.loads((root / "job.json").read_text())
    completed = json.loads((root / "training_complete.json").read_text())
    if completed["updates"] != 20000:
        raise ValueError("Final evaluation requires all 20,000 updates")
    best = json.loads((root / "best_candidate.json").read_text())
    from .libero_full_utils import full_eval_eligible

    eligible = full_eval_eligible(best["closed_loop"])
    write_json(
        root / "evaluation_eligibility.json",
        {
            "eligible": eligible,
            "best_update": best["update"],
            "selection_success_rate": best["closed_loop"]["success_rate"],
            "suite_successes": {k: v["successes"] for k, v in best["closed_loop"]["suites"].items()},
        },
    )
    if eligible:
        split = json.loads((root / "split.json").read_text())
        meta = LeRobotDatasetMetadata(job["repo_id"], root=job["dataset_root"])
        result = _run_final_eval(root, job, sorted(split["train"]), meta, best)
        write_json(
            root / "training_complete.json",
            {
                **completed,
                "full_eval_success_rate": result["all_2000"]["success_rate"],
                "remaining_1960_success_rate": result["remaining_1960"]["success_rate"],
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume-from")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-resume", action="store_true")
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--optimizer-mode", choices=MODES)
    parser.add_argument("--offload-ratio", type=float)
    args = parser.parse_args()
    try:
        if args.final_eval:
            evaluate_best(args.root)
        else:
            run(
                args.root,
                args.max_updates,
                args.resume_from,
                args.smoke,
                args.verify_resume,
                args.optimizer_mode,
                args.offload_ratio,
            )
    except BaseException as exc:
        root = Path(args.root).resolve()
        failure = root / "failure.json"
        if not failure.exists():
            try:
                memory = memory_snapshot()
            except BaseException as memory_exc:
                memory = {"memory_check_error": repr(memory_exc)}
            write_json(failure, {"stage": "startup", "error": repr(exc), "memory": memory})
        raise


if __name__ == "__main__":
    main()
