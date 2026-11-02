"""Full DreamZero training with multi-rank ZeRO-3 CPU parameter and optimizer offload."""

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig
from lerobot.policies.dreamzero.modeling_dreamzero import DreamZeroPolicy
from lerobot.policies.dreamzero.processor_dreamzero import make_dreamzero_pre_post_processors

from .libero_full_utils import BalancedWindows, memory_snapshot, write_json
from .train_libero_full import RelativeWindowSampler, _precompute_text, _validate


def make_config(job, world_size):
    micro_batch = job.get("micro_batch", 1)
    if micro_batch < 1 or world_size < 2 or job["accumulation"] % (world_size * micro_batch):
        raise ValueError("Global accumulation must be divisible by world size >= 2")
    config = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": job["accumulation"] // (world_size * micro_batch),
        "train_batch_size": job["accumulation"],
        "gradient_clipping": job["grad_clip"],
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": False},
            "offload_optimizer": {"device": "cpu", "pin_memory": False},
            "sub_group_size": 50_000_000,
            "reduce_bucket_size": 5_000_000,
            "stage3_prefetch_bucket_size": 5_000_000,
            "stage3_param_persistence_threshold": 0,
            "stage3_max_live_parameters": 100_000_000,
            "stage3_max_reuse_distance": 0,
            "contiguous_gradients": True,
            "overlap_comm": False,
            "reduce_scatter": True,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "optimizer": {"type": "AdamW", "params": {
            "lr": job["lr"], "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": job["weight_decay"]}},
        "scheduler": {"type": "WarmupLR", "params": {
            "warmup_min_lr": 0.0, "warmup_max_lr": job["lr"],
            "warmup_num_steps": job["warmup_updates"], "warmup_type": "linear"}},
        "steps_per_print": 1000,
    }
    if not job.get("parameter_cpu_offload", True):
        del config["zero_optimization"]["offload_param"]
    return config


def prepare(root, reference, gpu_parameter_shards=False, micro_batch=None, no_activation_checkpointing=False):
    if (root / "job.json").exists():
        existing = json.loads((root / "job.json").read_text())
        if gpu_parameter_shards and existing.get("parameter_cpu_offload", True):
            raise ValueError("Use a new run directory for a different parameter placement")
        if micro_batch is not None and micro_batch != existing.get("micro_batch", 1):
            raise ValueError("Use a new run directory for a different micro batch")
        if no_activation_checkpointing and existing.get("activation_checkpointing", True):
            raise ValueError("Use a new run directory for a different recomputation profile")
        return
    root.mkdir(parents=True, exist_ok=True)
    for name in ("split.json", "train_stats.json", "validation_indices.json", "data_audit.json"):
        shutil.copy2(reference / name, root / name)
    job = json.loads((reference / "job.json").read_text())
    job.update(root=str(root), reference_root=str(reference), worker_id="4461508",
               host_memory_limit_gib=230, gpu_memory_limit_gib=21,
               experiment="Full Wan2.2-5B DreamZero implementation, LIBERO only; not official paper reproduction")
    job["parameter_cpu_offload"] = not gpu_parameter_shards
    job["micro_batch"] = 1 if micro_batch is None else micro_batch
    job["activation_checkpointing"] = not no_activation_checkpointing
    config = DreamZeroConfig.from_pretrained(reference / "policy_config")
    if config.train_mode != "full":
        raise ValueError("Full training configuration required")
    config.text_encoder_device = "cpu"
    config.use_gradient_checkpointing = not no_activation_checkpointing
    config.save_pretrained(root / "policy_config")
    write_json(root / "job.json", job)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def run(args):
    import deepspeed
    from deepspeed.utils import safe_get_local_fp32_param

    if deepspeed.__version__ != "0.19.8":
        raise RuntimeError("Expected pinned DeepSpeed 0.19.8")
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.cpu_affinity:
        cpus = {int(value) for value in args.cpu_affinity.split(":")[local_rank].split(",")}
        if not cpus or not cpus <= os.sched_getaffinity(0):
            raise ValueError("CPU affinity must stay within the worker's allowed CPUs")
        for thread in Path("/proc/self/task").iterdir():
            os.sched_setaffinity(int(thread.name), cpus)
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    root = args.root.resolve()
    if rank == 0:
        prepare(root, args.reference, args.gpu_parameter_shards, args.micro_batch,
                args.no_activation_checkpointing)
    distributed.barrier()
    job = json.loads((root / "job.json").read_text())
    ds_config = make_config(job, world_size)
    if rank == 0:
        write_json(root / "deepspeed.json", ds_config)
    split = json.loads((root / "split.json").read_text())
    tasks = sorted(split["train"])
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    torch.set_num_threads(4)
    config = DreamZeroConfig.from_pretrained(root / "policy_config")
    config.device = f"cuda:{local_rank}"
    config.text_encoder_device = config.device
    meta = LeRobotDatasetMetadata(job["repo_id"], root=job["dataset_root"])
    datasets = {}
    for name in ("train", "validation"):
        ids = sorted(row["episode"] for group in split[name].values() for row in group)
        datasets[name] = LeRobotDataset(meta.repo_id, root=job["dataset_root"], episodes=ids,
            delta_timestamps=resolve_delta_timestamps(config, meta), video_backend="pyav",
            return_uint8=True, download_videos=False)
    print(f"rank={rank}: loading full policy", flush=True)
    policy = DreamZeroPolicy(config).to(device="cpu", dtype=torch.bfloat16)
    if not all(parameter.requires_grad for parameter in policy.model.blocks.parameters()):
        raise RuntimeError("Full DiT blocks must all be trainable")
    trainable_count = sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)
    print(f"rank={rank}: preparing text on GPU before moving the DiT", flush=True)
    _precompute_text(policy, tasks)
    components = policy._components
    cached_text = components.encode_text

    def encode_text(phrases):
        return cached_text(phrases).to(device=config.device, dtype=torch.bfloat16)

    components.encode_text = encode_text
    components.device = torch.device(config.device)
    components.vae.to(config.device)
    components.image.to(config.device)
    policy.to(device=config.device)
    gc.collect()
    torch.cuda.empty_cache()
    preprocessor, postprocessor = make_dreamzero_pre_post_processors(config, dataset_meta=meta)
    print(f"rank={rank}: initializing CPU-offloaded ZeRO-3", flush=True)
    engine, optimizer, _, _ = deepspeed.initialize(model=policy, model_parameters=policy.get_optim_params(),
                                                 config=ds_config)
    balanced = BalancedWindows(split["train"], seed=job["seed"] + rank)
    probe = policy.model.blocks[0].self_attn.q.weight

    def master_digest():
        value = safe_get_local_fp32_param(probe)
        if value is None or value.dtype != torch.float32:
            raise RuntimeError("Missing local FP32 DiT master")
        return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()

    if args.resume:
        loaded_path, state = engine.load_checkpoint(str(root / "checkpoints"), tag=args.resume)
        if not loaded_path or state["world_size"] != world_size or state["job"] != job:
            raise RuntimeError("Checkpoint missing or experiment changed")
        rank_state = state["ranks"][rank]
        balanced.load_state_dict(rank_state["balanced"])
        restore_rng(rank_state["rng"])
        if master_digest() != rank_state["master_digest"] or engine.global_steps != state["update"]:
            raise RuntimeError("Reloaded master or update counter differs")
        write_json(root / f"reload_rank{rank}.json", {"passed": True, "update": engine.global_steps})
    initial = engine.global_steps
    loader = DataLoader(datasets["train"], batch_size=job.get("micro_batch", 1),
        sampler=RelativeWindowSampler(balanced, datasets["train"].absolute_to_relative_idx), num_workers=0,
        generator=torch.Generator().manual_seed(job["seed"] + rank))
    iterator = iter(loader)
    validation = json.loads((root / "validation_indices.json").read_text())
    if args.smoke:
        validation = validation[:2]
    write_json(root / f"initialized_rank{rank}.json", {
        "trainable_parameters": trainable_count, "world_size": world_size,
        "global_batch_size": ds_config["train_batch_size"], "memory": memory_snapshot(),
        "device": torch.cuda.get_device_name(local_rank), "deepspeed": deepspeed.__version__})
    print(f"rank={rank}: optimizer initialized, update={initial}", flush=True)
    started = time.perf_counter()
    loss_sum = 0.0
    previous_digest = master_digest()
    phase = "benchmark" if args.benchmark else ("smoke" if args.smoke else "training")
    timings = dict.fromkeys(("data", "forward", "backward", "step"), 0.0)

    def timestamp():
        if args.benchmark:
            torch.cuda.synchronize()
        return time.perf_counter()

    with (root / f"{phase}_rank{rank}.jsonl").open("a") as log:
        while engine.global_steps < args.max_updates:
            stage_start = timestamp()
            batch = next(iterator)
            for key, value in batch.items():
                if key.endswith("_is_pad") and torch.as_tensor(value).any():
                    raise RuntimeError("Unexpected padded training window")
            for key in config.camera_keys:
                batch[key] = batch[key].float() / 255
            batch = preprocessor(batch)
            current = timestamp()
            timings["data"] += current - stage_start
            stage_start = current
            loss, details = engine(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            current = timestamp()
            timings["forward"] += current - stage_start
            stage_start = current
            engine.backward(loss)
            current = timestamp()
            timings["backward"] += current - stage_start
            stage_start = current
            boundary = engine.is_gradient_accumulation_boundary()
            loss_sum += loss.detach().float().item() / ds_config["gradient_accumulation_steps"]
            applied_lr = max(group["lr"] for group in optimizer.param_groups)
            engine.step()
            timings["step"] += timestamp() - stage_start
            if not boundary:
                continue
            update = engine.global_steps
            if not torch.isfinite(torch.as_tensor(engine.get_global_grad_norm())):
                raise FloatingPointError("Nonfinite global gradient norm")
            digest = master_digest()
            if applied_lr > 0 and digest == previous_digest:
                raise RuntimeError("Representative full DiT FP32 master did not update")
            previous_digest = digest
            reduced_loss = torch.tensor(loss_sum, device=config.device)
            distributed.all_reduce(reduced_loss)
            loss_sum = 0.0
            record = {"update": update, "mean_loss": reduced_loss.item() / world_size,
                "applied_lr": applied_lr,
                "stage_seconds": dict(timings),
                "seconds_per_update": (time.perf_counter() - started) / (update - initial),
                "memory": memory_snapshot(), "master_digest": digest}
            log.write(json.dumps(record) + "\n")
            log.flush()
            write_json(root / f"status_rank{rank}.json", {"phase": phase, **record})
            print(f"rank={rank}: {json.dumps(record)}", flush=True)
            timings = dict.fromkeys(timings, 0.0)
            save_due = update == args.max_updates or update % job["save_every"] == 0
            if save_due and not args.benchmark and not (args.smoke and args.resume):
                state = {"balanced": balanced.state_dict(), "rng": rng_state(), "master_digest": digest}
                rank_states = [None] * world_size
                distributed.all_gather_object(rank_states, state)
                saved = engine.save_checkpoint(str(root / "checkpoints"), tag=f"update_{update:06d}",
                    client_state={"ranks": rank_states, "world_size": world_size, "update": update, "job": job})
                if not saved:
                    raise RuntimeError("DeepSpeed did not confirm checkpoint completion")
                if rank == 0:
                    write_json(root / "latest.json", {"tag": f"update_{update:06d}", "update": update})
            validate_due = update == args.max_updates or (not args.smoke and update in job["eval_updates"])
            if validate_due and not args.benchmark:
                result = _validate(policy, datasets["validation"], validation,
                    datasets["validation"].absolute_to_relative_idx, tasks, root / f"rank{rank}", update, preprocessor)
                if rank == 0:
                    write_json(root / "validation" / f"update_{update:06d}.json", result)
    export = None
    if not args.smoke and not args.benchmark:
        export = (root / "final_model" if engine.global_steps == job["optimizer_updates"]
                  else root / "candidates" / f"update_{engine.global_steps:06d}")
        if not engine.save_16bit_model(str(export), save_filename="model.pt"):
            raise RuntimeError("Full model export failed")
        if rank == 0:
            policy.config.save_pretrained(export)
            preprocessor.save_pretrained(export, config_filename="policy_preprocessor.json")
            postprocessor.save_pretrained(export, config_filename="policy_postprocessor.json")
    distributed.barrier()
    if rank == 0:
        completion = {"update": engine.global_steps,
            "resumed_from": args.resume, "initial_update": initial,
            "elapsed_seconds": time.perf_counter() - started, "success_rate_evaluated": False,
            "export": str(export) if export is not None else None}
        if phase == "training" and engine.global_steps < job["optimizer_updates"]:
            write_json(root / "segments" / f"update_{engine.global_steps:06d}.json", completion)
        else:
            write_json(root / f"{phase}_complete.json", completion)
    distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=20000)
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-parameter-shards", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--cpu-affinity")
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        args.root.mkdir(parents=True, exist_ok=True)
        write_json(args.root / f"failure_rank{os.environ.get('RANK', 'unknown')}.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
