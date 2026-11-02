"""Single-rank ZeRO-3 profiles and device-aware optimizer accounting.

DeepSpeed 0.19.8 hybrid offload requires CPUAdam and constructs torch AdamW
for GPU subgroups. It cannot be combined with bitsandbytes AdamW8bit.
"""

import importlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch

from .libero_full_utils import check_memory, write_json

MODES = ("zero3_cpu_adam", "zero3_adamw8bit")
DEEPSPEED_VERSION = "0.19.8"


def resolve_runtime(root, mode=None, offload_ratio=None):
    """Reuse a run's selection, and reject changes after a completed checkpoint."""
    root = Path(root)
    path = root / "runtime.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    mode = mode or previous.get("mode", MODES[0])
    if mode not in MODES:
        raise ValueError(f"Unknown optimizer mode: {mode}")
    if offload_ratio is None:
        offload_ratio = (
            previous["offload_ratio"] if previous.get("mode") == mode else (0.5 if mode == MODES[0] else 0.0)
        )
    if not math.isfinite(offload_ratio) or not 0 <= offload_ratio <= 1:
        raise ValueError("offload_ratio must be between 0 and 1")
    if mode == MODES[1] and offload_ratio != 0:
        raise ValueError("AdamW8bit cannot use CPU offload; select zero3_cpu_adam or set ratio=0")
    if mode == MODES[0] and offload_ratio == 0:
        raise ValueError("zero3_cpu_adam requires a positive CPU offload ratio")
    runtime = {"mode": mode, "offload_ratio": offload_ratio}
    completed = any((root / "checkpoints").glob("*/COMPLETE.json"))
    if runtime != previous and (completed or (root / "SMOKE_PASSED.json").exists()):
        raise ValueError("Optimizer profile is sealed by a checkpoint; use a separate prepared run directory")
    write_json(path, runtime)
    return runtime


def runtime_versions(runtime):
    """Pin the ZeRO internals used by streaming checkpoint restoration."""
    version = importlib.metadata.version("deepspeed")
    if version != DEEPSPEED_VERSION:
        raise RuntimeError(f"This training path requires deepspeed=={DEEPSPEED_VERSION}; found {version}")
    result = {"deepspeed": version, "torch": str(torch.__version__)}
    if runtime["mode"] == MODES[1]:
        try:
            result["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("GPU AdamW8bit requires bitsandbytes in the training environment") from exc
    return result


def deepspeed_config(job, runtime):
    zero = {
        "stage": 3,
        # Limit conversion/gradient scratch space and provide enough subgroups
        # for a useful partial-offload split on a 5.6B model.
        "sub_group_size": 100_000_000,
        "reduce_bucket_size": 20_000_000,
        "stage3_prefetch_bucket_size": 20_000_000,
        "stage3_param_persistence_threshold": 0,
        "stage3_max_live_parameters": 200_000_000,
        "stage3_max_reuse_distance": 0,
        "contiguous_gradients": True,
        "overlap_comm": False,
        "reduce_scatter": True,
    }
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": job["accumulation"],
        "gradient_clipping": job["grad_clip"],
        "bf16": {"enabled": True},
        "zero_optimization": zero,
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": job["lr"],
                "warmup_num_steps": job["warmup_updates"],
                "warmup_type": "linear",
            },
        },
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }
    if runtime["mode"] == MODES[0]:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": False, "ratio": runtime["offload_ratio"]}
        config["optimizer"] = {"type": "AdamW", "params": optimizer_kwargs(job)}
    else:
        # Pass the actual bnb optimizer to initialize(), never a fictitious
        # DeepSpeed optimizer type or an optim_bits flag on CPUAdam.
        config["zero_allow_untested_optimizer"] = True
    return config


def optimizer_kwargs(job):
    return {"lr": job["lr"], "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": job["weight_decay"]}


def external_optimizer(parameters, job, runtime):
    if runtime["mode"] != MODES[1]:
        return None
    # Optional CUDA dependency, loaded only for the selected training profile.
    optim = importlib.import_module("bitsandbytes.optim")
    return optim.AdamW8bit(parameters, **optimizer_kwargs(job), min_8bit_size=4096, is_paged=False)


def optimizer_layout(optimizer):
    """Return masters and their owning optimizer, including the GPU backup."""
    owners = {"primary": optimizer.optimizer}
    if hasattr(optimizer, "fp32_partitioned_groups_flat"):
        masters = optimizer.fp32_partitioned_groups_flat
        if getattr(optimizer, "swap_optimizer", False):
            raise ValueError("This checkpoint format does not support NVMe swapping")
        backup = getattr(optimizer, "backup_optimizer", None)
        if backup is not None:
            owners["gpu"] = backup
        assignment = []
        for i, master in enumerate(masters):
            device = getattr(optimizer, "subgroup_to_device", {}).get(i, master.device.type)
            assignment.append(
                "gpu" if backup is not None and torch.device(device).type != "cpu" else "primary"
            )
    else:
        masters = optimizer.single_partition_of_fp32_groups
        assignment = ["primary"] * len(masters)
    if not masters or any(p.dtype != torch.float32 or p.numel() == 0 for p in masters):
        raise ValueError("Expected nonempty FP32 master partitions")
    return masters, owners, assignment


def optimizer_audit(optimizer, runtime, initialized=False):
    masters, owners, assignment = optimizer_layout(optimizer)
    sizes = {"cpu": 0, "cuda": 0}
    state_sizes = {"cpu": 0, "cuda": 0}
    dtypes = set()
    quantized = 0
    for master, owner_name in zip(masters, assignment, strict=True):
        sizes[master.device.type] += master.numel() * master.element_size()
        state = owners[owner_name].state.get(master, {})
        keys = ("state1", "state2") if runtime["mode"] == MODES[1] else ("exp_avg", "exp_avg_sq")
        if initialized:
            if not all(key in state and state[key].numel() == master.numel() for key in keys):
                raise RuntimeError(f"Missing or malformed Adam state for {owner_name}")
            expected = (
                torch.uint8 if runtime["mode"] == MODES[1] and master.numel() >= 4096 else torch.float32
            )
            if any(state[key].dtype != expected for key in keys):
                raise RuntimeError(f"Adam state precision differs from the selected profile: {owner_name}")
            if expected == torch.uint8:
                quantized += master.numel()
                if not all(key in state for key in ("qmap1", "qmap2", "absmax1", "absmax2")):
                    raise RuntimeError("8-bit Adam state is missing quantization metadata")
        for value in state.values():
            if isinstance(value, torch.Tensor):
                state_sizes[value.device.type] += value.numel() * value.element_size()
        dtypes.update(str(state[key].dtype) for key in keys if key in state)
    if (
        runtime["mode"] == MODES[0]
        and 0 < runtime["offload_ratio"] < 1
        and (not sizes["cpu"] or not sizes["cuda"])
    ):
        raise RuntimeError(
            "Partial offload did not allocate masters on both CPU and GPU; reduce subgroup size"
        )
    return {
        "mode": runtime["mode"],
        "fp32_master_verified": True,
        "master_bytes_by_device": sizes,
        "optimizer_state_bytes_by_device": state_sizes,
        "moment_dtypes": sorted(dtypes),
        "quantized_parameter_count": quantized,
        "optimizer_types": {name: type(owner).__name__ for name, owner in owners.items()},
        "actual_cpu_master_fraction": sizes["cpu"] / sum(sizes.values()),
    }


def check_first_allocation(optimizer, runtime, root):
    """Budget lazy moment creation on its actual device before the first step."""
    masters, owners, assignment = optimizer_layout(optimizer)
    additions = {"cpu": 0, "cuda": 0}
    for master, owner_name in zip(masters, assignment, strict=True):
        state = owners[owner_name].state.get(master, {})
        if state:
            continue
        n = master.numel()
        size = 2 * n + 8 * math.ceil(n / 256) + 2048 if runtime["mode"] == MODES[1] and n >= 4096 else 8 * n
        additions[master.device.type] += size
    memory = check_memory()
    host = max(memory.get("worker_nonreclaimable_gib", 0), memory.get("process_current_gib", 0))
    projected = {
        "cpu": host + additions["cpu"] / 1024**3,
        "cuda": memory.get("gpu_allocated_gib", 0) + additions["cuda"] / 1024**3,
    }
    write_json(
        Path(root) / "first_optimizer_allocation.json",
        {
            "memory_before": memory,
            "new_state_bytes_by_device": additions,
            "projected_gib": projected,
            "limits_gib": {"cpu": 90, "cuda": 72},
        },
    )
    if projected["cpu"] >= 90 or projected["cuda"] >= 72:
        raise MemoryError(f"First optimizer allocation exceeds the training memory budget: {projected}")
