"""Bounded-memory single-rank ZeRO checkpoints, including hybrid and 8-bit state.

Format 2 stores both CPU and GPU optimizers, original tensor dtypes/devices,
and all bitsandbytes quantization metadata. No full optimizer copy is loaded.
"""

import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import torch

from .libero_full_runtime import optimizer_layout
from .libero_full_utils import (
    check_memory,
    file_hash,
    policy_state_tensors,
    read_tensor_tree,
    write_json,
    write_tensor_tree,
)


def _layout(engine):
    zero = engine.optimizer
    if engine.dp_world_size != 1 or zero.dynamic_loss_scale:
        raise ValueError("Streaming checkpoints require single-rank BF16 ZeRO")
    return optimizer_layout(zero)


def _fingerprint(root):
    names = [
        "policy_config/config.json",
        "split.json",
        "train_stats.json",
        "deepspeed.json",
        "wan_dit_verified.json",
        "dataset_hashes.json",
        "protocol_verified.json",
    ]
    if (Path(root) / "runtime.json").exists():
        names.append("runtime.json")
    return {name: file_hash(Path(root) / name) for name in names}


def _versions(owners):
    result = {"deepspeed": importlib.metadata.version("deepspeed"), "torch": str(torch.__version__)}
    if any(type(owner).__module__.startswith("bitsandbytes") for owner in owners.values()):
        result["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
    return result


def _description(masters, owners, assignment):
    return {
        "masters": [
            {"shape": list(p.shape), "device": p.device.type, "owner": name}
            for p, name in zip(masters, assignment, strict=True)
        ],
        "owners": {
            name: f"{type(owner).__module__}.{type(owner).__qualname__}" for name, owner in owners.items()
        },
    }


def _check_state(state):
    if {"exp_avg", "exp_avg_sq"}.issubset(state):
        return
    if {"state1", "state2"}.issubset(state):
        if state["state1"].dtype == torch.uint8 and not {"qmap1", "qmap2", "absmax1", "absmax2"}.issubset(
            state
        ):
            raise ValueError("8-bit state has no quantization metadata")
        return
    raise ValueError("Adam state must be initialized before a complete save")


def save_resume(engine, balanced, update, root, rng):
    masters, owners, assignment = _layout(engine)
    target = Path(root) / "checkpoints" / f"update_{update:06d}"
    target.mkdir(parents=True, exist_ok=False)
    tensors = {f"model/{name}": value for name, value in policy_state_tensors(engine.module).items()}
    scalar_states, tensor_devices = [], {}
    for i, (master, name) in enumerate(zip(masters, assignment, strict=True)):
        tensors[f"master/{i}"] = master
        state = owners[name].state[master]
        _check_state(state)
        scalar_states.append({k: v for k, v in state.items() if not isinstance(v, torch.Tensor)})
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                tensor_name = f"adam/{i}/{key}"
                tensors[tensor_name] = value
                tensor_devices[tensor_name] = value.device.type
    write_tensor_tree(target / "tensors", tensors)
    zero = engine.optimizer
    metadata = {
        "format_version": 2,
        "versions": _versions(owners),
        "layout": _description(masters, owners, assignment),
        "fingerprint": _fingerprint(root),
        "update": update,
        "balanced": balanced.state_dict(),
        "rng": rng,
        "parameter_order": [name for name, p in engine.module.named_parameters() if p.requires_grad],
        "param_groups": {
            name: [{k: v for k, v in group.items() if k != "params"} for group in owner.param_groups]
            for name, owner in owners.items()
        },
        "scalar_states": scalar_states,
        "tensor_devices": tensor_devices,
        "scheduler": engine.lr_scheduler.state_dict(),
        "engine": {
            k: getattr(engine, k) for k in ("global_steps", "global_samples", "micro_steps", "skipped_steps")
        },
        "zero": {
            k: getattr(zero, k)
            for k in ("loss_scaler", "dynamic_loss_scale", "overflow", "clip_grad", "micro_step_id")
        },
    }
    state_dir = target / "training_state"
    state_dir.mkdir()
    torch.save(metadata, state_dir / "state.pt")
    write_json(
        target / "COMPLETE.json",
        {
            "update": update,
            "state_sha256": file_hash(state_dir / "state.pt"),
            "tensor_index_sha256": file_hash(target / "tensors/index.json"),
            "created_unix": time.time(),
            "memory": check_memory(),
        },
    )
    complete = sorted(p for p in target.parent.glob("update_*") if (p / "COMPLETE.json").is_file())
    for stale in complete[:-2]:
        shutil.rmtree(stale)
    return target


def resume(engine, balanced, checkpoint, root):
    checkpoint = Path(checkpoint)
    marker = json.loads((checkpoint / "COMPLETE.json").read_text())
    state_path = checkpoint / "training_state/state.pt"
    if (
        file_hash(state_path) != marker["state_sha256"]
        or file_hash(checkpoint / "tensors/index.json") != marker["tensor_index_sha256"]
    ):
        raise ValueError("Checkpoint metadata checksum mismatch")
    saved = torch.load(state_path, map_location="cpu", weights_only=False)
    masters, owners, assignment = _layout(engine)
    if (
        saved.get("format_version") != 2
        or saved["fingerprint"] != _fingerprint(root)
        or saved["versions"] != _versions(owners)
        or saved["layout"] != _description(masters, owners, assignment)
        or saved["parameter_order"] != [n for n, p in engine.module.named_parameters() if p.requires_grad]
    ):
        raise ValueError("Checkpoint format, runtime, data or parameter layout changed")
    tensors = {f"model/{name}": value for name, value in policy_state_tensors(engine.module).items()}
    index = json.loads((checkpoint / "tensors/index.json").read_text())
    additions = {"cpu": 0, "cuda": 0}
    for i, (master, owner_name) in enumerate(zip(masters, assignment, strict=True)):
        state = owners[owner_name].state.get(master, {})
        prefix = f"adam/{i}/"
        for name, spec in index.items():
            if name.startswith(prefix) and name.removeprefix(prefix) not in state:
                dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
                size = torch.Size(spec["shape"]).numel() * torch.empty((), dtype=dtype).element_size()
                additions[saved["tensor_devices"][name]] += size
    memory = check_memory()
    host = max(memory.get("process_current_gib", 0), memory.get("worker_nonreclaimable_gib", 0))
    if (
        host + additions["cpu"] / 1024**3 >= 90
        or memory.get("gpu_allocated_gib", 0) + additions["cuda"] / 1024**3 >= 72
    ):
        raise MemoryError("Restoring optimizer tensors would exceed the training memory budget")
    for i, (master, owner_name) in enumerate(zip(masters, assignment, strict=True)):
        tensors[f"master/{i}"] = master
        state = owners[owner_name].state[master]
        state.update(saved["scalar_states"][i])
        prefix = f"adam/{i}/"
        for name, spec in index.items():
            if name.startswith(prefix):
                key = name.removeprefix(prefix)
                dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
                device = saved["tensor_devices"][name]
                if key not in state:
                    state[key] = torch.empty(spec["shape"], device=device, dtype=dtype)
                if state[key].device.type != device:
                    raise ValueError(f"Optimizer state device mismatch: {name}")
                tensors[name] = state[key]
        _check_state(state)
    for name, owner in owners.items():
        groups = saved["param_groups"][name]
        if len(groups) != len(owner.param_groups):
            raise ValueError(f"Optimizer group count changed: {name}")
        for group, source in zip(owner.param_groups, groups, strict=True):
            group.update(source)
    read_tensor_tree(checkpoint / "tensors", tensors)
    # ZeRO may retain gathered copies of small parameters at initialization.
    # Keep those copies coherent with the restored partition storage.
    with torch.no_grad():
        for parameter in engine.module.parameters():
            if hasattr(parameter, "ds_tensor") and parameter.numel() == parameter.ds_numel:
                parameter.copy_(parameter.ds_tensor[: parameter.ds_numel].view(parameter.ds_shape))
    engine.lr_scheduler.load_state_dict(saved["scheduler"])
    if "gpu" in owners:
        for primary, backup in zip(owners["primary"].param_groups, owners["gpu"].param_groups, strict=True):
            backup["lr"] = primary["lr"]
    for name, value in saved["engine"].items():
        setattr(engine, name, value)
    for name, value in saved["zero"].items():
        setattr(engine.optimizer, name, value)
    balanced.load_state_dict(saved["balanced"])
    if engine.global_steps != saved["update"]:
        raise ValueError("Restored optimizer update counter mismatch")
    check_memory()
    return saved["update"], saved["rng"]
