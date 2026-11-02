"""Deterministic sampling, bounded-memory checkpoints, and LIBERO action contracts."""

import contextlib
import hashlib
import json
import os
import random
import resource
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def write_json(path, value):
    """Atomically publish a JSON result, rejecting nonfinite numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def file_hash(path, algorithm="sha256", git_blob=False):
    """Hash a file without materializing it."""
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {Path(path).stat().st_size}\0".encode())
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def split_episodes(episodes, seed=1000, tail=64):
    """Stratify entire episodes; never split adjacent windows across train/validation."""
    groups = defaultdict(list)
    excluded = []
    for ep in episodes:
        row = {
            "episode": int(ep["episode_index"]),
            "start": int(ep["dataset_from_index"]),
            "stop": int(ep["dataset_to_index"]) - tail,
            "end": int(ep["dataset_to_index"]),
        }
        if row["stop"] <= row["start"]:
            excluded.append(row["episode"])
            continue
        if len(ep["tasks"]) != 1:
            raise ValueError("LIBERO episodes must have exactly one instruction")
        groups[ep["tasks"][0]].append(row)
    rng = random.Random(seed)
    train, validation = {}, {}
    for task in sorted(groups):
        rows = sorted(groups[task], key=lambda row: row["episode"])
        if len(rows) < 2:
            raise ValueError(f"Not enough valid episodes for task {task}")
        rng.shuffle(rows)
        count = max(1, int(len(rows) * 0.1))
        validation[task], train[task] = rows[:count], rows[count:]
    return {
        "train": train,
        "validation": validation,
        "excluded_short_episodes": excluded,
        "seed": seed,
        "drop_n_last_frames": tail,
        "window_offsets": list(range(0, tail + 1, 2)),
    }


class BalancedWindows:
    """Sample task, episode, then valid start independently of episode length."""

    def __init__(self, groups, seed=1000):
        self.groups = groups
        self.tasks = sorted(groups)
        if not self.tasks or any(not groups[t] for t in self.tasks):
            raise ValueError("Every task needs a nonempty episode set")
        self.rng = random.Random(seed)
        self.samples = 0

    def next(self):
        """Return one absolute dataset index."""
        task = self.rng.choice(self.tasks)
        episode = self.rng.choice(self.groups[task])
        self.samples += 1
        return self.rng.randrange(episode["start"], episode["stop"])

    def state_dict(self):
        """Capture the next sample position, including RNG state."""
        return {"rng": self.rng.getstate(), "samples": self.samples}

    def load_state_dict(self, state):
        """Restore sampling without replaying or skipping data."""
        self.rng.setstate(state["rng"])
        self.samples = state["samples"]

    def __iter__(self):
        while True:
            yield self.next()


def validation_indices(groups, seed=1001):
    """Choose five fixed windows per task across held-out episodes."""
    rng = random.Random(seed)
    return [
        {"task": task, "index": rng.randrange(rows[i % len(rows)]["start"], rows[i % len(rows)]["stop"])}
        for task in sorted(groups)
        for rows in [groups[task]]
        for i in range(5)
    ]


def rng_state():
    """Capture Python, NumPy, CPU and all CUDA RNG streams."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    """Restore all RNG streams."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextlib.contextmanager
def isolated_rng(seed):
    """Run validation without perturbing future training randomness."""
    state = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(state)


def sim_action(raw):
    """Convert dataset gripper [closed=0, open=1] to LIBERO's OSC_POSE command."""
    result = np.asarray(raw, dtype=np.float32).reshape(7).copy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite LIBERO action")
    result[:6] = np.clip(result[:6], -1, 1)
    result[6] = np.sign(1 - 2 * result[6])
    return result


def camera_images(obs):
    """Match the existing LIBERO dataset camera order and 180-degree orientation."""
    return [
        np.ascontiguousarray(obs[key][::-1, ::-1]) for key in ("agentview_image", "robot0_eye_in_hand_image")
    ]


def memory_snapshot():
    """Report process high-water RSS and non-reclaimable worker memory."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    process_peak_gib = peak / (1024**2 if sys.platform.startswith("linux") else 1024**3)
    result = {"process_peak_gib": process_peak_gib}
    statm = Path("/proc/self/statm")
    if statm.exists():
        result["process_current_gib"] = (
            int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024**3
        )
    stat = Path("/sys/fs/cgroup/memory.stat")
    if stat.exists():
        values = {k: int(v) for k, v in (line.split() for line in stat.read_text().splitlines())}
        result["worker_nonreclaimable_gib"] = (values.get("anon", 0) + values.get("kernel", 0)) / 1024**3
        current = Path("/sys/fs/cgroup/memory.current")
        peak_path = Path("/sys/fs/cgroup/memory.peak")
        if current.exists():
            result["worker_current_gib"] = int(current.read_text()) / 1024**3
        if peak_path.exists():
            result["worker_peak_gib"] = int(peak_path.read_text()) / 1024**3
    else:
        # The retained Merlin worker uses cgroup v1 on some hosts.
        legacy = Path("/sys/fs/cgroup/memory/memory.stat")
        if legacy.exists():
            values = {k: int(v) for k, v in (line.split() for line in legacy.read_text().splitlines())}
            result["worker_nonreclaimable_gib"] = values.get("total_rss", values.get("rss", 0)) / 1024**3
            usage = legacy.parent / "memory.usage_in_bytes"
            result["worker_current_gib"] = int(usage.read_text()) / 1024**3
    if torch.cuda.is_available():
        result["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        result["gpu_allocated_gib"] = torch.cuda.memory_allocated() / 1024**3
        result["gpu_reserved_gib"] = torch.cuda.memory_reserved() / 1024**3
    return result


def check_memory():
    """Enforce the agreed 90 GiB host and 72 GiB device budgets."""
    values = memory_snapshot()
    # File cache is reclaimable; the worker's lifetime peak can include earlier jobs.
    # Enforce this run's RSS high water and the worker's current resident allocations.
    if max(values.get("worker_nonreclaimable_gib", 0), values["process_peak_gib"]) >= 90:
        raise MemoryError(f"Host memory budget exceeded: {values}")
    if max(values.get("gpu_peak_allocated_gib", 0), values.get("gpu_reserved_gib", 0)) >= 72:
        raise MemoryError(f"GPU memory budget exceeded: {values}")
    return values


def write_tensor_tree(directory, tensors, chunk_bytes=256 * 1024**2):
    """Save named tensors in bounded flat chunks; optimizer tensors may exceed RAM headroom."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {}
    counter = 0
    for name, tensor in tensors.items():
        value = tensor.detach().reshape(-1)
        width = max(1, chunk_bytes // tensor.element_size())
        pieces = []
        for start in range(0, value.numel(), width):
            filename = f"tensor-{counter:06d}.safetensors"
            part = value[start : start + width].cpu().contiguous()
            save_file({"value": part}, directory / filename, metadata={"format": "pt"})
            pieces.append(
                {
                    "file": filename,
                    "start": start,
                    "length": part.numel(),
                    "sha256": file_hash(directory / filename),
                }
            )
            counter += 1
            del part
            check_memory()
        manifest[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "pieces": pieces}
    write_json(directory / "index.json", manifest)


def read_tensor_tree(directory, tensors):
    """Restore tensor chunks directly into existing allocations without a second state dictionary."""
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    if set(index) != set(tensors):
        raise ValueError("Checkpoint tensor names do not match the optimizer")
    with torch.no_grad():
        for name, destination in tensors.items():
            spec = index[name]
            if list(destination.shape) != spec["shape"] or str(destination.dtype) != spec["dtype"]:
                raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
            flat = destination.reshape(-1)
            offset = 0
            for part in spec["pieces"]:
                path = directory / part["file"]
                if part["start"] != offset or file_hash(path) != part["sha256"]:
                    raise ValueError(f"Corrupt checkpoint chunk: {path}")
                with safe_open(path, framework="pt", device="cpu") as reader:
                    value = reader.get_tensor("value")
                    if value.numel() != part["length"]:
                        raise ValueError(f"Incorrect chunk length: {path}")
                    flat[offset : offset + value.numel()].copy_(value)
                offset += part["length"]
                del value
                check_memory()
            if offset != flat.numel():
                raise ValueError(f"Incomplete checkpoint tensor: {name}")


def policy_state_tensors(policy):
    """Expose single-rank ZeRO-3 partition storage without gathering the model."""
    tensors = policy.state_dict()
    parameters = dict(policy.named_parameters(remove_duplicate=False))
    for name, parameter in parameters.items():
        if name not in tensors or not hasattr(parameter, "ds_tensor"):
            continue
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise ValueError("Streaming policy export supports only one rank")
        if parameter.ds_tensor.numel() < parameter.ds_numel:
            raise ValueError(f"Incomplete local parameter partition: {name}")
        tensors[name] = parameter.ds_tensor[: parameter.ds_numel].view(parameter.ds_shape)
    return tensors


def export_policy(policy, directory, pre, post, chunk_bytes=256 * 1024**2):
    """Export normal Hugging Face safetensor shards, using bounded CPU staging."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    weights, weight_map, size, shard = {}, {}, 0, 0
    total = 0

    def flush():
        nonlocal weights, size, shard
        filename = f"model-{shard:05d}.safetensors"
        save_file(weights, directory / filename, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(weights, filename))
        weights, size, shard = {}, 0, shard + 1
        check_memory()

    for name, tensor in policy_state_tensors(policy).items():
        length = tensor.numel() * tensor.element_size()
        if weights and size + length > chunk_bytes:
            flush()
        weights[name] = tensor.detach().cpu().contiguous()
        size += length
        total += length
    if weights:
        flush()
    policy.config.save_pretrained(directory)
    pre.save_pretrained(directory, config_filename="policy_preprocessor.json")
    post.save_pretrained(directory, config_filename="policy_postprocessor.json")
    write_json(
        directory / "model.safetensors.index.json",
        {"metadata": {"total_size": total}, "weight_map": weight_map},
    )
    verify_export(directory)


def verify_export(directory):
    """Check every shard's header, byte length, ownership and complete tensor inventory."""
    directory = Path(directory)
    mapping = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    seen = set()
    for name in set(mapping.values()):
        if Path(name).name != name:
            raise ValueError("Invalid shard filename")
        path = directory / name
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
        keys = set(header) - {"__metadata__"}
        if seen & keys or any(mapping.get(k) != name for k in keys):
            raise ValueError("Duplicate or incorrectly indexed shard tensor")
        if 8 + length + max(header[k]["data_offsets"][1] for k in keys) != path.stat().st_size:
            raise ValueError("Truncated safetensor shard")
        seen |= keys
    if seen != set(mapping):
        raise ValueError("Incomplete model export")


def candidate_key(result, validation_mse):
    """Rank candidates by macro suite success, breaking ties on held-out action MSE."""
    rates = [suite["success_rate"] for suite in result["suites"].values()]
    return (sum(rates) / len(rates), -validation_mse)


def full_eval_eligible(result):
    """Require a complete selection evaluation and success in all four suites."""
    return (
        result["complete"]
        and result["episodes"] == 40
        and result["success_rate"] >= 0.2
        and len(result["suites"]) == 4
        and all(s["successes"] > 0 for s in result["suites"].values())
    )
