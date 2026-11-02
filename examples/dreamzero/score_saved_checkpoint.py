"""Export an intact ZeRO checkpoint and score the fixed matched LIBERO subset without training."""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from .libero_full_utils import file_hash, verify_export, write_json


def export_checkpoint(root, output, update):
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    torch.set_num_threads(8)
    tag = f"update_{update:06d}"
    directory = output / "model"
    if (output / "export_verified.json").exists():
        verify_export(directory)
        return
    source = root / "checkpoints" / tag
    for path in sorted(source.glob("*_model_states.pt")):
        metadata = torch.load(path, map_location="cpu", weights_only=False)
        if metadata["global_steps"] != update or metadata["update"] != update:
            raise ValueError("Checkpoint update identity mismatch")
    for path in sorted(source.glob("*.pt")):
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist():
                raise ValueError("Empty checkpoint archive")
    state = get_fp32_state_dict_from_zero_checkpoint(str(root / "checkpoints"), tag=tag, lazy_mode=True)
    directory.mkdir(parents=True, exist_ok=True)
    tensors, mapping, digests = {}, {}, {}
    shard_bytes, total_bytes, shard_index = 0, 0, 0

    def flush():
        nonlocal tensors, shard_bytes, shard_index
        name = f"model-{shard_index:05d}.safetensors"
        temporary = directory / (name + ".tmp")
        save_file(tensors, temporary, metadata={"format": "pt"})
        temporary.replace(directory / name)
        mapping.update(dict.fromkeys(tensors, name))
        digests[name] = file_hash(directory / name)
        tensors, shard_bytes, shard_index = {}, 0, shard_index + 1

    for name, lazy in state.items():
        tensor = lazy.contiguous()
        if tensor.is_floating_point():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite checkpoint tensor: {name}")
            tensor = tensor.to(torch.bfloat16)
        length = tensor.numel() * tensor.element_size()
        if tensors and shard_bytes + length > 256 * 1024**2:
            flush()
        tensors[name] = tensor.clone()
        shard_bytes += length
        total_bytes += length
    if tensors:
        flush()
    shutil.copy2(root / "policy_config/config.json", directory / "config.json")
    write_json(directory / "model.safetensors.index.json", {"metadata": {"total_size": total_bytes},
                                                           "weight_map": mapping})
    verify_export(directory)
    write_json(output / "export_verified.json", {"source": str(source), "update": update,
        "tensors": len(mapping), "bytes": total_bytes, "shards_sha256": digests,
        "source_archive_directories_readable": True, "source_full_crc_checked": False,
        "all_exported_tensors_finite": True, "dtype": "BF16", "additional_training_updates": 0})
    print(f"Export verified: {len(mapping)} tensors, {total_bytes} bytes", flush=True)


def evaluate(root, output, update, rank):
    from libero.libero.envs import OffScreenRenderEnv

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig
    from lerobot.policies.dreamzero.modeling_dreamzero import DreamZeroPolicy
    from lerobot.policies.dreamzero.processor_dreamzero import make_dreamzero_pre_post_processors

    from .evaluate_libero_full import RESOLUTION, SUITES, _run_episode, _task, protocol_fingerprint
    from .train_libero_full import _precompute_text

    if rank not in (0, 1):
        raise ValueError("Expected one of two evaluation ranks")
    if json.loads((output / "export_verified.json").read_text())["update"] != update:
        raise ValueError("Export belongs to another update")
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    device = f"cuda:{rank}"
    job = json.loads((root / "job.json").read_text())
    fingerprint = protocol_fingerprint(job["protocol_manifest"])
    assignments = [(suite, task_id) for suite in SUITES for task_id in (0, 1)][rank::2]
    tasks = [_task(suite, task_id) for suite, task_id in assignments]
    probe = OffScreenRenderEnv(bddl_file_name=str(tasks[0][2]), camera_heights=RESOLUTION,
        camera_widths=RESOLUTION, render_gpu_device_id=rank)
    try:
        observation = probe.reset()
        if observation["agentview_image"].shape != (RESOLUTION, RESOLUTION, 3):
            raise ValueError("Unexpected LIBERO camera shape")
    finally:
        probe.close()
    write_json(output / f"status_rank{rank}.json", {"stage": "loading_model", "render_smoke_passed": True})
    print(f"rank={rank}: rendering passed, loading checkpoint", flush=True)
    config = DreamZeroConfig.from_pretrained(output / "model")
    config.device, config.text_encoder_device = "cpu", device
    policy = DreamZeroPolicy.from_pretrained(output / "model", config=config).eval()
    _precompute_text(policy, [task.language for task, _, _ in tasks])
    frozen = policy._components
    cached_text = frozen.encode_text
    frozen.encode_text = lambda phrases: cached_text(phrases).to(device=device, dtype=torch.bfloat16)
    frozen.device = torch.device(device)
    frozen.vae.to(device)
    frozen.image.to(device)
    policy.to(device=device, dtype=torch.bfloat16)
    config.device = device
    gc.collect()
    torch.cuda.empty_cache()
    meta = LeRobotDatasetMetadata(job["repo_id"], root=job["dataset_root"])
    pre, post = make_dreamzero_pre_post_processors(config, dataset_meta=meta)
    path = output / f"episodes_rank{rank}.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    completed = {(row["suite"], row["task_id"], row["trial"]) for row in records}
    if len(completed) != len(records) or any(row["checkpoint_update"] != update for row in records):
        raise ValueError("Existing records are duplicated or refer to another checkpoint")
    with path.open("a") as stream, torch.inference_mode():
        for (suite, task_id), (task, states, bddl) in zip(assignments, tasks, strict=True):
            environment = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RESOLUTION,
                camera_widths=RESOLUTION, control_freq=20, hard_reset=True, render_gpu_device_id=rank)
            try:
                for trial in range(5):
                    if (suite, task_id, trial) in completed:
                        continue
                    write_json(output / f"status_rank{rank}.json", {"stage": "evaluating", "completed": len(records),
                        "suite": suite, "task_id": task_id, "trial": trial, "started_unix": time.time()})
                    row = _run_episode(environment, states[trial], task, policy, pre, post, suite, task_id, trial)
                    row["checkpoint_update"] = update
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    records.append(row)
                    print(json.dumps(row), flush=True)
            finally:
                environment.close()
    write_json(output / f"status_rank{rank}.json", {"stage": "completed", "completed": len(records),
        "successes": sum(row["success"] for row in records), "protocol_fingerprint": fingerprint})


def report(root, output, update):
    from .evaluate_libero_distributed import matched_fastwam_selection
    from .evaluate_libero_full import SUITES, _aggregate

    records = []
    for rank in (0, 1):
        records.extend(json.loads(line) for line in (output / f"episodes_rank{rank}.jsonl").read_text().splitlines())
    expected = {(suite, task_id, trial) for suite in SUITES for task_id in (0, 1) for trial in range(5)}
    if len(records) != 40 or {(row["suite"], row["task_id"], row["trial"]) for row in records} != expected:
        raise ValueError("The matched 40-episode evaluation is incomplete")
    if any(row["checkpoint_update"] != update for row in records):
        raise ValueError("Mixed checkpoint updates in evaluation")
    job = json.loads((root / "job.json").read_text())
    baseline_path = Path(job["previous_root"]).parent / "fastwam_libero_20261023/comparison.json"
    baseline = matched_fastwam_selection(json.loads(baseline_path.read_text()))
    result = {"complete": True, "checkpoint_update": update, "dreamzero": _aggregate(records, True, update),
        "fastwam_matched_40": baseline, "baseline_source": str(baseline_path),
        "scope": "40 matched LIBERO initial states; not the full 2000-episode benchmark",
        "note": "DreamZero checkpoint has only four optimizer updates; FastWAM is an official trained checkpoint. This is not a controlled architecture comparison."}
    write_json(output / "comparison.json", result)
    print(json.dumps({"dreamzero": result["dreamzero"]["success_rate"],
                      "fastwam": baseline["success_rate"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("export", "evaluate", "report", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--update", type=int, default=4)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("LOCAL_RANK", "0")))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "export":
        export_checkpoint(args.root, args.output, args.update)
    elif args.mode == "evaluate":
        evaluate(args.root, args.output, args.update, args.rank)
    elif args.mode == "run":
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                        "--module", "examples.dreamzero.score_saved_checkpoint", "evaluate",
                        "--root", str(args.root), "--output", str(args.output), "--update", str(args.update)],
                       check=True)
        report(args.root, args.output, args.update)
    else:
        report(args.root, args.output, args.update)
