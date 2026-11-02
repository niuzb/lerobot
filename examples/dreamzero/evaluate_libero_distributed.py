"""Evaluate the final full-training checkpoint against the recorded FastWAM baseline."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig
from lerobot.policies.dreamzero.modeling_dreamzero import DreamZeroPolicy
from lerobot.policies.dreamzero.processor_dreamzero import make_dreamzero_pre_post_processors

from .evaluate_libero_full import SUITES, evaluate_full, evaluate_selection
from .libero_full_utils import write_json
from .train_libero_full import _precompute_text


def matched_fastwam_selection(baseline):
    records = []
    selected = [task for task in baseline["tasks"] if task["task_id"] in (0, 1)]
    if len(selected) != 8 or {(row["suite"], row["task_id"]) for row in selected} != {
        (suite, task_id) for suite in SUITES for task_id in (0, 1)
    }:
        raise ValueError("Expected eight FastWAM tasks for the matched 40-episode subset")
    for task in selected:
        result = json.loads(Path(task["result_file"]).read_text())
        if result["task_suite"] != task["suite"] or result["task_id"] != task["task_id"]:
            raise ValueError("FastWAM result file belongs to a different task")
        successes, failures = set(result["success_episodes"]), set(result["failure_episodes"])
        if successes & failures or successes | failures != set(range(50)):
            raise ValueError("FastWAM episode identities are incomplete or overlapping")
        for trial in range(5):
            records.append({"suite": task["suite"], "task_id": task["task_id"],
                            "trial": trial, "success": trial in successes})
    return {"episodes": len(records), "successes": sum(row["success"] for row in records),
            "success_rate": sum(row["success"] for row in records) / len(records), "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--update", type=int, default=20000)
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    job = json.loads((root / "job.json").read_text())
    final = args.update == job["optimizer_updates"]
    receipt = root / "training_complete.json" if final else root / "segments" / f"update_{args.update:06d}.json"
    completed = json.loads(receipt.read_text())
    if completed["update"] != args.update:
        raise RuntimeError("Requested checkpoint budget has not been completed")
    directory = root / "final_model" if final else root / "candidates" / f"update_{args.update:06d}"
    if not (directory / "model.safetensors").exists():
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        temporary = directory / "model.safetensors.tmp"
        save_file({name: value.contiguous() for name, value in state.items()}, temporary, metadata={"format": "pt"})
        temporary.replace(directory / "model.safetensors")
        del state
    config = DreamZeroConfig.from_pretrained(directory)
    config.device = "cuda:0"
    config.text_encoder_device = "cpu"
    policy = DreamZeroPolicy.from_pretrained(directory, config=config).eval()
    policy.to(device="cuda:0", dtype=torch.bfloat16)
    split = json.loads((root / "split.json").read_text())
    _precompute_text(policy, sorted(split["train"]))
    meta = LeRobotDatasetMetadata(job["repo_id"], root=job["dataset_root"])
    preprocessor, postprocessor = make_dreamzero_pre_post_processors(config, dataset_meta=meta)
    with torch.inference_mode():
        selection = evaluate_selection(policy, preprocessor, postprocessor, root, completed["update"],
                                       protocol_manifest=job["protocol_manifest"])
        result = selection if args.selection_only else evaluate_full(
            policy, preprocessor, postprocessor, root, completed["update"],
            selection_result=selection, protocol_manifest=job["protocol_manifest"])
    baseline_path = Path(job["previous_root"]).parent / "fastwam_libero_20261023/comparison.json"
    baseline = json.loads(baseline_path.read_text())
    write_json(root / "comparisons" / f"update_{args.update:06d}.json", {
        "complete": result["complete"], "dreamzero": result,
        "update": args.update,
        "validation": json.loads((root / "validation" / f"update_{args.update:06d}.json").read_text()),
        "fastwam_matched_40": matched_fastwam_selection(baseline) if args.selection_only else None,
        "fastwam_full_2000_reference": baseline, "fastwam_source": str(baseline_path),
        "note": "Compare the selection-only run to fastwam_matched_40, not the full 2000-episode reference. Different initialization and training histories are not an architecture-controlled comparison.",
    })


if __name__ == "__main__":
    main()
