# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Convert upstream DreamZero weights: python -m lerobot.policies.dreamzero.convert_checkpoint."""

import argparse
import json
from pathlib import Path

import yaml

from .checkpoint import config_from_original, load_checked, load_original_adapter, resolve_checkpoint
from .modeling_dreamzero import DreamZeroPolicy
from .processor_dreamzero import make_dreamzero_pre_post_processors


def convert(source, output, *, embodiment="so101", base=None, revision=None, dataset_stats=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint directory: {output}")
    path = resolve_checkpoint(source, revision)
    original = json.loads((path / "config.json").read_text())
    metadata_path, experiment_path = path / "experiment_cfg/metadata.json", path / "experiment_cfg/conf.yaml"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    experiment = yaml.safe_load(experiment_path.read_text()) if experiment_path.exists() else None
    cfg = config_from_original(
        original,
        embodiment=embodiment,
        metadata=metadata,
        experiment=experiment,
        device="cpu",
        dtype="bfloat16",
        push_to_hub=False,
    )
    if embodiment == "so101" and dataset_stats is None:
        raise ValueError(
            "SO-101 conversion requires target dataset statistics; original robot statistics are incompatible"
        )
    if base and cfg.reset_action_heads:
        raise ValueError(
            "Adapter conversion must preserve its original embodiment; fine-tune the converted model separately"
        )
    policy = DreamZeroPolicy(cfg)
    if base:
        head = original["action_head_cfg"]["config"]
        policy.model, report = load_original_adapter(
            policy.model,
            path,
            resolve_checkpoint(base),
            rank=head["lora_rank"],
            alpha=head["lora_alpha"],
            targets=head["lora_target_modules"].split(","),
        )
    else:
        report = load_checked(
            policy.model,
            path,
            allow_missing=("state_encoder.", "action_encoder.", "action_decoder.")
            if cfg.reset_action_heads
            else (),
            discard_heads=cfg.reset_action_heads,
        )
    pre, post = make_dreamzero_pre_post_processors(cfg, dataset_stats=dataset_stats)
    output.mkdir(parents=True)
    policy.save_pretrained(output)
    pre.save_pretrained(output)
    post.save_pretrained(output)
    (output / "conversion_report.json").write_text(
        json.dumps({"source": source, "revision": revision, "embodiment": embodiment, **report}, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embodiment", choices=["so101", "oxe_droid", "agibot"], default="so101")
    parser.add_argument("--base", help="Explicit dense base for an upstream LoRA checkpoint")
    parser.add_argument("--revision")
    parser.add_argument("--dataset-stats", type=Path, help="SO-101 LeRobot meta/stats.json")
    args = parser.parse_args()
    stats = json.loads(args.dataset_stats.read_text()) if args.dataset_stats else None
    convert(
        args.source,
        args.output,
        embodiment=args.embodiment,
        base=args.base,
        revision=args.revision,
        dataset_stats=stats,
    )


if __name__ == "__main__":
    main()
