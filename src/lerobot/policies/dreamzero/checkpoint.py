# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Audited, shard-by-shard imports of raw Wan and original DreamZero weights."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.import_utils import _peft_available, require_package

from .configuration_dreamzero import DreamZeroConfig

if TYPE_CHECKING or _peft_available:
    from peft import LoraConfig, get_peft_model
else:
    LoraConfig = get_peft_model = None

HEAD_PREFIXES = ("state_encoder.", "action_encoder.", "action_decoder.")
FROZEN_PREFIXES = ("action_head.text_encoder.", "action_head.image_encoder.", "action_head.vae.")


def resolve_checkpoint(source: str, revision: str | None = None) -> Path:
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    return Path(
        snapshot_download(
            source,
            revision=revision,
            allow_patterns=["*.safetensors", "*.json", "experiment_cfg/*.json", "experiment_cfg/*.yaml"],
        )
    )


def weight_files(path: Path) -> tuple[list[Path], dict[str, str] | None]:
    indices = list(path.glob("*.safetensors.index.json"))
    if len(indices) > 1:
        raise ValueError("Ambiguous checkpoint: multiple safetensors indices")
    if indices:
        mapping = json.loads(indices[0].read_text())["weight_map"]
        filenames = sorted(set(mapping.values()))
        if any(Path(name).name != name for name in filenames):
            raise ValueError("Checkpoint shards must be files in the checkpoint directory")
        files = [path / name for name in filenames]
    else:
        files, mapping = sorted(path.glob("*.safetensors")), None
        if len(files) != 1:
            raise ValueError("Expected one safetensors file or an explicit shard index")
    if any(not f.is_file() for f in files):
        raise FileNotFoundError("Checkpoint is missing a safetensors shard")
    return files, mapping


def canonical_key(name: str) -> str | None:
    if name.startswith(FROZEN_PREFIXES):
        return None
    for prefix in ("action_head.model.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    return name.replace(".base_layer.", ".")


def load_checked(model, path: Path, *, allow_missing=(), discard_heads=False, adapter_only=False):
    """Audit ALL names and shapes before copying any tensor; never silently partial-load."""
    files, index = weight_files(path)
    target = model.state_dict()
    aliases = {canonical_key(key): key for key in target}
    mapping = {}
    raw_seen = set()
    ignored = []
    for shard in files:
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for raw in reader.keys():  # noqa: SIM118 - safe_open is not iterable
                if raw in raw_seen or (index is not None and index.get(raw) != shard.name):
                    raise ValueError(f"Duplicate or incorrectly indexed weight: {raw}")
                raw_seen.add(raw)
                key = canonical_key(raw)
                if key is None or (discard_heads and key.startswith(HEAD_PREFIXES)):
                    ignored.append(raw)
                    continue
                if key in mapping:
                    raise ValueError(f"Duplicate checkpoint alias: {key}")
                if key not in aliases:
                    raise ValueError(f"Unexpected checkpoint parameter: {raw}")
                dest = aliases[key]
                if tuple(reader.get_slice(raw).get_shape()) != tuple(target[dest].shape):
                    raise ValueError(f"Shape mismatch for {raw}: expected {tuple(target[dest].shape)}")
                mapping[key] = (shard, raw, dest)
    if index is not None and raw_seen != set(index):
        raise ValueError("Shard index does not match the tensors in the checkpoint")
    missing = sorted(set(aliases) - set(mapping))
    if adapter_only:
        required = {key for key in aliases if ".lora_" in key or key.startswith(HEAD_PREFIXES)}
        invalid_missing = sorted(required - set(mapping))
    else:
        invalid_missing = [key for key in missing if not key.startswith(tuple(allow_missing))]
    if invalid_missing:
        raise ValueError(f"Checkpoint missing required parameters: {invalid_missing[:12]}")
    # Avoid accumulating a second 14B state dict in memory.
    for shard in files:
        with safe_open(shard, framework="pt", device="cpu") as reader, torch.no_grad():
            for file, raw, dest in mapping.values():
                if file == shard:
                    target[dest].copy_(reader.get_tensor(raw))
    return {"loaded": len(mapping), "missing": missing, "ignored": ignored}


def initialize_model(model, config):
    path = resolve_checkpoint(config.init_from, config.init_revision)
    if "snapshots" in path.parts:
        config.init_revision = path.parts[path.parts.index("snapshots") + 1]
    allowed = HEAD_PREFIXES if config.reset_action_heads or config.init_kind == "wan" else ()
    if config.init_kind == "wan" and config.backbone == "wan22_5b":
        # Stock TI2V has no CLIP conditioning branch. Its new image projections
        # are trainable in both full and LoRA modes.
        allowed += ("img_emb.",)
        allowed += tuple(
            f"blocks.{i}.cross_attn.{part}."
            for i in range(len(model.blocks))
            for part in ("k_img", "v_img", "norm_k_img")
        )
    return load_checked(model, path, allow_missing=allowed, discard_heads=config.reset_action_heads)


def config_from_original(
    original: dict, *, embodiment: str = "so101", metadata=None, experiment=None, **overrides
):
    """Translate known configuration fields; never instantiate Hydra target strings."""
    head = original["action_head_cfg"]["config"]
    architecture = {
        k: v
        for k, v in head["diffusion_model_cfg"].items()
        if not k.startswith("_") and k != "diffusion_model_pretrained_path"
    }
    small = architecture.get("out_dim") == 48
    block = architecture["num_frame_per_block"]
    horizon = architecture["num_action_per_block"]
    if horizon % (4 * block):
        raise ValueError("Checkpoint's action/video ratio cannot be expressed as integer dataset strides")
    temporal = head.get("num_frames", 33)
    values = {
        "backbone": "wan22_5b" if small else "wan21_14b",
        "train_mode": "full",
        "num_frame_per_block": block,
        "num_action_per_block": horizon,
        "video_stride": horizon // (4 * block),
        "training_chunks": (temporal - 1) // (4 * block),
        "max_cache_chunks": architecture.get("max_chunk_size", 4),
        "max_action_dim": architecture.get("action_dim", head.get("max_action_dim", 32)),
        "max_state_dim": architecture.get("max_state_dim", head.get("max_state_dim", 64)),
        "max_text_length": architecture.get("text_len", 512),
        "dit_config": architecture,
        "image_height": head.get("target_video_height") or (160 if small else 352),
        "image_width": head.get("target_video_width") or (320 if small else 640),
    }
    # Geometry-dependent fields are derived from native config, not hardcoded strings.
    for key in (
        "frame_seqlen",
        "num_action_per_block",
        "num_frame_per_block",
        "max_chunk_size",
        "action_dim",
        "max_state_dim",
        "text_len",
    ):
        values["dit_config"].pop(key, None)
    if embodiment != "so101":
        if metadata is None or experiment is None:
            raise ValueError(
                "Original-embodiment conversion requires experiment_cfg metadata.json and conf.yaml"
            )
        if experiment.get("relative_action_per_horizon", False):
            raise ValueError("Per-horizon normalization is not supported; convert a per-block checkpoint")
        meta = metadata[embodiment]
        modalities = experiment["modality_configs"][embodiment]
        stats = meta["statistics"]
        state_keys = [k.removeprefix("state.") for k in modalities["state"]["modality_keys"]]
        action_keys = [k.removeprefix("action.") for k in modalities["action"]["modality_keys"]]
        cameras = [
            "observation.images." + k.removeprefix("video.") for k in modalities["video"]["modality_keys"]
        ]
        flat = {}
        for modality, names in (("state", state_keys), ("action", action_keys)):
            flat[modality] = {"min": [], "max": []}
            for name in names:
                group = stats[modality][name]
                for dest, source in (("min", "q01"), ("max", "q99")):
                    value = torch.as_tensor(group[source])
                    if value.ndim != 1:
                        raise ValueError("Expected per-channel, not per-horizon, normalization statistics")
                    flat[modality][dest].extend(value.tolist())
        relative = []
        offset = 0
        for name in action_keys:
            size = len(stats["action"][name]["q01"])
            if experiment.get("relative_action") and name in experiment.get("relative_action_keys", []):
                # Current native contract requires matching action/state channel order.
                state_offset = (
                    sum(len(stats["state"][k]["q01"]) for k in state_keys[: state_keys.index(name)])
                    if name in state_keys
                    else -1
                )
                if state_offset != offset:
                    raise ValueError(f"Relative channel mapping differs between state and action: {name}")
                relative.extend(range(offset, offset + size))
            offset += size
        layout = "droid" if embodiment == "oxe_droid" else "grid"
        template = (
            "A multi-view video shows that a robot {task} The video is split into three views: The top view shows the camera view from the robot's wrist, the bottom-left view shows the camera view from the left exterior camera, and the bottom-right view shows the camera view from the right exterior camera. During training, one of the two bottom exterior views may be a black screen (dropped view). The robot {task}"
            if layout == "droid"
            else "A multi-view video shows that a robot {task} The video is split into four views: The top-left view shows the camera view from the robot's head, the top-right view shows the camera view from the right hand, the bottom-left view shows the camera view from the left hand, and the bottom-right view is a black screen (inactive view). The robot {task}"
        )
        values.update(
            camera_keys=cameras,
            camera_layout=layout,
            state_dim=len(flat["state"]["min"]),
            action_dim=len(flat["action"]["min"]),
            state_stats=flat["state"],
            action_stats=flat["action"],
            relative_action_indices=relative,
            prompt_template=template,
            reset_action_heads=False,
            normalization_style="q99",
            lowercase_task=True,
        )
        fps = experiment.get("fps", {})
        if isinstance(fps, dict) and fps.get(embodiment):
            values["fps"] = int(fps[embodiment])
        else:
            video_metadata = meta.get("modalities", {}).get("video", {})
            rates = {view["fps"] for view in video_metadata.values() if "fps" in view}
            if len(rates) > 1:
                raise ValueError("Mixed camera frame rates require explicit dataset resampling")
            if rates:
                values["fps"] = rates.pop()
    values.update(overrides)
    config = DreamZeroConfig(**values)
    config.input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, config.image_height, config.image_width))
        for key in config.camera_keys
    }
    config.input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(config.state_dim,)
    )
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(config.action_dim,))}
    return config


def load_original_adapter(
    model, adapter_path: Path, base_path: Path, *, rank: int, alpha: int, targets: list[str]
):
    """Load an explicit dense DreamZero base, then its adapters AND trained heads."""
    require_package("peft", extra="dreamzero")
    load_checked(model, base_path)
    wrapped = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets))
    report = load_checked(wrapped, adapter_path, adapter_only=True)
    return wrapped.merge_and_unload(), report
