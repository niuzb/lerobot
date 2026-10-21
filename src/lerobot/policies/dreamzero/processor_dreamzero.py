# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Serializable DreamZero normalization and unnormalized block anchors."""

from dataclasses import dataclass

import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.factory import make_default_policy_processor_steps, make_policy_processor_pipelines

from .configuration_dreamzero import DreamZeroConfig

RAW_STATE = "observation.dreamzero_raw_state"
RAW_ACTION = "observation.dreamzero_raw_action"


@ProcessorStepRegistry.register("dreamzero_raw_state")
@dataclass
class DreamZeroRawStateStep(ProcessorStep):
    def __call__(self, transition):
        result = transition.copy()
        observation = dict(result.get(TransitionKey.OBSERVATION) or {})
        if "observation.state" in observation:
            observation[RAW_STATE] = observation["observation.state"].clone()
        if result.get(TransitionKey.ACTION) is not None:
            observation[RAW_ACTION] = result[TransitionKey.ACTION].clone()
        result[TransitionKey.OBSERVATION] = observation
        return result

    def transform_features(self, features):
        return features


def make_dreamzero_pre_post_processors(config: DreamZeroConfig, dataset_stats=None, dataset_meta=None):
    if dataset_meta is not None and dataset_meta.fps != config.fps:
        raise ValueError(f"DreamZero preset expects {config.fps} FPS, dataset has {dataset_meta.fps}")
    stats = dict(dataset_stats or {})
    for key, attr in (("action", "action_stats"), ("observation.state", "state_stats")):
        saved = getattr(config, attr)
        if saved is not None:
            # Converted policies carry their original statistics; never replace relative
            # statistics with absolute-action dataset statistics on an inference reload.
            stats[key] = {name: torch.as_tensor(value) for name, value in saved.items()}
        elif key in stats:
            setattr(
                config, attr, {name: torch.as_tensor(stats[key][name]).tolist() for name in ("min", "max")}
            )
    if config.relative_action_indices and not config.action_stats:
        raise ValueError("Relative DreamZero requires block-relative action statistics")
    steps = make_default_policy_processor_steps(config, stats, normalizer_device=config.device)
    return make_policy_processor_pipelines(
        [
            steps.rename_observations,
            steps.add_batch_dim,
            steps.to_device,
            DreamZeroRawStateStep(),
            steps.normalize,
        ],
        [steps.unnormalize, steps.to_cpu],
    )


def make_dreamzero_pre_post_processors_from_pretrained(
    config,
    pretrained_path,
    *,
    revision,
    dataset_stats,
    dataset_meta,
    preprocessor_overrides,
    postprocessor_overrides,
    preprocessor_config_filename,
    postprocessor_config_filename,
):
    # Absolute SO-101 checkpoints use the standard serialized pipeline path.
    if config.normalization_style != "q99":
        return None
    # Generic fine-tuning overrides contain absolute dataset statistics. A converted
    # relative policy must retain its original block-relative q01/q99 statistics.
    pre, post = make_dreamzero_pre_post_processors(config, dataset_meta=dataset_meta)
    overrides = preprocessor_overrides or {}
    pre.steps[0].rename_map = overrides.get("rename_observations_processor", {}).get("rename_map", {})
    if device := overrides.get("device_processor", {}).get("device"):
        pre.steps[2].device = device
    return pre, post
