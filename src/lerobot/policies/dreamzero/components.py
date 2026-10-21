# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Frozen Wan components loaded lazily, independently of the DreamZero checkout."""

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download

from lerobot.utils.import_utils import _transformers_available, require_package

from .wan.wan_video_image_encoder import WanImageEncoder
from .wan.wan_video_text_encoder import WanTextEncoder
from .wan.wan_video_vae import WanVideoVAE, WanVideoVAE38

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoTokenizer
else:
    AutoTokenizer = None


def component_file(config, path_attr: str, revision_attr: str, filename: str) -> Path:
    source = getattr(config, path_attr)
    path = Path(source).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        candidate = path / filename
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate.resolve()
    result = Path(hf_hub_download(source, filename, revision=getattr(config, revision_attr)))
    # Persist the resolved immutable revision for subsequent reloads.
    if "snapshots" in result.parts:
        setattr(config, revision_attr, result.parts[result.parts.index("snapshots") + 1])
    return result


def load_frozen(module, path, device, dtype, ignored_prefixes=()):
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {name: value for name, value in state.items() if not name.startswith(ignored_prefixes)}
    module.load_state_dict(state, strict=True)
    module.requires_grad_(False).eval().to(device=device, dtype=dtype)
    return module


class FrozenComponents:
    """Not an nn.Module: frozen weights are references, not copied into each policy checkpoint."""

    def __init__(self, config, device, dtype):
        require_package("transformers", extra="dreamzero")
        self.config = config
        self.device = device
        self.dtype = dtype
        small = config.backbone == "wan22_5b"
        vae_path = component_file(
            config, "wan_path", "wan_revision", "Wan2.2_VAE.pth" if small else "Wan2.1_VAE.pth"
        )
        self.vae = WanVideoVAE38() if small else WanVideoVAE()
        load_frozen(self.vae.model, vae_path, device, dtype)
        text_path = component_file(config, "wan_path", "wan_revision", "models_t5_umt5-xxl-enc-bf16.pth")
        self.text_device = torch.device(config.text_encoder_device or device)
        self.text = load_frozen(WanTextEncoder(), text_path, self.text_device, dtype)
        clip_path = component_file(
            config,
            "image_encoder_path",
            "image_encoder_revision",
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        )
        self.image = WanImageEncoder()
        load_frozen(self.image.model, clip_path, device, dtype, ignored_prefixes=("textual.",))
        # Pin the tokenizer independently; only tokenizer assets are fetched.
        component_file(config, "tokenizer_path", "tokenizer_revision", "tokenizer_config.json")
        tokenizer_source = config.tokenizer_path
        if Path(tokenizer_source).is_dir():
            tokenizer_source = str(Path(tokenizer_source).expanduser().resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=config.tokenizer_revision)

    @torch.no_grad()
    def encode_text(self, tasks: list[str]):
        tokens = self.tokenizer(
            tasks,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt",
        ).to(self.text_device)
        context = self.text(tokens.input_ids, tokens.attention_mask)
        context = context.masked_fill(~tokens.attention_mask.bool().unsqueeze(-1), 0)
        return context.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def encode_video(self, video):
        return self.vae.encode(video.to(device=self.device, dtype=self.dtype))

    @torch.no_grad()
    def image_condition(self, first_frame, num_frames):
        clip = self.image.encode_image(first_frame.transpose(1, 2).clone())
        if self.config.backbone == "wan22_5b":
            return clip, None
        zeros = first_frame.new_zeros(first_frame.shape[0], 3, num_frames - 1, *first_frame.shape[-2:])
        latent = self.encode_video(torch.cat((first_frame, zeros), dim=2))
        mask = latent.new_zeros(latent.shape[0], 4, *latent.shape[2:])
        mask[:, :, :1] = 1
        return clip, torch.cat((mask, latent), dim=1)
