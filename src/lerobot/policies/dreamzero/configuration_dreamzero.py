# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Native DreamZero configuration, including the temporal sampling contract."""

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import ConstantWithWarmupSchedulerConfig

COMPONENT_REVISIONS = {
    "Wan-AI/Wan2.1-I2V-14B-480P": "6b73f84e66371cdfe870c72acd6826e1d61cf279",
    "Wan-AI/Wan2.2-TI2V-5B": "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
    "google/umt5-xxl": "66cb9e7e85526fe440a945569e42c72fb6cbc0ad",
}


@PreTrainedConfig.register_subclass("dreamzero")
@dataclass
class DreamZeroConfig(PreTrainedConfig):
    backbone: str = "wan22_5b"
    train_mode: str = "lora"
    dtype: str = "bfloat16"
    n_obs_steps: int = 1
    camera_keys: list[str] = field(
        default_factory=lambda: ["observation.images.front", "observation.images.side"]
    )
    camera_layout: str = "horizontal"  # horizontal, droid (wrist on top), grid (AgiBot)
    image_height: int | None = None
    image_width: int | None = None
    action_dim: int = 6
    state_dim: int = 6
    max_action_dim: int = 32
    max_state_dim: int = 64
    fps: int = 30
    video_stride: int = 3
    num_frame_per_block: int = 2
    num_action_per_block: int = 24
    training_chunks: int = 4
    max_cache_chunks: int = 4
    n_action_steps: int | None = None
    max_text_length: int = 512
    prompt_template: str = "A multi-view video shows that a robot {task}"
    negative_prompt: str = (
        "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, painting, "
        "image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, ugly, mutilated, "
        "extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, fused fingers, "
        "stagnant image, cluttered background, three legs, many people in the background, walking backwards."
    )
    num_inference_steps: int = 16
    guidance_scale: float = 5.0
    sigma_shift: float = 5.0
    inference_seed: int | None = None
    use_gradient_checkpointing: bool = True
    lora_rank: int = 4
    lora_alpha: int = 4
    lora_targets: list[str] = field(default_factory=lambda: ["q", "k", "v", "o", "ffn.0", "ffn.2"])
    # Base initialization is different from --policy.path (a native resume/inference checkpoint).
    init_from: str | None = None
    init_revision: str | None = None
    init_kind: str = "wan"  # wan or dreamzero
    reset_action_heads: bool = True
    wan_path: str | None = None
    wan_revision: str | None = None
    image_encoder_path: str = "Wan-AI/Wan2.1-I2V-14B-480P"
    image_encoder_revision: str | None = None
    tokenizer_path: str = "google/umt5-xxl"
    tokenizer_revision: str | None = None
    text_encoder_device: str | None = None
    # Architecture overrides also allow real, small CPU networks in offline tests.
    dit_config: dict[str, Any] = field(default_factory=dict)
    # Native SO-101 is absolute. Converted upstream policies may use these channels as deltas.
    relative_action_indices: list[int] = field(default_factory=list)
    normalization_style: str = "min_max"  # q99 preserves upstream clipping/constant-channel behavior
    lowercase_task: bool = False
    action_stats: dict[str, list[float]] | None = None
    state_stats: dict[str, list[float]] | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-5
    scheduler_warmup_steps: int = 1000

    def __post_init__(self):
        super().__post_init__()
        if self.backbone not in ("wan21_14b", "wan22_5b"):
            raise ValueError("backbone must be wan21_14b or wan22_5b")
        if self.train_mode not in ("lora", "full") or self.init_kind not in ("wan", "dreamzero"):
            raise ValueError("Invalid train_mode or init_kind")
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ValueError("dtype must be float32, bfloat16 or float16")
        if self.normalization_style not in ("min_max", "q99"):
            raise ValueError("normalization_style must be min_max or q99")
        if self.normalization_style == "q99" and (self.action_stats is None or self.state_stats is None):
            raise ValueError("q99 requires original action and state statistics")
        if self.n_obs_steps != 1:
            raise ValueError("Streaming DreamZero receives one observation per select_action call")
        for name in (
            "fps",
            "video_stride",
            "num_frame_per_block",
            "num_action_per_block",
            "training_chunks",
            "max_cache_chunks",
            "num_inference_steps",
            "max_text_length",
            "action_dim",
            "state_dim",
            "lora_rank",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_action_per_block != self.video_stride * 4 * self.num_frame_per_block:
            raise ValueError(
                "Each block must align actions with video_stride × VAE temporal stride 4 × latent frames"
            )
        if self.training_chunks > self.max_cache_chunks:
            raise ValueError("training_chunks must not exceed max_cache_chunks")
        if self.action_dim > self.max_action_dim or self.state_dim > self.max_state_dim:
            raise ValueError("State/action dimensions exceed model capacity")
        if self.n_action_steps is None:
            self.n_action_steps = self.num_action_per_block
        if self.n_action_steps != self.num_action_per_block:
            raise ValueError("DreamZero streaming currently executes a complete action block")
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must be nonempty and unique")
        if self.camera_layout not in ("horizontal", "droid", "grid"):
            raise ValueError("Unknown camera_layout")
        if self.camera_layout in ("droid", "grid") and len(self.camera_keys) != 3:
            raise ValueError("droid/grid layouts require exactly three ordered cameras")
        if self.relative_action_indices and (
            min(self.relative_action_indices) < 0
            or max(self.relative_action_indices) >= min(self.action_dim, self.state_dim)
        ):
            raise ValueError("Relative action channels must have corresponding state channels")
        small = self.backbone == "wan22_5b"
        self.image_height = self.image_height or (160 if small else 352)
        self.image_width = self.image_width or (320 if small else 640)
        divisor = 32 if small else 16
        if self.image_height % divisor or self.image_width % divisor:
            raise ValueError(f"Image dimensions must be divisible by {divisor}")
        if self.wan_path is None:
            self.wan_path = "Wan-AI/Wan2.2-TI2V-5B" if small else "Wan-AI/Wan2.1-I2V-14B-480P"
        for source, revision in (
            ("wan_path", "wan_revision"),
            ("image_encoder_path", "image_encoder_revision"),
            ("tokenizer_path", "tokenizer_revision"),
        ):
            if getattr(self, revision) is None:
                setattr(self, revision, COMPONENT_REVISIONS.get(getattr(self, source)))

    @property
    def chunk_size(self) -> int:
        return self.num_action_per_block

    @property
    def num_video_frames(self) -> int:
        return 1 + 4 * self.num_frame_per_block * self.training_chunks

    @property
    def observation_delta_indices(self) -> list[int]:
        return self.image_observation_delta_indices

    @property
    def image_observation_delta_indices(self) -> list[int]:
        return [i * self.video_stride for i in range(self.num_video_frames)]

    @property
    def state_observation_delta_indices(self) -> list[int]:
        return [i * self.chunk_size for i in range(self.training_chunks)]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.training_chunks * self.chunk_size))

    @property
    def drop_n_last_frames(self) -> int:
        return self.image_observation_delta_indices[-1]

    @property
    def reward_delta_indices(self) -> None:
        return None

    def model_kwargs(self) -> dict[str, Any]:
        small = self.backbone == "wan22_5b"
        spatial_stride = 16 if small else 8
        result = {
            "model_type": "ti2v" if small else "i2v",
            "dim": 3072 if small else 5120,
            "in_dim": 48 if small else 36,
            "out_dim": 48 if small else 16,
            "ffn_dim": 14336 if small else 13824,
            "num_heads": 24 if small else 40,
            "num_layers": 30 if small else 40,
            "freq_dim": 256,
            "text_dim": 4096,
            "concat_first_frame_latent": not small,
            "text_len": self.max_text_length,
            "frame_seqlen": (self.image_height // spatial_stride // 2)
            * (self.image_width // spatial_stride // 2),
            "num_frame_per_block": self.num_frame_per_block,
            "num_action_per_block": self.chunk_size,
            "num_state_per_block": 1,
            "action_dim": self.max_action_dim,
            "max_state_dim": self.max_state_dim,
            "max_chunk_size": self.max_cache_chunks,
        }
        result.update(self.dit_config)
        return result

    def validate_features(self) -> None:
        for key in self.camera_keys:
            if key not in self.input_features or self.input_features[key].type != FeatureType.VISUAL:
                raise ValueError(f"DreamZero requires camera {key}")
        for features, key, dim in (
            (self.input_features, "observation.state", self.state_dim),
            (self.output_features, "action", self.action_dim),
        ):
            if key not in features or features[key].shape != (dim,):
                raise ValueError(f"Expected {key} with shape ({dim},)")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> ConstantWithWarmupSchedulerConfig:
        return ConstantWithWarmupSchedulerConfig(num_warmup_steps=self.scheduler_warmup_steps)
