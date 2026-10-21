# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""DreamZero joint video/action flow matching and observed-history streaming."""

from collections import deque
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.import_utils import _diffusers_available, _peft_available, require_package

from .configuration_dreamzero import DreamZeroConfig
from .processor_dreamzero import RAW_ACTION, RAW_STATE

if TYPE_CHECKING or _diffusers_available:
    from .wan.causal_wan import CausalWanModel
    from .wan.flow_match_scheduler import FlowMatchScheduler
    from .wan.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler
else:
    CausalWanModel = FlowMatchScheduler = FlowUniPCMultistepScheduler = None

if TYPE_CHECKING or _peft_available:
    from peft import LoraConfig, get_peft_model
else:
    LoraConfig = get_peft_model = None


def stack_cameras(batch: dict[str, Tensor], config: DreamZeroConfig) -> Tensor:
    """Pack ordered RGB views into [B,3,T,H,W], scaled once to [-1,1]."""
    views = []
    for key in config.camera_keys:
        value = batch[key]
        if value.ndim == 4:
            value = value[:, None]
        if value.ndim != 5 or value.shape[2] != 3:
            raise ValueError(f"{key} must have shape [B,T,3,H,W] or [B,3,H,W]")
        value = value.float() / 255 if value.dtype == torch.uint8 else value.float()
        if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError(f"{key} must contain RGB pixels in [0,1] or uint8")
        views.append(value)
    b, t = views[0].shape[:2]
    if any(v.shape[:2] != (b, t) for v in views):
        raise ValueError("Camera batch sizes and temporal windows must agree")
    h, w = config.image_height, config.image_width

    def resize(value, height, width):
        return F.interpolate(
            value.flatten(0, 1), size=(height, width), mode="bilinear", align_corners=False
        ).view(b, t, 3, height, width)

    if config.camera_layout == "horizontal":
        edges = [i * w // len(views) for i in range(len(views) + 1)]
        video = torch.cat([resize(v, h, edges[i + 1] - edges[i]) for i, v in enumerate(views)], dim=-1)
    else:
        video = views[0].new_zeros(b, t, 3, h, w)
        if config.camera_layout == "droid":
            video[..., : h // 2, :] = resize(views[2], h // 2, w)
            video[..., h // 2 :, : w // 2] = resize(views[0], h // 2, w // 2)
            video[..., h // 2 :, w // 2 :] = resize(views[1], h // 2, w // 2)
        else:
            video[..., : h // 2, : w // 2] = resize(views[0], h // 2, w // 2)
            video[..., h // 2 :, : w // 2] = resize(views[1], h // 2, w // 2)
            video[..., : h // 2, w // 2 :] = resize(views[2], h // 2, w // 2)
    return video.permute(0, 2, 1, 3, 4).contiguous().mul(2).sub(1)


class DreamZeroPolicy(PreTrainedPolicy):
    config_class = DreamZeroConfig
    name = "dreamzero"
    _fsdp_wrap_modules = ["CausalWanAttentionBlock"]
    supports_gradient_checkpointing = True

    def __init__(self, config: DreamZeroConfig, **kwargs):
        require_package("diffusers", extra="dreamzero")
        super().__init__(config)
        config.validate_features()
        self.model = CausalWanModel(**config.model_kwargs())
        self.model.gradient_checkpointing = config.use_gradient_checkpointing
        self._components = None
        # Initialize dense base weights before inserting adapters. Native reloads already
        # contain the dense weights AND adapters, so must not download an initialization.
        if config.init_from and not config.pretrained_path and not kwargs.get("_skip_base_loading", False):
            from .checkpoint import initialize_model

            initialize_model(self.model, config)
        if config.train_mode == "lora":
            require_package("peft", extra="dreamzero")
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    target_modules=config.lora_targets,
                    init_lora_weights=True,
                ),
            )
            for name, param in self.model.named_parameters():
                if any(
                    part in name.split(".")
                    for part in ("state_encoder", "action_encoder", "action_decoder", "img_emb")
                ):
                    param.requires_grad_(True)
                # TI2V lacks the I2V-only cross-attention image projections.
                if config.backbone == "wan22_5b" and any(
                    part in name.split(".") for part in ("k_img", "v_img", "norm_k_img")
                ):
                    param.requires_grad_(True)
        self.model.to(dtype=getattr(torch, config.dtype))
        self.scheduler = FlowMatchScheduler(shift=config.sigma_shift, sigma_min=0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.reset()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs["_skip_base_loading"] = True
        kwargs.setdefault("strict", True)
        return super().from_pretrained(*args, **kwargs)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def model_dtype(self):
        return next(self.model.parameters()).dtype

    def _frozen(self):
        if self._components is None:
            from .components import FrozenComponents

            self._components = FrozenComponents(self.config, self.device, self.model_dtype)
        if torch.device(self._components.device) != self.device or self._components.dtype != self.model_dtype:
            raise RuntimeError(
                "Device/dtype changed after frozen components loaded; reload the policy on its target device"
            )
        return self._components

    def get_optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def reset(self):
        self._action_queue = deque()
        self._history = {key: deque(maxlen=self.config.chunk_size + 1) for key in self.config.camera_keys}
        self._task = None
        self._clear_cache()

    def _clear_cache(self):
        self._kv = None
        self._contexts = None
        self._clip = None
        self._condition = None
        self._start_frame = 0
        self.last_predicted_latents = None

    def drop_queued_actions(self):
        # Interrupted blocks do not correspond to a complete observed latent block.
        self.reset()

    def _tasks(self, batch, batch_size):
        tasks = batch.get("task")
        if isinstance(tasks, str):
            tasks = [tasks]
        if (
            tasks is None
            or len(tasks) != batch_size
            or any(not isinstance(t, str) or not t.strip() for t in tasks)
        ):
            raise ValueError("DreamZero requires one nonempty task string per batch item")
        return [t.lower() if self.config.lowercase_task else t for t in tasks]

    @staticmethod
    def _q99_normalize(value, stats):
        if stats is None:
            raise ValueError("Converted DreamZero requires saved normalization statistics")
        lower = torch.as_tensor(stats["min"], device=value.device, dtype=value.dtype)
        upper = torch.as_tensor(stats["max"], device=value.device, dtype=value.dtype)
        width = upper - lower
        return torch.where(width != 0, 2 * (value - lower) / width.clamp_min(1e-8) - 1, value).clamp(-1, 1)

    def _state(self, batch):
        if self.config.normalization_style == "q99":
            return self._q99_normalize(batch[RAW_STATE].float(), self.config.state_stats).to(
                device=self.device, dtype=self.model_dtype
            )
        return batch["observation.state"].to(device=self.device, dtype=self.model_dtype)

    def _relative_offset(self, batch, action, training):
        ids = self.config.relative_action_indices
        if not ids:
            return torch.zeros_like(action)
        if RAW_STATE not in batch or self.config.action_stats is None:
            raise ValueError(
                "Use the DreamZero preprocessor to supply raw states and relative-action statistics"
            )
        state = batch[RAW_STATE].to(action)
        if state.ndim == 2:
            state = state[:, None]
        state = state.repeat_interleave(self.config.chunk_size, dim=1) if training else state[:, -1:]
        stats = self.config.action_stats
        width = (
            torch.as_tensor(stats["max"], device=action.device)
            - torch.as_tensor(stats["min"], device=action.device)
        ).clamp_min(1e-8)
        offset = torch.zeros_like(action)
        offset[..., ids] = state[..., ids] * (2 / width[ids])
        return offset

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        cfg = self.config
        for key, value in batch.items():
            if key.endswith("_is_pad") and torch.as_tensor(value).any():
                raise ValueError(
                    "DreamZero requires complete temporal windows; configure the episode-aware sampler"
                )
        video = stack_cameras(batch, cfg).to(device=self.device, dtype=self.model_dtype)
        if video.shape[2] != cfg.num_video_frames:
            raise ValueError(f"Expected {cfg.num_video_frames} video frames, got {video.shape[2]}")
        action = batch["action"].to(device=self.device, dtype=self.model_dtype)
        state = self._state(batch)
        expected = (video.shape[0], cfg.training_chunks * cfg.chunk_size, cfg.action_dim)
        if action.shape != expected or state.shape != (video.shape[0], cfg.training_chunks, cfg.state_dim):
            raise ValueError("Action/state windows do not align with the configured video blocks")
        if cfg.normalization_style == "q99":
            raw_action = batch[RAW_ACTION].float().clone()
            ids = cfg.relative_action_indices
            raw_action[..., ids] -= batch[RAW_STATE].float().repeat_interleave(cfg.chunk_size, 1)[..., ids]
            action = self._q99_normalize(raw_action, cfg.action_stats).to(action)
        else:
            action = action - self._relative_offset(batch, action, training=True)
        action = F.pad(action, (0, cfg.max_action_dim - cfg.action_dim))
        state = F.pad(state, (0, cfg.max_state_dim - cfg.state_dim))
        frozen = self._frozen()
        with torch.no_grad():
            latents = frozen.encode_video(video)
            context = frozen.encode_text(
                [cfg.prompt_template.format(task=t) for t in self._tasks(batch, video.shape[0])]
            )
            clip, condition = frozen.image_condition(video[:, :, :1], cfg.num_video_frames)
        return self._training_loss(latents, action, state, context, clip, condition)

    def _training_loss(self, latents, action, state, context, clip, condition):
        cfg = self.config
        b, _, frames, h, w = latents.shape
        if frames != 1 + cfg.training_chunks * cfg.num_frame_per_block:
            raise ValueError("VAE temporal compression does not match DreamZero block layout")
        ids = torch.randint(1000, (b, cfg.training_chunks + 1))
        video_ids = torch.cat((ids[:, :1], ids[:, 1:].repeat_interleave(cfg.num_frame_per_block, 1)), 1)
        action_ids = ids[:, 1:].repeat_interleave(cfg.chunk_size, 1)
        times = self.scheduler.timesteps[video_ids].to(latents.device)
        action_times = self.scheduler.timesteps[action_ids].to(latents.device)
        clean = latents.transpose(1, 2)
        video_noise, action_noise = torch.randn_like(clean), torch.randn_like(action)
        noisy_video = (
            self.scheduler.add_noise(clean.flatten(0, 1), video_noise.flatten(0, 1), times.flatten())
            .unflatten(0, (b, frames))
            .transpose(1, 2)
        )
        noisy_action = self.scheduler.add_noise(
            action.flatten(0, 1), action_noise.flatten(0, 1), action_times.flatten()
        ).reshape_as(action)
        video_pred, action_pred = self.model(
            noisy_video,
            timestep=times,
            timestep_action=action_times,
            context=context,
            seq_len=frames * (h // 2) * (w // 2),
            clean_x=latents,
            y=condition,
            clip_feature=clip,
            action=noisy_action,
            state=state,
            embodiment_id=torch.zeros(b, dtype=torch.long, device=latents.device),
        )
        video_error = (
            (video_pred.float() - (video_noise - clean).transpose(1, 2).float()).square().mean((1, 3, 4))
        )
        video_weight = self.scheduler.training_weight(times.flatten()).to(latents.device).reshape(b, frames)
        loss_video = (video_error * video_weight).mean()
        action_error = (action_pred.float() - (action_noise - action).float()).square()
        # Preserve upstream's padded-width reduction while excluding padded channels.
        mask = torch.arange(cfg.max_action_dim, device=action.device) < cfg.action_dim
        action_weight = (
            self.scheduler.training_weight(action_times.flatten()).to(action.device).reshape(action.shape[:2])
        )
        loss_action = ((action_error * mask).mean(-1) * action_weight).mean()
        return loss_video + loss_action, {
            "loss_video": loss_video.detach().item(),
            "loss_action": loss_action.detach().item(),
        }

    def _model_pass(self, latent, times, action, action_times, state, context, cache, start, condition):
        return self.model(
            latent,
            timestep=times,
            timestep_action=action_times,
            action=action,
            state=state,
            embodiment_id=torch.zeros(latent.shape[0], dtype=torch.long, device=latent.device),
            context=context,
            clip_feature=self._clip,
            y=condition,
            seq_len=latent.shape[2] * (latent.shape[3] // 2) * (latent.shape[4] // 2),
            kv_cache=cache,
            crossattn_cache=[],
            current_start_frame=start,
        )

    def _condition_slice(self, start, count):
        if self._condition is None:
            return None
        if start + count <= self._condition.shape[2]:
            return self._condition[:, :, start : start + count]
        return self._condition[:, :, -count:]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        cfg = self.config
        if self._action_queue:
            # A direct chunk call interrupts a partially consumed select_action block.
            self.reset()
        video = stack_cameras(batch, cfg).to(device=self.device, dtype=self.model_dtype)
        if video.shape[0] != 1:
            raise ValueError("DreamZero streaming inference supports one environment per policy instance")
        task = self._tasks(batch, 1)[0]
        observed_frames = 1 + 4 * cfg.num_frame_per_block
        if (
            task != self._task
            or video.shape[2] != observed_frames
            or self._start_frame >= 1 + cfg.max_cache_chunks * cfg.num_frame_per_block
        ):
            self._clear_cache()
        self._task = task
        state = self._state(batch)
        if state.ndim == 2:
            state = state[:, None]
        if state.shape[-1] != cfg.state_dim:
            raise ValueError("State dimension mismatch")
        state = F.pad(state[:, -1:], (0, cfg.max_state_dim - cfg.state_dim))
        frozen = self._frozen()
        if self._start_frame == 0:
            video = video[:, :, -1:]
            self._contexts = [frozen.encode_text([cfg.prompt_template.format(task=task)])]
            if cfg.guidance_scale != 1:
                self._contexts.append(frozen.encode_text([cfg.negative_prompt]))
            self._clip, self._condition = frozen.image_condition(video, cfg.num_video_frames)
            reference = frozen.encode_video(video)
            model_cfg = cfg.model_kwargs()
            self._kv = [
                [
                    reference.new_empty(
                        2, 1, 0, model_cfg["num_heads"], model_cfg["dim"] // model_cfg["num_heads"]
                    )
                    for _ in range(model_cfg["num_layers"])
                ]
                for _ in self._contexts
            ]
            start = 0
        else:
            # Only real observations enter the persistent cache. Predictions never do.
            reference = frozen.encode_video(video)[:, :, -cfg.num_frame_per_block :]
            start = self._start_frame - cfg.num_frame_per_block
        for i, context in enumerate(self._contexts):
            _, _, self._kv[i] = self._model_pass(
                reference,
                reference.new_zeros(1, reference.shape[2]),
                None,
                None,
                None,
                context,
                self._kv[i],
                start,
                self._condition_slice(start, reference.shape[2]),
            )
        if self._start_frame == 0:
            self._start_frame = 1
        generator = None
        if cfg.inference_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(cfg.inference_seed)
        latent = torch.randn(
            (1, reference.shape[1], cfg.num_frame_per_block, *reference.shape[-2:]),
            device=self.device,
            dtype=self.model_dtype,
            generator=generator,
        )
        action = torch.randn(
            (1, cfg.chunk_size, cfg.max_action_dim),
            device=self.device,
            dtype=self.model_dtype,
            generator=generator,
        )
        schedulers = [
            FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
            for _ in range(2)
        ]
        for scheduler in schedulers:
            scheduler.set_timesteps(cfg.num_inference_steps, device=self.device, shift=cfg.sigma_shift)
        for index, timestep in enumerate(schedulers[0].timesteps):
            predictions = [
                self._model_pass(
                    latent,
                    timestep.expand(1, cfg.num_frame_per_block),
                    action,
                    timestep.expand(1, cfg.chunk_size),
                    state,
                    context,
                    self._kv[i],
                    self._start_frame,
                    self._condition_slice(self._start_frame, cfg.num_frame_per_block),
                )
                for i, context in enumerate(self._contexts)
            ]
            video_flow, action_flow = predictions[0][:2]
            if len(predictions) == 2:
                video_flow = predictions[1][0] + cfg.guidance_scale * (video_flow - predictions[1][0])
            # Upstream guides video only; action uses the conditional flow.
            latent = schedulers[0].step(video_flow, timestep, latent, step_index=index, return_dict=False)[0]
            action = schedulers[1].step(action_flow, timestep, action, step_index=index, return_dict=False)[0]
        self._start_frame += cfg.num_frame_per_block
        self.last_predicted_latents = latent
        action = action[..., : cfg.action_dim].float()
        if cfg.normalization_style == "q99":
            lower = torch.as_tensor(cfg.action_stats["min"], device=action.device, dtype=action.dtype)
            width = torch.as_tensor(cfg.action_stats["max"], device=action.device, dtype=action.dtype) - lower
            absolute = (action + 1) * width / 2 + lower
            raw_state = batch[RAW_STATE].to(action)
            if raw_state.ndim == 2:
                raw_state = raw_state[:, None]
            absolute[..., cfg.relative_action_indices] += raw_state[:, -1:, cfg.relative_action_indices]
            # Re-encode absolute actions for the standard LeRobot postprocessor.
            return 2 * (absolute - lower) / torch.where(width == 0, 1e-8, width) - 1
        return action + self._relative_offset(batch, action, training=False)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        task = self._tasks(batch, 1)[0]
        if task != self._task:
            self.reset()
            self._task = task
        for key in self.config.camera_keys:
            frame = batch[key]
            if frame.ndim == 5:
                if frame.shape[1] != 1:
                    raise ValueError("select_action expects one current frame per camera")
                frame = frame[:, 0]
            self._history[key].append(frame.detach().clone())
        if not self._action_queue:
            inputs = dict(batch)
            for key, frames in self._history.items():
                history = list(frames)
                if len(history) == self.config.chunk_size + 1:
                    inputs[key] = torch.stack(history[:: self.config.video_stride], dim=1)
                else:
                    inputs[key] = history[-1]
            actions = self.predict_action_chunk(inputs, **kwargs)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
