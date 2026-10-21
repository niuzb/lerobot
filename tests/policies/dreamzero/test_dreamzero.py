"""Offline DreamZero contracts; frozen encoders are tiny stand-ins, DiT is real."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F  # noqa: N812

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.dreamzero.configuration_dreamzero import DreamZeroConfig

pytest.importorskip("diffusers")
pytest.importorskip("peft")

from lerobot.policies.dreamzero.checkpoint import (
    config_from_original,
    load_checked,
    load_original_adapter,
)
from lerobot.policies.dreamzero.modeling_dreamzero import DreamZeroPolicy, stack_cameras
from lerobot.policies.dreamzero.processor_dreamzero import RAW_STATE
from lerobot.policies.dreamzero.wan.attention import flash_attention
from lerobot.policies.dreamzero.wan.causal_wan import CausalWanModel


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**kwargs):
    options = {
        "device": "cpu",
        "dtype": "float32",
        "train_mode": "full",
        "image_height": 32,
        "image_width": 64,
        "training_chunks": 2,
        "max_cache_chunks": 3,
        "num_frame_per_block": 1,
        "num_action_per_block": 4,
        "video_stride": 1,
        "max_text_length": 8,
        "max_action_dim": 8,
        "max_state_dim": 8,
        "inference_seed": 7,
        "num_inference_steps": 2,
        "guidance_scale": 1,
        "use_gradient_checkpointing": False,
        "dit_config": {
            "dim": 32,
            "ffn_dim": 64,
            "num_heads": 2,
            "num_layers": 2,
            "freq_dim": 16,
            "text_dim": 16,
            "hidden_size": 16,
        },
    }
    options.update(kwargs)
    cfg = DreamZeroConfig(**options)
    cfg.input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)) for key in cfg.camera_keys
    }
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    return cfg


class TinyFrozen:
    def __init__(self, cfg):
        self.config = cfg
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.seen_videos = []

    def encode_video(self, video):
        self.seen_videos.append(video.clone())
        small = self.config.backbone == "wan22_5b"
        channels, stride = (48, 16) if small else (16, 8)
        video = F.avg_pool3d(video[:, :, ::4], kernel_size=(1, stride, stride))
        return video.mean(1, keepdim=True).repeat(1, channels, 1, 1, 1)

    def encode_text(self, tasks):
        values = torch.tensor([sum(map(ord, t)) % 10 for t in tasks], dtype=torch.float32)
        return values[:, None, None].expand(-1, self.config.max_text_length, 16) / 10

    def image_condition(self, image, frames):
        clip = torch.zeros(image.shape[0], 257, 1280)
        if self.config.backbone == "wan22_5b":
            return clip, None
        h, w = image.shape[-2:]
        return clip, torch.zeros(image.shape[0], 20, (frames - 1) // 4 + 1, h // 8, w // 8)


def policy_for(cfg):
    policy = DreamZeroPolicy(cfg)
    policy._components = TinyFrozen(cfg)
    return policy


def batch_for(cfg, training=True, value=0.4):
    batch = {
        key: torch.full((1, cfg.num_video_frames, 3, 32, 32) if training else (1, 3, 32, 32), value)
        for key in cfg.camera_keys
    }
    batch["observation.state"] = torch.zeros(1, cfg.training_chunks, 6) if training else torch.zeros(1, 6)
    batch["task"] = ["pick up the block"]
    if training:
        batch["action"] = torch.zeros(1, cfg.chunk_size * cfg.training_chunks, 6)
    return batch


def stats():
    return {
        key: {"min": torch.tensor([-90.0] * 5 + [0.0]), "max": torch.tensor([90.0] * 5 + [100.0])}
        for key in ("action", "observation.state")
    }


def test_factory_defaults_and_temporal_contract(tmp_path):
    cfg = make_policy_config("dreamzero", device="cpu")
    assert get_policy_class("dreamzero") is DreamZeroPolicy
    assert cfg.num_video_frames == 33
    assert cfg.action_delta_indices == list(range(96))
    assert cfg.state_observation_delta_indices == [0, 24, 48, 72]
    assert cfg.image_observation_delta_indices == list(range(0, 97, 3))
    assert cfg.drop_n_last_frames == 96
    cfg.save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert loaded == cfg
    metadata = SimpleNamespace(
        fps=30, features={key: {} for key in [*cfg.camera_keys, "observation.state", "action"]}
    )
    deltas = resolve_delta_timestamps(cfg, metadata)
    assert deltas["action"][-1] == 95 / 30
    assert deltas["observation.state"] == [0.0, 0.8, 1.6, 2.4]
    assert deltas[cfg.camera_keys[0]][-1] == 3.2
    sampler = EpisodeAwareSampler([0, 100], [100, 200], drop_n_last_frames=96, shuffle=False)
    assert list(sampler) == [0, 1, 2, 3, 100, 101, 102, 103]


@pytest.mark.parametrize(
    "args",
    [
        {"backbone": "bad"},
        {"video_stride": 2},
        {"n_action_steps": 3},
        {"image_height": 35},
        {"action_dim": 33},
        {"camera_keys": []},
    ],
)
def test_invalid_config(args):
    with pytest.raises(ValueError):
        DreamZeroConfig(device="cpu", **args)


def test_camera_order_and_scale():
    cfg = tiny_config()
    batch = batch_for(cfg, training=False, value=0.0)
    batch[cfg.camera_keys[1]].fill_(1)
    packed = stack_cameras(batch, cfg)
    assert packed.shape == (1, 3, 1, 32, 64)
    assert (packed[..., :32] == -1).all()
    assert (packed[..., 32:] == 1).all()
    batch[cfg.camera_keys[0]] = torch.full((1, 3, 32, 32), 255, dtype=torch.uint8)
    assert (stack_cameras(batch, cfg) == 1).all()
    batch[cfg.camera_keys[0]] = torch.full((1, 3, 32, 32), 256.0)
    with pytest.raises(ValueError, match="pixels"):
        stack_cameras(batch, cfg)


def test_processors_roundtrip_and_serialization(tmp_path):
    cfg = tiny_config()
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats())
    batch = batch_for(cfg)
    batch["action"].fill_(20)
    batch["observation.state"].fill_(30)
    processed = pre(batch)
    torch.testing.assert_close(post(processed["action"]), batch["action"])
    torch.testing.assert_close(processed[RAW_STATE], batch["observation.state"])
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    re_pre, re_post = make_pre_post_processors(cfg, pretrained_path=str(tmp_path))
    torch.testing.assert_close(re_post(re_pre(batch)["action"]), batch["action"])


@pytest.mark.parametrize("backbone", ["wan21_14b", "wan22_5b"])
@pytest.mark.parametrize("checkpointing", [True, False])
def test_real_core_forward_backward_and_optimizer(backbone, checkpointing):
    cfg = tiny_config(backbone=backbone, use_gradient_checkpointing=checkpointing)
    policy = policy_for(cfg)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    before = policy.model.action_decoder.layer2.W.detach().clone()
    loss, metrics = policy(batch_for(cfg))
    assert torch.isfinite(loss) and metrics["loss_action"] > 0 and metrics["loss_video"] > 0
    loss.backward()
    assert policy.model.action_decoder.layer2.W.grad.abs().sum() > 0
    assert policy.model.head.head.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, policy.model.action_decoder.layer2.W)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_lora_updates_adapters_and_new_heads(checkpointing):
    cfg = tiny_config(train_mode="lora", use_gradient_checkpointing=checkpointing)
    policy = policy_for(cfg)
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    assert any("lora_A" in n for n in trainable)
    assert any("action_decoder" in n for n in trainable)
    assert not any(n.endswith("patch_embedding.weight") for n in trainable)
    assert any("k_img" in n for n in trainable)
    loss, _ = policy(batch_for(cfg))
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for n, p in policy.named_parameters() if "lora_B" in n
    )


def test_teacher_forcing_has_no_future_leak():
    cfg = tiny_config()
    model = CausalWanModel(**cfg.model_kwargs()).eval()
    model.gradient_checkpointing = False
    torch.nn.init.normal_(model.head.head.weight, std=0.02)
    inputs = {
        "x": torch.randn(1, 48, 3, 2, 4),
        "clean_x": torch.randn(1, 48, 3, 2, 4),
        "timestep": torch.full((1, 3), 500.0),
        "timestep_action": torch.full((1, 8), 500.0),
        "action": torch.randn(1, 8, 8),
        "state": torch.randn(1, 2, 8),
        "context": torch.randn(1, 8, 16),
        "clip_feature": torch.randn(1, 257, 1280),
        "seq_len": 6,
        "embodiment_id": torch.zeros(1, dtype=torch.long),
    }
    with torch.no_grad():
        video, action = model(**inputs)
        changed = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        changed["x"][:, :, 2:] += 100
        changed["clean_x"][:, :, 1:] += 100  # Same-block clean future is also forbidden.
        changed["action"][:, 4:] += 100
        changed["state"][:, 1:] += 100
        next_video, next_action = model(**changed)
    torch.testing.assert_close(action[:, :4], next_action[:, :4], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(video[:, :, :2], next_video[:, :, :2], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(action[:, 4:], next_action[:, 4:])


def test_attention_respects_lengths():
    q, k, v = (torch.randn(2, 4, 2, 8) for _ in range(3))
    out = flash_attention(q, k, v, k_lens=torch.tensor([2, 4]))
    k[0, 2:] += 1000
    v[0, 2:] += 1000
    torch.testing.assert_close(out[0], flash_attention(q, k, v, k_lens=torch.tensor([2, 4]))[0])


def test_clip_flattened_attention_matches_multihead_reference():
    q, k, v = (torch.randn(2, 5, 16) for _ in range(3))
    reference = (
        F.scaled_dot_product_attention(
            q.unflatten(-1, (2, 8)).transpose(1, 2),
            k.unflatten(-1, (2, 8)).transpose(1, 2),
            v.unflatten(-1, (2, 8)).transpose(1, 2),
        )
        .transpose(1, 2)
        .flatten(2)
    )
    torch.testing.assert_close(flash_attention(q, k, v, num_heads=2, compatibility_mode=True), reference)


@pytest.mark.parametrize("backbone", ["wan21_14b", "wan22_5b"])
def test_streaming_cache_real_observations_and_resets(backbone):
    cfg = tiny_config(backbone=backbone, guidance_scale=2)
    policy = policy_for(cfg)
    initial = policy.select_action(batch_for(cfg, False, 0.1))
    assert initial.shape == (1, 6) and torch.isfinite(initial).all()
    initial_cache = policy._kv[0][0].clone()
    for i in range(1, cfg.chunk_size):
        policy.select_action(batch_for(cfg, False, 0.1 + i * 0.1))
    torch.testing.assert_close(policy._kv[0][0], initial_cache)
    policy.select_action(batch_for(cfg, False, 0.5))
    assert policy._start_frame == 3
    assert policy._kv[0][0].shape[2] > initial_cache.shape[2]
    observed = policy._components.seen_videos[-1]
    assert observed.shape[2] == 5
    torch.testing.assert_close(observed[0, 0, :, 0, 0], torch.tensor([-0.8, -0.6, -0.4, -0.2, 0.0]))
    changed = batch_for(cfg, False)
    changed["task"] = ["place the block"]
    policy.select_action(changed)
    assert policy._start_frame == 2 and len(policy._history[cfg.camera_keys[0]]) == 1
    policy.drop_queued_actions()
    assert policy._kv is None and not policy._action_queue


@pytest.mark.parametrize("train_mode", ["full", "lora"])
def test_save_reload_without_base_or_encoders(tmp_path, train_mode):
    cfg = tiny_config(train_mode=train_mode)
    policy = policy_for(cfg)
    expected = policy.predict_action_chunk(batch_for(cfg, False))
    policy.config.init_from = "/does/not/exist"
    policy.save_pretrained(tmp_path)
    loaded = DreamZeroPolicy.from_pretrained(tmp_path)
    assert loaded._components is None
    loaded._components = TinyFrozen(cfg)
    actual = loaded.predict_action_chunk(batch_for(cfg, False))
    torch.testing.assert_close(expected, actual)
    assert all(not k.startswith("_components") for k in loaded.state_dict())


def test_padding_and_temporal_mismatch_rejected():
    cfg = tiny_config()
    policy = policy_for(cfg)
    batch = batch_for(cfg)
    batch["action_is_pad"] = torch.tensor([[True]])
    with pytest.raises(ValueError, match="complete temporal"):
        policy(batch)
    del batch["action_is_pad"]
    batch["action"] = batch["action"][:, :-1]
    with pytest.raises(ValueError, match="align"):
        policy(batch)


def save_original(model, directory, shards=False):
    directory.mkdir()
    weights = {"action_head.model." + k: v.clone() for k, v in model.state_dict().items()}
    if not shards:
        save_file(weights, directory / "model.safetensors")
        return
    mapping = {}
    for number in range(2):
        name = f"model-{number}.safetensors"
        shard = {k: v for i, (k, v) in enumerate(weights.items()) if i % 2 == number}
        save_file(shard, directory / name)
        mapping.update(dict.fromkeys(shard, name))
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))


@pytest.mark.parametrize("shards", [True, False])
def test_audited_weight_loading(tmp_path, shards):
    cfg = tiny_config()
    source = CausalWanModel(**cfg.model_kwargs())
    target = CausalWanModel(**cfg.model_kwargs())
    save_original(source, tmp_path / "original", shards)
    report = load_checked(target, tmp_path / "original")
    assert report["missing"] == []
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key])
    incompatible = tiny_config(dit_config={**cfg.dit_config, "ffn_dim": 80})
    with pytest.raises(ValueError, match="Shape mismatch"):
        load_checked(CausalWanModel(**incompatible.model_kwargs()), tmp_path / "original")


def test_missing_and_duplicate_weights_rejected(tmp_path):
    cfg = tiny_config()
    model = CausalWanModel(**cfg.model_kwargs())
    path = tmp_path / "broken"
    path.mkdir()
    key, value = next(iter(model.state_dict().items()))
    save_file({key: value.clone()}, path / "model.safetensors")
    with pytest.raises(ValueError, match="missing required"):
        load_checked(model, path)
    save_file({key: value.clone(), "action_head.model." + key: value.clone()}, path / "model.safetensors")
    with pytest.raises(ValueError, match="Duplicate"):
        load_checked(model, path)


@pytest.mark.parametrize("small,horizon", [(False, 24), (False, 48), (True, 24)])
def test_original_config_preserves_architecture_and_horizon(small, horizon):
    original = {
        "action_head_cfg": {
            "config": {
                "num_frames": 33,
                "max_action_dim": 32,
                "max_state_dim": 64,
                "diffusion_model_cfg": {
                    "out_dim": 48 if small else 16,
                    "in_dim": 48 if small else 36,
                    "dim": 3072 if small else 5120,
                    "num_heads": 24 if small else 40,
                    "num_layers": 30 if small else 40,
                    "num_frame_per_block": 2,
                    "num_action_per_block": horizon,
                    "max_chunk_size": 4,
                },
            }
        }
    }
    cfg = config_from_original(original, device="cpu")
    assert cfg.chunk_size == horizon and cfg.video_stride == horizon // 8
    assert cfg.model_kwargs()["num_layers"] == (30 if small else 40)
    assert cfg.reset_action_heads


def test_lora_import_requires_base_and_trained_heads(tmp_path):
    from peft import LoraConfig, get_peft_model

    cfg = tiny_config()
    model = CausalWanModel(**cfg.model_kwargs())
    save_original(model, tmp_path / "base")
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=["q", "v"]))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.1)
    adapter = {
        "action_head.model." + k: v.clone()
        for k, v in model.state_dict().items()
        if "lora_" in k or any(h in k for h in ("state_encoder", "action_encoder", "action_decoder"))
    }
    (tmp_path / "adapter").mkdir()
    save_file(adapter, tmp_path / "adapter/model.safetensors")
    imported, report = load_original_adapter(
        CausalWanModel(**cfg.model_kwargs()),
        tmp_path / "adapter",
        tmp_path / "base",
        rank=2,
        alpha=2,
        targets=["q", "v"],
    )
    assert report["loaded"] > 0
    expected = model.merge_and_unload()
    for key, value in imported.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key])


def test_relative_actions_use_block_anchor_not_later_observation():
    cfg = tiny_config(relative_action_indices=list(range(5)))
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats())
    policy = policy_for(cfg)
    batch = batch_for(cfg, False)
    batch["observation.state"][:, :5] = 45
    processed = pre(batch)
    relative = torch.zeros(1, cfg.chunk_size, 6)
    absolute_normalized = relative + policy._relative_offset(processed, relative, training=False)
    result = post(absolute_normalized)
    torch.testing.assert_close(result[0, 0], torch.tensor([45.0] * 5 + [50.0]))


@pytest.mark.parametrize("small", [False, True])
def test_actual_small_vae_temporal_and_spatial_compression(small):
    from lerobot.policies.dreamzero.wan.wan_video_vae import VideoVAE38_, VideoVAE_

    options = {"dim": 4, "z_dim": 4, "num_res_blocks": 1}
    if small:
        options["dec_dim"] = 4
    model = (VideoVAE38_ if small else VideoVAE_)(**options).eval()
    scale = [torch.zeros(4), torch.ones(4)]
    video = torch.randn(1, 3, 9, 32, 32)
    with torch.no_grad():
        encoded = model.encode(video, scale)
        fresh = model.encode(video, scale)
        first = model.encode(video[:, :, :1], scale)
    assert encoded.shape == (1, 4, 3, 2 if small else 4, 2 if small else 4)
    torch.testing.assert_close(encoded, fresh)
    torch.testing.assert_close(encoded[:, :, :1], first)


def test_actual_small_text_and_vision_components():
    from lerobot.policies.dreamzero.wan.wan_video_image_encoder import VisionTransformer
    from lerobot.policies.dreamzero.wan.wan_video_text_encoder import WanTextEncoder

    encoder = WanTextEncoder(vocab=32, dim=16, dim_attn=16, dim_ffn=32, num_heads=2, num_layers=2).eval()
    tokens = torch.randint(32, (2, 8))
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1] * 8])
    with torch.no_grad():
        text = encoder(tokens, mask)
        tokens[0, 4:] = (tokens[0, 4:] + 1) % 32
        changed = encoder(tokens, mask)
    torch.testing.assert_close(text[0, :4], changed[0, :4])
    vision = VisionTransformer(
        image_size=28, patch_size=14, dim=16, num_heads=2, num_layers=2, out_dim=16
    ).eval()
    assert torch.isfinite(vision(torch.randn(1, 3, 28, 28), use_31_block=True)).all()


def test_upstream_quantile_normalization_constant_channels_and_relative_training():
    values = {"min": [0.0] * 6, "max": [10.0] * 5 + [0.0]}
    cfg = tiny_config(
        normalization_style="q99",
        action_stats=values,
        state_stats=values,
        relative_action_indices=[0, 1, 2, 3, 4],
    )
    pre, post = make_pre_post_processors(cfg)
    policy = policy_for(cfg)
    batch = batch_for(cfg)
    batch["observation.state"][:, 0] = 2
    batch["observation.state"][:, 1] = 7
    batch["action"][:, :4] = 5
    batch["action"][:, 4:] = 10
    observed = {}
    original = policy._training_loss

    def capture(latent, action, state, *args):
        observed["action"], observed["state"] = action, state
        return original(latent, action, state, *args)

    policy._training_loss = capture
    policy(pre(batch))
    torch.testing.assert_close(observed["action"][0, :, 0], torch.full((8,), -0.4))
    assert (observed["state"][..., 5] == 1).all()
    result = post(policy.predict_action_chunk(pre(batch_for(cfg, False))))
    assert result.shape == (1, 4, 6) and torch.isfinite(result).all()
    assert (result[..., 5] == 0).all()


def test_original_metadata_camera_order_relative_stats_and_horizon():
    cfg = tiny_config()
    original = {
        "action_head_cfg": {
            "config": {
                "num_frames": cfg.num_video_frames,
                "diffusion_model_cfg": cfg.model_kwargs(),
                "max_action_dim": 8,
                "max_state_dim": 8,
            }
        }
    }
    state = {"joint": {"q01": [-1.0] * 5, "q99": [1.0] * 5}, "gripper": {"q01": [0.0], "q99": [1.0]}}
    metadata = {"oxe_droid": {"statistics": {"state": state, "action": state}}}
    experiment = {
        "relative_action": True,
        "relative_action_keys": ["joint"],
        "modality_configs": {
            "oxe_droid": {
                "state": {"modality_keys": ["state.joint", "state.gripper"]},
                "action": {"modality_keys": ["action.joint", "action.gripper"]},
                "video": {"modality_keys": ["video.left", "video.right", "video.wrist"]},
            }
        },
    }
    converted = config_from_original(
        original,
        embodiment="oxe_droid",
        metadata=metadata,
        experiment=experiment,
        device="cpu",
        image_height=32,
        image_width=64,
    )
    assert converted.camera_keys == [
        "observation.images.left",
        "observation.images.right",
        "observation.images.wrist",
    ]
    assert converted.camera_layout == "droid" and converted.relative_action_indices == list(range(5))
    assert converted.action_stats["min"] == [-1.0] * 5 + [0.0]
    assert not converted.reset_action_heads


def test_direct_chunk_after_partial_execution_clears_old_queue():
    cfg = tiny_config()
    policy = policy_for(cfg)
    policy.select_action(batch_for(cfg, False))
    assert policy._action_queue
    policy.predict_action_chunk(batch_for(cfg, False))
    assert not policy._action_queue


def test_single_frame_direct_inference_restarts_context():
    cfg = tiny_config()
    policy = policy_for(cfg)
    first = policy.predict_action_chunk(batch_for(cfg, False))
    second = policy.predict_action_chunk(batch_for(cfg, False))
    torch.testing.assert_close(first, second)
    assert policy._start_frame == 2


def test_conversion_entrypoint_produces_native_checkpoint(tmp_path):
    from lerobot.policies.dreamzero.convert_checkpoint import convert

    cfg = tiny_config()
    source = CausalWanModel(**cfg.model_kwargs())
    save_original(source, tmp_path / "original", shards=True)
    original = {
        "action_head_cfg": {
            "config": {
                "num_frames": cfg.num_video_frames,
                "target_video_height": 32,
                "target_video_width": 64,
                "diffusion_model_cfg": cfg.model_kwargs(),
                "max_action_dim": 8,
                "max_state_dim": 8,
            }
        }
    }
    (tmp_path / "original/config.json").write_text(json.dumps(original))
    convert(str(tmp_path / "original"), tmp_path / "native", dataset_stats=stats())
    cfg_loaded = PreTrainedConfig.from_pretrained(tmp_path / "native")
    assert cfg_loaded.type == "dreamzero" and cfg_loaded.chunk_size == 4
    native = DreamZeroPolicy.from_pretrained(tmp_path / "native")
    torch.testing.assert_close(
        native.model.patch_embedding.weight.float(), source.patch_embedding.weight.bfloat16().float()
    )
    assert (tmp_path / "native/policy_preprocessor.json").exists()
    report = json.loads((tmp_path / "native/conversion_report.json").read_text())
    assert all(
        name.startswith(("state_encoder.", "action_encoder.", "action_decoder."))
        for name in report["missing"]
    )


def test_fps_validation_and_saved_relative_stats_override(tmp_path):
    cfg = tiny_config()
    with pytest.raises(ValueError, match="FPS"):
        make_pre_post_processors(cfg, dataset_stats=stats(), dataset_meta=SimpleNamespace(fps=15))
    values = {"min": [-1.0] * 6, "max": [1.0] * 6}
    cfg = tiny_config(
        normalization_style="q99",
        action_stats=values,
        state_stats=values,
        relative_action_indices=list(range(5)),
    )
    pre, post = make_pre_post_processors(cfg)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    re_pre, re_post = make_pre_post_processors(
        cfg,
        pretrained_path=str(tmp_path),
        dataset_stats=stats(),
        preprocessor_overrides={"normalizer_processor": {"stats": stats()}},
        postprocessor_overrides={"unnormalizer_processor": {"stats": stats()}},
    )
    batch = batch_for(cfg)
    torch.testing.assert_close(re_pre(batch)["action"], pre(batch)["action"])
    torch.testing.assert_close(re_post(torch.zeros(1, 4, 6)), post(torch.zeros(1, 4, 6)))


def test_frozen_vae_wrappers_do_not_require_cuda(monkeypatch):
    import lerobot.policies.dreamzero.wan.wan_video_vae as vae_module

    class EmptyVAE(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

    monkeypatch.setattr(vae_module, "VideoVAE_", EmptyVAE)
    monkeypatch.setattr(vae_module, "VideoVAE38_", EmptyVAE)
    for cls, channels in ((vae_module.WanVideoVAE, 16), (vae_module.WanVideoVAE38, 48)):
        wrapper = cls()
        assert wrapper.mean.device.type == "cpu" and wrapper.mean.shape == (channels,)
