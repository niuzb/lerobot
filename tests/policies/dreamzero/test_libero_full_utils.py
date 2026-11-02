"""Deterministic contracts for the DreamZero LIBERO full-training path."""

import random

import numpy as np
import pytest
import torch

from examples.dreamzero.libero_full_utils import (
    BalancedWindows,
    camera_images,
    isolated_rng,
    sim_action,
    split_episodes,
    validation_indices,
)


def fake_episodes(tasks=4, per_task=10, length=160):
    rows = []
    for task_id in range(tasks):
        for episode in range(per_task):
            idx = task_id * per_task + episode
            start = idx * length
            rows.append({
                "episode_index": idx,
                "dataset_from_index": start,
                "dataset_to_index": start + length,
                "tasks": [f"task-{task_id}"],
            })
    return rows


def test_split_is_episode_disjoint_and_windows_fit_episode():
    split = split_episodes(fake_episodes(), seed=1000, tail=64)
    assert len(split["train"]) == len(split["validation"]) == 4
    for task in split["train"]:
        train_ids = {row["episode"] for row in split["train"][task]}
        valid_ids = {row["episode"] for row in split["validation"][task]}
        assert len(train_ids) == 9
        assert len(valid_ids) == 1
        assert not train_ids & valid_ids
        assert all(row["stop"] == row["end"] - 64 for row in split["train"][task])
        # Offset 64 is inclusive, so the largest sampled start is end - 65.
        assert all(row["start"] < row["stop"] for row in split["validation"][task])


def test_balanced_window_sampler_restores_exact_next_draw():
    groups = split_episodes(fake_episodes(), seed=1000, tail=64)["train"]
    first = BalancedWindows(groups, seed=123)
    expected = [first.next() for _ in range(17)]
    state = first.state_dict()
    continuation = [first.next() for _ in range(10)]
    resumed = BalancedWindows(groups, seed=999)
    resumed.load_state_dict(state)
    assert [resumed.next() for _ in range(10)] == continuation
    assert len(set(expected)) > 1


def test_validation_windows_are_fixed_and_within_episode():
    groups = split_episodes(fake_episodes(), seed=1000, tail=64)["validation"]
    first = validation_indices(groups, seed=101)
    second = validation_indices(groups, seed=101)
    assert first == second
    assert len(first) == 20
    assert all(sum(row["task"] == task for row in first) == 5 for task in groups)
    for row in first:
        assert any(r["start"] <= row["index"] < r["stop"] for r in groups[row["task"]])


def test_libero_action_conversion_uses_dataset_gripper_convention():
    closed = sim_action([2, -2, 0.1, 0, 0, 0, 0])
    opened = sim_action([0, 0, 0, 0, 0, 0, 1])
    assert np.allclose(closed[:6], [1, -1, 0.1, 0, 0, 0])
    assert closed[6] == 1
    assert opened[6] == -1
    with pytest.raises(FloatingPointError):
        sim_action([0, 0, 0, 0, 0, 0, np.nan])


def test_libero_camera_orientation_and_order():
    main = np.arange(3 * 3 * 3).reshape(3, 3, 3)
    wrist = (100 + np.arange(3 * 3 * 3)).reshape(3, 3, 3)
    got = camera_images({"agentview_image": main, "robot0_eye_in_hand_image": wrist})
    assert np.array_equal(got[0], main[::-1, ::-1])
    assert np.array_equal(got[1], wrist[::-1, ::-1])
    assert got[0].flags.c_contiguous and got[1].flags.c_contiguous


def test_validation_rng_isolation_preserves_training_streams():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), np.random.rand(), torch.rand(()).item())
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    with isolated_rng(123):
        _ = (random.random(), np.random.rand(), torch.rand(()))
    actual = (random.random(), np.random.rand(), torch.rand(()).item())
    assert actual == expected


def test_libero_min_max_action_normalization_round_trip():
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor import TransitionKey, batch_to_transition
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    from lerobot.utils.constants import ACTION

    features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    norm_map = {FeatureType.ACTION: NormalizationMode.MIN_MAX}
    stats = {ACTION: {"min": torch.tensor([-1.0] * 7), "max": torch.tensor([1.0] * 7)}}
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
    source = torch.tensor([[-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0]])
    normalized = normalizer(batch_to_transition({ACTION: source}))[TransitionKey.ACTION]
    restored = unnormalizer(batch_to_transition({ACTION: normalized}))[TransitionKey.ACTION]
    torch.testing.assert_close(restored, source)


def test_streamed_optimizer_restore_reproduces_next_update(tmp_path, monkeypatch):
    """Restore into existing allocations and compare the next Adam update exactly."""
    from types import SimpleNamespace

    from examples.dreamzero import libero_full_checkpoint as checkpoint
    from examples.dreamzero.libero_full_utils import rng_state

    monkeypatch.setattr(checkpoint.importlib.metadata, "version", lambda _: "test-runtime")
    for name in ("policy_config/config.json", "split.json", "train_stats.json", "deepspeed.json",
                 "wan_dit_verified.json", "dataset_hashes.json", "protocol_verified.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    def make_engine():
        model = torch.nn.Linear(4, 3, bias=False)
        master = torch.nn.Parameter(model.weight.detach().flatten().clone())
        adam = torch.optim.AdamW([master], lr=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(adam, lambda _: 1.0)
        zero = SimpleNamespace(optimizer=adam, single_partition_of_fp32_groups=[master],
                               partition_gradients=True, cpu_offload=True, dynamic_loss_scale=False,
                               loss_scaler=None, overflow=False, clip_grad=1.0, micro_step_id=-1)
        return SimpleNamespace(module=model, optimizer=zero, lr_scheduler=scheduler, dp_world_size=1,
                               global_steps=1, global_samples=16, micro_steps=16, skipped_steps=0)

    first = make_engine()
    master = first.optimizer.single_partition_of_fp32_groups[0]
    master.grad = torch.arange(master.numel(), dtype=torch.float32) / 10
    first.optimizer.optimizer.step()
    groups = split_episodes(fake_episodes())["train"]
    sampler = BalancedWindows(groups)
    _ = [sampler.next() for _ in range(16)]
    target = checkpoint.save_resume(first, sampler, 1, tmp_path, rng_state())
    second = make_engine()
    resumed_sampler = BalancedWindows(groups, seed=0)
    update, _ = checkpoint.resume(second, resumed_sampler, target, tmp_path)
    assert update == 1 and second.micro_steps == 16
    assert resumed_sampler.next() == sampler.next()
    torch.testing.assert_close(second.module.weight, first.module.weight, rtol=0, atol=0)
    restored = second.optimizer.single_partition_of_fp32_groups[0]
    restored.grad = master.grad.clone()
    first.optimizer.optimizer.step()
    second.optimizer.optimizer.step()
    torch.testing.assert_close(restored, master, rtol=0, atol=0)
    # A corrupted shard is rejected before it can silently change an optimizer state.
    shard = next((target / "tensors").glob("tensor-*.safetensors"))
    payload = bytearray(shard.read_bytes())
    payload[-1] ^= 1
    shard.write_bytes(payload)
    with pytest.raises(ValueError, match="Corrupt checkpoint chunk"):
        checkpoint.resume(make_engine(), BalancedWindows(groups), target, tmp_path)
