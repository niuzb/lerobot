"""Profile compatibility, split optimizer state, and native CUDA smoke checks."""

import gc
import json
import socket
from types import SimpleNamespace

import pytest
import torch

from examples.dreamzero import libero_full_checkpoint as checkpoint
from examples.dreamzero.libero_full_runtime import (
    MODES,
    check_first_allocation,
    deepspeed_config,
    external_optimizer,
    optimizer_audit,
    optimizer_layout,
    resolve_runtime,
)
from examples.dreamzero.libero_full_utils import BalancedWindows, policy_state_tensors, rng_state

JOB = {"accumulation": 2, "grad_clip": 1.0, "lr": 1e-3, "warmup_updates": 2, "weight_decay": 1e-5}


def prepare_root(root):
    for name in (
        "policy_config/config.json",
        "split.json",
        "train_stats.json",
        "deepspeed.json",
        "wan_dit_verified.json",
        "dataset_hashes.json",
        "protocol_verified.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")


def sampler():
    return BalancedWindows({"task": [{"start": 0, "stop": 100}]}, seed=17)


def test_profiles_exclude_incompatible_offload_and_8bit(tmp_path):
    hybrid = resolve_runtime(tmp_path)
    cfg = deepspeed_config(JOB, hybrid)
    assert cfg["zero_optimization"]["stage"] == 3
    assert cfg["zero_optimization"]["offload_optimizer"]["ratio"] == 0.5
    assert cfg["optimizer"]["type"] == "AdamW"
    quantized = resolve_runtime(tmp_path, MODES[1])
    cfg = deepspeed_config(JOB, quantized)
    assert "offload_optimizer" not in cfg["zero_optimization"]
    assert "optimizer" not in cfg
    assert cfg["zero_allow_untested_optimizer"]
    assert resolve_runtime(tmp_path) == quantized
    with pytest.raises(ValueError, match="cannot use CPU offload"):
        resolve_runtime(tmp_path, MODES[1], 0.5)
    assert resolve_runtime(tmp_path) == quantized
    for ratio in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            resolve_runtime(tmp_path, MODES[0], ratio)
    marker = tmp_path / "checkpoints/update_000004/COMPLETE.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    with pytest.raises(ValueError, match="sealed"):
        resolve_runtime(tmp_path, MODES[0])


def fake_partitioned_engine():
    """Emulate two optimizer owners on CPU, without claiming CUDA coverage."""
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False))
    masters = []
    for parameter in model.parameters():
        parameter.ds_shape = parameter.shape
        parameter.ds_numel = parameter.numel()
        parameter.ds_tensor = parameter.detach().flatten().clone()
        masters.append(torch.nn.Parameter(parameter.ds_tensor.clone()))
        parameter.data = torch.empty(0)
    cpu = torch.optim.AdamW([masters[1]], lr=0.001)
    gpu = torch.optim.AdamW([masters[0]], lr=0.001)
    for i, master in enumerate(masters):
        master.grad = torch.arange(master.numel(), dtype=torch.float32) / (10 + i)
    cpu.step()
    gpu.step()
    zero = SimpleNamespace(
        optimizer=cpu,
        backup_optimizer=gpu,
        fp32_partitioned_groups_flat=masters,
        subgroup_to_device={0: "cuda", 1: "cpu"},
        swap_optimizer=False,
        dynamic_loss_scale=False,
        loss_scaler=None,
        overflow=False,
        clip_grad=1.0,
        micro_step_id=0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(cpu, lambda _: 1.0)
    return SimpleNamespace(
        module=model,
        optimizer=zero,
        lr_scheduler=scheduler,
        dp_world_size=1,
        global_steps=1,
        global_samples=2,
        micro_steps=2,
        skipped_steps=0,
    )


def test_hybrid_checkpoint_restores_both_owners_and_next_update(tmp_path, monkeypatch):
    prepare_root(tmp_path)
    monkeypatch.setattr(checkpoint.importlib.metadata, "version", lambda _: "test-runtime")
    first = fake_partitioned_engine()
    samples = sampler()
    samples.next()
    target = checkpoint.save_resume(first, samples, 1, tmp_path, rng_state())
    second = fake_partitioned_engine()
    # Force the restore path to allocate fresh moments rather than reuse state.
    second.optimizer.optimizer.state.clear()
    second.optimizer.backup_optimizer.state.clear()
    resumed_samples = sampler()
    checkpoint.resume(second, resumed_samples, target, tmp_path)
    assert resumed_samples.next() == samples.next()
    original, owners, assigned = optimizer_layout(first.optimizer)
    restored, resumed_owners, resumed_assignment = optimizer_layout(second.optimizer)
    assert assigned == resumed_assignment == ["gpu", "primary"]
    for p, q, name in zip(original, restored, assigned, strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                owners[name].state[p][key], resumed_owners[name].state[q][key], rtol=0, atol=0
            )
        q.grad = p.grad.clone()
    for name in owners:
        owners[name].step()
        resumed_owners[name].step()
    for p, q in zip(original, restored, strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    for name, tensor in policy_state_tensors(first.module).items():
        torch.testing.assert_close(tensor, policy_state_tensors(second.module)[name], rtol=0, atol=0)
        assert tensor.shape == (4, 4)
    # The public parameter tensors are still partitioned placeholders.
    assert all(p.numel() == 0 for p in second.module.parameters())


def test_quantized_checkpoint_keeps_uint8_and_scaling_metadata(tmp_path, monkeypatch):
    prepare_root(tmp_path)
    monkeypatch.setattr(checkpoint.importlib.metadata, "version", lambda _: "test-runtime")
    first = fake_partitioned_engine()
    masters, owners, assigned = optimizer_layout(first.optimizer)
    for master, name in zip(masters, assigned, strict=True):
        state = owners[name].state[master]
        state.clear()
        state.update(
            step=7,
            state1=torch.arange(master.numel(), dtype=torch.uint8),
            state2=torch.full(master.shape, 127, dtype=torch.uint8),
            qmap1=torch.linspace(-1, 1, 256),
            qmap2=torch.linspace(0, 1, 256),
            absmax1=torch.tensor([0.03]),
            absmax2=torch.tensor([0.002]),
        )
    target = checkpoint.save_resume(first, sampler(), 1, tmp_path, rng_state())
    second = fake_partitioned_engine()
    second.optimizer.optimizer.state.clear()
    second.optimizer.backup_optimizer.state.clear()
    checkpoint.resume(second, sampler(), target, tmp_path)
    restored, new_owners, _ = optimizer_layout(second.optimizer)
    for p, q, name in zip(masters, restored, assigned, strict=True):
        expected, actual = owners[name].state[p], new_owners[name].state[q]
        assert actual["step"] == expected["step"]
        assert actual["state1"].dtype == actual["state2"].dtype == torch.uint8
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


def test_first_allocation_budgets_actual_owner_device(tmp_path, monkeypatch):
    from examples.dreamzero import libero_full_runtime as runtime

    p = torch.nn.Parameter(torch.zeros(4096))
    adam = torch.optim.AdamW([p])
    zero = SimpleNamespace(optimizer=adam, single_partition_of_fp32_groups=[p])
    monkeypatch.setattr(runtime, "check_memory", lambda: {"process_current_gib": 2, "gpu_allocated_gib": 3})
    check_first_allocation(zero, {"mode": MODES[0]}, tmp_path)
    report = json.loads((tmp_path / "first_optimizer_allocation.json").read_text())
    assert report["new_state_bytes_by_device"] == {"cpu": p.numel() * 8, "cuda": 0}
    monkeypatch.setattr(runtime, "check_memory", lambda: {"process_current_gib": 89.999999})
    with pytest.raises(MemoryError):
        check_first_allocation(zero, {"mode": MODES[0]}, tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native ZeRO-3 optimizer kernels require CUDA")
@pytest.mark.parametrize("mode", MODES)
def test_cuda_zero3_checkpoint_reproduces_next_update(tmp_path, mode, monkeypatch):
    """Real kernels: two optimizer updates, save/rebuild, compare the third."""
    ds = pytest.importorskip("deepspeed")
    if ds.__version__ != "0.19.8":
        pytest.skip("Streaming runtime is pinned to DeepSpeed 0.19.8")
    if mode == MODES[1]:
        pytest.importorskip("bitsandbytes")
    prepare_root(tmp_path)
    profile = resolve_runtime(tmp_path, mode)
    config = deepspeed_config(JOB, profile)
    config["zero_optimization"]["sub_group_size"] = 16384
    config["zero_optimization"]["reduce_bucket_size"] = 16384
    (tmp_path / "deepspeed.json").write_text(json.dumps(config))
    owns_group = not torch.distributed.is_initialized()
    if owns_group:
        monkeypatch.setenv("LOCAL_RANK", "0")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        ds.init_distributed(
            dist_backend="nccl",
            auto_mpi_discovery=False,
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=1,
        )

    def make_engine():
        model = (
            torch.nn.Sequential(
                torch.nn.Linear(128, 128, bias=False), torch.nn.Tanh(), torch.nn.Linear(128, 128, bias=False)
            )
            .cuda()
            .bfloat16()
        )
        return ds.initialize(
            model=model,
            model_parameters=list(model.parameters()),
            config=config,
            optimizer=external_optimizer(list(model.parameters()), JOB, profile),
        )[0]

    torch.manual_seed(7)
    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    target = torch.randn_like(x)

    def update(engine):
        for _ in range(JOB["accumulation"]):
            loss = (engine(x).float() - target.float()).square().mean()
            engine.backward(loss)
            backup = getattr(engine.optimizer, "backup_optimizer", None)
            if backup is not None:
                for a, b in zip(engine.optimizer.optimizer.param_groups, backup.param_groups, strict=True):
                    b["lr"] = a["lr"]
            engine.step()

    engine = None
    try:
        engine = make_engine()
        update(engine)
        update(engine)
        optimizer_audit(engine.optimizer, profile, initialized=True)
        saved = checkpoint.save_resume(engine, sampler(), 2, tmp_path, rng_state())
        update(engine)
        expected = [p.detach().cpu().clone() for p in optimizer_layout(engine.optimizer)[0]]
        engine.optimizer.destroy()
        del engine
        engine = None
        gc.collect()
        torch.cuda.empty_cache()
        engine = make_engine()
        checkpoint.resume(engine, sampler(), saved, tmp_path)
        optimizer_audit(engine.optimizer, profile, initialized=True)
        update(engine)
        assert engine.global_steps == 3
        for actual, wanted in zip(optimizer_layout(engine.optimizer)[0], expected, strict=True):
            torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
    finally:
        if engine is not None:
            engine.optimizer.destroy()
        if owns_group:
            torch.distributed.destroy_process_group()
