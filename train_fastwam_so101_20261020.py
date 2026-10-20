"""Pinned SO-101 FastWAM experiment: read-only data, exact 4-real/4-sim updates."""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np


def write_json(path, value):
    """Write finite status data atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def valid_starts(episode_ids, horizon=32):
    """Require an entire future-video window inside one contiguous episode."""
    ids = np.asarray(episode_ids)
    if len(ids) <= horizon:
        return np.empty(0, dtype=np.int64)
    boundaries = np.r_[0, np.cumsum(ids[1:] != ids[:-1])]
    return np.flatnonzero(boundaries[:-horizon] == boundaries[horizon:])


def sample_update(real_indices, sim_indices, seed, update):
    """Make each optimizer update deterministic, with exactly four of each domain."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, update]))
    return [("real", int(i)) for i in rng.choice(real_indices, 4)] + [
        ("sim", int(i)) for i in rng.choice(sim_indices, 4)
    ]


def checkpoint_key(key):
    """Use the same legacy block-name conversion as the upstream MoT loader."""
    match = re.fullmatch(r"model\.mot\.mixtures\.([^.]+)\.blocks\.(\d+)\.(.+)", key)
    if match:
        expert, layer, rest = match.groups()
        return f"model.mot.layers.{layer}.blocks.{expert}.{rest}"
    return key


def audit_checkpoint(model_shapes, weight_shapes, full=False):
    """Allow only embodiment-specific heads to be absent from the generic base."""
    converted = {checkpoint_key(k): tuple(shape) for k, shape in weight_shapes.items()}
    if len(converted) != len(weight_shapes):
        raise ValueError("Duplicate checkpoint aliases")
    permitted = {
        f"model.mot.mixtures.action.{layer}.{kind}"
        for layer in ("action_encoder", "head")
        for kind in ("weight", "bias")
    } | {f"model.proprio_encoder.{kind}" for kind in ("weight", "bias")}
    missing = set(model_shapes) - set(converted)
    extra = set(converted) - set(model_shapes)
    mismatched = {k for k in converted.keys() & model_shapes.keys() if converted[k] != tuple(model_shapes[k])}
    allowed = set() if full else permitted
    if extra or not (missing | mismatched).issubset(allowed):
        raise ValueError(
            f"Invalid checkpoint coverage: missing={missing}, extra={extra}, mismatched={mismatched}"
        )
    return sorted(missing | mismatched)


def make_data(job):
    """Load fixed splits and real-only normalization without editing source data."""
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
    from lerobot.sim2real.config import JOINTS, TASK, TRAIN, VALIDATION, digest, load_config
    from lerobot.sim2real.data import SourceData

    scene = load_config(job["scene_config"])
    source = SourceData(scene)
    root = Path(job["job_root"])
    cfg = FastWAMConfig(
        **job["policy"],
        device="cpu",
        base_model_id=None,
        pretrained_path=root / "assets/base",
        push_to_hub=False,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    delta = {
        "observation.images.front": [i / 30 for i in cfg.observation_delta_indices],
        "observation.images.side": [i / 30 for i in cfg.observation_delta_indices],
        "observation.state": [0.0],
        "action": [i / 30 for i in cfg.action_delta_indices],
    }
    data = {}
    for name, episodes in (("real", TRAIN), ("validation", VALIDATION)):
        data[name] = LeRobotDataset(
            scene["dataset_repo_id"],
            root=scene["dataset_root"],
            episodes=episodes,
            delta_timestamps=delta,
            video_backend="pyav",
        )
    generated = Path(job["generated"])
    manifest = json.loads((generated / "generation_manifest.json").read_text())
    verification = json.loads((generated / "dataset_verification.json").read_text())
    if not manifest["passed"] or manifest["physics_only"] or manifest["accepted"] != 200:
        raise ValueError("Expected the completed 200-success rendered simulation dataset")
    if not verification["passed"] or verification["frames"] != 138000:
        raise ValueError("Simulation verification failed")
    for name, expected in manifest["dataset_sha256"].items():
        path = (generated / "dataset" / name).resolve()
        if not path.is_relative_to((generated / "dataset").resolve()) or digest(path) != expected:
            raise ValueError(f"Simulation data integrity failure: {name}")
    data["sim"] = LeRobotDataset(
        "local/so101_r2s2r_sim",
        root=generated / "dataset",
        delta_timestamps=delta,
        video_backend="pyav",
    )
    indices = {}
    for name, dataset in data.items():
        for key in ("observation.state", "action"):
            if dataset.meta.features[key]["names"] != [f"{j}.pos" for j in JOINTS]:
                raise ValueError(f"Wrong joint order: {name} {key}")
        ids = np.asarray(dataset.hf_dataset["episode_index"])
        expected_ids = TRAIN if name == "real" else VALIDATION if name == "validation" else range(200)
        if set(ids.tolist()) != set(expected_ids):
            raise ValueError(f"Unexpected episode split: {name}")
        indices[name] = valid_starts(ids)
        if not len(indices[name]):
            raise ValueError(f"No complete temporal windows in {name}")
        if dataset.meta.fps != 30:
            raise ValueError(f"Wrong frame rate: {name}")
    stats = {}
    for key, array in (("observation.state", source.state), ("action", source.action)):
        values = array[np.isin(source.ep, TRAIN)].astype(np.float64)
        stats[key] = {
            "mean": values.mean(0),
            "std": np.maximum(values.std(0), 1e-6),
            "min": values.min(0),
            "max": values.max(0),
            "count": np.array([len(values)]),
        }
    info = {
        "source": source.manifest(),
        "generated_manifest_sha256": digest(generated / "generation_manifest.json"),
        "valid_windows": {k: len(v) for k, v in indices.items()},
        "delta_timestamps": delta,
        "calibration_passed": False,
        "task": TASK,
    }
    return cfg, data, indices, stats, info


def batch_for(dataset, index):
    """Read one complete temporal sample and reject padding or mismatched semantics."""
    import torch
    from torch.utils.data import default_collate

    from lerobot.sim2real.config import TASK

    batch = default_collate([dataset[int(index)]])
    if batch["task"] != [TASK]:
        raise ValueError("Unexpected task text; cached language embedding would be wrong")
    batch["observation.state"] = batch["observation.state"][:, 0, :]
    if tuple(batch["action"].shape) != (1, 32, 6):
        raise ValueError("Wrong action window")
    for name in ("front", "side"):
        key = f"observation.images.{name}"
        if tuple(batch[key].shape) != (1, 9, 3, 480, 640):
            raise ValueError(f"Wrong temporal image shape for {key}")
        if not torch.isfinite(batch[key]).all():
            raise ValueError("Non-finite image")
        if batch.get(key + "_is_pad", torch.tensor(False)).any():
            raise ValueError("Padded future video entered training")
    if batch.get("action_is_pad", torch.tensor(False)).any():
        raise ValueError("Padded future action entered training")
    return batch


def check_smolvla_finished(job):
    """Enforce the user's dependency on both completed SmolVLA experiments."""
    from lerobot.sim2real.experiment_run import verify_checkpoint

    status = json.loads(Path(job["smolvla_status"]).read_text())
    if not status["completed"] or status["stage"] != "completed":
        raise ValueError("Wait for both SmolVLA candidates to finish")
    for name in job["smolvla_exitcodes"]:
        if Path(name).read_text().strip() != "0":
            raise ValueError("SmolVLA did not exit successfully")
    for name in ("r2s2r_baseline_20261018_t4", "r2s2r_mixed_20261018"):
        verify_checkpoint(Path("/root/code/lerobot/outputs/train") / name)


def build_policy(job, cfg, checkpoint=None):
    """Load pinned shared weights and initialize only the SO-101-specific heads."""
    import torch
    from safetensors import safe_open

    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    from lerobot.policies.fastwam.wan import components

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("FastWAM job requires a CUDA GPU with native bfloat16 support")
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < job["minimum_gpu_gib"] * 1024**3:
        raise RuntimeError(f"Need >= {job['minimum_gpu_gib']} GiB; got {props.total_memory / 1024**3:.1f}")
    root = Path(job["job_root"])
    assets = json.loads((root / "preparation_status.json").read_text())
    if not assets["completed"]:
        raise ValueError("Pinned model assets have not passed integrity verification")
    cfg.device = "cuda"
    cfg.text_encoder_model_id = str(root / "assets/wan")
    cfg.tokenizer_model_id = str(root / "assets/tokenizer")
    # The upstream VAE loader uses this module constant. Pin only this process's asset locator;
    # retain the original model code and record the exact local component files in the manifest.
    components.WAN22_DIFFUSERS_MODEL_ID = str(root / "assets/wan")
    model = FastWAMPolicy(cfg)
    weight = Path(checkpoint) / "model.safetensors" if checkpoint else root / "assets/base/model.safetensors"
    state = model.state_dict()
    with safe_open(weight, framework="pt") as reader:
        keys = reader.keys()
        shapes = {k: reader.get_slice(k).get_shape() for k in keys}
        fresh = audit_checkpoint({k: tuple(v.shape) for k, v in state.items()}, shapes, full=bool(checkpoint))
        weights = {checkpoint_key(k): reader.get_tensor(k) for k in keys if checkpoint_key(k) not in fresh}
        result = model.load_state_dict(weights, strict=bool(checkpoint))
        if set(result.missing_keys) != set(fresh) or result.unexpected_keys:
            raise ValueError("Loaded weights differ from the audited coverage")
    model.eval()
    return model, {"gpu": props.name, "vram_gib": props.total_memory / 1024**3, "freshly_initialized": fresh}


def evaluate(policy, pre, dataset, indices, context, mask, smoke):
    """Evaluate held-out complete windows with fixed noise and no training RNG side effects."""
    import torch

    policy.eval()
    totals = {"loss": 0.0, "loss_action": 0.0, "loss_video": 0.0}
    chosen = indices[:2] if smoke else indices
    # Keep validation noise fixed across checkpoints and preserve the training RNG.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(20261020)
        with torch.inference_mode():
            for index in chosen:
                batch = pre(batch_for(dataset, index))
                batch.update(context=context, context_mask=mask)
                value, metrics = policy(batch)
                if not torch.isfinite(value):
                    raise RuntimeError("Non-finite validation loss")
                totals["loss"] += value.item()
                for key in ("loss_action", "loss_video"):
                    totals[key] += float(metrics[key])
    return {**{k: v / len(chosen) for k, v in totals.items()}, "windows": len(chosen), "smoke": smoke}


def run(job, mode, checkpoint=None):
    """Run CPU preflight, GPU smoke, production training, or saved-policy reload checks."""
    import torch

    from lerobot.policies.fastwam.modeling_fastwam import _stack_video_from_images
    from lerobot.policies.fastwam.processor_fastwam import make_fastwam_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    torch.set_num_threads(4)
    check_smolvla_finished(job)
    cfg, data, indices, stats, info = make_data(job)
    pre, _ = make_fastwam_pre_post_processors(cfg, stats)
    for name, dataset in data.items():
        for index in (indices[name][0], indices[name][len(indices[name]) // 2], indices[name][-1]):
            batch = pre(batch_for(dataset, index))
            video = _stack_video_from_images(batch, cfg)
            if tuple(video.shape) != (1, 3, 9, 224, 448):
                raise ValueError("FastWAM temporal/camera adaptation failed")
    root = Path(job["job_root"])
    write_json(root / "data_preflight.json", {**info, "passed": True})
    if mode == "preflight":
        print(json.dumps({"passed": True, "valid_windows": info["valid_windows"]}), flush=True)
        return
    set_seed(job["seed"])
    policy, hardware = build_policy(job, cfg, checkpoint)
    pre, post = make_fastwam_pre_post_processors(cfg, stats)
    if mode == "validate":
        if checkpoint is None:
            raise ValueError("Validation requires a saved checkpoint")
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(
                pre(batch_for(data["validation"], indices["validation"][0]))
            )
            actions = post(normalized)
        if tuple(actions.shape) != (1, 32, 6) or not torch.isfinite(actions).all():
            raise ValueError("Reloaded policy produced invalid actions")
        write_json(
            Path(checkpoint).parent / "reload_check.json",
            {"passed": True, "action_shape": list(actions.shape)},
        )
        print("CHECKPOINT_RELOAD_PASSED", flush=True)
        return
    smoke = mode == "smoke"
    out = root / "smoke" if smoke else Path(job["output"])
    out.mkdir(parents=True, exist_ok=False)
    steps = 2 if smoke else job["steps"]
    manifest = {
        **info,
        "job": job,
        "hardware": hardware,
        "steps": steps,
        "smoke": smoke,
        "completed": False,
        "step_unit": "optimizer_updates",
        "effective_batch": 8,
        "domain_batch": {"real": 4, "sim": 4},
        "started_unix": time.time(),
        "base_revision": job["models"][0]["revision"],
        "full_video_action_training": True,
    }
    write_json(out / "training_manifest.json", manifest)
    prompt = cfg.prompt_template.format(task=info["task"])
    context, mask = policy.model.encode_prompt([prompt])
    # The task is constant. Cache its exact frozen T5 embedding once, then free its GPU memory.
    policy.model.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    params = policy.get_optim_params()
    if len(params) != len({id(p) for p in params}):
        raise ValueError("Duplicate optimizer parameters")
    optimizer = torch.optim.AdamW(params, lr=job["learning_rate"], weight_decay=0.01, foreach=False)
    started = time.monotonic()
    for step in range(1, steps + 1):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for domain, index in sample_update(indices["real"], indices["sim"], job["seed"], step):
            batch = pre(batch_for(data[domain], index))
            batch.update(context=context, context_mask=mask)
            loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            (loss / 8).backward()
            total += loss.item() / 8
        norm = torch.nn.utils.clip_grad_norm_(params, 10.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0 or step <= 2:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": total,
                        "grad_norm": float(norm),
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    }
                ),
                flush=True,
            )
        if step % job["save_every"] == 0 or step == steps:
            metrics = evaluate(policy, pre, data["validation"], indices["validation"], context, mask, smoke)
            destination = out / "checkpoints" / f"{step:06d}"
            destination.mkdir(parents=True, exist_ok=False)
            pretrained = destination / "pretrained_model"
            policy.save_pretrained(pretrained)
            pre.save_pretrained(pretrained)
            post.save_pretrained(pretrained)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                destination / "optimizer_and_rng.pt",
            )
            write_json(destination / "validation.json", {"step": step, **metrics})
            print(json.dumps({"step": step, "validation": metrics}), flush=True)
    manifest.update(completed=True, finished_unix=time.time(), final_step=steps)
    write_json(out / "training_manifest.json", manifest)


def main():
    """Keep GPU execution separate from read-only CPU preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["preflight", "smoke", "train", "validate"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    job = json.loads(args.config.read_text())
    run(job, args.mode, args.checkpoint)


if __name__ == "__main__":
    main()
