"""Real-checkpoint CUDA density/backward preflight, with zero optimizer updates."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from eval_real_loss import save
from flowvla_rl.sde import gaussian_log_prob, marginal_preserving_transition
from flowvla_rl.smolvla_sde import prepare_policy_prefix, recompute_log_probs
from grpo_actor import Actor


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    actor = Actor(
        SimpleNamespace(
            root=args.root,
            checkpoint=Path(
                "/root/data/leisaac/runs/tape-mixed-20261110-01/train/checkpoints/020000/pretrained_model"
            ),
            eta=0.05,
        )
    )
    ds = LeRobotDataset(
        "local/so101_tape100_framewise_20261109",
        root="/root/data/leisaac/runs/tape-train100-20261109-01/dataset",
        episodes=[0],
        video_backend="pyav",
        return_uint8=True,
    )
    frame = ds[0]
    observation = {
        "state": frame["observation.state"].numpy(),
        **{name: frame["observation.images." + name].permute(1, 2, 0).numpy() for name in ["front", "side"]},
    }
    actor.begin({"episode": 0, "epoch": 0, "group_seed": 0, "sample_seed": 20261110, "mode": "sde"})
    action = actor.act(observation)
    trace = actor.current["chunks"][0]["trajectory"].to("cuda")
    with torch.no_grad():
        masks, cache = prepare_policy_prefix(actor.policy, actor.prepare(observation))
        scored = recompute_log_probs(actor.policy.model, masks, cache, trace)
        drift = float(((scored - trace.log_probs).exp() - 1).abs().max())
    assert drift <= 0.05
    with torch.no_grad():
        masks, cache = prepare_policy_prefix(actor.policy, actor.prepare(observation))
    velocity = actor.policy.model.denoise_step(
        prefix_pad_masks=masks,
        past_key_values=cache,
        x_t=trace.states[:, 3],
        timestep=torch.tensor([0.7], device="cuda"),
    )
    transition = marginal_preserving_transition(trace.states[:, 3], velocity, 0.7, 0.1, 0.05)
    logp = gaussian_log_prob(trace.states[:, 4], transition.mean, transition.std)
    gradients = torch.autograd.grad(logp.sum(), actor.parameters, allow_unused=True)
    norm = float(sum(g.square().sum() for g in gradients if g is not None).sqrt())
    assert np.isfinite(action).all() and np.isfinite(norm) and norm > 0
    report = {
        "passed": True,
        "source": "actual mixed-SFT checkpoint and recorded synthetic observation; API preflight only",
        "action_shape": list(action.shape),
        "ratio_max_drift": drift,
        "gradient_norm": norm,
        "optimizer_steps": 0,
        "weights_updated": False,
        "device": "cuda",
        "precision": "float32",
    }
    save(args.root / "density_preflight.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
