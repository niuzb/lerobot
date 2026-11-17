"""Binary group-normalized, asymmetric-clipped GRPO with exactly zero KL."""

import torch


def advantages(rewards):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.shape != (8,) or not torch.all((rewards == 0) | (rewards == 1)):
        raise ValueError("Require eight actual binary simulator outcomes")
    rate = float(rewards.mean())
    if not 0.1 <= rate <= 0.9:
        return None
    return ((rewards - rewards.mean()) / rewards.std(unbiased=False)).detach()


def clipped_loss(log_probability, old_log_probability, advantage, epsilon_low=0.2, epsilon_high=0.28):
    if log_probability.shape != old_log_probability.shape:
        raise ValueError("Current and behavior density shapes differ")
    log_ratio = (log_probability.float() - old_log_probability.detach().float()).clamp(-20, 20)
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1 - epsilon_low, 1 + epsilon_high)
    advantage = torch.as_tensor(advantage, device=ratio.device).detach()
    loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    metrics = {
        "ratio_mean": float(ratio.detach().mean()),
        "ratio_max_drift": float((ratio.detach() - 1).abs().max()),
        "clip_fraction": float(
            ((ratio.detach() < 1 - epsilon_low) | (ratio.detach() > 1 + epsilon_high)).float().mean()
        ),
    }
    return loss, metrics
