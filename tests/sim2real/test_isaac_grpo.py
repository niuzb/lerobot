"""GRPO reward filtering, clipping, valid gradients and denoising densities."""

import socket
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

EXAMPLE = Path(__file__).resolve().parents[2] / "scripts/simulation/so101_tape_leisaac"
with patch.object(sys, "path", [str(EXAMPLE), *sys.path]):
    from flowvla_rl.sde import gaussian_log_prob, marginal_preserving_transition
    from grpo_objective import advantages, clipped_loss
    from grpo_wire import receive, send


def test_mixed_binary_group_is_standardized_and_uniform_groups_are_skipped():
    torch.testing.assert_close(
        advantages([1, 0, 1, 0, 0, 1, 0, 1]), torch.tensor([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0])
    )
    assert advantages([0] * 8) is None
    assert advantages([1] * 8) is None
    with pytest.raises(ValueError):
        advantages([0.5] * 8)


def test_asymmetric_ppo_clipping_for_both_advantage_signs():
    old = torch.zeros(2)
    logp = torch.tensor([1.5, 0.5]).log().requires_grad_()
    loss, _ = clipped_loss(logp, old, 1.0)
    torch.testing.assert_close(loss, torch.tensor(-(1.28 + 0.5) / 2))
    loss.backward()
    assert logp.grad[0] == 0 and logp.grad[1] < 0
    logp2 = torch.tensor([1.5, 0.5]).log().requires_grad_()
    loss, _ = clipped_loss(logp2, old, -1.0)
    torch.testing.assert_close(loss, torch.tensor((1.5 + 0.8) / 2))
    loss.backward()
    assert logp2.grad[0] > 0 and logp2.grad[1] == 0


def test_unchanged_policy_ratio_and_real_gradient_improve_rewarded_density():
    mean = torch.nn.Parameter(torch.tensor([0.0]))
    sampled = torch.tensor([[1.0], [-1.0]])
    behavior = gaussian_log_prob(sampled, mean.detach().expand(2, 1), torch.ones(1)).detach()
    optimizer = torch.optim.SGD([mean], lr=0.05)
    current = gaussian_log_prob(sampled, mean.expand(2, 1), torch.ones(1))
    loss, metrics = clipped_loss(current, behavior, torch.tensor([1.0, -1.0]))
    assert metrics["ratio_max_drift"] == 0
    loss.backward()
    assert mean.grad < 0
    optimizer.step()
    assert mean > 0


def test_sde_variance_and_full_latent_log_density():
    x = torch.randn(1, 50, 32)
    transition = marginal_preserving_transition(x, torch.zeros_like(x), 0.5, 0.1, 0.05)
    torch.testing.assert_close(transition.std.square().reshape(-1), torch.tensor([0.05**2 * 0.5 * 0.1]))
    summed = gaussian_log_prob(x, transition.mean, transition.std)
    element = gaussian_log_prob(x, transition.mean, transition.std, reduction="none")
    torch.testing.assert_close(summed, element.sum(dim=(1, 2)))


def test_numpy_rpc_cross_process_format_without_pickle():
    left, right = socket.socketpair()
    values = {
        "state": np.arange(25, dtype=np.float32),
        "front": np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
    }
    thread = threading.Thread(target=send, args=(left, {"operation": "act"}), kwargs=values)
    thread.start()
    metadata, recovered = receive(right)
    thread.join()
    assert metadata == {"operation": "act"}
    for key in values:
        np.testing.assert_array_equal(recovered[key], values[key])
    left.close()
    right.close()
