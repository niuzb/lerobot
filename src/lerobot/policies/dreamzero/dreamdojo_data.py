"""SO-101 action preparation for NVIDIA DreamDojo's 2B action layout."""

import torch
from torch import Tensor

ACTION_DIM = 384
SO101_ACTION_START = 169
SO101_ACTION_END = 220
ACTION_CHUNK_SIZE = 4


def encode_so101_actions(absolute_actions: Tensor) -> Tensor:
    """Encode 13 absolute action targets as 12 chunk-relative DreamDojo actions.

    The first target is the baseline for the first four predicted frames; each
    subsequent chunk uses the last target of the previous chunk as its baseline.
    Leading batch dimensions are preserved.
    """
    if absolute_actions.ndim < 2 or absolute_actions.shape[-2] != 13:
        raise ValueError("Expected 13 action targets, including the baseline")
    action_dim = absolute_actions.shape[-1]
    if not 0 < action_dim <= SO101_ACTION_END - SO101_ACTION_START:
        raise ValueError("SO-101 action dimension must fit the reserved action channels")
    if not torch.is_floating_point(absolute_actions) or not torch.isfinite(absolute_actions).all():
        raise ValueError("Action targets must be finite floating-point values")

    targets = absolute_actions[..., 1:, :]
    baselines = absolute_actions[..., :-1:ACTION_CHUNK_SIZE, :].repeat_interleave(
        ACTION_CHUNK_SIZE, dim=-2
    )
    encoded = absolute_actions.new_zeros(*absolute_actions.shape[:-2], 12, ACTION_DIM)
    encoded[..., SO101_ACTION_START : SO101_ACTION_START + action_dim] = targets - baselines
    return encoded
