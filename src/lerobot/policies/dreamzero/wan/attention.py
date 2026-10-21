# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Device-independent SDPA used by the vendored Wan modules.

DreamZero passes each causal block's permitted keys explicitly. Do not replace
these calls with an unmasked attention over the complete training sequence.
"""

import torch
from torch import nn
from torch.nn import functional as F


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    **kwargs,
):
    """Upstream-compatible entry point; no FlashAttention installation required."""
    flattened = q.ndim == 3
    if flattened:
        # Wan CLIP uses [B,S,H*D]; the causal DiT uses [B,S,H,D].
        heads = kwargs.get("num_heads")
        if heads is None or any(t.shape[-1] % heads for t in (q, k, v)):
            raise ValueError("Flattened attention requires a valid num_heads")
        q, k, v = (t.unflatten(-1, (heads, t.shape[-1] // heads)) for t in (q, k, v))
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    mask = None
    if q_lens is not None or k_lens is not None or window_size != (-1, -1):
        qi = torch.arange(q.shape[-2], device=q.device)[:, None]
        ki = torch.arange(k.shape[-2], device=q.device)[None, :]
        mask = torch.ones((q.shape[0], 1, q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.bool)
        if k_lens is not None:
            mask &= ki[None, None] < torch.as_tensor(k_lens, device=q.device)[:, None, None, None]
        if q_lens is not None:
            mask &= qi[None, None] < torch.as_tensor(q_lens, device=q.device)[:, None, None, None]
        if causal:
            mask &= ki <= qi
        left, right = window_size
        if left >= 0:
            mask &= ki >= qi - left
        if right >= 0:
            mask &= ki <= qi + right
    if q_scale is not None:
        q = q * q_scale
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=causal and mask is None,
        scale=softmax_scale,
    )
    out = out.transpose(1, 2).contiguous()
    return out.flatten(2) if flattened else out


class AttentionModule(nn.Module):
    def __init__(self, num_heads, head_dim, causal=False, **kwargs):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v):
        return flash_attention(q, k, v, causal=self.causal)
