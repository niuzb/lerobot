# Copyright 2025 NVIDIA Corporation. All rights reserved.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0. See LICENSE and README.md.

import torch
import torch.nn as nn

from .attention import flash_attention

ENABLE_TENSORRT = False


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half, dtype=position.dtype, device=position.device).div(half)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    if ENABLE_TENSORRT:
        return rope_params_no_polar(max_seq_len, dim, theta)
    else:
        return rope_params_polar(max_seq_len, dim, theta)


def rope_params_polar(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_params_no_polar(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    assert dim % 2 == 0
    inv_freq = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32) / dim)
    t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.stack((freqs.cos(), freqs.sin()), dim=-1).flatten(-2)
    return emb


def rope_apply(x, grid_sizes, freqs):
    if ENABLE_TENSORRT:
        return rope_apply_no_polar(x, freqs)
    else:
        return rope_apply_polar(x, freqs)


def rope_apply_polar(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(x.to(torch.float64).reshape(B, seq_len, n, -1, 2))

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x


def rope_apply_no_polar(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    B, seq_len, n, D = x.shape

    # Reshape freqs to be broadcastable: (1, seq_len, 1, D)
    freqs = freqs.unsqueeze(0).unsqueeze(2)

    x0, x1 = x.chunk(2, dim=-1)
    freqs_cos, freqs_sin = freqs.chunk(2, dim=-1)

    rotated_x0 = x0 * freqs_cos - x1 * freqs_sin
    rotated_x1 = x1 * freqs_cos + x0 * freqs_sin
    x_rotated = torch.cat((rotated_x0, rotated_x1), dim=-1)
    return x_rotated


def rope_action_apply(
    x,
    freqs,
    freqs_action,
    freqs_state,
    action_register_length,
    num_action_per_block=32,
    num_state_per_block=1,
):
    if ENABLE_TENSORRT:
        return rope_action_apply_no_polar(
            x,
            freqs,
            freqs_action,
            freqs_state,
            action_register_length,
            num_action_per_block,
            num_state_per_block,
        )
    else:
        return rope_action_apply_polar(
            x,
            freqs,
            freqs_action,
            freqs_state,
            action_register_length,
            num_action_per_block,
            num_state_per_block,
        )


def rope_action_apply_no_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int,
    num_action_per_block: int = 32,
    num_state_per_block: int = 1,
) -> torch.Tensor:
    B, seq_len, n, D = x.shape

    if action_register_length is not None:
        chunk_size = action_register_length // (num_action_per_block + num_state_per_block)
        freqs_1d_action = freqs_action[: chunk_size * num_action_per_block]
        freqs_1d_state = freqs_state[: chunk_size * num_state_per_block]
        freqs = torch.cat([freqs, freqs_1d_action, freqs_1d_state], dim=0)

    # Reshape freqs to be broadcastable: (1, seq_len, 1, D)
    freqs = freqs.unsqueeze(0).unsqueeze(2)

    x0, x1 = x.chunk(2, dim=-1)
    freqs_cos, freqs_sin = freqs.chunk(2, dim=-1)

    rotated_x0 = x0 * freqs_cos - x1 * freqs_sin
    rotated_x1 = x1 * freqs_cos + x0 * freqs_sin
    x_rotated = torch.cat((rotated_x0, rotated_x1), dim=-1)

    return x_rotated


def rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int | None = None,
    num_state_per_block: int | None = None,
) -> torch.Tensor:
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(x.to(torch.float64).reshape(B, seq_len, n, -1, 2))

    if action_register_length is not None:
        assert num_action_per_block is not None
        assert num_state_per_block is not None

        chunk_size = action_register_length // (num_action_per_block + num_state_per_block)

        freqs_1d_action = freqs_action[: chunk_size * num_action_per_block].view(
            chunk_size * num_action_per_block, 1, -1
        )
        freqs_1d_state = freqs_state[: chunk_size * num_state_per_block].view(
            chunk_size * num_state_per_block, 1, -1
        )
        freqs = torch.cat([freqs, freqs_1d_action, freqs_1d_state], dim=0)

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)
    return x


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, freqs), k=rope_apply(k, freqs), v=v, k_lens=seq_lens, window_size=self.window_size
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanGanCrossAttention(WanSelfAttention):
    def forward(self, x, context, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        qq = self.norm_q(self.q(context)).view(b, 1, -1, d)

        kk = self.norm_k(self.k(x)).view(b, -1, n, d)
        vv = self.v(x).view(b, -1, n, d)

        # compute attention
        x = flash_attention(qq, kk, vv)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            # compute query, key, value
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
        x = flash_attention(q, k, v, k_lens=None)

        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


WAN_CROSSATTENTION_CLASSES = {"t2v_cross_attn": WanT2VCrossAttention, "i2v_cross_attn": WanI2VCrossAttention}
