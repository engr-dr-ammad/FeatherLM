"""Causal self-attention with grouped-query attention, RoPE and a KV cache.

The same module serves the Transformer baseline (global causal attention) and
the Feather backend's sparse local layers (sliding-window attention, whose KV
cache is truncated to the window so inference memory stays bounded).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FeatherConfig


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]  # (T, hd/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, hd). Rotates pairs (x[..., :hd/2], x[..., hd/2:])."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None]
    sin = sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: FeatherConfig, window: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = cfg.head_dim
        self.window = window  # None -> global causal attention
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.hd, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, start_pos: int = 0, cache: dict | None = None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv, self.hd)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)

        q_pos = torch.arange(start_pos, start_pos + T, device=x.device)
        cos, sin = rope_cos_sin(q_pos, self.hd, self.cfg.rope_theta, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        new_cache = None
        k_start = start_pos
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
                k_start = cache["start"]
            new_cache = {"k": k, "v": v, "start": k_start}

        rep = self.n_heads // self.n_kv
        kr = k.repeat_interleave(rep, dim=1) if rep > 1 else k
        vr = v.repeat_interleave(rep, dim=1) if rep > 1 else v

        drop = self.dropout if self.training else 0.0
        if self.window is None and k_start == start_pos and kr.shape[2] == T:
            out = F.scaled_dot_product_attention(q, kr, vr, is_causal=True, dropout_p=drop)
        else:
            k_pos = torch.arange(k_start, k_start + kr.shape[2], device=x.device)
            rel = q_pos[:, None] - k_pos[None, :]
            mask = rel >= 0
            if self.window is not None:
                mask = mask & (rel < self.window)
            out = F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask, dropout_p=drop)

        if new_cache is not None and self.window is not None:
            keep = self.window - 1  # the next query needs at most window-1 past keys
            if new_cache["k"].shape[2] > keep:
                drop_n = new_cache["k"].shape[2] - keep
                new_cache = {
                    "k": new_cache["k"][:, :, drop_n:],
                    "v": new_cache["v"][:, :, drop_n:],
                    "start": new_cache["start"] + drop_n,
                }

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.hd)
        return self.o_proj(out), new_cache
