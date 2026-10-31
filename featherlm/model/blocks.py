"""Shared building blocks: RMSNorm, SwiGLU feed-forward and the residual block."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .config import FeatherConfig
from .recurrent import GatedRecurrence


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.gate_up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(g) * u))


class Block(nn.Module):
    """Pre-norm residual block: x + mixer(norm(x)); x + ffn(norm(x))."""

    def __init__(self, cfg: FeatherConfig, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        if kind == "attn":
            self.mixer = CausalSelfAttention(cfg, window=None)
        elif kind == "local_attn":
            self.mixer = CausalSelfAttention(cfg, window=cfg.local_window)
        elif kind == "rec":
            self.mixer = GatedRecurrence(cfg)
        else:
            raise ValueError(f"unknown block kind {kind}")
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, start_pos: int = 0, cache: dict | None = None):
        if self.kind == "rec":
            h, new_cache = self.mixer(self.norm1(x), cache)
        else:
            h, new_cache = self.mixer(self.norm1(x), start_pos, cache)
        x = x + self.drop(h)
        x = x + self.ffn(self.norm2(x))
        return x, new_cache
