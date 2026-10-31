"""Gated linear recurrence: the sequence mixer of the experimental Feather core.

Established ideas this builds on (not claimed as new):
  * Real-gated linear recurrent unit (RG-LRU) from Griffin/Hawk (De et al., 2024):
      a_t = exp(-c * softplus(L) * r_t),  h_t = a_t * h_{t-1} + sqrt(1 - a_t^2) * (i_t * u_t)
  * Short causal depthwise convolution before the recurrence (H3, Mamba, Griffin).
  * Multiplicative output gating (GLU-style).

Implementation choices specific to FeatherLM (hypotheses, to be evaluated):
  * Low-rank gate projections (d -> d/4 -> 2d) instead of full/block-diagonal ones,
    to cut parameters of the recurrence gates by ~4x.
  * A chunked log-space scan that keeps activation memory O(T * d) (no T x T or
    C x C x d intermediates), made numerically safe by bounding log a_t from below.

The recurrence is causal by construction: h_t depends only on inputs at times <= t.
A slow, obviously-correct sequential reference (`reference_scan`) is provided and
tested against the chunked implementation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FeatherConfig

# Lower bound on log(a_t). With chunks of <= 20 steps the largest exponent in the
# chunked scan is 20 * 4 = 80 < log(float32 max) ~= 88.7, so nothing overflows.
LOG_A_MIN = -4.0
GATE_C = 8.0


def reference_scan(log_a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
    """Sequential h_t = exp(log_a_t) * h_{t-1} + b_t. Shapes (B, T, D)."""
    B, T, D = b.shape
    h = torch.zeros(B, D, dtype=b.dtype, device=b.device) if h0 is None else h0
    out = []
    for t in range(T):
        h = torch.exp(log_a[:, t]) * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def chunked_scan(log_a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None, chunk: int = 16) -> torch.Tensor:
    """Same result as `reference_scan`, vectorised within chunks.

    Within a chunk starting at step 0 with carried state h0:
        A_t = sum_{r<=t} log_a_r
        h_t = exp(A_t) * (h0 + sum_{s<=t} exp(-A_s) * b_s)
    Requires log_a >= LOG_A_MIN and chunk <= 20 for float32 safety.
    """
    B, T, D = b.shape
    h = torch.zeros(B, D, dtype=b.dtype, device=b.device) if h0 is None else h0
    outs = []
    for s in range(0, T, chunk):
        la = log_a[:, s : s + chunk].float()
        bc = b[:, s : s + chunk].float()
        A = torch.cumsum(la, dim=1)
        acc = torch.cumsum(torch.exp(-A) * bc, dim=1)
        hc = torch.exp(A) * (h.float().unsqueeze(1) + acc)
        hc = hc.to(b.dtype)
        outs.append(hc)
        h = hc[:, -1]
    return torch.cat(outs, dim=1)


class GatedRecurrence(nn.Module):
    """Conv -> RG-LRU-style gated linear recurrence -> gated output projection."""

    def __init__(self, cfg: FeatherConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.d = d
        self.K = cfg.conv_kernel
        self.chunk = cfg.scan_chunk
        rank = max(16, d // 4)
        self.in_proj = nn.Linear(d, 2 * d, bias=False)  # u (recurrent branch) and g (gate branch)
        self.conv_weight = nn.Parameter(torch.zeros(d, 1, self.K))
        self.conv_bias = nn.Parameter(torch.zeros(d))
        self.gate_down = nn.Linear(d, rank, bias=False)
        self.gate_up = nn.Linear(rank, 2 * d, bias=True)  # r_t (recurrence gate) and i_t (input gate)
        self.decay = nn.Parameter(torch.zeros(d))  # Lambda
        self.out_proj = nn.Linear(d, d, bias=False)
        self.reset_recurrent_parameters()

    def reset_recurrent_parameters(self) -> None:
        # Initialise decay so a = exp(-c * softplus(L)) is uniform in [0.9, 0.999] at r = 1.
        a = torch.empty(self.d).uniform_(0.9, 0.999)
        sp = -torch.log(a) / GATE_C  # softplus(L)
        self.decay.data.copy_(torch.log(torch.expm1(sp)))  # inverse softplus
        nn.init.normal_(self.conv_weight, std=1.0 / math.sqrt(self.K))
        self.conv_weight.data[:, :, -1] += 1.0  # start close to identity
        nn.init.zeros_(self.conv_bias)

    # ------------------------------------------------------------------
    def _conv(self, u: torch.Tensor, conv_state: torch.Tensor | None):
        """Causal depthwise conv over time. u: (B, T, d). conv_state: (B, K-1, d)."""
        B, T, d = u.shape
        if conv_state is None:
            conv_state = u.new_zeros(B, self.K - 1, d)
        padded = torch.cat([conv_state, u], dim=1)  # (B, T+K-1, d)
        y = F.conv1d(padded.transpose(1, 2), self.conv_weight, self.conv_bias, groups=d).transpose(1, 2)
        new_state = padded[:, padded.shape[1] - (self.K - 1) :] if self.K > 1 else padded[:, :0]
        return y, new_state

    def gates(self, u: torch.Tensor):
        r, i = torch.sigmoid(self.gate_up(self.gate_down(u))).chunk(2, dim=-1)
        log_a = -GATE_C * F.softplus(self.decay) * r
        log_a = log_a.clamp(min=LOG_A_MIN)
        a2 = torch.exp(2 * log_a)
        b = torch.sqrt((1 - a2).clamp(min=1e-6)) * (i * u)
        return log_a, b

    def forward(self, x: torch.Tensor, state: dict | None = None, use_reference: bool = False):
        u, g = self.in_proj(x).chunk(2, dim=-1)
        conv_state = state.get("conv") if state else None
        h0 = state.get("h") if state else None
        u, new_conv = self._conv(u, conv_state)
        log_a, b = self.gates(u)
        if use_reference or x.shape[1] == 1:
            h = reference_scan(log_a, b, h0)
        else:
            h = chunked_scan(log_a, b, h0, self.chunk)
        y = self.out_proj(h * F.gelu(g))
        new_state = {"h": h[:, -1], "conv": new_conv} if state is not None else None
        return y, new_state
