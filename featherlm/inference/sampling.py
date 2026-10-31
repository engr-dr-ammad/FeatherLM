"""Logit processing and token sampling."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.8  # 0 -> greedy
    top_k: int = 0  # 0 -> disabled
    top_p: float = 1.0  # 1 -> disabled
    repetition_penalty: float = 1.0  # 1 -> disabled
    stop: list[str] = field(default_factory=list)
    seed: int | None = None

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")


def apply_repetition_penalty(logits: torch.Tensor, prev_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """CTRL-style penalty. logits: (V,), prev_ids: (N,)."""
    if penalty == 1.0 or prev_ids.numel() == 0:
        return logits
    ids = torch.unique(prev_ids)
    vals = logits[ids]
    logits = logits.clone()
    logits[ids] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    remove = cum - probs > p  # keep the smallest prefix whose mass >= p (always keeps the top token)
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_idx, sorted_logits)


def sample_next(
    logits: torch.Tensor, prev_ids: torch.Tensor, params: SamplingParams, generator: torch.Generator | None = None
) -> int:
    """logits: (V,) for the last position. Returns a token id."""
    logits = logits.float()
    logits = apply_repetition_penalty(logits, prev_ids, params.repetition_penalty)
    if params.temperature == 0:
        return int(torch.argmax(logits))
    logits = logits / params.temperature
    logits = top_k_filter(logits, params.top_k)
    logits = top_p_filter(logits, params.top_p)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
