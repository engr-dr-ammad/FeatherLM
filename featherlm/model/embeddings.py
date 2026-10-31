"""Token embeddings and the vocabulary projection.

Supports weight tying and factorized (ALBERT-style) embeddings: a V x e table
followed by an e -> d projection, where e < d. With tying, the output head
reuses both matrices, so the head adds no extra parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FeatherConfig


class TokenEmbedding(nn.Module):
    def __init__(self, cfg: FeatherConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.table = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.factorized = cfg.embed_dim < cfg.d_model
        self.up = nn.Linear(cfg.embed_dim, cfg.d_model, bias=False) if self.factorized else None
        if not cfg.tie_embeddings:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.head = None

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.table(idx)
        if self.up is not None:
            x = self.up(x)
        return x

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        if self.head is not None:
            return self.head(h)
        if self.up is not None:
            h = F.linear(h, self.up.weight.t())  # d -> e using the transposed up-projection
        return F.linear(h, self.table.weight)


def embedding_param_report(cfg: FeatherConfig) -> dict[str, int]:
    """Analytic parameter counts for the embedding/head choices (for documentation)."""
    v, d, e = cfg.vocab_size, cfg.d_model, cfg.embed_dim
    untied_full = 2 * v * d
    tied_full = v * d
    tied_factorized = v * e + e * d
    return {
        "untied_full": untied_full,
        "tied_full": tied_full,
        "tied_factorized": tied_factorized,
        "this_config": (v * e + (e * d if e < d else 0) + (0 if cfg.tie_embeddings else v * d)),
    }
