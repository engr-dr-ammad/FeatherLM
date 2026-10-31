"""FeatherLM: decoder-only autoregressive language model with two backends."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import Block, RMSNorm
from .config import FeatherConfig
from .embeddings import TokenEmbedding

IGNORE_INDEX = -100


class FeatherLM(nn.Module):
    def __init__(self, cfg: FeatherConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbedding(cfg)
        self.drop = nn.Dropout(cfg.dropout)

        # Depth-wise parameter sharing: per mixer kind, layers cycle through
        # min(count, n_unique_blocks) unique blocks (0 disables sharing).
        kinds = cfg.layer_kinds()
        self.blocks = nn.ModuleList()
        unique: dict[str, list[int]] = {}
        seen: dict[str, int] = {}
        self.layer_to_block: list[int] = []
        for kind in kinds:
            j = seen.get(kind, 0)
            seen[kind] = j + 1
            pool = unique.setdefault(kind, [])
            limit = cfg.n_unique_blocks or cfg.n_layers
            if len(pool) < limit:
                self.blocks.append(Block(cfg, kind))
                pool.append(len(self.blocks) - 1)
            self.layer_to_block.append(pool[j % len(pool)])
        self.layer_kinds = kinds

        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.apply(self._init_weights)
        for blk in self.blocks:
            if hasattr(blk.mixer, "reset_recurrent_parameters"):
                blk.mixer.reset_recurrent_parameters()
        # Scaled init for residual output projections (GPT-2 style)
        scale = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("out_proj.weight") or name.endswith("ffn.down.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    # ------------------------------------------------------------------
    def num_params(self, trainable_only: bool = False) -> int:
        """Unique parameters (shared blocks counted once)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def param_breakdown(self) -> dict[str, int]:
        out = {"embedding": sum(p.numel() for p in self.embed.parameters()), "final_norm": self.norm.weight.numel()}
        for kind in set(self.layer_kinds):
            out[f"blocks_{kind}"] = sum(
                p.numel() for i, b in enumerate(self.blocks) if b.kind == kind for p in b.parameters()
            )
        out["total"] = self.num_params()
        out["unique_blocks"] = len(self.blocks)
        out["layers"] = self.cfg.n_layers
        return out

    def new_cache(self) -> list[dict]:
        return [{} for _ in range(self.cfg.n_layers)]

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        start_pos: int = 0,
        caches: list[dict] | None = None,
        last_only: bool = False,
    ):
        """idx: (B, T) token ids. targets: (B, T) with IGNORE_INDEX for masked positions.

        Returns (logits, loss, caches). With `caches` (from `new_cache()`), returns the
        updated per-layer inference state for incremental decoding.
        """
        x = self.drop(self.embed(idx))
        new_caches = [] if caches is not None else None
        use_ckpt = self.cfg.gradient_checkpointing and self.training and caches is None
        for layer, bi in enumerate(self.layer_to_block):
            blk = self.blocks[bi]
            cache = caches[layer] if caches is not None else None
            if use_ckpt:
                x, c = checkpoint(blk, x, start_pos, cache, use_reentrant=False)
            else:
                x, c = blk(x, start_pos, cache)
            if new_caches is not None:
                new_caches.append(c)
        x = self.norm(x)
        if last_only:
            x = x[:, -1:]
        logits = self.embed.logits(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=IGNORE_INDEX
            )
        return logits, loss, new_caches
