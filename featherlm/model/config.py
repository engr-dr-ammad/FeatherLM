"""Model configuration for FeatherLM.

Developed by Ammad Aslam with AI-assisted tooling.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

BACKENDS = ("transformer", "feather")


@dataclass
class FeatherConfig:
    """Hyper-parameters for both FeatherLM backends.

    backend="transformer": efficient Transformer baseline (GQA + RoPE + RMSNorm + SwiGLU).
    backend="feather":     experimental core: gated linear recurrence blocks with
                           depth-wise parameter sharing and sparse local attention.
    """

    name: str = "featherlm-nano"
    backend: str = "feather"
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 6
    # Attention
    n_heads: int = 6
    n_kv_heads: int = 2
    rope_theta: float = 10000.0
    max_seq_len: int = 256
    # Feed-forward (SwiGLU). If 0, computed as ~8/3 * d_model rounded to 64.
    d_ff: int = 0
    # Embeddings
    embed_dim: int = 0  # 0 -> same as d_model; smaller -> factorized embedding
    tie_embeddings: bool = True
    # Regularisation / numerics
    dropout: float = 0.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    # Feather backend specific
    n_unique_blocks: int = 0  # 0 -> no sharing (one block per layer)
    attn_every: int = 3  # every k-th layer uses local attention (0 disables)
    local_window: int = 64
    conv_kernel: int = 4
    scan_chunk: int = 16  # must be <= 20 (see recurrent.py: LOG_A_MIN)
    # Memory
    gradient_checkpointing: bool = False
    # Chat template name if the model was instruction-tuned (None = plain LM)
    chat_template: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.d_ff == 0:
            self.d_ff = int(((8 * self.d_model / 3) + 63) // 64 * 64)
        if self.embed_dim == 0:
            self.embed_dim = self.d_model
        self.validate()

    # ------------------------------------------------------------------
    def validate(self) -> None:
        errs: list[str] = []
        if self.backend not in BACKENDS:
            errs.append(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        for k in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "max_seq_len", "d_ff"):
            if getattr(self, k) <= 0:
                errs.append(f"{k} must be > 0")
        if self.d_model % self.n_heads != 0:
            errs.append("d_model must be divisible by n_heads")
        elif (self.d_model // self.n_heads) % 2 != 0:
            errs.append("head_dim must be even for rotary embeddings")
        if self.n_kv_heads > 0 and self.n_heads % self.n_kv_heads != 0:
            errs.append("n_heads must be divisible by n_kv_heads")
        if self.embed_dim > self.d_model:
            errs.append("embed_dim must be <= d_model")
        if not 0.0 <= self.dropout < 1.0:
            errs.append("dropout must be in [0, 1)")
        if self.n_unique_blocks < 0 or self.n_unique_blocks > self.n_layers:
            errs.append("n_unique_blocks must be in [0, n_layers]")
        if self.attn_every < 0:
            errs.append("attn_every must be >= 0")
        if self.local_window <= 0 or self.conv_kernel <= 0:
            errs.append("local_window and conv_kernel must be > 0")
        if not 1 <= self.scan_chunk <= 20:
            errs.append("scan_chunk must be in [1, 20] for numerically safe chunked scans")
        if errs:
            raise ValueError("Invalid FeatherConfig: " + "; ".join(errs))

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def layer_kinds(self) -> list[str]:
        """Return the mixer kind for each layer ("attn", "local_attn" or "rec")."""
        if self.backend == "transformer":
            return ["attn"] * self.n_layers
        kinds = []
        for i in range(self.n_layers):
            if self.attn_every and (i + 1) % self.attn_every == 0:
                kinds.append("local_attn")
            else:
                kinds.append("rec")
        return kinds

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FeatherConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeatherConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
