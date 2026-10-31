"""Optimizers and learning-rate schedule."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Decay matrices (linear weights); skip norms, biases, embeddings, decay vectors, conv taps.
        if p.ndim == 2 and "table" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def build_optimizer(model: nn.Module, name: str = "adamw", lr: float = 3e-4, weight_decay: float = 0.1,
                    betas=(0.9, 0.95)) -> torch.optim.Optimizer:
    name = name.lower()
    groups = param_groups(model, weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, foreach=False)
    if name == "adafactor":
        # Memory-efficient: factorised second moments, no first moment.
        return torch.optim.Adafactor(groups, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)
    raise ValueError(f"unknown optimizer {name!r} (adamw | adafactor | sgd)")


def optimizer_state_bytes(opt: torch.optim.Optimizer) -> int:
    total = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


def lr_lambda_factory(warmup: int, total: int, min_ratio: float = 0.1):
    """Linear warmup then cosine decay to min_ratio * peak lr."""

    def f(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if total <= warmup:
            return 1.0
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    return f


def build_scheduler(opt, warmup: int, total: int, min_ratio: float = 0.1):
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(warmup, total, min_ratio))
