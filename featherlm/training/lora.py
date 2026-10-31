"""LoRA (Hu et al., 2021) adapters for FeatherLM linear layers."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("q_proj", "kv_proj", "o_proj", "in_proj", "out_proj", "gate_up", "down")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    @property
    def weight(self):  # allows code that inspects .weight to keep working
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scale

    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.data = self.base.weight.data + (self.lora_B @ self.lora_A) * self.scale
        if self.base.bias is not None:
            lin.bias.data = self.base.bias.data.clone()
        return lin


def apply_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0,
               targets=DEFAULT_TARGETS) -> int:
    """Freeze the model and wrap target linear layers with LoRA. Returns #trainable params."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r, alpha, dropout))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_lora(model: nn.Module) -> nn.Module:
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged())
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if "lora_" in k}


def merged_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict with LoRA folded into base weights and plain-model key names (non-destructive)."""
    out = {}
    lora_mods = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    for k, v in model.state_dict().items():
        if "lora_" in k:
            continue
        if ".base." in k:
            mod_name = k.split(".base.")[0]
            m = lora_mods[mod_name]
            new_k = k.replace(".base.", ".")
            if k.endswith("weight"):
                v = v + (m.lora_B @ m.lora_A).detach() * m.scale
            out[new_k] = v.detach()
        else:
            out[k] = v.detach()
    return out
