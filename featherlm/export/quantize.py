"""Post-training weight quantization (FP16, INT8 per-channel, INT4 group-wise).

Weight-only, symmetric quantization; activations stay in floating point. This is
an inference optimisation only: it says nothing about training efficiency.

    python -m featherlm.export.quantize --model runs/featherlm-nano/final --mode int8 --out models/featherlm-nano-int8

Two runtime options when loading a quantized checkpoint:
  * dequantize (default): weights are expanded to FP32 at load time. Smaller files,
    same speed and RAM as FP32 at runtime.
  * packed: linear layers keep int8 / packed-int4 weights and dequantize per call
    (lower resident RAM, slower in pure PyTorch on CPU).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MODES = ("fp16", "int8", "int4")
INT4_GROUP = 32


def _quantizable(name: str, t: torch.Tensor) -> bool:
    return t.ndim == 2 and t.dtype == torch.float32 and t.shape[1] >= INT4_GROUP and t.numel() >= 4096


def quantize_int8(w: torch.Tensor):
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(1).to(torch.float32)


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale[:, None]


def quantize_int4(w: torch.Tensor, group: int = INT4_GROUP):
    out_f, in_f = w.shape
    if in_f % group != 0:
        raise ValueError(f"in_features {in_f} not divisible by group {group}")
    g = w.reshape(out_f, in_f // group, group)
    scale = g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(g / scale), -8, 7).to(torch.int8).reshape(out_f, in_f)
    u = (q + 8).to(torch.uint8)  # 0..15
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
    return packed, scale.squeeze(2).to(torch.float16)


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, group: int = INT4_GROUP) -> torch.Tensor:
    lo = (packed & 0x0F).to(torch.int8) - 8
    hi = (packed >> 4).to(torch.int8) - 8
    q = torch.stack([lo, hi], dim=2).reshape(packed.shape[0], packed.shape[1] * 2)
    out_f, in_f = q.shape
    w = q.reshape(out_f, in_f // group, group).to(torch.float32) * scale.to(torch.float32)[:, :, None]
    return w.reshape(out_f, in_f)


def quantize_state_dict(state: dict[str, torch.Tensor], mode: str) -> tuple[dict[str, torch.Tensor], dict]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    out: dict[str, torch.Tensor] = {}
    quantized: list[str] = []
    for name, t in state.items():
        t = t.detach().cpu()
        if mode == "fp16":
            out[name] = t.to(torch.float16) if t.is_floating_point() else t
            if t.is_floating_point():
                quantized.append(name)
        elif _quantizable(name, t):
            if mode == "int8":
                q, s = quantize_int8(t)
            else:
                q, s = quantize_int4(t)
            out[name + ".qweight"] = q
            out[name + ".qscale"] = s
            quantized.append(name)
        else:
            out[name] = t.contiguous()
    info = {"mode": mode, "group_size": INT4_GROUP if mode == "int4" else None,
            "quantized_tensors": len(quantized), "scheme": "symmetric weight-only"}
    return {k: v.contiguous() for k, v in out.items()}, info


def dequantize_state_dict(qstate: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
    out = {}
    for name, t in qstate.items():
        if name.endswith(".qscale"):
            continue
        if name.endswith(".qweight"):
            base = name[: -len(".qweight")]
            s = qstate[base + ".qscale"]
            out[base] = dequantize_int8(t, s) if mode == "int8" else dequantize_int4(t, s)
        else:
            out[name] = t.to(torch.float32) if t.is_floating_point() else t
    return out


class QuantLinear(nn.Module):
    """Linear layer that stores quantized weights and dequantizes on each forward."""

    def __init__(self, qweight: torch.Tensor, qscale: torch.Tensor, mode: str, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.mode = mode
        self.register_buffer("qweight", qweight)
        self.register_buffer("qscale", qscale)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "int8":
            w = dequantize_int8(self.qweight, self.qscale)
        else:
            w = dequantize_int4(self.qweight, self.qscale)
        return F.linear(x, w.to(x.dtype), self.bias)


def load_quantized_into(model: nn.Module, path: str | Path, mode: str, packed: bool = False) -> nn.Module:
    qstate = load_file(str(path))
    if mode == "fp16" or not packed:
        missing, unexpected = model.load_state_dict(dequantize_state_dict(qstate, mode), strict=False)
        if missing or unexpected:
            raise ValueError(f"quantized checkpoint mismatch: missing={missing} unexpected={unexpected}")
        return model
    # Packed runtime: swap nn.Linear modules whose weights were quantized.
    plain = {k: v for k, v in dequantize_state_dict(
        {k: v for k, v in qstate.items() if not k.endswith((".qweight", ".qscale"))}, mode).items()}
    for mod_name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if isinstance(child, nn.Linear) and f"{full}.weight.qweight" in qstate:
                bias = qstate.get(f"{full}.bias")
                setattr(mod, child_name, QuantLinear(qstate[f"{full}.weight.qweight"], qstate[f"{full}.weight.qscale"],
                                                     mode, bias))
    # Remaining quantized tensors (e.g. embedding table) are dequantized.
    for k in qstate:
        if k.endswith(".qweight"):
            base = k[: -len(".qweight")]
            mod_path = base.rsplit(".", 1)[0]
            try:
                sub = model.get_submodule(mod_path)
            except AttributeError:
                sub = None
            if not isinstance(sub, QuantLinear):
                plain[base] = dequantize_int8(qstate[k], qstate[base + ".qscale"]) if mode == "int8" else \
                    dequantize_int4(qstate[k], qstate[base + ".qscale"])
    model.load_state_dict(plain, strict=False)
    return model


def quantize_checkpoint(src: str | Path, dst: str | Path, mode: str) -> dict:
    src, dst = Path(src), Path(dst)
    meta = json.loads((src / "metadata.json").read_text())
    if meta.get("quantization", "none") != "none":
        raise ValueError("source checkpoint is already quantized")
    state = load_file(str(src / "model.safetensors"))
    qstate, info = quantize_state_dict(state, mode)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "tokenizer.json"):
        shutil.copy2(src / f, dst / f)
    save_file(qstate, str(dst / "model.safetensors"), metadata={"format": "pt", "featherlm_quant": mode})
    meta.update({"quantization": mode, "quantization_info": info, "quantized_from": str(src),
                 "name": f"{meta.get('name', 'featherlm')}-{mode}"})
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2))
    sizes = {"source_bytes": (src / "model.safetensors").stat().st_size,
             "quantized_bytes": (dst / "model.safetensors").stat().st_size}
    sizes["ratio"] = round(sizes["quantized_bytes"] / sizes["source_bytes"], 4)
    return {**info, **sizes}


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize a FeatherLM checkpoint")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(json.dumps(quantize_checkpoint(args.model, args.out, args.mode), indent=2))


if __name__ == "__main__":
    main()
