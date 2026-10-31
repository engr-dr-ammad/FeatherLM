"""ONNX export.

Exports a fixed-window "full forward" graph: input_ids (1, T) -> logits (1, T, V),
with T = model max_seq_len. Because the model is strictly causal, a shorter prompt
can be right-padded and the logits read at position len-1 (padding never
influences earlier positions; this is tested in tests/test_export.py).

Why fixed T: the Feather backend's chunked scan is a Python loop over chunks; under
tracing it is unrolled for a specific T. A fixed window keeps one graph that works
in ONNX Runtime (desktop and mobile) for both backends.

Known limitation: each generated token re-runs the full window (O(T) per token).
A stateful single-step export (recurrent state / KV cache as graph inputs) is the
next optimisation for mobile.

    python -m featherlm.export.onnx_export --model runs/featherlm-nano/final --out models/featherlm-nano-onnx
"""
from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from ..training.checkpoint import load_model


class LogitsWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(input_ids)
        return logits


def export_onnx(model_dir: str | Path, out_dir: str | Path, opset: int = 17) -> Path:
    model, tok, meta = load_model(model_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T = model.cfg.max_seq_len
    dummy = torch.zeros(1, T, dtype=torch.long)
    path = out / "model.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(LogitsWrapper(model).eval(), (dummy,), str(path), input_names=["input_ids"],
                          output_names=["logits"], opset_version=opset, dynamo=False)
    shutil.copy2(Path(model_dir) / "tokenizer.json", out / "tokenizer.json")
    info = {
        "name": meta.get("name"),
        "architecture": model.cfg.backend,
        "window": T,
        "vocab_size": model.cfg.vocab_size,
        "pad_id": tok.pad_id,
        "eos_id": tok.eos_id,
        "opset": opset,
        "usage": "right-pad input_ids to window, read logits[0, len-1]",
        "credits": "Developed by Ammad Aslam with AI-assisted tooling.",
    }
    (out / "onnx_config.json").write_text(json.dumps(info, indent=2))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export FeatherLM to ONNX")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    p = export_onnx(args.model, args.out, args.opset)
    print(f"exported {p} ({p.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
