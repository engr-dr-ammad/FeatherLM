"""Checkpoint save/restore.

Model weights are stored with safetensors (no pickle, safe to load from untrusted
sources). Optimizer state, needed only to resume local training, is stored with
torch.save and loaded with weights_only=True; never accept optimizer files from
untrusted users.

Layout of a checkpoint directory:
    config.json         model config
    model.safetensors   weights
    tokenizer.json      tokenizer
    metadata.json       name, parameter count, quantization, provenance
    trainer_state.json  step, best val loss, RNG state (training checkpoints only)
    optimizer.pt        optimizer + scheduler state (training checkpoints only)
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..model.config import FeatherConfig
from ..model.model import FeatherLM
from ..tokenizer.tokenizer import ByteBPETokenizer

AUTHOR = "Developed by Ammad Aslam with AI-assisted tooling."


def _atomic_dir_write(final: Path, writer) -> None:
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp", dir=final.parent))
    try:
        writer(tmp)
        if final.exists():
            old = final.with_name(f".{final.name}.old{os.getpid()}")
            final.rename(old)
            tmp.rename(final)
            shutil.rmtree(old, ignore_errors=True)
        else:
            tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def save_model(path: str | Path, model: FeatherLM, tok: ByteBPETokenizer, extra_meta: dict | None = None,
               optimizer=None, scheduler=None, trainer_state: dict | None = None,
               state: dict | None = None, extra_files: dict | None = None) -> Path:
    """Write a checkpoint directory atomically. `state` overrides model.state_dict()
    (used to save merged LoRA weights); `extra_files` maps filename -> tensor dict (safetensors)."""
    path = Path(path)
    if state is None:
        state = model.state_dict()
    state = {k: v.detach().contiguous().cpu() for k, v in state.items()}

    def writer(d: Path) -> None:
        model.cfg.save(d / "config.json")
        save_file(state, str(d / "model.safetensors"), metadata={"format": "pt", "featherlm": "1"})
        tok.save(d / "tokenizer.json")
        meta = {
            "name": model.cfg.name,
            "architecture": model.cfg.backend,
            "parameters": model.num_params(),
            "quantization": "none",
            "context_length": model.cfg.max_seq_len,
            "chat_template": model.cfg.chat_template,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "credits": AUTHOR,
        }
        meta.update(extra_meta or {})
        (d / "metadata.json").write_text(json.dumps(meta, indent=2))
        if trainer_state is not None:
            (d / "trainer_state.json").write_text(json.dumps(trainer_state, indent=2))
        for fname, tensors in (extra_files or {}).items():
            save_file(tensors, str(d / fname))
        if optimizer is not None:
            torch.save({"optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None},
                       d / "optimizer.pt")

    _atomic_dir_write(path, writer)
    return path


def load_model(path: str | Path, device: str = "cpu", dtype: torch.dtype | None = None):
    """Load (model, tokenizer, metadata). Handles quantized checkpoints transparently."""
    path = Path(path)
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{path} is not a FeatherLM checkpoint (missing config.json)")
    cfg = FeatherConfig.load(path / "config.json")
    meta = json.loads((path / "metadata.json").read_text()) if (path / "metadata.json").exists() else {}
    tok = ByteBPETokenizer.load(path / "tokenizer.json")
    model = FeatherLM(cfg)
    q = meta.get("quantization", "none")
    if q != "none":
        from ..export.quantize import load_quantized_into

        load_quantized_into(model, path / "model.safetensors", q)
    else:
        state = load_file(str(path / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    if dtype is not None:
        model = model.to(dtype)
    model.to(device).eval()
    if tok.vocab_size > cfg.vocab_size:
        raise ValueError("tokenizer vocab larger than model vocab")
    return model, tok, meta


def load_training_state(path: str | Path, optimizer, scheduler=None) -> dict:
    path = Path(path)
    state = json.loads((path / "trainer_state.json").read_text())
    opt_file = path / "optimizer.pt"
    if opt_file.exists():
        blob = torch.load(opt_file, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(blob["optimizer"])
        if scheduler is not None and blob.get("scheduler") is not None:
            scheduler.load_state_dict(blob["scheduler"])
    return state


def encode_rng(t: torch.Tensor) -> str:
    return base64.b64encode(t.numpy().tobytes()).decode()


def decode_rng(s: str) -> torch.Tensor:
    import numpy as np

    return torch.from_numpy(np.frombuffer(base64.b64decode(s), dtype=np.uint8).copy())


def safe_child(root: str | Path, name: str) -> Path:
    """Resolve `name` under `root`, refusing path traversal and absolute paths."""
    root = Path(root).resolve()
    if not name or "\x00" in name or Path(name).is_absolute():
        raise ValueError("invalid path")
    p = (root / name).resolve()
    if p != root and root not in p.parents:
        raise ValueError("path escapes the allowed directory")
    return p
