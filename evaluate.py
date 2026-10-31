#!/usr/bin/env python3
"""Evaluate a FeatherLM checkpoint on held-out text.

Reports validation loss, perplexity, bits-per-byte (tokenizer independent, so it
compares models with different vocabularies) and next-token top-1/top-5 accuracy.

    python evaluate.py --model runs/featherlm-nano/final --data data/shakespeare
    python evaluate.py --model runs/featherlm-nano/final --text-file my_heldout.txt

Developed by Ammad Aslam with AI-assisted tooling.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from featherlm.training.checkpoint import load_model


@torch.no_grad()
def evaluate(model, tok, ids: np.ndarray, seq_len: int, batch_size: int = 8, max_windows: int = 0) -> dict:
    model.eval()
    n_win = (len(ids) - 1) // seq_len
    if max_windows:
        n_win = min(n_win, max_windows)
    if n_win < 1:
        raise ValueError("not enough tokens for one window")
    tot_nll, tot_tok, top1, top5, tot_bytes = 0.0, 0, 0, 0, 0
    for b in range(0, n_win, batch_size):
        wins = [ids[i * seq_len : i * seq_len + seq_len + 1] for i in range(b, min(b + batch_size, n_win))]
        t = torch.from_numpy(np.stack(wins).astype(np.int64))
        x, y = t[:, :-1], t[:, 1:]
        logits = model(x)[0].float()
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        tot_nll += nll.item()
        tot_tok += y.numel()
        top = logits.topk(5, dim=-1).indices
        top1 += (top[..., 0] == y).sum().item()
        top5 += (top == y[..., None]).any(-1).sum().item()
        tot_bytes += sum(len(tok.decode_bytes(row.tolist(), skip_special=False)) for row in y)
    loss = tot_nll / tot_tok
    return {"tokens": tot_tok, "val_loss": round(loss, 4), "perplexity": round(math.exp(loss), 3),
            "bits_per_byte": round(tot_nll / (math.log(2) * tot_bytes), 4),
            "top1_accuracy": round(top1 / tot_tok, 4), "top5_accuracy": round(top5 / tot_tok, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="prepared dataset dir (uses val.bin)")
    src.add_argument("--text-file", help="raw held-out text")
    ap.add_argument("--seq-len", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--out", help="write JSON here")
    a = ap.parse_args()
    model, tok, meta = load_model(a.model)
    seq = a.seq_len or model.cfg.max_seq_len
    if a.data:
        ids = np.fromfile(Path(a.data) / "val.bin", dtype=np.uint16)
    else:
        ids = np.array(tok.encode(Path(a.text_file).read_text(encoding="utf-8")), dtype=np.int64)
    res = {"model": a.model, "architecture": model.cfg.backend, "parameters": model.num_params(),
           "quantization": meta.get("quantization", "none"), **evaluate(model, tok, ids, seq, max_windows=a.max_windows)}
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
