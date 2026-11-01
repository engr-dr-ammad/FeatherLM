#!/usr/bin/env python3
"""Reference implementation of the mobile inference loop, using only numpy + onnxruntime.

It mirrors FeatherModel.kt / FeatherModel.swift step for step (tokenizer from JSON,
right-padded fixed window, temperature/top-k sampling), so the exported model and the
mobile algorithm can be tested on a desktop before building the apps.

    python mobile/reference_runtime.py --onnx models/featherlm-nano-onnx --prompt "ROMEO:" --tokens 40
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort

PATTERN = r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""


class MiniTokenizer:
    """Dependency-free byte-level BPE (same algorithm as the Kotlin/Swift ports)."""

    def __init__(self, path: Path):
        d = json.loads(path.read_text())
        self.n_special = len(d["special_tokens"])
        self.ranks = {(a, b): i for i, (a, b) in enumerate(d["merges"])}
        self.vocab = [t.encode() for t in d["special_tokens"]] + [bytes([i]) for i in range(256)]
        for a, b in d["merges"]:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.pat = re.compile(PATTERN)

    def encode(self, text: str) -> list[int]:
        out = []
        for m in self.pat.finditer(unicodedata.normalize("NFC", text)):
            ids = [b + self.n_special for b in m.group().encode()]
            while len(ids) > 1:
                pairs = [(self.ranks.get((ids[i], ids[i + 1]), 1 << 30), i) for i in range(len(ids) - 1)]
                r, i = min(pairs)
                if r == 1 << 30:
                    break
                ids = ids[:i] + [self.n_special + 256 + r] + ids[i + 2:]
            out += ids
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids if i >= self.n_special).decode("utf-8", "replace")


def generate(onnx_dir: Path, prompt: str, n_tokens: int, temperature: float = 0.0, top_k: int = 40, seed: int = 0):
    cfg = json.loads((onnx_dir / "onnx_config.json").read_text())
    tok = MiniTokenizer(onnx_dir / "tokenizer.json")
    sess = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    window, pad = cfg["window"], cfg["pad_id"]
    ids = tok.encode(prompt) or [1]
    out = []
    for _ in range(n_tokens):
        ctx = ids[-window:]
        buf = np.full((1, window), pad, dtype=np.int64)
        buf[0, : len(ctx)] = ctx
        logits = sess.run(["logits"], {"input_ids": buf})[0][0, len(ctx) - 1].astype(np.float64)
        logits[len(tok.vocab):] = -np.inf
        if temperature <= 0:
            nxt = int(np.argmax(logits))
        else:
            logits /= temperature
            if top_k:
                kth = np.sort(logits)[-top_k]
                logits[logits < kth] = -np.inf
            p = np.exp(logits - logits.max())
            p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
        if nxt == cfg["eos_id"]:
            break
        ids.append(nxt)
        out.append(nxt)
    return tok.decode(out), out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    text, _ = generate(Path(a.onnx), a.prompt, a.tokens, a.temperature)
    sys.stdout.write(a.prompt + text + "\n")
