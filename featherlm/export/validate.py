"""Validate quantized / exported models against the FP32 reference.

    python -m featherlm.export.validate --reference runs/featherlm-nano/final \
        --candidates models/featherlm-nano-int8 models/featherlm-nano-int4 --onnx models/featherlm-nano-onnx \
        --data data/shakespeare --out results/quantization.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import TokenStream
from ..training.checkpoint import load_model
from ..training.metrics import PeakMemoryMonitor


@torch.no_grad()
def perplexity(model, stream: TokenStream, batch_size: int = 8, max_batches: int = 10) -> float:
    tot, n = 0.0, 0
    for x, y in stream.sequential_batches(batch_size, max_batches):
        tot += model(x, y)[1].item()
        n += 1
    return math.exp(tot / n)


@torch.no_grad()
def compare_logits(ref, cand, batches) -> dict:
    max_diff, agree, total = 0.0, 0, 0
    for x, _ in batches:
        a, b = ref(x)[0], cand(x)[0]
        max_diff = max(max_diff, (a - b).abs().max().item())
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
        total += a.shape[0] * a.shape[1]
    return {"max_abs_logit_diff": max_diff, "top1_agreement": agree / total}


@torch.no_grad()
def decode_speed(model, n_tokens: int = 64) -> float:
    from ..inference.sampling import SamplingParams
    from ..model.generation import generate_ids

    params = SamplingParams(max_new_tokens=n_tokens, temperature=0.0)
    t = time.perf_counter()
    n = sum(1 for _ in generate_ids(model, [1, 100, 200], params))
    return n / (time.perf_counter() - t)


def onnx_check(onnx_dir: Path, ref, stream: TokenStream) -> dict:
    import onnxruntime as ort

    cfg = json.loads((onnx_dir / "onnx_config.json").read_text())
    sess = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    T = cfg["window"]
    x, _ = next(stream.sequential_batches(1, 1))
    x = x[:, :T]
    with torch.no_grad():
        ref_logits = ref(x)[0].numpy()
    t = time.perf_counter()
    out = sess.run(["logits"], {"input_ids": x.numpy().astype(np.int64)})[0]
    dt = time.perf_counter() - t
    # Right-padding check: a shorter prompt must give identical logits at its last position.
    L = T // 3
    padded = np.full((1, T), cfg["pad_id"], dtype=np.int64)
    padded[0, :L] = x[0, :L].numpy()
    out_pad = sess.run(["logits"], {"input_ids": padded})[0]
    return {
        "max_abs_logit_diff_vs_pytorch": float(np.abs(out - ref_logits).max()),
        "top1_agreement": float((out.argmax(-1) == ref_logits.argmax(-1)).mean()),
        "right_padding_max_diff": float(np.abs(out_pad[0, L - 1] - ref_logits[0, L - 1]).max()),
        "window_forward_ms": round(dt * 1000, 1),
        "file_bytes": (onnx_dir / "model.onnx").stat().st_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate quantized/exported FeatherLM models")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidates", nargs="*", default=[])
    ap.add_argument("--onnx")
    ap.add_argument("--data", required=True, help="dataset dir containing val.bin")
    ap.add_argument("--out", default="results/quantization.json")
    ap.add_argument("--max-batches", type=int, default=8)
    args = ap.parse_args()

    ref, _, ref_meta = load_model(args.reference)
    seq = ref.cfg.max_seq_len
    stream = TokenStream(Path(args.data) / "val.bin", seq)
    results = {"reference": args.reference, "models": []}

    def measure(path: str, packed: bool = False) -> dict:
        mon = PeakMemoryMonitor().start()
        if packed:
            from .quantize import load_quantized_into
            from ..model.config import FeatherConfig
            from ..model.model import FeatherLM

            meta = json.loads((Path(path) / "metadata.json").read_text())
            m = FeatherLM(FeatherConfig.load(Path(path) / "config.json"))
            load_quantized_into(m, Path(path) / "model.safetensors", meta["quantization"], packed=True)
            m.eval()
        else:
            m, _, meta = load_model(path)
        ppl = perplexity(m, stream, max_batches=args.max_batches)
        cmp = compare_logits(ref, m, stream.sequential_batches(4, 2))
        tps = decode_speed(m)
        mon.stop()
        resident = sum(b.numel() * b.element_size() for b in list(m.parameters()) + list(m.buffers()))
        size = sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))
        return {"model": path, "quantization": meta.get("quantization", "none"),
                "runtime": "packed" if packed else "dequantized", "val_perplexity": round(ppl, 3),
                **cmp, "decode_tokens_per_sec": round(tps, 1), "file_bytes": size,
                "resident_weight_bytes": resident, "peak_rss_mb_during_eval": round(mon.peak_mb, 1)}

    results["models"].append(measure(args.reference))
    for c in args.candidates:
        results["models"].append(measure(c))
        meta = json.loads((Path(c) / "metadata.json").read_text())
        if meta.get("quantization") in ("int8", "int4"):
            results["models"].append(measure(c, packed=True))
    if args.onnx:
        results["onnx"] = onnx_check(Path(args.onnx), ref, stream)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
