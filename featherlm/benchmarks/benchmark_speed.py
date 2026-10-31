"""CPU inference speed: time to first token (prefill) and decode throughput vs context length.

    python -m featherlm.benchmarks.benchmark_speed --configs configs/nano_cpu.yaml configs/nano_transformer_cpu.yaml \
        --contexts 128 512 2048 --out results/speed
    python -m featherlm.benchmarks.benchmark_speed --models runs/featherlm-nano/final --out results/speed_ckpt
Throughput does not depend on weight values, so configs are benchmarked with random weights.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import yaml

from ..model.config import FeatherConfig
from ..model.model import FeatherLM
from .common import run_worker, write_results


@torch.no_grad()
def measure(model: FeatherLM, prompt_len: int, decode_tokens: int, repeats: int) -> dict:
    model.eval()
    V = model.cfg.vocab_size
    ttfts, tps = [], []
    for r in range(repeats + 1):  # first repeat is warm-up
        prompt = torch.randint(0, V, (1, prompt_len))
        t0 = time.perf_counter()
        caches = model.new_cache()
        logits, _, caches = model(prompt, caches=caches, last_only=True)
        nxt = logits[0, -1].argmax().view(1, 1)
        t1 = time.perf_counter()
        pos = prompt_len
        for _ in range(decode_tokens):
            logits, _, caches = model(nxt, start_pos=pos, caches=caches)
            nxt = logits[0, -1].argmax().view(1, 1)
            pos += 1
        t2 = time.perf_counter()
        if r:
            ttfts.append(t1 - t0)
            tps.append(decode_tokens / (t2 - t1))
    return {"prompt_len": prompt_len, "decode_tokens": decode_tokens,
            "ttft_ms": round(statistics.median(ttfts) * 1000, 2),
            "decode_tokens_per_sec": round(statistics.median(tps), 1)}


def worker(src: str, is_model: bool, ctx: int, decode: int, repeats: int, threads: int) -> dict:
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(0)
    if is_model:
        from ..training.checkpoint import load_model

        model, _, meta = load_model(src)
    else:
        m = dict(yaml.safe_load(open(src))["model"])
        m["max_seq_len"] = max(m.get("max_seq_len", 256), ctx + decode)
        model = FeatherLM(FeatherConfig.from_dict(m))
    r = measure(model, ctx, decode, repeats)
    return {"source": src, "backend": model.cfg.backend, "params": model.num_params(),
            "threads": torch.get_num_threads(), **r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--contexts", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--src", default="")
    ap.add_argument("--is-model", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--out", default="results/speed")
    a = ap.parse_args()
    if a.worker:
        print(json.dumps(worker(a.src, bool(a.is_model), a.ctx, a.decode, a.repeats, a.threads)))
        return
    rows = []
    jobs = [(c, 0) for c in a.configs] + [(m, 1) for m in a.models]
    for src, is_model in jobs:
        for ctx in a.contexts:
            r = run_worker("featherlm.benchmarks.benchmark_speed",
                           ["--src", src, "--is-model", str(is_model), "--ctx", str(ctx), "--decode", str(a.decode),
                            "--repeats", str(a.repeats), "--threads", str(a.threads)])
            print(f"{src} ctx={ctx}: TTFT {r['ttft_ms']} ms, decode {r['decode_tokens_per_sec']} tok/s")
            rows.append(r)
    write_results(rows, a.out)


if __name__ == "__main__":
    main()
