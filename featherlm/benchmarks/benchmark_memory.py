"""Peak RAM during training steps and during inference, each in an isolated process.

    python -m featherlm.benchmarks.benchmark_memory --configs configs/nano_cpu.yaml configs/nano_transformer_cpu.yaml \
        --steps 5 --out results/memory
Random weights are used: memory does not depend on weight values.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import yaml

from ..model.config import FeatherConfig
from ..model.model import FeatherLM
from ..training.metrics import PeakMemoryMonitor
from ..training.optimizer import build_optimizer, optimizer_state_bytes
from .common import run_worker, write_results


def worker(config_path: str, mode: str, steps: int, batch: int, seq: int, ckpt: bool, ctx: int, optimizer: str) -> dict:
    torch.manual_seed(0)
    raw = yaml.safe_load(open(config_path))
    mcfg = dict(raw["model"])
    mcfg["gradient_checkpointing"] = ckpt
    mcfg["dropout"] = 0.0
    if mode == "infer":
        mcfg["max_seq_len"] = max(mcfg.get("max_seq_len", 256), ctx)
    cfg = FeatherConfig.from_dict(mcfg)
    mon = PeakMemoryMonitor(interval=0.01).start()
    base = mon.current_mb
    model = FeatherLM(cfg)
    res = {"config": config_path, "backend": cfg.backend, "params": model.num_params(), "mode": mode,
           "baseline_rss_mb": round(base, 1)}
    if mode == "train":
        opt = build_optimizer(model, optimizer)
        model.train()
        for _ in range(steps):
            x = torch.randint(0, cfg.vocab_size, (batch, seq))
            loss = model(x, x)[1]
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        res.update(batch_size=batch, seq_len=seq, gradient_checkpointing=ckpt, optimizer=optimizer,
                   optimizer_state_mb=round(optimizer_state_bytes(opt) / 2**20, 1))
    else:
        model.eval()
        with torch.no_grad():
            caches = model.new_cache()
            _, _, caches = model(torch.randint(0, cfg.vocab_size, (1, 8)), caches=caches, last_only=True)
            for pos in range(8, ctx):
                _, _, caches = model(torch.randint(0, cfg.vocab_size, (1, 1)), start_pos=pos, caches=caches)
        cache_bytes = 0
        for c in caches:
            for v in (c or {}).values():
                if torch.is_tensor(v):
                    cache_bytes += v.numel() * v.element_size()
        res.update(context=ctx, cache_mb=round(cache_bytes / 2**20, 3))
    mon.stop()
    res["peak_rss_mb"] = round(mon.peak_mb, 1)
    res["peak_over_baseline_mb"] = round(mon.peak_mb - base, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--contexts", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--checkpointing", choices=["off", "on", "both"], default="both")
    ap.add_argument("--optimizers", nargs="+", default=["adamw"])
    ap.add_argument("--mode", default="train")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--ckpt", type=int, default=0)
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--out", default="results/memory")
    a = ap.parse_args()
    if a.worker:
        print(json.dumps(worker(a.configs[0], a.mode, a.steps, a.batch, a.seq, bool(a.ckpt), a.ctx, a.optimizer)))
        return
    rows = []
    ck = {"off": [0], "on": [1], "both": [0, 1]}[a.checkpointing]
    for cfgp in a.configs:
        for c in ck:
            for optn in a.optimizers:
                t = time.time()
                r = run_worker("featherlm.benchmarks.benchmark_memory",
                               ["--configs", cfgp, "--mode", "train", "--steps", str(a.steps), "--batch", str(a.batch),
                                "--seq", str(a.seq), "--ckpt", str(c), "--optimizer", optn])
                print(f"train {cfgp} ckpt={c} opt={optn}: peak {r['peak_rss_mb']} MB ({time.time() - t:.0f}s)")
                rows.append(r)
        for ctx in a.contexts:
            r = run_worker("featherlm.benchmarks.benchmark_memory",
                           ["--configs", cfgp, "--mode", "infer", "--ctx", str(ctx)])
            print(f"infer {cfgp} ctx={ctx}: peak {r['peak_rss_mb']} MB cache {r['cache_mb']} MB")
            rows.append(r)
    write_results(rows, a.out, {"note": "random weights; peak RSS measured in isolated processes"})


if __name__ == "__main__":
    main()
