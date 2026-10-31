"""Matched-budget training comparison across architectures and seeds.

Every run uses the same dataset, tokenizer, sequence length, batch size, number of
steps (=> identical training-token budget), optimizer and precision. Only the model
section differs. Results: per-run rows + per-architecture mean/std, CSV/JSON + plots.

    python -m featherlm.benchmarks.benchmark_quality --spec configs/bench_micro.yaml --out results/quality
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .common import write_results


def run_one(base: dict, arch_name: str, model_cfg: dict, seed: int, out_root: Path) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg["model"] = {**cfg.get("model", {}), **model_cfg}
    cfg["training"]["seed"] = seed
    cfg["training"]["resume"] = False
    cfg["training"]["early_stopping_patience"] = 0
    run_dir = out_root / f"{arch_name}_seed{seed}"
    cfg["output_dir"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "bench_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    t = time.time()
    proc = subprocess.run([sys.executable, "train.py", "--config", str(cfg_path), "--no-resume"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{arch_name} seed {seed} failed:\n{proc.stderr[-3000:]}")
    s = json.loads((run_dir / "summary.json").read_text())
    curve = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in curve if r["type"] == "eval"]
    print(f"  {arch_name} seed={seed}: val {s['final_val_loss']:.4f} params {s['params']:,} "
          f"peak {s['peak_rss_mb']} MB {s['mean_tokens_per_sec']} tok/s ({time.time() - t:.0f}s)")
    return {"arch": arch_name, "seed": seed, "params": s["params"], "final_val_loss": s["final_val_loss"],
            "final_val_ppl": math.exp(s["final_val_loss"]), "best_val_loss": s["best_val_loss"],
            "last_train_loss": s["last_train_loss"], "peak_train_rss_mb": s["peak_rss_mb"],
            "train_tokens_per_sec": s["mean_tokens_per_sec"], "train_time_s": s["elapsed_s"],
            "tokens_seen": s["tokens_seen"], "eval_curve": [(e["step"], e["val_loss"]) for e in evals]}


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    for arch in dict.fromkeys(r["arch"] for r in rows):
        rs = [r for r in rows if r["arch"] == arch]
        def ms(k):
            v = [r[k] for r in rs]
            return round(statistics.mean(v), 4), round(statistics.stdev(v), 4) if len(v) > 1 else 0.0
        vl, vls = ms("final_val_loss")
        out.append({"arch": arch, "n_seeds": len(rs), "params": rs[0]["params"], "val_loss_mean": vl,
                    "val_loss_std": vls, "val_ppl_mean": round(math.exp(vl), 2),
                    "peak_train_rss_mb_mean": ms("peak_train_rss_mb")[0],
                    "train_tokens_per_sec_mean": ms("train_tokens_per_sec")[0],
                    "train_time_s_mean": ms("train_time_s")[0]})
    return out


def plot(rows: list[dict], out_prefix: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {}
    for r in rows:
        c = colors.setdefault(r["arch"], f"C{len(colors)}")
        steps, losses = zip(*r["eval_curve"]) if r["eval_curve"] else ([], [])
        ax.plot(steps, losses, color=c, alpha=0.8, label=r["arch"] if r["seed"] == rows[0]["seed"] else None)
    ax.set_xlabel("optimizer step (equal token budget)")
    ax.set_ylabel("validation loss (nats/token)")
    ax.set_title("Matched-budget validation loss (each line = one seed)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_prefix.with_name(out_prefix.name + "_val_loss.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="YAML with base config, architectures and seeds")
    ap.add_argument("--out", default="results/quality")
    a = ap.parse_args()
    spec = yaml.safe_load(Path(a.spec).read_text())
    out = Path(a.out)
    runs_root = Path(spec.get("runs_dir", "runs/bench"))
    rows = []
    for arch_name, mcfg in spec["architectures"].items():
        for seed in spec["seeds"]:
            rows.append(run_one(spec["base"], arch_name, mcfg, seed, runs_root))
    summary = aggregate(rows)
    write_results([{k: v for k, v in r.items() if k != "eval_curve"} for r in rows], out,
                  {"summary": summary, "curves": {f"{r['arch']}_seed{r['seed']}": r["eval_curve"] for r in rows},
                   "spec": spec})
    write_results(summary, out.with_name(out.name + "_summary"))
    plot(rows, out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
