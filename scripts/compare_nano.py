#!/usr/bin/env python3
"""Evaluate the best checkpoints of the Nano Feather and Nano Transformer runs -> results/nano_comparison.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import evaluate  # noqa: E402
from featherlm.benchmarks.common import write_results  # noqa: E402
from featherlm.training.checkpoint import load_model  # noqa: E402

RUNS = {"featherlm-nano": ROOT / "runs/featherlm-nano", "featherlm-nano-transformer": ROOT / "runs/featherlm-nano-transformer"}


def main():
    ids = np.fromfile(ROOT / "data/shakespeare/val.bin", dtype=np.uint16)
    rows = []
    for name, run in RUNS.items():
        s = json.loads((run / "summary.json").read_text())
        meta = json.loads((run / "best" / "metadata.json").read_text())
        model, tok, _ = load_model(run / "best")
        ev = evaluate(model, tok, ids, model.cfg.max_seq_len)
        rows.append({"model": name, "backend": model.cfg.backend, "params": model.num_params(), "steps": s["steps"],
                     "stop_reason": s["status"], "best_val_step": meta["training"]["step"], **ev,
                     "peak_train_rss_mb": s["peak_rss_mb"], "train_tokens_per_sec": s["mean_tokens_per_sec"],
                     "train_time_min": round(s["elapsed_s"] / 60, 1), "first_train_loss": s["first_train_loss"],
                     "last_train_loss": s["last_train_loss"]})
        print(json.dumps(rows[-1], indent=2))
    write_results(rows, ROOT / "results/nano_comparison")


if __name__ == "__main__":
    main()
