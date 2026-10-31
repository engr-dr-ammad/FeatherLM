"""Shared helpers for benchmarks: isolated subprocess runs and result writing."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from ..training.metrics import system_info


def run_worker(module: str, args: list[str], timeout: float = 3600) -> dict:
    """Run `python -m module --worker ...` in a fresh process (clean RSS) and parse its JSON line."""
    cmd = [sys.executable, "-m", module, "--worker", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed ({' '.join(cmd)}):\n{proc.stderr[-2000:]}")
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"worker produced no JSON:\n{proc.stdout[-2000:]}")


def write_results(rows: list[dict], out_prefix: str | Path, extra: dict | None = None) -> None:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"system": system_info(), "rows": rows, **(extra or {})}
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2))
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with out_prefix.with_suffix(".csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
