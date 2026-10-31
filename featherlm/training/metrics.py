"""Runtime metrics: peak RSS monitoring, memory budget checks and JSONL logging."""
from __future__ import annotations

import json
import os
import platform
import resource
import sys
import threading
import time
from pathlib import Path

import psutil

MB = 1024 * 1024


class PeakMemoryMonitor:
    """Samples process RSS in a background thread and tracks the peak."""

    def __init__(self, interval: float = 0.05) -> None:
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = self.proc.memory_info().rss
        self.baseline = self.peak
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def start(self) -> "PeakMemoryMonitor":
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1)
        self.peak = max(self.peak, self.proc.memory_info().rss)

    def reset(self) -> None:
        self.peak = self.proc.memory_info().rss
        self.baseline = self.peak

    @property
    def current_mb(self) -> float:
        return self.proc.memory_info().rss / MB

    @property
    def peak_mb(self) -> float:
        return self.peak / MB

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def ru_maxrss_mb() -> float:
    """OS-reported lifetime peak RSS of this process."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / MB if sys.platform == "darwin" else r / 1024


def usable_memory_mb(budget_mb: float | None = None, reserve_mb: float = 1024) -> float:
    """Memory we allow ourselves: based on *currently available* RAM minus a reserve for the
    OS and other programs, capped by the user's budget. Never based on total installed RAM."""
    available = psutil.virtual_memory().available / MB - reserve_mb
    available = max(available, 256.0)
    return min(available, budget_mb) if budget_mb else available


def estimate_training_memory_mb(n_params: int, batch_size: int, seq_len: int, d_model: int, n_layers: int,
                                vocab_size: int, optimizer: str = "adamw", checkpointing: bool = False,
                                runtime_overhead_mb: float | None = None) -> float:
    """Rough fp32 estimate: weights + grads + optimizer state + activations + logits + runtime overhead.
    runtime_overhead_mb defaults to the current process RSS (Python + PyTorch already loaded)."""
    opt_mult = {"adamw": 2.0, "adafactor": 0.1, "sgd": 1.0}.get(optimizer, 2.0)
    state = n_params * 4 * (2 + opt_mult)
    per_layer = batch_size * seq_len * d_model * 4 * (8 if checkpointing else 48)  # calibrated on nano runs
    logits = batch_size * seq_len * vocab_size * 4 * 3
    if runtime_overhead_mb is None:
        runtime_overhead_mb = psutil.Process(os.getpid()).memory_info().rss / MB
    overhead = runtime_overhead_mb * MB
    return (state + per_layer * n_layers + logits + overhead) / MB


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def system_info() -> dict:
    import torch

    vm = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": cpu_model(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "torch_threads": torch.get_num_threads(),
        "ram_total_mb": round(vm.total / MB),
        "ram_available_mb": round(vm.available / MB),
        "cuda": torch.cuda.is_available(),
    }


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        record = {"time": time.time(), **record}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def write_status(path: str | Path, status: dict) -> None:
    """Atomically write a status JSON file (polled by the API/dashboard)."""
    p = Path(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(p)
