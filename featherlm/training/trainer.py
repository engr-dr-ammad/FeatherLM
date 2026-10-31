"""CPU-first training loop for pretraining, continued pretraining and fine-tuning."""
from __future__ import annotations

import copy
import json
import math
import os
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml
from safetensors.torch import load_file

from ..data.dataset import CHAT_TEMPLATE, JsonlSFTDataset, TokenStream
from ..model.config import FeatherConfig
from ..model.model import FeatherLM
from ..tokenizer.tokenizer import ByteBPETokenizer
from .checkpoint import decode_rng, encode_rng, load_model, load_training_state, save_model
from .lora import apply_lora, lora_state_dict, merged_state_dict
from .metrics import (JsonlLogger, PeakMemoryMonitor, estimate_training_memory_mb, system_info,
                      usable_memory_mb, write_status)
from .optimizer import build_optimizer, build_scheduler, optimizer_state_bytes


class MemoryBudgetError(RuntimeError):
    pass


@dataclass
class TrainConfig:
    seed: int = 1337
    batch_size: int = 8
    grad_accum: int = 4
    max_steps: int = 1000
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    optimizer: str = "adamw"
    precision: str = "fp32"  # fp32 | bf16 (autocast)
    eval_interval: int = 100
    eval_batches: int = 20
    log_interval: int = 10
    save_interval: int = 200
    early_stopping_patience: int = 0  # evals without improvement; 0 disables
    threads: int = 0  # 0 -> all logical cores
    memory_budget_mb: float = 0  # 0 -> derive from available RAM
    reserve_mb: float = 1024
    allow_over_budget: bool = False
    max_minutes: float = 0  # wall-clock limit; 0 disables
    resume: bool = True


@dataclass
class DataConfig:
    path: str = "data/shakespeare"
    format: str = "bin"  # bin | jsonl
    train_file: str = ""  # jsonl only
    val_file: str = ""  # jsonl only
    seq_len: int = 256


@dataclass
class FinetuneConfig:
    init_from: str = ""  # checkpoint dir; empty -> train from scratch
    mode: str = "full"  # full | lora
    lora_r: int = 8
    lora_alpha: float = 16.0
    set_chat_template: bool = False


@dataclass
class RunConfig:
    model: dict[str, Any] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    output_dir: str = "runs/nano_cpu"

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict | None = None) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in (overrides or {}).items():  # dotted overrides, e.g. training.max_steps=10
            sect, _, key = k.partition(".")
            if key:
                raw.setdefault(sect, {})[key] = v
            else:
                raw[sect] = v
        allowed = {"model", "data", "training", "finetune", "output_dir"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        return cls(
            model=raw.get("model", {}),
            data=DataConfig(**raw.get("data", {})),
            training=TrainConfig(**raw.get("training", {})),
            finetune=FinetuneConfig(**raw.get("finetune", {})),
            output_dir=raw.get("output_dir", "runs/nano_cpu"),
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


def _parse_override(s: str):
    k, _, v = s.partition("=")
    try:
        val = yaml.safe_load(v)
    except yaml.YAMLError:
        val = v
    return k, val


class Trainer:
    def __init__(self, rc: RunConfig, log=print) -> None:
        self.rc = rc
        self.tc = rc.training
        self.log = log
        self.out = Path(rc.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stop_requested = False
        self.status_path = self.out / "status.json"
        self.metrics = JsonlLogger(self.out / "metrics.jsonl")
        self.is_lora = False

    # ------------------------------------------------------------------
    def _handle_signal(self, signum, frame) -> None:
        self.log(f"signal {signum} received: finishing current step, saving checkpoint, exiting")
        self.stop_requested = True

    def _seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _status(self, **kw) -> None:
        base = {"pid": os.getpid(), "output_dir": str(self.out), "updated": time.time()}
        prev = {}
        if self.status_path.exists():
            try:
                prev = json.loads(self.status_path.read_text())
            except json.JSONDecodeError:
                prev = {}
        prev.update(base)
        prev.update(kw)
        write_status(self.status_path, prev)

    # ------------------------------------------------------------------
    def _build(self):
        rc, ft = self.rc, self.rc.finetune
        if ft.init_from:
            model, tok, _ = load_model(ft.init_from)
            model.train()
            cfg = model.cfg
            if rc.model:
                # Allow overriding non-architectural knobs (dropout, checkpointing, name).
                for k in ("dropout", "gradient_checkpointing", "name", "max_seq_len"):
                    if k in rc.model:
                        setattr(cfg, k, rc.model[k])
            if ft.set_chat_template:
                cfg.chat_template = CHAT_TEMPLATE
            if ft.mode == "lora":
                n = apply_lora(model, ft.lora_r, ft.lora_alpha)
                self.is_lora = True
                self.log(f"LoRA enabled: {n:,} trainable params")
            elif ft.mode != "full":
                raise ValueError("finetune.mode must be 'full' or 'lora'")
        else:
            tok_path = Path(rc.data.path) / "tokenizer.json"
            tok = ByteBPETokenizer.load(tok_path)
            cfg = FeatherConfig.from_dict({**rc.model, "max_seq_len": rc.model.get("max_seq_len", rc.data.seq_len)})
            if tok.vocab_size > cfg.vocab_size:
                raise ValueError(f"tokenizer vocab {tok.vocab_size} > model vocab_size {cfg.vocab_size}")
            model = FeatherLM(cfg)
        if rc.data.seq_len > cfg.max_seq_len:
            raise ValueError("data.seq_len must be <= model max_seq_len")
        return model, tok, cfg

    def _data(self, tok):
        d = self.rc.data
        if d.format == "bin":
            train = TokenStream(Path(d.path) / "train.bin", d.seq_len)
            val_path = Path(d.path) / "val.bin"
            val = TokenStream(val_path, d.seq_len) if val_path.exists() and val_path.stat().st_size > 2 * (d.seq_len + 1) else None
        elif d.format == "jsonl":
            train = JsonlSFTDataset(d.train_file, tok, d.seq_len)
            val = JsonlSFTDataset(d.val_file, tok, d.seq_len) if d.val_file else None
        else:
            raise ValueError("data.format must be 'bin' or 'jsonl'")
        return train, val

    @torch.no_grad()
    def evaluate(self, model, val) -> float | None:
        if val is None:
            return None
        model.eval()
        total, n = 0.0, 0
        for x, y in val.sequential_batches(self.tc.batch_size, self.tc.eval_batches):
            with self._autocast():
                _, loss, _ = model(x, y)
            total += loss.item()
            n += 1
        model.train()
        return total / max(n, 1)

    def _autocast(self):
        if self.tc.precision == "bf16":
            return torch.autocast("cpu", dtype=torch.bfloat16)
        import contextlib

        return contextlib.nullcontext()

    def _save(self, name: str, model, tok, opt, sched, state: dict, with_optimizer: bool = True) -> Path:
        extra = {"training": {"step": state["step"], "best_val_loss": state.get("best_val_loss"),
                              "tokens_seen": state.get("tokens_seen"), "mode": self._mode()}}
        if self.is_lora:
            extra["training"]["lora_r"] = self.rc.finetune.lora_r
            p = save_model(self.out / name, model, tok, extra, opt if with_optimizer else None,
                           sched if with_optimizer else None, state, state=merged_state_dict(model),
                           extra_files={"lora_adapter.safetensors": lora_state_dict(model)})
        else:
            p = save_model(self.out / name, model, tok, extra, opt if with_optimizer else None,
                           sched if with_optimizer else None, state)
        return p

    def _mode(self) -> str:
        if not self.rc.finetune.init_from:
            return "pretrain_from_scratch"
        if self.is_lora:
            return "lora_adapter_finetune"
        return "sft_full" if self.rc.data.format == "jsonl" else "continued_pretraining"

    # ------------------------------------------------------------------
    def run(self) -> dict:
        tc = self.tc
        if tc.threads:
            torch.set_num_threads(tc.threads)
        self._seed(tc.seed)
        model, tok, cfg = self._build()
        train, val = self._data(tok)
        n_params = model.num_params()
        n_train = model.num_params(trainable_only=True)

        est = estimate_training_memory_mb(n_train, tc.batch_size, self.rc.data.seq_len, cfg.d_model, cfg.n_layers,
                                          cfg.vocab_size, tc.optimizer, cfg.gradient_checkpointing)
        est += (n_params - n_train) * 4 / 2**20
        budget = usable_memory_mb(tc.memory_budget_mb or None, tc.reserve_mb)
        self.log(f"model {cfg.name} [{cfg.backend}] params={n_params:,} trainable={n_train:,}")
        self.log(f"estimated peak training memory ~{est:.0f} MB; usable budget {budget:.0f} MB")
        if est > budget and not tc.allow_over_budget:
            raise MemoryBudgetError(
                f"estimated {est:.0f} MB exceeds budget {budget:.0f} MB. Reduce batch_size/seq_len, "
                "enable model.gradient_checkpointing, use optimizer=adafactor, or set training.allow_over_budget=true")

        opt = build_optimizer(model, tc.optimizer, tc.lr, tc.weight_decay)
        sched = build_scheduler(opt, tc.warmup_steps, tc.max_steps, tc.min_lr_ratio)
        rng = np.random.default_rng(tc.seed)
        state = {"step": 0, "best_val_loss": None, "tokens_seen": 0, "evals_without_improvement": 0}

        latest = self.out / "latest"
        if tc.resume and (latest / "trainer_state.json").exists():
            if self.is_lora:
                adapter = load_file(str(latest / "lora_adapter.safetensors"))
                model.load_state_dict(adapter, strict=False)
            else:
                model.load_state_dict(load_file(str(latest / "model.safetensors")))
            state = load_training_state(latest, opt, sched)
            if "np_rng" in state:
                rng.bit_generator.state = state["np_rng"]
            if "torch_rng" in state:
                torch.set_rng_state(decode_rng(state["torch_rng"]))
            self.log(f"resumed from {latest} at step {state['step']}")

        (self.out / "run_config.json").write_text(json.dumps({**self.rc.to_dict(), "system": system_info()}, indent=2))
        old_int = signal.signal(signal.SIGINT, self._handle_signal)
        old_term = signal.signal(signal.SIGTERM, self._handle_signal)
        mon = PeakMemoryMonitor().start()
        proc = psutil.Process()
        proc.cpu_percent(None)
        tokens_per_step = tc.batch_size * tc.grad_accum * self.rc.data.seq_len
        t_start = time.time()
        t_last = t_start
        tokens_since = 0
        status = "running"
        self._status(state="running", step=state["step"], max_steps=tc.max_steps, model=cfg.name,
                     backend=cfg.backend, dataset=self.rc.data.path, params=n_params, mode=self._mode(),
                     started=t_start)
        history = []
        val_loss = None
        model.train()
        try:
            while state["step"] < tc.max_steps:
                opt.zero_grad(set_to_none=True)
                loss_acc = 0.0
                for _ in range(tc.grad_accum):
                    x, y = train.random_batch(tc.batch_size, rng)
                    with self._autocast():
                        _, loss, _ = model(x, y)
                    (loss / tc.grad_accum).backward()
                    loss_acc += loss.item() / tc.grad_accum
                if not math.isfinite(loss_acc):
                    raise FloatingPointError(f"non-finite loss at step {state['step']}")
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], tc.grad_clip) if tc.grad_clip > 0 else torch.tensor(0.0)
                opt.step()
                sched.step()
                state["step"] += 1
                state["tokens_seen"] += tokens_per_step
                tokens_since += tokens_per_step
                step = state["step"]

                if step % tc.log_interval == 0 or step == 1:
                    now = time.time()
                    tps = tokens_since / max(now - t_last, 1e-9)
                    t_last, tokens_since = now, 0
                    lr = sched.get_last_lr()[0]
                    rec = {"step": step, "train_loss": round(loss_acc, 4), "lr": lr, "grad_norm": float(gnorm),
                           "tokens_per_sec": round(tps, 1), "rss_mb": round(mon.current_mb, 1),
                           "peak_rss_mb": round(mon.peak_mb, 1), "cpu_percent": proc.cpu_percent(None),
                           "tokens_seen": state["tokens_seen"], "elapsed_s": round(now - t_start, 1)}
                    eta = (tc.max_steps - step) * tokens_per_step / max(tps, 1e-9)
                    self.log(f"step {step:>6}/{tc.max_steps} loss {loss_acc:.4f} lr {lr:.2e} "
                             f"{tps:,.0f} tok/s rss {mon.current_mb:.0f}MB peak {mon.peak_mb:.0f}MB eta {eta/60:.1f}m")
                    self.metrics.log({"type": "train", **rec})
                    history.append(rec)
                    self._status(state="running", step=step, train_loss=rec["train_loss"], lr=lr,
                                 tokens_per_sec=rec["tokens_per_sec"], rss_mb=rec["rss_mb"],
                                 peak_rss_mb=rec["peak_rss_mb"], cpu_percent=rec["cpu_percent"],
                                 tokens_seen=state["tokens_seen"])

                if val is not None and (step % tc.eval_interval == 0 or step == tc.max_steps):
                    val_loss = self.evaluate(model, val)
                    improved = state["best_val_loss"] is None or val_loss < state["best_val_loss"]
                    if improved:
                        state["best_val_loss"] = val_loss
                        state["evals_without_improvement"] = 0
                    else:
                        state["evals_without_improvement"] += 1
                    self.log(f"  eval step {step}: val_loss {val_loss:.4f} ppl {math.exp(min(val_loss, 50)):.2f}"
                             + (" (best)" if improved else ""))
                    self.metrics.log({"type": "eval", "step": step, "val_loss": val_loss,
                                      "val_ppl": math.exp(min(val_loss, 50))})
                    self._status(val_loss=val_loss, best_val_loss=state["best_val_loss"])
                    if improved:
                        self._save("best", model, tok, opt, sched, self._snap(state, rng), with_optimizer=False)
                    if tc.early_stopping_patience and state["evals_without_improvement"] >= tc.early_stopping_patience:
                        self.log("early stopping: validation loss stopped improving")
                        status = "early_stopped"
                        break

                if step % tc.save_interval == 0:
                    self._save("latest", model, tok, opt, sched, self._snap(state, rng))
                    self._status(last_checkpoint=str(self.out / "latest"), checkpoint_step=step)

                if self.stop_requested:
                    status = "stopped"
                    break
                if tc.max_minutes and (time.time() - t_start) > tc.max_minutes * 60:
                    self.log("time limit reached")
                    status = "time_limit"
                    break
            else:
                status = "completed"
        except Exception as e:
            self._status(state="failed", error=f"{type(e).__name__}: {e}")
            raise
        finally:
            mon.stop()
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)

        self._save("latest", model, tok, opt, sched, self._snap(state, rng))
        self._save("final", model, tok, None, None, self._snap(state, rng), with_optimizer=False)
        elapsed = time.time() - t_start
        summary = {
            "status": status,
            "steps": state["step"],
            "tokens_seen": state["tokens_seen"],
            "params": n_params,
            "trainable_params": n_train,
            "first_train_loss": history[0]["train_loss"] if history else None,
            "last_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": val_loss,
            "best_val_loss": state["best_val_loss"],
            "peak_rss_mb": round(mon.peak_mb, 1),
            "estimated_memory_mb": round(est, 1),
            "optimizer_state_mb": round(optimizer_state_bytes(opt) / 2**20, 2),
            "elapsed_s": round(elapsed, 1),
            "mean_tokens_per_sec": round(sum(h["tokens_per_sec"] for h in history[1:]) / max(len(history) - 1, 1), 1),
            "mode": self._mode(),
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2))
        self._status(state=status, step=state["step"], last_checkpoint=str(self.out / "latest"),
                     final_model=str(self.out / "final"), summary=summary)
        self.log(json.dumps(summary, indent=2))
        return summary

    def _snap(self, state: dict, rng) -> dict:
        s = copy.deepcopy(state)
        s["np_rng"] = rng.bit_generator.state
        s["torch_rng"] = encode_rng(torch.get_rng_state())
        return s
