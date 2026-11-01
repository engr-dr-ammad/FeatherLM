"""Inference engine: load a checkpoint and stream text with timing/memory stats."""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch

from ..data.dataset import CHAT_TEMPLATE, format_chat
from ..model.generation import generate_ids
from ..training.checkpoint import load_model
from .sampling import SamplingParams
from .streaming import StopSequenceFilter


class EngineBusy(RuntimeError):
    pass


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_to_first_token_s: float = 0.0
    total_time_s: float = 0.0
    tokens_per_sec: float = 0.0
    rss_mb: float = 0.0
    finish_reason: str = "length"


class InferenceEngine:
    def __init__(self, model_dir: str | Path, device: str = "cpu", threads: int = 0) -> None:
        if threads:
            torch.set_num_threads(threads)
        self.model_dir = Path(model_dir)
        self.model, self.tok, self.meta = load_model(self.model_dir, device=device)
        self.device = device
        self.lock = threading.Lock()  # one generation at a time per engine (CPU-bound)

    @property
    def supports_chat(self) -> bool:
        return self.model.cfg.chat_template == CHAT_TEMPLATE

    def encode_prompt(self, prompt: str) -> list[int]:
        ids = self.tok.encode(prompt)
        return ids if ids else [self.tok.bos_id]

    def encode_chat(self, messages: list[dict], reserve: int = 64) -> list[int]:
        """Format a conversation for the chat template, keeping only the most recent turns that fit.

        Small models are trained on short conversations within max_seq_len tokens; feeding them a
        long history makes them drift (e.g. continue an earlier answer). System messages are kept;
        the oldest user/assistant turns are dropped until prompt + `reserve` new tokens fit.
        """
        if not self.supports_chat:
            raise ValueError("this model has no chat template; use the completions endpoint")
        budget = max(32, self.model.cfg.max_seq_len - reserve)
        system = [m for m in messages if m.get("role") == "system"]
        turns = [m for m in messages if m.get("role") != "system"]
        ids, _ = format_chat(system + turns, self.tok, add_generation_prompt=True)
        while len(ids) > budget and len(turns) > 1:
            turns = turns[1:]
            while len(turns) > 1 and turns[0].get("role") != "user":  # start at a user turn
                turns = turns[1:]
            ids, _ = format_chat(system + turns, self.tok, add_generation_prompt=True)
        return ids

    def stream(self, prompt_ids: list[int], params: SamplingParams, cancel: threading.Event | None = None,
               stats: GenerationStats | None = None, lock_timeout: float | None = None) -> Iterator[str]:
        """Yield text pieces. Fills `stats` as it goes. Raises EngineBusy if the engine stays
        busy longer than lock_timeout seconds (None = wait indefinitely)."""
        stats = stats if stats is not None else GenerationStats()
        stats.prompt_tokens = len(prompt_ids)
        max_ctx = self.model.cfg.max_seq_len
        if self.model.cfg.backend == "transformer" and len(prompt_ids) > max_ctx:
            prompt_ids = prompt_ids[-max_ctx:]
        eos = {self.tok.eos_id}
        if self.supports_chat:
            eos.add(self.tok.token_id("<|end|>"))
        stop_filter = StopSequenceFilter(self.tok, params.stop)
        t0 = time.perf_counter()
        n = 0
        proc = psutil.Process()
        if not self.lock.acquire(timeout=-1 if lock_timeout is None else lock_timeout):
            raise EngineBusy("model is busy with another request")
        try:
            for tid in generate_ids(self.model, prompt_ids, params, eos, cancel, vocab_limit=self.tok.vocab_size):
                if n == 0:
                    stats.time_to_first_token_s = time.perf_counter() - t0
                n += 1
                if tid in eos:
                    stats.finish_reason = "stop"
                    break
                piece, hit = stop_filter.push(tid)
                if piece:
                    yield piece
                if hit:
                    stats.finish_reason = "stop"
                    break
            else:
                stats.finish_reason = "cancelled" if cancel is not None and cancel.is_set() else "length"
            if stats.finish_reason != "stop":
                tail = stop_filter.flush()
                if tail:
                    yield tail
        finally:
            self.lock.release()
        stats.completion_tokens = n
        stats.total_time_s = time.perf_counter() - t0
        stats.tokens_per_sec = n / stats.total_time_s if stats.total_time_s > 0 else 0.0
        stats.rss_mb = proc.memory_info().rss / 2**20

    def generate(self, prompt: str, params: SamplingParams) -> tuple[str, GenerationStats]:
        stats = GenerationStats()
        text = "".join(self.stream(self.encode_prompt(prompt), params, stats=stats))
        return text, stats

    def info(self) -> dict:
        cfg = self.model.cfg
        size = sum(f.stat().st_size for f in self.model_dir.glob("*.safetensors"))
        return {
            "name": self.meta.get("name", cfg.name),
            "architecture": cfg.backend,
            "parameters": self.model.num_params(),
            "file_size_bytes": size,
            "quantization": self.meta.get("quantization", "none"),
            "context_length": cfg.max_seq_len,
            "chat_template": cfg.chat_template,
        }
