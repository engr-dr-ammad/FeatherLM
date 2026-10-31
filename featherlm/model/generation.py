"""Incremental autoregressive decoding with KV cache / recurrent state."""
from __future__ import annotations

import threading
from collections.abc import Iterator

import torch

from ..inference.sampling import SamplingParams, sample_next
from .model import FeatherLM


@torch.no_grad()
def generate_ids(
    model: FeatherLM,
    prompt_ids: list[int],
    params: SamplingParams,
    eos_ids: set[int] | None = None,
    cancel: threading.Event | None = None,
    vocab_limit: int | None = None,
) -> Iterator[int]:
    """Yield generated token ids one at a time.

    `vocab_limit` (usually the tokenizer's vocab size) masks output rows the tokenizer
    cannot decode: the model's vocab_size may be padded above the tokenizer's.

    Stops at max_new_tokens, an EOS id, or when `cancel` is set. Text-level stop
    sequences are handled by the caller (they need detokenisation).

    The Transformer backend has a hard context limit (max_seq_len); when it is
    reached the context is cropped and re-encoded. The Feather backend keeps a
    constant-size state (recurrent state + windowed KV cache), so it can continue
    past max_seq_len, although quality beyond the training length is not guaranteed.
    """
    params.validate()
    model.eval()
    device = next(model.parameters()).device
    gen = None
    if params.seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(params.seed)
    eos_ids = eos_ids or set()
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    max_len = model.cfg.max_seq_len
    bounded = model.cfg.backend == "transformer"
    history = list(prompt_ids)
    ctx = history[-max_len:] if bounded else history

    caches = model.new_cache()
    logits, _, caches = model(torch.tensor([ctx], device=device), caches=caches, last_only=True)
    pos = len(ctx)

    for _ in range(params.max_new_tokens):
        if cancel is not None and cancel.is_set():
            return
        prev = torch.tensor(history[-256:], dtype=torch.long)
        last = logits[0, -1].cpu()
        if vocab_limit is not None and vocab_limit < last.shape[0]:
            last = last.clone()
            last[vocab_limit:] = float("-inf")
        nxt = sample_next(last, prev, params, gen)
        history.append(nxt)
        yield nxt
        if nxt in eos_ids:
            return
        if bounded and pos >= max_len:
            ctx = history[-(max_len - 1) :] if max_len > 1 else history[-1:]
            caches = model.new_cache()
            logits, _, caches = model(torch.tensor([ctx], device=device), caches=caches, last_only=True)
            pos = len(ctx)
        else:
            logits, _, caches = model(torch.tensor([[nxt]], device=device), start_pos=pos, caches=caches)
            pos += 1
