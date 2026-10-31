"""Datasets: memory-mapped token streams (pretraining) and JSONL instruction data (SFT)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..model.model import IGNORE_INDEX
from ..tokenizer.tokenizer import ByteBPETokenizer
from .packing import window_starts


class TokenStream:
    """Random/sequential windows over a uint16 .bin file via np.memmap.

    The file is never loaded into RAM in full, so dataset size does not drive
    memory use (streaming from disk).
    """

    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.path = Path(path)
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        if len(self.data) < seq_len + 1:
            raise ValueError(f"{path} has {len(self.data)} tokens, need at least {seq_len + 1}")

    def __len__(self) -> int:
        return len(self.data)

    def _batch_from_starts(self, starts) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.stack([self.data[s : s + self.seq_len + 1].astype(np.int64) for s in starts])
        t = torch.from_numpy(x)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def random_batch(self, batch_size: int, generator: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        starts = generator.integers(0, len(self.data) - self.seq_len - 1, size=batch_size)
        return self._batch_from_starts(starts)

    def sequential_batches(self, batch_size: int, max_batches: int | None = None):
        """Deterministic non-overlapping windows (used for validation)."""
        starts = window_starts(len(self.data), self.seq_len)
        n = 0
        for i in range(0, len(starts), batch_size):
            if max_batches is not None and n >= max_batches:
                return
            yield self._batch_from_starts(starts[i : i + batch_size])
            n += 1


# ----------------------------------------------------------------------
CHAT_TEMPLATE = "featherlm-v1"


def format_chat(messages: list[dict], tok: ByteBPETokenizer, add_generation_prompt: bool = False):
    """featherlm-v1 template: <|bos|><|role|>content<|end|>... Returns (ids, loss_mask).

    Loss is computed only on assistant content and its <|end|> token.
    Message content is encoded with allow_special=False so user text cannot inject control tokens.
    """
    roles = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
    ids, mask = [tok.bos_id], [0]
    end = tok.token_id("<|end|>")
    for m in messages:
        role = m.get("role")
        if role not in roles:
            raise ValueError(f"unsupported role {role!r}")
        content = str(m.get("content", ""))
        ids.append(tok.token_id(roles[role]))
        mask.append(0)
        c = tok.encode(content)
        ids += c + [end]
        train = 1 if role == "assistant" else 0
        mask += [train] * (len(c) + 1)
    if add_generation_prompt:
        ids.append(tok.token_id("<|assistant|>"))
        mask.append(0)
    return ids, mask


class JsonlSFTDataset:
    """Instruction data. Each line: {"messages":[{"role":..,"content":..},...]}
    or {"prompt": "...", "completion": "..."}."""

    def __init__(self, path: str | Path, tok: ByteBPETokenizer, seq_len: int) -> None:
        self.examples: list[tuple[list[int], list[int]]] = []
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e})") from e
            if "messages" in row:
                msgs = row["messages"]
            elif "prompt" in row and "completion" in row:
                msgs = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["completion"]}]
            else:
                raise ValueError(f"{path}:{n}: need 'messages' or 'prompt'+'completion'")
            ids, mask = format_chat(msgs, tok)
            ids, mask = ids[: seq_len + 1], mask[: seq_len + 1]
            if sum(mask[1:]) > 0:
                self.examples.append((ids, mask))
        if not self.examples:
            raise ValueError(f"{path}: no usable examples")
        self.seq_len = seq_len
        self.pad = tok.pad_id

    def __len__(self) -> int:
        return len(self.examples)

    def collate(self, items) -> tuple[torch.Tensor, torch.Tensor]:
        T = max(len(i) for i, _ in items) - 1
        x = torch.full((len(items), T), self.pad, dtype=torch.long)
        y = torch.full((len(items), T), IGNORE_INDEX, dtype=torch.long)
        for b, (ids, mask) in enumerate(items):
            ids_t = torch.tensor(ids)
            m = torch.tensor(mask[1:], dtype=torch.bool)
            x[b, : len(ids) - 1] = ids_t[:-1]
            tgt = ids_t[1:].clone()
            tgt[~m] = IGNORE_INDEX
            y[b, : len(ids) - 1] = tgt
        return x, y

    def random_batch(self, batch_size: int, generator: np.random.Generator):
        idx = generator.integers(0, len(self.examples), size=batch_size)
        return self.collate([self.examples[i] for i in idx])

    def sequential_batches(self, batch_size: int, max_batches: int | None = None):
        for n, i in enumerate(range(0, len(self.examples), batch_size)):
            if max_batches is not None and n >= max_batches:
                return
            yield self.collate(self.examples[i : i + batch_size])
