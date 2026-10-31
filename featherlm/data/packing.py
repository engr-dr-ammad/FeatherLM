"""Sequence packing: concatenate tokenized documents into one stream separated by EOS."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def pack_documents(docs: Iterable[list[int]], eos_id: int, dtype=np.uint16) -> np.ndarray:
    """Flatten documents into a single token stream, appending EOS after each document."""
    max_val = np.iinfo(dtype).max
    chunks = []
    for ids in docs:
        if not ids:
            continue
        arr = np.asarray(list(ids) + [eos_id], dtype=np.int64)
        if arr.max() > max_val:
            raise ValueError(f"token id {arr.max()} does not fit in {np.dtype(dtype).name}")
        chunks.append(arr.astype(dtype))
    if not chunks:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(chunks)


def window_starts(n_tokens: int, seq_len: int, stride: int | None = None) -> np.ndarray:
    """Start offsets of non-overlapping (or strided) windows of length seq_len + 1."""
    stride = stride or seq_len
    last = n_tokens - (seq_len + 1)
    if last < 0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, last + 1, stride, dtype=np.int64)
