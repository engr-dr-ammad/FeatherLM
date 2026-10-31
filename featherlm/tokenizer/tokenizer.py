"""Byte-level BPE tokenizer (pure Python, no external dependencies).

Design (established technique, GPT-2 style):
  * Text is UTF-8 encoded; the 256 byte values are base tokens, so any Unicode
    string round-trips exactly and no <unk> is ever needed for normal text.
  * A regex pre-tokenizer splits text into word-like chunks; merges never cross
    chunk boundaries.
  * Special tokens get fixed ids 0..N-1, bytes get N..N+255, merges follow.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]
# Python `re` version of the GPT-2 pre-tokenizer pattern (\w is Unicode-aware).
PRETOKENIZE_PATTERN = r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+"""
FORMAT_VERSION = 1


class ByteBPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None, special_tokens: list[str] | None = None,
                 normalize: str | None = "NFC") -> None:
        self.special_tokens = list(special_tokens or SPECIAL_TOKENS)
        self.n_special = len(self.special_tokens)
        self.byte_offset = self.n_special
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self.normalize = normalize
        self.ranks: dict[tuple[int, int], int] = {p: i for i, p in enumerate(self.merges)}
        self.special_to_id = {t: i for i, t in enumerate(self.special_tokens)}
        self._pat = re.compile(PRETOKENIZE_PATTERN)
        self._special_pat = re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
        # id -> bytes table
        self.vocab: list[bytes] = [t.encode("utf-8") for t in self.special_tokens]
        self.vocab += [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._encode_chunk = lru_cache(maxsize=65536)(self._encode_chunk_uncached)

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_id(self, name: str) -> int:
        return self.special_to_id[name]

    @property
    def pad_id(self) -> int:
        return self.special_to_id["<|pad|>"]

    @property
    def bos_id(self) -> int:
        return self.special_to_id["<|bos|>"]

    @property
    def eos_id(self) -> int:
        return self.special_to_id["<|eos|>"]

    # ------------------------------------------------------------------
    def _encode_chunk_uncached(self, chunk: bytes) -> tuple[int, ...]:
        ids = [b + self.byte_offset for b in chunk]
        ranks = self.ranks
        while len(ids) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            new_id = self.byte_offset + 256 + best_rank
            pair = (ids[best_i], ids[best_i + 1])
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return tuple(ids)

    def _encode_ordinary(self, text: str) -> list[int]:
        if self.normalize:
            text = unicodedata.normalize(self.normalize, text)
        out: list[int] = []
        for m in self._pat.finditer(text):
            out.extend(self._encode_chunk(m.group().encode("utf-8")))
        return out

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False, allow_special: bool = False) -> list[int]:
        """Encode text. Special-token strings are only parsed when allow_special=True
        (so untrusted user text cannot inject control tokens)."""
        ids: list[int] = [self.bos_id] if add_bos else []
        if allow_special:
            for part in self._special_pat.split(text):
                if not part:
                    continue
                if part in self.special_to_id:
                    ids.append(self.special_to_id[part])
                else:
                    ids.extend(self._encode_ordinary(part))
        else:
            ids.extend(self._encode_ordinary(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode_bytes(self, ids: list[int], skip_special: bool = True) -> bytes:
        parts = []
        for i in ids:
            if i < 0 or i >= len(self.vocab):
                raise ValueError(f"token id {i} out of range")
            if skip_special and i < self.n_special:
                continue
            parts.append(self.vocab[i])
        return b"".join(parts)

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self.decode_bytes(ids, skip_special).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": "featherlm-byte-bpe",
            "version": FORMAT_VERSION,
            "special_tokens": self.special_tokens,
            "normalize": self.normalize,
            "pattern": PRETOKENIZE_PATTERN,
            "merges": [list(m) for m in self.merges],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, d: dict) -> "ByteBPETokenizer":
        if d.get("format") != "featherlm-byte-bpe":
            raise ValueError("not a FeatherLM tokenizer file")
        if d.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported tokenizer version {d.get('version')}")
        if d.get("pattern") != PRETOKENIZE_PATTERN:
            raise ValueError("tokenizer was trained with a different pre-tokenizer pattern")
        specials = d["special_tokens"]
        if not isinstance(specials, list) or not all(isinstance(t, str) and t for t in specials):
            raise ValueError("invalid special_tokens")
        merges = d["merges"]
        base = len(specials) + 256
        for i, m in enumerate(merges):
            if (not isinstance(m, list) or len(m) != 2 or not all(isinstance(v, int) for v in m)
                    or not all(0 <= v < base + i for v in m)):
                raise ValueError(f"invalid merge {i}: {m!r}")
        return cls(merges=[tuple(m) for m in merges], special_tokens=specials, normalize=d.get("normalize"))

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text()))


class IncrementalDecoder:
    """Turns a stream of token ids into text without splitting UTF-8 characters."""

    def __init__(self, tok: ByteBPETokenizer) -> None:
        self.tok = tok
        self.buf = b""

    def push(self, token_id: int) -> str:
        self.buf += self.tok.decode_bytes([token_id])
        for cut in range(len(self.buf), max(len(self.buf) - 4, -1), -1):
            try:
                text = self.buf[:cut].decode("utf-8")
            except UnicodeDecodeError:
                continue
            self.buf = self.buf[cut:]
            return text
        # Invalid bytes that will never complete: flush with replacement.
        text, self.buf = self.buf.decode("utf-8", errors="replace"), b""
        return text

    def flush(self) -> str:
        text, self.buf = self.buf.decode("utf-8", errors="replace"), b""
        return text
