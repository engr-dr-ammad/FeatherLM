"""Train a byte-level BPE tokenizer.

Usage:
    python -m featherlm.tokenizer.train_tokenizer --input data/raw/*.txt --vocab-size 8192 --out tokenizer.json
"""
from __future__ import annotations

import argparse
import glob
import heapq
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable

from .tokenizer import PRETOKENIZE_PATTERN, SPECIAL_TOKENS, ByteBPETokenizer


def _pairs(word: list[int]):
    return zip(word, word[1:])


def train_bpe(texts: Iterable[str], vocab_size: int = 8192, normalize: str | None = "NFC",
              min_frequency: int = 2, verbose: bool = False) -> ByteBPETokenizer:
    n_special = len(SPECIAL_TOKENS)
    base = n_special + 256
    if vocab_size < base:
        raise ValueError(f"vocab_size must be >= {base}")
    pat = re.compile(PRETOKENIZE_PATTERN)
    counts: Counter[bytes] = Counter()
    for text in texts:
        if normalize:
            text = unicodedata.normalize(normalize, text)
        counts.update(m.group().encode("utf-8") for m in pat.finditer(text))

    words = [[b + n_special for b in w] for w in counts]
    freqs = list(counts.values())
    pair_counts: Counter[tuple[int, int]] = Counter()
    where: dict[tuple[int, int], set[int]] = {}
    for wi, (w, f) in enumerate(zip(words, freqs)):
        for p in _pairs(w):
            pair_counts[p] += f
            where.setdefault(p, set()).add(wi)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    t0 = time.time()
    target = vocab_size - base
    while len(merges) < target and heap:
        neg, pair = heapq.heappop(heap)
        cur = pair_counts.get(pair, 0)
        if cur != -neg:  # stale heap entry
            if cur > 0:
                heapq.heappush(heap, (-cur, pair))
            continue
        if cur < min_frequency:
            break
        new_id = base + len(merges)
        merges.append(pair)
        changed: set[tuple[int, int]] = set()
        for wi in list(where.get(pair, ())):
            w, f = words[wi], freqs[wi]
            for p in _pairs(w):
                pair_counts[p] -= f
                changed.add(p)
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == pair[0] and w[i + 1] == pair[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            words[wi] = out
            for p in _pairs(out):
                pair_counts[p] += f
                where.setdefault(p, set()).add(wi)
                changed.add(p)
        where.pop(pair, None)
        pair_counts.pop(pair, None)
        for p in changed:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            elif p in pair_counts:
                del pair_counts[p]
        if verbose and len(merges) % 1000 == 0:
            print(f"  merges {len(merges)}/{target}  ({time.time() - t0:.1f}s)")
    if verbose:
        print(f"trained {len(merges)} merges in {time.time() - t0:.1f}s -> vocab {base + len(merges)}")
    return ByteBPETokenizer(merges=merges, normalize=normalize)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a FeatherLM byte-level BPE tokenizer")
    ap.add_argument("--input", nargs="+", required=True, help="text files or globs")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted({f for pattern in args.input for f in glob.glob(pattern)})
    if not files:
        raise SystemExit("no input files matched")
    texts = (open(f, encoding="utf-8", errors="replace").read() for f in files)
    tok = train_bpe(texts, args.vocab_size, verbose=True)
    tok.save(args.out)
    print(f"saved {args.out} (vocab_size={tok.vocab_size})")


if __name__ == "__main__":
    main()
