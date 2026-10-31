"""Dataset preparation: clean -> (train tokenizer) -> tokenize -> pack -> train/val .bin files.

Usage examples:
    # Public-domain Shakespeare (downloaded from the char-rnn repository)
    python -m featherlm.data.prepare --download tinyshakespeare --out data/shakespeare --vocab-size 8192
    # Your own licensed text files
    python -m featherlm.data.prepare --input "corpus/*.txt" --out data/mycorpus --vocab-size 8192
    # Synthetic data for smoke tests (no download)
    python -m featherlm.data.prepare --synthetic 2000 --out data/synthetic --vocab-size 512

Only use text you have the right to use. Nothing here scrapes the web.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np

from ..tokenizer.tokenizer import ByteBPETokenizer
from ..tokenizer.train_tokenizer import train_bpe
from .packing import pack_documents

KNOWN_DATASETS = {
    "tinyshakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "license": "Works of William Shakespeare (public domain); compilation from karpathy/char-rnn (MIT).",
    }
}

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def synthetic_corpus(n_docs: int, seed: int = 0) -> list[str]:
    """Small, deterministic, learnable corpus (simple grammar + arithmetic patterns)."""
    rng = random.Random(seed)
    names = ["Ada", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana"]
    animals = ["cat", "dog", "fox", "owl", "bee", "cow"]
    verbs = ["sees", "likes", "follows", "helps", "finds"]
    places = ["the park", "the river", "the market", "the hill"]
    docs = []
    for _ in range(n_docs):
        lines = []
        for _ in range(rng.randint(2, 5)):
            kind = rng.random()
            if kind < 0.4:
                lines.append(f"{rng.choice(names)} {rng.choice(verbs)} a {rng.choice(animals)} at {rng.choice(places)}.")
            elif kind < 0.7:
                a, b = rng.randint(0, 9), rng.randint(0, 9)
                lines.append(f"{a} plus {b} is {a + b}.")
            else:
                start = rng.randint(1, 5)
                lines.append("Count: " + " ".join(str(start + i) for i in range(5)) + ".")
        docs.append(" ".join(lines))
    return docs


def download(name: str, dest: Path) -> Path:
    info = KNOWN_DATASETS[name]
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.txt"
    if not path.exists():
        print(f"downloading {name} from {info['url']}")
        with urllib.request.urlopen(info["url"], timeout=60) as r:  # noqa: S310 (fixed, known URL)
            path.write_bytes(r.read())
    (dest / "SOURCE.txt").write_text(f"{name}\n{info['url']}\nLicense: {info['license']}\n")
    return path


def prepare(docs: list[str], out_dir: Path, vocab_size: int, val_fraction: float = 0.05,
            tokenizer_path: str | None = None, seed: int = 1337, source: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = [d for d in (clean_text(x) for x in docs) if d]
    if not docs:
        raise ValueError("no non-empty documents")
    rng = random.Random(seed)

    if tokenizer_path:
        tok = ByteBPETokenizer.load(tokenizer_path)
    else:
        # Train the tokenizer on training documents only (avoid val leakage).
        tok = None
    if len(docs) >= 20:
        order = list(range(len(docs)))
        rng.shuffle(order)
        n_val = max(1, int(len(docs) * val_fraction))
        val_docs = [docs[i] for i in order[:n_val]]
        train_docs = [docs[i] for i in order[n_val:]]
    else:
        # Few large documents: split each document by characters (tail goes to validation).
        train_docs, val_docs = [], []
        for d in docs:
            cut = int(len(d) * (1 - val_fraction))
            cut = d.rfind("\n", 0, cut) if d.rfind("\n", 0, cut) > 0 else cut
            train_docs.append(d[:cut])
            val_docs.append(d[cut:].strip())
    if tok is None:
        tok = train_bpe(train_docs, vocab_size, verbose=True)
    tok.save(out_dir / "tokenizer.json")

    train = pack_documents((tok.encode(d) for d in train_docs), tok.eos_id)
    val = pack_documents((tok.encode(d) for d in val_docs if d), tok.eos_id)
    train.tofile(out_dir / "train.bin")
    val.tofile(out_dir / "val.bin")
    meta = {
        "source": source,
        "tokenizer_vocab_size": tok.vocab_size,
        "dtype": "uint16",
        "train_tokens": int(train.size),
        "val_tokens": int(val.size),
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "chars": sum(len(d) for d in docs),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a FeatherLM dataset")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", nargs="+", help="text files or glob patterns (one document per file)")
    src.add_argument("--download", choices=sorted(KNOWN_DATASETS))
    src.add_argument("--synthetic", type=int, metavar="N_DOCS")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--tokenizer", help="reuse an existing tokenizer.json")
    ap.add_argument("--split-paragraphs", action="store_true", help="treat blank-line separated paragraphs as documents")
    args = ap.parse_args()

    out = Path(args.out)
    if args.synthetic:
        docs, source = synthetic_corpus(args.synthetic), f"synthetic:{args.synthetic}"
    elif args.download:
        path = download(args.download, out / "raw")
        docs, source = [path.read_text(encoding="utf-8")], args.download
    else:
        files = sorted({f for p in args.input for f in glob.glob(p)})
        if not files:
            raise SystemExit("no input files matched")
        docs = [Path(f).read_text(encoding="utf-8", errors="replace") for f in files]
        source = f"files:{len(files)}"
    if args.split_paragraphs:
        docs = [p for d in docs for p in re.split(r"\n\s*\n", d) if p.strip()]
    meta = prepare(docs, out, args.vocab_size, args.val_fraction, args.tokenizer, source=source)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
