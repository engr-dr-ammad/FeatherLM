"""Teaching datasets for the chatbot, and building chat fine-tuning runs from them.

Layout:  <data_dir>/chat/<dataset>/examples.jsonl   (one {"id", "messages", "created"} per line)
The built-in starter conversations live in <data_dir>/chat/starter (read-only here).
"""
from __future__ import annotations

import csv
import io
import json
import math
import random
import re
import threading
import time
import uuid
from pathlib import Path

from featherlm.data.chat_starter import _vary
from featherlm.training.checkpoint import safe_child

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")
STARTER = "starter"
MAX_EXAMPLES = 5000
MAX_CONTENT = 2000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class TeachError(ValueError):
    pass


def validate_messages(msgs) -> list[dict]:
    if not isinstance(msgs, list) or not msgs:
        raise TeachError("messages must be a non-empty list")
    out = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
            raise TeachError("each message needs role system, user or assistant")
        c = str(m.get("content", "")).strip()
        if not c:
            raise TeachError("message content must not be empty")
        if len(c) > MAX_CONTENT:
            raise TeachError(f"message content longer than {MAX_CONTENT} characters")
        out.append({"role": m["role"], "content": c})
    if out[-1]["role"] != "assistant":
        raise TeachError("the last message must be the assistant's answer")
    if not any(m["role"] == "user" for m in out):
        raise TeachError("at least one user message is required")
    return out


def parse_upload(filename: str, data: bytes) -> list[list[dict]]:
    """Accepts .jsonl ({messages} | {prompt, completion} | {question, answer}), .csv (question,answer) and
    .txt (blocks of 'Q: ...' / 'A: ...' separated by blank lines)."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise TeachError("file larger than 5 MB")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise TeachError("file must be UTF-8 text") from e
    name = filename.lower()
    convs: list[list[dict]] = []

    def qa(q, a):
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]

    if name.endswith(".jsonl") or name.endswith(".json"):
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise TeachError(f"line {n}: invalid JSON") from e
            if "messages" in row:
                convs.append(row["messages"])
            elif "prompt" in row and "completion" in row:
                convs.append(qa(row["prompt"], row["completion"]))
            elif "question" in row and "answer" in row:
                convs.append(qa(row["question"], row["answer"]))
            else:
                raise TeachError(f"line {n}: need messages, prompt+completion or question+answer")
    elif name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        qk = cols.get("question") or cols.get("prompt")
        ak = cols.get("answer") or cols.get("completion")
        if not qk or not ak:
            raise TeachError("CSV needs columns question,answer (or prompt,completion)")
        for row in reader:
            if (row.get(qk) or "").strip() and (row.get(ak) or "").strip():
                convs.append(qa(row[qk], row[ak]))
    elif name.endswith(".txt"):
        for block in re.split(r"\n\s*\n", text):
            m = re.match(r"\s*Q:\s*(.+?)\n\s*A:\s*(.+)\s*$", block, re.S | re.I)
            if m:
                convs.append(qa(m.group(1).strip(), m.group(2).strip()))
        if not convs:
            raise TeachError("no 'Q: ...' / 'A: ...' pairs found")
    else:
        raise TeachError("upload a .jsonl, .csv or .txt file")
    if not convs:
        raise TeachError("no examples found in the file")
    return [validate_messages(c) for c in convs]


class TeachService:
    def __init__(self, settings) -> None:
        self.s = settings
        self.root = settings.data_dir / "chat"
        self._lock = threading.Lock()

    # ---------------- datasets
    def _dir(self, name: str) -> Path:
        if not NAME_RE.match(name or ""):
            raise TeachError("dataset name: 1-48 chars of letters, digits, '_' or '-'")
        return safe_child(self.root, name)

    def _file(self, name: str) -> Path:
        return self._dir(name) / "examples.jsonl"

    def _read(self, name: str) -> list[dict]:
        if name == STARTER:
            p = self._dir(STARTER) / "train.jsonl"
            if not p.exists():
                return []
            return [{"id": f"s{i}", **json.loads(l)} for i, l in enumerate(p.read_text().splitlines()) if l.strip()]
        p = self._file(name)
        if not p.exists():
            raise KeyError(name)
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    def _write(self, name: str, rows: list[dict]) -> None:
        p = self._file(name)
        tmp = p.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(p)

    def list(self) -> list[dict]:
        out = []
        if self.root.exists():
            for d in sorted(self.root.iterdir()):
                if d.name == STARTER and (d / "train.jsonl").exists():
                    n = sum(1 for l in (d / "train.jsonl").read_text().splitlines() if l.strip())
                    out.append({"name": STARTER, "examples": n, "builtin": True,
                                "updated": (d / "train.jsonl").stat().st_mtime})
                elif (d / "examples.jsonl").exists():
                    f = d / "examples.jsonl"
                    n = sum(1 for l in f.read_text(encoding="utf-8").splitlines() if l.strip())
                    out.append({"name": d.name, "examples": n, "builtin": False, "updated": f.stat().st_mtime})
        return out

    def create(self, name: str) -> dict:
        if name == STARTER:
            raise TeachError("'starter' is the built-in dataset")
        d = self._dir(name)
        with self._lock:
            if (d / "examples.jsonl").exists():
                raise TeachError(f"dataset '{name}' already exists")
            d.mkdir(parents=True, exist_ok=True)
            (d / "examples.jsonl").write_text("")
        return {"name": name, "examples": 0, "builtin": False}

    def ensure(self, name: str) -> None:
        if name != STARTER and not self._file(name).exists():
            self.create(name)

    def delete(self, name: str) -> None:
        if name == STARTER:
            raise TeachError("the built-in dataset cannot be deleted")
        p = self._file(name)
        if not p.exists():
            raise KeyError(name)
        p.unlink()
        try:
            p.parent.rmdir()
        except OSError:
            pass

    def examples(self, name: str, limit: int = 500) -> list[dict]:
        rows = self._read(name)
        return rows[-limit:][::-1]  # newest first

    def add(self, name: str, convs: list[list[dict]]) -> list[dict]:
        if name == STARTER:
            raise TeachError("the built-in dataset is read-only; create your own dataset")
        with self._lock:
            rows = self._read(name)
            if len(rows) + len(convs) > MAX_EXAMPLES:
                raise TeachError(f"a dataset holds at most {MAX_EXAMPLES} examples")
            new = [{"id": uuid.uuid4().hex[:10], "messages": validate_messages(c), "created": time.time()}
                   for c in convs]
            self._write(name, rows + new)
        return new

    def remove(self, name: str, ex_id: str) -> None:
        if name == STARTER:
            raise TeachError("the built-in dataset is read-only")
        with self._lock:
            rows = self._read(name)
            keep = [r for r in rows if r["id"] != ex_id]
            if len(keep) == len(rows):
                raise KeyError(ex_id)
            self._write(name, keep)

    # ---------------- training runs
    def build_run(self, run: str, dataset: str, base_dir: Path, steps: int = 200, mode: str = "full",
                  include_starter: bool = True, seed: int = 0) -> dict:
        """Write training/validation files and return a finetune config for a chat run."""
        if mode not in ("full", "lora"):
            raise TeachError("mode must be 'full' or 'lora'")
        user = [r["messages"] for r in self._read(dataset)]
        if dataset != STARTER and not user:
            raise TeachError("add at least one example before training")
        out_dir = safe_child(self.s.runs_dir, run)
        if out_dir.exists() and any(out_dir.iterdir()):
            raise TeachError(f"a run named '{run}' already exists; choose another name")
        rng = random.Random(seed)
        train: list[dict] = []
        if dataset != STARTER:
            # Repeat taught examples (with light rewording of the user's text) so a few examples are learned
            # reliably, and mix in starter conversations so the bot keeps what it already knew.
            reps = max(4, min(60, math.ceil(400 / len(user))))
            for conv in user:
                for _ in range(reps):
                    train.append({"messages": [{**m, "content": _vary(m["content"], rng)} if m["role"] == "user"
                                               else m for m in conv]})
        starter = [r["messages"] for r in self._read(STARTER)]
        if include_starter or dataset == STARTER:
            k = len(starter) if dataset == STARTER else min(len(starter), max(300, len(train)))
            train += [{"messages": m} for m in rng.sample(starter, k)] if starter else []
        rng.shuffle(train)
        val = [{"messages": c} for c in (user if user else starter[:200])]
        data_dir = out_dir / "teach_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train))
        (data_dir / "val.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val))
        steps = int(steps)
        base_cfg = json.loads((Path(base_dir) / "config.json").read_text())
        seq_len = min(256, int(base_cfg.get("max_seq_len", 256)))  # never longer than the base model's context
        cfg = {
            "model": {"name": run},
            "data": {"path": str(data_dir), "format": "jsonl", "train_file": str(data_dir / "train.jsonl"),
                     "val_file": str(data_dir / "val.jsonl"), "seq_len": seq_len},
            "training": {"seed": seed, "batch_size": 16, "grad_accum": 1, "max_steps": steps,
                         "lr": 3.0e-4 if mode == "full" else 2.0e-3, "warmup_steps": min(20, steps // 5),
                         "min_lr_ratio": 0.1, "weight_decay": 0.0, "eval_interval": max(10, steps // 4),
                         "eval_batches": 8, "log_interval": max(5, steps // 20), "save_interval": max(10, steps // 2),
                         "early_stopping_patience": 0, "memory_budget_mb": self.s.training_memory_budget_mb},
            "finetune": {"init_from": str(base_dir), "mode": mode, "lora_r": 16, "lora_alpha": 32,
                         "set_chat_template": True},
            "output_dir": str(out_dir),
        }
        return {"config": cfg, "train_examples": len(train), "taught_examples": len(user) if dataset != STARTER else 0}
