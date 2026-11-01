#!/usr/bin/env python3
"""Measure teaching: add answers the way the Teach tab does, train, then test.

Reports accuracy on (a) the taught questions, (b) rephrasings of them never shown in
training, and (c) the starter held-out set before and after (does it forget?).

    python scripts/eval_teaching.py --base runs/featherlm-nano-chat/best --out results/teach_eval.json

Developed by Ammad Aslam with AI-assisted tooling.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.services.settings import Settings  # noqa: E402
from api.services.teach import TeachService  # noqa: E402
from featherlm.benchmarks.chat_eval import evaluate, norm  # noqa: E402
from featherlm.inference.engine import InferenceEngine  # noqa: E402
from featherlm.inference.sampling import SamplingParams  # noqa: E402

ALT = {  # extra phrasings a user might add ("one per line" in the Teach tab); never the test rephrasing
    "What are your opening hours?": ["When do you open?", "What time do you close?"],
    "Where is the shop?": ["What is your address?", "Where can I find you?"],
    "Do you offer delivery?": ["Do you deliver?", "Can I get it delivered?"],
    "How can I contact you?": ["How do I reach you?", "What is your email?"],
    "What is your return policy?": ["Can I return an item?", "How do returns work?"],
    "Who owns the shop?": ["Who runs the shop?", "Whose shop is this?"],
}
FAQ = [  # (taught question, answer, unseen rephrasing)
    ("What are your opening hours?", "We are open Monday to Friday, from 9am to 5pm.", "when are you open"),
    ("Where is the shop?", "The shop is at 12 Main Street, Lisbon.", "where are you located?"),
    ("Do you offer delivery?", "Yes, we deliver within Lisbon for 5 euros.", "can you deliver to my house?"),
    ("How can I contact you?", "Email us at hello@example.com or call 210 000 000.", "what is your phone number?"),
    ("What is your return policy?", "You can return any item within 30 days with the receipt.", "can i return something i bought?"),
    ("Who owns the shop?", "The shop is owned by Ammad Aslam.", "who is the owner?"),
]


def ask(eng, q):
    return "".join(eng.stream(eng.encode_chat([{"role": "user", "content": q}]),
                              SamplingParams(max_new_tokens=60, temperature=0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/featherlm-nano-chat/best")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--heldout", default="data/chat/starter/eval_heldout.jsonl")
    ap.add_argument("--alt", action="store_true", help="also teach 2 extra phrasings per answer")
    ap.add_argument("--out", default="results/teach_eval.json")
    a = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="teach_eval_", dir=ROOT / "runs"))
    s = Settings()
    s.root, s.data_dir, s.runs_dir = ROOT, work / "data", work / "runs"
    (s.data_dir / "chat").mkdir(parents=True)
    shutil.copytree(ROOT / "data/chat/starter", s.data_dir / "chat" / "starter")
    teach = TeachService(s)
    teach.create("shop")
    qs = [(q, ans) for q, ans, _ in FAQ]
    if a.alt:
        qs += [(alt, ans) for q, ans, _ in FAQ for alt in ALT[q]]
    teach.add("shop", [[{"role": "user", "content": q}, {"role": "assistant", "content": ans}] for q, ans in qs])
    built = teach.build_run("shop-bot", "shop", ROOT / a.base, steps=a.steps, mode=a.mode, include_starter=True)
    cfg_path = work / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(built["config"]))
    t0 = time.time()
    proc = subprocess.run([sys.executable, "finetune.py", "--config", str(cfg_path), "--no-resume"], cwd=ROOT,
                          capture_output=True, text=True)
    if proc.returncode:
        sys.exit(proc.stderr[-3000:])
    train_s = time.time() - t0
    model = s.runs_dir / "shop-bot" / "best"
    eng = InferenceEngine(model)
    rows = []
    for q, ans, para in FAQ:
        for kind, question in (("taught", q), ("rephrased", para)):
            reply = ask(eng, question)
            rows.append({"kind": kind, "question": question, "expected": ans, "reply": reply,
                         "correct": norm(reply) == norm(ans)})
    before = evaluate(str(ROOT / a.base), a.heldout)
    after = evaluate(str(model), a.heldout)
    acc = lambda k: sum(r["correct"] for r in rows if r["kind"] == k) / len(FAQ)  # noqa: E731
    res = {"base": a.base, "steps": a.steps, "mode": a.mode, "taught_examples": len(qs),
           "phrasings_per_answer": len(qs) // len(FAQ),
           "training_rows": built["train_examples"], "train_time_s": round(train_s, 1),
           "taught_accuracy": acc("taught"), "rephrased_accuracy": acc("rephrased"),
           "starter_heldout_before": before["accuracy"], "starter_heldout_after": after["accuracy"],
           "rows": rows, "summary": json.loads((s.runs_dir / "shop-bot" / "summary.json").read_text())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "summary")}, indent=2))
    for r in rows:
        print(f"  {'✓' if r['correct'] else '✗'} [{r['kind']}] {r['question']!r} -> {r['reply'][:90]!r}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
