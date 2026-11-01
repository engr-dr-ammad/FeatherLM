"""Score a chat model on held-out questions (phrasings never seen in training).

    python -m featherlm.benchmarks.chat_eval --model runs/featherlm-nano-chat/best \
        --data data/chat/starter/eval_heldout.jsonl --out results/chat_eval.json

A reply counts as correct when it matches any accepted answer for that intent (exactly,
ignoring case and surrounding whitespace). Shakespeare replies must start with the
expected prefix; maths replies must contain the correct sentence.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..data.chat_starter import INTENTS
from ..inference.engine import InferenceEngine
from ..inference.sampling import SamplingParams


def norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def correct(intent: str, expected: str, reply: str) -> bool:
    r = norm(reply)
    if intent == "shakespeare":
        return r.startswith(norm(expected)) and len(r) > len(norm(expected)) + 10
    if intent in INTENTS:
        return any(r == norm(a) for a in INTENTS[intent][1])
    return norm(expected) in r


def evaluate(model_dir: str, data: str, max_new_tokens: int = 80) -> dict:
    eng = InferenceEngine(model_dir)
    params = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0)
    per = defaultdict(lambda: [0, 0])
    rows = []
    for line in Path(data).read_text().splitlines():
        ex = json.loads(line)
        msgs = ex["messages"][:-1]
        expected = ex["messages"][-1]["content"]
        reply = "".join(eng.stream(eng.encode_chat(msgs), params))
        ok = correct(ex.get("intent", ""), expected, reply)
        per[ex.get("intent", "custom")][0] += ok
        per[ex.get("intent", "custom")][1] += 1
        rows.append({"intent": ex.get("intent"), "question": msgs[-1]["content"], "reply": reply, "correct": ok})
    total = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    return {"model": model_dir, "accuracy": round(total[0] / total[1], 4), "correct": total[0], "total": total[1],
            "per_intent": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per.items())}, "examples": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = evaluate(a.model, a.data)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=2))
    for r in res["examples"]:
        if not r["correct"]:
            print(f"  ✗ [{r['intent']}] {r['question']!r} -> {r['reply'][:100]!r}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
