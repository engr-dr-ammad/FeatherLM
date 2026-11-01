#!/usr/bin/env python3
"""Generate text with a trained FeatherLM checkpoint (CPU by default).

    python inference.py --model runs/featherlm-nano/final --prompt "ROMEO:"
    python inference.py --model runs/featherlm-nano/final --chat        # interactive terminal
    python inference.py --model runs/featherlm-nano/final --prompt "KING" --temperature 0 --max-new-tokens 50

Press Ctrl+C during generation to cancel it.

Developed by Ammad Aslam with AI-assisted tooling.
"""
import argparse
import sys
import threading

import torch

from featherlm.inference.engine import GenerationStats, InferenceEngine
from featherlm.inference.sampling import SamplingParams


def run_once(engine, ids, params) -> GenerationStats:
    stats = GenerationStats()
    cancel = threading.Event()
    try:
        for piece in engine.stream(ids, params, cancel=cancel, stats=stats):
            sys.stdout.write(piece)
            sys.stdout.flush()
    except KeyboardInterrupt:
        cancel.set()
        print("\n[cancelled]")
    print()
    print(f"[{stats.completion_tokens} tokens | TTFT {stats.time_to_first_token_s * 1000:.0f} ms | "
          f"{stats.tokens_per_sec:.1f} tok/s | RSS {stats.rss_mb:.0f} MB | finish: {stats.finish_reason}]",
          file=sys.stderr)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument("--prompt", help="prompt text (omit with --chat)")
    ap.add_argument("--chat", action="store_true", help="interactive terminal session")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--stop", action="append", default=[])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--history", action="store_true",
                    help="chat: send earlier turns too (small models answer best one question at a time)")
    args = ap.parse_args()
    if not args.chat and args.prompt is None:
        ap.error("--prompt is required unless --chat is given")

    engine = InferenceEngine(args.model, device=args.device, threads=args.threads)
    info = engine.info()
    print(f"[{info['name']} | {info['architecture']} | {info['parameters']:,} params | "
          f"quant={info['quantization']} | ctx={info['context_length']} | device={args.device}]", file=sys.stderr)
    params = SamplingParams(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                            top_p=args.top_p, repetition_penalty=args.repetition_penalty, stop=args.stop,
                            seed=args.seed)

    if not args.chat:
        run_once(engine, engine.encode_prompt(args.prompt), params)
        return 0

    if engine.supports_chat:
        print("Chat mode (instruction-tuned template). Type /exit to quit, /reset to clear.", file=sys.stderr)
    else:
        print("This model is a plain language model (no chat template): each line you type is continued as text. "
              "Type /exit to quit.", file=sys.stderr)
    history: list[dict] = []
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.strip() == "/exit":
            return 0
        if line.strip() == "/reset":
            history = []
            continue
        if engine.supports_chat:
            history.append({"role": "user", "content": line})
            ids = engine.encode_chat(history if args.history else history[-1:])
            pieces = []
            stats = GenerationStats()
            for p in engine.stream(ids, params, stats=stats):
                pieces.append(p)
                sys.stdout.write(p)
                sys.stdout.flush()
            print()
            history.append({"role": "assistant", "content": "".join(pieces)})
        else:
            run_once(engine, engine.encode_prompt(line), params)


if __name__ == "__main__":
    sys.exit(main())
