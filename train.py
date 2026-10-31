#!/usr/bin/env python3
"""Train FeatherLM from scratch (or continue a run) on CPU.

    python train.py --config configs/nano_cpu.yaml
    python train.py --config configs/nano_cpu.yaml --set training.max_steps=50 --set output_dir=runs/quick

Developed by Ammad Aslam with AI-assisted tooling.
"""
import argparse
import sys

from featherlm.training.trainer import MemoryBudgetError, RunConfig, Trainer, _parse_override


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML run config")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. training.max_steps=100")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint in output_dir")
    args = ap.parse_args()
    overrides = dict(_parse_override(s) for s in args.set)
    if args.no_resume:
        overrides["training.resume"] = False
    rc = RunConfig.from_yaml(args.config, overrides)
    try:
        Trainer(rc, log=lambda m: print(m, flush=True)).run()
    except MemoryBudgetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
