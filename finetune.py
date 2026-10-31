#!/usr/bin/env python3
"""Fine-tune an existing FeatherLM checkpoint.

Modes (set in the config's `finetune` and `data` sections):
  * continued pretraining : finetune.init_from=<ckpt>, data.format=bin
  * supervised instruction tuning (full) : finetune.init_from=<ckpt>, data.format=jsonl, finetune.mode=full
  * LoRA adapter fine-tuning : finetune.mode=lora (saves adapter + merged weights)

    python finetune.py --config configs/sft_nano.yaml

Developed by Ammad Aslam with AI-assisted tooling.
"""
import argparse
import json
import sys

from featherlm.training.trainer import RunConfig, Trainer, _parse_override


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    overrides = dict(_parse_override(s) for s in args.set)
    if args.no_resume:
        overrides["training.resume"] = False
    rc = RunConfig.from_yaml(args.config, overrides)
    if not rc.finetune.init_from:
        print("error: finetune.init_from must point to a checkpoint (use train.py to train from scratch)", file=sys.stderr)
        return 2
    trainer = Trainer(rc, log=lambda m: print(m, flush=True))
    before = None
    try:
        from featherlm.training.checkpoint import load_model
        base, tok, _ = load_model(rc.finetune.init_from)
        _, val = trainer._data(tok)
        before = trainer.evaluate(base.to(trainer.device).train(), val)
        del base
    except Exception as e:  # evaluation is informative only
        print(f"(pre-finetune evaluation skipped: {e})")
    summary = trainer.run()
    report = {"val_loss_before": before, "val_loss_after": summary.get("final_val_loss"), "mode": summary["mode"]}
    print(json.dumps(report, indent=2))
    (trainer.out / "finetune_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
