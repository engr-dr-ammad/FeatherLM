import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(min(2, os.cpu_count() or 1))

from featherlm.data.prepare import prepare, synthetic_corpus  # noqa: E402
from featherlm.training.trainer import RunConfig, Trainer  # noqa: E402

TINY_MODEL = {"name": "featherlm-test", "backend": "feather", "vocab_size": 512, "d_model": 64, "n_layers": 3,
              "n_heads": 4, "n_kv_heads": 2, "max_seq_len": 64, "attn_every": 3, "local_window": 16}


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    prepare(synthetic_corpus(600, seed=0), out, vocab_size=512)
    return out


def make_run(data_dir, out_dir, model=None, **training):
    tr = {"batch_size": 8, "grad_accum": 1, "max_steps": 120, "lr": 3e-3, "warmup_steps": 10, "eval_interval": 40,
          "eval_batches": 4, "log_interval": 20, "save_interval": 40, "memory_budget_mb": 4000, "seed": 0}
    tr.update(training)
    import yaml

    cfg = {"model": model or dict(TINY_MODEL), "data": {"path": str(data_dir), "format": "bin", "seq_len": 64},
           "training": tr, "output_dir": str(out_dir)}
    p = Path(out_dir).with_suffix(".yaml")
    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg))
    return RunConfig.from_yaml(p)


@pytest.fixture(scope="session")
def trained_model(synthetic_data, tmp_path_factory):
    """Train a tiny model once per session; returns (run_dir, summary)."""
    run = tmp_path_factory.mktemp("runs") / "tiny"
    summary = Trainer(make_run(synthetic_data, run), log=lambda *_: None).run()
    return run, summary
