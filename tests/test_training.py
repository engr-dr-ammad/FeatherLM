import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from featherlm.data.dataset import CHAT_TEMPLATE, JsonlSFTDataset, TokenStream, format_chat
from featherlm.data.packing import pack_documents, window_starts
from featherlm.inference.sampling import SamplingParams
from featherlm.model.generation import generate_ids
from featherlm.model.model import IGNORE_INDEX
from featherlm.training.checkpoint import load_model, safe_child, save_model
from featherlm.training.lora import apply_lora, merge_lora, merged_state_dict
from featherlm.training.metrics import usable_memory_mb
from featherlm.training.optimizer import lr_lambda_factory
from featherlm.training.trainer import MemoryBudgetError, Trainer

from .conftest import ROOT, TINY_MODEL, make_run


def test_smoke_train_save_reload_generate(trained_model):
    """Spec smoke test: train on synthetic data (CPU), checkpoint, reload, generate."""
    run, s = trained_model
    assert s["status"] == "completed"
    assert s["last_train_loss"] < s["first_train_loss"] * 0.5, s
    assert s["final_val_loss"] < 2.5
    assert s["peak_rss_mb"] > 0
    model, tok, meta = load_model(run / "final")
    assert meta["parameters"] == model.num_params()
    ids = list(generate_ids(model, tok.encode("Ada sees"), SamplingParams(max_new_tokens=12, temperature=0)))
    text = tok.decode(ids)
    assert len(ids) >= 1 and isinstance(text, str)


def test_checkpoint_round_trip_exact(trained_model, tmp_path):
    run, _ = trained_model
    m1, tok, _ = load_model(run / "final")
    save_model(tmp_path / "copy", m1, tok)
    m2, _, _ = load_model(tmp_path / "copy")
    x = torch.randint(0, 300, (2, 30))
    with torch.no_grad():
        assert torch.equal(m1(x)[0], m2(x)[0])
    assert (tmp_path / "copy" / "model.safetensors").exists()
    assert not list((tmp_path / "copy").glob("*.pt"))  # inference checkpoints carry no pickle


def test_resume_continues_from_checkpoint(synthetic_data, tmp_path):
    run = tmp_path / "resume"
    s1 = Trainer(make_run(synthetic_data, run, max_steps=40, save_interval=20), log=lambda *_: None).run()
    assert s1["steps"] == 40
    s2 = Trainer(make_run(synthetic_data, run, max_steps=60, save_interval=20), log=lambda *_: None).run()
    assert s2["steps"] == 60
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    steps = [l["step"] for l in lines if l["type"] == "train"]
    assert max(steps) == 60 and steps == sorted(steps)  # resumed at 41, did not restart from 1


def test_deterministic_training(synthetic_data, tmp_path):
    a = Trainer(make_run(synthetic_data, tmp_path / "a", max_steps=15), log=lambda *_: None).run()
    b = Trainer(make_run(synthetic_data, tmp_path / "b", max_steps=15), log=lambda *_: None).run()
    assert a["last_train_loss"] == b["last_train_loss"]
    wa = load_file(str(tmp_path / "a" / "final" / "model.safetensors"))
    wb = load_file(str(tmp_path / "b" / "final" / "model.safetensors"))
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_memory_budget_enforced(synthetic_data, tmp_path):
    with pytest.raises(MemoryBudgetError):
        Trainer(make_run(synthetic_data, tmp_path / "m", memory_budget_mb=50), log=lambda *_: None).run()
    assert usable_memory_mb(500) <= 500
    assert usable_memory_mb(None, reserve_mb=10**9) == 256.0  # never trusts total RAM


def test_lr_schedule():
    f = lr_lambda_factory(10, 100, 0.1)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(100) == pytest.approx(0.1) and 0.1 < f(55) < 1.0


def test_packing_and_stream(tmp_path):
    arr = pack_documents([[5, 6], [], [7]], eos_id=2)
    assert arr.tolist() == [5, 6, 2, 7, 2]
    assert window_starts(10, 4).tolist() == [0, 4]
    np.arange(100, dtype=np.uint16).tofile(tmp_path / "t.bin")
    ts = TokenStream(tmp_path / "t.bin", 8)
    x, y = ts.random_batch(4, np.random.default_rng(0))
    assert torch.equal(x[:, 1:], y[:, :-1])
    with pytest.raises(ValueError):
        pack_documents([[70000]], eos_id=2)


def test_sft_masking(trained_model, tmp_path):
    run, _ = trained_model
    _, tok, _ = load_model(run / "final")
    ids, mask = format_chat([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], tok)
    assert ids[0] == tok.bos_id and tok.token_id("<|assistant|>") in ids
    # Only assistant content (+ <|end|>) is trained on.
    trained = [i for i, m in zip(ids, mask) if m]
    assert tok.decode(trained) == "yo" and trained[-1] == tok.token_id("<|end|>")
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"prompt": "a", "completion": "b"}) + "\n" +
                 json.dumps({"messages": [{"role": "user", "content": "<|assistant|>x"},
                                          {"role": "assistant", "content": "z"}]}) + "\n")
    ds = JsonlSFTDataset(p, tok, 32)
    x, y = ds.collate(ds.examples)
    assert (y != IGNORE_INDEX).sum() > 0
    # user text containing a special-token string is encoded as plain text
    assert ds.examples[1][0].count(tok.token_id("<|assistant|>")) == 1


def test_lora_and_merge(trained_model):
    run, _ = trained_model
    m, _, _ = load_model(run / "final")
    total = m.num_params()
    n = apply_lora(m, r=4, alpha=8)
    assert 0 < n < total * 0.2
    for name, p in m.named_parameters():
        assert p.requires_grad == ("lora_" in name)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.05)
    x = torch.randint(0, 300, (1, 20))
    with torch.no_grad():
        y_lora = m(x)[0]
        merged = merged_state_dict(m)
        y_merged = merge_lora(m)(x)[0]
    assert torch.allclose(y_lora, y_merged, atol=1e-4)
    assert all("lora" not in k and ".base." not in k for k in merged)


def test_sft_finetune_cli(trained_model, tmp_path):
    """Full SFT via finetune.py sets the chat template and evaluates before/after."""
    run, _ = trained_model
    data = tmp_path / "sft.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"add {a} and {b}"},
                          {"role": "assistant", "content": f"{a} plus {b} is {a + b}."}]}
            for a in range(6) for b in range(6)]
    data.write_text("\n".join(json.dumps(r) for r in rows))
    import yaml

    cfg = {"model": {}, "data": {"path": str(tmp_path), "format": "jsonl", "train_file": str(data),
                                 "val_file": str(data), "seq_len": 64},
           "training": {"batch_size": 8, "grad_accum": 1, "max_steps": 30, "lr": 1e-3, "warmup_steps": 2,
                        "eval_interval": 15, "eval_batches": 3, "log_interval": 10, "save_interval": 30,
                        "memory_budget_mb": 4000},
           "finetune": {"init_from": str(run / "final"), "mode": "lora", "lora_r": 4, "set_chat_template": True},
           "output_dir": str(tmp_path / "sft")}
    (tmp_path / "sft.yaml").write_text(yaml.safe_dump(cfg))
    proc = subprocess.run([sys.executable, "finetune.py", "--config", str(tmp_path / "sft.yaml")], cwd=ROOT,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads((tmp_path / "sft" / "finetune_report.json").read_text())
    assert report["mode"] == "lora_adapter_finetune"
    assert report["val_loss_after"] < report["val_loss_before"]
    m, _, meta = load_model(tmp_path / "sft" / "final")
    assert m.cfg.chat_template == CHAT_TEMPLATE
    assert (tmp_path / "sft" / "final" / "lora_adapter.safetensors").exists()


def test_safe_child(tmp_path):
    assert safe_child(tmp_path, "a/b") == (tmp_path / "a" / "b").resolve()
    for bad in ("../x", "/etc", "a/../../x", ""):
        with pytest.raises(ValueError):
            safe_child(tmp_path, bad)


def test_train_cli_runs(synthetic_data, tmp_path):
    rc_path = tmp_path / "cfg.yaml"
    import yaml

    rc_path.write_text(yaml.safe_dump({"model": TINY_MODEL, "data": {"path": str(synthetic_data), "seq_len": 64},
                                       "training": {"max_steps": 5, "eval_interval": 5, "log_interval": 1,
                                                    "save_interval": 5, "memory_budget_mb": 4000},
                                       "output_dir": str(tmp_path / "cli")}))
    proc = subprocess.run([sys.executable, "train.py", "--config", str(rc_path), "--set", "training.batch_size=4"],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads((tmp_path / "cli" / "summary.json").read_text())["steps"] == 5
    assert (tmp_path / "cli" / "latest" / "optimizer.pt").exists()
