import threading

import pytest
import torch

from featherlm.inference.engine import InferenceEngine
from featherlm.inference.sampling import (SamplingParams, apply_repetition_penalty, sample_next, top_k_filter,
                                          top_p_filter)
from featherlm.inference.streaming import StopSequenceFilter
from featherlm.model.config import FeatherConfig
from featherlm.model.generation import generate_ids
from featherlm.model.model import FeatherLM
from featherlm.tokenizer.train_tokenizer import train_bpe


def model(backend="feather", max_seq_len=32):
    torch.manual_seed(0)
    return FeatherLM(FeatherConfig(backend=backend, vocab_size=300, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
                                   max_seq_len=max_seq_len, attn_every=2, local_window=8)).eval()


def test_greedy_is_deterministic():
    m = model()
    p = SamplingParams(max_new_tokens=20, temperature=0)
    assert list(generate_ids(m, [5, 6, 7], p)) == list(generate_ids(m, [5, 6, 7], p))


def test_seeded_sampling_is_reproducible():
    m = model()
    a = list(generate_ids(m, [5, 6], SamplingParams(max_new_tokens=30, temperature=1.0, seed=42)))
    b = list(generate_ids(m, [5, 6], SamplingParams(max_new_tokens=30, temperature=1.0, seed=42)))
    c = list(generate_ids(m, [5, 6], SamplingParams(max_new_tokens=30, temperature=1.0, seed=43)))
    assert a == b and a != c


def test_max_tokens_eos_and_cancel():
    m = model()
    assert len(list(generate_ids(m, [1], SamplingParams(max_new_tokens=7, temperature=0)))) == 7
    first = next(generate_ids(m, [1], SamplingParams(max_new_tokens=5, temperature=0)))
    out = list(generate_ids(m, [1], SamplingParams(max_new_tokens=50, temperature=0), eos_ids={first}))
    assert out == [first]
    ev = threading.Event()
    gen = generate_ids(m, [1], SamplingParams(max_new_tokens=50, temperature=0), cancel=ev)
    next(gen)
    ev.set()
    assert list(gen) == []


@pytest.mark.parametrize("backend", ["transformer", "feather"])
def test_generation_beyond_context(backend):
    m = model(backend, max_seq_len=16)
    out = list(generate_ids(m, list(range(10, 30)), SamplingParams(max_new_tokens=40, temperature=0)))
    assert len(out) == 40


def test_filters():
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
    k = top_k_filter(logits, 2)
    assert torch.isinf(k[0]) and torch.isinf(k[3]) and k[1] == 3.0
    p = top_p_filter(torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05])), 0.7)
    assert torch.isfinite(p[:2]).all() and torch.isinf(p[2:]).all()
    assert torch.isfinite(top_p_filter(torch.tensor([10.0, 0.0]), 0.01)[0])  # top token always kept
    r = apply_repetition_penalty(torch.tensor([2.0, -2.0, 1.0]), torch.tensor([0, 1]), 2.0)
    assert r.tolist() == [1.0, -4.0, 1.0]
    assert sample_next(torch.tensor([0.0, 5.0, 1.0]), torch.tensor([]), SamplingParams(temperature=0)) == 1


def test_sampling_param_validation():
    for bad in (dict(temperature=-1), dict(top_p=0), dict(max_new_tokens=0), dict(top_k=-1)):
        with pytest.raises(ValueError):
            SamplingParams(**bad).validate()


def test_stop_sequence_filter():
    tok = train_bpe(["hello world END more text " * 20], vocab_size=300)
    f = StopSequenceFilter(tok, ["END"])
    out, hit = "", False
    for i in tok.encode("hello world END more"):
        piece, hit = f.push(i)
        out += piece
        if hit:
            break
    assert hit and out == "hello world "


def test_engine_end_to_end(trained_model):
    run, _ = trained_model
    eng = InferenceEngine(run / "final")
    text, stats = eng.generate("Ada sees", SamplingParams(max_new_tokens=15, temperature=0))
    assert isinstance(text, str) and stats.completion_tokens >= 1
    assert stats.tokens_per_sec > 0 and stats.rss_mb > 0
    text2, _ = eng.generate("Ada sees", SamplingParams(max_new_tokens=15, temperature=0))
    assert text == text2
    t3, s3 = eng.generate("Ada sees", SamplingParams(max_new_tokens=30, temperature=0, stop=["."]))
    assert "." not in t3


def test_vocab_limit_masks_undecodable_ids():
    m = model()
    with torch.no_grad():
        m.embed.table.weight[250:] += 5.0  # make padded rows attractive
    out = list(generate_ids(m, [5], SamplingParams(max_new_tokens=30, temperature=1.0, seed=0), vocab_limit=250))
    assert max(out) < 250


def test_encode_chat_keeps_recent_turns_within_context(trained_model):
    from featherlm.training.checkpoint import load_model, save_model

    run, _ = trained_model
    m, tok, _ = load_model(run / "final")
    m.cfg.chat_template = "featherlm-v1"
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp()) / "chat"
    save_model(d, m, tok)
    eng = InferenceEngine(d)
    long_turns = []
    for i in range(30):
        long_turns += [{"role": "user", "content": f"question number {i} about cats and dogs"},
                       {"role": "assistant", "content": f"answer number {i} about cats and dogs at the river"}]
    msgs = [{"role": "system", "content": "be brief"}] + long_turns + [{"role": "user", "content": "last question"}]
    ids = eng.encode_chat(msgs, reserve=16)
    assert len(ids) <= m.cfg.max_seq_len - 16
    text = eng.tok.decode(ids, skip_special=False)
    assert "be brief" in text and "last question" in text and "question number 0 " not in text
    assert text.endswith("<|assistant|>")
    short = eng.encode_chat([{"role": "user", "content": "hi"}])
    assert eng.tok.decode(short, skip_special=False).endswith("<|user|>hi<|end|><|assistant|>")
