import math

import pytest
import torch

from featherlm.model.config import FeatherConfig
from featherlm.model.embeddings import embedding_param_report
from featherlm.model.model import IGNORE_INDEX, FeatherLM
from featherlm.model.recurrent import chunked_scan, reference_scan

BASE = dict(vocab_size=300, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=96, local_window=8,
            attn_every=2, scan_chunk=8)


def make(backend="feather", **kw):
    torch.manual_seed(0)
    return FeatherLM(FeatherConfig(backend=backend, **{**BASE, **kw})).eval()


@pytest.mark.parametrize("backend", ["transformer", "feather"])
def test_forward_shapes_and_loss(backend):
    m = make(backend)
    x = torch.randint(0, 300, (3, 40))
    logits, loss, caches = m(x, x)
    assert logits.shape == (3, 40, 300)
    assert caches is None
    manual = torch.nn.functional.cross_entropy(logits.reshape(-1, 300), x.reshape(-1))
    assert torch.allclose(loss, manual, atol=1e-5)
    assert abs(m(x, torch.randint(0, 300, (3, 40)))[1].item() - math.log(300)) < 1.5  # near-uniform at init


def test_loss_ignores_masked_targets():
    m = make()
    x = torch.randint(0, 300, (2, 20))
    y = x.clone()
    y[:, :10] = IGNORE_INDEX
    logits, loss, _ = m(x, y)
    manual = torch.nn.functional.cross_entropy(logits[:, 10:].reshape(-1, 300), x[:, 10:].reshape(-1))
    assert torch.allclose(loss, manual, atol=1e-5)


@pytest.mark.parametrize("backend", ["transformer", "feather"])
def test_causal_masking(backend):
    m = make(backend)
    x = torch.randint(0, 300, (1, 64))
    x2 = x.clone()
    x2[0, 40:] = torch.randint(0, 300, (24,))
    with torch.no_grad():
        a, b = m(x)[0], m(x2)[0]
    assert torch.allclose(a[0, :40], b[0, :40], atol=1e-5)
    assert not torch.allclose(a[0, 40:], b[0, 40:], atol=1e-3)


def test_local_attention_window_is_enforced():
    # Pure local attention model: token at position t must not see tokens older than t - window + 1.
    m = make("feather", attn_every=1, n_layers=1)
    w = BASE["local_window"]
    x = torch.randint(0, 300, (1, 40))
    x2 = x.clone()
    x2[0, :10] = torch.randint(0, 300, (10,))
    with torch.no_grad():
        a, b = m(x)[0], m(x2)[0]
    assert torch.allclose(a[0, 10 + w - 1 :], b[0, 10 + w - 1 :], atol=1e-5)


@pytest.mark.parametrize("T,chunk", [(1, 4), (7, 4), (33, 8), (50, 16), (64, 20)])
def test_chunked_scan_matches_reference(T, chunk):
    torch.manual_seed(1)
    la = -torch.rand(2, T, 16) * 4.0
    b = torch.randn(2, T, 16)
    h0 = torch.randn(2, 16)
    assert torch.allclose(reference_scan(la, b, h0), chunked_scan(la, b, h0, chunk), atol=1e-4, rtol=1e-4)


def test_recurrent_block_reference_equivalence():
    m = make("feather", attn_every=0)
    blk = m.blocks[0].mixer
    x = torch.randn(2, 37, 64)
    with torch.no_grad():
        fast, _ = blk(x)
        slow, _ = blk(x, use_reference=True)
    assert torch.allclose(fast, slow, atol=1e-5)


@pytest.mark.parametrize("backend", ["transformer", "feather"])
def test_incremental_decoding_matches_full_forward(backend):
    m = make(backend)
    x = torch.randint(0, 300, (2, 50))  # longer than local window (8) and scan chunk (8)
    with torch.no_grad():
        full = m(x)[0]
        caches = m.new_cache()
        out, _, caches = m(x[:, :13], caches=caches)
        outs = [out]
        for t in range(13, 50):
            o, _, caches = m(x[:, t : t + 1], start_pos=t, caches=caches)
            outs.append(o)
    assert torch.allclose(full, torch.cat(outs, 1), atol=1e-4)
    if backend == "feather":
        for kind, c in zip(m.layer_kinds, caches):
            if kind == "local_attn":
                assert c["k"].shape[2] == BASE["local_window"] - 1  # bounded KV cache


def test_parameter_counting_and_sharing():
    m = make("feather", attn_every=0, n_unique_blocks=0)
    shared = make("feather", attn_every=0, n_unique_blocks=1)
    assert m.num_params() == sum(p.numel() for p in m.parameters())
    per_block = sum(p.numel() for p in m.blocks[0].parameters())
    assert m.num_params() - shared.num_params() == 3 * per_block
    assert len(shared.blocks) == 1 and shared.layer_to_block == [0, 0, 0, 0]
    bd = m.param_breakdown()
    assert bd["total"] == m.num_params()


def test_embedding_tying_and_factorization():
    tied = make("transformer")
    untied = make("transformer", tie_embeddings=False)
    fact = make("transformer", embed_dim=16)
    V, d = 300, 64
    assert untied.num_params() - tied.num_params() == V * d
    assert tied.num_params() - fact.num_params() == V * d - (V * 16 + 16 * d)
    rep = embedding_param_report(fact.cfg)
    assert rep["this_config"] == V * 16 + 16 * 64
    x = torch.randint(0, 300, (1, 10))
    assert fact(x)[0].shape == (1, 10, 300)


@pytest.mark.parametrize("backend", ["transformer", "feather"])
def test_gradients_reach_all_parameters(backend):
    m = make(backend).train()
    x = torch.randint(0, 300, (2, 32))
    m(x, x)[1].backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_gradient_checkpointing_same_gradients():
    a = make("feather").train()
    b = make("feather", gradient_checkpointing=True).train()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 300, (2, 32))
    a(x, x)[1].backward()
    b(x, x)[1].backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5), n


@pytest.mark.parametrize("bad", [dict(backend="rnn"), dict(d_model=65), dict(n_heads=3, n_kv_heads=2),
                                 dict(embed_dim=128), dict(n_unique_blocks=9), dict(scan_chunk=32),
                                 dict(dropout=1.0), dict(vocab_size=0)])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        FeatherConfig(**{**BASE, **bad})


def test_config_round_trip(tmp_path):
    c = FeatherConfig(**BASE)
    c.save(tmp_path / "c.json")
    assert FeatherConfig.load(tmp_path / "c.json") == c
    with pytest.raises(ValueError):
        FeatherConfig.from_dict({**c.to_dict(), "nonsense": 1})
