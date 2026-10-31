import json

import numpy as np
import pytest
import torch

from featherlm.export.onnx_export import export_onnx
from featherlm.export.quantize import (dequantize_int4, dequantize_int8, load_quantized_into, quantize_checkpoint,
                                       quantize_int4, quantize_int8)
from featherlm.model.config import FeatherConfig
from featherlm.model.model import FeatherLM
from featherlm.training.checkpoint import load_model


def test_int8_int4_round_trip_error():
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.05
    q8, s8 = quantize_int8(w)
    assert q8.dtype == torch.int8
    assert (dequantize_int8(q8, s8) - w).abs().max() <= s8.max() / 2 + 1e-7
    q4, s4 = quantize_int4(w)
    assert q4.dtype == torch.uint8 and q4.shape == (64, 64)  # two weights per byte
    err4 = (dequantize_int4(q4, s4) - w).abs().max()
    assert err4 <= s4.float().max() / 2 + 1e-3


@pytest.mark.parametrize("mode", ["fp16", "int8", "int4"])
def test_quantized_checkpoint_outputs_close(trained_model, tmp_path, mode):
    run, _ = trained_model
    info = quantize_checkpoint(run / "final", tmp_path / mode, mode)
    assert info["quantized_bytes"] < info["source_bytes"]
    ref, _, _ = load_model(run / "final")
    q, _, meta = load_model(tmp_path / mode)
    assert meta["quantization"] == mode
    x = torch.randint(3, 300, (2, 48))
    with torch.no_grad():
        a, b = ref(x)[0], q(x)[0]
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    assert agree >= {"fp16": 0.99, "int8": 0.95, "int4": 0.7}[mode]
    if mode in ("int8", "int4"):
        cfg = FeatherConfig.load(tmp_path / mode / "config.json")
        packed = load_quantized_into(FeatherLM(cfg), tmp_path / mode / "model.safetensors", mode, packed=True).eval()
        with torch.no_grad():
            assert torch.allclose(packed(x)[0], b, atol=1e-4)
        resident = sum(t.numel() * t.element_size() for t in list(packed.parameters()) + list(packed.buffers()))
        assert resident < sum(t.numel() * t.element_size() for t in ref.parameters())


@pytest.mark.parametrize("backend", ["feather", "transformer"])
def test_onnx_export_matches_pytorch(trained_model, tmp_path, backend):
    ort = pytest.importorskip("onnxruntime")
    run, _ = trained_model
    src = run / "final"
    if backend == "transformer":
        from featherlm.training.checkpoint import save_model

        m, tok, _ = load_model(src)
        cfg = FeatherConfig(**{**m.cfg.to_dict(), "backend": "transformer"})
        torch.manual_seed(0)
        save_model(tmp_path / "tf", FeatherLM(cfg), tok)
        src = tmp_path / "tf"
    out = tmp_path / f"onnx-{backend}"
    export_onnx(src, out)
    ref, _, _ = load_model(src)
    cfg = json.loads((out / "onnx_config.json").read_text())
    T = cfg["window"]
    x = torch.randint(3, 300, (1, T))
    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input_ids": x.numpy()})[0]
    with torch.no_grad():
        want = ref(x)[0].numpy()
    assert np.abs(got - want).max() < 1e-3
    # Right padding does not change logits at the last real position (causality).
    L = T // 2
    padded = np.full((1, T), cfg["pad_id"], dtype=np.int64)
    padded[0, :L] = x[0, :L].numpy()
    got_pad = sess.run(["logits"], {"input_ids": padded})[0]
    assert np.abs(got_pad[0, L - 1] - want[0, L - 1]).max() < 1e-3
