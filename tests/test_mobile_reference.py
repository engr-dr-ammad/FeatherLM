"""The mobile inference loop (mirrored by mobile/reference_runtime.py) must match the PyTorch engine."""
import importlib.util

import pytest

from featherlm.export.onnx_export import export_onnx
from featherlm.inference.engine import InferenceEngine
from featherlm.inference.sampling import SamplingParams

from .conftest import ROOT


def _load_reference():
    spec = importlib.util.spec_from_file_location("reference_runtime", ROOT / "mobile" / "reference_runtime.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reference_runtime_matches_engine(trained_model, tmp_path):
    pytest.importorskip("onnxruntime")
    run, _ = trained_model
    export_onnx(run / "final", tmp_path / "onnx")
    ref = _load_reference()
    eng = InferenceEngine(run / "final")
    mini = ref.MiniTokenizer(tmp_path / "onnx" / "tokenizer.json")
    for text in ("Ada sees a cat at the river.", "3 plus 4 is 7. Count: 1 2 3", "unicode café 東京 🙂"):
        assert mini.encode(text) == eng.tok.encode(text)
    prompt = "Ben likes"
    _, mobile_ids = ref.generate(tmp_path / "onnx", prompt, n_tokens=20, temperature=0.0)
    from featherlm.model.generation import generate_ids

    engine_ids = list(generate_ids(eng.model, eng.tok.encode(prompt), SamplingParams(max_new_tokens=20, temperature=0),
                                   eos_ids={eng.tok.eos_id}, vocab_limit=eng.tok.vocab_size))
    if engine_ids and engine_ids[-1] == eng.tok.eos_id:
        engine_ids = engine_ids[:-1]
    assert mobile_ids == engine_ids
