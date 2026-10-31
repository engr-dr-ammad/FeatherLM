import json

import pytest

from featherlm.tokenizer.tokenizer import SPECIAL_TOKENS, ByteBPETokenizer, IncrementalDecoder
from featherlm.tokenizer.train_tokenizer import train_bpe

CORPUS = ("The quick brown fox jumps over the lazy dog. " * 50 + "Unicode: café naïve 東京 🙂 Ωmega. " * 30
          + "numbers 12345 and 678 tabs\tand\nnewlines\n\n")


@pytest.fixture(scope="module")
def tok():
    return train_bpe([CORPUS], vocab_size=600)


@pytest.mark.parametrize("text", ["", "hello world", "  leading spaces", "café naïve", "東京タワー",
                                  "emoji 🙂🎉👩‍👩‍👧", "tabs\tand\nnewlines\r\n", "never seen: Žluťoučký kůň ﷺ"])
def test_round_trip(tok, text):
    import unicodedata

    assert tok.decode(tok.encode(text)) == unicodedata.normalize("NFC", text)


def test_merges_compress(tok):
    s = "The quick brown fox jumps over the lazy dog."
    assert len(tok.encode(s)) < len(s.encode()) / 2


def test_special_tokens_not_injected_by_default(tok):
    ids = tok.encode("<|user|>hi")
    assert tok.token_id("<|user|>") not in ids
    assert tok.decode(ids) == "<|user|>hi"
    ids2 = tok.encode("<|user|>hi", allow_special=True)
    assert ids2[0] == tok.token_id("<|user|>")


def test_special_layout(tok):
    assert tok.special_tokens == SPECIAL_TOKENS
    assert (tok.pad_id, tok.bos_id, tok.eos_id) == (0, 1, 2)
    assert tok.encode("x", add_bos=True, add_eos=True)[0] == tok.bos_id
    assert tok.decode([tok.bos_id, *tok.encode("x"), tok.eos_id]) == "x"


def test_serialization(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(p)
    t2 = ByteBPETokenizer.load(p)
    assert t2.vocab_size == tok.vocab_size
    assert t2.encode(CORPUS) == tok.encode(CORPUS)


def test_rejects_bad_files(tok, tmp_path):
    d = tok.to_dict()
    d["format"] = "other"
    (tmp_path / "a.json").write_text(json.dumps(d))
    with pytest.raises(ValueError):
        ByteBPETokenizer.load(tmp_path / "a.json")
    d = tok.to_dict()
    d["merges"].append([10**6, 5])
    (tmp_path / "b.json").write_text(json.dumps(d))
    with pytest.raises(ValueError):
        ByteBPETokenizer.load(tmp_path / "b.json")


def test_vocab_size_targets():
    text = " ".join(f"word{i} tok{i % 97}x{i % 13}" for i in range(20000))
    for target in (300, 1000):
        assert train_bpe([text], vocab_size=target).vocab_size == target
    # 8192/16384/32768 targets are respected as upper bounds (small corpora run out of frequent pairs).
    t = train_bpe([text], vocab_size=8192)
    assert t.vocab_size <= 8192
    with pytest.raises(ValueError):
        train_bpe([text], vocab_size=100)


def test_incremental_decoder_utf8(tok):
    text = "a東🙂b"
    ids = tok.encode(text)
    # Force byte-level pieces to split multi-byte characters.
    byte_ids = [b + tok.byte_offset for b in text.encode()]
    for seq in (ids, byte_ids):
        dec = IncrementalDecoder(tok)
        out = "".join(dec.push(i) for i in seq) + dec.flush()
        assert out == text
