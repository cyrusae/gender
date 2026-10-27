import pytest

from gbleed import chat_format as cf


def _tok(name):
    tr = pytest.importorskip("transformers")
    try:
        return tr.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError:
        pytest.skip(f"{name} tokenizer not cached")


@pytest.mark.parametrize("name", ["Qwen/Qwen3-0.6B", "google/gemma-3-4b-it"])
def test_positions_land_on_noun_and_after(name):
    tok = _tok(name)
    for fmt in cf.FORMATS:
        pre, post = cf.scaffold(tok, fmt, "es")
        ids, last, after, ok = cf.encode(tok, "puente", pre, post)
        w = tok("puente", add_special_tokens=False)["input_ids"]
        assert ids[last - len(w) + 1 : last + 1] == w
        assert after == last + 1 and after < len(ids)
        if fmt != "U":  # A: AFTER is the end-of-turn marker
            assert tok.decode([ids[after]]).strip() in ("<end_of_turn>", "<|im_end|>")
        assert ok


def test_qwen_non_thinking_and_lang_prompts():
    tok = _tok("Qwen/Qwen3-0.6B")
    pre, _ = cf.scaffold(tok, "A-lang", "de")
    assert "Sag etwas." in pre and "<think>\n\n</think>" in pre
