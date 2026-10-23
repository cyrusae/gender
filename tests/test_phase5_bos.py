"""Phase 5 prompts get exactly one <bos> on Gemma and none elsewhere (Qwen3/EuroLLM unchanged)."""

from gbleed.phase5 import Prompt, _encode


class FakeTok:
    """Whitespace tokenizer with offsets; prepends bos when add_special_tokens and adds_bos."""

    def __init__(self, name, bos_token_id, adds_bos):
        self.name_or_path, self.bos_token_id, self.adds_bos = name, bos_token_id, adds_bos

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        ids, offs, i = [], [], 0
        for w in text.split(" "):
            ids.append(100 + len(ids))
            offs.append((i, i + len(w)))
            i += len(w) + 1
        if add_special_tokens and self.adds_bos:
            ids, offs = [self.bos_token_id, *ids], [(0, 0), *offs]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out


P = Prompt("Mi casa es muy", "casa", extra=[7])


def test_gemma_gets_one_bos_and_shifted_noun():
    ids, pos = _encode(FakeTok("google/gemma-3-4b-pt", 2, True), P)
    assert ids == [2, 100, 101, 102, 103, 7]
    assert pos == [2]


def test_other_families_unchanged_even_if_tokenizer_adds_bos():
    for tok in (FakeTok("Qwen/Qwen3-8B-Base", None, False),
                FakeTok("utter-project/EuroLLM-1.7B", 1, True)):  # fmt: skip
        ids, pos = _encode(tok, P)
        assert ids == [100, 101, 102, 103, 7]
        assert pos == [1]


def test_bos_all_flag_adds_bos_only_where_the_tokenizer_would(monkeypatch):
    import gbleed.phase5 as p5

    monkeypatch.setattr(p5, "BOS_ALL", True)
    ids, pos = _encode(FakeTok("utter-project/EuroLLM-1.7B", 1, True), P)
    assert ids == [1, 100, 101, 102, 103, 7] and pos == [2]
    ids, pos = _encode(FakeTok("Qwen/Qwen3-8B-Base", None, False), P)
    assert ids == [100, 101, 102, 103, 7] and pos == [1]
