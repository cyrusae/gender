"""Gemma 3 support: softcapping and embedding scaling match the model's own forward
(tiny random Gemma3ForCausalLM, CPU, float32)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")


def _tiny(cap=5.0):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    cfg = Gemma3TextConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=8, sliding_window=3, final_logit_softcapping=cap)  # fmt: skip
    return Gemma3ForCausalLM(cfg).eval()


def test_softcap_is_active():
    from gbleed.scoring import softcap

    model = _tiny(cap=5.0)
    assert softcap(model, torch.tensor([100.0])).item() == pytest.approx(5.0, abs=1e-4)
    assert torch.equal(softcap(_tiny(cap=None), torch.tensor([100.0])), torch.tensor([100.0]))


@pytest.mark.parametrize("k", [0, 1, 3])
def test_gemma3_steer_batch_matches_hook(k):
    from gbleed import steering as st

    model = _tiny()
    ids = torch.tensor([[2, 6, 7, 8, 9], [2, 11, 12, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    where = torch.zeros_like(ids, dtype=torch.bool)
    where[0, 1:3] = True
    where[1, 1] = True
    slot = mask.sum(1) - 1
    vecs = torch.randn(3, 32)
    alphas = torch.tensor([0.0, 2.0, -3.0])
    item = torch.tensor([0, 1, 1])
    fast = st.steer_batch(model, ids, mask, k, where, slot, vecs, alphas, item)
    for j in range(3):
        i = item[j].item()
        sl = slice(i, i + 1)
        ref = st.hook_reference(model, ids[sl], mask[sl], k, where[sl], slot[sl], vecs[j],
                                alphas[j].item())[0]  # fmt: skip
        assert torch.allclose(fast[j], ref, atol=1e-5)
    plain = torch.log_softmax(model(input_ids=ids[:1], attention_mask=mask[:1]).logits[0, -1], -1)
    assert torch.allclose(fast[0], plain, atol=1e-5)


def test_gemma3_lower_k0_is_scaled_embedding():
    from gbleed import steering as st

    model = _tiny()
    ids = torch.tensor([[2, 6, 7]])
    h = st.lower(model, ids, torch.ones_like(ids), 0)
    hs = model(input_ids=ids, output_hidden_states=True).hidden_states[0]
    assert torch.allclose(h, hs, atol=1e-6)
    raw = model.get_input_embeddings().weight[ids]
    assert not torch.allclose(h, raw)  # sqrt(d) scale applied


def test_gemma3_batch_scorer_matches_forward():
    from gbleed.scoring import continuation_logprobs_batch

    model = _tiny()

    class Tok:
        pad_token_id = 0
        eos_token_id = 1

    import gbleed.scoring as sc

    enc = {"a": [2, 6, 7], "b": [2, 11], " x": [8], " yz": [9, 10]}
    orig = sc._ids
    sc._ids = lambda tok, text, special: enc[text]
    try:
        got = continuation_logprobs_batch(model, Tok(), [("a", " x"), ("b", " yz")], batch_size=2)
    finally:
        sc._ids = orig
    for (p, c), g in zip([("a", " x"), ("b", " yz")], got, strict=True):
        seq = torch.tensor([enc[p] + enc[c]])
        lp = torch.log_softmax(model(input_ids=seq).logits[0].float(), -1)
        n = len(enc[p])
        want = sum(lp[n - 1 + t, enc[c][t]].item() for t in range(len(enc[c])))
        assert g == pytest.approx(want, abs=1e-4)


def test_gemma3_multimodal_class_matches_forward():
    """The 4B/12B checkpoints load as Gemma3ForConditionalGeneration: decoder, head and (absent)
    softcapping must reproduce its own logits."""
    from transformers import AutoModelForCausalLM, Gemma3Config, Gemma3TextConfig, SiglipVisionConfig

    from gbleed import steering as st

    torch.manual_seed(0)
    tc = Gemma3TextConfig(vocab_size=300, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=8, sliding_window=3, final_logit_softcapping=0.1)  # fmt: skip
    vc = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                            num_attention_heads=2, image_size=28, patch_size=14)  # fmt: skip
    cfg = Gemma3Config(text_config=tc, vision_config=vc, mm_tokens_per_image=4,
                       image_token_index=299, boi_token_index=297, eoi_token_index=298)  # fmt: skip
    m = AutoModelForCausalLM.from_config(cfg).eval()
    ids = torch.tensor([[2, 6, 7, 8, 9]])
    mask = torch.ones_like(ids)
    where = torch.zeros_like(ids, dtype=torch.bool)
    ref = torch.log_softmax(m(input_ids=ids).logits[0, -1].float(), -1)
    for k in (0, 2):
        fast = st.steer_batch(m, ids, mask, k, where, torch.tensor([4]), torch.zeros(1, 32),
                              torch.tensor([0.0]), torch.tensor([0]))  # fmt: skip
        assert torch.allclose(fast[0], ref, atol=1e-5)


def test_text_only_shares_multimodal_weights():
    from transformers import (Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig,
                              SiglipVisionConfig)  # fmt: skip

    from gbleed.models import text_only

    torch.manual_seed(0)
    tc = Gemma3TextConfig(vocab_size=300, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=8, sliding_window=3)  # fmt: skip
    vc = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                            num_attention_heads=2, image_size=28, patch_size=14)  # fmt: skip
    mm = Gemma3ForConditionalGeneration(Gemma3Config(
        text_config=tc, vision_config=vc, mm_tokens_per_image=4, image_token_index=299,
        boi_token_index=297, eoi_token_index=298)).eval()  # fmt: skip
    ids = torch.tensor([[2, 6, 7, 8, 9]])
    want = mm(input_ids=ids).logits
    txt = text_only(mm).eval()
    assert txt.config.hidden_size == 32 and txt.config.num_hidden_layers == 4
    assert not any(p.is_meta for p in txt.parameters())
    got = txt(input_ids=ids, output_hidden_states=True)
    assert torch.allclose(got.logits, want, atol=1e-6)
    assert len(got.hidden_states) == 5


def test_gemma_scope_jumprelu_encode():
    from gbleed import sae as S

    torch.manual_seed(0)
    d, m = 8, 20
    s = {"W_enc": torch.randn(m, d), "b_enc": torch.randn(m), "W_dec": torch.randn(d, m),
         "b_dec": torch.randn(d), "threshold": torch.rand(m)}  # fmt: skip
    X = torch.randn(5, d).numpy()
    f = S.encode(s, X)
    pre = torch.as_tensor(X) @ s["W_enc"].T + s["b_enc"]
    assert torch.equal(f != 0, pre > s["threshold"])
    assert torch.allclose(f[f != 0], pre[f != 0])
