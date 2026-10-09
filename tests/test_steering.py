"""Split, batched steering equals a plain forward hook (tiny random Qwen3, CPU, float32)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")


def _tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8)  # fmt: skip
    return Qwen3ForCausalLM(cfg).eval()


@pytest.mark.parametrize("k", [0, 1, 3])
def test_steer_batch_matches_hook(k):
    from gbleed import steering as st

    model = _tiny()
    ids = torch.tensor([[5, 6, 7, 8, 9], [10, 11, 12, 0, 0]])
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
    # alpha 0 is the unsteered model
    plain = torch.log_softmax(model(input_ids=ids[:1], attention_mask=mask[:1]).logits[0, -1], -1)
    assert torch.allclose(fast[0], plain, atol=1e-5)
