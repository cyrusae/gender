"""The batched run() and damage() equal their earlier one-chunk-at-a-time versions (tiny random
Qwen3 with the real Qwen3 tokenizer, CPU, float32)."""

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
from gbleed.phase5 import (
    KL_TOKENS,
    TOPK,
    Prompt,
    _encode,
    _head,
    _pad,
    damage,
    head_dtype,
    lower,
    run,
    upper,
)


# ---- reference copies of the pre-2026-10-10 implementations --------------------------
@torch.no_grad()
def ref_run(model, tok, prompts: list[Prompt], k: int, conds: list[tuple[int, int, float]],
        vecs: torch.Tensor, read_ids: list[int] | None, prompt_batch: int = 16,
        row_batch: int = 64, topk: int = TOPK, unit: float = 1.0,
        positions: str = "noun") -> dict:  # fmt: skip
    """conds: (prompt index, vector index, dose in units); vecs [n_vec, d] unit vectors.
    positions: "noun" (the noun's tokens, P13) or "all" (every token of the prompt, P14).
    Returns per condition: log-probs at read_ids [n_cond, len(read_ids)] and top-k ids/logprobs."""
    assert positions in ("noun", "all"), positions
    enc = [_encode(tok, p) for p in prompts]
    by_prompt: dict[int, list[int]] = {}
    for ci, (pi, _, _) in enumerate(conds):
        by_prompt.setdefault(pi, []).append(ci)
    order = sorted(by_prompt, key=lambda i: len(enc[i][0]))
    dev = model.device
    n = len(conds)
    out_lp = np.zeros((n, len(read_ids or [])), np.float32)
    out_top = np.zeros((n, topk), np.int32)
    out_toplp = np.zeros((n, topk), np.float32)
    rid = torch.tensor(read_ids or [0], device=dev)
    vecs = vecs.to(dev)
    for b0 in range(0, len(order), prompt_batch):
        pis = order[b0 : b0 + prompt_batch]
        ids, mask = _pad(tok, [enc[i][0] for i in pis])
        where = torch.zeros_like(ids, dtype=torch.bool)
        for r, i in enumerate(pis):
            where[r, enc[i][1]] = True
        if positions == "all":
            where = mask.bool()
        ids, mask, where = ids.to(dev), mask.to(dev), where.to(dev)
        slot = mask.sum(1) - 1
        h = lower(model, ids, mask, k)
        cis = [c for i in pis for c in by_prompt[i]]
        local = {i: r for r, i in enumerate(pis)}
        for c0 in range(0, len(cis), row_batch):
            chunk = cis[c0 : c0 + row_batch]
            item = torch.tensor([local[conds[c][0]] for c in chunk], device=dev)
            vi = torch.tensor([conds[c][1] for c in chunk], device=dev)
            al = torch.tensor([conds[c][2] * unit for c in chunk], device=dev, dtype=h.dtype)
            hc = h[item].clone()
            hc += (al[:, None, None] * vecs[vi].to(h.dtype)[:, None, :]) * where[item][..., None]
            lp = upper(model, hc, mask[item], k, slot[item])
            if read_ids:
                out_lp[chunk] = lp[:, rid].cpu().numpy()
            t = lp.topk(topk, dim=-1)
            out_top[chunk] = t.indices.cpu().numpy()
            out_toplp[chunk] = t.values.cpu().numpy()
    return {"lp": out_lp, "top": out_top, "toplp": out_toplp}


@torch.no_grad()
def ref_damage(model, tok, prompts: list[Prompt], k: int, conds, vecs, unit: float,
           batch_rows: int = 16, positions: str = "noun") -> np.ndarray:  # fmt: skip
    """Mean KL(unsteered || steered) over the KL_TOKENS positions after the noun, per condition."""
    from gbleed.steering import _only_layers_from

    enc = [_encode(tok, p) for p in prompts]
    dev = model.device
    W, bias = _head(model, head_dtype(model))
    body = model.get_decoder()
    vecs = vecs.to(dev)
    out = np.zeros(len(conds), np.float32)
    by_prompt: dict[int, list[int]] = {}
    for ci, (pi, _, _) in enumerate(conds):
        by_prompt.setdefault(pi, []).append(ci)
    for pi, cis in by_prompt.items():
        ids_l, pos = enc[pi]
        ids, mask = _pad(tok, [ids_l])
        ids, mask = ids.to(dev), mask.to(dev)
        last = pos[-1]
        span = list(range(last, min(last + KL_TOKENS, len(ids_l) - 1)))
        h = lower(model, ids, mask, k)

        def logp(hh, mm, span=span):
            with _only_layers_from(body, k):
                x = body(inputs_embeds=hh, attention_mask=mm, use_cache=False).last_hidden_state
            lg = x[:, span].to(W.dtype) @ W.T
            if bias is not None:
                lg = lg + bias
            return torch.log_softmax(lg.float(), -1)

        base = logp(h, mask)[0]
        for c0 in range(0, len(cis), batch_rows):
            chunk = cis[c0 : c0 + batch_rows]
            hc = h.repeat(len(chunk), 1, 1)
            al = torch.tensor([conds[c][2] * unit for c in chunk], device=dev, dtype=h.dtype)
            vi = torch.tensor([conds[c][1] for c in chunk], device=dev)
            add = al[:, None] * vecs[vi].to(h.dtype)
            where = pos if positions == "noun" else list(range(len(ids_l)))
            hc[:, where] += add[:, None, :]
            lp = logp(hc, mask.repeat(len(chunk), 1))
            kl = (base.exp()[None] * (base[None] - lp)).sum(-1).mean(-1)
            out[chunk] = kl.cpu().numpy()
    return out


def _setup():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8)  # fmt: skip
    model = Qwen3ForCausalLM(cfg).eval()
    prompts = [Prompt("The bridge is very", "bridge"), Prompt("I think the moon is very", "moon"),
               Prompt("Yesterday we talked about the house over dinner and then went home.",
                      "house"), Prompt("Mi puente es muy", "puente")]  # fmt: skip
    vecs = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)
    conds = [(p, v, a) for p in range(4) for v in range(4) for a in (-1.0, 0.0, 0.5)]
    return model, tok, prompts, vecs, conds


@pytest.mark.parametrize("positions", ["noun", "all"])
def test_run_matches_reference(positions):
    model, tok, prompts, vecs, conds = _setup()
    read = [10, 20, 30]
    new = run(model, tok, prompts, 2, conds, vecs, read, prompt_batch=3, row_batch=5,
              topk=7, positions=positions)  # fmt: skip
    old = ref_run(model, tok, prompts, 2, conds, vecs, read, prompt_batch=3, row_batch=5,
                  topk=7, positions=positions)  # fmt: skip
    assert np.allclose(new["lp"], old["lp"], atol=1e-5)
    assert np.allclose(new["toplp"], old["toplp"], atol=1e-5)


@pytest.mark.parametrize("positions", ["noun", "all"])
def test_damage_matches_reference(positions):
    model, tok, prompts, vecs, conds = _setup()
    new = damage(model, tok, prompts, 2, conds, vecs, 1.0, batch_rows=5, positions=positions,
                 prompt_batch=3)  # fmt: skip
    old = ref_damage(model, tok, prompts, 2, conds, vecs, 1.0, batch_rows=5, positions=positions)
    assert np.allclose(new, old, atol=1e-5)
    assert np.allclose(new[[i for i, c in enumerate(conds) if c[2] == 0.0]], 0, atol=1e-6)
