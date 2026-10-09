"""Activation steering primitives for Phase 5 (efficient, batched).

Steering adds alpha * v to the residual stream at chosen positions of one layer boundary. Layer
index k follows `activations.py`: k = 0 is the token embeddings, k = 1..L-1 the output of layer
k (1-indexed), i.e. the input to `layers[k]` (0-indexed). Directions fitted on hidden_states[k]
are added at the same k.

Three savings over a plain forward hook with one run per condition:
1. **Split forward.** Nothing below k depends on the steering, so layers[:k] run once per item
   (`lower`) and only layers[k:] run per condition (`upper`).
2. **Conditions in the batch.** Each row of the upper pass carries its own vector and dose, so
   many conditions (doses, real and random directions) share one forward pass.
3. **Output layer only at the slot**, in `scoring.head_dtype` (fp32 with GBLEED_FP32_HEAD=1).

`hook_reference` is the plain one-run-per-condition version, kept for the equivalence test.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from .scoring import _head, head_dtype


class _Stop(Exception):
    pass


def _check(model) -> None:
    mt = model.config.model_type
    if mt.startswith("gemma") or getattr(model.config, "final_logit_softcapping", None):
        raise NotImplementedError(f"{mt}: embedding scaling / softcapping not handled")


def _out(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def lower(model, ids: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """Residual stream at boundary k for every position: [b, t, d] (layers[:k] only)."""
    _check(model)
    body = model.get_decoder()
    if k == 0:
        return body.embed_tokens(ids)
    box = {}

    def grab(_m, _i, o):
        box["h"] = _out(o)
        raise _Stop

    h = body.layers[k - 1].register_forward_hook(grab)
    try:
        body(input_ids=ids, attention_mask=mask, use_cache=False)
    except _Stop:
        pass
    finally:
        h.remove()
    return box["h"]


@contextmanager
def _only_layers_from(body, k: int):
    """Temporarily make the body run layers[k:] only (forward() then takes the boundary-k
    residual stream as inputs_embeds; masks, positions and the final norm are its own)."""
    cfg = body.config
    layers, n, types = body.layers, cfg.num_hidden_layers, getattr(cfg, "layer_types", None)
    body.layers = nn.ModuleList(list(layers)[k:])
    cfg.num_hidden_layers = n - k
    if types is not None:
        cfg.layer_types = types[k:]
    try:
        yield
    finally:
        body.layers, cfg.num_hidden_layers = layers, n
        if types is not None:
            cfg.layer_types = types


@torch.no_grad()
def upper(model, h: torch.Tensor, mask: torch.Tensor, k: int, slot: torch.Tensor) -> torch.Tensor:
    """Run layers[k:] on boundary-k states h [b, t, d]; log-probabilities over the vocabulary at
    position slot[i] of each row (the prediction for the next token): [b, V] float32."""
    body = model.get_decoder()
    with _only_layers_from(body, k):
        last = body(inputs_embeds=h, attention_mask=mask, use_cache=False).last_hidden_state
    W, bias = _head(model, head_dtype(model))
    hs = last[torch.arange(len(h), device=h.device), slot]
    lg = hs.to(W.dtype) @ W.T
    if bias is not None:
        lg = lg + bias
    lg = lg.float()
    if not torch.isfinite(lg).all():
        raise FloatingPointError("Non-finite logits (fp16 overflow?). Retry with --dtype float32.")
    return torch.log_softmax(lg, dim=-1)


@torch.no_grad()
def steer_batch(model, ids, mask, k: int, where: torch.Tensor, slot: torch.Tensor,
                vecs: torch.Tensor, alphas: torch.Tensor, item: torch.Tensor) -> torch.Tensor:  # fmt: skip
    """Many (item, vector, dose) conditions in one upper pass.

    ids, mask [n_items, t]: right-padded prompts; where [n_items, t] bool: positions to steer
    (e.g. the noun's tokens); slot [n_items]: position whose next-token distribution is read.
    vecs [m, d], alphas [m], item [m] (long): condition j adds alphas[j] * vecs[j] to item[j]'s
    steered positions. Returns [m, V] log-probabilities. alpha 0 = the unsteered baseline.
    """
    h = lower(model, ids, mask, k)
    hc = h[item].clone()
    add = alphas.to(hc.dtype)[:, None, None] * vecs.to(hc.dtype)[:, None, :]
    hc += add * where[item].to(hc.dtype)[..., None]
    return upper(model, hc, mask[item], k, slot[item])


@torch.no_grad()
def hook_reference(model, ids, mask, k: int, where, slot, vec, alpha: float) -> torch.Tensor:
    """Plain version for checking: a forward hook adds alpha * vec at boundary k, full model,
    full-vocabulary logits at the slot. [n_items, V]."""
    body = model.get_decoder()
    target = body.embed_tokens if k == 0 else body.layers[k - 1]

    def add(_m, _i, o):
        x = _out(o)
        x = x + alpha * vec.to(x.dtype) * where.to(x.dtype)[..., None]
        return (x, *o[1:]) if isinstance(o, tuple) else x

    hk = target.register_forward_hook(add)
    try:
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    finally:
        hk.remove()
    lg = logits[torch.arange(len(ids), device=ids.device), slot].float()
    return torch.log_softmax(lg, dim=-1)
