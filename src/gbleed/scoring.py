"""Log-probability scoring primitives."""

from __future__ import annotations

import torch


def _ids(tok, text: str, special: bool) -> list[int]:
    return tok(text, add_special_tokens=special)["input_ids"]


@torch.no_grad()
def _token_logprobs(model, ids: list[int]) -> torch.Tensor:
    """Log P(ids[i] | ids[:i]) for i >= 1, computed in float32."""
    x = torch.tensor([ids], device=model.device)
    logits = model(x).logits[0, :-1].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits (fp16 overflow?). Retry with --dtype float32.")
    lp = torch.log_softmax(logits, dim=-1)
    return lp.gather(1, x[0, 1:, None]).squeeze(1)


@torch.no_grad()
def next_token_logprobs(model, tok, prompt: str) -> torch.Tensor:
    """Full next-token log-distribution after `prompt` (float32, on CPU)."""
    x = torch.tensor([_ids(tok, prompt, special=True)], device=model.device)
    logits = model(x).logits[0, -1].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits (fp16 overflow?). Retry with --dtype float32.")
    return torch.log_softmax(logits, dim=-1).cpu()


def continuation_logprob(model, tok, prompt: str, continuation: str) -> float:
    """log P(continuation | prompt). Prompt and continuation are tokenized
    separately and concatenated, so start `continuation` with a space."""
    p = _ids(tok, prompt, special=True)
    c = _ids(tok, continuation, special=False)
    return _token_logprobs(model, p + c)[len(p) - 1 :].sum().item()


def candidate_logprobs(model, tok, prompt: str, candidates: list[str]) -> list[float]:
    """log P(cand | prompt) for each candidate; one forward pass when every
    candidate is a single token, falling back to full scoring otherwise."""
    cand_ids = [_ids(tok, c, special=False) for c in candidates]
    if all(len(c) == 1 for c in cand_ids):
        lp = next_token_logprobs(model, tok, prompt)
        return [lp[c[0]].item() for c in cand_ids]
    return [continuation_logprob(model, tok, prompt, c) for c in candidates]


def sentence_logprob(model, tok, text: str) -> float:
    """Total log-probability of `text` (first token is unscored; when two
    sentences share a prefix that cancels out in their difference)."""
    return _token_logprobs(model, _ids(tok, text, special=True)).sum().item()


def tokens_of(tok, text: str) -> list[str]:
    ids = _ids(tok, text, special=False)
    return [tok.decode([i]) for i in ids]


HEAD_DTYPE_ENV = "GBLEED_FP32_HEAD"  # =1: output layer (unembedding) in float32, see head_dtype()


def head_dtype(model) -> torch.dtype:
    """dtype for the output layer in batched scoring: the model's own, or float32 when
    GBLEED_FP32_HEAD=1 (session-2 precision fix: bf16 logits carry a few tenths of a nat of
    rounding noise; computing the last projection in fp32 removes most of it). Recorded in
    run metadata as `head_dtype`."""
    import os

    return torch.float32 if os.environ.get(HEAD_DTYPE_ENV) == "1" else model.dtype


def _head(model, dtype: torch.dtype):
    """(weight, bias) of the output layer in `dtype`, cached on the model (fp32 copy made once)."""
    cache = getattr(model, "_gbleed_head", None)
    if cache is None or cache[0] != dtype:
        lin = model.get_output_embeddings()
        w = lin.weight.to(dtype)
        b = lin.bias.to(dtype) if lin.bias is not None else None
        model._gbleed_head = cache = (dtype, w, b)
    return cache[1], cache[2]


@torch.no_grad()
def continuation_logprobs_batch(
    model, tok, requests: list[tuple[str, str]], batch_size: int = 32, desc: str = "scoring"
) -> list[float]:
    """log P(continuation | prompt) for many (prompt, continuation) pairs at once.

    Same quantity as `continuation_logprob`, computed in right-padded batches with an
    attention mask (padding sits after each sequence, so causal attention never sees it).
    Requests are sorted by length to minimise padding; results come back in input order.
    For a single-token continuation this equals the next-token log-probability.

    The transformer body runs on the whole batch; the output layer (vocabulary-sized) runs only
    at the positions that predict continuation tokens, in head_dtype(model), and results are
    summed on the device (one transfer per batch).
    """
    from .models import progress

    if getattr(model.config, "final_logit_softcapping", None):
        raise NotImplementedError("logit softcapping (Gemma) isn't applied by this scorer")
    body = model.get_decoder()
    W, bias = _head(model, head_dtype(model))
    seqs, spans = [], []
    for prompt, cont in requests:
        p = _ids(tok, prompt, special=True)
        c = _ids(tok, cont, special=False)
        seqs.append(p + c)
        spans.append((len(p), len(p) + len(c)))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [0.0] * len(seqs)
    chunks = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    for chunk in progress(chunks, desc=desc, unit="batch"):
        width = max(len(seqs[i]) for i in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        rows, pos = [], []
        for row, i in enumerate(chunk):
            ids[row, : len(seqs[i])] = torch.tensor(seqs[i])
            mask[row, : len(seqs[i])] = 1
            s, e = spans[i]
            rows += [row] * (e - s)
            pos += range(s - 1, e - 1)  # predictions for tokens s..e-1
        dev = model.device
        h = body(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False)
        h = h.last_hidden_state
        rows_t, pos_t = torch.tensor(rows, device=dev), torch.tensor(pos, device=dev)
        lg = h[rows_t, pos_t].to(W.dtype) @ W.T
        if bias is not None:
            lg = lg + bias
        lg = lg.float()
        if not torch.isfinite(lg).all():
            raise FloatingPointError(
                "Non-finite logits (fp16 overflow?). Retry with --dtype float32."
            )
        tgt = ids.to(dev)[rows_t, pos_t + 1]
        lp = torch.log_softmax(lg, dim=-1).gather(1, tgt[:, None]).squeeze(1)
        sums = torch.zeros(len(chunk), device=dev, dtype=lp.dtype).index_add_(0, rows_t, lp)
        for row, v in zip(range(len(chunk)), sums.cpu().tolist(), strict=True):
            out[chunk[row]] = v
    return out
