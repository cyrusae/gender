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
