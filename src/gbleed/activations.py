"""Extract residual-stream activations at the last token of each word, at every layer.

Uses plain Hugging Face `output_hidden_states` (no hook library needed to *read*).
For Qwen3 (checked on 0.6B):
  hidden_states[0]      token embeddings (input to layer 1)
  hidden_states[1..L-1] output of layers 1..L-1 (the residual stream)
  hidden_states[L]      output of the last layer AFTER the final RMSNorm (feeds the
                        unembedding directly), so it's on a different scale.

Each word is presented "bare" (no article or context), but never on an attention-sink position.
Position 0 is a sink (Qwen3-0.6B, layer 10: norm ~6,700 vs ~37 elsewhere), and so, for some
single-token words, is the first token after a document separator. So the input is
  [<|endoftext|>] + tokens("\n") + tokens(" " + word)
(fixed 2026-10-06; the earlier format without the newline put single-token words on a sink).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .models import model_slug, progress

ACT_DIR = Path("activations")


INPUT_FORMAT = "[<|endoftext|>] + tokens('\\n') + tokens(' ' + text); last token"


def prefix_ids(tok) -> list[int]:
    """Document separator, then a newline as a buffer. The first content token after the
    separator can itself be an attention sink (Qwen3-1.7B, layer 8: ~2,500 vs ~10 on one
    outlier dimension, for some single-token words only); the newline absorbs that instead."""
    tid = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return [tid, *tok("\n", add_special_tokens=False)["input_ids"]]


def check_no_sink(X, words, ratio: float = 10.0) -> None:
    """Warn if any measured vector is abnormally large (a sign it sits on a sink position)."""
    norms = np.linalg.norm(X[:, X.shape[1] // 2], axis=-1)  # a middle layer
    med = np.median(norms)
    bad = [w for w, n in zip(words, norms, strict=True) if n > ratio * med]
    if bad:
        print(f"!! {len(bad)} measured vectors are >{ratio:.0f}x the median norm "
              f"(attention-sink position?): {bad[:10]}")  # fmt: skip


AFTER = "\n"  # readout token appended after the word for the AFTER position


@torch.no_grad()
def states_at(model, tok, words: list[str], after: str | None = None, batch_size: int = 64,
              desc: str = "extract"):  # fmt: skip
    """Residual stream at every layer for each text (`prefix + " " + text`).

    Returns (X_last, X_after, tokens): X_last at the text's last token; X_after at the last token
    of `after` appended behind it (None if after is None). The model reads left to right, so
    appending `after` doesn't change X_last; one pass records both readout positions.
    """
    pre = prefix_ids(tok)
    suf = tok(after, add_special_tokens=False)["input_ids"] if after else []
    body = [tok(" " + w, add_special_tokens=False)["input_ids"] for w in words]
    seqs = [pre + b + suf for b in body]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out_last, out_after = [], []
    toks = [tok.convert_ids_to_tokens(b) for b in body]
    for i in progress(range(0, len(seqs), batch_size), desc=desc, unit="batch"):
        batch, bodies = seqs[i : i + batch_size], body[i : i + batch_size]
        width = max(map(len, batch))
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, sq in enumerate(batch):  # right-padding; causal attention ignores the pad tail
            ids[j, : len(sq)] = torch.tensor(sq)
            mask[j, : len(sq)] = 1
        # The transformer body only: the output layer's vocabulary-sized logits aren't needed.
        hs = model.get_decoder()(
            input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
            output_hidden_states=True,
        ).hidden_states  # fmt: skip
        rows = torch.arange(len(batch), device=model.device)
        last = torch.tensor([len(pre) + len(b) - 1 for b in bodies], device=model.device)
        # pick the readout positions per layer, then stack: [b, L+1, d]
        out_last.append(torch.stack([h[rows, last] for h in hs], 1).float().cpu().numpy())
        if suf:
            end = torch.tensor([len(sq) - 1 for sq in batch], device=model.device)
            out_after.append(torch.stack([h[rows, end] for h in hs], 1).float().cpu().numpy())
    X_last = np.concatenate(out_last)
    check_no_sink(X_last, words)
    X_after = np.concatenate(out_after) if suf else None
    if X_after is not None:
        check_no_sink(X_after, [f"{w}+after" for w in words])
    return X_last, X_after, toks


def last_token_states(model, tok, words: list[str], batch_size: int = 64, desc: str = "extract"):
    """Return (X [n, L+1, d] float32, tokens [n lists of str]) at each text's last token."""
    X, _, toks = states_at(model, tok, words, None, batch_size, desc)
    return X, toks


def save(model_id: str, name: str, X: np.ndarray, words: list[str], toks, meta: dict) -> Path:
    d = ACT_DIR / model_slug(model_id)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}.npy", X)
    (d / f"{name}.json").write_text(
        json.dumps(
            {**meta, "words": words, "tokens": toks, "shape": list(X.shape)},
            ensure_ascii=False,
            indent=1,
        )
    )
    return d / f"{name}.npy"


def load(model_id: str, name: str) -> tuple[np.ndarray, dict]:
    d = ACT_DIR / model_slug(model_id)
    return np.load(d / f"{name}.npy"), json.loads((d / f"{name}.json").read_text())
