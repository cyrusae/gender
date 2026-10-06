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


@torch.no_grad()
def last_token_states(model, tok, words: list[str], batch_size: int = 64, desc: str = "extract"):
    """Return (X [n, L+1, d] float32, tokens [n lists of str]) for bare words."""
    pre = prefix_ids(tok)
    seqs = [pre + tok(" " + w, add_special_tokens=False)["input_ids"] for w in words]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = []
    toks = [tok.convert_ids_to_tokens(s[len(pre) :]) for s in seqs]
    for i in progress(range(0, len(seqs), batch_size), desc=desc, unit="batch"):
        batch = seqs[i : i + batch_size]
        width = max(map(len, batch))
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, s in enumerate(batch):  # right-padding; causal attention ignores the pad tail
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        hs = model(
            input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
            output_hidden_states=True,
        ).hidden_states  # fmt: skip
        last = torch.tensor([len(s) - 1 for s in batch], device=model.device)
        stacked = torch.stack(hs, dim=1)  # [b, L+1, t, d]
        sel = stacked[torch.arange(len(batch)), :, last]  # [b, L+1, d]
        out.append(sel.float().cpu().numpy())
    X = np.concatenate(out)
    check_no_sink(X, words)
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
