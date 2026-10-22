"""Qwen-Scope sparse autoencoders (SAEs) on saved activations (exploratory).

Qwen-Scope (Qwen, 2026; arXiv 2605.11887; `Qwen/SAE-Res-Qwen3-*`): TopK SAEs on the residual
stream after each transformer block ("resid_post"), k = 50 active features per token. Each
`layer{L}.sae.pt` holds W_enc (d_sae x d), b_enc (d_sae), W_dec (d x d_sae), b_dec (d).

  features f = TopK_k(W_enc x + b_enc)        reconstruction x_hat = W_dec f + b_dec

Index mapping: SAE layer L = output of block L = our `hidden_states[L + 1]` (index 0 is the
embeddings). Our last index has the final RMSNorm applied, so the last SAE layer doesn't match it.
`alignment()` checks the mapping empirically: the matching index should reconstruct best.

Gemma Scope 2 (`google/gemma-scope-2-{4b,12b}-pt`, path
`resid_post/layer_{L}_width_16k_l0_small/params.safetensors`): JumpReLU SAEs on "model.layers.L
.output" (same index mapping). Stored transposed (w_enc d x d_sae); `load` converts to the layout
above and adds a `threshold`, which `encode` uses instead of TopK:

  features f = pre * (pre > threshold),  pre = W_enc x + b_enc

Licence (Qwen-Scope): Qwen's own (see the repo's LICENSE): scientific research use; no generating harmful or
discriminatory content.
"""

from __future__ import annotations

import numpy as np
import torch

REPOS = {
    "Qwen/Qwen3-1.7B-Base": "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
    "Qwen/Qwen3-8B-Base": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
    "Qwen/Qwen3-30B-A3B-Base": "Qwen/SAE-Res-Qwen3-30B-A3B-Base-W32K-L0_50",
}
K = 50
GEMMA_REPOS = {
    "google/gemma-3-4b-pt": "google/gemma-scope-2-4b-pt",
    "google/gemma-3-12b-pt": "google/gemma-scope-2-12b-pt",
}
GEMMA_SAE = "resid_post/layer_{L}_width_16k_l0_small"
# the only resid_post layers Gemma Scope 2 publishes (repo listing, 2026-10-22)
GEMMA_LAYERS = {"google/gemma-3-4b-pt": (9, 17, 22, 29), "google/gemma-3-12b-pt": (12, 24, 31, 41)}


def load(model_id: str, layer: int) -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download

    if model_id in GEMMA_REPOS:
        from safetensors.torch import load_file

        path = hf_hub_download(GEMMA_REPOS[model_id], GEMMA_SAE.format(L=layer) + "/params.safetensors")
        p = {k: v.float() for k, v in load_file(path).items()}
        return {"W_enc": p["w_enc"].T.contiguous(), "b_enc": p["b_enc"],
                "W_dec": p["w_dec"].T.contiguous(), "b_dec": p["b_dec"],
                "threshold": p["threshold"]}  # fmt: skip
    path = hf_hub_download(REPOS[model_id], f"layer{layer}.sae.pt")
    sae = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.float() for k, v in sae.items()}


def encode(sae: dict, X: np.ndarray, k: int = K) -> torch.Tensor:
    """(n, d) activations -> (n, d_sae) sparse feature activations (top-k kept; JumpReLU for
    an SAE with a `threshold`)."""
    x = torch.as_tensor(X, dtype=torch.float32)
    pre = x @ sae["W_enc"].T + sae["b_enc"]
    if "threshold" in sae:
        return pre * (pre > sae["threshold"])
    vals, idx = pre.topk(k, dim=-1)
    f = torch.zeros_like(pre)
    f.scatter_(-1, idx, vals)
    return f


def decode(sae: dict, f: torch.Tensor) -> torch.Tensor:
    return f @ sae["W_dec"].T + sae["b_dec"]


def fvu(sae: dict, X: np.ndarray) -> float:
    """Fraction of variance unexplained by the reconstruction (0 = perfect; ~1 = no better than
    the mean). Variance is taken around the items' own mean."""
    x = torch.as_tensor(X, dtype=torch.float32)
    err = ((x - decode(sae, encode(sae, X))) ** 2).sum()
    return float(err / ((x - x.mean(0)) ** 2).sum())


def alignment(sae: dict, X_all: np.ndarray, layer: int, offsets=(-1, 0, 1, 2)) -> dict[int, float]:
    """FVU of SAE layer `layer` applied to our hidden_states[layer + offset] (offset 1 should win)."""
    out = {}
    for o in offsets:
        i = layer + o
        if 0 <= i < X_all.shape[1]:
            out[o] = fvu(sae, X_all[:, i].astype(np.float32))
    return out
