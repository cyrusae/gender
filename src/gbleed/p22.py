"""P22: the model's own woman/man axis in its output vocabulary, validated against Glasgow
(exploratory; design: phase5-design.md P22). No forward passes: output (unembedding) weights only.

Axis per language = mean of the output-vector differences of suppletive woman/man pairs:
en woman/man, mother/father (also reported with she/he); es mujer/hombre, madre/padre; de
Frau/Mann, Mutter/Vater. Each word's first token with a leading space; multi-token words are
flagged. A word's axis score = cosine of its first-token output vector with the axis (> 0 =
toward "woman"). Validation: Spearman correlation with the Glasgow gender rating (sign flipped
so > 0 = agrees), raw and partial on valence, over the rated adjectives (English) and their
Wiktionary translations (Spanish, German; same English rating).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = {"en": [("woman", "man"), ("mother", "father")],
         "es": [("mujer", "hombre"), ("madre", "padre")],
         "de": [("Frau", "Mann"), ("Mutter", "Vater")]}  # fmt: skip


def output_matrix(model_id: str) -> np.ndarray:
    """The output (unembedding) matrix, read from the checkpoint without loading the model
    (the input embedding when the model ties them)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    name = "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight"
    try:
        idx = json.loads(
            Path(hf_hub_download(model_id, "model.safetensors.index.json")).read_text()
        )
        files = [idx["weight_map"][name]]
    except (OSError, KeyError, ValueError):  # single-file checkpoint
        files = ["model.safetensors"]
    with safe_open(hf_hub_download(model_id, files[0]), framework="pt") as f:
        return f.get_tensor(name).float().numpy()


def _first(tok, w: str) -> tuple[int, int]:
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    return ids[0], len(ids)


def _rank(x):
    return pd.Series(x).rank().to_numpy()


def _partial(x, y, z) -> float:
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    res = [r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0] for r in (rx, ry)]
    return float(np.corrcoef(*res)[0, 1])


def run(model_id: str, out_root: str = "results/p22") -> dict:
    from transformers import AutoTokenizer

    from .models import model_slug
    from .phase5 import family
    from .phase5_stimuli import ADJ_TRANS

    tok = AutoTokenizer.from_pretrained(model_id)
    W = output_matrix(model_id)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    fam = family(model_id)

    def axis(pairs):
        d = [W[_first(tok, f)[0]] - W[_first(tok, m)[0]] for f, m in pairs]
        a = np.mean(d, 0)
        return a / np.linalg.norm(a)

    axes = {lang: axis(p) for lang, p in PAIRS.items()}
    axes["en_she_he"] = axis([*PAIRS["en"], ("she", "he")])
    adj = pd.read_csv(f"data/stimuli/phase5_adjectives_{fam}_v1.csv", keep_default_na=False)
    tr = pd.read_csv(ADJ_TRANS, keep_default_na=False)
    rows, out = [], {"model": model_id, "axis_cos": {}, "validation": {}}
    sets = {"en": (adj.word, adj.GEND, adj.VAL), "en_she_he": (adj.word, adj.GEND, adj.VAL)}
    for lang in ("es", "de"):
        t = tr[tr[lang] != ""]
        sets[lang] = (t[lang], t.GEND, t.VAL)
    for nm, (words, gend, val) in sets.items():
        ids = [_first(tok, w) for w in words]
        s = Wn[[i for i, _ in ids]] @ axes[nm]
        g, v = -pd.to_numeric(gend).to_numpy(), pd.to_numeric(val).to_numpy()
        single = np.array([n == 1 for _, n in ids])
        out["validation"][nm] = {
            "n": len(s), "n_single_token": int(single.sum()),
            "rho": float(pd.Series(s).corr(pd.Series(g), "spearman")),
            "rho_partial_val": _partial(s, g, v),
            "rho_single_token": float(pd.Series(s[single]).corr(pd.Series(g[single]), "spearman")),
        }  # fmt: skip
        rows += [{"axis": nm, "word": w, "score": float(x), "single_token": bool(o)}
                 for w, x, o in zip(words, s, single, strict=True)]  # fmt: skip
    for a in ("es", "de"):
        for b in ("en", "de"):
            if a != b:
                out["axis_cos"][f"{a}_{b}"] = float(axes[a] @ axes[b])
    sc = pd.DataFrame(rows)
    # cross-language: translation-equivalent adjectives' axis scores
    piv = {}
    for lang in ("en", "es", "de"):
        m = sc[sc.axis == lang].drop_duplicates("word").set_index("word").score
        piv[lang] = pd.Series(tr[lang].map(m).to_numpy(), index=tr.en)
    cross = pd.DataFrame(piv).dropna()
    out["cross_rho"] = {f"{a}_{b}": float(cross[a].corr(cross[b], "spearman"))
                        for a, b in (("es", "en"), ("de", "en"), ("es", "de"))}  # fmt: skip
    out["n_cross"] = len(cross)
    d = Path(out_root) / model_slug(model_id)
    d.mkdir(parents=True, exist_ok=True)
    sc.to_csv(d / "scores.csv", index=False)
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out
