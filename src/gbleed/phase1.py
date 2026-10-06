"""Phase 1: build and validate a spelling eraser.

Spanish -a/-o correlates with gender, so a "gender direction" could just be a spelling
detector. Phase 1 trains a LEACE eraser for -o vs -a on verb pairs (hablo/habla), where
the ending marks person, not gender, and checks whether it removes -o/-a in general by
testing on nonce words (brelda/breldo).

THE KEY RULE: a probe used to test an eraser must be trained on words the eraser was NOT
fit on. LEACE makes the class means equal on its fit data, and with equal class means the
optimal logistic regression has zero weights (Belrose et al., Thm 2.3). So a probe trained
on the eraser's own fit data scores chance by construction, whatever information remains.

Per layer (all probes: standardised features + L2 logistic regression; acc and ROC AUC):
  verbs_before / verbs_after    -o/-a on verbs. "after": eraser fit on 4/5 of verbs, fresh
                                probe cross-validated (by verb) WITHIN the unseen 1/5.
  nonce_before / nonce_after    PRE-REGISTERED test: probe trained on train-split nonce
                                stems, tested on test-split stems, before / after the VERB
                                eraser (which never saw nonce words, so this is valid).
  nonce_after_random            same after erasing a random direction (control).
  nonce_cv_<eraser>             probe cross-validated (by stem) WITHIN the test-split stems,
                                which no eraser ever sees. Erasers: none, verb, pooled
                                (verbs + train stems as one concept: design-doc fallback),
                                rank2 (verb ending and nonce ending as two concepts),
                                nonceonly (train stems only). Exploratory, added after the
                                pre-registered run (decisions.md).
  nouns_<eraser>                gender probe on regular -o/-a nouns (5-fold). Diagnostic
                                only: for these nouns, ending = gender.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from concept_erasure import LeaceEraser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import activations as acts
from .models import git_state, load_model, model_slug, pick_device, pick_dtype, run_metadata

STIM = {
    "verbs": "data/stimuli/phase1_verbs_v1.csv",
    "nonce": "data/stimuli/phase1_nonce_v1.csv",
    "nouns": "data/stimuli/phase1_nouns_v1.csv",
}
N_RANDOM = 5  # random-direction controls averaged per layer


def _probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def load_stimuli():
    v = pd.read_csv(STIM["verbs"])
    n = pd.read_csv(STIM["nonce"])
    g = pd.read_csv(STIM["nouns"])
    return {
        # label 1 = ends in -a
        "verbs": (list(v.form_o) + list(v.form_a), np.r_[np.zeros(len(v)), np.ones(len(v))],
                  np.r_[v.lemma, v.lemma]),
        "nonce": (list(n.form_o) + list(n.form_a), np.r_[np.zeros(len(n)), np.ones(len(n))],
                  np.r_[n.stem, n.stem], np.r_[n.split, n.split]),
        "nouns": (list(g.lemma), (g.gender == "f").to_numpy(float)),
    }  # fmt: skip


def extract(model_id: str, device=None, dtype=None) -> None:
    st = load_stimuli()
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["input_format"] = "[<|endoftext|>] + tokens(' ' + word)"
    for name, s in st.items():
        X, toks = acts.last_token_states(model, tok, s[0])
        acts.save(model_id, f"phase1_{name}", X, s[0], toks, {**meta, "stimuli": STIM[name]})
        print(f"{name}: {X.shape}, multi-token words {np.mean([len(t) > 1 for t in toks]):.0%}")


def _fit(x, z) -> LeaceEraser:
    return LeaceEraser.fit(torch.from_numpy(x), torch.from_numpy(z))


def _apply(e, x) -> np.ndarray:
    return x if e is None else e(torch.from_numpy(x)).numpy()


def _scores(Xtr, ytr, Xte, yte) -> tuple[float, float]:
    p = _probe().fit(Xtr, ytr)
    s = p.decision_function(Xte)
    auc = roc_auc_score(yte, s) if np.ptp(s) > 0 else 0.5  # constant scores = no information
    return float(p.score(Xte, yte)), float(auc)


def _cv(X, y, groups=None, k=5, seed=0) -> tuple[float, float]:
    if groups is not None:
        splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    else:
        splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y))
    res = [_scores(X[a], y[a], X[b], y[b]) for a, b in splits]
    return float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))


def _verbs_after(X, y, groups, k=5) -> tuple[float, float]:
    """Eraser fit on k-1 folds of verbs; fresh probe cross-validated within the unseen fold."""
    res = []
    for a, b in GroupKFold(n_splits=k).split(X, y, groups):
        e = _fit(X[a], y[a].astype(np.int64))
        res.append(_cv(_apply(e, X[b]), y[b], groups[b], k=k))
    return float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))


def _erase_random(X, rng, Xfit) -> np.ndarray:
    """Remove one random direction (orthogonal projection about the fit-data mean)."""
    v = rng.standard_normal(X.shape[1])
    v /= np.linalg.norm(v)
    return X - np.outer((X - Xfit.mean(0)) @ v, v)


def analyze(model_id: str, out_root: str = "results/phase1", seed: int = 0) -> pd.DataFrame:
    st = load_stimuli()
    Xv, mv = acts.load(model_id, "phase1_verbs")
    Xn, mn = acts.load(model_id, "phase1_nonce")
    Xg, _ = acts.load(model_id, "phase1_nouns")
    _, yv, gv = st["verbs"]
    _, yn, gn, split = st["nonce"]
    _, yg = st["nouns"]
    tr, te = split == "train", split == "test"
    assert not set(gn[tr]) & set(gn[te]), "nonce stem appears in both train and test"
    n_test = int(te.sum())
    chance_upper = 0.5 + 1.96 * math.sqrt(0.25 / n_test)
    rng = np.random.default_rng(seed)
    yvi, yni = yv.astype(np.int64), yn.astype(np.int64)

    rows = []
    for layer in range(Xv.shape[1]):
        xv, xn, xg = (a[:, layer].astype(np.float64) for a in (Xv, Xn, Xg))
        z2 = np.zeros((len(xv) + int(tr.sum()), 2))
        z2[: len(xv), 0] = yv - 0.5
        z2[len(xv) :, 1] = yn[tr] - 0.5
        erasers = {
            "none": None,
            "verb": _fit(xv, yvi),
            "pooled": _fit(np.r_[xv, xn[tr]], np.r_[yvi, yni[tr]]),
            "rank2": _fit(np.r_[xv, xn[tr]], z2),
            "nonceonly": _fit(xn[tr], yni[tr]),
        }
        r = {"layer": layer}
        r["verbs_before"], r["verbs_before_auc"] = _cv(xv, yv, gv)
        r["verbs_after"], r["verbs_after_auc"] = _verbs_after(xv, yv, gv)
        xn_v = _apply(erasers["verb"], xn)
        r["nonce_before"], _ = _scores(xn[tr], yn[tr], xn[te], yn[te])
        r["nonce_after"], r["nonce_after_auc"] = _scores(xn_v[tr], yn[tr], xn_v[te], yn[te])
        r["nonce_after_random"] = float(np.mean([
            _scores(xr[tr], yn[tr], xr[te], yn[te])[0]
            for xr in (_erase_random(xn, rng, xv) for _ in range(N_RANDOM))
        ]))  # fmt: skip
        for name, e in erasers.items():
            r[f"nonce_cv_{name}"], r[f"nonce_cv_{name}_auc"] = _cv(
                _apply(e, xn)[te], yn[te], gn[te]
            )
            r[f"nouns_{name}"], r[f"nouns_{name}_auc"] = _cv(_apply(e, xg), yg, seed=seed)
        rows.append(r)
        print(f"layer {layer:2d}  verbs {r['verbs_before']:.2f}->{r['verbs_after']:.2f}  "
              f"nonce(prereg) {r['nonce_before']:.2f}->{r['nonce_after']:.2f}  nonce-cv AUC: "
              + " ".join(f"{n} {r[f'nonce_cv_{n}_auc']:.2f}" for n in erasers)
              + f"  nouns: {r['nouns_none']:.2f}->{r['nouns_rank2']:.2f} (rank2)")  # fmt: skip
    df = pd.DataFrame(rows)
    df["nonce_after_near_chance"] = df.nonce_after < chance_upper

    toks = mn["tokens"]
    # Design-doc caveat: if the ending is its own token (breld|a), pre-erasure accuracy is
    # trivial (the last token IS the ending); the erasure comparison still holds.
    ending_alone = [t[-1].lstrip("Ġ▁") in ("a", "o") for t in toks]
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "layers.csv", index=False)
    inner = df.iloc[1:-1]  # skip embeddings and the post-norm final state
    summary = {
        "meta": {k: mv[k] for k in mv if k not in ("words", "tokens", "shape")},
        "analysis_git": git_state(),
        "n_test_items": n_test,
        "chance_upper": chance_upper,
        "nonce_multitoken_frac": float(np.mean([len(t) > 1 for t in toks])),
        "nonce_ending_own_token_frac": float(np.mean(ending_alone)),
        "prereg_layers_near_chance_after": int(inner.nonce_after_near_chance.sum()),
        "n_inner_layers": len(inner),
        "mean_inner": inner.drop(columns=["layer", "nonce_after_near_chance"])
        .mean()
        .round(3)
        .to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=2))
    return df
