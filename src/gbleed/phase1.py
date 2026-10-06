"""Phase 1: build and validate a spelling eraser.

Spanish -a/-o correlates with gender, so a "gender direction" could just be a spelling
detector. Phase 1 trains a LEACE eraser for -o vs -a on verb pairs (hablo/habla), where
the ending marks person, not gender, and checks whether it removes -o/-a in general by
testing on nonce words it has never seen (brelda/breldo).

Per layer:
  verbs_before / verbs_after   probe for -o/-a on verbs (5-fold, grouped by verb), before
                               and after an eraser fit on the training folds only: does
                               erasure generalise to unseen verbs? (sanity check)
  nonce_before / nonce_after   probe trained on train-split nonce stems, tested on
                               test-split stems, before / after the verb eraser. THE test.
  nonce_after_random           same, after erasing a random direction instead (control:
                               removing *any* one direction shouldn't matter).
  nonce_after_plus             eraser fit on verbs + train-split nonce stems (design-doc
                               fallback), tested on test stems.
  nouns_before / nouns_after   gender probe on regular -o/-a nouns (5-fold), before/after
                               the verb eraser. Diagnostic for "did it erase gender too?"
  *_rank2, *_nonceonly         exploratory, added after the pre-registered run: a rank-2
                               eraser (separate verb-ending and nonce-ending concepts, fit
                               on verbs + train nonce stems) and a nonce-only eraser.

Probes: standardised features + L2 logistic regression. Balanced classes, so accuracy
0.5 = chance. "Near chance" is pre-registered (docs/decisions.md) as below CHANCE_UPPER,
the 95% upper bound of chance accuracy for the nonce test set.
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


def _cv_acc(X, y, groups=None, k=5, seed=0) -> float:
    if groups is not None:
        splits = GroupKFold(n_splits=k).split(X, y, groups)
    else:
        splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y)
    accs = [_probe().fit(X[tr], y[tr]).score(X[te], y[te]) for tr, te in splits]
    return float(np.mean(accs))


def _verbs_oos(X, y, groups, k=5) -> float:
    """Eraser fit on training-fold verbs only, probe tested on unseen verbs. (Fitting the
    eraser on all verbs and then cross-validating gives systematically BELOW-chance scores:
    with class means equalised overall, each training fold's mean difference points the
    opposite way to its held-out fold's.)"""
    accs = []
    for a, b in GroupKFold(n_splits=k).split(X, y, groups):
        e = LeaceEraser.fit(torch.from_numpy(X[a]), torch.from_numpy(y[a]).long())
        xa, xb = (e(torch.from_numpy(x)).numpy() for x in (X[a], X[b]))
        accs.append(_probe().fit(xa, y[a]).score(xb, y[b]))
    return float(np.mean(accs))


def _heldout_acc(Xtr, ytr, Xte, yte) -> float:
    return float(_probe().fit(Xtr, ytr).score(Xte, yte))


def _erase_random(X, rng, Xfit) -> np.ndarray:
    """Remove one random direction (orthogonal projection about the fit-data mean)."""
    v = rng.standard_normal(X.shape[1])
    v /= np.linalg.norm(v)
    mu = Xfit.mean(0)
    return X - np.outer((X - mu) @ v, v)


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

    rows = []
    for layer in range(Xv.shape[1]):
        xv, xn, xg = (a[:, layer].astype(np.float64) for a in (Xv, Xn, Xg))
        er = LeaceEraser.fit(torch.from_numpy(xv), torch.from_numpy(yv).long())
        apply = lambda x, e=er: e(torch.from_numpy(x)).numpy()
        xv_e, xn_e, xg_e = apply(xv), apply(xn), apply(xg)
        er_plus = LeaceEraser.fit(
            torch.from_numpy(np.r_[xv, xn[tr]]), torch.from_numpy(np.r_[yv, yn[tr]]).long()
        )
        xn_p = er_plus(torch.from_numpy(xn)).numpy()
        # Exploratory (added after the pre-registered run, see decisions.md):
        # rank-2 eraser with SEPARATE columns for the verb ending and the nonce ending,
        # so the two word types aren't forced onto one direction.
        z2 = np.zeros((len(xv) + int(tr.sum()), 2))
        z2[: len(xv), 0] = yv - 0.5
        z2[len(xv) :, 1] = yn[tr] - 0.5
        er2 = LeaceEraser.fit(torch.from_numpy(np.r_[xv, xn[tr]]), torch.from_numpy(z2))
        xn_2, xg_2 = (er2(torch.from_numpy(x)).numpy() for x in (xn, xg))
        er_n = LeaceEraser.fit(torch.from_numpy(xn[tr]), torch.from_numpy(yn[tr]).long())
        xn_n, xg_n = (er_n(torch.from_numpy(x)).numpy() for x in (xn, xg))
        rand = [
            _heldout_acc(xr[tr], yn[tr], xr[te], yn[te])
            for xr in (_erase_random(xn, rng, xv) for _ in range(N_RANDOM))
        ]
        rows.append(
            {
                "layer": layer,
                "verbs_before": _cv_acc(xv, yv, gv),
                "verbs_after_insample": _cv_acc(xv_e, yv, gv),
                "verbs_after": _verbs_oos(xv, yv, gv),
                "nonce_before": _heldout_acc(xn[tr], yn[tr], xn[te], yn[te]),
                "nonce_after": _heldout_acc(xn_e[tr], yn[tr], xn_e[te], yn[te]),
                "nonce_after_random": float(np.mean(rand)),
                "nonce_after_plus": _heldout_acc(xn_p[tr], yn[tr], xn_p[te], yn[te]),
                "nouns_before": _cv_acc(xg, yg, seed=seed),
                "nouns_after": _cv_acc(xg_e, yg, seed=seed),
                "nonce_after_rank2": _heldout_acc(xn_2[tr], yn[tr], xn_2[te], yn[te]),
                "nouns_after_rank2": _cv_acc(xg_2, yg, seed=seed),
                "nonce_after_nonceonly": _heldout_acc(xn_n[tr], yn[tr], xn_n[te], yn[te]),
                "nouns_after_nonceonly": _cv_acc(xg_n, yg, seed=seed),
            }
        )
        r = rows[-1]
        print(f"layer {layer:2d}  verbs {r['verbs_before']:.2f}->{r['verbs_after']:.2f}  "
              f"nonce {r['nonce_before']:.2f}->{r['nonce_after']:.2f} (rand {r['nonce_after_random']:.2f}, "
              f"plus {r['nonce_after_plus']:.2f}, rank2 {r['nonce_after_rank2']:.2f})  "
              f"nouns {r['nouns_before']:.2f}->{r['nouns_after']:.2f} (rank2 {r['nouns_after_rank2']:.2f})")  # fmt: skip
    df = pd.DataFrame(rows)
    df["nonce_after_near_chance"] = df.nonce_after < chance_upper
    df["nonce_after_rank2_near_chance"] = df.nonce_after_rank2 < chance_upper

    toks = mn["tokens"]
    # Design-doc caveat: if the ending is its own token (brel|da, breld|a), pre-erasure
    # accuracy is trivial (the last token IS the ending); the erasure comparison still holds.
    ending_alone = [t[-1].lstrip("Ġ▁") in ("a", "o") for t in toks]
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "layers.csv", index=False)
    inner = df.iloc[1:-1]  # skip embeddings and the post-norm final state
    summary = {
        "meta": {k: mv[k] for k in mv if k not in ("words", "n_tokens", "shape")},
        "analysis_git": git_state(),
        "n_test_items": n_test,
        "chance_upper": chance_upper,
        "nonce_multitoken_frac": float(np.mean([len(t) > 1 for t in toks])),
        "nonce_ending_own_token_frac": float(np.mean(ending_alone)),
        "layers_near_chance_after": int(inner.nonce_after_near_chance.sum()),
        "n_inner_layers": len(inner),
        "layers_near_chance_after_rank2": int(inner.nonce_after_rank2_near_chance.sum()),
        "mean_inner": inner.drop(
            columns=["layer", "nonce_after_near_chance", "nonce_after_rank2_near_chance"]
        ).mean().round(3).to_dict(),
    }  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "meta"}, indent=2))
    return df
