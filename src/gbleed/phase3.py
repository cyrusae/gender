"""Phase 3: German grammatical gender (m/f/n), separate masculine and feminine vectors.

Pre-registration: docs/design/phases-3-5-plan.md ("Phase 2 (v4) and Phase 3 analysis", adopted
2026-10-07). Stimuli: data/stimuli/phase3_final_v3.csv. All analyses use activations residualised
on the stratification cells (ending x loan status) + frequency, token count and concreteness.

Per inner layer:
  axis      m vs f logistic probe (stratified) on strat3's m/f nouns:
              cv_auc        grouped CV (by cell) on the training nouns
              comp_auc      compound_test m/f compounds (gender = head's)
              head_auc      PRIMARY: conflict compounds (first part m/f, head the other) scored
                            against the head's gender; AUC against the first part's = 1 - head_auc
              suffix_p      gender index of feminine-suffix nouns (0 = training masc mean,
                            1 = training fem mean)
  three     three-class probe on strat3: balanced accuracy on all conflict compounds against the
            head's vs the first part's gender (ba_head, ba_first, ba_diff)
  geometry  split-half len2_m, len2_f, cos, rel_m, rel_f on strat3 (neuter = reference), with the
            within-cell shuffle null and bootstrap CIs; the same on matched3 (equal counts); plain
            plug-in versions for comparison.
Readings (inner layers, majority): follows head = head_auc lower bound > 0.5; markedness (1) =
len2_f - len2_m bootstrap lower bound > 0 and above the null's 95th percentile; cosine reported
only if rel_m and rel_f lower bounds > 0 and the cosine CI lies outside the null's central 95%.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from . import activations as acts
from .estimators import (
    auc,
    boot_within,
    fit_logistic,
    nuisance,
    plain_geometry,
    probe_strat,
    residualise,
    shuffle_within,
    split_half_geometry,
)
from .models import (
    git_state,
    load_model,
    model_slug,
    pick_device,
    pick_dtype,
    progress,
    run_metadata,
    stage,
)  # fmt: skip

FINAL = "data/stimuli/phase3_final_v3.csv"
SEED = 0
N_BOOT = 1000  # probe refits (resampling training nouns within cells) and test nouns
N_SPLITS = 200  # split-half averaging (pre-registered)
N_NULL = 1000  # within-cell shuffles (pre-registered)
N_BOOT_GEOM = 1000


def _suffix(position: str) -> str:
    return {"last": "", "after": "_after"}[position]


def _final() -> pd.DataFrame:
    df = pd.read_csv(FINAL, keep_default_na=False)
    tr, te = set(df[df.split == "train"].lemma), set(df[df.split == "test"].lemma)
    assert not tr & te, "train/test overlap"
    return df


def extract(model_id: str, device=None, dtype=None) -> None:
    """Bare nouns, both readout positions in one pass (LAST is unchanged by appending AFTER)."""
    words = sorted(set(_final().lemma))
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["input_format"] = acts.INPUT_FORMAT + " (AFTER: + newline, read there)"
    meta["stimuli"] = FINAL
    stage(f"{model_id}: extracting {len(words)} German nouns at LAST + AFTER")
    X_last, X_after, toks = acts.states_at(model, tok, words, acts.AFTER, desc="phase3")
    acts.save(model_id, "phase3_bare", X_last, words, toks, {**meta, "position": "last"})
    acts.save(model_id, "phase3_bare_after", X_after, words, toks, {**meta, "position": "after"})


def _auc(y, s) -> float:
    return auc(y, s)


def _probe3(X, y, Z):
    R = residualise(X, Z)
    sc = StandardScaler().fit(R)
    lr = fit_logistic(sc.transform(R), y)
    return sc, lr


def _ci(vals) -> tuple[float, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    return (
        (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        if len(v)
        else (np.nan, np.nan)
    )


def analyze(model_id: str, position: str = "last", out_root: str | None = None) -> dict:
    from .lexicon import load_lexicon
    from .norms import rate

    df = _final()
    out_root = out_root or f"results/phase3{_suffix(position)}"
    Xall, meta = acts.load(model_id, "phase3_bare" + _suffix(position))
    row = {w: i for i, w in enumerate(meta["words"])}
    ntok_of = {w: len(t) for w, t in zip(meta["words"], meta["tokens"], strict=True)}
    gl = load_lexicon("de").drop_duplicates("lemma").set_index("lemma").gloss

    def block(sel: pd.DataFrame):
        """Row indices into Xall + covariates for a subset of the final list."""
        idx = np.array([row[w] for w in sel.lemma])
        rated = [rate(gl.get(w, c))[0] for w, c in zip(sel.lemma, sel.concept_en, strict=True)]
        conc = np.array([np.nan if v is None else v for v in rated], dtype=float)
        miss = np.isnan(conc)
        conc[miss] = np.nanmean(conc) if (~miss).any() else 0.0
        covs = [conc, miss.astype(float), pd.to_numeric(sel.zipf).to_numpy(),
                np.array([ntok_of[w] for w in sel.lemma], dtype=float)]  # fmt: skip
        cells = (sel.ending + "|" + sel.loan).to_numpy()
        return idx, covs, cells, sel.gender.to_numpy()

    s3 = df[df.set == "strat3"]
    i3, c3, k3, g3 = block(s3)
    mf = g3 != "n"
    m3 = df[df.set == "matched3"]
    im, cm, km, gm = block(m3)
    comp = df[(df.set == "compound_test") & df.gender.isin(["m", "f", "n"])]
    ic = np.array([row[w] for w in comp.lemma])
    conflict = (comp.conflict.astype(str) == "True").to_numpy()
    first_g = comp.first_gender.to_numpy()
    head_g = comp.gender.to_numpy()
    cf_bin = conflict & np.isin(head_g, ["m", "f"]) & np.isin(first_g, ["m", "f"])
    cf_all = conflict & np.isin(first_g, ["m", "f", "n"])
    no_suffix = (comp.de_suffix == "").to_numpy()
    suf = df[(df.set == "suffix") & (df.gender == "f")]
    isuf = np.array([row[w] for w in suf.lemma])

    L = Xall.shape[1]
    rng = np.random.default_rng(SEED)
    rows = []
    stage(f"{model_id} [{position}]: Phase 3 analysis over {L - 2} inner layers "
          f"(strat3 {dict(pd.Series(g3).value_counts())}, conflict compounds {int(cf_bin.sum())} "
          f"m/f + {int((cf_all & ~cf_bin).sum())} with neuter)")  # fmt: skip
    for layer in progress(range(1, L - 1), desc=f"phase3 {position}", unit="layer"):
        X = Xall[:, layer].astype(np.float64)
        Xs, Xm, Xc, Xsuf = X[i3], X[im], X[ic], X[isuf]
        y_mf = (g3[mf] == "f").astype(int)

        def axis_metrics(tr, Xs=Xs, y_mf=y_mf, Xc=Xc, Xsuf=Xsuf):
            """tr: indices into the m/f training rows (bootstrap or full)."""
            Xt, yt, kt = Xs[mf][tr], y_mf[tr], k3[mf][tr]
            Z = nuisance(kt, [c[mf][tr] for c in c3])
            d = probe_strat(Xt, yt, Z)
            s_tr = Xs[mf] @ d
            mu_m, mu_f = s_tr[y_mf == 0].mean(), s_tr[y_mf == 1].mean()
            sc = Xc @ d
            o = {
                "comp_auc": _auc((head_g[np.isin(head_g, ["m", "f"])] == "f").astype(int),
                                 sc[np.isin(head_g, ["m", "f"])]),
                "head_auc": _auc((head_g[cf_bin] == "f").astype(int), sc[cf_bin]),
                "head_auc_nosuffix": _auc((head_g[cf_bin & no_suffix] == "f").astype(int),
                                          sc[cf_bin & no_suffix]),
                "suffix_p": float(((Xsuf @ d - mu_m) / (mu_f - mu_m)).mean()),
            }  # fmt: skip
            return o, d

        def three_metrics(tr, Xs=Xs, Xc=Xc):
            Z = nuisance(k3[tr], [c[tr] for c in c3])
            sc, lr = _probe3(Xs[tr], g3[tr], Z)
            # Test compounds have no training cells to residualise on, so they are centred by the
            # training mean (residualised training rows have mean 0); descriptive only.
            pred = lr.predict(sc.transform(Xc[cf_all] - Xs[tr].mean(0)))
            return {
                "ba_head": float(balanced_accuracy_score(head_g[cf_all], pred)),
                "ba_first": float(balanced_accuracy_score(first_g[cf_all], pred)),
            }

        full_mf = np.arange(mf.sum())
        o, _ = axis_metrics(full_mf)
        o |= three_metrics(np.arange(len(i3)))
        o["ba_diff"] = o["ba_head"] - o["ba_first"]
        # grouped CV on training nouns
        n_groups = len(set(k3[mf]))
        aucs = []
        for a, b in GroupKFold(n_splits=min(5, n_groups)).split(Xs[mf], y_mf, k3[mf]):
            Z = nuisance(k3[mf][a], [c[mf][a] for c in c3])
            dd = probe_strat(Xs[mf][a], y_mf[a], Z)
            aucs.append(_auc(y_mf[b], Xs[mf][b] @ dd))
        o["cv_auc"] = float(np.nanmean(aucs))
        boots = []
        for _ in range(N_BOOT):
            tb = boot_within(k3[mf], y_mf, rng)
            ob, _ = axis_metrics(tb)
            boots.append(ob)
        for k in ("comp_auc", "head_auc", "head_auc_nosuffix", "suffix_p"):
            o[f"{k}_lo"], o[f"{k}_hi"] = _ci([b[k] for b in boots])
        b3 = []
        for _ in range(N_BOOT // 5):  # three-class: descriptive, fewer rounds
            r3 = three_metrics(boot_within(k3, g3, rng))
            r3["ba_diff"] = r3["ba_head"] - r3["ba_first"]
            b3.append(r3)
        for k in ("ba_head", "ba_first", "ba_diff"):
            o[f"{k}_lo"], o[f"{k}_hi"] = _ci([b[k] for b in b3])

        # geometry (strat3 primary, matched3 equal-count check)
        for tag, Xg, gg, kk, cc in (("", Xs, g3, k3, c3), ("_eq", Xm, gm, km, cm)):
            Kg = Xg @ Xg.T  # Gram matrix, shared by the null and (indexed) the bootstrap
            obs = split_half_geometry(Xg, gg, kk, cc, ["m", "f"], "n", rng, N_SPLITS, gram=Kg)
            o |= {f"{k}{tag}": v for k, v in obs.items()}
            o[f"diff{tag}"] = obs["len2_f"] - obs["len2_m"]
            pl = plain_geometry(Xg, gg, kk, cc, ["m", "f"])
            o |= {f"plain_{k}{tag}": v for k, v in pl.items()}
            null = [split_half_geometry(Xg, shuffle_within(gg, kk, rng), kk, cc, ["m", "f"], "n",
                                        rng, N_SPLITS, gram=Kg) for _ in range(N_NULL)]  # fmt: skip
            nd = [n["len2_f"] - n["len2_m"] for n in null]
            o[f"diff_null95{tag}"] = float(np.percentile(nd, 95))
            o[f"cos_null_lo{tag}"], o[f"cos_null_hi{tag}"] = _ci([n["cos"] for n in null])
            bg = []
            for _ in range(N_BOOT_GEOM):
                bi = boot_within(kk, gg, rng)
                cb = [c[bi] for c in cc]
                bg.append(split_half_geometry(Xg[bi], gg[bi], kk[bi], cb, ["m", "f"], "n", rng,
                                              N_SPLITS, gram=Kg[np.ix_(bi, bi)],
                                              items=bi))  # fmt: skip
            for k in ("len2_m", "len2_f", "cos", "rel_m", "rel_f"):
                o[f"{k}{tag}_lo"], o[f"{k}{tag}_hi"] = _ci([b[k] for b in bg])
            o[f"diff{tag}_lo"], o[f"diff{tag}_hi"] = _ci([b["len2_f"] - b["len2_m"] for b in bg])
        rows.append({"layer": layer, **o})
    res = pd.DataFrame(rows)
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "layers.csv", index=False)
    marked = (res.diff_lo > 0) & (res["diff"] > res.diff_null95)
    interp = ((res.rel_m_lo > 0) & (res.rel_f_lo > 0)
              & ((res.cos_lo > res.cos_null_hi) | (res.cos_hi < res.cos_null_lo)))  # fmt: skip
    n = len(res)
    summary = {
        "model": model_id, "position": position, "meta": {k: meta[k] for k in meta
                                                          if k not in ("words", "tokens", "shape")},
        "analysis_git": git_state(), "inner_layers": n,
        "n": {"strat3": dict(pd.Series(g3).value_counts()), "matched3": dict(pd.Series(gm).value_counts()),
              "conflict_mf": int(cf_bin.sum()), "conflict_all": int(cf_all.sum()), "suffix_f": len(isuf)},
        "follows_head_layers": int((res.head_auc_lo > 0.5).sum()),
        "follows_first_layers": int((res.head_auc_hi < 0.5).sum()),
        "markedness_layers": int(marked.sum()),
        "markedness_eq_layers": int(((res.diff_eq_lo > 0) & (res.diff_eq > res.diff_null95_eq)).sum()),
        "cosine_interpretable_layers": int(interp.sum()),
        "cosine_median_interpretable": float(res.cos[interp].median()) if interp.any() else None,
        "mean": res.drop(columns=["layer"]).mean().round(3).to_dict(),
    }  # fmt: skip
    summary["reading"] = {
        "compounds": "follow the head"
        if summary["follows_head_layers"] > n / 2
        else "follow the first part"
        if summary["follows_first_layers"] > n / 2
        else "neither",
        "markedness_1": "supported" if summary["markedness_layers"] > n / 2 else "not supported",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["reading"]), {k: summary[k] for k in
          ("follows_head_layers", "markedness_layers", "cosine_interpretable_layers")})  # fmt: skip
    return summary
