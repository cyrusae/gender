"""Phase 4 for N languages (`docs/design/new-language-playbook.md`, one-time tooling).

The same estimators and bootstrap as `phase4.analyze`, over any set of languages with a training
loader in `TRAIN`:
  T1  every ordered pair (a's direction scores b's training nouns, AUC within b's cells)
  T2  every unordered pair (cross-fitted cosine of the two class vectors, cross-term shuffle null)
  T3  flipped pairs, for language pairs that have them (currently de-es)
  POOLED direction = all languages stacked (cells = language x cell)
With `langs=("es", "de")` it reproduces `phase4.analyze`'s T1/T2/T3 columns exactly (same random
draws in the same order); `check_against` compares. Decision rules for pairs other than de-es
are not set here: they belong in the pre-registration that adds a language.
"""

from __future__ import annotations

import json
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

from . import activations as acts
from .estimators import boot_within, shuffle_within, split_half_geometry
from .models import git_state, model_slug, progress, stage
from .phase4 import (
    N_BOOT,
    N_SHUFFLE,
    N_SPLITS,
    PAIRS_FINAL,
    PAIRS_POOL,
    SEED,
    _ci,
    _fit,
    _sx,
    de_train,
    es_train,
    paired_auc,
    within_cell_auc,
)

N_SHARED_COVS = (
    4  # every loader's first four covariates: concreteness, its missing flag, Zipf, tokens
)
TRAIN = {"es": es_train, "de": de_train}  # add a language: a loader with the same dict shape


def pooled_n(ds: dict[str, dict]) -> dict:
    """All languages stacked: cells = language:cell; the shared covariates stacked; each
    language's own extra covariates zero elsewhere (for es+de: identical to `phase4.pooled`)."""
    langs = list(ds)
    for lg in langs:
        assert len(ds[lg]["covs"]) >= N_SHARED_COVS, (
            f"{lg}: needs the {N_SHARED_COVS} shared covariates"
        )
    n = {lg: len(ds[lg]["y"]) for lg in langs}
    covs = [np.concatenate([ds[lg]["covs"][i] for lg in langs]) for i in range(N_SHARED_COVS)]
    for lg in langs:
        for c in ds[lg]["covs"][N_SHARED_COVS:]:
            covs.append(np.concatenate([c if o == lg else np.zeros(n[o]) for o in langs]))
    return {"y": np.concatenate([ds[lg]["y"] for lg in langs]),
            "cells": np.concatenate([np.char.add(f"{lg}:", ds[lg]["cells"].astype(str))
                                     for lg in langs]),
            "covs": covs, "offsets": np.cumsum([0] + [n[lg] for lg in langs])[:-1]}  # fmt: skip


def _pairs(model_id: str, sx: str, pairs_path: str):
    pairs = pd.read_csv(pairs_path, keep_default_na=False)
    pool = pd.read_csv(PAIRS_POOL, keep_default_na=False)
    Xp_de, _ = acts.load(model_id, "phase4_pairs_de" + sx)
    Xp_es, _ = acts.load(model_id, "phase4_pairs_es" + sx)
    prow = {pid: i for i, pid in enumerate(pool.pair_id)}
    pi = np.array([prow[x] for x in pairs.pair_id])
    return {"n": len(pairs), "Xde": Xp_de[pi], "Xes": Xp_es[pi],
            "de_f": (pairs.de_gender == "f").to_numpy(),
            "cog": pairs.en_cognate.astype(str).eq("True").to_numpy()}  # fmt: skip


def analyze(model_id: str, langs=("es", "de"), position: str = "last", out_root: str | None = None,
            pairs_path: str = PAIRS_FINAL, n_boot: int = N_BOOT, layers=None) -> pd.DataFrame:  # fmt: skip
    sx = _sx(position)
    ds = {lg: TRAIN[lg](model_id, position) for lg in langs}
    po = pooled_n(ds)
    pr = _pairs(model_id, sx, pairs_path) if {"es", "de"} <= set(langs) else None
    L = ds[langs[0]]["X"].shape[1]
    rng = np.random.default_rng(SEED)
    rows = []
    stage(f"{model_id} [{position}]: Phase 4 (N languages: {', '.join(langs)}) over {L - 2} inner "
          f"layers, {n_boot} bootstrap rounds")  # fmt: skip
    for layer in progress(layers or range(1, L - 1), desc=f"phase4n {position}", unit="layer"):
        X = {lg: ds[lg]["X"][:, layer].astype(np.float64) for lg in langs}
        Xpo = np.concatenate([X[lg] for lg in langs])
        P = None if pr is None else (pr["Xde"][:, layer].astype(np.float64),
                                     pr["Xes"][:, layer].astype(np.float64))  # fmt: skip

        def metrics(tr, te, pair_idx, kind="probe", X=X, Xpo=Xpo, P=P):
            dv = {lg: _fit(X[lg], ds[lg], tr[lg], kind) for lg in langs}
            dv["pooled"] = _fit(Xpo, po, np.concatenate(
                [po["offsets"][i] + tr[lg] for i, lg in enumerate(langs)]), kind)  # fmt: skip
            r = {}
            for a, b in permutations(langs, 2):
                r[f"t1_{a}_on_{b}"] = within_cell_auc(ds[b]["y"][te[b]], X[b][te[b]] @ dv[a],
                                                      ds[b]["cells"][te[b]])  # fmt: skip
            if P is not None:
                Pde, Pes = P
                de_f, cog = pr["de_f"][pair_idx], pr["cog"][pair_idx]
                for nm in ("es", "de", "pooled"):
                    diff = (Pde[pair_idx] - Pes[pair_idx]) @ dv[nm]
                    r[f"t3_{nm}"] = paired_auc(diff, de_f)
                    r[f"t3_{nm}_share"] = float(np.mean(np.sign(diff) == np.where(de_f, 1, -1)))
                    r[f"t3_{nm}_nocog"] = paired_auc(diff[~cog], de_f[~cog])
                r["t3_samelang"] = paired_auc(Pde[pair_idx] @ dv["de"] - Pes[pair_idx] @ dv["es"],
                                              de_f)  # fmt: skip
            return r

        allr = {lg: np.arange(len(ds[lg]["y"])) for lg in langs}
        allp = np.arange(pr["n"]) if pr is not None else None
        point = metrics(allr, allr, allp)
        point_dom = metrics(allr, allr, allp, kind="dom")
        boots = []
        for _ in range(
            n_boot
        ):  # draw order as phase4.analyze: every training set, every test set, pairs
            tr = {lg: boot_within(ds[lg]["cells"], ds[lg]["y"], rng) for lg in langs}
            te = {lg: boot_within(ds[lg]["cells"], ds[lg]["y"], rng) for lg in langs}
            ip = rng.choice(pr["n"], pr["n"]) if pr is not None else None
            boots.append(metrics(tr, te, ip))
        bt = pd.DataFrame(boots)
        row = {"layer": layer, **point, **{f"dom_{k}": v for k, v in point_dom.items()}}
        for k in bt.columns:
            for lev, tag in ((0.95, ""), (0.975, "_holm")):
                row[f"{k}_lo{tag}"], row[f"{k}_hi{tag}"] = _ci(bt[k], lev)
        for a, b in combinations(langs, 2):  # T2 per unordered pair
            sub = {a: ds[a], b: ds[b]}
            p2 = pooled_n(sub)
            Xab = np.concatenate([X[a], X[b]])
            labels = np.r_[np.where(ds[a]["y"] == 1, f"{a}_f", "m"),
                           np.where(ds[b]["y"] == 1, f"{b}_f", "m")]  # fmt: skip
            K = Xab @ Xab.T
            g2 = split_half_geometry(Xab, labels, p2["cells"], p2["covs"], [f"{a}_f", f"{b}_f"],
                                     "m", rng, n_splits=N_SPLITS, gram=K)  # fmt: skip
            null = [split_half_geometry(Xab, shuffle_within(labels, p2["cells"], rng), p2["cells"],
                                        p2["covs"], [f"{a}_f", f"{b}_f"], "m", rng, n_splits=20,
                                        gram=K) for _ in range(N_SHUFFLE)]  # fmt: skip
            row |= {f"t2_{a}_{b}_cos": g2["cos"], f"t2_{a}_{b}_cross": g2["cross"],
                    f"t2_{a}_{b}_cross_null_hi": float(np.nanpercentile([n["cross"] for n in null], 97.5)),
                    f"t2_{a}_{b}_rel_{a}": g2[f"rel_{a}_f"], f"t2_{a}_{b}_rel_{b}": g2[f"rel_{b}_f"]}  # fmt: skip
        rows.append(row)
    res = pd.DataFrame(rows)
    out = Path(out_root or f"results/phase4n{sx}") / model_slug(model_id) / "_".join(langs)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "layers.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary(res, langs) | {
        "model": model_id, "langs": list(langs), "analysis_git": git_state(),
        "n_train": {lg: len(ds[lg]["y"]) for lg in langs}, "n_boot": n_boot}, indent=1, default=float))  # fmt: skip
    return res


def summary(res: pd.DataFrame, langs) -> dict:
    """Layer counts (lower bound > 0.5) per ordered T1 pair and T3; T2 layers above the null.
    No verdicts for new pairs: those rules go in the pre-registration that adds the language."""
    out = {"inner_layers": len(res)}
    for c in [c for c in res.columns if c.endswith("_lo") and c.startswith(("t1_", "t3_"))]:
        k = c[:-3]
        out[f"{k}_layers"] = int((res[c] > 0.5).sum())
        out[f"{k}_layers_holm"] = int((res[f"{k}_lo_holm"] > 0.5).sum())
        out[f"{k}_mean"] = float(res[k].mean())
    for a, b in combinations(langs, 2):
        out[f"t2_{a}_{b}_cos_mean"] = float(res[f"t2_{a}_{b}_cos"].mean())
        out[f"t2_{a}_{b}_layers_above_null"] = int(
            (res[f"t2_{a}_{b}_cross"] > res[f"t2_{a}_{b}_cross_null_hi"]).sum())  # fmt: skip
    return out


def check_against(res: pd.DataFrame, published_layers: str, atol: float = 1e-9) -> pd.DataFrame:
    """Column-by-column max |difference| against a `phase4.analyze` layers.csv (es+de)."""
    pub = pd.read_csv(published_layers)
    pub = pub[pub.layer.isin(res.layer)].set_index("layer").loc[res.layer]
    ren = {"t2_cos": "t2_es_de_cos", "t2_cross": "t2_es_de_cross",
           "t2_cross_null_hi": "t2_es_de_cross_null_hi"}  # fmt: skip
    rows = []
    for c in pub.columns:
        n = ren.get(c, c)
        if n in res:
            d = float(np.nanmax(np.abs(pub[c].to_numpy(float) - res[n].to_numpy(float))))
            rows.append({"column": c, "max_abs_diff": d, "match": d <= atol})
    return pd.DataFrame(rows)
