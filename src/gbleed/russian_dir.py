"""Russian gender direction (docs/design/russian-direction-prereg.md, adopted PI 2026-10-25).

R-A: does a direction fitted on the Russian -ь nouns score Spanish/German nouns by gender, and
vice versa (four co-primary within-cell AUCs, Bonferroni 98.75% bootstrap intervals, majority of
inner layers)? R-B: the confidence-control (c) frequency cosine on the Russian fit, with the
validity gate. Sensitivity: both refitted on concreteness-matched f-m pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/russian_dir")
MODELS = ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-14B-Base")
SEED = 25
N_BOOT = 1000
LEVEL = 0.9875  # Bonferroni over four co-primary tests
CALIPER_SD = 0.25
MIN_PAIRS = 20
R_A = ("ru_on_es", "ru_on_de", "es_on_ru", "de_on_ru")


def ru_train(model_id: str, position: str = "last") -> dict:
    """The reviewed -ь list with the Phase 3 covariates (concreteness, missing flag, Zipf, tokens)."""
    from . import activations as acts
    from .phase4 import RU_FINAL, _conc, _sx

    df = pd.read_csv(RU_FINAL, keep_default_na=False)
    X, meta = acts.load(model_id, "phase4_ru" + _sx(position))
    assert list(meta["words"]) == list(df.lemma)
    conc = _conc("ru", df.lemma, df.concept_en)
    miss = np.isnan(conc)
    conc_raw = conc.copy()
    conc[miss] = np.nanmean(conc)
    covs = [conc, miss.astype(float), pd.to_numeric(df.zipf).to_numpy(),
            np.array([len(t) for t in meta["tokens"]], dtype=float)]  # fmt: skip
    return {"X": X, "y": (df.gender == "f").to_numpy(int), "cells": df.cell.to_numpy(),
            "covs": covs, "lemma": df.lemma.to_numpy(), "conc_raw": conc_raw}  # fmt: skip


def matched_pairs(d: dict, seed: int = SEED) -> list[tuple[int, int]]:
    """Within cells: feminine to masculine 1:1, |dconcreteness| <= CALIPER_SD x SD, rated nouns
    only, greedy from the most abstract feminine noun, nearest match, seeded shuffle for ties."""
    rng = np.random.default_rng(seed)
    c = d["conc_raw"]
    cal = CALIPER_SD * np.nanstd(c)
    pairs = []
    for cell in sorted(set(d["cells"])):
        k = (d["cells"] == cell) & np.isfinite(c)
        fem = rng.permutation(np.flatnonzero(k & (d["y"] == 1))).tolist()
        mas = set(np.flatnonzero(k & (d["y"] == 0)).tolist())
        for i in sorted(fem, key=lambda i: c[i]):
            cand = [j for j in rng.permutation(sorted(mas)).tolist() if abs(c[j] - c[i]) <= cal]
            if cand:
                j = min(cand, key=lambda j: abs(c[j] - c[i]))
                pairs.append((i, j))
                mas.discard(j)
    return pairs


def subset(d: dict, idx: np.ndarray) -> dict:
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(d["y"]) else v)
            for k, v in d.items()} | {"covs": [c[idx] for c in d["covs"]]}  # fmt: skip


def _pool(a: dict, b: dict, na: str, nb: str) -> dict:
    """Two languages stacked: cells prefixed, the four shared covariates stacked, extra
    covariates of either side zero on the other."""
    covs = [np.r_[x, y] for x, y in zip(a["covs"][:4], b["covs"][:4], strict=True)]
    covs += [np.r_[c, np.zeros(len(b["y"]))] for c in a["covs"][4:]]
    covs += [np.r_[np.zeros(len(a["y"])), c] for c in b["covs"][4:]]
    cells = np.r_[np.char.add(f"{na}:", a["cells"].astype(str)),
                  np.char.add(f"{nb}:", b["cells"].astype(str))]  # fmt: skip
    labels = np.r_[np.where(a["y"] == 1, f"{na}_f", "m"), np.where(b["y"] == 1, f"{nb}_f", "m")]
    return {"cells": cells, "covs": covs, "labels": labels}


def r_a(model_id: str, ru: dict, es: dict, de: dict, rng, n_boot=N_BOOT, cosines=True):
    from .estimators import boot_within, split_half_geometry
    from .models import progress
    from .phase4 import _fit, within_cell_auc

    L = ru["X"].shape[1]
    rows = []
    for layer in progress(range(1, L - 1), desc=f"russian {model_id.split('/')[1]}", unit="layer"):
        Xr = ru["X"][:, layer].astype(np.float64)
        Xe = es["X"][:, layer].astype(np.float64)
        Xd = de["X"][:, layer].astype(np.float64)

        def metrics(ir, ie, idd, tr, te, td, Xr=Xr, Xe=Xe, Xd=Xd):
            d_ru, d_es, d_de = _fit(Xr, ru, ir), _fit(Xe, es, ie), _fit(Xd, de, idd)
            return {
                "ru_on_es": within_cell_auc(es["y"][te], Xe[te] @ d_ru, es["cells"][te]),
                "ru_on_de": within_cell_auc(de["y"][td], Xd[td] @ d_ru, de["cells"][td]),
                "es_on_ru": within_cell_auc(ru["y"][tr], Xr[tr] @ d_es, ru["cells"][tr]),
                "de_on_ru": within_cell_auc(ru["y"][tr], Xr[tr] @ d_de, ru["cells"][tr]),
            }

        a_r, a_e, a_d = (np.arange(len(x["y"])) for x in (ru, es, de))
        row = {"layer": layer, **metrics(a_r, a_e, a_d, a_r, a_e, a_d)}
        bw = lambda d: boot_within(d["cells"], d["y"], rng)
        bt = pd.DataFrame([metrics(bw(ru), bw(es), bw(de), bw(ru), bw(es), bw(de))
                           for _ in range(n_boot)])  # fmt: skip
        a = (1 - LEVEL) / 2
        for k in R_A:
            row[f"{k}_lo"], row[f"{k}_hi"] = np.nanquantile(bt[k], [a, 1 - a])
        if cosines:  # descriptive cross-fitted cosine, RU vs each language
            for nm, other, Xo in (("es", es, Xe), ("de", de, Xd)):
                p = _pool(ru, other, "ru", nm)
                g = split_half_geometry(np.r_[Xr, Xo], p["labels"], p["cells"], p["covs"],
                                        ["ru_f", f"{nm}_f"], "m", rng, n_splits=200)  # fmt: skip
                row[f"cos_ru_{nm}"] = g["cos"]
        rows.append(row)
    return pd.DataFrame(rows)


def r_a_reading(res: pd.DataFrame) -> dict:
    maj = len(res) // 2 + 1
    n = {k: int((res[f"{k}_lo"] > 0.5).sum()) for k in R_A}
    ok = {k: v >= maj for k, v in n.items()}
    if all(ok.values()):
        r = "aligns"
    elif (ok["es_on_ru"] and ok["de_on_ru"]) or (ok["ru_on_es"] and ok["ru_on_de"]):
        r = "one-way"
    elif any(ok.values()):
        r = "partial"
    else:
        r = "no alignment"
    return {"layers": len(res), "majority": maj, "layers_pass": n, "reading": r}


def r_b(model_id: str, ru: dict, rng, n_splits=200, n_perm=200) -> pd.DataFrame:
    """Confidence control (c) on the Russian fit, four Phase 5 layers, validity gate."""
    from .confidence_ctl import OVERLAP_MIN, SPLITS_PER_PERM, ZIPF_COV, _geometry
    from .estimators import nuisance
    from .phase5 import LAYER_FRACS

    y = ru["y"].astype(float)
    Z = nuisance(ru["cells"], ru["covs"])
    gi, fi = 0, 1 + len(set(ru["cells"])) + ZIPF_COV
    n_layers = ru["X"].shape[1] - 1
    strata = [np.flatnonzero((ru["cells"] == c) & (ru["y"] == g))
              for c in set(ru["cells"]) for g in (0, 1)]  # fmt: skip
    rows = []
    for k in sorted({max(1, round(f * n_layers)) for f in LAYER_FRACS}):
        X = ru["X"][:, k].astype(np.float64)
        M = np.column_stack([y, Z])
        r = _geometry(M, X, gi, fi, ru["cells"], ru["y"], rng, n_splits)
        null, zc = [], Z[:, fi - 1].copy()
        for _ in range(n_perm):
            zp = zc.copy()
            for s in strata:
                zp[s] = zc[rng.permutation(s)]
            Mp = M.copy()
            Mp[:, fi] = zp
            null.append(abs(_geometry(Mp, X, gi, fi, ru["cells"], ru["y"], rng,
                                      SPLITS_PER_PERM)["cross_gender_zipf"]))  # fmt: skip
        p95 = float(np.quantile(null, 0.95))
        c, x = r["cos_gender_zipf"], r["cross_gender_zipf"]
        valid = bool(np.isfinite(c) and abs(c) <= 1)
        rows.append({"model": model_id, "layer": k, **r, "null_abs_cross_p95": p95,
                     "valid": valid,
                     "overlap": bool(valid and abs(x) > p95 and abs(c) >= OVERLAP_MIN)})  # fmt: skip
    return pd.DataFrame(rows)


def r_b_reading(g: pd.DataFrame) -> str:
    ok, ov = g.valid.astype(bool), g.overlap.astype(bool)
    if (~ok).sum() >= 2:
        return "no valid test"
    if (ok & ov).sum() >= 3:
        return "overlap"
    if (ok & ~ov).sum() >= 3:
        return "no overlap"
    return "mixed"


def main(models=MODELS) -> dict:
    from .phase4 import de_train, es_train

    OUT.mkdir(parents=True, exist_ok=True)
    res, a_rows, b_rows = {}, [], []
    for m in models:
        rng = np.random.default_rng(SEED)
        ru, es, de = ru_train(m), es_train(m, "last"), de_train(m, "last")
        pr = matched_pairs(ru)
        mi = np.array(sorted(i for p in pr for i in p))
        out = {"n_ru": len(ru["y"]), "n_pairs": len(pr),
               "mean_dconc": float(np.mean([ru["conc_raw"][i] - ru["conc_raw"][j]
                                            for i, j in pr]))}  # fmt: skip
        for tag, d in (("all", ru), ("matched", subset(ru, mi) if len(pr) >= MIN_PAIRS else None)):
            if d is None:
                out[tag] = {"too_few": True}
                continue
            a = r_a(m, d, es, de, rng, cosines=(tag == "all")).assign(model=m, set=tag)
            b = r_b(m, d, rng).assign(set=tag)
            a_rows.append(a)
            b_rows.append(b)
            out[tag] = {"R_A": r_a_reading(a), "R_B": r_b_reading(b)}
            print(m, tag, json.dumps(out[tag]), flush=True)
        res[m] = out
    pd.concat(a_rows).to_csv(OUT / "r_a_layers.csv", index=False)
    pd.concat(b_rows).to_csv(OUT / "r_b_layers.csv", index=False)
    rb = {s: [res[m][s]["R_B"] for m in models if "R_B" in res[m].get(s, {})]
          for s in ("all", "matched")}  # fmt: skip
    res["R_B_across_models"] = {s: ("shared" if v and all(x == "overlap" for x in v) else
                                    "German-only" if v and all(x == "no overlap" for x in v)
                                    else "mixed") for s, v in rb.items()}  # fmt: skip
    (OUT / "readings.json").write_text(json.dumps(res, indent=2, default=float))
    return res


# ---- follow-up S: size-matched random subsets ---------------------------------------------
N_SUBSETS = 200
N_SUBSETS_BOOT = 50
SEED_S = 26


def random_subsets(ru: dict, pairs, n: int, seed: int = SEED_S) -> list[np.ndarray]:
    """Random subsets with the matched set's composition: per cell, the same numbers of f and m."""
    rng = np.random.default_rng(seed)
    comp = {}
    for i, j in pairs:
        for k in (i, j):
            key = (ru["cells"][k], ru["y"][k])
            comp[key] = comp.get(key, 0) + 1
    out = []
    for _ in range(n):
        idx = [rng.choice(np.flatnonzero((ru["cells"] == c) & (ru["y"] == g)), k, replace=False)
               for (c, g), k in sorted(comp.items())]  # fmt: skip
        out.append(np.sort(np.concatenate(idx)))
    return out


def _ru_out(Xr, Xe, Xd, d, es, de, ir=None, te=None, td=None):
    from .phase4 import _fit, within_cell_auc

    te = np.arange(len(es["y"])) if te is None else te
    td = np.arange(len(de["y"])) if td is None else td
    v = _fit(Xr, d, ir)
    return (within_cell_auc(es["y"][te], Xe[te] @ v, es["cells"][te]),
            within_cell_auc(de["y"][td], Xd[td] @ v, de["cells"][td]))  # fmt: skip


def size_control(models=MODELS, boot_model="Qwen/Qwen3-8B-Base") -> pd.DataFrame:
    from .estimators import boot_within
    from .models import progress
    from .phase4 import de_train, es_train

    OUT.mkdir(parents=True, exist_ok=True)
    main_a = pd.read_csv(OUT / "r_a_layers.csv")
    rows, brows, summ = [], [], {}
    for m in models:
        ru, es, de = ru_train(m), es_train(m, "last"), de_train(m, "last")
        pr = matched_pairs(ru)
        subs = random_subsets(ru, pr, N_SUBSETS)
        L = ru["X"].shape[1]
        layers = range(1, L - 1)
        Xs = {k: (ru["X"][:, k].astype(np.float64), es["X"][:, k].astype(np.float64),
                  de["X"][:, k].astype(np.float64)) for k in layers}  # fmt: skip
        for s_i, idx in enumerate(progress(subs, desc=f"size {m.split('/')[1]}", unit="subset")):
            d = subset(ru, idx)
            aucs = np.array([_ru_out(Xs[k][0][idx], Xs[k][1], Xs[k][2], d, es, de) for k in layers])
            rows.append({"model": m, "subset": s_i, "ru_on_es": aucs[:, 0].mean(),
                         "ru_on_de": aucs[:, 1].mean()})  # fmt: skip
        mt = main_a[(main_a.model == m) & (main_a.set == "matched")]
        r = pd.DataFrame([x for x in rows if x["model"] == m])
        summ[m] = {}
        for t in ("ru_on_es", "ru_on_de"):
            obs = float(mt[t].mean())
            lo, hi = np.quantile(r[t], [0.025, 0.975])
            summ[m][t] = {"matched": obs, "random_lo": float(lo), "random_hi": float(hi),
                          "random_median": float(r[t].median()),
                          "reading": "size explains it" if lo <= obs <= hi else
                          ("concreteness matters" if obs < lo else "matched better than random")}  # fmt: skip
        if m == boot_model:  # S2 (descriptive)
            rng = np.random.default_rng(SEED_S + 1)
            a = (1 - LEVEL) / 2
            for s_i, idx in enumerate(progress(subs[:N_SUBSETS_BOOT], desc="S2 boot",
                                               unit="subset")):  # fmt: skip
                d = subset(ru, idx)
                n_pass = {"ru_on_es": 0, "ru_on_de": 0}
                for k in layers:
                    Xr = Xs[k][0][idx]
                    bt = np.array([_ru_out(Xr, Xs[k][1], Xs[k][2], d, es, de,
                                           boot_within(d["cells"], d["y"], rng),
                                           boot_within(es["cells"], es["y"], rng),
                                           boot_within(de["cells"], de["y"], rng))
                                   for _ in range(N_BOOT)])  # fmt: skip
                    for j, t in enumerate(("ru_on_es", "ru_on_de")):
                        n_pass[t] += int(np.nanquantile(bt[:, j], a) > 0.5)
                brows.append({"model": m, "subset": s_i, **n_pass})
                pd.DataFrame(brows).to_csv(OUT / "size_control_boot.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "size_control.csv", index=False)
    models_ = list(summ)
    per = [summ[m][t]["reading"] for m in models_ for t in ("ru_on_es", "ru_on_de")]
    conc_both = all(any(summ[m][t]["reading"] == "concreteness matters"
                        for t in ("ru_on_es", "ru_on_de")) for m in models_)  # fmt: skip
    summ["overall"] = ("size explains the drop" if all(x == "size explains it" for x in per)
                       else "concreteness matters" if conc_both else "mixed")  # fmt: skip
    (OUT / "size_control.json").write_text(json.dumps(summ, indent=2, default=float))
    print(json.dumps(summ, indent=2, default=float))
    return pd.DataFrame(rows)
