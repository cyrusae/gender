"""Why the German gender direction points toward "rarer" (`docs/design/frequency-axis-prereg.md`;
diagnostic). A: flexible frequency control; B: frequency axis projected out, Phase 4 rerun;
C: frequency-matched pairs (no regression); D: complex nouns dropped (D1) / as a covariate (D2).
Cosines are cross-fitted and judged by the confidence control's (c) overlap rule.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .confidence_ctl import OVERLAP_MIN
from .estimators import _halves, _strata, nuisance
from .models import model_slug

OUT = Path("results/frequency_axis")
MODELS = ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-14B-Base")
SEED = 22
N_SPLITS = 200
N_PERM = 200
SPLITS_PER_PERM = 20
ZIPF, NTOK = 2, 3  # covariate indices in es_train / de_train
CALIPER = 0.3
MIN_PAIRS = 10
NUMERALS = {"eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn",
            "elf", "zwölf", "zwanzig", "dreißig", "vierzig", "fünfzig", "hundert", "tausend"}  # fmt: skip
LINKS = ("", "s", "n", "en", "es", "e")


# ---- the complexity rule (D) -------------------------------------------------------------
def complexity_rule():
    from .phase3_stimuli import german_verbs

    lex = pd.read_csv("data/lexicon/de_nouns.csv", keep_default_na=False)
    nouns = set(lex.lemma.str.lower())
    verbs = set(german_verbs())

    def first_noun(p: str) -> bool:
        return any(p.endswith(k) and len(p) - len(k) >= 3 and p[: len(p) - len(k)] in nouns
                   for k in LINKS)  # fmt: skip

    def why(w: str) -> str:
        w = w.lower()
        r = []
        if w.endswith("er") and w[:-2] in NUMERALS:
            r.append("numeral+er")
        if any(w[i:] in nouns and first_noun(w[:i]) for i in range(3, len(w) - 3)):
            r.append("noun+noun")
        if w.endswith("er") and w[:-2] + "en" in verbs:
            r.append("verb+er")
        if w.endswith("er") and any(first_noun(w[:i]) and w[i:-2] + "en" in verbs
                                    and len(w) - i >= 5 for i in range(3, len(w) - 4)):  # fmt: skip
            r.append("noun+verb+er")
        return ",".join(r)

    return why


# ---- data and fits -----------------------------------------------------------------------
def _data(model_id: str, lang: str) -> dict:
    from .phase4 import de_train, es_train

    return es_train(model_id, "last") if lang == "es" else de_train(model_id, "last")


def _flex_covs(covs: list, zipf: np.ndarray) -> list:
    """A's covariates: the linear Zipf and token count replaced by a cubic in standardised Zipf,
    token-count categories (1 = reference, 2, 3, 4+) and Zipf x token count."""
    z = (zipf - zipf.mean()) / zipf.std()
    nt = covs[NTOK]
    rest = [c for i, c in enumerate(covs) if i not in (ZIPF, NTOK)]
    return [*rest, z, z**2, z**3, (nt == 2) * 1.0, (nt == 3) * 1.0, (nt >= 4) * 1.0, z * nt]


def _design(d: dict, zipf: np.ndarray, kind: str, extra=None) -> tuple[np.ndarray, int]:
    """[gender, cells, covariates] and the Zipf column's index (-1 if Zipf isn't linear)."""
    covs = list(d["covs"])
    covs[ZIPF] = zipf
    n_lev = len(set(d["cells"]))
    if kind == "flex":
        covs, zi = _flex_covs(covs, zipf), -1
    else:
        zi = 1 + n_lev + ZIPF
    if extra is not None:
        covs = [*covs, extra]
    return np.column_stack([d["y"].astype(float), nuisance(d["cells"], covs)]), zi


def _coef(M, X, rows, j) -> np.ndarray:
    return (np.linalg.pinv(M[rows]) @ X[rows])[j]


def _cross(halves, g_est, f_est) -> dict:
    acc = np.zeros(3)
    for A in halves:
        gA, gB, fA, fB = g_est(A), g_est(~A), f_est(A), f_est(~A)
        acc += [(gA @ fB + gB @ fA) / 2, gA @ gB, fA @ fB]
    acc /= len(halves)
    cos = acc[0] / np.sqrt(acc[1] * acc[2]) if acc[1] > 0 and acc[2] > 0 else np.nan
    return {"cos": float(cos), "cross": float(acc[0]), "len2_g": float(acc[1])}


def _perm_zipf(d, rng) -> np.ndarray:
    z = np.asarray(d["covs"][ZIPF], dtype=float)
    out = z.copy()
    for s in _strata(d["cells"], d["y"]):
        out[s] = z[rng.permutation(s)]
    return out


def _judge(r: dict, null: list[float], ref_overlap: bool | None) -> dict:
    p95 = float(np.quantile(null, 0.95))
    ov = bool(abs(r["cross"]) > p95 and abs(r["cos"]) >= OVERLAP_MIN)
    return {**r, "null_abs_cross_p95": p95, "overlap": ov, "overlap_in_c": ref_overlap}


def regression_variant(d, X, kind: str = "lin", rng=None, keep=None, extra=None, n_splits=N_SPLITS,
                       n_perm=N_PERM) -> dict:  # fmt: skip
    """Cross-fitted cos(g, f): g from the variant's fit (`kind` lin/flex, rows `keep`, an
    `extra` covariate), f = the linear Zipf coefficient of the original fit on the same rows'
    other half. The null permutes Zipf in both fits (as (c))."""
    keep = np.ones(len(d["y"]), bool) if keep is None else keep
    sub = {k: (v[keep] if isinstance(v, np.ndarray) else [c[keep] for c in v])
           for k, v in d.items() if k in ("y", "cells", "covs", "lemma")}  # fmt: skip
    Xk = X[keep]
    ex = None if extra is None else extra[keep]

    def run(zipf, splits):
        Mg, _ = _design(sub, zipf, kind, ex)
        Mf, zi = _design(sub, zipf, "lin")
        strata = _strata(sub["cells"], sub["y"])
        halves = [_halves(sub["cells"], sub["y"], rng, strata) for _ in range(splits)]
        return _cross(halves, lambda h: _coef(Mg, Xk, h, 0), lambda h: _coef(Mf, Xk, h, zi))

    z = np.asarray(sub["covs"][ZIPF], dtype=float)
    r = run(z, n_splits)
    null = [abs(run(_perm_zipf(sub, rng), SPLITS_PER_PERM)["cross"]) for _ in range(n_perm)]
    return r | {"null": null, "n": int(keep.sum())}


# ---- C: frequency-matched pairs ----------------------------------------------------------
def matched_pairs(d: dict, seed: int = SEED) -> list[tuple[int, int]]:
    """Within each ending cell: feminine to masculine 1:1, same token count, |dZipf| <= CALIPER,
    greedy from the rarest feminine noun, nearest Zipf (ties broken by a seeded shuffle)."""
    rng = np.random.default_rng(seed)
    z, nt = np.asarray(d["covs"][ZIPF], float), np.asarray(d["covs"][NTOK])
    pairs = []
    for c in sorted(set(d["cells"])):
        fem = [i for i in np.flatnonzero((d["cells"] == c) & (d["y"] == 1))]
        mas = set(np.flatnonzero((d["cells"] == c) & (d["y"] == 0)).tolist())
        fem = sorted(rng.permutation(fem).tolist(), key=lambda i: z[i])
        for i in fem:
            cand = [j for j in rng.permutation(sorted(mas)).tolist()
                    if nt[j] == nt[i] and abs(z[j] - z[i]) <= CALIPER]  # fmt: skip
            if cand:
                j = min(cand, key=lambda j: abs(z[j] - z[i]))
                pairs.append((i, j))
                mas.discard(j)
    return pairs


def pairs_variant(d, X, rng, n_splits=N_SPLITS, n_perm=N_PERM) -> dict:
    pr = matched_pairs(d)
    out = {"n_pairs": len(pr)}
    if len(pr) < MIN_PAIRS:
        return out | {"too_few": True}
    z = np.asarray(d["covs"][ZIPF], float)
    out["mean_dzipf"] = float(np.mean([z[i] - z[j] for i, j in pr]))
    pid = np.full(len(d["y"]), -1)
    for k, (i, j) in enumerate(pr):
        pid[[i, j]] = k
    P = np.array(pr)

    def halves(splits):
        hs = []
        for _ in range(splits):
            a_pair = rng.random(len(pr)) < 0.5
            h = rng.random(len(d["y"])) < 0.5  # unpaired nouns: random half
            h[pid >= 0] = a_pair[pid[pid >= 0]]
            hs.append(h)
        return hs

    def g_est(h):
        sel = h[P[:, 0]]
        return (X[P[sel, 0]] - X[P[sel, 1]]).mean(0)

    def run(zipf, splits):
        Mf, zi = _design(d, zipf, "lin")
        return _cross(halves(splits), g_est, lambda h: _coef(Mf, X, h, zi))

    r = run(z, n_splits)
    null = [abs(run(_perm_zipf(d, rng), SPLITS_PER_PERM)["cross"]) for _ in range(n_perm)]
    return out | r | {"null": null}


# ---- A, C, D per layer ---------------------------------------------------------------------
def run_acd(model_id: str, n_splits=N_SPLITS, n_perm=N_PERM) -> pd.DataFrame:
    from .phase5 import LAYER_FRACS

    c_ref = pd.read_csv("results/confidence_ctl/c_frequency_cosine.csv")
    why = complexity_rule()
    rng = np.random.default_rng(SEED)
    rows = []
    for lang in ("es", "de"):
        d = _data(model_id, lang)
        n_layers = d["X"].shape[1] - 1
        cx = np.array([why(w) != "" for w in d["lemma"]]) if lang == "de" else None
        for k in sorted({max(1, round(f * n_layers)) for f in LAYER_FRACS}):
            X = d["X"][:, k].astype(np.float64)
            ref = c_ref[(c_ref.model == model_id) & (c_ref.lang == lang) & (c_ref.layer == k)]
            ref_ov = bool(ref.overlap.iloc[0]) if len(ref) else None
            kw = {"n_splits": n_splits, "n_perm": n_perm}
            variants = [("A_flex", "reg", {"kind": "flex"}), ("C_pairs", "pairs", {})]
            if lang == "de":
                variants += [("D1_no_complex", "reg", {"kind": "lin", "keep": ~cx}),
                             ("D2_complex_cov", "reg", {"kind": "lin", "extra": cx.astype(float)})]  # fmt: skip
            for name, how, args in variants:
                r = (regression_variant(d, X, rng=rng, **args, **kw) if how == "reg"
                     else pairs_variant(d, X, rng, **kw))  # fmt: skip
                base = {"model": model_id, "lang": lang, "layer": k, "variant": name}
                if r.get("too_few"):
                    rows.append(base | {k2: v for k2, v in r.items() if k2 != "null"})
                else:
                    null = r.pop("null")
                    rows.append(base | _judge(r, null, ref_ov))
                print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def readings(acd: pd.DataFrame) -> pd.DataFrame:
    """Per model x language x variant: does the overlap disappear at a majority of the layers
    where (c) found it? (prereg readings for A and D1)"""
    out = []
    for (m, lang, v), g in acd.groupby(["model", "lang", "variant"]):
        g = g[g.overlap_in_c.astype(str) == "True"] if "overlap_in_c" in g else g.iloc[:0]
        n = len(g)
        gone = int((g.overlap.astype(str) != "True").sum()) if n else 0
        out.append({"model": m, "lang": lang, "variant": v, "layers_with_c_overlap": n,
                    "overlap_gone": gone, "majority_gone": bool(n and gone > n // 2)})  # fmt: skip
    return pd.DataFrame(out)


# ---- B: frequency axis projected out, Phase 4 rerun --------------------------------------
def freq_axes(model_id: str, langs=("de",)) -> np.ndarray:
    """Unit Zipf coefficients of the full linear fit, per layer: [layers, r, d] (orthonormal)."""
    vs = []
    for lang in langs:
        d = _data(model_id, lang)
        M, zi = _design(d, np.asarray(d["covs"][ZIPF], float), "lin")
        P = np.linalg.pinv(M)[zi]  # [n]
        X = d["X"].astype(np.float64)
        vs.append(np.einsum("n,nld->ld", P, X))
    V = np.stack(vs, axis=1)  # [L, r, d]
    Q = np.stack([np.linalg.qr(v.T)[0].T for v in V])  # orthonormal rows per layer
    return Q


@contextmanager
def projected(Q: np.ndarray):
    from . import activations as acts

    orig = acts.load

    def load(model_id, name):
        X, meta = orig(model_id, name)
        X = X.astype(np.float32)
        for k in range(X.shape[1]):
            q = Q[k].astype(np.float32)  # [r, d]
            X[:, k] -= (X[:, k] @ q.T) @ q
        return X, meta

    acts.load = load
    try:
        yield
    finally:
        acts.load = orig


def run_b(model_id: str) -> dict:
    from . import phase4

    out = {}
    for tag, langs in (("baseline", None), ("minus_de_freq", ("de",)),
                       ("minus_both_freq", ("de", "es"))):  # fmt: skip
        root = str(OUT / "B" / tag)
        if langs is None:
            out[tag] = phase4.analyze(model_id, "last", out_root=root)
        else:
            with projected(freq_axes(model_id, langs)):
                out[tag] = phase4.analyze(model_id, "last", out_root=root)
    keys = ["T1_transfers", "T3_pairs_follow_gender", "t1_es_on_de_layers_holm",
            "t1_de_on_es_layers_holm", "t3_es_layers_holm", "t3_de_layers_holm", "t2_cos_mean",
            "t2_layers_above_null", "inner_layers"]  # fmt: skip
    pub = json.loads(
        (Path("results/runpod2/phase4") / model_slug(model_id) / "summary.json").read_text()
    )
    tab = {"published": {k: pub.get(k) for k in keys}}
    tab |= {t: {k: s.get(k) for k in keys} for t, s in out.items()}
    b, p = out["baseline"], out["minus_de_freq"]
    tab["reading"] = {
        "verdicts_unchanged": bool(b["T1_transfers"] == p["T1_transfers"]
                                   and b["T3_pairs_follow_gender"] == p["T3_pairs_follow_gender"]),
        "t2_cos_change": float(p["t2_cos_mean"] - b["t2_cos_mean"]),
    }  # fmt: skip
    tab["reading"]["robust"] = bool(tab["reading"]["verdicts_unchanged"]
                                    and abs(tab["reading"]["t2_cos_change"]) < 0.05)  # fmt: skip
    return tab


def main(parts: str = "acdb") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if any(p in parts for p in "acd"):
        acd = pd.concat(run_acd(m) for m in MODELS)
        acd.to_csv(OUT / "acd_layers.csv", index=False)
        readings(acd).to_csv(OUT / "acd_readings.csv", index=False)
    if "b" in parts:
        res = {m: run_b(m) for m in MODELS}
        (OUT / "b_phase4.json").write_text(json.dumps(res, indent=1, default=float))


# ---- open threads 1-3 (exploratory, 2026-10-22; no pre-set rules) -------------------------
def _within_spearman(s, z, groups) -> float:
    """Spearman of s with z after removing group means of both ranks (groups = cell x gender)."""
    from scipy.stats import rankdata

    rs, rz = rankdata(s), rankdata(z)
    g = pd.Series(groups)
    rs = rs - pd.Series(rs).groupby(g).transform("mean").to_numpy()
    rz = rz - pd.Series(rz).groupby(g).transform("mean").to_numpy()
    return float(np.corrcoef(rs, rz)[0, 1])


def _auc_by_dzipf(y, s, cells, z, edges=(-0.3, 0.3)) -> dict:
    """Within-cell f-vs-m AUC over comparison pairs, binned by dZipf = zipf_f - zipf_m."""
    out = {k: [0.0, 0] for k in ("f_rarer", "balanced", "f_commoner")}
    for c in np.unique(cells):
        k = cells == c
        f, m = np.flatnonzero(k & (y == 1)), np.flatnonzero(k & (y == 0))
        if not len(f) or not len(m):
            continue
        dz = z[f][:, None] - z[m][None, :]
        win = (s[f][:, None] > s[m][None, :]) + 0.5 * (s[f][:, None] == s[m][None, :])
        for key, sel in (("f_rarer", dz < edges[0]), ("balanced", (dz >= edges[0]) & (dz <= edges[1])),
                         ("f_commoner", dz > edges[1])):  # fmt: skip
            out[key][0] += float(win[sel].sum())
            out[key][1] += int(sel.sum())
    return {f"auc_{k}": (v[0] / v[1] if v[1] else np.nan) for k, v in out.items()} | {
        f"n_{k}": v[1] for k, v in out.items()
    }


def threads12(model_id: str) -> pd.DataFrame:
    """Thread 1: within Spanish cell x gender, Spearman of the German direction's score with
    Spanish Zipf (T1's German probe direction, as Phase 4), before and after projecting out the
    German frequency axis. Thread 2: T1 AUC binned by the Zipf gap of each f-m comparison."""
    from .phase4 import _fit

    es, de = _data(model_id, "es"), _data(model_id, "de")
    Q = freq_axes(model_id, ("de",))
    z = np.asarray(es["covs"][ZIPF], float)
    grp = np.char.add(es["cells"].astype(str), es["y"].astype(str))
    rows = []
    for k in range(1, es["X"].shape[1] - 1):
        Xe, Xd = es["X"][:, k].astype(np.float64), de["X"][:, k].astype(np.float64)
        q = Q[k]
        for tag, (A, B) in (("baseline", (Xe, Xd)),
                            ("minus_de_freq", (Xe - (Xe @ q.T) @ q, Xd - (Xd @ q.T) @ q))):  # fmt: skip
            d_de = _fit(B, de)
            s = A @ d_de
            rows.append({"model": model_id, "layer": k, "version": tag,
                         "spearman_score_zipf_within": _within_spearman(s, z, grp),
                         **_auc_by_dzipf(es["y"], s, es["cells"], z)})  # fmt: skip
    return pd.DataFrame(rows)


def thread3(model_id: str, n_show: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """German training nouns along the German frequency axis once Zipf (and the rest of the
    design) is accounted for: residual of x . f_hat on the fit's design, per layer. Returns the
    per-noun table (mean residual rank over the steering layers) and the layer-to-layer
    consistency (Spearman between steering layers)."""
    from scipy.stats import rankdata

    from .phase5 import LAYER_FRACS

    d = _data(model_id, "de")
    Q = freq_axes(model_id, ("de",))
    M, _ = _design(d, np.asarray(d["covs"][ZIPF], float), "lin")
    n_layers = d["X"].shape[1] - 1
    ks = sorted({max(1, round(f * n_layers)) for f in LAYER_FRACS})
    res = {}
    for k in ks:
        p = d["X"][:, k].astype(np.float64) @ Q[k, 0]
        r = p - M @ np.linalg.lstsq(M, p, rcond=None)[0]
        res[k] = r / r.std()
    R = pd.DataFrame(res)
    cons = R.corr(method="spearman")
    lex = pd.read_csv("data/lexicon/de_nouns.csv", keep_default_na=False).drop_duplicates("lemma")
    gl = lex.set_index("lemma")
    why = complexity_rule()
    t = pd.DataFrame({"lemma": d["lemma"], "gender": np.where(d["y"] == 1, "f", "m"),
                      "zipf": np.asarray(d["covs"][ZIPF], float),
                      "ntok": np.asarray(d["covs"][NTOK]), "cell": d["cells"],
                      "gloss": [gl.concept_en.get(w, "") or gl.gloss.get(w, "") for w in d["lemma"]],
                      "complex": [why(w) for w in d["lemma"]],
                      "resid_mean_z": R.mean(1).to_numpy(),
                      "resid_rank_mean": np.mean([rankdata(R[k]) for k in ks], axis=0)})  # fmt: skip
    t = t.sort_values("resid_mean_z")
    return t, cons


def main_threads() -> None:
    out = OUT / "threads"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(threads12(m) for m in MODELS).to_csv(out / "threads12_layers.csv", index=False)
    for m in MODELS:
        t, cons = thread3(m)
        t.to_csv(out / f"thread3_nouns_{model_slug(m)}.csv", index=False)
        cons.to_csv(out / f"thread3_layer_consistency_{model_slug(m)}.csv")
