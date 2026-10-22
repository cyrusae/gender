"""Confidence/flattening control (`docs/design/confidence-ctl-prereg.md`; diagnostic).

(a) Every graded score gets one more covariate, the adjective's midpoint log-probability (mean of
    the two compared conditions, per row). Implemented by running each experiment's own analysis
    unchanged with three functions swapped for versions that carry the midpoint alongside the
    shift (`Shift`), so rows, exclusions, bootstraps and seeds are identical to the originals.
(b) Spearman correlation of GEND with the adjective's mean baseline log-probability, raw and
    partial (other covariates).
(c) Cross-fitted cosine of the gender coefficient with the Zipf coefficient of the same
    steering-vector fit, per steering layer, vs the number direction and a permutation null.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .models import model_slug
from .p29 import _contrast as _contrast0
from .p31 import _pair as _pair0
from .phase5_analysis import shifts as _shifts0

OUT = Path("results/confidence_ctl")
SEED = 21
N_SPLITS = 200
N_PERM = 200
SPLITS_PER_PERM = 20
OVERLAP_MIN = 0.3
KEEP_MIN = 0.5


# ---- (a) the midpoint covariate -------------------------------------------------------
class Shift(np.ndarray):
    """A [rows, adjectives] shift (B - A) carrying `mid` = (A + B) / 2 of the same shape; 2-D
    indexing slices both."""

    def __new__(cls, d, mid):
        o = np.asarray(d).view(cls)
        o.mid = np.asarray(mid)
        assert o.mid.shape == o.shape
        return o

    def __getitem__(self, idx):
        out = np.asarray(self)[idx]
        m = getattr(self, "mid", None)
        if m is not None and np.ndim(out) == 2:
            return Shift(out, m[idx])
        return out


def graded_mid(shift: np.ndarray, rat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`phase5_analysis.graded` plus a per-row midpoint covariate (when `shift` carries one).
    Frisch-Waugh: with m_r the row's midpoints, d_r = pinv(X) m_r and m~_r = m_r - X d_r,
    beta_r = pinv(X) y_r - d_r * (m~_r . y_r) / (m~_r . m~_r). Scaling m_r doesn't change beta."""
    from .phase5_analysis import COVS

    mid = getattr(shift, "mid", None)
    y = np.asarray(shift, dtype=float)
    cols = ["GEND", *COVS]
    X = np.column_stack([np.ones(len(rat))] + [pd.to_numeric(rat[c]).to_numpy() for c in cols])
    X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)
    P = np.linalg.pinv(X)  # [p, adjectives]
    B = P @ y.T  # [p, rows]
    if mid is not None:
        M = np.asarray(mid, dtype=float)
        D = P @ M.T
        Mt = M - (X @ D).T
        den = (Mt * Mt).sum(1)
        g = np.where(den > 0, (Mt * y).sum(1) / np.where(den > 0, den, 1), 0.0)
        B = B - D * g
    return -B[1], B[1 + cols.index("SIZE")]


def _pivot_avg(keys: pd.DataFrame, X: np.ndarray, a: str, b: str):
    """Per word: (X[a] - X[b], (X[a] + X[b]) / 2), each averaged over the wordings present."""
    k = keys.reset_index(drop=True)
    idx = k.pivot_table(index=["word", "wording"], columns="gender", values="row", aggfunc="first")
    xa, xb = X[idx[a].to_numpy()], X[idx[b].to_numpy()]
    per = idx.reset_index()[["word"]]
    words_ = per.word.unique()
    wi = {w: i for i, w in enumerate(words_)}
    n = per.groupby("word").size().reindex(words_).to_numpy()[:, None]
    out = []
    for v in (xa - xb, (xa + xb) / 2):
        acc = np.zeros((len(words_), X.shape[1]))
        np.add.at(acc, per.word.map(wi).to_numpy(), v)
        out.append(acc / n)
    return words_, Shift(*out)


def _contrast_mid(keys, X):
    meta, d = _contrast0(keys, X)
    words_, s = _pivot_avg(keys, X, "f", "m")
    assert (meta.word.to_numpy() == words_).all()
    assert np.allclose(np.asarray(s), d, equal_nan=True)
    return meta, s


def _pair_mid(keys, X, a, b):
    meta, d = _pair0(keys, X, a, b)
    words_, s = _pivot_avg(keys, X, a, b)
    assert (meta.word.to_numpy() == words_).all()
    assert np.allclose(np.asarray(s), d, equal_nan=True)
    return meta, s


def _shifts_mid(keys, lp):
    k, sh = _shifts0(keys, lp)
    return k, Shift(sh, lp - sh / 2)  # (row + base) / 2


@contextmanager
def midpoint():
    """Swap in the midpoint-carrying versions for the duration (restored afterwards)."""
    from . import p29, p30, p31, phase5_analysis

    swaps = [(phase5_analysis, "graded", graded_mid), (p29, "_contrast", _contrast_mid),
             (p30, "_contrast", _contrast_mid), (p31, "_pair", _pair_mid),
             (phase5_analysis, "shifts", _shifts_mid)]  # fmt: skip
    old = [(m, n, getattr(m, n)) for m, n, _ in swaps]
    for m, n, f in swaps:
        setattr(m, n, f)
    try:
        yield
    finally:
        for m, n, f in old:
            setattr(m, n, f)


def _rerun(mod, model_id: str, src_root: str, tag: str) -> tuple[dict, dict]:
    """The experiment's analysis twice: as published (from its analysis.json) and with the
    midpoint covariate, run in a directory of symlinks so the original isn't overwritten."""
    src = Path(src_root) / model_slug(model_id)
    orig = json.loads((src / "analysis.json").read_text())
    dst = OUT / tag / model_slug(model_id)
    dst.mkdir(parents=True, exist_ok=True)
    for f in paths.visible(src.iterdir()):
        if f.name != "analysis.json" and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve())
    with midpoint():
        new = mod.analyze(model_id, out_root=str(OUT / tag))
    return orig, new


def _keep(o: float, n: float) -> bool:
    return bool(np.sign(o) == np.sign(n) and abs(n) >= KEEP_MIN * abs(o))


def run_a_small() -> pd.DataFrame:
    """(a) on P29 (M1 all / M2 / M3 whole-word), P30 and P31 (S per context), every model run."""
    from . import p29, p30, p31

    rows = []
    for mod, root, tag in ((p29, "results/p29", "p29"), (p30, "results/p30", "p30"),
                           (p31, "results/p31", "p31")):  # fmt: skip
        for d in paths.visible(sorted(Path(root).iterdir())):
            if not (d / "analysis.json").exists():
                continue
            model_id = json.loads((d / "meta.json").read_text())["meta"]["model_id"]
            o, n = _rerun(mod, model_id, root, tag)
            if tag == "p29":
                for name, get in (("M1_all", lambda r: r["primary"]["M1_all"]),
                                  ("M2_invariant", lambda r: r["primary"]["M2_invariant"]),
                                  ("M3_whole_word", lambda r: r["primary"]["M3"]["M1_all_whole_word"])):  # fmt: skip
                    so, sn = get(o), get(n)
                    rows.append({"exp": tag, "model": model_id, "stat": name, "orig": so["A"],
                                 "mid": sn["A"], "mid_lo95": sn["lo95"], "mid_hi95": sn["hi95"],
                                 "verdict_orig": f"M1_passed={o['M1_passed']}",
                                 "verdict_mid": f"M1_passed={n['M1_passed']}"})  # fmt: skip
            else:
                for k in ("S_object", "S_neutral", "S_person", "S_object_das", "S_neutral_das",
                          "S_person_das"):  # fmt: skip
                    if k in o:
                        rows.append({"exp": tag, "model": model_id, "stat": k, "orig": o[k]["S"],
                                     "mid": n[k]["S"], "mid_lo95": n[k]["lo95"],
                                     "mid_hi95": n[k]["hi95"], "verdict_orig": o["verdict"],
                                     "verdict_mid": n["verdict"]})  # fmt: skip
    df = pd.DataFrame(rows)
    df["verdict_same"] = df.verdict_orig == df.verdict_mid
    df["keeps_half"] = [_keep(a, b) for a, b in zip(df.orig, df.mid, strict=True)]
    df["robust"] = df.verdict_same & df.keeps_half
    return df


P5_MODELS = ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-14B-Base")
P5_ROOT = "/Volumes/Lexar/gbleed/gender-s2/results/phase5"
P5_RC = "../gender-s2/results/phase5_readoutcheck"


def run_a_phase5(model_id: str, root: str = P5_ROOT, rc: str = P5_RC) -> pd.DataFrame:
    """(a) on the Phase 5 Spanish sweeps: per-layer working tests and verdicts, R1 / R1-EN,
    as published and with the midpoint covariate."""
    from transformers import AutoTokenizer

    from .phase5 import family
    from .phase5_analysis import analyze_run, failed_wordings, verdicts

    fam = family(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    failed = failed_wordings(model_id, rc)
    dirs = [r for r in paths.visible(sorted((Path(root) / model_slug(model_id)).iterdir()))
            if (r / "vectors.json").exists() and r.name.startswith("es_")]  # fmt: skip
    out = []
    for tag, ctx in (("orig", None), ("mid", midpoint)):
        runs = []
        for r in dirs:
            if ctx is None:
                runs.append(analyze_run(r, fam, tok, failed=failed))
            else:
                with ctx():
                    runs.append(analyze_run(r, fam, tok, failed=failed))
        v = verdicts(runs)
        v = v[v.readout.isin(["R1", "R1-EN"])].assign(version=tag, model=model_id)
        per = []
        for run in runs:
            for n in ("R1", "R1-EN"):
                w = run["readouts"].get(n, {}).get("working")
                if w:
                    per.append({"model": model_id, "version": tag, "positions": run["positions"],
                                "layer": run["layer"], "readout": n, **w})  # fmt: skip
        out.append((v, pd.DataFrame(per)))
    return pd.concat([o[0] for o in out]), pd.concat([o[1] for o in out])


# ---- (b) gender rating vs baseline probability -------------------------------------------
def _spearman(g, lpm, Z) -> tuple[float, float]:
    from scipy.stats import rankdata, spearmanr

    raw = float(spearmanr(g, lpm).statistic)
    R = np.column_stack([np.ones(len(g))] + [rankdata(z) for z in Z.T])
    res = [x - R @ np.linalg.lstsq(R, x, rcond=None)[0] for x in (rankdata(g), rankdata(lpm))]
    return raw, float(np.corrcoef(*res)[0, 1])


def run_b() -> pd.DataFrame:
    from .phase5 import family, r3_n_invariant
    from .phase5_analysis import COVS, PRIMARY_WORDINGS, load, ratings_for

    rows = []

    def add(src, model_id, name, lpm, rat):
        raw, part = _spearman(pd.to_numeric(rat.GEND).to_numpy(), lpm,
                              np.column_stack([pd.to_numeric(rat[c]).to_numpy() for c in COVS]))  # fmt: skip
        rows.append({"source": src, "model": model_id, "readout": name, "n_adj": len(rat),
                     "spearman": raw, "partial_spearman": part})  # fmt: skip

    for model_id in P5_MODELS:
        fam = family(model_id)
        d = Path(P5_ROOT) / model_slug(model_id)
        # unsteered rows are the same in every noun-only run: any one will do
        run = min(r for r in paths.visible(d.iterdir()) if r.name.startswith("es_")
                  and not r.name.endswith("_all"))  # fmt: skip
        for name in ("R1", "R1-EN", "R3"):
            keys, lp, read = load(run, name)
            rat = ratings_for(name, fam, read)
            m = (
                (keys.vec == "v") & (keys.mult == 0.0) & keys.wording.isin(PRIMARY_WORDINGS[name])
            ).to_numpy()
            add("phase5", model_id, name, lp[m][:, : len(rat)].mean(0), rat)
    for root, src in (("results/p30", "p30"), ("results/p31", "p31")):
        for d in paths.visible(sorted(Path(root).iterdir())):
            if not (d / "next.npz").exists():
                continue
            model_id = json.loads((d / "meta.json").read_text())["meta"]["model_id"]
            fam = family(model_id)
            keys = pd.read_csv(d / "prompts.csv.gz", keep_default_na=False)
            z = np.load(d / "next.npz")
            m = (keys.kind == "nonce").to_numpy()
            if src == "p30":
                rat = ratings_for("R3", fam, z["read"])
                assert len(rat) == r3_n_invariant(fam)
            else:
                rat = pd.read_csv(d / "readout.csv", keep_default_na=False)
            add(
                src,
                model_id,
                "invariant" if src == "p30" else "first-token",
                z["lp"][m][:, : len(rat)].mean(0),
                rat,
            )
    return pd.DataFrame(rows)


# ---- (c) gender direction vs frequency direction ------------------------------------------
ZIPF_COV = 2  # index of Zipf in es_train / de_train covariates


def _coefs(M: np.ndarray, X: np.ndarray, rows: np.ndarray, which: list[int]) -> np.ndarray:
    return (np.linalg.pinv(M[rows]) @ X[rows])[which]


def _number_pairs(model_id: str, lang: str):
    from . import activations as acts
    from .phase5_stimuli import NUMBER

    df = pd.read_csv(NUMBER, keep_default_na=False)
    df = df[df.lang == lang]
    Xp, mp = acts.load(model_id, "phase5_number")
    Xs, ms = acts.load(model_id, "phase2_bare" if lang == "es" else "phase3_bare")
    rp = {w: i for i, w in enumerate(mp["words"])}
    rs = {w: i for i, w in enumerate(ms["words"])}
    return df.lemma.to_numpy(), Xp[[rp[w] for w in df.plural]], Xs[[rs[w] for w in df.lemma]]


def _geometry(M, X, gi, fi, cells, y, rng, n_splits, num=None):
    """Cross-fitted cos(gender, zipf) and, with `num` = (lemmas, lemma_of_row, Xp, Xs),
    cos(number, zipf): halves within cell x gender strata; a number pair follows its
    singular's half (pairs whose singular isn't in the fit: a random half)."""
    from .estimators import _halves, _strata

    strata = _strata(cells, y)
    acc = np.zeros(5)  # g.f, g.g, f.f, n.f, n.n
    for _ in range(n_splits):
        A = _halves(cells, y, rng, strata)
        gA, fA = _coefs(M, X, A, [gi, fi])
        gB, fB = _coefs(M, X, ~A, [gi, fi])
        acc[:3] += [(gA @ fB + gB @ fA) / 2, gA @ gB, fA @ fB]
        if num is not None:
            lem, row_lem, Xp, Xs = num
            half = dict(zip(row_lem, A, strict=True))
            inA = np.array([half.get(w, rng.random() < 0.5) for w in lem])
            nA = (Xp[inA] - Xs[inA]).mean(0)
            nB = (Xp[~inA] - Xs[~inA]).mean(0)
            acc[3:] += [(nA @ fB + nB @ fA) / 2, nA @ nB]
    acc /= n_splits
    cos = lambda xy, xx, yy: float(xy / np.sqrt(xx * yy)) if xx > 0 and yy > 0 else np.nan
    out = {
        "cos_gender_zipf": cos(acc[0], acc[1], acc[2]),
        "cross_gender_zipf": acc[0],
        "len2_gender": acc[1],
        "len2_zipf": acc[2],
    }
    if num is not None:
        out["cos_number_zipf"] = cos(acc[3], acc[4], acc[2])
    return out


def run_c(model_id: str) -> pd.DataFrame:
    from .estimators import nuisance
    from .phase4 import de_train, es_train

    rng = np.random.default_rng(SEED)
    rows = []
    for lang in ("es", "de"):
        d = es_train(model_id, "last") if lang == "es" else de_train(model_id, "last")
        y = d["y"].astype(float)
        Z = nuisance(d["cells"], d["covs"])
        n_lev = len(set(d["cells"]))
        gi, fi = 0, 1 + n_lev + ZIPF_COV
        lem, Xp, Xs = _number_pairs(model_id, lang)
        n_layers = d["X"].shape[1] - 1
        from .phase5 import LAYER_FRACS

        for k in sorted({max(1, round(f * n_layers)) for f in LAYER_FRACS}):
            X = d["X"][:, k].astype(np.float64)
            M = np.column_stack([y, Z])
            num = (lem, d["lemma"], Xp[:, k].astype(np.float64), Xs[:, k].astype(np.float64))
            r = _geometry(M, X, gi, fi, d["cells"], d["y"], rng, N_SPLITS, num)
            null = []
            strata = [np.flatnonzero((d["cells"] == c) & (d["y"] == g))
                      for c in set(d["cells"]) for g in (0, 1)]  # fmt: skip
            zc = Z[:, fi - 1].copy()
            for _ in range(N_PERM):
                zp = zc.copy()
                for s in strata:
                    zp[s] = zc[rng.permutation(s)]
                Mp = M.copy()
                Mp[:, fi] = zp
                g = _geometry(Mp, X, gi, fi, d["cells"], d["y"], rng, SPLITS_PER_PERM)
                null.append(abs(g["cross_gender_zipf"]))
            # null on the cross term g.f, not the cosine: the permuted Zipf coefficient's squared
            # length is ~0, which makes the null cosine ill-conditioned (as Phase 4's shuffle null)
            p95 = float(np.quantile(null, 0.95))
            c, x = r["cos_gender_zipf"], r["cross_gender_zipf"]
            rows.append({"model": model_id, "lang": lang, "layer": k, **r,
                         "null_abs_cross_p95": p95,
                         "overlap": bool(abs(x) > p95 and abs(c) >= OVERLAP_MIN)})  # fmt: skip
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main(parts: str = "abc") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "a" in parts:
        run_a_small().to_csv(OUT / "a_unsteered.csv", index=False)
        vs, per = zip(*(run_a_phase5(m) for m in P5_MODELS), strict=True)
        pd.concat(vs).to_csv(OUT / "a_phase5_verdicts.csv", index=False)
        pd.concat(per).to_csv(OUT / "a_phase5_layers.csv", index=False)
    if "b" in parts:
        run_b().to_csv(OUT / "b_rating_vs_baseline.csv", index=False)
    if "c" in parts:
        pd.concat(run_c(m) for m in P5_MODELS).to_csv(OUT / "c_frequency_cosine.csv", index=False)
