"""Phase 5 analysis (design: docs/design/phase5-design.md §5.3-5.4, §5.7, P9, P14, P18, P25).

Reads the sweep outputs of `phase5.sweep` (one directory per model x language x layer x
positions) and computes, per readout:

  score     per steered row: R1/R1-EN/R3/RN-R1 = the graded score, i.e. the coefficient of the
            Glasgow gender rating when the row's per-adjective log-probability shift (steered
            minus the same prompt unsteered) is regressed on gender, valence, arousal, size and
            frequency; sign flipped so that > 0 = toward feminine-rated adjectives (GEND runs
            1 = feminine to 7 = masculine). R2/R2-EN/RN-R2 = shift of log P(female option) -
            log P(male option) (man/woman, male/female, boy/girl; narrative: she/he).
  noun      rows averaged per noun over the primary wordings (R1/R1-EN: W1-W3; R2: the six
            framings); W4 (consensus) and W1nv are kept for their own contrasts.
  test      at the working dose (+1 x alpha*): the real direction's mean noun score against the
            100 random directions' (rank p-value, one-sided toward feminine, P9), and a noun
            bootstrap CI of (real - mean random). Holm across R1 and R1-EN.
  window    doses whose median damage (KL on neutral sentences) is <= the random directions'
            median damage at alpha* (§5.3); the dose-response slope within it, real vs the 20
            random directions on the dose subset (descriptive).
  extras    the number and social directions' scores (specificity, positive control), and the
            correlation of the real direction's per-adjective shift profile with the social one.

Verdicts across layers (majority of passing layers) and languages (sign agreement) are made by
`verdicts` over the per-run tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
N_BOOT = 2000
PRIMARY_WORDINGS = {"R1": {"W1", "W2", "W3"}, "R1-EN": {"W1", "W2", "W3"},
                    "RN-R1": {"W1", "W2", "W3"}, "R3": {"W1"}}  # fmt: skip
COVS = ["VAL", "AROU", "SIZE", "zipf"]
FEMALE_FIRST = {"woman_man", "female_male", "girl_boy"}


def load(run_dir: Path, name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    k = pd.read_csv(run_dir / f"{name}_conds.csv.gz", keep_default_na=False)
    z = np.load(run_dir / f"{name}.npz")
    return k, z["lp"], z["read"]


def ratings_for(name: str, fam: str, read: np.ndarray) -> pd.DataFrame:
    """Rating rows aligned with the readout's read ids."""
    if name == "R3":
        from wordfreq import zipf_frequency

        from .phase5_stimuli import ADJ_TRANS

        t = pd.read_csv(ADJ_TRANS, keep_default_na=False)
        t = t[(t.es_invariant.astype(str) == "True") & (t[f"es_ntok_{fam}"] == 1)]
        t = t.rename(columns={"en": "word"}).reset_index(drop=True)
        t["zipf"] = [zipf_frequency(w, "es") for w in t.es]  # the Spanish word's frequency
        assert len(t) == len(read)
        return t
    a = pd.read_csv(f"data/stimuli/phase5_adjectives_{fam}_v1.csv", keep_default_na=False)
    assert (a.token_id.to_numpy() == read).all()
    return a


def graded(shift: np.ndarray, rat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the feminine-direction gender coefficient (-beta_GEND) of the shift regressed on
    [1, GEND, covariates] across adjectives, and the size coefficient (the §5.4 control)."""
    cols = ["GEND", *COVS]
    X = np.column_stack([np.ones(len(rat))] + [pd.to_numeric(rat[c]).to_numpy() for c in cols])
    X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0)
    B = np.linalg.lstsq(X, shift.T, rcond=None)[0]  # [p, n_rows]
    return -B[1], B[1 + cols.index("SIZE")]


def row_scores(name: str, keys: pd.DataFrame, lp: np.ndarray, read: np.ndarray,
               fam: str, tok=None) -> pd.DataFrame:  # fmt: skip
    """Each row's score as a shift from its prompt's unsteered row (v at dose 0)."""
    k = keys.copy()
    pk = ["readout", "wording", "lemma", "lang"]
    k["prompt"] = k.groupby(pk, sort=False).ngroup()
    base = k[(k.vec == "v") & (k.mult == 0.0)].reset_index().set_index("prompt")["index"]
    b = base.loc[k.prompt].to_numpy()
    shift = lp - lp[b]
    if name in ("R2", "R2-EN", "RN-R2"):
        col = {int(t): j for j, t in enumerate(read)}
        ids = _r2_ids(tok)
        s = np.zeros(len(k))
        for w, (fi, mi) in ids.items():
            m = (k.wording == w).to_numpy()
            s[m] = shift[m, col[fi]] - shift[m, col[mi]]
        k["score"] = s
        k["size_coef"] = np.nan
    else:
        rat = ratings_for(name, fam, read)
        k["score"], k["size_coef"] = graded(shift, rat)
    return k


def _r2_ids(tok) -> dict[str, tuple[int, int]]:
    """wording -> (female token id, male token id)."""
    from .phase5 import R2_CHOICES, single_ids

    out = {}
    for w, (_, _, ws) in R2_CHOICES.items():
        a, b = single_ids(tok, list(ws))
        out[w] = (a, b) if w in FEMALE_FIRST else (b, a)
    she, he = single_ids(tok, [" she", " he"])
    out["narrative"] = (she, he)
    return out


def noun_scores(name: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Mean score per (vec, mult, noun) over the primary wordings."""
    prim = PRIMARY_WORDINGS.get(name)
    r = rows if prim is None else rows[rows.wording.isin(prim)]
    if name in ("R2", "R2-EN", "RN-R2"):
        r = rows[rows.wording != "narrative"]
    return r.groupby(["vec", "mult", "lemma"], as_index=False).score.mean()


def working_test(ns: pd.DataFrame, rng, n_boot: int = N_BOOT) -> dict:
    """Real direction at +1 vs every random direction at +1 (P9)."""
    w = ns[ns.mult == 1.0]
    piv = w.pivot_table(index="lemma", columns="vec", values="score")
    rand = [c for c in piv.columns if c.startswith("rand")]
    real = piv["v"].to_numpy()
    R = piv[rand].to_numpy()
    e_v, e_r = real.mean(), R.mean(0)
    p = (1 + int((e_r >= e_v).sum())) / (1 + len(rand))
    d = real - R.mean(1)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    out = {"n_nouns": len(real), "n_random": len(rand), "effect": float(e_v),
           "random_mean": float(e_r.mean()), "random_p95": float(np.quantile(e_r, 0.95)),
           "diff": float(d.mean()), "p": p}  # fmt: skip
    for lvl, q in (("95", (0.025, 0.975)), ("975", (0.0125, 0.9875))):
        out[f"diff_lo_{lvl}"], out[f"diff_hi_{lvl}"] = map(float, np.quantile(bs, q))
    for ex in ("number", "social"):
        if ex in piv:
            out[f"{ex}_effect"] = float(piv[ex].mean())
    return out


def window(damage: pd.DataFrame) -> list[float]:
    """Dose multiples where the real direction's median KL <= the random directions' median KL
    at alpha* (+1), always including 0."""
    lim = damage[damage.vec.str.startswith("rand") & (damage.mult == 1.0)].kl.median()
    v = damage[damage.vec == "v"].groupby("mult").kl.median()
    return sorted({0.0, *v[v <= lim].index.tolist()})


def slopes(ns: pd.DataFrame, mults: list[float]) -> dict:
    """Per-noun least-squares slope of the score on the dose multiple within the window, averaged;
    real vs random directions that have the full dose set (the dose subset)."""
    w = ns[ns.mult.isin(mults)]
    base = ns[(ns.vec == "v") & (ns.mult == 0.0)].set_index("lemma").score
    out = {}
    for vec, g in w.groupby("vec"):
        g = pd.concat([g, base.loc[g.lemma.unique()].reset_index().assign(vec=vec, mult=0.0)])
        if g.mult.nunique() < len(mults):
            continue
        s = g.groupby("lemma").apply(lambda x: np.polyfit(x.mult, x.score, 1)[0])
        out[vec] = (float(s.mean()), len(s))
    rand = [v for k, (v, _) in out.items() if k.startswith("rand")]
    res = {"window": mults, "slope_v": out.get("v", (np.nan, 0))[0], "n_random_slopes": len(rand)}
    if rand:
        res["slope_random_p95"] = float(np.quantile(rand, 0.95))
    for ex in ("number", "social"):
        if ex in out:
            res[f"slope_{ex}"] = out[ex][0]
    return res


def analyze_run(run_dir: str | Path, fam: str, tok, rng=None) -> dict:
    """Every readout of one sweep directory."""
    run_dir = Path(run_dir)
    rng = rng or np.random.default_rng(SEED)
    info = json.loads((run_dir / "vectors.json").read_text())
    dmg_path = run_dir / "damage.csv.gz"
    win = window(pd.read_csv(dmg_path, keep_default_na=False)) if dmg_path.exists() else None
    res = {"run": run_dir.name, **{k: info[k] for k in ("lang", "layer", "positions")},
           "window": win, "readouts": {}}  # fmt: skip
    for f in sorted(run_dir.glob("*_conds.csv.gz")):
        name = f.name.removesuffix("_conds.csv.gz")
        keys, lp, read = load(run_dir, name)
        rows = row_scores(name, keys, lp, read, fam, tok)
        ns = noun_scores(name, rows)
        r = {"working": working_test(ns, rng)}
        if win:
            r["slopes"] = slopes(ns, win)
        res["readouts"][name] = r
    return res
