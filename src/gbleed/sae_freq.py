"""G2 (R3): SAE features behind the German f−m / frequency overlap (Qwen3-8B first).

Pre-registered in `docs/design/gemma-slim-prereg.md` (G2 + "G2 implementation notes").
Discovery on the strat3 training nouns; the confirmatory test on the held-out `compound_test`
m/f nouns: remove the 20 candidate features (x' = x − Σ f_i W_dec[:, i], SAE error kept) and ask
whether the refit f−m direction's |cos| with the Zipf coefficient halves, while gender decoding
drops < 5 points, beating 100 firing-matched random feature sets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import sae as S
from .estimators import _halves, _strata
from .frequency_axis import ZIPF, _design, _judge, regression_variant

MODEL = "Qwen/Qwen3-8B-Base"
LAYERS = (9, 14, 20, 25)  # hidden_states indices (Phase 5 layers); SAE layer = index − 1
SEED = 21  # the (c) seed
N_SPLITS = 200
N_CAND = 20
N_RAND = 100
OUT = Path("results/sae_freq")


def heldout(model_id: str = MODEL) -> dict:
    """compound_test m/f nouns, shaped like `phase4.de_train`."""
    from . import activations as acts
    from .phase4 import P3_FINAL, _conc

    df = pd.read_csv(P3_FINAL, keep_default_na=False)
    X, meta = acts.load(model_id, "phase3_bare")
    row = {w: i for i, w in enumerate(meta["words"])}
    ntok_of = {w: len(t) for w, t in zip(meta["words"], meta["tokens"], strict=True)}
    s = df[(df.set == "compound_test") & df.gender.isin(["m", "f"])]
    conc = _conc("de", s.lemma, s.concept_en)
    miss = np.isnan(conc)
    conc[miss] = np.nanmean(conc)
    covs = [conc, miss.astype(float), pd.to_numeric(s.zipf).to_numpy(),
            np.array([ntok_of[w] for w in s.lemma], dtype=float)]  # fmt: skip
    return {"X": X[np.array([row[w] for w in s.lemma])], "y": (s.gender == "f").to_numpy(int),
            "cells": (s.ending + "|" + s.loan).to_numpy(), "covs": covs,
            "lemma": s.lemma.to_numpy()}  # fmt: skip


def score_features(d: dict, F: np.ndarray) -> pd.DataFrame:
    """Per feature: f−m effect and Zipf slope from the direction-fit regression; firing count."""
    M, zi = _design(d, np.asarray(d["covs"][ZIPF], float), "lin")
    B = np.linalg.pinv(M) @ F  # [p, d_sae]
    return pd.DataFrame({"feature": np.arange(F.shape[1]), "fm": B[0], "zipf_slope": B[zi],
                         "n_fire": (F > 0).sum(0)})  # fmt: skip


def candidates(sc: pd.DataFrame, overlap_sign: float, n: int = N_CAND) -> np.ndarray:
    ok = sc[(sc.n_fire > 0) & (np.sign(sc.fm) * np.sign(sc.zipf_slope) == overlap_sign)]
    return ok.reindex(ok.fm.abs().sort_values(ascending=False).index).feature.to_numpy()[:n]


def random_sets(sc: pd.DataFrame, cand: np.ndarray, rng, n_sets: int = N_RAND) -> list:
    """Each candidate swapped for a feature with the nearest training firing count."""
    pool = sc[(sc.n_fire > 0) & ~sc.feature.isin(cand)]
    fire, feat = pool.n_fire.to_numpy(), pool.feature.to_numpy()
    out = []
    for _ in range(n_sets):
        chosen: list[int] = []
        for c in cand:
            gap = np.abs(fire - sc.n_fire.iat[c]).astype(float)
            gap[np.isin(feat, chosen)] = np.inf
            ties = np.flatnonzero(gap == gap.min())
            chosen.append(int(feat[rng.choice(ties)]))
        out.append(np.array(chosen))
    return out


class CrossCos:
    """(c)'s cross-fitted cos(g, f) on fixed halves, fast for X − F_set W_set^T (linear in X)."""

    def __init__(self, d: dict, rng, n_splits: int = N_SPLITS):
        Mg, zi = _design(d, np.asarray(d["covs"][ZIPF], float), "lin")
        strata = _strata(d["cells"], d["y"])
        self.P = []  # per split: (gA, fA, gB, fB) projection rows [n]
        for _ in range(n_splits):
            A = _halves(d["cells"], d["y"], rng, strata)
            rows = []
            for h in (A, ~A):
                p = np.zeros((2, len(A)))
                p[:, h] = np.linalg.pinv(Mg[h])[[0, zi]]
                rows.append(p)
            self.P.append(np.concatenate(rows))  # [4, n]: gA, fA, gB, fB
        self.P = np.stack(self.P)  # [S, 4, n]

    def coefs(self, X: np.ndarray) -> np.ndarray:
        return np.einsum("skn,nd->skd", self.P, X)

    @staticmethod
    def cos(C: np.ndarray) -> float:
        gA, fA, gB, fB = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
        cross = (np.einsum("sd,sd->s", gA, fB) + np.einsum("sd,sd->s", gB, fA)).mean() / 2
        gg, ff = np.einsum("sd,sd->s", gA, gB).mean(), np.einsum("sd,sd->s", fA, fB).mean()
        return float(cross / np.sqrt(gg * ff)) if gg > 0 and ff > 0 else np.nan

    def removed(self, C0: np.ndarray, F: np.ndarray, W: np.ndarray, idx: np.ndarray) -> float:
        """cos after X' = X − F[:, idx] W[:, idx]^T, from C0 = coefs(X)."""
        return self.cos(C0 - np.einsum("skn,nj,dj->skd", self.P, F[:, idx], W[:, idx]))


def decode_acc(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv).mean())


def check_alignment(model_id: str, X: np.ndarray) -> pd.DataFrame:
    rows = []
    for L in LAYERS:
        sae = S.load(model_id, L - 1)
        al = S.alignment(sae, X, L - 1)
        rows.append({"hidden": L, **{f"fvu_off{o:+d}": v for o, v in al.items()},
                     "ok": min(al, key=al.get) == 1})  # fmt: skip
    return pd.DataFrame(rows)


def run_layer(model_id: str, L: int, tr: dict, ho: dict, rng) -> tuple[dict, pd.DataFrame]:
    sae = S.load(model_id, L - 1)
    Xtr = tr["X"][:, L].astype(np.float32)
    Xho = ho["X"][:, L].astype(np.float64)
    sc = score_features(tr, S.encode(sae, Xtr).numpy())
    Fho = S.encode(sae, Xho.astype(np.float32)).numpy().astype(np.float64)
    W = sae["W_dec"].numpy().astype(np.float64)  # [d, d_sae]

    # baseline overlap on held-out (the (c) rule, with its permutation null)
    base = regression_variant(ho, Xho, rng=rng)
    j = _judge(base, base["null"], None)
    valid = bool(j["overlap"])
    sign = float(np.sign(base["cos"]))
    cand = candidates(sc, sign)

    cc = CrossCos(ho, rng)
    C0 = cc.coefs(Xho)
    cos0 = cc.cos(C0)
    cos1 = cc.removed(C0, Fho, W, cand)
    red = abs(cos0) - abs(cos1)
    rand = [abs(cos0) - abs(cc.removed(C0, Fho, W, s)) for s in random_sets(sc, cand, rng)]
    p95 = float(np.percentile(rand, 95))
    Xrm = Xho - Fho[:, cand] @ W[:, cand].T
    acc0, acc1 = decode_acc(Xho, ho["y"]), decode_acc(Xrm, ho["y"])
    halves = abs(cos1) <= abs(cos0) / 2
    small_drop = (acc0 - acc1) * 100 < 5
    beats = red > p95
    res = {"model": model_id, "hidden": L, "sae_layer": L - 1, "n_train": len(tr["y"]),
           "n_heldout": len(ho["y"]), "base_cos_c": base["cos"], "base_overlap": valid,
           "null_p95": j["null_abs_cross_p95"], "cos_before": cos0, "cos_after": cos1,
           "reduction": red, "rand_p95": p95, "rand_median": float(np.median(rand)),
           "acc_before": acc0, "acc_after": acc1, "halves": bool(halves),
           "small_drop": bool(small_drop), "beats_random": bool(beats),
           "pass": bool(valid and halves and small_drop and beats)}  # fmt: skip
    top = (
        sc.set_index("feature").loc[cand].reset_index().assign(hidden=L, rank=np.arange(len(cand)))
    )
    return res, top


def describe(model_id: str, L: int, feats: np.ndarray, n_show: int = 15) -> pd.DataFrame:
    """Descriptive: top-activating German lexicon nouns per candidate feature (exploratory).

    Uses the Phase 3 bare-noun activations (the lexicon slice we have saved activations for)."""
    from . import activations as acts

    X, meta = acts.load(model_id, "phase3_bare")
    sae = S.load(model_id, L - 1)
    F = S.encode(sae, X[:, L].astype(np.float32)).numpy()
    df = pd.read_csv("data/stimuli/phase3_final_v3.csv", keep_default_na=False)
    info = df.drop_duplicates("lemma").set_index("lemma")
    rows = []
    for f in feats:
        order = np.argsort(-F[:, f])[:n_show]
        for r, i in enumerate(order):
            if F[i, f] <= 0:
                break
            w = meta["words"][i]
            g = info.loc[w] if w in info.index else None
            rows.append({"hidden": L, "feature": int(f), "rank": r, "noun": w,
                         "act": float(F[i, f]),
                         "gender": None if g is None else g.gender,
                         "gloss": None if g is None else g.concept_en,
                         "set": None if g is None else g.set,
                         "zipf": None if g is None else g.zipf})  # fmt: skip
    return pd.DataFrame(rows)


def main(model_id: str = MODEL, layers=LAYERS) -> pd.DataFrame:
    from .phase4 import de_train

    OUT.mkdir(parents=True, exist_ok=True)
    tr, ho = de_train(model_id, "last"), heldout(model_id)
    al = check_alignment(model_id, ho["X"])
    al.to_csv(OUT / "alignment.csv", index=False)
    print(al.to_string(index=False), flush=True)
    if not al.ok.all():
        raise SystemExit("SAE layer mapping check failed; stopping as pre-registered")
    rng = np.random.default_rng(SEED)
    res, tops, desc = [], [], []
    for L in layers:
        r, t = run_layer(model_id, L, tr, ho, rng)
        res.append(r)
        tops.append(t)
        desc.append(describe(model_id, L, t.feature.to_numpy()))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        pd.DataFrame(res).to_csv(OUT / "g2_layers.csv", index=False)
    pd.concat(tops).to_csv(OUT / "g2_candidates.csv", index=False)
    pd.concat(desc).to_csv(OUT / "g2_top_nouns.csv", index=False)
    out = pd.DataFrame(res)
    v = out[out.base_overlap]
    reading = ("no valid test" if v.empty
               else "features found" if v["pass"].sum() > len(v) / 2 else "not found")  # fmt: skip
    print(f"G2 reading ({model_id}): {reading} — pass {int(v['pass'].sum())}/{len(v)} valid layers",
          flush=True)  # fmt: skip
    (OUT / "g2_reading.txt").write_text(f"{model_id}\t{reading}\t{int(v['pass'].sum())}/{len(v)}\n")
    return out


if __name__ == "__main__":
    main()
