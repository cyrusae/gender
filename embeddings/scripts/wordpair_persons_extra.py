"""Person-noun baseline, remaining retrieval conditions (docs/wordpair-persons-prereg.md).
- invariant cues (67 frozen Spanish invariant adjectives): PRE-REGISTERED descriptive output, omitted
  from the first run's script by mistake;
- masculine cues (the 87 gendered pairs' masculine forms) and plain translation (query = the English
  noun alone, no cue): EXPLORATORY additions, 2026-11-11, after the main results.
Same nouns, space and scoring as wordpair_persons.py (original Spanish space, reciprocal rank, 0
outside the top 999). Gap per noun = mean over cues of (RR_masc − RR_fem). Bootstrap 10,000 (seed 0)
over nouns within group and cues (shared across groups).

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/wordpair_persons_extra.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from wordpair_mechanism import K, rr_table
from zhou_shift import Z, load_vec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/wordpair_persons"
N_BOOT, SEED = 10_000, 0
FLAGS = ["fem_also_noun", "masc_also_noun", "language_name", "masc_identical_en", "place_name"]


def plain(EN, T, en_w2i, w2i, i2w, occ):
    """[1, noun, 2] reciprocal ranks of (masc, fem) for the English noun alone."""
    Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
    out = np.zeros((1, len(occ), 2))
    for oi, (f, m, e) in enumerate(occ):
        q = EN[en_w2i[e]]
        s = Tn @ (q / np.linalg.norm(q))
        top = [i2w[i] for i in np.argsort(-s)[:K]][:999]
        pos = {w: r + 1 for r, w in enumerate(top)}
        out[0, oi] = [1 / pos[m] if m in pos else 0, 1 / pos[f] if f in pos else 0]
    return out


def summarise(G, grp, cond, sel, rng):
    groups = {g: np.flatnonzero(grp == g) for g in ("person", "occupation", "identity")}
    pt = {g: G[:, ix].mean() for g, ix in groups.items()}
    pt["person-occupation"] = pt["person"] - pt["occupation"]
    nc, B = G.shape[0], []
    for _ in range(N_BOOT):
        ci = rng.integers(0, nc, nc)
        b = {g: G[np.ix_(ci, rng.choice(ix, len(ix)))].mean() for g, ix in groups.items()}
        b["person-occupation"] = b["person"] - b["occupation"]
        B.append(b)
    B = pd.DataFrame(B)
    return [{"set": sel, "condition": cond, "measure": k, "gap_m_minus_f": v,
             "lo": np.percentile(B[k], 2.5), "hi": np.percentile(B[k], 97.5),
             "n_nouns": len(groups[k]) if k in groups else f"{len(groups['person'])}/{len(groups['occupation'])}",
             "n_cues": nc} for k, v in pt.items()]  # fmt: skip


def main():
    nouns = pd.read_csv(ROOT / "data/wp_persons_es_frozen_v1.csv", keep_default_na=False, dtype=str)
    cues = pd.read_csv(ROOT / "data/wp_cues_frozen_v1.csv", keep_default_na=False, dtype=str)
    es = cues[(cues.lang == "es") & (cues.subset == "primary")]
    G, I = es[es.type == "gendered"], es[es.type == "invariant"]
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    T, w2i = load_vec(Z / "data/wiki.multi.es.vec.txt")
    i2w = {i: w for w, i in w2i.items()}
    nouns = nouns[nouns.en.isin(en_w2i) & nouns.masc.isin(w2i) & nouns.fem.isin(w2i)].reset_index(drop=True)
    occ = list(zip(nouns.fem, nouns.masc, nouns.en))
    tabs = {
        "invariant cues (pre-registered, descriptive)": rr_table(en, T, en_w2i, w2i, i2w, [(e, x, "inv") for e, x in zip(I.en, I.masc)], occ),
        "masculine cues (exploratory)": rr_table(en, T, en_w2i, w2i, i2w, [(e, x, "m") for e, x in zip(G.en, G.masc)], occ),
        "plain translation (exploratory)": plain(en, T, en_w2i, w2i, i2w, occ),
    }
    grp = nouns.group.to_numpy()
    clean = ~nouns[FLAGS].eq("True").any(axis=1).to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    for cond, R in tabs.items():
        Gp = R[..., 0] - R[..., 1]
        rows += summarise(Gp, grp, cond, "all", rng)
        rows += summarise(Gp[:, clean], grp[clean], cond, "no flagged nouns", rng)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "extra_conditions.csv", index=False)
    with pd.option_context("display.width", 200):
        print(res.round(4).to_string(index=False))
    print("=== WPPX done")


if __name__ == "__main__":
    main()
