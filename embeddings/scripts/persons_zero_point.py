"""Person-noun zero point (docs/persons-zero-point-prereg.md, ADOPTED 2026-11-11). Committed before any
score. Reuses zhou_zero_point.Setup (lean, zero point, A). Spanish, original space, corrected
attribute words (default). A per pair = A / n. Groups: person, occupation (rule-built), Zhou's list
(reference). Shared bootstrap draws (attribute words, epicene references) across groups; pairs within
group. Readings as the prereg.

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/persons_zero_point.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from c1_run import EN_F, EN_M
from zhou_shift import WEAT_F, WEAT_M, Z, load_vec
from zhou_zero_point import Setup, anchors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/persons_zero_point"
N_BOOT, SEED, FLOOR = 10_000, 0, 30
FLAGS = ["fem_also_noun", "masc_also_noun", "language_name", "masc_identical_en", "place_name"]


def q1(a, lo, hi):
    return "asymmetry: feminine further" if hi < 0 else ("masculine further" if lo > 0 else "inconclusive")


def run(E, w2i, groups, AM, AF, ref, tag):
    refs = {"epicene_strict": anchors(ref)["epicene_strict"]}
    S = {g: Setup(E, w2i, p, AM, AF, refs) for g, p in groups.items()}
    g_ref = S["person"].ref["epicene_strict"][2]
    full = {"epicene_strict": np.arange(len(g_ref))}
    pt = {g: s.stats(slice(None), slice(None), np.arange(s.n), full)["epicene_strict|A"] / s.n for g, s in S.items()}
    rng = np.random.default_rng(SEED)
    nm, nf = len(AM), len(AF)
    B = []
    for _ in range(N_BOOT):
        am, af = rng.integers(0, nm, nm), rng.integers(0, nf, nf)
        ridx = {"epicene_strict": np.concatenate([rng.choice(np.flatnonzero(g_ref == x), (g_ref == x).sum())
                                                  for x in np.unique(g_ref)])}  # fmt: skip
        B.append({g: s.stats(am, af, rng.integers(0, s.n, s.n), ridx)["epicene_strict|A"] / s.n for g, s in S.items()})
    B = pd.DataFrame(B)
    B["D"], pt["D"] = B.person - B.occupation, pt["person"] - pt["occupation"]
    rows = []
    for k in pt:
        lo, hi = np.percentile(B[k], [2.5, 97.5])
        if k == "D":
            m = 0.5 * abs(pt["occupation"])
            v = ("occupation-specific" if lo > 0 and pt["occupation"] < 0 else
                 "shared with person nouns" if (lo > -m and hi < m) else "not distinguishable")  # fmt: skip
        else:
            v = q1(pt[k], lo, hi)
        rows.append({**tag, "measure": k, "A_per_pair": pt[k], "lo": lo, "hi": hi, "verdict": v,
                     "n_pairs": S[k].n if k in S else f"{S['person'].n}/{S['occupation'].n}",
                     "floor_met": all(S[g].n >= FLOOR for g in ("person", "occupation"))})  # fmt: skip
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    E, w2i = load_vec(Z / "data/wiki.multi.es.vec.txt")
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    nouns = pd.read_csv(ROOT / "data/wp_persons_es_frozen_v1.csv", keep_default_na=False, dtype=str)
    nouns = nouns[nouns.masc.isin(w2i) & nouns.fem.isin(w2i)]
    zhou = [line.rstrip("\n").split("\t") for line in open(Z / "ES_occupation_words_with_EN_translations")]
    zhou = [(m, f) for f, m, _ in zhou if m in w2i and f in w2i]
    ref = pd.read_csv(ROOT / "data/zero_ref_es_frozen_v1.csv")
    taken = set(nouns.masc) | set(nouns.fem) | {w for p in zhou for w in p} | set(WEAT_M) | set(WEAT_F)
    ref = ref[ref["lemma"].isin(w2i) & ~ref["lemma"].isin(taken)].reset_index(drop=True)
    yard = {"native": ([E[w2i[w]] for w in WEAT_M], [E[w2i[w]] for w in WEAT_F]),
            "english": ([en[en_w2i[w]] for w in EN_M], [en[en_w2i[w]] for w in EN_F])}  # fmt: skip
    rows = []
    for sel, nn in (("all", nouns), ("no flagged nouns", nouns[~nouns[FLAGS].eq("True").any(axis=1)])):
        groups = {g: list(zip(nn[nn.group == g].masc, nn[nn.group == g].fem)) for g in ("person", "occupation")}
        if sel == "all":
            groups["zhou"] = zhou
        for y, (AM, AF) in yard.items():
            rows += run(E, w2i, groups, AM, AF, ref, {"set": sel, "yardstick": y})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)
    with pd.option_context("display.width", 200):
        print(res.round(4).to_string(index=False))
    print("=== PZP done")


if __name__ == "__main__":
    main()
