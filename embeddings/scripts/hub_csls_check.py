"""Exploratory checks raised 2026-11-11 (outside reviewer via the PI): hubness, CSLS retrieval and the
masculine-vs-feminine epicene gap. Nothing here is pre-registered.

1. Epicene gap: mean lean of the masculine minus the feminine strict-tier epicenes (Spanish, corrected
   native attribute words and English attribute words). True epicenes have fixed grammatical gender
   but no sex reference, so a gap is grammatical gender leaking into the attribute yardstick.
   Bootstrap 10,000 (seed 0) over epicenes (within gender) and attribute words.
2. Hubness: k-occurrence N_10 of every Spanish word (first 50k) over the first 50k English words as
   queries (cosine). For the person-noun study's nouns: hub counts of the masculine and feminine
   forms, and the correlation of log((N_m+1)/(N_f+1)) with the per-noun intrusion (feminine cues).
3. CSLS retrieval: the person-noun study's conditions rerun with CSLS ranking, score(y) = 2 cos(q, y) −
   r(y), r(y) = mean cosine of Spanish word y to its 10 nearest English neighbours (the query-side
   term is constant per query and drops out of the ranking). Same scoring otherwise.

    uv run --no-project --python 3.12 --with numpy,pandas,scipy,scikit-learn python embeddings/scripts/hub_csls_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from c1_run import EN_F, EN_M
from scipy.stats import spearmanr
from wordpair_persons_extra import summarise
from zhou_shift import WEAT_F, WEAT_M, Z, load_vec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/hub_csls_check"
K, KH = 1000, 10
FLAGS = ["fem_also_noun", "masc_also_noun", "language_name", "masc_identical_en", "place_name"]


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def topk_counts_and_r(S, Q, k=KH, chunk=2000):
    """For rows of Q (queries), count how often each row of S is in the top k; and r(S) vs Q."""
    counts = np.zeros(len(S), int)
    r = np.zeros(len(S))
    for i in range(0, len(Q), chunk):
        sims = Q[i:i + chunk] @ S.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        np.add.at(counts, top.ravel(), 1)
    for i in range(0, len(S), chunk):
        sims = S[i:i + chunk] @ Q.T
        r[i:i + chunk] = -np.partition(-sims, k, axis=1)[:, :k].mean(1) * -1
        r[i:i + chunk] = np.partition(sims, -k, axis=1)[:, -k:].mean(1)
    return counts, r


def retrieve(Tn, rS, q, exclude, i2w, csls):
    s = Tn @ (q / np.linalg.norm(q))
    if csls:
        s = 2 * s - rS
    top = [i2w[i] for i in np.argsort(-s)[:K] if i2w[i] != exclude][:999]
    return {w: r + 1 for r, w in enumerate(top)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    T, w2i = load_vec(Z / "data/wiki.multi.es.vec.txt")
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    Tn, EN50 = unit(T), unit(en[:50000])
    i2w = {i: w for w, i in w2i.items()}
    # 1. epicene gap
    ref = pd.read_csv(ROOT / "data/zero_ref_es_frozen_v1.csv")
    epi = ref[(ref["set"] == "epicene") & (ref["tier"] == "strict") & ref["lemma"].isin(w2i)]
    gap_rows = []
    for yard, (AM, AF) in {"native (corrected)": ([T[w2i[w]] for w in WEAT_M], [T[w2i[w]] for w in WEAT_F]),
                           "english": ([en[en_w2i[w]] for w in EN_M], [en[en_w2i[w]] for w in EN_F])}.items():  # fmt: skip
        AM, AF = np.array(AM), np.array(AF)
        Vm = np.array([T[w2i[w]] for w in epi[epi.gender == "m"].lemma])
        Vf = np.array([T[w2i[w]] for w in epi[epi.gender == "f"].lemma])

        def g(vm, vf, am, af):
            return ((vm @ am.T).mean(1) - (vm @ af.T).mean(1)).mean() - ((vf @ am.T).mean(1) - (vf @ af.T).mean(1)).mean()

        pt = g(Vm, Vf, AM, AF)
        B = [g(Vm[rng.integers(0, len(Vm), len(Vm))], Vf[rng.integers(0, len(Vf), len(Vf))],
               AM[rng.integers(0, len(AM), len(AM))], AF[rng.integers(0, len(AF), len(AF))]) for _ in range(10_000)]  # fmt: skip
        lo, hi = np.percentile(B, [2.5, 97.5])
        gap_rows.append({"yardstick": yard, "gap_masc_minus_fem_epicenes": pt, "lo": lo, "hi": hi,
                         "n_m": len(Vm), "n_f": len(Vf)})  # fmt: skip
    pd.DataFrame(gap_rows).to_csv(OUT / "epicene_gap.csv", index=False)
    # 2. hubness
    counts, rS = topk_counts_and_r(Tn, EN50)
    nouns = pd.read_csv(ROOT / "data/wp_persons_es_frozen_v1.csv", keep_default_na=False, dtype=str)
    nouns = nouns[nouns.en.isin(en_w2i) & nouns.masc.isin(w2i) & nouns.fem.isin(w2i)].reset_index(drop=True)
    per = pd.read_csv(ROOT / "results/wordpair_persons/per_noun.csv", keep_default_na=False)
    nouns = nouns.merge(per[["masc", "intrusion"]], on="masc", validate="one_to_one")
    nouns["hub_m"], nouns["hub_f"] = nouns.masc.map(lambda w: counts[w2i[w]]), nouns.fem.map(lambda w: counts[w2i[w]])
    nouns["log_hub_ratio"] = np.log((nouns.hub_m + 1) / (nouns.hub_f + 1))
    hub_rows = []
    for grp, d in nouns.groupby("group"):
        rho, p = spearmanr(d.log_hub_ratio, d.intrusion)
        hub_rows.append({"group": grp, "n": len(d), "median_hub_m": d.hub_m.median(), "median_hub_f": d.hub_f.median(),
                         "share_masc_bigger_hub": (d.hub_m > d.hub_f).mean(),
                         "spearman_logratio_vs_intrusion": rho, "p": p})  # fmt: skip
    pd.DataFrame(hub_rows).to_csv(OUT / "hubness.csv", index=False)
    nouns[["masc", "fem", "group", "hub_m", "hub_f", "intrusion"]].to_csv(OUT / "hub_per_noun.csv", index=False)
    print("global N10: share of words that are never a top-10 neighbour", (counts == 0).mean().round(3),
          "| max", counts.max(), i2w[int(counts.argmax())])  # fmt: skip
    # 3. CSLS retrieval, person-noun conditions
    cues = pd.read_csv(ROOT / "data/wp_cues_frozen_v1.csv", keep_default_na=False, dtype=str)
    es = cues[(cues.lang == "es") & (cues.subset == "primary")]
    G, I = es[es.type == "gendered"], es[es.type == "invariant"]
    conds = {"feminine cues": list(zip(G.en, G.fem)), "masculine cues": list(zip(G.en, G.masc)),
             "invariant cues": list(zip(I.en, I.masc)), "plain translation": [(None, None)]}  # fmt: skip
    grp = nouns.group.to_numpy()
    clean = ~nouns[FLAGS].eq("True").any(axis=1).to_numpy()
    rows = []
    for csls in (False, True):
        for cond, cl in conds.items():
            R = np.zeros((len(cl), len(nouns), 2))
            for ci, (ce, cx) in enumerate(cl):
                for oi, (f, m, e) in enumerate(zip(nouns.fem, nouns.masc, nouns.en)):
                    q = en[en_w2i[e]] if ce is None else en[en_w2i[e]] - en[en_w2i[ce]] + T[w2i[cx]]
                    pos = retrieve(Tn, rS, q, cx, i2w, csls)
                    R[ci, oi] = [1 / pos[m] if m in pos else 0, 1 / pos[f] if f in pos else 0]
            Gp = R[..., 0] - R[..., 1]
            tag = f"{cond} ({'CSLS' if csls else 'cosine'})"
            rows += summarise(Gp, grp, tag, "all", rng)
            rows += summarise(Gp[:, clean], grp[clean], tag, "no flagged nouns", rng)
            print("done", tag, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "csls_retrieval.csv", index=False)
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(gap_rows).round(4).to_string(index=False))
        print(pd.DataFrame(hub_rows).round(3).to_string(index=False))
        print(res[res.set == "all"].round(4).to_string(index=False))
    print("=== HUB done")


if __name__ == "__main__":
    main()
