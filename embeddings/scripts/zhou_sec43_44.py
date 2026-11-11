"""Zhou et al. 2019 §4.3 word translation (Table 2 top) and §4.4 bias correlation (Fig. 4)
(exploratory, PI 2026-11-10). Committed before any score. No released code exists for either.

§4.3: MUSE test dictionaries (`{src}-{tgt}.5000-6500.txt`, downloaded 2026-11-10), nearest
neighbour by cosine (the paper's 79.2 matches MUSE's NN, not CSLS, figure). Source queries in the
first 200k source words; candidates the first 200k target words for → EN, the first 50k for → ES/FR
(our rebuilt spaces hold 50k). A query counts if any listed translation is in the top k. Spaces:
our rebuilds (Original, Shift Ori, Hybrid Ori) from zhou_wordpair.spaces.
§4.4: footnote 3 says WEAT for English and MWEAT for Spanish, outer absolute value removed. Per
occupation (original space): EN s(e) = mean cos with EN male definitional words − female ones;
ES per pair two readings, since "remove the outer absolute" is ambiguous at pair level:
  (i) |s(masc)| − |s(fem)|   (the Eq. 1 term for one pair)
  (ii) s(masc) + s(fem)      (signed pair lean)
Raw dot products for Spanish (as their mweat), cosine for English. Spearman ρ with a permutation
p-value (10,000, seed 0), over Zhou's 58 pairs (published) and the 54-pair clean list.

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/zhou_sec43_44.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from c1_run import EN_F, EN_M
from zhou_shift import WEAT_F, WEAT_M, Z, load_vec
from zhou_wordpair import spaces

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/zhou_sec43_44"
DD = Path("/Volumes/Lexar/gbleed/shared/embeddings/muse/dictionaries")
TABLE2 = {("en-es", "original"): (79.2, 89.0), ("en-es", "shift_ori"): (80.7, 90.3), ("en-es", "hybrid_ori"): (80.7, 90.3),
          ("es-en", "original"): (79.2, 89.0), ("es-en", "shift_ori"): (79.2, 89.0), ("es-en", "hybrid_ori"): (79.2, 89.0),
          ("en-fr", "original"): (78.2, 89.4), ("en-fr", "shift_ori"): (79.9, 91.1), ("en-fr", "hybrid_ori"): (79.9, 91.1),
          ("fr-en", "original"): (76.1, 88.1), ("fr-en", "shift_ori"): (76.1, 88.1), ("fr-en", "hybrid_ori"): (76.1, 88.1)}  # fmt: skip


def unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def precision(S, s_w2i, T, t_w2i, pairs):
    gold = {}
    for a, b in pairs:
        if a in s_w2i and b in t_w2i:
            gold.setdefault(a, set()).add(t_w2i[b])
    q = list(gold)
    Tn = unit(T)
    hit1 = hit5 = 0
    for i in range(0, len(q), 512):
        Q = unit(S[[s_w2i[w] for w in q[i:i + 512]]])
        top = np.argsort(-(Q @ Tn.T), axis=1)[:, :5]
        for w, row in zip(q[i:i + 512], top):
            hit1 += row[0] in gold[w]
            hit5 += bool(gold[w] & set(row))
    return 100 * hit1 / len(q), 100 * hit5 / len(q), len(q)


def spearman_perm(x, y, rng, n=10_000):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    r = np.corrcoef(rx, ry)[0, 1]
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return r, (np.sum(np.abs(null) >= abs(r)) + 1) / (n + 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    rows = []
    for lang in ("es", "fr"):
        sp, w2i, occ = spaces(lang, en, en_w2i)
        for name, (EN, _, T) in sp.items():
            for d, (S, sw, Tt, tw) in ((f"en-{lang}", (EN, en_w2i, T, w2i)), (f"{lang}-en", (T, w2i, EN, en_w2i))):
                pairs = [line.split() for line in open(DD / f"{d}.5000-6500.txt", encoding="utf-8")]
                p1, p5, n = precision(S, sw, Tt, tw, pairs)
                rows.append({"dir": d, "space": name, "P@1": p1, "P@5": p5, "n_queries": n,
                             "paper_P@1": TABLE2[(d, name)][0], "paper_P@5": TABLE2[(d, name)][1]})  # fmt: skip
    tr = pd.DataFrame(rows)
    tr.to_csv(OUT / "translation.csv", index=False)

    # §4.4, Spanish original space
    es, es_w2i = load_vec(Z / "data/wiki.multi.es.vec.txt")
    En = unit(en)
    am, af = [En[en_w2i[w]] for w in EN_M if w in en_w2i], [En[en_w2i[w]] for w in EN_F if w in en_w2i]
    occ = [line.rstrip("\n").split("\t") for line in open(Z / "ES_occupation_words_with_EN_translations")]
    mp = pd.read_csv(ROOT / "data/occupation_mapping_v1.csv")
    keep = mp[mp.lang == "es"].clean_keep.to_numpy(bool)

    def s_es(w):
        v = es[es_w2i[w]]
        return np.mean([es[es_w2i[a]] @ v for a in WEAT_M]) - np.mean([es[es_w2i[a]] @ v for a in WEAT_F])

    per = []
    for (f, m, e), k in zip(occ, keep):
        v = En[en_w2i[e]]
        sm, sf = s_es(m), s_es(f)
        per.append({"en": e, "masc": m, "fem": f, "clean": k, "weat_en": np.mean([a @ v for a in am]) - np.mean([a @ v for a in af]),
                    "mweat_es_i": abs(sm) - abs(sf), "mweat_es_ii": sm + sf})  # fmt: skip
    per = pd.DataFrame(per)
    per.to_csv(OUT / "per_occupation.csv", index=False)
    rng = np.random.default_rng(0)
    corr = []
    for lst, d in (("as_published (58)", per), ("clean (54)", per[per.clean])):
        for col in ("mweat_es_i", "mweat_es_ii"):
            r, p = spearman_perm(d.weat_en, d[col], rng)
            corr.append({"list": lst, "spanish_score": col, "spearman": r, "p_perm": p, "n": len(d)})
    corr = pd.DataFrame(corr)
    corr.to_csv(OUT / "correlation.csv", index=False)
    with pd.option_context("display.width", 200):
        print(tr.round(1).to_string(index=False))
        print("paper §4.4: Spearman 0.45, p 0.0004")
        print(corr.round(4).to_string(index=False))
    print("=== SEC4344 done")


if __name__ == "__main__":
    main()
