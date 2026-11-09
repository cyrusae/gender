"""C1: Zhou et al.'s inanimate-noun score after Zhou et al.'s mitigation (embeddings/docs/c1-prereg.md,
adopted 2026-11-08). Test nouns: frozen embeddings/data/c1_nouns_es_v1.csv.

Arms: 1 original; 2 Shift Ori "as implemented" = OUR RECONSTRUCTION (released code never applies
Eq. 2; see docs/zhou-shift-identity.md) (occupation pairs shifted, the semantic direction
projected out of Zhou et al.'s own noun lists only, so our nouns are untouched); 3 Shift Ori as
specified (Eq. 2: semantic direction projected out of every inanimate noun, ours included);
4 Hybrid Ori (Spanish mapped onto hard-debiased English, `zhou_hybrid.py`), as implemented and as
specified. Attributes: (a) Zhou's Spanish 8 + 8 WEAT words; (b) their English definitional words,
taken from the English space each arm is aligned to (original or hard-debiased English).

s(w) = mean cos(w, male words) − mean cos(w, female words); Δ = mean s(masc) − mean s(fem).
95% bootstrap CI (2,000 resamples within gender, seed 0); two-sided label-permutation p (10,000).
Directions (semantic PCA, grammatical LinearSVC, orthogonalised) from the first 50k words, as in
`zhou_shift.py`; vocab 50k primary, 200k sensitivity.

    uv run --no-project --python 3.12 --with numpy,scikit-learn,pandas python embeddings/scripts/c1_run.py [es|fr]
French (Addendum A): French attributes = Zhou's French definitional pairs; spelling split = fr_suffix.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from zhou_hybrid import DICT, hard_debias, procrustes
from zhou_shift import WEAT_F, WEAT_M, Z, load_vec, nouns, pca_dir, shift

ROOT = Path(__file__).resolve().parents[1]
SHA = {"es": "2c58b883bb1ba3ebf613351c65ce00ad2a19782e03cb112a8bee76bebbc4d0c2",
       "fr": "0e8687a39caae54f7094b42d0c6fada7fafab748cf8d2c389c3f450f0196e51c"}  # fmt: skip
EN_DEFS = json.load(open(Z / "EN_definitional_pairs.json"))
EN_F, EN_M = [f for f, _ in EN_DEFS], [m for _, m in EN_DEFS]


def unit(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def directions(E, w2i50, lang):
    d = pca_dir(json.load(open(Z / f"{lang.upper()}_definitional_pairs.json")), E, w2i50)
    gp = list(zip(nouns(f"feminine_nouns_{lang}", w2i50), nouns(f"masculine_nouns_{lang}", w2i50)))[:3000]
    X = np.array([E[w2i50[w]] for p in gp for w in p])
    g = LinearSVC().fit(X, np.tile([1, 2], len(gp))).coef_.reshape(-1)
    g /= np.linalg.norm(g)
    ds = d - (d @ g) * g
    return ds / np.linalg.norm(ds), d @ g


def project_out(E, rows, v):
    E = E.copy()
    E[rows] -= np.outer(E[rows] @ v, v)
    return E


def scores(E, idx, M, F):
    N = unit(E[idx])
    return (N @ unit(M).T).mean(1) - (N @ unit(F).T).mean(1)


def delta_stats(s, is_m, rng):
    sm, sf = s[is_m], s[~is_m]
    d = sm.mean() - sf.mean()
    boot = [rng.choice(sm, len(sm)).mean() - rng.choice(sf, len(sf)).mean() for _ in range(2000)]
    perm = []
    for _ in range(10000):
        p = rng.permutation(is_m)
        perm.append(s[p].mean() - s[~p].mean())
    return d, *np.percentile(boot, [2.5, 97.5]), float(np.mean(np.abs(perm) >= abs(d)))


def main(lang="es"):
    nouns_csv = ROOT / f"data/c1_nouns_{lang}_v1.csv"
    assert hashlib.sha256(nouns_csv.read_bytes()).hexdigest() == SHA[lang], "frozen noun list changed"
    OUT = ROOT / f"results/c1_{lang}"
    OUT.mkdir(parents=True, exist_ok=True)
    nl = pd.read_csv(nouns_csv)
    if lang == "es":
        att_m, att_f = WEAT_M, WEAT_F
        regular = nl["es_ending"].isin(["o", "a"]).to_numpy() & nl["es_exception"].isna().to_numpy()
        other = (nl["es_ending"] == "other").to_numpy()
        exc = nl["es_exception"].notna().to_numpy()
    else:  # Addendum A: French attributes = Zhou et al.'s French definitional pairs
        fdefs = json.load(open(Z / "FR_definitional_pairs.json"))
        att_m, att_f = [b for _, b in fdefs], [a for a, _ in fdefs]
        regular = nl["fr_suffix"].notna().to_numpy()
        other = ~regular
        exc = (nl["fr_suffix_regular"] == "no").to_numpy()
    es, w2i = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt", nmax=200000)
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    w2i50 = {w: i for w, i in w2i.items() if i < 50000}
    en_db, _ = hard_debias(en, en_w2i)
    seen, src, tgt = set(), [], []
    for line in open(Path(str(DICT).replace("es-en", f"{lang}-en")), encoding="utf-8"):
        a, b = line.split()
        if a in w2i50 and b in en_w2i and a not in seen:
            seen.add(a)
            src.append(w2i[a])
            tgt.append(en_w2i[b])
    o = np.argsort(src)[:5000]
    es_da = es @ procrustes(es[np.array(src)[o]], en_db[np.array(tgt)[o]]).T

    occ = [line.rstrip("\n").split("\t") for line in open(Z / f"{lang.upper()}_occupation_words_with_EN_translations")]
    pairs = [(m, f) for f, m, _ in occ]
    zhou_rows = [w2i[w] for w in set(nouns(f"masculine_nouns_{lang}", w2i50) + nouns(f"feminine_nouns_{lang}", w2i50))]
    test_rows = nl["lemma"].map(w2i).to_numpy()
    assert not set(test_rows) & set(zhou_rows)

    spaces, info = {}, {}
    for base, E, EN in [("shift", es, en), ("hybrid", es_da, en_db)]:
        ds, cos = directions(E, w2i50, lang)
        info[f"{base}_cos_sem_gram"] = float(abs(cos))
        impl = project_out(shift(E, w2i, pairs, ds), zhou_rows, ds)
        spec = project_out(impl, test_rows, ds)
        if base == "shift":
            spaces["1_original"] = (E, EN)
            spaces["2_shift_ori_as_implemented"] = (impl, EN)
            spaces["3_shift_ori_as_specified"] = (spec, EN)
        else:
            spaces["4a_hybrid_ori_as_implemented"] = (impl, EN)
            spaces["4b_hybrid_ori_as_specified"] = (spec, EN)

    rows, per_noun = [], {}
    for arm, (E, EN) in spaces.items():
        attrs = {
            lang: (E[[w2i[w] for w in att_m]], E[[w2i[w] for w in att_f]]),
            "en": (EN[[en_w2i[w] for w in EN_M]], EN[[en_w2i[w] for w in EN_F]]),
        }
        for att, (M, F) in attrs.items():
            s_all = scores(E, test_rows, M, F)
            per_noun[f"{arm}|{att}"] = s_all
            for vocab in ("50k", "200k"):
                keep = nl["in_50k"].to_numpy() if vocab == "50k" else np.ones(len(nl), bool)
                for subset, mask in [("all", keep), ("regular", keep & regular), ("other_ending", keep & other)]:
                    rng = np.random.default_rng(0)
                    is_m = (nl["gender"] == "m").to_numpy()[mask]
                    d, lo, hi, p = delta_stats(s_all[mask], is_m, rng)
                    rows.append({"arm": arm, "attributes": att, "vocab": vocab, "subset": subset,
                                 "n_m": int(is_m.sum()), "n_f": int((~is_m).sum()), "delta": d,
                                 "ci_lo": lo, "ci_hi": hi, "p_perm": p,
                                 "mean_abs_bw": float(np.abs(s_all[mask]).mean())})  # fmt: skip
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "c1_deltas.csv", index=False)
    pn = nl[["lemma", "gender", "es_ending" if lang == "es" else "fr_suffix", "concept_en", "zipf", "in_50k"]].copy()
    for k, v in per_noun.items():
        pn[k] = v
    pn.to_csv(OUT / "c1_per_noun.csv", index=False)
    # exceptions, descriptive only
    desc = {k: {g: float(v[exc & (nl["gender"] == g).to_numpy()].mean()) for g in "mf"} for k, v in per_noun.items()}
    json.dump({"info": info, "exceptions_mean_s": desc}, open(OUT / "c1_info.json", "w"), indent=1)

    # pre-registered reading, primary = 50k, all nouns
    verdicts = []
    for att in (lang, "en"):
        base = res.query("arm=='1_original' and attributes==@att and vocab=='50k' and subset=='all'").iloc[0]
        for arm in spaces:
            if arm == "1_original":
                continue
            r = res.query("arm==@arm and attributes==@att and vocab=='50k' and subset=='all'").iloc[0]
            excl0 = r.ci_lo > 0 or r.ci_hi < 0
            if att == "en" and base.ci_lo <= 0 <= base.ci_hi:
                v = "no social-gender lean to remove"
            elif excl0 and abs(r.delta) >= 0.5 * abs(base.delta):
                v = "survives"
            elif not excl0 and abs(r.delta) < 0.25 * abs(base.delta):
                v = "removed"
            else:
                v = "reduced"
            verdicts.append({"attributes": att, "arm": arm, "delta_original": base.delta,
                             "delta_arm": r.delta, "ratio": r.delta / base.delta, "verdict": v})  # fmt: skip
    pd.DataFrame(verdicts).to_csv(OUT / "c1_verdicts.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.query("subset=='all'").round(4).to_string(index=False))
        print(pd.DataFrame(verdicts).round(4).to_string(index=False))
        print(json.dumps(info))
    print("=== C1 done")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "es")
