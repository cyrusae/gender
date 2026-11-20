"""English-default addendum A: expanded attribute words (docs/english-default-prereg.md, addendum A,
ADOPTED 2026-11-20). Committed before any score.

Reruns Q1–Q3 of scripts/english_default.py with two attribute lists instead of Caliskan et al.'s sixteen:
  expanded  38 sex-referring (male, female) pairs, data/en_attr_frozen_v1.csv (primary for the addendum);
            bootstrap resamples whole pairs.
  contrast  the 40 most masculine and 40 most feminine Glasgow words, data/en_attr_contrast_glasgow_v0.csv
            (stereotypes included on purpose); bootstrap resamples words within each side.
Everything else as english_default.py: same anchors, nouns, plain-translation gaps, regressions.

    uv run --no-project --python 3.12 --with numpy,pandas,scipy python embeddings/scripts/english_default_attr.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from english_default import FLAGS, GLOSSES_IN, N_BOOT, NS, SEED, plain_gap, reg
from scipy.stats import spearmanr
from zhou_shift import Z, load_vec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/english_default_attr"


def lists():
    p = pd.read_csv(ROOT / "data/en_attr_frozen_v1.csv")
    c = pd.read_csv(ROOT / "data/en_attr_contrast_glasgow_v0.csv")
    return {"expanded": (list(p.male), list(p.female), True),
            "contrast": (list(c[c.side == "male"].word), list(c[c.side == "female"].word), False)}  # fmt: skip


def boot_weights(nm, nf, paired, rng):
    W = np.zeros((N_BOOT, nm + nf))
    for b in range(N_BOOT):
        im = rng.integers(0, nm, nm)
        jf = im if paired else rng.integers(0, nf, nf)
        np.add.at(W[b], im, 1 / nm)
        np.add.at(W[b], nm + jf, -1 / nf)
    return W


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    anc = pd.read_csv(ROOT / "data/en_zero_frozen_v1.csv")
    sets = {"primary": list(anc[anc["set"] == "primary"].word),
            "epicene_gloss": list(anc[anc["set"] == "epicene_gloss"].word),
            "glosses_in": list(anc[anc["set"].isin(["primary", "glosses_in_extra"])].word)}  # fmt: skip
    rows1, rows23, lean_rows, info = [], [], [], {}
    leans = {}
    for lname, (M, F, paired) in lists().items():
        assert all(w in en_w2i for w in M + F), [w for w in M + F if w not in en_w2i]
        A = np.array([en[en_w2i[w]] for w in M + F])
        nm, nf = len(M), len(F)
        Call = en @ A.T
        sall = Call[:, :nm].mean(1) - Call[:, nm:].mean(1)
        leans[lname] = sall
        C = Call[:100_000]
        W = boot_weights(nm, nf, paired, rng)
        for tag, words in sets.items():
            idx = np.array([en_w2i[w] for w in words if w in en_w2i and en_w2i[w] < 100_000])
            z = sall[idx].mean()
            ba = rng.integers(0, len(idx), (N_BOOT, len(idx)))
            sh = {n: np.empty(N_BOOT) for n in NS}
            sh0 = {n: np.empty(N_BOOT) for n in NS}
            for i in range(0, N_BOOT, 500):
                S = C @ W[i:i + 500].T
                zb = np.array([S[idx[ba[i + j]], j].mean() for j in range(S.shape[1])])
                for n in NS:
                    sh[n][i:i + 500] = (S[:n] > zb).mean(0)
                    sh0[n][i:i + 500] = (S[:n] > 0).mean(0)
            for n in NS:
                lo, hi = np.percentile(sh[n], [2.5, 97.5])
                rows1.append({"attributes": lname, "anchors": tag, "N": n, "z": z,
                              "share_male_caliskan_zero": (sall[:n] > 0).mean(),
                              "caliskan_lo": np.percentile(sh0[n], 2.5), "caliskan_hi": np.percentile(sh0[n], 97.5),
                              "share_male_fair_zero": (sall[:n] > z).mean(), "fair_lo": lo, "fair_hi": hi,
                              "verdict_fair": "robust to the zero point" if lo > 0.5 else "reversed" if hi < 0.5
                              else "explained by the zero point"})  # fmt: skip
            info[(lname, tag)] = {"n_attr": f"{nm}/{nf}", "n_anchors": len(idx), "z": z,
                                  "spearman_fair_lean_vs_log_rank": spearmanr(sall[:100_000] - z, np.log(np.arange(1, 100_001)))[0]}  # fmt: skip
    q1 = pd.DataFrame(rows1)
    p = q1[(q1.anchors == "primary") & (q1.N == 1000)].set_index("attributes")
    half = (p.loc["expanded", "fair_hi"] - p.loc["expanded", "fair_lo"]) / 2
    diff = p.loc["contrast", "share_male_fair_zero"] - p.loc["expanded", "share_male_fair_zero"]
    choice = {"expanded_minus_contrast_halfwidth": half, "contrast_minus_expanded_share": diff,
              "verdict": "attribute choice matters" if abs(diff) > half else "not shown"}  # fmt: skip
    q1.to_csv(OUT / "q1_shares.csv", index=False)
    pd.DataFrame(info).T.to_csv(OUT / "q1_info.csv")
    pd.DataFrame([choice]).to_csv(OUT / "attribute_choice.csv", index=False)
    for lang in ("es", "fr"):
        T, w2i = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt")
        i2w = {i: w for w, i in w2i.items()}
        nouns = pd.read_csv(ROOT / f"data/wp_persons_{lang}_frozen_v1.csv", keep_default_na=False, dtype=str)
        nouns = nouns[nouns.group.isin(["person", "occupation"]) & nouns.en.isin(en_w2i)
                      & nouns.masc.isin(w2i) & nouns.fem.isin(w2i)].reset_index(drop=True)  # fmt: skip
        pn = pd.read_csv(ROOT / f"results/wordpair_persons{'' if lang == 'es' else '_fr'}/per_noun.csv", keep_default_na=False)
        nouns = nouns.merge(pn[["masc", "intrusion"]], on="masc", how="left", validate="one_to_one")
        nouns["gap_plain"] = plain_gap(en, en_w2i, T, w2i, i2w, nouns)
        nouns["log_rank_ratio"] = np.log(nouns.rank_f.astype(float) + 1) - np.log(nouns.rank_m.astype(float) + 1)
        nouns["occupation"] = (nouns.group == "occupation").astype(float)
        clean = ~nouns[FLAGS].eq("True").any(axis=1)
        for lname, sall in leans.items():
            nouns["en_lean_fair"] = [sall[en_w2i[e]] for e in nouns.en]  # zero point drops out of Q2/Q3
            for sel, d in (("all", nouns), ("no flagged nouns", nouns[clean]),
                           ("glosses_in (nouns dropped)", nouns[~nouns.en.isin(GLOSSES_IN)])):  # fmt: skip
                d = d.reset_index(drop=True)
                for r in reg(d, "gap_plain", ["en_lean_fair", "log_rank_ratio", "occupation"], rng):
                    rows23.append({"attributes": lname, "lang": lang, "set": sel, "question": "Q2 plain-translation gap", **r})
                dd = d.dropna(subset=["intrusion"]).reset_index(drop=True)
                for r in reg(dd, "intrusion", ["occupation", "en_lean_fair", "log_rank_ratio"], rng):
                    rows23.append({"attributes": lname, "lang": lang, "set": sel, "question": "Q3 feminine-cue intrusion", **r})
                lp, lo_ = d[d.group == "person"].en_lean_fair.to_numpy(), d[d.group == "occupation"].en_lean_fair.to_numpy()
                B = [rng.choice(lo_, len(lo_)).mean() - rng.choice(lp, len(lp)).mean() for _ in range(N_BOOT)]
                lean_rows.append({"attributes": lname, "lang": lang, "set": sel, "occ_minus_person": lo_.mean() - lp.mean(),
                                  "lo": np.percentile(B, 2.5), "hi": np.percentile(B, 97.5)})  # fmt: skip
    res = pd.DataFrame(rows23)

    def verdict(r):
        if r.question.startswith("Q2") and r.term == "en_lean_fair":
            return "English lean contributes" if r.lo > 0 else ("reversed" if r.hi < 0 else "not shown")
        if r.question.startswith("Q3") and r.term == "occupation":
            return "occupation-specific beyond English lean" if r.lo > 0 else "not shown"
        return ""

    res["verdict"] = res.apply(verdict, axis=1)
    res.to_csv(OUT / "q2_q3.csv", index=False)
    pd.DataFrame(lean_rows).to_csv(OUT / "lean_person_vs_occupation.csv", index=False)
    with pd.option_context("display.width", 220):
        print(q1.round(3).to_string(index=False))
        print(choice)
        print(res[res.term.isin(["en_lean_fair", "occupation"])].round(3).to_string(index=False))
        print(pd.DataFrame(lean_rows).round(4).to_string(index=False))
    print("=== ENDEFATTR done")


if __name__ == "__main__":
    main()
