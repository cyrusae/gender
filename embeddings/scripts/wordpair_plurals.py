"""Word-pair test with number (paper B; docs/wordpair-plurals-prereg.md, ADOPTED 2026-11-24). Committed
before any score.

Original MUSE space; target vocabulary the first 200k words of the language (PLURALS_VOCAB to override,
used for the 50k sensitivity). For each query, the reciprocal rank (1/rank among the top 999, cue form
removed; 0 otherwise) of the four occupation forms: masc sg, masc pl, fem sg, fem pl.
  singular arm (Q1-Q3, Q5): EN[occupation] - EN[cue] + T[cue form]
  plural arm (Q4, Q5):      EN[occupation pl] - EN[cue] + T[cue plural form]
Occupations: clean list with all four forms in the target vocabulary (and, for Q4, the English plural
in the first 200k English words). Cues: frozen v1 primary subset; plural forms from
data/wp_plural_cues_v0.csv (cues without both plural forms in the vocabulary are dropped from Q4).
Bootstrap 10,000 (seed 0): occupations, gendered pairs (masc and fem cue together) and invariant cues,
resampled independently; all measures computed on the same draw (paired differences).
Check before scoring: singular intrusion with PLURALS_VOCAB=50000 on the full clean list reproduces the
cue study (ES 0.291, FR 0.191); run with PLURALS_CHECK=1.

    uv run --no-project --python 3.12 --with numpy,pandas python embeddings/scripts/wordpair_plurals.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from zhou_shift import Z, load_vec

ROOT = Path(__file__).resolve().parents[1]
NV = int(os.environ.get("PLURALS_VOCAB", "200000"))
CHECK = os.environ.get("PLURALS_CHECK") == "1"
OUT = ROOT / ("results/wordpair_plurals" + ("" if NV == 200000 else f"_{NV // 1000}k"))
N_BOOT, SEED, TOP = 10_000, 0, 999
FORMS = ("m_sg", "m_pl", "f_sg", "f_pl")


def rr(EN, en_w2i, Tn, w2i, queries, forms_idx):
    """queries: list of (en_occ, en_cue, cue_form); forms_idx: [n, 4] target indices. Returns [n, 4] RRs."""
    out = np.zeros((len(queries), 4))
    for s in range(0, len(queries), 256):
        qs = queries[s:s + 256]
        Q = np.stack([EN[en_w2i[o]] - EN[en_w2i[c]] + Tn_raw[w2i[x]] for o, c, x in qs])
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        S = Tn @ Q.T  # [V, b]
        for j, (_, _, x) in enumerate(qs):
            col = S[:, j]
            cue = col[w2i[x]]
            for k, t in enumerate(forms_idx[s + j]):
                v = col[t]
                r = int((col > v).sum()) - int(cue > v) + 1
                out[s + j, k] = 1.0 / r if r <= TOP else 0.0
    return out


def measures(R, pi, ii, oi):
    """R: dict arm -> dict cue type -> array [cue, occ, 4]. Returns flat dict of measures."""
    m = {}
    for arm in R:
        X = {t: R[arm][t][np.ix_(ii if t == "inv" else pi, oi)] for t in R[arm]}
        for t, A in X.items():
            mean = A.reshape(-1, 4).mean(0)
            tot = mean.sum()
            for k, f in enumerate(FORMS):
                m[f"{arm}|{t}|rr_{f}"] = mean[k]
            num_err = (mean[1] + mean[3]) / tot if arm == "sg" else (mean[0] + mean[2]) / tot
            m[f"{arm}|{t}|number_error_share"] = num_err
            if t == "m":
                m[f"{arm}|{t}|gender_error_share"] = (mean[2] + mean[3]) / tot
            if t == "f":
                m[f"{arm}|{t}|gender_error_share"] = (mean[0] + mean[1]) / tot
        F = X["f"]
        sing = (F[..., 0] - F[..., 2]).mean()
        if arm == "sg":
            m["Q1_gender_level_intrusion"] = (np.maximum(F[..., 0], F[..., 1]) - np.maximum(F[..., 2], F[..., 3])).mean()
            m["S_sum_intrusion"] = ((F[..., 0] + F[..., 1]) - (F[..., 2] + F[..., 3])).mean()
            m["singular_intrusion"] = sing
            m["Q2_gender_level_minus_singular"] = m["Q1_gender_level_intrusion"] - sing
            m["Q3_mpl_minus_fpl_under_fem"] = (F[..., 1] - F[..., 3]).mean()
            m["Q3_mpl_minus_msg_under_fem"] = (F[..., 1] - F[..., 0]).mean()
        else:
            m["Q4_plural_intrusion"] = (F[..., 1] - F[..., 3]).mean()
    if "pl" in R:
        m["Q4_plural_minus_singular_intrusion"] = m["Q4_plural_intrusion"] - m["singular_intrusion"]
    return m


def main():
    global Tn_raw
    OUT.mkdir(parents=True, exist_ok=True)
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    cues = pd.read_csv(ROOT / "data/wp_cues_frozen_v1.csv", keep_default_na=False, dtype=str)
    pcues = pd.read_csv(ROOT / "data/wp_plural_cues_v0.csv", keep_default_na=False, dtype=str)
    gm = pd.read_csv(ROOT / "data/gm_pairs_frozen_v1.csv", keep_default_na=False, dtype=str)
    mp = pd.read_csv(ROOT / "data/occupation_mapping_v1.csv")
    enp = pd.read_csv(ROOT / "data/wp_plural_occ_en_v0.csv", keep_default_na=False, dtype=str)
    summary, info, verdicts = [], {"vocab": NV}, []
    rng = np.random.default_rng(SEED)
    for lang in ("es", "fr"):
        T, w2i = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt", nmax=NV)
        Tn_raw = T
        Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
        m = mp[(mp.lang == lang) & mp.clean_keep].merge(gm[gm.lang == lang][["masc", "fem", "masc_pl", "fem_pl"]],
                                                         on=["masc", "fem"], how="left")  # fmt: skip
        m = m.merge(enp[enp.lang == lang][["masc", "en_pl"]], on="masc", how="left")
        m = m[m.en.isin(en_w2i)]
        if CHECK:
            occ = m[m.masc.isin(w2i) & m.fem.isin(w2i)]
        else:
            occ = m[m[["masc", "fem", "masc_pl", "fem_pl"]].isin(w2i).all(axis=1)]
        occ = occ.reset_index(drop=True)
        cs = cues[(cues.lang == lang) & (cues.subset == "primary")]
        G = cs[cs.type == "gendered"].reset_index(drop=True)
        I = cs[cs.type == "invariant"].reset_index(drop=True)
        pc = pcues[pcues.lang == lang]
        Gp = G.merge(pc[pc.type == "gendered"][["masc", "masc_pl", "fem_pl"]], on="masc", how="left")
        Ip = I.merge(pc[pc.type == "invariant"][["masc", "masc_pl"]], on="masc", how="left")
        g_ok = Gp.masc_pl.isin(w2i) & Gp.fem_pl.isin(w2i)
        i_ok = Ip.masc_pl.isin(w2i)
        info[lang] = {"n_occupations": len(occ), "occupations_dropped": sorted(set(m.masc) - set(occ.masc)),
                      "n_gendered": len(G), "n_invariant": len(I),
                      "n_gendered_plural": int(g_ok.sum()), "n_invariant_plural": int(i_ok.sum())}  # fmt: skip
        fidx = np.array([[w2i.get(r[c], 0) for c in ("masc", "masc_pl", "fem", "fem_pl")] for _, r in occ.iterrows()])
        if CHECK:
            fidx[:, 1] = fidx[:, 0]
            fidx[:, 3] = fidx[:, 2]

        def table(cue_rows, form_col, occ_col):
            q, fi = [], []
            for _, c in cue_rows.iterrows():
                for oi, o in occ.iterrows():
                    q.append((o[occ_col], c.en, c[form_col]))
                    fi.append(fidx[oi])
            return rr(en, en_w2i, Tn, w2i, q, np.array(fi)).reshape(len(cue_rows), len(occ), 4)

        R = {"sg": {"m": table(G, "masc", "en"), "f": table(G, "fem", "en"), "inv": table(I, "masc", "en")}}
        if CHECK:
            si = (R["sg"]["f"][..., 0] - R["sg"]["f"][..., 2]).mean()
            print(f"CHECK {lang} singular intrusion at {NV}: {si:.4f} (cue study: ES 0.291, FR 0.191)", flush=True)
            continue
        occ_pl_ok = occ.en_pl.isin(en_w2i).all()
        assert occ_pl_ok, "English plural outside 200k"
        Gq, Iq = Gp[g_ok].reset_index(drop=True), Ip[i_ok].reset_index(drop=True)
        R["pl"] = {"m": table(Gq, "masc_pl", "en_pl"), "f": table(Gq, "fem_pl", "en_pl"), "inv": table(Iq, "masc_pl", "en_pl")}
        # gendered pairs that lack plural forms are dropped from the plural arm only; resample within each arm's pairs
        n_o = len(occ)
        pt = measures({"sg": R["sg"]}, np.arange(len(G)), np.arange(len(I)), np.arange(n_o))
        pt_pl = measures({"pl": R["pl"]}, np.arange(len(Gq)), np.arange(len(Iq)), np.arange(n_o))
        draws = []
        for _ in range(N_BOOT):
            oi = rng.integers(0, n_o, n_o)
            a = measures({"sg": R["sg"]}, rng.integers(0, len(G), len(G)), rng.integers(0, len(I), len(I)), oi)
            b = measures({"pl": R["pl"]}, rng.integers(0, len(Gq), len(Gq)), rng.integers(0, len(Iq), len(Iq)), oi)
            a.update(b)
            a["Q4_plural_minus_singular_intrusion"] = a["Q4_plural_intrusion"] - a["singular_intrusion"]
            draws.append(a)
        pt.update(pt_pl)
        pt["Q4_plural_minus_singular_intrusion"] = pt["Q4_plural_intrusion"] - pt["singular_intrusion"]
        for k, v in pt.items():
            lo, hi = np.percentile([d[k] for d in draws], [2.5, 97.5])
            summary.append({"lang": lang, "measure": k, "value": v, "lo": lo, "hi": hi})
        S = {r["measure"]: r for r in summary if r["lang"] == lang}

        def pna(k):
            r = S[k]
            return "present" if r["lo"] > 0 else "absent" if r["hi"] < 0 else "inconclusive"

        q2 = S["Q2_gender_level_minus_singular"]
        verdicts.append({"lang": lang, "role": "primary" if lang == "es" else "descriptive",
                         "Q1": pna("Q1_gender_level_intrusion"),
                         "Q2": "singular-only overstates" if q2["hi"] < 0 else "singular-only understates" if q2["lo"] > 0
                         else "no clear difference",
                         "Q4": pna("Q4_plural_intrusion")})  # fmt: skip
        print("done", lang, flush=True)
    if CHECK:
        return
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(verdicts).to_csv(OUT / "verdicts.csv", index=False)
    json.dump(info, open(OUT / "info.json", "w"), indent=1, ensure_ascii=False)
    print(pd.DataFrame(verdicts).to_string(index=False))
    d = pd.DataFrame(summary)
    print(d[~d.measure.str.contains(r"\|rr_")].round(3).to_string(index=False))
    print("=== WPPLURALS done", NV)


if __name__ == "__main__":
    main()
