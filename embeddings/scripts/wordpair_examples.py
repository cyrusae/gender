"""Representative word-pair queries for the paper's worked example (exploratory, 2026-11-23, PI request).

Same queries and scoring as the cue study (original space, clean occupations, frozen cues v1). Selection
fixed before looking at any query (no cherry-picking):
  1. occupation = the one at the median of per-occupation intrusion (mean over feminine cues of
     RR masc − RR fem); with an even count, the lower-middle one;
  2. for that occupation, the gendered adjective pair at the median of its feminine-cue intrusion, and the
     invariant adjective at the median of its RR masc − RR fem.
For each selected query: rank of the masculine and of the feminine occupation form (0 = outside the top
999) and the top five retrieved words. Output: results/wordpair_examples/examples.md and .csv.

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/wordpair_examples.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from wordpair_mechanism import K, rr_table
from zhou_shift import Z, load_vec
from zhou_wordpair import spaces

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/wordpair_examples"


def query(EN, T, en_w2i, w2i, i2w, ce, cx, e):
    Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
    q = EN[en_w2i[e]] - EN[en_w2i[ce]] + T[w2i[cx]]
    s = Tn @ (q / np.linalg.norm(q))
    return [i2w[i] for i in np.argsort(-s)[:K] if i2w[i] != cx][:999]


def median_index(x):
    order = np.argsort(x, kind="stable")
    return int(order[(len(x) - 1) // 2])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cues = pd.read_csv(ROOT / "data/wp_cues_frozen_v1.csv", keep_default_na=False, dtype=str)
    mp = pd.read_csv(ROOT / "data/occupation_mapping_v1.csv")
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    rows, md = [], ["# Representative word-pair queries (original space)\n"]
    for lang in ("es", "fr"):
        sp, w2i, occ_all = spaces(lang, en, en_w2i)
        EN, _, T = sp["original"]
        i2w = {i: w for w, i in w2i.items()}
        m = mp[mp.lang == lang].reset_index(drop=True)
        keep = m.clean_keep.to_numpy(bool) & np.array([e in en_w2i for _, _, e in occ_all])
        occ = [o for o, k in zip(occ_all, keep) if k]
        cs = cues[(cues.lang == lang) & (cues.subset == "primary")]
        G, Inv = cs[cs.type == "gendered"].reset_index(drop=True), cs[cs.type == "invariant"].reset_index(drop=True)
        fc = [(e, x, "f") for e, x in zip(G.en, G.fem)]
        ic = [(e, x, "inv") for e, x in zip(Inv.en, Inv.masc)]
        Rf = rr_table(EN, T, en_w2i, w2i, i2w, fc, occ)
        Ri = rr_table(EN, T, en_w2i, w2i, i2w, ic, occ)
        per_occ = (Rf[..., 0] - Rf[..., 1]).mean(0)
        oi = median_index(per_occ)
        f, mm, e = occ[oi]
        pi = median_index(Rf[:, oi, 0] - Rf[:, oi, 1])
        ii = median_index(Ri[:, oi, 0] - Ri[:, oi, 1])
        md.append(f"## {lang.upper()}: *{e}* ({mm} / {f}); occupation intrusion {per_occ[oi]:.3f} "
                  f"(median of {len(occ)}; mean over occupations {per_occ.mean():.3f})\n")  # fmt: skip
        md.append("| Cue type | English | Cue | Query | Rank of masc. | Rank of fem. | Top 5 |")
        md.append("| --- | --- | --- | --- | --- | --- | --- |")
        sel = (("masculine", G.en[pi], G.masc[pi]), ("feminine", G.en[pi], G.fem[pi]), ("invariant", Inv.en[ii], Inv.masc[ii]))
        for typ, ce, cx in sel:
            top = query(EN, T, en_w2i, w2i, i2w, ce, cx, e)
            rk = {w: r + 1 for r, w in enumerate(top)}
            r_m, r_f = rk.get(mm, 0), rk.get(f, 0)
            rows.append({"lang": lang, "occupation_en": e, "masc": mm, "fem": f, "cue_type": typ, "cue_en": ce, "cue": cx,
                         "rank_masc": r_m, "rank_fem": r_f, "top5": " ".join(top[:5])})  # fmt: skip
            md.append(f"| {typ} | {ce} | *{cx}* | {ce} : {e} = {cx} : ? | {r_m or '>999'} | {r_f or '>999'} | "
                      + ", ".join(f"*{w}*" for w in top[:5]) + " |")  # fmt: skip
        md.append("")
        print("done", lang, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "examples.csv", index=False)
    (OUT / "examples.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print("=== WPEX done")


if __name__ == "__main__":
    main()
