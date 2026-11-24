"""Word-pair test as counts (exploratory, 2026-11-24, PI request: readers want "how often", which MRR is not).

Same queries, frozen v1 cues (primary), clean occupations (54 / 22) and scoring as the cue study; spaces
Original, Shift Ori, Hybrid Ori (ZHOU_CENTRE=midpoint for the described centring, default released).
For every query (cue x occupation), the ranks of the masculine and feminine occupation forms; reported per
cue type (masculine, invariant, feminine) and space, as counts and shares of queries:
  masc_top1 / fem_top1     the form is the first word retrieved
  masc_top10 / fem_top10   the form is among the first ten
  masc_wins / fem_wins     the form ranks above the other one (a form outside the top 999 counts as last)
  neither                  neither form in the top 999
No intervals (descriptive companion to the MRR results).

    ZHOU_CENTRE=midpoint uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/wordpair_counts.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from wordpair_mechanism import rr_table
from zhou_shift import Z, load_vec
from zhou_wordpair import spaces

ROOT = Path(__file__).resolve().parents[1]
CENTRE = os.environ.get("ZHOU_CENTRE", "notebook")
OUT = ROOT / "results/wordpair_counts"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cues = pd.read_csv(ROOT / "data/wp_cues_frozen_v1.csv", keep_default_na=False, dtype=str)
    mp = pd.read_csv(ROOT / "data/occupation_mapping_v1.csv")
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    rows = []
    for lang in ("es", "fr"):
        sp, w2i, occ_all = spaces(lang, en, en_w2i)
        i2w = {i: w for w, i in w2i.items()}
        m = mp[mp.lang == lang].reset_index(drop=True)
        keep = m.clean_keep.to_numpy(bool) & np.array([e in en_w2i for _, _, e in occ_all])
        occ = [o for o, k in zip(occ_all, keep) if k]
        cs = cues[(cues.lang == lang) & (cues.subset == "primary")]
        G, I = cs[cs.type == "gendered"], cs[cs.type == "invariant"]
        qs = {"masculine": [(e, x, "m") for e, x in zip(G.en, G.masc)],
              "feminine": [(e, x, "f") for e, x in zip(G.en, G.fem)],
              "invariant": [(e, x, "inv") for e, x in zip(I.en, I.masc)]}  # fmt: skip
        for s, (EN, _, T) in sp.items():
            for t, q in qs.items():
                R = rr_table(EN, T, en_w2i, w2i, i2w, q, occ).reshape(-1, 2)
                rm, rf = R[:, 0], R[:, 1]
                n = len(R)
                c = {"masc_top1": (rm == 1).sum(), "fem_top1": (rf == 1).sum(),
                     "masc_top10": (rm >= 0.1).sum(), "fem_top10": (rf >= 0.1).sum(),
                     "masc_wins": (rm > rf).sum(), "fem_wins": (rf > rm).sum(), "neither": ((rm == 0) & (rf == 0)).sum()}  # fmt: skip
                rows.append({"lang": lang, "centring": CENTRE, "space": s, "cue_type": t, "n_queries": n,
                             **{k: int(v) for k, v in c.items()}, **{f"{k}_share": v / n for k, v in c.items()}})  # fmt: skip
        print("done", lang, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"{CENTRE}.csv", index=False)
    print(d[["lang", "space", "cue_type", "n_queries"] + [c for c in d.columns if c.endswith("_share")]].round(3).to_string(index=False))
    print("=== WPCOUNTS done", CENTRE)


if __name__ == "__main__":
    main()
