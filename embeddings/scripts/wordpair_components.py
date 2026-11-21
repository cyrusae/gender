"""Component MRRs of the word-pair test as defined (exploratory, 2026-11-20; PI request for the paper's
"Table 2 as defined"). The cue study (wordpair_cues.py) saved only the agreement gap
M-MRR − F-MRR for mitigated spaces; this recomputes its two parts per space, with the same queries,
cues, occupations and scoring:
  M-MRR = mean reciprocal rank of the masculine form under the masculine cue of each gendered pair
  F-MRR = mean reciprocal rank of the feminine form under the feminine cue
Bootstrap 10,000 (seed 0) over gendered pairs and occupations, as in wordpair_cues.py. Run once per
centring: ZHOU_CENTRE=notebook (released code, default) or midpoint (as described); output
results/wordpair_components/<centring>.csv. The difference reproduces wordpair_cues' agreement_gap.

    ZHOU_CENTRE=midpoint uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/wordpair_components.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from wordpair_mechanism import rr_table
from zhou_shift import Z, load_vec
from zhou_wordpair import spaces

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/wordpair_components"
N_BOOT, SEED = 10_000, 0
CENTRE = os.environ.get("ZHOU_CENTRE", "notebook")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cues = pd.read_csv(ROOT / "data/wp_cues_frozen_v1.csv", keep_default_na=False, dtype=str)
    mp = pd.read_csv(ROOT / "data/occupation_mapping_v1.csv")
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    rng = np.random.default_rng(SEED)
    rows = []
    for lang in ("es", "fr"):
        sp, w2i, occ_all = spaces(lang, en, en_w2i)
        i2w = {i: w for w, i in w2i.items()}
        m = mp[mp.lang == lang].reset_index(drop=True)
        keep = m.clean_keep.to_numpy(bool) & np.array([e in en_w2i for _, _, e in occ_all])
        occ = [o for o, k in zip(occ_all, keep) if k]
        G = cues[(cues.lang == lang) & (cues.subset == "primary") & (cues.type == "gendered")].reset_index(drop=True)
        mc = [(e, x, "m") for e, x in zip(G.en, G.masc)]
        fc = [(e, x, "f") for e, x in zip(G.en, G.fem)]
        tabs = {name: (rr_table(EN, T, en_w2i, w2i, i2w, mc, occ)[..., 0], rr_table(EN, T, en_w2i, w2i, i2w, fc, occ)[..., 1])
                for name, (EN, _, T) in sp.items()}  # fmt: skip
        n_p, n_o = len(G), len(occ)
        draws = {(s, k): [] for s in tabs for k in ("M_MRR", "F_MRR", "diff")}
        for _ in range(N_BOOT):
            pi, oi = rng.integers(0, n_p, n_p), rng.integers(0, n_o, n_o)
            for s, (M, F) in tabs.items():
                a, b = M[np.ix_(pi, oi)].mean(), F[np.ix_(pi, oi)].mean()
                draws[(s, "M_MRR")].append(a)
                draws[(s, "F_MRR")].append(b)
                draws[(s, "diff")].append(a - b)
        for s, (M, F) in tabs.items():
            pt = {"M_MRR": M.mean(), "F_MRR": F.mean(), "diff": M.mean() - F.mean()}
            for k, v in pt.items():
                lo, hi = np.percentile(draws[(s, k)], [2.5, 97.5])
                rows.append({"lang": lang, "centring": CENTRE, "space": s, "measure": k, "value": v, "lo": lo, "hi": hi,
                             "n_pairs": n_p, "n_occ": n_o})  # fmt: skip
        print("done", lang, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"{CENTRE}.csv", index=False)
    print(d.round(3).to_string(index=False))
    print("=== WPCOMP done", CENTRE)


if __name__ == "__main__":
    main()
