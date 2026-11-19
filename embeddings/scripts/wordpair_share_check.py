"""Share-of-queries version of the feminine-cue word-pair test (exploratory, 2026-11-14).

Zhou et al.'s correction draft (footnote 12) reports the share of feminine-cue queries in which the
masculine occupation form outranks the feminine one. This computes that share on our frozen cue
set (data/wp_cues_frozen_v1.csv, primary gendered pairs) beside our own measures, with queries,
spaces and scoring exactly as wordpair_cues.py (misses = reciprocal rank 0, so a form outside the
top 999 loses to one inside it; both outside = tie).
  share_m_outranks = share of (feminine cue, occupation) queries with RR_m > RR_f
  intrusion        = mean over feminine cues of (RR_m − RR_f)        (as wordpair_cues.py)
  agreement_gap    = mean RR_m over masculine cues − mean RR_f over feminine cues
No intervals: descriptive only (intervals for intrusion are in results/wordpair_cues/gaps.csv).

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/wordpair_share_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from wordpair_mechanism import rr_table
from zhou_shift import Z, load_vec
from zhou_wordpair import spaces

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/wordpair_share_check"


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
        assert [tuple(x) for x in m[["fem", "masc", "en"]].values] == [tuple(o) for o in occ_all]
        keep = m.clean_keep.to_numpy(bool) & np.array([e in en_w2i for _, _, e in occ_all])
        occ = [o for o, k in zip(occ_all, keep) if k]
        G = cues[(cues.lang == lang) & (cues.subset == "primary") & (cues.type == "gendered")]
        fc = [(e, x, "f") for e, x in zip(G.en, G.fem)]
        mc = [(e, x, "m") for e, x in zip(G.en, G.masc)]
        for name, (EN, _, T) in sp.items():
            Rf = rr_table(EN, T, en_w2i, w2i, i2w, fc, occ)
            Rm = rr_table(EN, T, en_w2i, w2i, i2w, mc, occ)
            rows.append({"lang": lang, "space": name, "n_cues": len(fc), "n_occ": len(occ),
                         "share_m_outranks": (Rf[..., 0] > Rf[..., 1]).mean(),
                         "share_tie": (Rf[..., 0] == Rf[..., 1]).mean(),
                         "M_MRR_femcue": Rf[..., 0].mean(), "F_MRR_femcue": Rf[..., 1].mean(),
                         "intrusion": (Rf[..., 0] - Rf[..., 1]).mean(),
                         "agreement_gap": Rm[..., 0].mean() - Rf[..., 1].mean()})  # fmt: skip
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "share_check.csv", index=False)
    print(D.round(3).to_string(index=False))
    print("=== WPSHARE done")


if __name__ == "__main__":
    main()
