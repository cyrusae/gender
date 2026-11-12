"""Exploratory check for upstream Z23 (2026-11-11): Zhou's Spanish WEAT lists pair *masculino* (male
side) with *femenina* (female side): the markers are the grammatical-gender terms, and they don't
match in form. Table 1 Original MWEAT (Spanish) and the epicene zero-point A (native yardstick, as
zhou_zero_point.py, original space) under: published (tio), corrected (tío; primary), corrected +
femenino for femenina, corrected without the masculino/femenina pair.

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/z23_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from zhou_shift import Z, load_vec, mweat
from zhou_zero_point import Setup, anchors, verdict

ROOT = Path(__file__).resolve().parents[1]
M_PUB = ["hombre", "niño", "padre", "masculino", "abuelo", "esposo", "hijo", "tio"]
F_PUB = ["niña", "femenina", "tía", "hija", "esposa", "mujer", "madre", "abuela"]
M_COR = [w if w != "tio" else "tío" for w in M_PUB]
VARIANTS = {
    "published (tio)": (M_PUB, F_PUB),
    "corrected (tío; primary)": (M_COR, F_PUB),
    "corrected + femenino": (M_COR, [w if w != "femenina" else "femenino" for w in F_PUB]),
    "corrected, no masculino/femenina": ([w for w in M_COR if w != "masculino"], [w for w in F_PUB if w != "femenina"]),
}


def main():
    E, w2i = load_vec(Z / "data/wiki.multi.es.vec.txt")
    occ = [line.rstrip("\n").split("\t") for line in open(Z / "ES_occupation_words_with_EN_translations")]
    pairs_fm = [(f, m) for f, m, _ in occ if f in w2i and m in w2i]
    pairs = [(m, f) for f, m in pairs_fm]
    ref0 = pd.read_csv(ROOT / "data/zero_ref_es_frozen_v1.csv")
    rows = []
    for name, (am, af) in VARIANTS.items():
        taken = {w for p in pairs for w in p} | set(M_PUB) | set(M_COR) | set(F_PUB) | {"femenino"}
        ref = ref0[ref0["lemma"].isin(w2i) & ~ref0["lemma"].isin(taken)].reset_index(drop=True)
        st = Setup(E, w2i, pairs, [E[w2i[w]] for w in am], [E[w2i[w]] for w in af], {"epicene_strict": anchors(ref)["epicene_strict"]})
        pt, B = st.point(), st.boot(10_000)
        a, (lo, hi) = pt["epicene_strict|A"], np.percentile(B["epicene_strict|A"], [2.5, 97.5])
        rows.append({"variant": name, "n_male": len(am), "n_female": len(af),
                     "table1_original_mweat": mweat(E, w2i, pairs_fm, am=am, af=af),
                     "zero_point_A": a, "A_lo": lo, "A_hi": hi, "verdict": verdict(a, lo, hi, pt["raw"])})  # fmt: skip
    res = pd.DataFrame(rows)
    out = ROOT / "results/z23_check"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "results.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
