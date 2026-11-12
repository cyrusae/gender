"""Check that the MUSE vectors are unit length (2026-11-12). Zhou et al.'s MWEAT and our scripts use raw
dot products; these equal cosines only for unit vectors. Writes results/unit_norm_check.csv and fails
if any loaded vector's length differs from 1 by more than 1e-5. (Vectors after Zhou's Shift are not
renormalised; that is part of upstream Z14, not checked here.)

    uv run --no-project --python 3.12 --with numpy,pandas python embeddings/scripts/check_unit_norm.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from zhou_shift import Z, load_vec

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-5


def main():
    rows = []
    for lang in ("es", "fr", "en"):
        E, _ = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt", nmax=200_000)
        n = np.linalg.norm(E, axis=1)
        for k in (50_000, 200_000):
            m = n[:k]
            rows.append({"lang": lang, "first_n_words": k, "min": m.min(), "max": m.max(),
                         "max_abs_dev_from_1": np.abs(1 - m).max()})  # fmt: skip
    d = pd.DataFrame(rows)
    d.to_csv(ROOT / "results/unit_norm_check.csv", index=False)
    print(d.to_string(index=False))
    assert (d.max_abs_dev_from_1 < TOL).all(), "non-unit vectors"
    print("all unit length within", TOL)


if __name__ == "__main__":
    main()
