"""Gender ValNorm, Omrani Sabbaghi & Caliskan erasure arm (docs/gvalnorm-prereg.md, adopted
2026-11-08; French: Addendum A). Reads the vector dumps from gg_dump.py.

s(w) = mean cos(w, their man words) − mean cos(w, their woman words), before and after erasure
(attribute words erased too, as in their method). Nouns: the Glasgow-matched C1 nouns in the CC
vocabulary, minus any word in their erasure training pairs. Same model and reading rule as
gvalnorm_run.py.

    uv run --no-project --python 3.12 --with numpy,pandas,scipy python embeddings/scripts/gvalnorm_omrani.py es 14 28
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gvalnorm_run import fit, verdict

ROOT = Path(__file__).resolve().parents[1]


def sc(W, M, F):
    u = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: E731
    W, M, F = u(W), u(M), u(F)
    return (W @ M.T).mean(1) - (W @ F.T).mean(1)


def main(lang, iters):
    nouns = pd.read_csv(ROOT / f"results/gvalnorm_{lang}/gvalnorm_nouns.csv")
    rows, ver, info = [], [], {}
    for n in iters:
        z = np.load(ROOT / f"results/gg_dump_{lang}/vectors_iter{n}.npz")
        train = set(z["train_words"])
        df = pd.DataFrame({"lemma": z["words"],
                           "s_before": sc(z["before"], z["man_before"], z["wom_before"]),
                           "s_after": sc(z["after"], z["man_after"], z["wom_after"])})  # fmt: skip
        d = nouns.merge(df, on="lemma")
        info[n] = {"in_cc_vocab": len(d), "dropped_in_training_pairs": int(d["lemma"].isin(train).sum()),
                   "final_heldout_acc": float(z["final_acc"]), "attributes": [list(z["man"]), list(z["wom"])]}  # fmt: skip
        d = d[~d["lemma"].isin(train)]
        for vocab in ("50k", "200k"):
            dv = d[d["in_50k"]] if vocab == "50k" else d
            fb, fa = fit(dv, "s_before"), fit(dv, "s_after")
            rows += [{"iters": n, "vocab": vocab, "space": "before", "n": len(dv), **fb},
                     {"iters": n, "vocab": vocab, "space": "after", "n": len(dv), **fa}]  # fmt: skip
            if vocab == "50k":
                ver.append({"iters": n, "beta_H_before": fb["beta_H"], "beta_H_after": fa["beta_H"],
                            "tracks_human_ratings_before": fb["beta_H_lo"] > 0,
                            "beta_H_verdict": verdict(fa, fb, "beta_H"),
                            "beta_G_before": fb["beta_G"], "beta_G_after": fa["beta_G"],
                            "beta_G_verdict": verdict(fa, fb, "beta_G")})  # fmt: skip
    out = ROOT / f"results/gvalnorm_{lang}"
    pd.DataFrame(rows).to_csv(out / "gvalnorm_omrani_fits.csv", index=False)
    pd.DataFrame(ver).to_csv(out / "gvalnorm_omrani_verdicts.csv", index=False)
    json.dump(info, open(out / "gvalnorm_omrani_info.json", "w"), indent=1, ensure_ascii=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(json.dumps(info, indent=1, ensure_ascii=False))
        print(pd.DataFrame(rows)[["iters", "vocab", "space", "n", "beta_G", "beta_G_lo", "beta_G_hi",
                                  "beta_H", "beta_H_lo", "beta_H_hi", "r_gend"]].round(3).to_string(index=False))  # fmt: skip
        print(pd.DataFrame(ver).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], [int(a) for a in sys.argv[2:]])
