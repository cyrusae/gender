"""Which of the three pieces of the cross-fitted cosine is unstable on Gemma? (outcome-blind)

German training nouns, cos(f−m, (f+m)/2 − n) as in outlier_fix_diagnostics.py: for the overlap
term and both squared lengths, the mean over 200 split-halves and its standard error
(SD / sqrt(200)); snr = mean / SE. Methods raw and B (drop dimensions with >10% of the variance).

    uv run python scripts/outlier_fix_components.py
"""

import numpy as np
import outlier_fix_diagnostics as od
import pandas as pd

rows = []
for m, (actdir, layers) in od.MODELS.items():
    X, g, cells = od.nouns(actdir, m, "de")
    for L in layers:
        x0 = od.resid(X[:, L, :].astype(np.float64), cells)
        for meth in ("raw", "B"):
            x = od.transform(x0, meth)
            rng = np.random.default_rng(0)
            parts = []
            for _ in range(od.N_SPLIT):
                a = np.zeros(len(g), bool)
                for lab in np.unique(g):
                    i = np.flatnonzero(g == lab)
                    a[rng.choice(i, len(i) // 2, replace=False)] = True
                b = ~a
                mu = {(h, lab): x[hh & (g == lab)].mean(0) for h, hh in (("a", a), ("b", b))
                      for lab in "mfn"}  # fmt: skip
                va, vb = mu["a", "f"] - mu["a", "m"], mu["b", "f"] - mu["b", "m"]
                na = (mu["a", "f"] + mu["a", "m"]) / 2 - mu["a", "n"]
                nb = (mu["b", "f"] + mu["b", "m"]) / 2 - mu["b", "n"]
                plug = np.linalg.norm(va + vb) ** 2 / 4, np.linalg.norm(na + nb) ** 2 / 4
                parts.append(((va @ nb + vb @ na) / 2, va @ vb, na @ nb, *plug))
            p = np.array(parts)
            mean, se = p.mean(0), p.std(0) / np.sqrt(len(p))
            rows.append({"model": m.split("/")[1], "layer": L, "method": meth,
                         "n_m_f_n": "/".join(str((g == k).sum()) for k in "mfn"),
                         "cross": mean[0], "len2_fm": mean[1], "len2_n": mean[2],
                         "snr_cross": mean[0] / se[0], "snr_len2_fm": mean[1] / se[1],
                         "snr_len2_n": mean[2] / se[2],
                         "fm_true_share_of_plugin": mean[1] / mean[3],
                         "n_true_share_of_plugin": mean[2] / mean[4],
                         "cos": mean[0] / np.sqrt(mean[1] * mean[2]) if min(mean[1:3]) > 0 else np.nan})  # fmt: skip
d = pd.DataFrame(rows)
d.to_csv("results/outlier_fix/components.csv", index=False)
cols = ["n_m_f_n", "snr_cross", "snr_len2_fm", "snr_len2_n", "fm_true_share_of_plugin",
        "n_true_share_of_plugin", "cos"]  # fmt: skip
print(d.set_index(["model", "layer", "method"])[cols].round(2).to_string())
