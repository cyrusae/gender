"""Masculine lead by cue type, before and after mitigation (exploratory, 2026-11-23, PI request).

Same queries, cues, occupations and scoring as the cue study (wordpair_cues.py): for each English cue +
occupation, RR of the masculine and of the feminine Spanish/French occupation form (0 outside the top 999).
Three cue types: masculine cue of each gendered pair (only the masculine form agrees), invariant cue
(either agrees), feminine cue (only the feminine form agrees; its gap is the cue study's "intrusion").
Per cue type and space:
  gap   = mean (RR_m − RR_f)                 (0 = no preference)
  share = mean RR_m / (mean RR_m + mean RR_f) (50% = parity)
Per-cue values (gap averaged over occupations) are saved for the dot plot. Bootstrap 10,000 (seed 0):
gendered pairs (masculine and feminine cue together), invariant cues and occupations, as in the cue study.
Run per centring: ZHOU_CENTRE=notebook (released code, default) or midpoint.

    ZHOU_CENTRE=midpoint uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn,matplotlib python embeddings/scripts/wordpair_cue_gradient.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from figpanels import save_panels  # noqa: E402
from wordpair_mechanism import rr_table  # noqa: E402
from zhou_shift import Z, load_vec  # noqa: E402
from zhou_wordpair import spaces  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/wordpair_cue_gradient"
N_BOOT, SEED = 10_000, 0
CENTRE = os.environ.get("ZHOU_CENTRE", "notebook")
TYPES = ("masculine", "invariant", "feminine")
SPACE_LAB = {"original": "Original", "shift_ori": "Shift", "hybrid_ori": "Hybrid"}
COL = {"original": "#333333", "shift_ori": "#3b7bbf", "hybrid_ori": "#2e8b57"}


def summarise(R, pi, ii, oi):
    out = {}
    for t, X in (("masculine", R["m"][np.ix_(pi, oi)]), ("invariant", R["inv"][np.ix_(ii, oi)]),
                 ("feminine", R["f"][np.ix_(pi, oi)])):  # fmt: skip
        a, b = X[..., 0].mean(), X[..., 1].mean()
        out[(t, "gap")], out[(t, "share")] = a - b, a / (a + b)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cues = pd.read_csv(ROOT / "data/wp_cues_frozen_v1.csv", keep_default_na=False, dtype=str)
    mp = pd.read_csv(ROOT / "data/occupation_mapping_v1.csv")
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    rng = np.random.default_rng(SEED)
    rows, per_cue = [], []
    for lang in ("es", "fr"):
        sp, w2i, occ_all = spaces(lang, en, en_w2i)
        i2w = {i: w for w, i in w2i.items()}
        m = mp[mp.lang == lang].reset_index(drop=True)
        keep = m.clean_keep.to_numpy(bool) & np.array([e in en_w2i for _, _, e in occ_all])
        occ = [o for o, k in zip(occ_all, keep) if k]
        cs = cues[(cues.lang == lang) & (cues.subset == "primary")]
        G, Inv = cs[cs.type == "gendered"].reset_index(drop=True), cs[cs.type == "invariant"].reset_index(drop=True)
        qs = {"m": [(e, x, "m") for e, x in zip(G.en, G.masc)], "f": [(e, x, "f") for e, x in zip(G.en, G.fem)],
              "inv": [(e, x, "inv") for e, x in zip(Inv.en, Inv.masc)]}  # fmt: skip
        tabs = {s: {k: rr_table(EN, T, en_w2i, w2i, i2w, q, occ) for k, q in qs.items()} for s, (EN, _, T) in sp.items()}
        n_p, n_i, n_o = len(G), len(Inv), len(occ)
        full = summarise(tabs["original"], np.arange(n_p), np.arange(n_i), np.arange(n_o))
        draws = {s: [] for s in tabs}
        for _ in range(N_BOOT):
            pi, ii, oi = rng.integers(0, n_p, n_p), rng.integers(0, n_i, n_i), rng.integers(0, n_o, n_o)
            for s, R in tabs.items():
                draws[s].append(summarise(R, pi, ii, oi))
        for s, R in tabs.items():
            pt = summarise(R, np.arange(n_p), np.arange(n_i), np.arange(n_o))
            for key, v in pt.items():
                lo, hi = np.percentile([d[key] for d in draws[s]], [2.5, 97.5])
                rows.append({"lang": lang, "centring": CENTRE, "space": s, "cue_type": key[0], "measure": key[1],
                             "value": v, "lo": lo, "hi": hi, "n_pairs": n_p, "n_invariant": n_i, "n_occ": n_o})  # fmt: skip
                if s != "original":  # paired change from Original
                    dl = [d[key] - o[key] for d, o in zip(draws[s], draws["original"])]
                    lo, hi = np.percentile(dl, [2.5, 97.5])
                    rows.append({"lang": lang, "centring": CENTRE, "space": f"{s}-original", "cue_type": key[0],
                                 "measure": key[1], "value": v - full[key], "lo": lo, "hi": hi,
                                 "n_pairs": n_p, "n_invariant": n_i, "n_occ": n_o})  # fmt: skip
            for k, t, words in (("m", "masculine", G.masc), ("inv", "invariant", Inv.masc), ("f", "feminine", G.fem)):
                g = (R[k][..., 0] - R[k][..., 1]).mean(1)
                per_cue += [{"lang": lang, "centring": CENTRE, "space": s, "cue_type": t, "cue": w, "gap": x}
                            for w, x in zip(words, g)]  # fmt: skip
        print("done", lang, flush=True)
    d, pc = pd.DataFrame(rows), pd.DataFrame(per_cue)
    d.to_csv(OUT / f"summary_{CENTRE}.csv", index=False)
    pc.to_csv(OUT / f"per_cue_{CENTRE}.csv", index=False)
    plot(d, pc)
    print(d[(d.measure == "gap")].round(3).to_string(index=False))
    print("=== WPGRAD done", CENTRE)


def plot(d, pc):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), sharey="row")
    rng = np.random.default_rng(1)
    for j, (lang, name) in enumerate((("es", "Spanish"), ("fr", "French"))):
        for i, meas in enumerate(("gap", "share")):
            a = ax[i, j]
            for t_i, t in enumerate(TYPES):
                for s_i, s in enumerate(SPACE_LAB):
                    x = t_i * 4 + s_i
                    if meas == "gap":
                        v = pc[(pc.lang == lang) & (pc.space == s) & (pc.cue_type == t)].gap
                        a.scatter(x + rng.uniform(-0.25, 0.25, len(v)), v, s=5, c=COL[s], alpha=0.25, lw=0)
                    r = d[(d.lang == lang) & (d.space == s) & (d.cue_type == t) & (d.measure == meas)].iloc[0]
                    a.errorbar(x, r.value, yerr=[[r.value - r.lo], [r.hi - r.value]], fmt="o", c=COL[s], ms=7,
                               capsize=3, label=SPACE_LAB[s] if t_i == 0 else None)  # fmt: skip
            a.axhline(0 if meas == "gap" else 0.5, c="black", lw=0.8)
            a.set_xticks([1, 5, 9], [f"{t} cue\n" + {"masculine": "(only masc. agrees)", "invariant": "(either agrees)",
                                                     "feminine": "(only fem. agrees)"}[t] for t in TYPES])  # fmt: skip
            letter = "ABCD"[i * 2 + j]
            a.set_title(f"{letter}. {name}: " + ("masculine lead (RR masc − RR fem)" if meas == "gap"
                                                  else "masculine share of retrieval"))  # fmt: skip
            a.set_ylabel("RR masc − RR fem (dots = cues)" if meas == "gap" else "RR masc / (RR masc + RR fem)")
            if meas == "share":
                a.set_ylim(0, 1)
            if letter == "A":
                a.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / f"cue_gradient_{CENTRE}.png", dpi=150)
    save_panels(fig, OUT, f"cue_gradient_{CENTRE}")


if __name__ == "__main__":
    main()
