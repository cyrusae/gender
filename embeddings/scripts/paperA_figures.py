"""Figures F2–F4 for paper A (replication report on Zhou et al. 2019). Descriptive plots of results
already computed and recorded elsewhere, plus the one small recomputation each figure needs (2026-11-19).
Reference for what each figure is for: drafts/paperA-tables-figures.md (local) and
results/paperA_figures/README.md.

F2  MWEAT permutation null (Z10): Original and Shift Ori, ES and FR, Zhou's published attribute lists
    and the notebook's `a + b/2` centring (i.e. the setting that produced Table 1). 10,000 random
    re-partitions of the occupation words (seed 0), the statistic per partition, and the observed value.
F3  "Same choices, two measures" (Z14, Z19, Z33): Shift Ori MWEAT under the four combinations of
    centring (notebook / midpoint) × attribute lists (published / corrected), beside the change in
    intrusion under Shift for the two centrings (from results/wordpair_cues/ and
    results/wordpair_cues_midpoint/; intrusion uses no attribute words).
F4  Mitigation measured with feminine cues: intrusion in the original, Shift Ori and Hybrid Ori spaces,
    ES and FR, with 95% intervals, for the released code's centring (results/wordpair_cues/) and the
    centring the paper describes (results/wordpair_cues_midpoint/).

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn,matplotlib python embeddings/scripts/paperA_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import c1_run  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import zhou_shift  # noqa: E402
from zhou_shift import Z, load_vec, mweat, shift  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paperA_figures"
N_PERM = 10_000

ATTR = {  # (male, female)
    ("es", "published"): (["hombre", "niño", "padre", "masculino", "abuelo", "esposo", "hijo", "tio"],
                          ["niña", "femenina", "tía", "hija", "esposa", "mujer", "madre", "abuela"]),
    ("es", "corrected"): (["hombre", "niño", "padre", "masculino", "abuelo", "esposo", "hijo", "tío"],
                          ["niña", "femenina", "tía", "hija", "esposa", "mujer", "madre", "abuela"]),
    ("fr", "corrected"): (["homme", "mari", "garçon", "fils", "il", "père", "mâle", "frère", "oncle"],
                          ["femme", "fille", "elle", "mère", "femelle", "sœur", "tante"]),
}  # fmt: skip


def fr_published():
    d = json.load(open(Z / "FR_definitional_pairs.json"))
    return [b for _, b in d], [a for a, _ in d]


def setup(lang):
    E, w2i = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt")
    occ = [line.rstrip("\n").split("\t") for line in open(Z / f"{lang.upper()}_occupation_words_with_EN_translations")]
    pairs = [(m, f) for f, m, _ in occ if m in w2i and f in w2i]
    return E, w2i, pairs


def direction(E, w2i, lang, midpoint):
    zhou_shift.CENTRE_MIDPOINT = midpoint
    ds, _ = c1_run.directions(E, w2i, lang)
    zhou_shift.CENTRE_MIDPOINT = False
    return ds


def perm_null(X, w2i, pairs, am, af, rng):
    a = np.mean([X[w2i[w]] for w in am], 0) - np.mean([X[w2i[w]] for w in af], 0)
    words = [w for p in pairs for w in p]
    s = np.array([X[w2i[w]] @ a for w in words])
    is_m = np.zeros(len(words), bool)
    is_m[0::2] = True
    obs = abs(abs(s[is_m].sum()) - abs(s[~is_m].sum()))
    tot = s.sum()
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        x = s[rng.choice(len(s), len(pairs), replace=False)].sum()
        null[i] = abs(abs(x) - abs(tot - x))
    return obs, null


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows2, rows3 = [], []
    fig2, ax2 = plt.subplots(2, 2, figsize=(9, 6))
    for li, lang in enumerate(("es", "fr")):
        E, w2i, pairs = setup(lang)
        pub = ATTR[("es", "published")] if lang == "es" else fr_published()
        cor = ATTR[(lang, "corrected")]
        dirs = {"notebook": direction(E, w2i, lang, False), "midpoint": direction(E, w2i, lang, True)}
        S = {k: shift(E, w2i, pairs, d) for k, d in dirs.items()}
        # F2
        for si, (sname, X) in enumerate((("Original", E), ("Shift Ori", S["notebook"]))):
            obs, null = perm_null(X, w2i, pairs, *pub, rng)
            ties = np.mean(np.abs(null - obs) < 1e-6)
            rows2.append({"lang": lang, "space": sname, "observed": obs, "share_greater": np.mean(null > obs + 1e-6),
                          "share_tied": ties, "null_max": null.max()})  # fmt: skip
            a = ax2[li, si]
            a.hist(null, bins=60, color="0.6")
            a.axvline(obs, color="crimson", lw=2)
            a.set_title(f"{lang.upper()} {sname}: {ties:.0%} of permutations tie, {np.mean(null > obs + 1e-6):.0%} exceed")
            a.set_xlabel("MWEAT under random re-partition")
        # F3
        for c, X in S.items():
            for an, (am, af) in (("published", pub), ("corrected", cor)):
                rows3.append({"lang": lang, "centring": c, "attributes": an, "shift_ori_mweat": mweat(X, w2i, pairs, am=am, af=af),
                              "original_mweat": mweat(E, w2i, pairs, am=am, af=af)})  # fmt: skip
    fig2.tight_layout()
    fig2.savefig(OUT / "F2_permutation_null.png", dpi=150)
    pd.DataFrame(rows2).to_csv(OUT / "F2_permutation_null.csv", index=False)
    d3 = pd.DataFrame(rows3)
    d3.to_csv(OUT / "F3_shift_mweat_by_choice.csv", index=False)

    def gaps(path):
        g = pd.read_csv(path)
        return g[(g["set"] == "primary") & (g.measure == "intrusion")].set_index(["lang", "space"])

    G = {"notebook": gaps(ROOT / "results/wordpair_cues/gaps.csv"), "midpoint": gaps(ROOT / "results/wordpair_cues_midpoint/gaps.csv")}
    fig3, ax3 = plt.subplots(1, 2, figsize=(10, 4))
    for li, lang in enumerate(("es", "fr")):
        sub = d3[d3.lang == lang]
        lab = [f"{r.centring}\n{r.attributes}" for r in sub.itertuples()]
        ax3[0].bar(np.arange(4) + 5 * li, sub.shift_ori_mweat, color="steelblue" if lang == "es" else "darkorange", label=lang.upper())
        for i, t in enumerate(lab):
            ax3[0].text(i + 5 * li, -0.02, t, ha="center", va="top", fontsize=6, rotation=90)
        for ci, c in enumerate(("notebook", "midpoint")):
            r = G[c].loc[(lang, "shift_ori-original")]
            ax3[1].errorbar(ci + 3 * li, r.value, yerr=[[r.value - r.lo], [r.hi - r.value]], fmt="o",
                            color="steelblue" if lang == "es" else "darkorange")  # fmt: skip
            ax3[1].text(ci + 3 * li, 0.005, f"{lang.upper()}\n{c}", ha="center", fontsize=7)
    ax3[0].set_title("Shift Ori MWEAT (Table 1 statistic)\nby centring × attribute list")
    ax3[0].set_xticks([])
    ax3[0].legend()
    ax3[1].axhline(0, color="k", lw=0.5)
    ax3[1].set_xticks([])
    ax3[1].set_title("Change in intrusion under Shift Ori\n(feminine cues; uses no attribute words)")
    fig3.tight_layout()
    fig3.savefig(OUT / "F3_same_choices_two_measures.png", dpi=150)
    # F4: both centrings side by side (released code = notebook a + b/2; as described = midpoint)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    rows4 = []
    for li, lang in enumerate(("es", "fr")):
        for si, sp in enumerate(("original", "shift_ori", "hybrid_ori")):
            for ci, (c, mk) in enumerate((("notebook", "o"), ("midpoint", "s"))):
                if sp == "original" and c == "midpoint":
                    continue  # the original space doesn't depend on the centring
                r = G[c].loc[(lang, sp)]
                x = si * 1.0 + (0.18 * (ci * 2 - 1) if sp != "original" else 0) + 4 * li
                ax4.errorbar(x, r.value, yerr=[[r.value - r.lo], [r.hi - r.value]], fmt=mk,
                             color=["0.2", "steelblue", "seagreen"][si], mfc="white" if c == "midpoint" else None)  # fmt: skip
                rows4.append({"lang": lang, "space": sp, "centring": c, "value": r.value, "lo": r.lo, "hi": r.hi, "verdict": r.verdict})
            ax4.text(si + 4 * li, -0.15, f"{lang.upper()}\n{sp.replace('_ori', '')}", ha="center", fontsize=7)
    ax4.axhline(0, color="k", lw=0.5)
    ax4.set_xticks([])
    ax4.set_ylim(-0.22, 0.5)
    ax4.set_ylabel("intrusion (RR masc − RR fem, feminine cues)")
    ax4.set_title("Masculine intrusion before and after mitigation\nfilled: released code (a + b/2 centring); open: as described (midpoint centring)")
    fig4.tight_layout()
    fig4.savefig(OUT / "F4_mitigation_intrusion.png", dpi=150)
    pd.DataFrame(rows4).to_csv(OUT / "F4_mitigation_intrusion.csv", index=False)
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows2).round(4).to_string(index=False))
        print(d3.round(4).to_string(index=False))
    print("=== PAPERA FIGS done")


if __name__ == "__main__":
    main()
