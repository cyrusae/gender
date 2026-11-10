"""Offset-corrected asymmetry for Zhou et al.'s occupation pairs (docs/zhou-zero-point-prereg.md,
ADOPTED 2026-11-10; lists frozen ebd47b1). Committed before any score.

Per-word lean s(w) = mean_{x in male attr} w·x − mean_{y in female attr} w·y (raw dot products,
as Zhou et al.'s notebook). T_m, T_f = sums over the n occupation pairs' masculine / feminine
forms. Zero point c_R of a reference set: ½(mean m + mean f), or the plain mean for common-gender
nouns. A_R = |T_m − n c_R| − |T_f − n c_R| (> 0: masculine forms lean further from zero); raw =
|T_m| − |T_f|. Joint bootstrap over occupation pairs, reference words (within gender) and
attribute words. Spaces: original, Shift Ori (our rebuild). Yardsticks: native attributes
(primary), English attributes in the aligned MUSE space.

    uv run --no-project --python 3.12 --with numpy,pandas,scikit-learn python embeddings/scripts/zhou_zero_point.py [es|fr]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from c1_run import EN_F, EN_M, directions
from zhou_shift import WEAT_F, WEAT_M, Z, load_vec, shift

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 15
N_MAIN, N_SENS = 10_000, 2_000
SENS = {"S1": "s1_identical_to_gloss", "S2": "s2_en_more_frequent", "S3": "s3_pi_loan_flag",
        "S5": "s5_in_zhou_lists", "S6": "s6_pi_borderline"}  # fmt: skip


def anchors(ref):
    epi = ref["set"].isin(["epicene", "collective"])
    return {"epicene_strict": ref[epi & (ref["tier"] == "strict")],
            "epicene_broad": ref[epi],
            "inanimate": ref[ref["set"] == "inanimate"],
            "common": ref[ref["set"] == "common"]}  # fmt: skip


def verdict(a, lo, hi, raw):
    if lo > 0 or hi < 0:
        return "asymmetry: masculine further" if a > 0 else "asymmetry: feminine further"
    if abs(a) < 0.25 * abs(raw):
        return "offset explains raw statistic"
    return "inconclusive"


class Setup:
    """Dot products of every needed word with every attribute word, in one space."""

    def __init__(self, E, w2i, pairs, AM, AF, refs):
        self.n = len(pairs)
        self.Pm_m = np.array([[E[w2i[m]] @ x for x in AM] for m, _ in pairs])
        self.Pf_m = np.array([[E[w2i[m]] @ y for y in AF] for m, _ in pairs])
        self.Pm_f = np.array([[E[w2i[f]] @ x for x in AM] for _, f in pairs])
        self.Pf_f = np.array([[E[w2i[f]] @ y for y in AF] for _, f in pairs])
        self.ref = {}
        for name, r in refs.items():
            V = np.array([E[w2i[w]] for w in r["lemma"]])
            self.ref[name] = (V @ np.array(AM).T, V @ np.array(AF).T, r["gender"].to_numpy())

    def stats(self, am, af, pi, ridx):
        sm = self.Pm_m[pi][:, am].mean(1) - self.Pf_m[pi][:, af].mean(1)
        sf = self.Pm_f[pi][:, am].mean(1) - self.Pf_f[pi][:, af].mean(1)
        Tm, Tf = sm.sum(), sf.sum()
        out = {"raw": abs(Tm) - abs(Tf)}
        for name, (Rm, Rf, g) in self.ref.items():
            i = ridx[name]
            s = Rm[i][:, am].mean(1) - Rf[i][:, af].mean(1)
            gg = g[i]
            c = s.mean() if (gg == "multi").all() else 0.5 * (s[gg == "m"].mean() + s[gg == "f"].mean())
            out[f"{name}|c"] = c
            out[f"{name}|A"] = abs(Tm - self.n * c) - abs(Tf - self.n * c)
        return out

    def point(self, ridx=None):
        ridx = ridx or {k: np.arange(len(v[2])) for k, v in self.ref.items()}
        return self.stats(slice(None), slice(None), np.arange(self.n), ridx)

    def boot(self, n_draws, seed=0):
        rng = np.random.default_rng(seed)
        nm, nf = self.Pm_m.shape[1], self.Pf_m.shape[1]
        draws = []
        for _ in range(n_draws):
            ridx = {}
            for k, (_, _, g) in self.ref.items():
                ridx[k] = np.concatenate([rng.choice(np.flatnonzero(g == x), (g == x).sum())
                                          for x in np.unique(g)])  # fmt: skip
            draws.append(self.stats(rng.integers(0, nm, nm), rng.integers(0, nf, nf),
                                    rng.integers(0, self.n, self.n), ridx))  # fmt: skip
        return pd.DataFrame(draws)


def summarise(st, pt, B, tag):
    rows = []
    for k in st.ref:
        a, (lo, hi) = pt[f"{k}|A"], np.percentile(B[f"{k}|A"], [2.5, 97.5])
        clo, chi = np.percentile(B[f"{k}|c"], [2.5, 97.5])
        g = st.ref[k][2]
        rows.append({**tag, "anchor": k, "n_m": int((g == "m").sum()), "n_f": int((g == "f").sum()),
                     "n_multi": int((g == "multi").sum()), "c": pt[f"{k}|c"], "c_lo": clo, "c_hi": chi,
                     "raw": pt["raw"], "A": a, "A_lo": lo, "A_hi": hi, "verdict": verdict(a, lo, hi, pt["raw"])})  # fmt: skip
    return rows


def main(lang):
    out = ROOT / f"results/zhou_zero_point_{lang}"
    out.mkdir(parents=True, exist_ok=True)
    E, w2i = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt")
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    occ = [line.rstrip("\n").split("\t") for line in open(Z / f"{lang.upper()}_occupation_words_with_EN_translations")]
    pairs = [(m, f) for f, m, _ in occ]
    defs = json.load(open(Z / f"{lang.upper()}_definitional_pairs.json"))
    am, af = (WEAT_M, WEAT_F) if lang == "es" else ([b for _, b in defs], [a for a, _ in defs])
    ref = pd.read_csv(ROOT / f"data/zero_ref_{lang}_frozen_v1.csv")
    taken = {w for p in pairs for w in p} | set(am) | set(af)
    info = {"n_pairs": len(pairs), "listed": len(ref),
            "dropped_not_in_50k": sorted(ref.loc[~ref["lemma"].isin(w2i), "lemma"]),
            "dropped_overlap_occupation_or_attribute": sorted(ref.loc[ref["lemma"].isin(taken), "lemma"])}  # fmt: skip
    ref = ref[ref["lemma"].isin(w2i) & ~ref["lemma"].isin(taken)].reset_index(drop=True)
    epi = anchors(ref)["epicene_strict"]
    info["strict_epicenes_after_coverage"] = epi["gender"].value_counts().to_dict()
    info["floor_met"] = bool(min((epi["gender"] == "m").sum(), (epi["gender"] == "f").sum()) >= FLOOR)
    info["primary_anchor"] = "epicene_strict" if info["floor_met"] else "inanimate"
    ds, _ = directions(E, w2i, lang)
    spaces = {"original": E, "shift_ori": shift(E, w2i, pairs, ds)}
    yard = {"native": ([E[w2i[w]] for w in am], [E[w2i[w]] for w in af]),
            "english": ([en[en_w2i[w]] for w in EN_M], [en[en_w2i[w]] for w in EN_F])}  # fmt: skip
    summ, loo, sens = [], [], []
    for sp, S in spaces.items():
        for y, (AM, AF) in yard.items():
            tag = {"lang": lang, "space": sp, "yardstick": y}
            st = Setup(S, w2i, pairs, AM, AF, anchors(ref))
            pt, B = st.point(), st.boot(N_MAIN)
            rows = summarise(st, pt, B, tag)
            summ += rows
            # rule 0: leave-one-out (exact point; CI shifted by the point change)
            for r in rows:
                k = r["anchor"]
                g = st.ref[k][2]
                full = {kk: np.arange(len(v[2])) for kk, v in st.ref.items()}
                names = anchors(ref)[k]["lemma"].to_numpy()
                for j in range(len(g)):
                    ridx = dict(full)
                    ridx[k] = np.delete(np.arange(len(g)), j)
                    a = st.point(ridx)[f"{k}|A"]
                    d = a - r["A"]
                    loo.append({**tag, "anchor": k, "dropped": names[j], "A": a,
                                "verdict": verdict(a, r["A_lo"] + d, r["A_hi"] + d, r["raw"])})  # fmt: skip
            # S1-S3, S5, S6: drop flagged reference words, re-bootstrap
            for s, col in SENS.items():
                sub = ref[~ref[col].astype(bool)]
                st2 = Setup(S, w2i, pairs, AM, AF, anchors(sub))
                for r in summarise(st2, st2.point(), st2.boot(N_SENS), {**tag, "check": s}):
                    sens.append(r)
    summ, loo, sens = pd.DataFrame(summ), pd.DataFrame(loo), pd.DataFrame(sens)
    summ.to_csv(out / "summary.csv", index=False)
    loo.to_csv(out / "leave_one_out.csv", index=False)
    sens.to_csv(out / "sensitivity.csv", index=False)
    # reading: rule 1 (anchor agreement), rule 0 (LOO), sensitivity stability
    read = []
    for (sp, y), g in summ.groupby(["space", "yardstick"]):
        e, i = g.set_index("anchor").loc["epicene_strict"], g.set_index("anchor").loc["inanimate"]
        p = g.set_index("anchor").loc[info["primary_anchor"]]
        lv = loo[(loo.space == sp) & (loo.yardstick == y) & (loo.anchor == info["primary_anchor"])]
        flips = sorted(lv.loc[lv.verdict != p.verdict, "dropped"])
        sv = sens[(sens.space == sp) & (sens.yardstick == y) & (sens.anchor == info["primary_anchor"])]
        s_flip = sorted(sv.loc[sv.verdict != p.verdict, "check"])
        broad = g.set_index("anchor").loc["epicene_broad"]
        read.append({"space": sp, "yardstick": y, "anchors_agree": not (e.c_hi < i.c_lo or i.c_hi < e.c_lo),
                     "primary": info["primary_anchor"], "verdict_primary": p.verdict,
                     "A": p.A, "A_lo": p.A_lo, "A_hi": p.A_hi, "raw": p.raw,
                     "loo_depends_on": flips, "S1-S6_depends_on": s_flip,
                     "S4_broad_verdict": broad.verdict,
                     "label": "algebraically constrained (Z14)" if sp == "shift_ori" else ""})  # fmt: skip
    read = pd.DataFrame(read)
    read.to_csv(out / "reading.csv", index=False)
    json.dump(info, open(out / "info.json", "w"), indent=1, ensure_ascii=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60):
        print(json.dumps(info, indent=1, ensure_ascii=False))
        print(summ[["space", "yardstick", "anchor", "n_m", "n_f", "n_multi", "c", "c_lo", "c_hi", "raw",
                    "A", "A_lo", "A_hi", "verdict"]].round(4).to_string(index=False))  # fmt: skip
        print(read.round(4).to_string(index=False))
    print("=== ZEROPOINT done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "es")
