"""Crime-noun probe (docs/crime-noun-probe-prereg.md, ADOPTED 2026-11-10). Committed before any score.

s(w) = mean_{x in male attr} w·x − mean_{y in female attr} w·y (raw dot products, as the
zero-point study). c = strict-epicene zero point from data/zero_ref_{lang}_frozen_v1.csv,
restricted to the first 50k words exactly as in zhou_zero_point.py (so c is unchanged).
  L     = mean s(crime) − c                      (> 0: toward the male attributes)
  K     = mean s(crime) − mean s(common reference set)
  L_act = mean s(act) − c;  D_act = L_act − L
Crime words up to the first 200k MUSE words (primary). Joint bootstrap (10,000, seed 0) over crime
words, act words, epicene reference words (within gender), common words and attribute words.
Robustness: leave-one-out over crime words; inanimate anchor; 50k-only crime words; without the
sex-coded words (capo, gangster).

    uv run --no-project --python 3.12 --with numpy,pandas python embeddings/scripts/crime_probe.py [es|fr]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from c1_run import EN_F, EN_M
from zhou_shift import attributes, Z

ROOT = Path(__file__).resolve().parents[1]
FLOOR, N_BOOT = 8, 10_000


def load(path, nmax):
    vecs, w2i = [], {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for i, line in enumerate(f):
            if i == nmax:
                break
            w, v = line.rstrip().split(" ", 1)
            if w not in w2i:  # keep the first (most frequent) occurrence
                w2i[w] = len(vecs)
                vecs.append(np.fromstring(v, sep=" "))
    return np.vstack(vecs), w2i


def verdict(lo, hi, pos="male-leaning", neg="female-leaning"):
    return pos if lo > 0 else neg if hi < 0 else "inconclusive"


class Probe:
    def __init__(self, E, w2i, AM, AF, sets):
        AM, AF = np.array(AM), np.array(AF)
        self.P = {k: (np.array([E[w2i[w]] for w in v["lemma"]]) @ AM.T,
                      np.array([E[w2i[w]] for w in v["lemma"]]) @ AF.T, v["gender"].to_numpy())
                  for k, v in sets.items()}  # fmt: skip
        self.nm, self.nf = AM.shape[0], AF.shape[0]

    def stats(self, am, af, idx):
        s = {k: Pm[idx[k]][:, am].mean(1) - Pf[idx[k]][:, af].mean(1) for k, (Pm, Pf, _) in self.P.items()}
        out = {}
        for anc in ("epicene", "inanimate"):
            g = self.P[anc][2][idx[anc]]
            c = 0.5 * (s[anc][g == "m"].mean() + s[anc][g == "f"].mean())
            out[f"c_{anc}"] = c
            out[f"L_{anc}"] = s["crime"].mean() - c
            if "act" in s:
                out[f"Lact_{anc}"] = s["act"].mean() - c
                out[f"Dact_{anc}"] = s["act"].mean() - s["crime"].mean()
        out["K"] = s["crime"].mean() - s["common"].mean()
        return out

    def full(self):
        return {k: np.arange(len(v[2])) for k, v in self.P.items()}

    def boot(self, n, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            idx = {}
            for k, (_, _, g) in self.P.items():
                if k in ("epicene", "inanimate"):
                    idx[k] = np.concatenate([rng.choice(np.flatnonzero(g == x), (g == x).sum()) for x in "mf"])
                else:
                    idx[k] = rng.integers(0, len(g), len(g))
            rows.append(self.stats(rng.integers(0, self.nm, self.nm), rng.integers(0, self.nf, self.nf), idx))
        return pd.DataFrame(rows)


def run(E, w2i, AM, AF, sets, tag, n=N_BOOT):
    pr = Probe(E, w2i, AM, AF, sets)
    pt, B = pr.stats(slice(None), slice(None), pr.full()), pr.boot(n)
    rows = []
    for k, v in pt.items():
        lo, hi = np.percentile(B[k], [2.5, 97.5])
        rows.append({**tag, "measure": k, "value": v, "lo": lo, "hi": hi,
                     "verdict": verdict(lo, hi) if k.startswith(("L", "Lact")) else
                     verdict(lo, hi, "more male than other common nouns", "less male") if k == "K" else
                     verdict(lo, hi, "act more male", "act less male") if k.startswith("Dact") else ""})  # fmt: skip
    return rows, pr, pt, B


def main(lang):
    out = ROOT / f"results/crime_probe_{lang}"
    out.mkdir(parents=True, exist_ok=True)
    E, w2i = load(Z / f"data/wiki.multi.{lang}.vec.txt", 200_000)
    en, en_w2i = load(Z / "data/wiki.multi.en.vec.txt", 200_000)
    defs = json.load(open(Z / f"{lang.upper()}_definitional_pairs.json"))
    am, af = attributes(lang, defs)
    occ = {w for line in open(Z / f"{lang.upper()}_occupation_words_with_EN_translations")
           for w in line.rstrip("\n").split("\t")[:2]}  # fmt: skip
    ref = pd.read_csv(ROOT / f"data/zero_ref_{lang}_frozen_v1.csv")
    ref = ref[ref["lemma"].map(lambda w: w2i.get(w, 10**9) < 50_000) & ~ref["lemma"].isin(occ | set(am) | set(af))]
    crime_all = pd.read_csv(ROOT / f"data/crime_{lang}_frozen_v1.csv")
    crime_all = crime_all[crime_all["lemma"].isin(w2i)]
    crime = crime_all[crime_all["set"] == "crime"]
    act = crime_all[crime_all["set"] == "act"]
    base = {"epicene": ref[ref["set"].isin(["epicene"]) & (ref["tier"] == "strict")],
            "inanimate": ref[ref["set"] == "inanimate"], "common": ref[ref["set"] == "common"]}  # fmt: skip
    info = {"n_crime": len(crime), "n_act": len(act), "crime_words": crime["lemma"].tolist(),
            "act_words": act["lemma"].tolist(), "n_epicene": base["epicene"]["gender"].value_counts().to_dict(),
            "floor_met": len(crime) >= FLOOR}  # fmt: skip
    yard = {"native": ([E[w2i[w]] for w in am], [E[w2i[w]] for w in af]),
            "english": ([en[en_w2i[w]] for w in EN_M], [en[en_w2i[w]] for w in EN_F])}  # fmt: skip
    rows, loo = [], []
    for y, (AM, AF) in yard.items():
        sets = {**base, "crime": crime, **({"act": act} if len(act) else {})}
        r, pr, pt, B = run(E, w2i, AM, AF, sets, {"lang": lang, "yardstick": y, "check": "main"})
        rows += r
        lo, hi = np.percentile(B["L_epicene"], [2.5, 97.5])
        for j, w in enumerate(crime["lemma"]):
            idx = pr.full()
            idx["crime"] = np.delete(idx["crime"], j)
            a = pr.stats(slice(None), slice(None), idx)["L_epicene"]
            d = a - pt["L_epicene"]
            loo.append({"yardstick": y, "dropped": w, "L_epicene": a, "verdict": verdict(lo + d, hi + d)})
        for chk, sub in (("50k_only", crime[crime["in_50k"]]), ("no_sex_coded", crime[~crime["sens_sex_coded"]])):
            if len(sub) >= FLOOR:
                rows += run(E, w2i, AM, AF, {**base, "crime": sub}, {"lang": lang, "yardstick": y, "check": chk}, 2000)[0]
            else:
                rows.append({"lang": lang, "yardstick": y, "check": chk, "measure": "L_epicene",
                             "verdict": f"too few ({len(sub)} < {FLOOR})"})  # fmt: skip
    res, loo = pd.DataFrame(rows), pd.DataFrame(loo)
    res.to_csv(out / "results.csv", index=False)
    loo.to_csv(out / "leave_one_out.csv", index=False)
    json.dump(info, open(out / "info.json", "w"), indent=1, ensure_ascii=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(json.dumps(info, ensure_ascii=False))
        print(res.round(4).to_string(index=False))
        for y, g in loo.groupby("yardstick"):
            main_v = res.query("yardstick==@y and check=='main' and measure=='L_epicene'").verdict.iloc[0]
            print(y, "LOO flips:", sorted(g.loc[g.verdict != main_v, "dropped"]))
    print("=== CRIME done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "es")
