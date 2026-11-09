"""Gender ValNorm Addendum B (and C with --fixed-en): the ds-orthogonal yardstick (docs/gvalnorm-prereg.md, frozen 63c5812).

Rebuilds C1's MUSE arms exactly as c1_run.py does, then scores each Glasgow-matched noun with
s⊥(w) = unit(w) · a⊥, a⊥ = a − (a·ds)ds, a = mean(unit male attr) − mean(unit female attr),
ds = the arm's own orthogonalised semantic direction. Same model and reading as gvalnorm_run.py.

    uv run --no-project --python 3.12 --with numpy,scikit-learn,pandas,scipy python embeddings/scripts/gvalnorm_orth.py [es|fr]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from c1_run import EN_F, EN_M, directions, project_out, unit
from gvalnorm_run import fit, verdict
from zhou_hybrid import DICT, hard_debias, procrustes
from zhou_shift import WEAT_F, WEAT_M, Z, load_vec, nouns, shift

ROOT = Path(__file__).resolve().parents[1]


def main(lang="es"):
    out = ROOT / f"results/gvalnorm_{lang}"
    gn = pd.read_csv(out / "gvalnorm_nouns.csv")
    if lang == "es":
        att_m, att_f = WEAT_M, WEAT_F
    else:
        fdefs = json.load(open(Z / "FR_definitional_pairs.json"))
        att_m, att_f = [b for _, b in fdefs], [a for a, _ in fdefs]
    es, w2i = load_vec(Z / f"data/wiki.multi.{lang}.vec.txt", nmax=200000)
    en, en_w2i = load_vec(Z / "data/wiki.multi.en.vec.txt", nmax=200000)
    w2i50 = {w: i for w, i in w2i.items() if i < 50000}
    en_db, _ = hard_debias(en, en_w2i)
    seen, src, tgt = set(), [], []
    for line in open(Path(str(DICT).replace("es-en", f"{lang}-en")), encoding="utf-8"):
        a, b = line.split()
        if a in w2i50 and b in en_w2i and a not in seen:
            seen.add(a)
            src.append(w2i[a])
            tgt.append(en_w2i[b])
    o = np.argsort(src)[:5000]
    es_da = es @ procrustes(es[np.array(src)[o]], en_db[np.array(tgt)[o]]).T
    occ = [line.rstrip("\n").split("\t") for line in open(Z / f"{lang.upper()}_occupation_words_with_EN_translations")]
    pairs = [(m, f) for f, m, _ in occ]
    zhou_rows = [w2i[w] for w in set(nouns(f"masculine_nouns_{lang}", w2i50) + nouns(f"feminine_nouns_{lang}", w2i50))]
    test_rows = gn["lemma"].map(w2i).to_numpy()

    spaces = {}
    for base, E, EN in [("shift", es, en), ("hybrid", es_da, en_db)]:
        ds, _ = directions(E, w2i50, lang)
        impl = project_out(shift(E, w2i, pairs, ds), zhou_rows, ds)
        spec = project_out(impl, test_rows, ds)
        names = (["1_original", "2_shift_ori_as_implemented", "3_shift_ori_as_specified"] if base == "shift"
                 else [None, "4a_hybrid_ori_as_implemented", "4b_hybrid_ori_as_specified"])  # fmt: skip
        for name, S in zip(names, (E, impl, spec)):
            if name:
                spaces[name] = (S, EN, ds)

    fixed = "--fixed-en" in sys.argv  # Addendum C: Hybrid arms scored with ORIGINAL English attributes
    if fixed:
        spaces = {k: (S, en, ds) for k, (S, _, ds) in spaces.items() if k.startswith(("1_", "4"))}
    rows, desc = [], []
    for arm, (E, EN, ds) in spaces.items():
        for att, (M, F) in {lang: (E[[w2i[w] for w in att_m]], E[[w2i[w] for w in att_f]]),
                            "en": (EN[[en_w2i[w] for w in EN_M]], EN[[en_w2i[w] for w in EN_F]])}.items():  # fmt: skip
            a = unit(M).mean(0) - unit(F).mean(0)
            a_perp = a - (a @ ds) * ds
            gn[f"{arm}|{att}|perp"] = unit(E[test_rows]) @ a_perp
            gn[f"{arm}|{att}|full"] = unit(E[test_rows]) @ a
            desc.append({"arm": arm, "attributes": att, "cos_a_ds": float(a @ ds / np.linalg.norm(a))})
            for vocab in ("50k", "200k"):
                dv = gn[gn["in_50k"]] if vocab == "50k" else gn
                main_col = f"{arm}|{att}|full" if fixed else f"{arm}|{att}"
                fm = fit(dv, main_col)
                rows.append({"vocab": vocab, "arm": arm, "attributes": att, **fit(dv, f"{arm}|{att}|perp"),
                             "beta_H_main": fm["beta_H"], "beta_H_main_lo": fm["beta_H_lo"], "beta_H_main_hi": fm["beta_H_hi"]})  # fmt: skip
    res = pd.DataFrame(rows)
    res["share_beta_H_off_axis"] = res["beta_H"] / res["beta_H_main"]
    tag = "fixed_en" if fixed else "orth"
    res.to_csv(out / f"gvalnorm_{tag}_fits.csv", index=False)
    pd.DataFrame(desc).to_csv(out / f"gvalnorm_{tag}_cos.csv", index=False)
    ver = []
    for att in res["attributes"].unique():
        p = res.query("vocab=='50k' and attributes==@att").set_index("arm")
        base = p.loc["1_original"]
        for arm in p.index:
            ver.append({"attributes": att, "arm": arm, "beta_H": p.loc[arm].beta_H,
                        "beta_H_lo": p.loc[arm].beta_H_lo, "beta_H_hi": p.loc[arm].beta_H_hi,
                        "tracks_off_axis": bool(p.loc[arm].beta_H_lo > 0) if arm == "1_original" else None,
                        "beta_H_verdict": None if arm == "1_original" else verdict(p.loc[arm], base, "beta_H"),
                        "share_off_axis": p.loc[arm].share_beta_H_off_axis,
                        "beta_H_full": p.loc[arm].beta_H_main, "beta_H_full_lo": p.loc[arm].beta_H_main_lo,
                        "beta_H_full_hi": p.loc[arm].beta_H_main_hi,
                        "full_verdict": None if arm == "1_original" else verdict(
                            {"beta_H": p.loc[arm].beta_H_main, "beta_H_lo": p.loc[arm].beta_H_main_lo,
                             "beta_H_hi": p.loc[arm].beta_H_main_hi}, {"beta_H": base.beta_H_main}, "beta_H")})  # fmt: skip
    pd.DataFrame(ver).to_csv(out / f"gvalnorm_{tag}_verdicts.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(pd.DataFrame(desc).round(3).to_string(index=False))
        print(pd.DataFrame(ver).round(3).to_string(index=False))
    print("=== GVALNORM ORTH done")


if __name__ == "__main__":
    main(next((a for a in sys.argv[1:] if not a.startswith("--")), "es"))
