"""Phase 4: is grammatical gender shared across languages? (design: docs/design/phase4-design.md,
adopted Q1-Q6, 2026-10-08).

Directions (fitted per layer with the Phase 2/3 stratified estimators, f = positive):
  ES      Phase 2 stratified training set (`strat` + `matched`), cells = ending
  DE      Phase 3 `strat3`, masculine vs feminine, cells = ending x loan status
  POOLED  ES + DE nouns, cells = language x cell (centring within language)
Each as a probe (`probe_strat`, primary) and a stratified difference of means (`class_betas`,
what Phase 5 steers with).

Tests
  T1  transfer: ES scores German strat3 m/f nouns, DE scores Spanish strat nouns; AUC within the
      test language's cells (all within-cell m-f comparisons pooled), bootstrap resampling the
      training nouns (refit) and the test nouns, both within cell x gender.
  T2  geometry: cross-fitted cosine between the ES and DE class vectors (split halves within
      language x cell x gender), with a within-cell label-shuffle null.
  T3  flipped pairs: one direction scores both nouns of each pair; paired AUC = P(score(de) -
      score(es) is larger for a German-f/Spanish-m pair than for a German-m/Spanish-f pair).
      0.5 under the null (a meaning direction scores both nouns alike); a constant language
      offset cancels. Share of pairs in the predicted direction is reported too (it doesn't
      cancel an offset). Bootstrap over pairs, directions refitted on resampled training nouns.
  T4  Russian -ь nouns (4B and up: 1.7B dropped for Russian), AUC within the list's cells.
  Q5  German neuter (strat3 neuter nouns, never in ES/POOLED training): position between the
      masculine (0) and feminine (1) means on ES and POOLED, within cells (descriptive).

Decision rules (layer counts over inner layers, as Phases 2-3) are applied in `verdicts()`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import activations as acts
from .estimators import (
    auc,
    boot_within,
    class_betas,
    nuisance,
    probe_strat,
    shuffle_within,
    split_half_geometry,
)
from .models import git_state, model_slug, progress, stage

PAIRS_SRC = "data/lexicon/pairs_de_es.csv"
FLAGS = "docs/reports/flipped-pairs-flags.csv"
PAIRS_POOL = "data/stimuli/phase4_pairs_pool_v1.csv"
PAIRS_ES_KNOWN_INPUT = "data/stimuli/phase4_pairs_es_v1.csv"  # the Spanish nouns, phase0 format
PAIRS_FINAL = "data/stimuli/phase4_pairs_final_v1.csv"
RU_FINAL = "data/stimuli/phase4_ru_final_v1.csv"
P2_FINAL = "data/stimuli/phase2_final_v4.csv"
P3_FINAL = "data/stimuli/phase3_final_v3.csv"
KNOWN_MODELS = ["Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]  # as Phases 2-3
SEED = 0
N_BOOT = 1000
N_SPLITS = 200
N_SHUFFLE = 100
COGNATE_FLAG = "English cognate"


# ---- stimuli ------------------------------------------------------------------
def training_lemmas() -> tuple[set[str], set[str]]:
    """Every Spanish / German noun used to train any Phase 2 / Phase 3 direction."""
    p2 = pd.read_csv(P2_FINAL, keep_default_na=False)
    p3 = pd.read_csv(P3_FINAL, keep_default_na=False)
    return set(p2[p2.split == "train"].lemma), set(p3[p3.split == "train"].lemma)


def build_pairs() -> pd.DataFrame:
    """The 389 flipped pairs: every flagged pair dropped except English cognates (kept, flagged
    for the pre-registered sensitivity analysis). Asserts no noun is in Phase 2/3 training."""
    p = pd.read_csv(PAIRS_SRC, keep_default_na=False)
    p = p[p.flipped.astype(str) == "True"]
    f = pd.read_csv(FLAGS, keep_default_na=False)
    cog = f.flag.str.startswith(COGNATE_FLAG)
    drop = set(zip(f[~cog].de_lemma, f[~cog].es_lemma, strict=True))
    cognate = set(zip(f[cog].de_lemma, f[cog].es_lemma, strict=True))
    key = list(zip(p.de_lemma, p.es_lemma, strict=True))
    p = p[[k not in drop for k in key]].copy()
    p["en_cognate"] = [k in cognate for k in zip(p.de_lemma, p.es_lemma, strict=True)]
    es_tr, de_tr = training_lemmas()
    assert not set(p.es_lemma) & es_tr, "flipped-pair Spanish noun in Phase 2 training"
    assert not set(p.de_lemma) & de_tr, "flipped-pair German noun in Phase 3 training"
    out = p[["concept_en", "de_lemma", "de_gender", "es_lemma", "es_gender", "de_zipf", "es_zipf",
             "en_cognate"]].reset_index(drop=True)  # fmt: skip
    out.insert(0, "pair_id", range(len(out)))
    out["split"] = "test"
    out.to_csv(PAIRS_POOL, index=False)
    es = pd.DataFrame({"lang": "es", "lemma": out.es_lemma, "gender": out.es_gender,
                       "concept_en": out.concept_en, "set": "pair", "source": PAIRS_SRC,
                       "zipf": out.es_zipf})  # fmt: skip
    es.to_csv(PAIRS_ES_KNOWN_INPUT, index=False)
    print(f"{len(out)} pairs ({int(out.en_cognate.sum())} English cognates); "
          f"{out.groupby(['de_gender', 'es_gender']).size().to_dict()}")  # fmt: skip
    return out


def known(model_id: str, device=None, dtype=None, which: str = "both") -> None:
    """Known checks on both nouns of every pair: German three-way (Phase 3 frames), Spanish
    Phase 0 frames."""
    from . import phase0, phase3_known
    from .cli import _free_memory

    p = pd.read_csv(PAIRS_POOL, keep_default_na=False)
    de = pd.DataFrame({"lemma": p.de_lemma, "gender": p.de_gender})
    if which in ("both", "de"):
        phase3_known.score(model_id, device, dtype, out_root="results/phase4_known_de", items=de)
        _free_memory()
    if which in ("both", "es"):
        phase0.run(model_id, PAIRS_ES_KNOWN_INPUT, "results/phase4_known_es", device, dtype)


def known_sets(model_id: str) -> tuple[set[str], set[str]]:
    slug = model_slug(model_id)
    de = pd.read_csv(f"results/phase4_known_de/{slug}/scores.csv", keep_default_na=False)
    es = pd.read_csv(f"results/phase4_known_es/{slug}/items.csv", keep_default_na=False)
    return set(de[de.status == "known"].lemma), set(es[es.passed.astype(str) == "True"].lemma)


def finalize(models=KNOWN_MODELS) -> pd.DataFrame:
    """Pairs whose German and Spanish nouns are both known by every model in `models`."""
    p = pd.read_csv(PAIRS_POOL, keep_default_na=False)
    keep = np.ones(len(p), bool)
    for m in models:
        kde, kes = known_sets(m)
        keep &= p.de_lemma.isin(kde).to_numpy() & p.es_lemma.isin(kes).to_numpy()
    out = p[keep].reset_index(drop=True)
    out.to_csv(PAIRS_FINAL, index=False)
    print(f"{len(out)} of {len(p)} pairs known by {', '.join(models)}: "
          f"{out.groupby(['de_gender', 'es_gender']).size().to_dict()}, "
          f"{int(out.en_cognate.sum())} cognates")  # fmt: skip
    return out


# ---- extraction ---------------------------------------------------------------
def extract(model_id: str, device=None, dtype=None) -> None:
    """Pair nouns (both languages, all 389, so per-model known filters need no re-extraction)
    and the Russian list; LAST and AFTER from one pass, same input format as Phases 2-3."""
    from datetime import UTC, datetime

    from .models import load_model, pick_device, pick_dtype, run_metadata

    p = pd.read_csv(PAIRS_POOL, keep_default_na=False)
    texts = {"phase4_pairs_de": list(p.de_lemma), "phase4_pairs_es": list(p.es_lemma)}
    if Path(RU_FINAL).exists():
        texts["phase4_ru"] = list(pd.read_csv(RU_FINAL, keep_default_na=False).lemma)
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    for name, ws in texts.items():
        stage(f"{model_id}: extracting {name} ({len(ws)} words)")
        X_last, X_after, toks = acts.states_at(model, tok, ws, acts.AFTER, desc=name)
        for pos, X in (("last", X_last), ("after", X_after)):
            m = {**meta, "position": pos, "input_format": acts.INPUT_FORMAT
                 + (" (+ newline, read there)" if pos == "after" else "")}  # fmt: skip
            acts.save(model_id, name + ("_after" if pos == "after" else ""), X, ws, toks, m)


# ---- training data (as Phases 2-3) -------------------------------------------
def _sx(position: str) -> str:
    return {"last": "", "after": "_after"}[position]


def _conc(lang: str, lemmas, concepts) -> np.ndarray:
    from .lexicon import load_lexicon
    from .norms import rate

    gl = load_lexicon(lang).drop_duplicates("lemma").set_index("lemma").gloss
    r = [rate(gl.get(w, c))[0] for w, c in zip(lemmas, concepts, strict=True)]
    return np.array([np.nan if v is None else v for v in r], dtype=float)


def es_train(model_id: str, position: str) -> dict:
    """Phase 2 stratified training set with the Phase 2 covariates (incl. the initial-a
    amendment); rows of the phase2_bare activations."""
    df = pd.read_csv(P2_FINAL, keep_default_na=False)
    X, meta = acts.load(model_id, "phase2_bare" + _sx(position))
    assert list(meta["words"]) == list(df.lemma)
    ntok = np.array([len(t) for t in meta["tokens"]], dtype=float)
    conc = _conc("es", df.lemma, df.concept_en)
    miss = np.isnan(conc)
    conc[miss] = np.nanmean(conc[(df.set == "matched").to_numpy()])  # as Phase 2
    sel = df.set.isin(["strat", "matched"]).to_numpy()
    loan = df.loan.to_numpy()
    a_init = df.lemma.str.lower().str.match(r"^[aá]").to_numpy().astype(float)
    covs = [conc, miss.astype(float), pd.to_numeric(df.zipf).to_numpy(), ntok,
            (loan == "loan").astype(float), (loan == "unknown").astype(float), a_init]  # fmt: skip
    return {"X": X[sel], "y": (df.gender[sel] == "f").to_numpy(int),
            "cells": df.ending[sel].to_numpy(), "covs": [c[sel] for c in covs],
            "lemma": df.lemma[sel].to_numpy()}  # fmt: skip


def de_train(model_id: str, position: str, genders=("m", "f")) -> dict:
    """Phase 3 strat3 (default m/f) with the Phase 3 covariates; rows of phase3_bare."""
    df = pd.read_csv(P3_FINAL, keep_default_na=False)
    X, meta = acts.load(model_id, "phase3_bare" + _sx(position))
    row = {w: i for i, w in enumerate(meta["words"])}
    ntok_of = {w: len(t) for w, t in zip(meta["words"], meta["tokens"], strict=True)}
    s3 = df[df.set == "strat3"]
    conc = _conc("de", s3.lemma, s3.concept_en)
    miss = np.isnan(conc)
    conc[miss] = np.nanmean(conc)  # as Phase 3 (mean over the block)
    covs = [conc, miss.astype(float), pd.to_numeric(s3.zipf).to_numpy(),
            np.array([ntok_of[w] for w in s3.lemma], dtype=float)]  # fmt: skip
    sel = s3.gender.isin(genders).to_numpy()
    idx = np.array([row[w] for w in s3.lemma])
    return {"X": X[idx][sel], "y": (s3.gender[sel] == "f").to_numpy(int),
            "g": s3.gender[sel].to_numpy(), "cells": (s3.ending + "|" + s3.loan)[sel].to_numpy(),
            "covs": [c[sel] for c in covs], "lemma": s3.lemma[sel].to_numpy()}  # fmt: skip


def pooled(es: dict, de: dict) -> dict:
    """ES + DE stacked; cells = language x cell; covariates shared (concreteness, missing flag,
    frequency, token count), Spanish-only ones (loan dummies, initial a) zero for German."""
    n_es, n_de = len(es["y"]), len(de["y"])
    covs = [np.r_[a, b] for a, b in zip(es["covs"][:4], de["covs"], strict=True)]
    covs += [np.r_[c, np.zeros(n_de)] for c in es["covs"][4:]]
    return {"X": None, "y": np.r_[es["y"], de["y"]],
            "cells": np.r_[np.char.add("es:", es["cells"].astype(str)),
                           np.char.add("de:", de["cells"].astype(str))],
            "covs": covs, "n_es": n_es}  # fmt: skip


# ---- per-layer computations -------------------------------------------------------
def _fit(X, d: dict, idx=None, kind: str = "probe") -> np.ndarray:
    idx = np.arange(len(d["y"])) if idx is None else idx
    Z = nuisance(d["cells"][idx], [c[idx] for c in d["covs"]])
    if kind == "probe":
        return probe_strat(X[idx], d["y"][idx], Z)
    return class_betas(X[idx], d["y"][idx][:, None].astype(float), Z)[0]


def within_cell_auc(y, s, cells) -> float:
    """AUC pooled over within-cell m-f comparisons (cells with both classes only)."""
    num = den = 0.0
    for c in np.unique(cells):
        k = cells == c
        yc, sc = y[k], s[k]
        n1, n0 = int(yc.sum()), int((1 - yc).sum())
        if n1 and n0:
            num += auc(yc, sc) * n1 * n0
            den += n1 * n0
    return num / den if den else float("nan")


def paired_auc(d_diff: np.ndarray, de_f: np.ndarray) -> float:
    return auc(de_f.astype(int), d_diff)


def _ci(v, level: float) -> tuple[float, float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    a = (1 - level) / 2 * 100
    return (float(np.percentile(v, a)), float(np.percentile(v, 100 - a))) if len(v) else (
        np.nan, np.nan)  # fmt: skip


def analyze(model_id: str, position: str = "last", out_root: str | None = None,
            pairs_path: str = PAIRS_FINAL, n_boot: int = N_BOOT) -> dict:  # fmt: skip
    """All Phase 4 tests for one model and readout position."""
    sx = _sx(position)
    out = Path(out_root or f"results/phase4{sx}") / model_slug(model_id)
    es, de = es_train(model_id, position), de_train(model_id, position)
    de3 = de_train(model_id, position, ("m", "f", "n"))
    po = pooled(es, de)
    pairs = pd.read_csv(pairs_path, keep_default_na=False)
    pool = pd.read_csv(PAIRS_POOL, keep_default_na=False)
    Xp_de, _ = acts.load(model_id, "phase4_pairs_de" + sx)
    Xp_es, _ = acts.load(model_id, "phase4_pairs_es" + sx)
    prow = {pid: i for i, pid in enumerate(pool.pair_id)}
    pi = np.array([prow[x] for x in pairs.pair_id])
    Xp_de, Xp_es = Xp_de[pi], Xp_es[pi]
    de_f = (pairs.de_gender == "f").to_numpy()
    cog = pairs.en_cognate.astype(str).eq("True").to_numpy()
    ru = None
    small = any(t in model_id for t in ("0.6B", "1.7B"))  # Russian dropped below 4B (decisions)
    if Path(RU_FINAL).exists() and not small:
        try:
            Xru, _ = acts.load(model_id, "phase4_ru" + sx)
            rdf = pd.read_csv(RU_FINAL, keep_default_na=False)
            ru = (Xru, (rdf.gender == "f").to_numpy(int), rdf.cell.to_numpy())
        except FileNotFoundError:
            pass
    L = es["X"].shape[1]
    rng = np.random.default_rng(SEED)
    rows = []
    stage(f"{model_id} [{position}]: Phase 4 over {L - 2} inner layers; ES {len(es['y'])}, "
          f"DE {len(de['y'])} training nouns, {len(pairs)} pairs, {n_boot} bootstrap rounds")  # fmt: skip
    for layer in progress(range(1, L - 1), desc=f"phase4 {position}", unit="layer"):
        Xe = es["X"][:, layer].astype(np.float64)
        Xd = de["X"][:, layer].astype(np.float64)
        Xpo = np.r_[Xe, Xd]
        Pde, Pes = Xp_de[:, layer].astype(np.float64), Xp_es[:, layer].astype(np.float64)
        Xn = de3["X"][:, layer].astype(np.float64)

        def metrics(ie, idd, pair_idx, te_es, te_de, kind="probe", Xe=Xe, Xd=Xd, Xpo=Xpo,
                    Pde=Pde, Pes=Pes, Xn=Xn, layer=layer):  # fmt: skip
            """ie/idd: training rows (ES/DE, maybe resampled); pair_idx: pairs; te_*: test rows."""
            d_es = _fit(Xe, es, ie, kind)
            d_de = _fit(Xd, de, idd, kind)
            ip = np.r_[ie, len(es["y"]) + idd]
            d_po = _fit(Xpo, po, ip, kind)
            r = {
                "t1_es_on_de": within_cell_auc(
                    de["y"][te_de], Xd[te_de] @ d_es, de["cells"][te_de]
                ),
                "t1_de_on_es": within_cell_auc(
                    es["y"][te_es], Xe[te_es] @ d_de, es["cells"][te_es]
                ),
            }
            for nm, dv in (("es", d_es), ("de", d_de), ("pooled", d_po)):
                diff = (Pde[pair_idx] - Pes[pair_idx]) @ dv
                r[f"t3_{nm}"] = paired_auc(diff, de_f[pair_idx])
                pred = np.where(de_f[pair_idx], 1, -1)
                r[f"t3_{nm}_share"] = float(np.mean(np.sign(diff) == pred))
                nc = ~cog[pair_idx]
                r[f"t3_{nm}_nocog"] = paired_auc(diff[nc], de_f[pair_idx][nc])
                # same-language (descriptive): each language's own nouns on its own direction
            r["t3_samelang"] = paired_auc(Pde[pair_idx] @ d_de - Pes[pair_idx] @ d_es,
                                          de_f[pair_idx])  # fmt: skip
            if ru is not None:
                Xr = ru[0][:, layer].astype(np.float64)
                for nm, dv in (("es", d_es), ("de", d_de), ("pooled", d_po)):
                    r[f"t4_{nm}"] = within_cell_auc(ru[1], Xr @ dv, ru[2])
            # Q5: German neuter position between m (0) and f (1), within cells
            for nm, dv in (("es", d_es), ("pooled", d_po)):
                s = Xn @ dv
                pos = []
                for c in np.unique(de3["cells"]):
                    k = de3["cells"] == c
                    g = de3["g"][k]
                    if {"m", "f", "n"} <= set(g):
                        mm, mf = s[k][g == "m"].mean(), s[k][g == "f"].mean()
                        if mf != mm:
                            pos += list((s[k][g == "n"] - mm) / (mf - mm))
                r[f"q5_neuter_pos_{nm}"] = float(np.median(pos)) if pos else float("nan")
            return r

        all_es, all_de, all_p = (
            np.arange(len(es["y"])),
            np.arange(len(de["y"])),
            np.arange(len(pairs)),
        )
        point = metrics(all_es, all_de, all_p, all_es, all_de)
        point_dom = metrics(all_es, all_de, all_p, all_es, all_de, kind="dom")
        boots = []
        for _ in range(n_boot):
            ie = boot_within(es["cells"], es["y"], rng)
            idd = boot_within(de["cells"], de["y"], rng)
            te_es = boot_within(es["cells"], es["y"], rng)
            te_de = boot_within(de["cells"], de["y"], rng)
            ip = rng.choice(len(pairs), len(pairs))
            boots.append(metrics(ie, idd, ip, te_es, te_de))
        bt = pd.DataFrame(boots)
        row = {"layer": layer, **point, **{f"dom_{k}": v for k, v in point_dom.items()}}
        for k in bt.columns:
            for lev, tag in ((0.95, ""), (0.975, "_holm")):
                lo, hi = _ci(bt[k], lev)
                row[f"{k}_lo{tag}"], row[f"{k}_hi{tag}"] = lo, hi
        # T2: cross-fitted ES-DE cosine (class vectors at equal cell/covariates), shuffle null
        labels = np.r_[np.where(es["y"] == 1, "es_f", "m"), np.where(de["y"] == 1, "de_f", "m")]
        K = Xpo @ Xpo.T
        g2 = split_half_geometry(Xpo, labels, po["cells"], po["covs"], ["es_f", "de_f"], "m",
                                 rng, n_splits=N_SPLITS, gram=K)  # fmt: skip
        null = [split_half_geometry(Xpo, shuffle_within(labels, po["cells"], rng), po["cells"],
                                    po["covs"], ["es_f", "de_f"], "m", rng, n_splits=20,
                                    gram=K)["cos"] for _ in range(N_SHUFFLE)]  # fmt: skip
        row |= {"t2_cos": g2["cos"], "t2_null_hi": float(np.nanpercentile(null, 97.5)),
                "t2_rel_es": g2["rel_es_f"], "t2_rel_de": g2["rel_de_f"]}  # fmt: skip
        rows.append(row)
    res = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "layers.csv", index=False)
    summary = {"model": model_id, "position": position, "analysis_git": git_state(),
               "pairs": pairs_path, "n_pairs": len(pairs), "n_es": len(es["y"]),
               "n_de": len(de["y"]), "n_boot": n_boot, **verdicts(res)}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "analysis_git"}, indent=1,
                     default=float))  # fmt: skip
    return summary


def _layers_above(res: pd.DataFrame, key: str, tag: str = "") -> int:
    return int((res[f"{key}_lo{tag}"] > 0.5).sum())


def verdicts(res: pd.DataFrame) -> dict:
    """Layer counts per test. A test "passes" if its lower bound is > 0.5 in a majority of inner
    layers. Holm across the co-primaries T1 and T3 (proposed implementation, see decisions.md):
    both are first judged with 97.5% intervals (alpha/2); if one passes there, the other is
    judged with 95% intervals. T1 = both transfer directions; T3 = ES-on-both and DE-on-both
    (proposed); POOLED, share, no-cognate and same-language versions are secondary."""
    n = len(res)
    maj = n // 2 + 1
    out = {"inner_layers": n}
    keys = ["t1_es_on_de", "t1_de_on_es", "t3_es", "t3_de", "t3_pooled", "t3_es_nocog",
            "t3_de_nocog", "t3_pooled_nocog", "t3_samelang", "t4_es", "t4_de", "t4_pooled"]  # fmt: skip
    for k in keys:
        if f"{k}_lo" in res:
            out[f"{k}_layers"] = _layers_above(res, k)
            out[f"{k}_layers_holm"] = _layers_above(res, k, "_holm")
            out[f"{k}_mean"] = float(res[k].mean())
    t1s = all(out.get(f"{k}_layers_holm", 0) >= maj for k in ("t1_es_on_de", "t1_de_on_es"))
    t3s = all(out.get(f"{k}_layers_holm", 0) >= maj for k in ("t3_es", "t3_de"))
    t1 = all(out.get(f"{k}_layers", 0) >= maj for k in ("t1_es_on_de", "t1_de_on_es"))
    t3 = all(out.get(f"{k}_layers", 0) >= maj for k in ("t3_es", "t3_de"))
    out["T1_transfers"] = bool(t1s or (t3s and t1))
    out["T3_pairs_follow_gender"] = bool(t3s or (t1s and t3))
    out["t2_cos_mean"] = float(res.t2_cos.mean())
    out["t2_layers_above_null"] = int((res.t2_cos > res.t2_null_hi).sum())
    for nm in ("es", "pooled"):
        out[f"q5_neuter_pos_{nm}_median"] = float(res[f"q5_neuter_pos_{nm}"].median())
    return out
