"""Suffix follow-up test set `suffix_ctrl` (pre-registration adopted 2026-10-08, D1-D4).

Derived German nouns of every gender, so the single-root gender direction can be tested on
whether it *ranks* suffix nouns by gender (S1), not only where feminine-suffix nouns sit.
  masculine  -ismus (borrowed)                     (-ling dropped, D2)
  neuter     -tum (native), -ment (borrowed)       (some are masculine: der Reichtum, der Moment;
                                                    genders always come from Wiktionary)
  feminine   top-ups for -ung, -keit (native; the v3 suffix set kept only 1 known noun each)

Same filters as the Phase 3 pool (phase3_stimuli.exclusions), with one fix: the compound check
there treats the suffix itself as a head (Kapital-ismus: "Ismus" is a Wiktionary noun), which is
why the v3 suffix set has no -ismus nouns. Here an ending only marks a compound if it is a lexicon
noun *longer than the suffix* (Bundes-regierung still is one). Nouns in any Phase 3 training set
are excluded (asserted). Up to 20 per suffix, Zipf >= 2.5, seeded.
"""

from __future__ import annotations

import pandas as pd

from .lexicon import load_lexicon, source_tag
from .phase2_stimuli import flags
from .phase3_stimuli import SEED, _base, etymology, exclusions, german_verbs

POOL = "data/stimuli/phase3_suffixctrl_pool_v1.csv"
SUFFIXES = {"ismus": "m", "tum": "n", "ment": "n", "ung": "f", "keit": "f"}
N_PER = 20
ZMIN = 2.5
SOCIAL_GENDER = {"Feminismus", "Chauvinismus"}  # PI review 2026-10-08
TRAIN_SETS = ["strat3", "matched3", "matched3_end", "matched2", "compound_train"]


def _suffix(w: str) -> str:
    for s in SUFFIXES:
        if w.lower().endswith(s) and len(w) >= len(s) + 3:
            return s
    return ""


def _real_compound(w: str, nouns: set[str], suffix: str) -> bool:
    lw = w.lower()
    return any(lw[i:] in nouns and len(lw) - i > len(suffix) for i in range(3, len(lw) - 2))


def _latin_sourced(words: set[str]) -> set[str]:
    """Nouns whose etymology templates name a Latin source (la, la-new, ML.): Latin -um words
    (Ultimatum, Faktum, Rektum) as opposed to the native suffix -tum (Irrtum < MHG irretuom)."""
    import json

    from .lexicon import dump_path

    out = set()
    with open(dump_path("de"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("lang_code") != "de" or r.get("pos") != "noun" or r.get("word") not in words:
                continue
            for t in r.get("etymology_templates", []):
                vals = [str(v) for k, v in t.get("args", {}).items() if k != "1"]
                if any(
                    v.split(":")[0] in ("la", "la-new", "ML.") or v.startswith(":bor")
                    for v in vals[:2]
                ):
                    out.add(r["word"])
    return out


def build() -> pd.DataFrame:
    lex = load_lexicon("de")
    nouns = set(lex.lemma.str.lower())
    base = _base(lex).copy()
    base["suffix"] = base.lemma.map(_suffix)
    base = base[(base.suffix != "") & (base.zipf >= ZMIN)].copy()
    why = exclusions(base, nouns, german_verbs())
    # Replace the generic compound verdict with the suffix-aware one.
    real = [_real_compound(w, nouns, s) for w, s in zip(base.lemma, base.suffix, strict=True)]
    why = [
        "; ".join([r for r in x.split("; ") if r and r != "compound"] + (["compound"] if c else []))
        for x, c in zip(why, real, strict=True)
    ]
    base["excluded"] = why
    v3 = pd.read_csv("data/stimuli/phase3_final_v3.csv", keep_default_na=False)
    train = set(v3[v3.set.isin(TRAIN_SETS)].lemma)
    existing = set(v3[v3.set == "suffix"].lemma)
    base.loc[base.lemma.isin(train), "excluded"] = "Phase 3 training noun"
    # PI review 2026-10-08: (1) -tum must be the native German suffix (Wachstum, Eigentum), not a
    # Latin -um word that happens to end in -tum (Ultimatum, Faktum, Praeteritum, Rektum);
    # (2) concepts carrying social gender in their meaning (as the sex-associated rule).
    latin_tum = (base.suffix == "tum") & base.lemma.isin(_latin_sourced(set(base.lemma)))
    base.loc[latin_tum, "excluded"] = "Latin -um word, not the suffix -tum"
    base.loc[base.lemma.isin(SOCIAL_GENDER), "excluded"] = "social gender in meaning"
    keep = base[base.excluded == ""]
    keep = keep[~keep.lemma.isin(existing)]  # already in the v3 suffix test set
    parts = [g.sample(min(N_PER, len(g)), random_state=SEED) for _, g in keep.groupby("suffix")]
    out = pd.concat(parts)
    assert not set(out.lemma) & train, "a training noun in suffix_ctrl"
    out = flags(out)
    out["loan"] = out.lemma.map(etymology(set(out.lemma))).fillna("unknown")
    out["set"] = "suffix_ctrl"
    out["lang"] = "de"
    out["source"] = source_tag("de")
    cols = ["lang", "lemma", "gender", "concept_en", "gloss", "set", "suffix", "loan", "zipf",
            "en_overlap", "sex_assoc", "source"]  # fmt: skip
    out = out[cols].sort_values(["suffix", "gender", "zipf"], ascending=[True, True, False])
    out.to_csv(POOL, index=False)
    base[["lemma", "gender", "suffix", "zipf", "excluded"]].to_csv(
        POOL.replace(".csv", "_screening.csv"), index=False
    )
    return out


# ---- known check, extraction, analysis (S1, S2) ------------------------------------------------
KNOWN_ROOT = "results/suffixctrl_known"
KNOWN_MODELS = ["Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]
ACTS = "suffixctrl_bare"
BORROWED_F = ["-tät", "-enz", "-anz", "-ik", "-ur"]  # v3 suffix set, feminine loan suffixes
NATIVE_F = ["-heit", "-keit", "-ung", "-ei"]
N_BOOT = 1000


def known(model_id: str, device=None, dtype=None) -> None:
    """German three-way known check (Phase 3 frames) on the suffix_ctrl pool."""
    from .phase3_known import score

    items = pd.read_csv(POOL, keep_default_na=False)
    score(model_id, device, dtype, out_root=KNOWN_ROOT, items=items)


def final() -> pd.DataFrame:
    """suffix_ctrl nouns known by every KNOWN_MODELS model."""
    from .models import model_slug

    df = pd.read_csv(POOL, keep_default_na=False)
    ok = None
    for m in KNOWN_MODELS:
        s = pd.read_csv(f"{KNOWN_ROOT}/{model_slug(m)}/scores.csv", keep_default_na=False)
        k = set(s[s.status == "known"].lemma)
        ok = k if ok is None else ok & k
    return df[df.lemma.isin(ok)]


def extract(model_id: str, device=None, dtype=None) -> None:
    """Bare nouns at LAST, same input format as Phase 3 (phase3_bare)."""
    from datetime import UTC, datetime

    from . import activations as acts
    from .models import load_model, pick_device, pick_dtype, run_metadata, stage

    words = sorted(set(pd.read_csv(POOL, keep_default_na=False).lemma))
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["input_format"] = acts.INPUT_FORMAT
    meta["stimuli"] = POOL
    stage(f"{model_id}: extracting {len(words)} suffix_ctrl nouns at LAST")
    X, _, toks = acts.states_at(model, tok, words, acts.AFTER, desc="suffix_ctrl")
    acts.save(model_id, ACTS, X, words, toks, {**meta, "position": "last"})


def analyze(model_id: str, out_root: str = "results/suffix_followup") -> dict:
    """S1 (confirmatory at 8B/14B; exploratory here) and S2 (descriptive), per the adopted
    pre-registration (docs/design/suffix-followup-prereg.md).

    Direction: the single-root m/f probe of Phase 3 (stratified, strat3, residualised on cells +
    covariates), unchanged. S1: AUC of feminine borrowed-suffix nouns (-tät/-enz/-anz/-ik/-ur, v3
    suffix set) over masculine -ismus nouns; 95% CI from a bootstrap that resamples suffixes,
    then nouns within suffixes. "Ranks correctly" = lower bound > 0.5. Spelling check: the gender
    gap (mean of feminine suffix-group means minus the -ismus mean) must exceed the 95th
    percentile of the gaps between feminine suffix groups. S2: each suffix group's position
    (0 = training masculine mean, 1 = feminine) before and after removing the covariate part
    (frequency, token count, concreteness) with the training fit's coefficients.
    """
    import json
    from pathlib import Path

    import numpy as np

    from . import activations as acts
    from .estimators import auc, nuisance, probe_strat
    from .lexicon import load_lexicon
    from .models import git_state, model_slug
    from .norms import rate

    v3 = pd.read_csv("data/stimuli/phase3_final_v3.csv", keep_default_na=False)
    sc = final()
    X3, m3 = acts.load(model_id, "phase3_bare")
    XC, mc = acts.load(model_id, ACTS)
    gl = load_lexicon("de").drop_duplicates("lemma").set_index("lemma").gloss

    def covs(words, zipf, toks_of):
        rated = [rate(gl.get(w, w))[0] for w in words]
        conc = np.array([np.nan if v is None else v for v in rated], dtype=float)
        miss = np.isnan(conc)
        conc[miss] = np.nanmean(conc) if (~miss).any() else 0.0
        ntok = np.array([toks_of[w] for w in words], dtype=float)
        return np.column_stack([conc, miss.astype(float), np.asarray(zipf, float), ntok])

    t3 = {w: len(t) for w, t in zip(m3["words"], m3["tokens"], strict=True)}
    tc = {w: len(t) for w, t in zip(mc["words"], mc["tokens"], strict=True)}
    r3 = {w: i for i, w in enumerate(m3["words"])}
    rc = {w: i for i, w in enumerate(mc["words"])}

    tr = v3[(v3.set == "strat3") & v3.gender.isin(["m", "f"])]
    i_tr = np.array([r3[w] for w in tr.lemma])
    C_tr = covs(tr.lemma, pd.to_numeric(tr.zipf), t3)
    cells = (tr.ending + "|" + tr.loan).to_numpy()
    y = (tr.gender == "f").to_numpy().astype(int)

    # test groups: v3 suffix nouns of the suffix's own gender + suffix_ctrl nouns
    sv3 = v3[(v3.set == "suffix") & (v3.gender == "f")]
    groups = {}  # suffix -> (source, row indices, covariates)
    for suf, g in sv3.groupby("de_suffix"):
        groups[suf] = (
            "v3",
            np.array([r3[w] for w in g.lemma]),
            covs(g.lemma, pd.to_numeric(g.zipf), t3),
        )
    for suf, g in sc.groupby("suffix"):
        want = SUFFIXES[suf]
        g = g[g.gender == want] if suf != "tum" else g  # -tum: Irrtum (m) kept, reported
        key = "-" + suf + ("_ctrl" if "-" + suf in groups else "")
        groups[key] = (
            "ctrl",
            np.array([rc[w] for w in g.lemma]),
            covs(g.lemma, pd.to_numeric(g.zipf), tc),
        )
    f_bor = [s for s in BORROWED_F if s in groups]
    rng = np.random.default_rng(SEED)
    L = X3.shape[1]
    rows = []
    for layer in range(1, L - 1):
        Xt = X3[i_tr, layer].astype(np.float64)
        Z = nuisance(cells, list(C_tr.T))
        d = probe_strat(Xt, y, Z)
        # covariate coefficients from the training fit (cells absorb the intercept)
        B = np.linalg.lstsq(Z, Xt, rcond=None)[0][-C_tr.shape[1] :]
        cmean = C_tr.mean(0)
        s_tr = Xt @ d
        s_tr_adj = (Xt - (C_tr - cmean) @ B) @ d
        mu = {
            k: (v[y == 0].mean(), v[y == 1].mean()) for k, v in (("raw", s_tr), ("adj", s_tr_adj))
        }
        score = {}
        out = {"layer": layer}
        for key, (src, idx, C) in groups.items():
            X = (X3 if src == "v3" else XC)[idx, layer].astype(np.float64)
            s, s_adj = X @ d, (X - (C - cmean) @ B) @ d
            score[key] = s
            for k, v in (("raw", s), ("adj", s_adj)):
                m0, m1 = mu[k]
                out[f"pos_{k}{key}"] = float(((v - m0) / (m1 - m0)).mean())
        fm = np.concatenate([score[s] for s in f_bor])
        mm = score["-ismus"]
        out["S1_auc"] = auc(np.r_[np.zeros(len(mm)), np.ones(len(fm))], np.r_[mm, fm])
        boots = []
        for _ in range(N_BOOT):
            fs = [score[s] for s in rng.choice(f_bor, len(f_bor))]
            fb = np.concatenate([rng.choice(v, len(v)) for v in fs])
            mb = rng.choice(mm, len(mm))
            boots.append(auc(np.r_[np.zeros(len(mb)), np.ones(len(fb))], np.r_[mb, fb]))
        out["S1_auc_lo"], out["S1_auc_hi"] = (float(np.percentile(boots, q)) for q in (2.5, 97.5))
        f_means = np.array([score[s].mean() for s in f_bor])
        out["gender_gap"] = float(f_means.mean() - mm.mean())
        pair = np.abs(f_means[:, None] - f_means[None, :])[np.triu_indices(len(f_means), 1)]
        out["between_f_gap95"] = float(np.percentile(pair, 95))
        out["spelling_check"] = bool(out["gender_gap"] > out["between_f_gap95"])
        rows.append(out)
    res = pd.DataFrame(rows)
    outd = Path(out_root) / model_slug(model_id)
    outd.mkdir(parents=True, exist_ok=True)
    res.to_csv(outd / "layers.csv", index=False)
    n = len(res)
    ranks = res.S1_auc_lo > 0.5
    summ = {
        "model": model_id, "analysis_git": git_state(), "inner_layers": n,
        "n": {k: len(v[1]) for k, v in groups.items()},
        "S1_ranks_correctly_layers": int(ranks.sum()),
        "S1_spelling_check_layers": int(res.spelling_check.sum()),
        "S1_both_layers": int((ranks & res.spelling_check).sum()),
        "S1_reading": ("offset, not failure (ranks correctly beyond spelling)"
                       if (ranks & res.spelling_check).sum() > n / 2 else
                       "ranks correctly but not separable from spelling" if ranks.sum() > n / 2
                       else "two routes supported (does not rank)"),
        "mean": {c: float(res[c].mean()) for c in res.columns if c.startswith(("S1_auc", "pos_", "gender_gap", "between"))},
    }  # fmt: skip
    (outd / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summ.items() if k != "mean"}, indent=1, ensure_ascii=False))
    return summ
