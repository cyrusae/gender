"""Phase 2: a Spanish grammatical-gender direction, tested where spelling and gender disagree.

Pre-registered reading: docs/decisions.md (2026-10-06). Explainer: docs/explainers/02-*.md.

Items: nouns known (Phase 0 status "known") by all three Qwen sizes; see finalize().
Directions (fem - masc, per layer):
  dom_matched     difference of class means on the ending-matched set   <- PRIMARY
  probe_matched   logistic-regression weights on the ending-matched set
  dom_regular     difference of means on regular -o/-a training nouns (comparison)
  dom_regular_verberase / dom_regular_rank2   same, after the Phase 1 erasers
Score s(x) = projection of x on the direction (after the direction's eraser, if any).
Gender index p(x) = (s - mean s[regular test m]) / (mean s[regular test f] - mean s[regular test m]).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from concept_erasure import LeaceEraser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from . import activations as acts
from .models import (
    git_state,
    load_model,
    model_slug,
    pick_device,
    pick_dtype,
    progress,
    run_metadata,
    stage,
)
from .phase1 import load_stimuli as load_phase1

POOL = "data/stimuli/phase2_pool_v3.csv"
MULTI = "data/stimuli/phase2_multi_v2.csv"
FINAL = "data/stimuli/phase2_final_v3.csv"
KNOWN_MODELS = ["Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]  # 0.6B dropped (decisions.md)
N_BOOT = 1000
# Primary direction for confirmatory runs (decisions.md, 2026-10-06: single estimator for all
# sizes, chosen by the pre-registered training-only CV rule). Earlier dev runs used adom_matched.
PRIMARY = "probe_matched"
SEED = 0


# ---- stimuli -------------------------------------------------------------------
def finalize(known_root: str = "results/phase2_known") -> pd.DataFrame:
    """Keep nouns known by every model in KNOWN_MODELS; freeze the regular train/test split."""
    pool = pd.read_csv(POOL, keep_default_na=False)
    known = None
    for m in KNOWN_MODELS:
        it = pd.read_csv(Path(known_root) / model_slug(m) / "items.csv", keep_default_na=False)
        k = set(it[it.passed.astype(str) == "True"].lemma)
        known = k if known is None else known & k
    df = pool[pool.lemma.isin(known)].copy()
    rng = np.random.default_rng(SEED)
    # The known filter breaks the matched set's per-ending m/f balance: restore it.
    keep = []
    for _, g in df[df.set == "matched"].groupby("ending"):
        k = min((g.gender == "m").sum(), (g.gender == "f").sum())
        for _, gg in g.groupby("gender"):
            keep += list(gg.sort_values("lemma").index[:k])
    df = df[(df.set != "matched") | df.index.isin(keep)]
    df["split"] = "test"
    df.loc[df.set == "matched", "split"] = "train"
    for g in ("m", "f"):
        idx = df.index[(df.set == "regular") & (df.gender == g)].to_numpy().copy()
        rng.shuffle(idx)
        df.loc[idx[: round(len(idx) * 2 / 3)], "split"] = "train"
    df.to_csv(FINAL, index=False)
    print(df.groupby(["set", "split", "gender"]).size().unstack(fill_value=0).to_string())
    print(f"dropped as not known by all models: {len(pool) - len(df)} of {len(pool)}")
    return df


def _final():
    df = pd.read_csv(FINAL, keep_default_na=False)
    test_lemmas = set(df[df.split == "test"].lemma)
    assert not test_lemmas & set(df[df.split == "train"].lemma), "train/test overlap"
    return df


# ---- extraction --------------------------------------------------------------
def _suffix(position: str) -> str:
    """Activation-name suffix per readout position (LAST is primary; see readout.py)."""
    return {"last": "", "after": "_after"}[position]


def extract(model_id: str, device=None, dtype=None, position: str = "last") -> None:
    df, multi = _final(), pd.read_csv(MULTI, keep_default_na=False)
    homo = df[df.set == "homograph"]
    texts = {
        "phase2_bare": list(df.lemma),
        "phase2_homo_noun": [f"mi {w}" for w in homo.lemma],
        "phase2_homo_verb": [
            f.format(w=w) for f, w in zip(homo.verb_frame, homo.lemma, strict=True)
        ],
        "phase2_multi_m": [f"el {w}" for w in multi.lemma],
        "phase2_multi_f": [f"la {w}" for w in multi.lemma],
    }
    if position == "after":  # the Phase 1 erasers must be fit at the same position
        from .phase1 import load_stimuli

        texts |= {
            f"phase1_{k}": list(v[0]) for k, v in load_stimuli().items() if k in ("verbs", "nonce")
        }
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["input_format"] = acts.INPUT_FORMAT + (
        " (+ newline, read there)" if position == "after" else ""
    )
    meta["position"] = position
    after = acts.AFTER if position == "after" else None
    for name, ws in texts.items():
        stage(f"{model_id}: extracting {name} ({len(ws)} texts) at {position.upper()}")
        X_last, X_after, toks = acts.states_at(model, tok, ws, after, desc=name)
        X = X_after if position == "after" else X_last
        acts.save(model_id, name + _suffix(position), X, ws, toks, {**meta, "stimuli": FINAL})
        print(f"{name}: {X.shape}")


# ---- analysis helpers ---------------------------------------------------------
def _adom(X, y, covs):
    """Adjusted difference of means: per activation dimension, least squares
    x = a + b*female + sum_k c_k*cov_k; returns b (the gender coefficients)."""
    Z = np.column_stack([np.ones(len(y)), y, *[(c - c.mean()) for c in covs]])
    B, *_ = np.linalg.lstsq(Z, X, rcond=None)
    return B[1]


def _dom(X, y):
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def _probe_dir(X, y):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), y)
    return lr.coef_[0] / sc.scale_  # direction in raw activation space


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def _boot(fn, n, rng, n_boot=N_BOOT):
    """95% percentile CI of fn(idx) over bootstrap resamples of n items."""
    vals = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _boot_two(neg, pos, rng, n_boot=N_BOOT):
    """95% CI of AUC(pos > neg), resampling each group separately."""
    vals = []
    for _ in range(n_boot):
        a, b = rng.choice(neg, len(neg)), rng.choice(pos, len(pos))
        vals.append(_auc(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


CI_KEYS = ["A_f", "A_m", "A_f_noen", "A_m_noen", "A_f_abs", "A_m_abs", "A_f_con", "A_m_con", "exc_auc", "homo_auc_noun", "homo_auc_verb",
           "homo_auc_diff", "multi_p_diff"]  # fmt: skip


def _core(d, idx, Xt, hn, hv, mm, mf, y, homo_y) -> dict:
    """Metrics that get CIs, for direction d on the test groups selected by idx."""
    s = Xt @ d
    rf, rm = s[idx["reg_f"]], s[idx["reg_m"]]
    out = {}
    # PRIMARY (amended before results): masculine -a exceptions vs regular nouns.
    #   A_f = P(regular fem -a noun above a masc -a exception): same ending, different gender
    #   A_m = P(masc -a exception above a regular masc -o noun): same gender, different ending
    for tag, key in (("", "excm"), ("_noen", "excm_noen")):
        xe = s[idx[key]]
        out[f"A_f{tag}"] = _auc(np.r_[np.zeros(len(xe)), np.ones(len(rf))], np.r_[xe, rf])
        out[f"A_m{tag}"] = _auc(np.r_[np.zeros(len(rm)), np.ones(len(xe))], np.r_[rm, xe])
    for tag in ("abs", "con"):
        if f"excm_{tag}" in idx:
            xe, rf2, rm2 = s[idx[f"excm_{tag}"]], s[idx[f"reg_f_{tag}"]], s[idx[f"reg_m_{tag}"]]
            out[f"A_f_{tag}"] = _auc(np.r_[np.zeros(len(xe)), np.ones(len(rf2))], np.r_[xe, rf2])
            out[f"A_m_{tag}"] = _auc(np.r_[np.zeros(len(rm2)), np.ones(len(xe))], np.r_[rm2, xe])
    out["exc_auc"] = _auc(y[idx["exc"]], s[idx["exc"]])
    h = idx["homo"]
    out["homo_auc_noun"] = _auc(homo_y[h], hn[h] @ d)
    out["homo_auc_verb"] = _auc(homo_y[h], hv[h] @ d)
    out["homo_auc_diff"] = out["homo_auc_noun"] - out["homo_auc_verb"]
    scale = rf.mean() - rm.mean()
    mi = idx["multi"]
    out["multi_p_diff"] = float(((mf[mi] @ d) - (mm[mi] @ d)).mean() / scale)
    return out


def _verdict(f_lo, f_hi, m_lo, m_hi) -> str:
    """Pre-registered (amended) reading of the masculine-exception test."""
    gender, spelling = f_lo > 0.5, m_lo > 0.5
    if gender and not spelling:
        return "gender"
    if spelling and not gender:
        return "spelling"
    if gender and spelling:
        return "mixed"
    return "neither"


def _leans(model_id: str, lemmas) -> np.ndarray:
    p = Path("results/multigender") / model_slug(model_id) / "items.csv"
    it = pd.read_csv(p, keep_default_na=False)
    it = it[it.lang == "es"].set_index("lemma")
    lean = np.sign(it.ctx1_margin + it.ctx2_margin)  # >0 = leans masculine
    return np.array([lean.get(w, 0.0) for w in lemmas])


def analyze(model_id: str, out_root: str | None = None, position: str = "last") -> pd.DataFrame:
    df, multi = _final(), pd.read_csv(MULTI, keep_default_na=False)
    sx = _suffix(position)
    out_root = out_root or f"results/phase2{sx}"
    Xb, mb = acts.load(model_id, "phase2_bare" + sx)
    Xhn, _ = acts.load(model_id, "phase2_homo_noun" + sx)
    Xhv, _ = acts.load(model_id, "phase2_homo_verb" + sx)
    Xmm, _ = acts.load(model_id, "phase2_multi_m" + sx)
    Xmf, _ = acts.load(model_id, "phase2_multi_f" + sx)
    Xv1, _ = acts.load(model_id, "phase1_verbs" + sx)
    Xn1, _ = acts.load(model_id, "phase1_nonce" + sx)
    p1 = load_phase1()
    _, yv1, _ = p1["verbs"]
    _, yn1, _, split1 = p1["nonce"]
    tr1 = split1 == "train"

    y = (df.gender == "f").to_numpy(int)
    s_ = df.set.to_numpy()
    matched = s_ == "matched"
    reg_tr = (s_ == "regular") & (df.split == "train").to_numpy()
    reg_te = (s_ == "regular") & (df.split == "test").to_numpy()
    exc = np.char.startswith(s_.astype(str), "exception")
    excm = exc & (y == 0)  # masculine -a exceptions (el problema, el día)
    en = (df.en_overlap.astype(str) == "True").to_numpy()
    homo_y = (df[df.set == "homograph"].gender == "f").to_numpy(int)
    endings = df.ending.to_numpy()
    from .lexicon import load_lexicon
    from .norms import rate

    gl = load_lexicon("es").drop_duplicates("lemma").set_index("lemma").gloss
    rated = [rate(gl.get(w, c))[0] for w, c in zip(df.lemma, df.concept_en, strict=True)]
    conc = np.array([np.nan if v is None else v for v in rated], dtype=float)
    conc_missing = np.isnan(conc)
    conc[conc_missing] = np.nanmean(conc[matched])  # impute the training-set mean (flagged)
    zipf = pd.to_numeric(df.zipf, errors="coerce").to_numpy()
    covs = [conc, zipf]

    def fit_adom(Xt, yy, idx):
        return _adom(Xt[idx], yy[idx], [c[idx] for c in covs])

    lean = _leans(model_id, multi.lemma)
    rng = np.random.default_rng(SEED)
    full_idx = {  # test groups, as row indices into their own arrays
        "reg_f": np.where(reg_te & (y == 1))[0],
        "reg_m": np.where(reg_te & (y == 0))[0],
        "excm": np.where(excm)[0],
        "excm_noen": np.where(excm & ~en)[0],
        "exc": np.where(exc)[0],
        "homo": np.arange(len(homo_y)),
        "multi": np.arange(len(multi)),
    }
    # Test-side meaning control: split regular test nouns and masc exceptions at the median rating
    # of the regular test nouns, and compare within each half (sensitivity, not primary).
    med = np.median(conc[reg_te & ~conc_missing])
    for tag, half in (("abs", conc < med), ("con", conc >= med)):
        ok = half & ~conc_missing
        full_idx[f"reg_f_{tag}"] = np.where(reg_te & (y == 1) & ok)[0]
        full_idx[f"reg_m_{tag}"] = np.where(reg_te & (y == 0) & ok)[0]
        full_idx[f"excm_{tag}"] = np.where(excm & ok)[0]

    rows = []
    stage(
        f"{model_id}: Phase 2 analysis, {Xb.shape[1]} layers x 6 directions x {N_BOOT} bootstrap rounds"
    )
    for layer in progress(range(Xb.shape[1]), desc="layers", unit="layer"):
        X = Xb[:, layer].astype(np.float64)
        hn, hv = Xhn[:, layer].astype(np.float64), Xhv[:, layer].astype(np.float64)
        mm, mf = Xmm[:, layer].astype(np.float64), Xmf[:, layer].astype(np.float64)
        xv1, xn1 = Xv1[:, layer].astype(np.float64), Xn1[:, layer].astype(np.float64)
        e_verb = LeaceEraser.fit(torch.from_numpy(xv1), torch.from_numpy(yv1).long())
        z2 = np.zeros((len(xv1) + int(tr1.sum()), 2))
        z2[: len(xv1), 0] = yv1 - 0.5
        z2[len(xv1) :, 1] = yn1[tr1] - 0.5
        e_r2 = LeaceEraser.fit(torch.from_numpy(np.r_[xv1, xn1[tr1]]), torch.from_numpy(z2))
        ident = lambda a: a
        ev = lambda a, e=e_verb: e(torch.from_numpy(a)).numpy()
        e2 = lambda a, e=e_r2: e(torch.from_numpy(a)).numpy()
        # name -> (transform applied to activations, training mask, fitting function)
        plain = lambda f: lambda A, yy, idx: f(A[idx], yy[idx])
        directions = {
            "adom_matched": (ident, matched, fit_adom),  # PRIMARY (amended before results)
            "dom_matched": (ident, matched, plain(_dom)),
            "probe_matched": (ident, matched, plain(_probe_dir)),
            "dom_regular": (ident, reg_tr, plain(_dom)),
            "dom_regular_verberase": (ev, reg_tr, plain(_dom)),
            "dom_regular_rank2": (e2, reg_tr, plain(_dom)),
        }
        # concreteness direction (diagnostic): slope of activations on rated concreteness,
        # controlling for gender, on regular training nouns
        k = reg_tr & ~conc_missing
        Zc = np.column_stack([np.ones(k.sum()), y[k], conc[k] - conc[k].mean()])
        cdir = np.linalg.lstsq(Zc, X[k], rcond=None)[0][2]
        for name, (T, trmask, fit) in directions.items():
            Xt, hnt, hvt, mmt, mft = T(X), T(hn), T(hv), T(mm), T(mf)
            tr_idx = np.where(trmask)[0]
            d = fit(Xt, y, tr_idx)
            core = lambda d, idx, Xt=Xt, hnt=hnt, hvt=hvt, mmt=mmt, mft=mft: _core(
                d, idx, Xt, hnt, hvt, mmt, mft, y, homo_y
            )
            r = {"layer": layer, "direction": name, **core(d, full_idx)}
            r["cos_concreteness"] = float(d @ cdir / np.linalg.norm(d) / np.linalg.norm(cdir))
            # 95% CIs: resample training nouns (refit the direction) AND test nouns
            boots = []
            for _ in progress(range(N_BOOT), desc=f"L{layer} {name}", unit="round", leave=False):
                tb = np.concatenate([rng.choice(tr_idx[y[tr_idx] == g], (y[tr_idx] == g).sum())
                                     for g in (0, 1)])  # fmt: skip
                ib = {k: rng.choice(v, len(v)) for k, v in full_idx.items()}
                boots.append(core(fit(Xt, y, tb), ib))
            for k in CI_KEYS:
                vals = np.array([b[k] for b in boots], dtype=float)
                vals = vals[np.isfinite(vals)]
                r[f"{k}_lo"], r[f"{k}_hi"] = (
                    (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
                    if len(vals)
                    else (float("nan"), float("nan"))
                )
            s = Xt @ d
            m0, m1 = s[reg_te & (y == 0)].mean(), s[reg_te & (y == 1)].mean()
            p = (s - m0) / (m1 - m0)
            r["regular_test_auc"] = _auc(y[reg_te], s[reg_te])
            for sub in ("exception_ma", "exception_clipping", "exception_true"):
                for g, gl in ((0, "m"), (1, "f")):
                    k = (s_ == sub) & (y == g)
                    if k.any():
                        r[f"p_{sub.removeprefix('exception_')}_{gl}"] = float(p[k].mean())
            r["n_excm"], r["n_excm_noen"] = int(excm.sum()), int((excm & ~en).sum())
            r["n_conc_imputed_train"] = int((conc_missing & trmask).sum())
            for w in df.lemma[exc & (y == 1)]:  # the few feminine -o exceptions, item by item
                r[f"p_item_{w}"] = float(p[(df.lemma == w).to_numpy()][0])
            dp = ((mft @ d) - (mmt @ d)) / (m1 - m0)
            r["multi_p_diff_lean_m"] = (
                float(dp[lean > 0].mean()) if (lean > 0).any() else float("nan")
            )
            r["multi_p_diff_lean_f"] = (
                float(dp[lean < 0].mean()) if (lean < 0).any() else float("nan")
            )
            if trmask is matched:  # in-domain: CV grouped by ending
                aucs = []
                for a, b in GroupKFold(n_splits=5).split(Xt[matched], y[matched], endings[matched]):
                    dd = fit(Xt, y, np.where(matched)[0][a])
                    aucs.append(_auc(y[matched][b], Xt[matched][b] @ dd))
                r["matched_cv_auc"] = float(np.nanmean(aucs))
            rows.append(r)
        r0 = next(x for x in rows[-len(directions) :] if x["direction"] == PRIMARY)
        tqdm.write(f"layer {layer:2d} adom_matched: A_f {r0['A_f']:.2f} [{r0['A_f_lo']:.2f},{r0['A_f_hi']:.2f}] "
              f"A_m {r0['A_m']:.2f} [{r0['A_m_lo']:.2f},{r0['A_m_hi']:.2f}]  p(-ma m)={r0.get('p_ma_m', float('nan')):.2f}  "
              f"homo diff {r0['homo_auc_diff']:+.2f}  multi {r0['multi_p_diff']:+.2f}  "
              f"regular AUC {r0['regular_test_auc']:.2f}  matched CV {r0['matched_cv_auc']:.2f}")  # fmt: skip
    res = pd.DataFrame(rows)
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "layers.csv", index=False)
    L = Xb.shape[1]
    inner = res[(res.layer >= 1) & (res.layer <= L - 2)]
    outcome = {}
    for name, g in inner.groupby("direction"):
        n = len(g)
        outcome[name] = {
            "inner_layers": n,
            "primary_verdict_layers": g.apply(
                lambda x: _verdict(x.A_f_lo, x.A_f_hi, x.A_m_lo, x.A_m_hi), axis=1
            )
            .value_counts()
            .to_dict(),
            "primary_verdict_layers_noen": g.apply(
                lambda x: _verdict(x.A_f_noen_lo, x.A_f_noen_hi, x.A_m_noen_lo, x.A_m_noen_hi),
                axis=1,
            )
            .value_counts()
            .to_dict(),
            "exc_follows_gender_layers": int((g.exc_auc_lo > 0.5).sum()),
            "exc_follows_spelling_layers": int((g.exc_auc_hi < 0.5).sum()),
            "homo_gender_beyond_spelling_layers": int((g.homo_auc_diff_lo > 0).sum()),
            "multi_reads_article_layers": int((g.multi_p_diff_lo > 0).sum()),
            "mean": g.drop(columns=["layer", "direction"]).mean().round(3).to_dict(),
        }
    counts = df.groupby("set").size().to_dict()
    summary = {
        "meta": {k: mb[k] for k in mb if k not in ("words", "tokens", "shape")},
        "analysis_git": git_state(),
        "n_items": counts,
        "n_multi": len(multi),
        "position": position,
        "primary_direction": PRIMARY,
        "outcome": outcome,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return res
