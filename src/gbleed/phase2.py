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

from . import activations as acts
from .models import git_state, load_model, model_slug, pick_device, pick_dtype, run_metadata
from .phase1 import load_stimuli as load_phase1

POOL = "data/stimuli/phase2_pool_v2.csv"
MULTI = "data/stimuli/phase2_multi_v2.csv"
FINAL = "data/stimuli/phase2_final_v2.csv"
KNOWN_MODELS = ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]
N_BOOT = 1000
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
    df["split"] = "test"
    df.loc[df.set == "matched", "split"] = "train"
    for g in ("m", "f"):
        idx = df.index[(df.set == "regular") & (df.gender == g)].to_numpy()
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
def extract(model_id: str, device=None, dtype=None) -> None:
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
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["input_format"] = "[<|endoftext|>] + tokens(' ' + text); last token"
    for name, ws in texts.items():
        X, toks = acts.last_token_states(model, tok, ws)
        acts.save(model_id, name, X, ws, toks, {**meta, "stimuli": FINAL})
        print(f"{name}: {X.shape}")


# ---- analysis helpers ---------------------------------------------------------
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


def analyze(model_id: str, out_root: str = "results/phase2") -> pd.DataFrame:
    df, multi = _final(), pd.read_csv(MULTI, keep_default_na=False)
    Xb, mb = acts.load(model_id, "phase2_bare")
    Xhn, _ = acts.load(model_id, "phase2_homo_noun")
    Xhv, _ = acts.load(model_id, "phase2_homo_verb")
    Xmm, _ = acts.load(model_id, "phase2_multi_m")
    Xmf, _ = acts.load(model_id, "phase2_multi_f")
    Xv1, _ = acts.load(model_id, "phase1_verbs")
    Xn1, _ = acts.load(model_id, "phase1_nonce")
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
    lean = _leans(model_id, multi.lemma)
    rng = np.random.default_rng(SEED)

    rows = []
    for layer in range(Xb.shape[1]):
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
        directions = {
            "dom_matched": (ident, _dom(X[matched], y[matched])),
            "probe_matched": (ident, _probe_dir(X[matched], y[matched])),
            "dom_regular": (ident, _dom(X[reg_tr], y[reg_tr])),
            "dom_regular_verberase": (ev, _dom(ev(X[reg_tr]), y[reg_tr])),
            "dom_regular_rank2": (e2, _dom(e2(X[reg_tr]), y[reg_tr])),
        }
        for name, (T, d) in directions.items():
            s = T(X) @ d
            m0, m1 = s[reg_te & (y == 0)].mean(), s[reg_te & (y == 1)].mean()
            pidx = lambda v, m0=m0, m1=m1: (v - m0) / (m1 - m0)
            p = pidx(s)
            r = {"layer": layer, "direction": name}
            r["regular_test_auc"] = _auc(y[reg_te], s[reg_te])
            # exceptions: do feminine exceptions outrank masculine ones?
            ye, se = y[exc], s[exc]
            r["exc_auc"] = _auc(ye, se)
            r["exc_auc_lo"], r["exc_auc_hi"] = _boot(
                lambda i, ye=ye, se=se: _auc(ye[i], se[i]), len(ye), rng
            )
            for sub in ("exception_ma", "exception_clipping", "exception_true"):
                for g, gl in ((0, "m"), (1, "f")):
                    k = (s_ == sub) & (y == g)
                    if k.any():
                        r[f"p_{sub.removeprefix('exception_')}_{gl}"] = float(p[k].mean())
            # PRIMARY (amended before results): masculine -a exceptions vs regular nouns.
            #   A_f = P(regular fem -a noun scores above a masc -a exception): same ending,
            #         different gender -> high if the direction tracks gender, ~0.5 if spelling.
            #   A_m = P(masc -a exception scores above a regular masc -o noun): same gender,
            #         different ending -> ~0.5 if gender, high if spelling.
            for tag, keep_ex in (("", excm), ("_noen", excm & ~en)):
                rf, rm, xe = s[reg_te & (y == 1)], s[reg_te & (y == 0)], s[keep_ex]
                lab_f = np.r_[np.zeros(len(xe)), np.ones(len(rf))]
                lab_m = np.r_[np.zeros(len(rm)), np.ones(len(xe))]
                r[f"A_f{tag}"] = _auc(lab_f, np.r_[xe, rf])
                r[f"A_m{tag}"] = _auc(lab_m, np.r_[rm, xe])
                r[f"A_f{tag}_lo"], r[f"A_f{tag}_hi"] = _boot_two(xe, rf, rng)
                r[f"A_m{tag}_lo"], r[f"A_m{tag}_hi"] = _boot_two(rm, xe, rng)
            r["n_excm"], r["n_excm_noen"] = int(excm.sum()), int((excm & ~en).sum())
            for w in df.lemma[exc & (y == 1)]:  # the few feminine -o exceptions, item by item
                r[f"p_item_{w}"] = float(p[(df.lemma == w).to_numpy()][0])
            # homographs: same strings as noun vs verb
            sn, sv = T(hn) @ d, T(hv) @ d
            r["homo_auc_noun"], r["homo_auc_verb"] = _auc(homo_y, sn), _auc(homo_y, sv)
            r["homo_auc_diff"] = r["homo_auc_noun"] - r["homo_auc_verb"]
            r["homo_diff_lo"], r["homo_diff_hi"] = _boot(
                lambda i, sn=sn, sv=sv: _auc(homo_y[i], sn[i]) - _auc(homo_y[i], sv[i]),
                len(homo_y),
                rng,
            )
            # mar-type: la X vs el X
            dp = pidx(T(mf) @ d) - pidx(T(mm) @ d)
            r["multi_p_diff"] = float(dp.mean())
            r["multi_p_diff_lo"], r["multi_p_diff_hi"] = _boot(
                lambda i, dp=dp: dp[i].mean(), len(dp), rng
            )
            r["multi_p_diff_lean_m"] = (
                float(dp[lean > 0].mean()) if (lean > 0).any() else float("nan")
            )
            r["multi_p_diff_lean_f"] = (
                float(dp[lean < 0].mean()) if (lean < 0).any() else float("nan")
            )
            if name.endswith("_matched"):  # in-domain: CV grouped by ending
                aucs = []
                for a, b in GroupKFold(n_splits=5).split(X[matched], y[matched], endings[matched]):
                    fit = _dom if name == "dom_matched" else _probe_dir
                    dd = fit(X[matched][a], y[matched][a])
                    aucs.append(_auc(y[matched][b], X[matched][b] @ dd))
                r["matched_cv_auc"] = float(np.nanmean(aucs))
            rows.append(r)
        r0 = next(x for x in rows[-len(directions) :] if x["direction"] == "dom_matched")
        print(f"layer {layer:2d} dom_matched: A_f {r0['A_f']:.2f} [{r0['A_f_lo']:.2f},{r0['A_f_hi']:.2f}] "
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
            "homo_gender_beyond_spelling_layers": int((g.homo_diff_lo > 0).sum()),
            "multi_reads_article_layers": int((g.multi_p_diff_lo > 0).sum()),
            "mean": g.drop(columns=["layer", "direction"]).mean().round(3).to_dict(),
        }
    counts = df.groupby("set").size().to_dict()
    summary = {
        "meta": {k: mb[k] for k in mb if k not in ("words", "tokens", "shape")},
        "analysis_git": git_state(),
        "n_items": counts,
        "n_multi": len(multi),
        "outcome": outcome,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return res
