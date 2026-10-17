"""P29: is the R3-POOL pooled adjective readout agreement-neutral? (PI 2026-10-17;
`docs/design/pooling-neutrality-prereg.md`)

Unsteered. Nonce nouns (the R-NONCE Spanish set) behind a masculine or feminine article, so that
only agreement differs between the two prompts of a word; read at the adjective slot after *muy*:
  - the R3 read ids (R3 proper's invariant adjectives, then the R3-POOL lemma ids), next token;
  - the gate adjectives' -o/-a endings (stem appended), for the manipulation check M0;
  - both whole forms of every pooled lemma (primary words only), for the fidelity check M3.
Analysis: M0 (article flips agreement), M1 (R3-POOL graded score, La minus El), M2 (the same on
the invariant adjectives), M3 (first-token vs whole-word pooling).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import model_slug

OUT_ROOT = "results/p29"
FRAMES = {  # wording: (template, article by gender); the P20 Spanish wordings
    "W1": ("{art} {N} es muy", {"m": "El", "f": "La"}),
    "W2": ("Es {art} {N} muy", {"m": "un", "f": "una"}),
}
PRIMARY_ENDINGS = ("e", "iz")  # endings that don't settle gender: the article does
# M1 pass bound: half of the smallest noun-only R3-POOL effect (subset "all") on file
# (results/runpod2/phase5_analysis/*/r3_pool.csv), fixed 2026-10-17 before any P29 data
R3_POOL_MIN_EFFECT = {"Qwen/Qwen3-8B-Base": 0.052016, "Qwen/Qwen3-14B-Base": 0.037602}
M0_MIN_FRAC = 0.90
SEED = 29
N_BOOT = 2000


def words() -> pd.DataFrame:
    from .phase5 import SMOKE, nonce_words

    nw = nonce_words()
    nw = nw[nw.set == "es"].reset_index(drop=True)
    nw["primary"] = nw.ending.isin(PRIMARY_ENDINGS)
    if SMOKE:
        nw = nw[nw.stem.isin(nw.stem.unique()[:4])].reset_index(drop=True)
    return nw


def prompt_keys(nw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in nw.itertuples():
        for w, (t, arts) in FRAMES.items():
            for g, art in arts.items():
                rows.append({"word": r.word, "stem": r.stem, "ending": r.ending,
                             "primary": r.primary, "wording": w, "gender": g,
                             "prompt": t.format(art=art, N=r.word)})  # fmt: skip
    return pd.DataFrame(rows)


@torch.no_grad()
def run(model_id: str, out_root: str = OUT_ROOT, device=None, dtype=None, model_tok=None):
    from datetime import UTC, datetime

    from .models import load_model, pick_device, pick_dtype, run_metadata
    from .phase5 import Prompt, family, r3_ids
    from .phase5 import run as fwd
    from .phase5_stimuli import es_pooled, gate_adjectives
    from .scoring import continuation_logprobs_batch

    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = model_tok or load_model(model_id, dev, dt)
    fam = family(model_id)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    keys = prompt_keys(words())
    V = torch.zeros((1, model.config.hidden_size))
    # 1) next token after "muy": R3 ids (invariant, then pooled), unsteered (dose 0)
    read = r3_ids(tok, fam)
    ps = [Prompt(p, w) for p, w in zip(keys.prompt, keys.word, strict=True)]
    r = fwd(model, tok, ps, 0, [(i, 0, 0.0) for i in range(len(ps))], V, read)
    # 2) gate adjectives: stem appended, read the two final tokens; margin f - m per adjective
    ga = gate_adjectives(fam)
    gp = [Prompt(p, w, extra=list(stem))
          for p, w in zip(keys.prompt, keys.word, strict=True) for _, _, stem, _, _ in ga]  # fmt: skip
    gread = sorted({i for *_, im, i_f in ga for i in (im, i_f)})
    pos = {t: j for j, t in enumerate(gread)}
    gr = fwd(model, tok, gp, 0, [(i, 0, 0.0) for i in range(len(gp))], V, gread)["lp"]
    gm = np.array([gr[j, pos[ga[j % len(ga)][4]]] - gr[j, pos[ga[j % len(ga)][3]]]
                   for j in range(len(gp))]).reshape(len(keys), len(ga))  # fmt: skip
    keys["margin"] = gm.mean(1)
    # 3) whole forms of every pooled lemma (primary words): log P(" form" | prompt)
    pooled = es_pooled(fam)
    prim = keys[keys.primary]
    reqs, wk = [], []
    for pi, p in zip(prim.index, prim.prompt, strict=True):
        for j, (am, af) in enumerate(zip(pooled.es_m, pooled.es_f, strict=True)):
            for form, f in (("m", am), ("f", af)):
                reqs.append((p, " " + f))
                wk.append((pi, j, form))
    lpw = continuation_logprobs_batch(model, tok, reqs, desc="P29 whole words")
    whole = pd.DataFrame(wk, columns=["prompt_row", "lemma_idx", "form"])
    whole["lp"] = lpw
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    keys.drop(columns=["prompt"]).to_csv(out / "prompts.csv.gz", index=False)
    np.savez_compressed(out / "next.npz", lp=r["lp"], read=np.asarray(read))
    whole.to_csv(out / "whole.csv.gz", index=False)
    (out / "meta.json").write_text(json.dumps({"meta": meta, "n_prompts": len(keys),
                                               "n_gate_adj": len(ga), "n_whole": len(reqs)},
                                              indent=1, default=str))  # fmt: skip
    print(model_id, "P29 done:", len(keys), "prompts,", len(reqs), "whole-word requests")
    return keys


# ---- analysis (CPU) ------------------------------------------------------------------------
def _contrast(keys: pd.DataFrame, X: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Per word: feminine-article row minus masculine-article row, averaged over wordings."""
    k = keys.reset_index(drop=True)
    idx = k.pivot_table(index=["word", "wording"], columns="gender", values="row", aggfunc="first")
    d = X[idx["f"].to_numpy()] - X[idx["m"].to_numpy()]
    per = idx.reset_index()[["word", "wording"]]
    words_ = per.word.unique()
    wi = {w: i for i, w in enumerate(words_)}
    acc = np.zeros((len(words_), X.shape[1]))
    np.add.at(acc, per.word.map(wi).to_numpy(), d)
    acc /= per.groupby("word").size().reindex(words_).to_numpy()[:, None]
    meta = k.drop_duplicates("word").set_index("word").loc[words_, ["stem", "ending", "primary"]]
    return meta.reset_index(), acc


def _boot(score: np.ndarray, stems: np.ndarray, rng, n_boot: int = N_BOOT) -> tuple[float, float]:
    """95% CI of the mean, resampling stems (a stem's words stay together)."""
    us = np.unique(stems)
    groups = [np.flatnonzero(stems == s) for s in us]
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(us), len(us))
        bs.append(score[np.concatenate([groups[i] for i in pick])].mean())
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def _summ(score, meta, rng) -> dict:
    lo, hi = _boot(score, meta.stem.to_numpy(), rng)
    return {"A": float(score.mean()), "lo95": lo, "hi95": hi, "n_words": len(score)}


def analyze(model_id: str, out_root: str = OUT_ROOT) -> dict:
    from .phase5 import family, r3_n_invariant
    from .phase5_analysis import graded, pooled_ratings, ratings_for

    fam = family(model_id)
    d = Path(out_root) / model_slug(model_id)
    keys = pd.read_csv(d / "prompts.csv.gz", keep_default_na=False)
    keys["primary"] = keys.primary.astype(str) == "True"
    keys["row"] = np.arange(len(keys))
    z = np.load(d / "next.npz")
    lp, read = z["lp"], z["read"]
    n_inv = r3_n_invariant(fam)
    g = pooled_ratings(fam)
    col = {int(t): j for j, t in enumerate(read)}
    lem = np.stack([np.logaddexp.reduce(lp[:, [col[i] for i in ids]], axis=1) for ids in g.ids],
                   axis=1)  # fmt: skip
    res: dict = {"model": model_id}
    for set_name, sel in (("primary", keys.primary), ("o_a", ~keys.primary)):
        ks = keys[sel.to_numpy()]
        # M0: agreement margin flips with the article
        mg = ks.pivot_table(index="word", columns="gender", values="margin", aggfunc="mean")
        m0 = {"frac_f_above_m": float((mg.f > mg.m).mean()), "median_f": float(mg.f.median()),
              "median_m": float(mg.m.median())}  # fmt: skip
        m0["passed"] = bool(m0["frac_f_above_m"] >= M0_MIN_FRAC and m0["median_f"] > 0
                            and m0["median_m"] < 0)  # fmt: skip
        rng = np.random.default_rng(SEED)
        meta, dl = _contrast(ks, lem)
        out = {"M0": m0}
        for sub, keep in (("all", np.ones(len(g), bool)),
                          ("both_single", (g[f"kind_{fam}"] == "both_single").to_numpy()),
                          ("no_homograph", (g.homograph == "").to_numpy())):  # fmt: skip
            s, _ = graded(dl[:, keep], g[keep].reset_index(drop=True))
            out[f"M1_{sub}"] = _summ(s, meta, rng)
        for w in FRAMES:  # per wording (reported)
            mw, dw = _contrast(ks[ks.wording == w], lem)
            out[f"M1_all_{w}"] = _summ(graded(dw, g)[0], mw, rng)
        # M2: invariant adjectives, same contrast; their slot share (R3's frame-check quantity)
        rat = ratings_for("R3", fam, read)
        mi, di = _contrast(ks, lp[:, : len(rat)])
        out["M2_invariant"] = _summ(graded(di, rat)[0], mi, rng)
        rows = ks.row.to_numpy()
        out["M2_invariant_share_median"] = float(np.median(np.exp(lp[rows, :n_inv]).sum(1)))
        out["pooled_share_median"] = float(np.median(np.exp(lp[rows, n_inv:]).sum(1)))
        res[set_name] = out
    # M3: whole-word pooling on the primary words
    wh = pd.read_csv(d / "whole.csv.gz")
    piv = wh.pivot_table(index=["prompt_row", "lemma_idx"], columns="form", values="lp")
    wl = np.logaddexp(piv.m, piv.f).unstack("lemma_idx")  # [prompt rows, lemmas]
    W = np.full((len(keys), len(g)), np.nan)
    W[wl.index.to_numpy()] = wl.to_numpy()
    kp = keys[keys.primary]
    meta, dw = _contrast(kp, W)
    _, df_ = _contrast(kp, lem)
    sf = (g[f"kind_{fam}"] == "shared_first").to_numpy()
    from scipy.stats import spearmanr

    rho = spearmanr(dw[:, sf].mean(0), df_[:, sf].mean(0))
    rng = np.random.default_rng(SEED)
    res["primary"]["M3"] = {
        "rho_lemma_shift_first_vs_whole_shared_first": float(rho.statistic),
        "n_lemmas_shared_first": int(sf.sum()),
        "M1_all_whole_word": _summ(graded(dw, g)[0], meta, rng),
        "agreement_logodds_shift_median": float(np.median(_agree_shift(piv, kp))),
    }
    m0, m1 = res["primary"]["M0"], res["primary"]["M1_all"]
    bound = R3_POOL_MIN_EFFECT.get(model_id, np.nan) / 2
    res["bound"] = bound
    res["valid"] = m0["passed"]
    res["M1_passed"] = bool(m0["passed"] and m1["hi95"] < bound)
    (d / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def _agree_shift(piv: pd.DataFrame, kp: pd.DataFrame) -> np.ndarray:
    """Per (word, lemma): change of log P(f form) - log P(m form) from El to La (wordings
    averaged); M0's agreement flip seen on the pooled lemmas' own forms."""
    lo = (piv.f - piv.m).unstack("lemma_idx")
    X = np.full((kp.row.max() + 1, lo.shape[1]), np.nan)
    X[lo.index.to_numpy()] = lo.to_numpy()
    _, d = _contrast(kp, X)
    return d.ravel()
