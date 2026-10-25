"""German rarity-lean held-out test (docs/design/rarity-lean-heldout-prereg.md).

Stage 1: freeze the held-out simplex list H (`build`). Stage 2: extract H at LAST on the same
device/precision as the training activations (`extract`, pod). Stage 3: the within-gender rank
test (`analyze`, `reading`, `main`; CPU).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .frequency_axis import complexity_rule
from .models import model_slug
from .phase2_stimuli import _clean
from .phase3_known import FINAL, KNOWN_MODELS
from .phase3_stimuli import POOL

HELDOUT = "data/stimuli/rarity_heldout_v1.csv"


def build(known_root: str = "results/phase3_known", out: str = HELDOUT) -> pd.DataFrame:
    """Known (Phase 3 filter), clean, unused, non-complex simplex nouns; m/f/n only."""
    pool = pd.read_csv(POOL, keep_default_na=False)
    pool["en_overlap"] = pool.en_overlap.astype(str).eq("True")
    pool["zipf"] = pd.to_numeric(pool.zipf, errors="coerce")
    known = None
    for m in KNOWN_MODELS:
        s = pd.read_csv(Path(known_root) / model_slug(m) / "scores.csv", keep_default_na=False)
        k = set(s[s.status == "known"].lemma)
        known = k if known is None else known & k
    used = set(pd.read_csv(FINAL, keep_default_na=False).lemma)
    h = _clean(pool[(pool.set == "simplex") & pool.lemma.isin(known)])
    h = h[~h.lemma.isin(used) & h.gender.isin(["m", "f", "n"])]
    why = complexity_rule()
    h = h[h.lemma.map(why) == ""]
    h = h.sort_values(["gender", "lemma"]).reset_index(drop=True)
    h.to_csv(out, index=False)
    digest = hashlib.sha256(Path(out).read_bytes()).hexdigest()
    print(
        h.gender.value_counts().to_dict(),
        "-ie f:",
        int(((h.gender == "f") & (h.ending == "ie")).sum()),
    )
    print(f"{out} sha256 {digest}")
    return h


# ---- extraction (GPU; same device and precision as the training activations: CUDA bf16) --------
ACTS = "rarity_heldout"


def extract(model_id: str, device=None, dtype=None) -> None:
    """Bare held-out nouns at LAST, the Phase 3 input format."""
    from datetime import UTC, datetime

    from . import activations as acts
    from .models import load_model, pick_device, pick_dtype, run_metadata

    words = list(pd.read_csv(HELDOUT, keep_default_na=False).lemma)
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta |= {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
             "input_format": acts.INPUT_FORMAT, "stimuli": HELDOUT, "position": "last"}  # fmt: skip
    X, toks = acts.last_token_states(model, tok, words, desc="rarity")
    acts.save(model_id, ACTS, X, words, toks, meta)


# ---- analysis (CPU) ---------------------------------------------------------------------
MODELS = ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-14B-Base")
N_PERM = 2000
FLOOR = 0.1
ALPHA = 0.025  # one-sided, predicted direction (rho < 0)
MIN_M = 100
AUC_CHANCE = 0.6  # decoding sanity check: AUC below this at a majority of layers = no valid test
SEED = 23
OUT = Path("results/rarity_heldout")


def directions(model_id: str):
    """{layer: (unit f−m coefficient, unit German number direction, training-midpoint threshold
    on the f−m projection)} at the four Phase 5 layers; the full-sample training fit used by
    confidence control (c) and frequency-axis A–D."""
    import numpy as np

    from .confidence_ctl import _number_pairs
    from .estimators import nuisance
    from .phase4 import de_train
    from .phase5 import LAYER_FRACS

    d = de_train(model_id, "last")
    y = d["y"].astype(float)
    M = np.column_stack([y, nuisance(d["cells"], d["covs"])])
    _, Xp, Xs = _number_pairs(model_id, "de")
    n_layers = d["X"].shape[1] - 1
    out = {}
    for k in sorted({max(1, round(f * n_layers)) for f in LAYER_FRACS}):
        X = d["X"][:, k].astype(np.float64)
        g = (np.linalg.pinv(M) @ X)[0]
        g /= np.linalg.norm(g)
        num = (Xp[:, k] - Xs[:, k]).astype(np.float64).mean(0)
        num /= np.linalg.norm(num)
        pr = X @ g
        thr = (pr[y == 1].mean() + pr[y == 0].mean()) / 2
        out[k] = (g, num, thr)
    return out


def partial_spearman(p, z, ntok, ending) -> float:
    """Spearman(p, z) controlling for token count and ending (fixed effects): ranks residualised
    on [1, token count, ending dummies], then correlated."""
    import numpy as np
    from scipy.stats import rankdata

    ending = np.asarray(ending)
    Z = np.column_stack([np.ones(len(p)), np.asarray(ntok, float),
                         *[(ending == e).astype(float) for e in sorted(set(ending))[1:]]])  # fmt: skip
    H = Z @ np.linalg.pinv(Z)
    rp, rz = rankdata(p), rankdata(z)
    ep, ez = rp - H @ rp, rz - H @ rz
    den = np.sqrt((ep @ ep) * (ez @ ez))
    return float(ep @ ez / den) if den > 0 else float("nan")


def rho_test(p, z, ntok, ending, rng, n_perm: int = N_PERM) -> dict:
    """Observed rho, permutation null (Zipf shuffled within ending cells), pass = below the
    null's ALPHA quantile and rho <= -FLOOR."""
    import numpy as np

    ending = np.asarray(ending)
    z = np.asarray(z, float)
    rho = partial_spearman(p, z, ntok, ending)
    cells = [np.flatnonzero(ending == e) for e in set(ending)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        zp = z.copy()
        for c in cells:
            zp[c] = z[rng.permutation(c)]
        null[i] = partial_spearman(p, zp, ntok, ending)
    q = float(np.quantile(null, ALPHA))
    return {"rho": rho, "null_q025": q, "n": len(p),
            "passes": bool(rho < q and rho <= -FLOOR)}  # fmt: skip


def analyze(model_id: str) -> pd.DataFrame:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from . import activations as acts

    h = pd.read_csv(HELDOUT, keep_default_na=False)
    X, meta = acts.load(model_id, ACTS)
    assert list(meta["words"]) == list(h.lemma)
    ntok = np.array([len(t) for t in meta["tokens"]], float)
    zipf = pd.to_numeric(h.zipf).to_numpy()
    rng = np.random.default_rng(SEED)
    groups = {"m": h.gender == "m", "f": h.gender == "f",
              "f_no_ie": (h.gender == "f") & (h.ending != "ie"), "n": h.gender == "n"}  # fmt: skip
    mf = h.gender.isin(["m", "f"]).to_numpy()
    rows = []
    for k, (g, num, thr) in directions(model_id).items():
        x = X[:, k].astype(np.float64)
        pg, pn = x @ g, x @ num
        yf = (h.gender[mf] == "f").to_numpy()
        base = {"model": model_id, "layer": k, "auc_mf": float(roc_auc_score(yf, pg[mf])),
                "acc_mf_midpoint": float(np.mean((pg[mf] > thr) == yf))}  # fmt: skip
        for dname, p in (("gender", pg), ("number", pn)):
            for gname, sel in groups.items():
                s = sel.to_numpy()
                r = rho_test(p[s], zipf[s], ntok[s], h.ending[s].to_numpy(), rng)
                rows.append({**base, "direction": dname, "group": gname, **r})
        print(model_id, k, base, flush=True)
    return pd.DataFrame(rows)


def reading(t: pd.DataFrame) -> dict:
    """Prereg readings for one model."""
    n_m = int(t[(t.group == "m")].n.iloc[0])
    gd = t[t.direction == "gender"].pivot_table(index="layer", columns="group", values="passes")
    nm = t[t.direction == "number"].pivot_table(index="layer", columns="group", values="passes")
    auc = t.drop_duplicates("layer").set_index("layer").auc_mf
    maj = len(gd) // 2 + 1
    both = int((gd.m.astype(bool) & gd.f.astype(bool)).sum())
    out = {"layers": len(gd), "n_m": n_m, "layers_m_pass": int(gd.m.sum()),
           "layers_f_pass": int(gd.f.sum()), "layers_both_pass": both,
           "layers_f_no_ie_pass": int(gd.f_no_ie.sum()),
           "layers_number_both_pass": int((nm.m.astype(bool) & nm.f.astype(bool)).sum()),
           "layers_auc_below_chance_bar": int((auc < AUC_CHANCE).sum())}  # fmt: skip
    if n_m < MIN_M or out["layers_auc_below_chance_bar"] >= maj:
        v = "no valid test"
    elif both >= maj and out["layers_number_both_pass"] < maj:
        v = "lean generalises" if out["layers_f_no_ie_pass"] >= maj else "mixed"
    elif out["layers_m_pass"] < maj and out["layers_f_pass"] < maj:
        v = "training-set property"
    else:
        v = "mixed"
    return out | {"reading": v}


def main(models=MODELS) -> None:
    import json

    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.concat(analyze(m) for m in models)
    t.to_csv(OUT / "layers.csv", index=False)
    res = {m: reading(t[t.model == m]) for m in models}
    (OUT / "readings.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
