"""P30: does a feminine article pull toward feminine-rated adjectives when the noun is clearly an
object? (exploratory; `docs/design/p30-object-context-prereg.md`)

Unsteered. Nonce nouns (R-NONCE Spanish, all endings) and real common-gender person nouns behind
*El* / *La* in object, neutral and person contexts; read at the adjective slot after *muy*:
  - the R3 read ids (invariant adjectives = primary readout, then the R3-POOL lemma ids);
  - the manipulation probe on the same stem with *es* in place of *es muy*:
    log P(" una persona") - log P(" una cosa").
Analysis: C1 (contexts change the person/thing reading), C2 (combined share >= 5%), S(context)
on the invariant adjectives, verdict on S(object).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import model_slug
from .p29 import _boot, _contrast

OUT_ROOT = "results/p30"
PERSON_NOUNS = "data/stimuli/p30_person_nouns_v1.csv"
# PI review 2026-10-17: dedicated feminine forms exist (sacerdotisa, poetisa), colloquial, rare
P30_PERSON_DROP = {"sacerdote", "poeta", "nene", "falte"}
CONTEXTS = {  # wording: (context, template); fixed on grammatical grounds
    "O1": ("object", "{art} {N} que compré ayer es muy"),
    "O2": ("object", "{art} {N} que guardé en el cajón es muy"),
    "N1": ("neutral", "{art} {N} es muy"),
    "P1": ("person", "{art} {N} con quien hablé ayer es muy"),
    "P2": ("person", "{art} {N} que conocí ayer es muy"),
}
ARTS = {"m": "El", "f": "La"}
PROBE = {"person": " una persona", "thing": " una cosa"}
C1_MARGIN = 1.0  # nats
SHARE_MIN = 0.05
SEED = 30


def build_person_nouns() -> pd.DataFrame:
    """Common-gender Spanish person nouns, by rule from the Wiktionary lexicon (see the prereg),
    minus the PI drops. Written once to a versioned stimulus file."""
    d = pd.read_csv("data/lexicon/es_nouns.csv", keep_default_na=False)
    c = d[(d.gender == "multi") & (d.multi_type == "free_variation") & (d.wordnet == "noun.person")
          & (d.zipf >= 3.5) & (d.n_entries == 1) & (d.also_pos == "") & (d.es_ending != "o")
          & (d.lemma.str.len() >= 4) & ~d.lemma.str.endswith("s")]  # fmt: skip
    c = c[pd.to_numeric(c.zipf_en, errors="coerce").fillna(0) < 3]  # not an English loanword
    c = c[~c.lemma.isin(P30_PERSON_DROP)]
    out = c[["lemma", "concept_en", "zipf"]].sort_values("zipf", ascending=False)
    out["source"] = "wiktionary es_nouns rule (p30.build_person_nouns); PI drops 2026-10-17"
    out.to_csv(PERSON_NOUNS, index=False)
    return out


def prompt_keys() -> pd.DataFrame:
    from .p29 import words

    rows = []
    nw = words()
    for r in nw.itertuples():
        for w, (ctx, t) in CONTEXTS.items():
            for g, art in ARTS.items():
                rows.append({"kind": "nonce", "word": r.word, "stem": r.stem, "ending": r.ending,
                             "context": ctx, "wording": w, "gender": g,
                             "prompt": t.format(art=art, N=r.word)})  # fmt: skip
    pn = pd.read_csv(PERSON_NOUNS, keep_default_na=False)
    from .phase5 import SMOKE

    pn = pn.head(4) if SMOKE else pn
    for r in pn.itertuples():
        for w, (ctx, t) in CONTEXTS.items():
            if ctx == "object":  # "the president I bought": nonsense for real persons
                continue
            for g, art in ARTS.items():
                rows.append({"kind": "person", "word": r.lemma, "stem": r.lemma, "ending": "",
                             "context": ctx, "wording": w, "gender": g,
                             "prompt": t.format(art=art, N=r.lemma)})  # fmt: skip
    return pd.DataFrame(rows)


@torch.no_grad()
def run(model_id: str, out_root: str = OUT_ROOT, device=None, dtype=None, model_tok=None):
    from datetime import UTC, datetime

    from .models import load_model, pick_device, pick_dtype, run_metadata
    from .phase5 import Prompt, family, r3_ids
    from .phase5 import run as fwd
    from .scoring import continuation_logprobs_batch

    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = model_tok or load_model(model_id, dev, dt)
    fam = family(model_id)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    keys = prompt_keys()
    V = torch.zeros((1, model.config.hidden_size))
    read = r3_ids(tok, fam)
    ps = [Prompt(p, w) for p, w in zip(keys.prompt, keys.word, strict=True)]
    r = fwd(model, tok, ps, 0, [(i, 0, 0.0) for i in range(len(ps))], V, read)
    stems = [p.removesuffix(" muy") for p in keys.prompt]
    assert all(s.endswith(" es") for s in stems)
    reqs = [(s, c) for s in stems for c in PROBE.values()]
    lp = np.asarray(continuation_logprobs_batch(model, tok, reqs, desc="P30 probe"))
    keys["lp_person"], keys["lp_thing"] = lp[0::2], lp[1::2]
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    keys.drop(columns=["prompt"]).to_csv(out / "prompts.csv.gz", index=False)
    np.savez_compressed(out / "next.npz", lp=r["lp"], read=np.asarray(read))
    (out / "meta.json").write_text(json.dumps({"meta": meta, "n_prompts": len(keys)}, indent=1,
                                              default=str))  # fmt: skip
    print(model_id, "P30 scored:", len(keys), "prompts")
    return keys


def analyze(model_id: str, out_root: str = OUT_ROOT) -> dict:
    from .phase5 import family, r3_n_invariant
    from .phase5_analysis import graded, pooled_ratings, ratings_for

    fam = family(model_id)
    d = Path(out_root) / model_slug(model_id)
    keys = pd.read_csv(d / "prompts.csv.gz", keep_default_na=False)
    keys["row"] = np.arange(len(keys))
    keys["primary"] = keys.kind == "nonce"  # column p29._contrast carries along
    z = np.load(d / "next.npz")
    lp, read = z["lp"], z["read"]
    n_inv = r3_n_invariant(fam)
    rat = ratings_for("R3", fam, read)
    g = pooled_ratings(fam)
    col = {int(t): j for j, t in enumerate(read)}
    lem = np.stack([np.logaddexp.reduce(lp[:, [col[i] for i in ids]], axis=1) for ids in g.ids],
                   axis=1)  # fmt: skip
    keys["share"] = np.exp(lp).sum(1)  # every read token: invariant + pooled (C2)
    keys["probe"] = keys.lp_person - keys.lp_thing
    non = keys[keys.kind == "nonce"]
    res: dict = {"model": model_id, "wordings": {}}
    med = non.groupby("wording").probe.median()
    real_n1 = float(keys[(keys.kind == "person") & (keys.wording == "N1")].probe.median())
    for w, (ctx, _) in CONTEXTS.items():
        share = float(non[non.wording == w].share.median())
        c1 = {"object": med[w] < 0 and med[w] <= med["N1"] - C1_MARGIN,
              "neutral": True, "person": med[w] >= med["N1"] + C1_MARGIN}[ctx]  # fmt: skip
        res["wordings"][w] = {"context": ctx, "probe_median": float(med[w]), "share": share,
                              "C1": bool(c1), "C2": share >= SHARE_MIN,
                              "valid": bool(c1 and share >= SHARE_MIN)}  # fmt: skip
    res["probe_real_person_N1"] = real_n1
    ok = {w for w, v in res["wordings"].items() if v["valid"]}
    obj = [w for w in ("O1", "O2") if w in ok]
    res["valid"] = bool(obj and "N1" in ok and real_n1 > 0)
    # S(context) on the invariant adjectives (primary) and the pooled lemmas (secondary)
    rng = np.random.default_rng(SEED)
    scores = {}
    for ctx in ("object", "neutral", "person"):
        ws = [w for w, (c, _) in CONTEXTS.items() if c == ctx and w in ok]
        if not ws:
            continue
        ks = non[non.wording.isin(ws)]
        m, di = _contrast(ks, lp[:, : len(rat)])
        _, dl = _contrast(ks, lem)
        scores[ctx] = (m, graded(di, rat)[0])
        lo, hi = _boot(scores[ctx][1], m.stem.to_numpy(), rng)
        res[f"S_{ctx}"] = {"S": float(scores[ctx][1].mean()), "lo95": lo, "hi95": hi,
                           "wordings": ws, "n_words": len(m)}  # fmt: skip
        res[f"S_{ctx}_pooled"] = float(graded(dl, g)[0].mean())
        res[f"S_{ctx}_by_ending"] = {e: float(scores[ctx][1][(m.ending == e).to_numpy()].mean())
                                     for e in ("a", "o", "e", "iz")}  # fmt: skip
        pk = keys[(keys.kind == "person") & keys.wording.isin(ws)]
        if len(pk):
            mp, dp = _contrast(pk, lp[:, : len(rat)])
            sp = graded(dp, rat)[0]
            res[f"S_real_{ctx}"] = {"S": float(sp.mean()),
                                    **dict(zip(("lo95", "hi95"), _boot(sp, mp.stem.to_numpy(), rng),
                                               strict=True))}  # fmt: skip
    if "object" in scores and "neutral" in scores:  # ratio, same stems resampled for both
        (mo, so), (mn, sn) = scores["object"], scores["neutral"]
        assert (mo.word.to_numpy() == mn.word.to_numpy()).all()
        stems = mo.stem.to_numpy()
        us = np.unique(stems)
        grp = [np.flatnonzero(stems == s) for s in us]
        bs = []
        for _ in range(2000):
            idx = np.concatenate([grp[i] for i in rng.integers(0, len(us), len(us))])
            bs.append(so[idx].mean() / sn[idx].mean())
        res["ratio_object_neutral"] = {"ratio": float(so.mean() / sn.mean()),
                                       "lo95": float(np.quantile(bs, 0.025)),
                                       "hi95": float(np.quantile(bs, 0.975))}  # fmt: skip
    res["n_invariant"] = n_inv
    if res["valid"] and res["S_neutral"]["lo95"] <= 0:  # no P29 pull to explain (prereg rule)
        res["verdict"] = "no neutral pull to explain (descriptive only)"
    elif res["valid"]:
        s, sn = res["S_object"], res["S_neutral"]["S"]
        res["verdict"] = ("pull survives an object reading" if s["lo95"] > 0
                          else "mostly the 'a woman' guess" if s["hi95"] < 0.5 * sn
                          else "partial")  # fmt: skip
    else:
        res["verdict"] = "no valid test"
    (d / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res
