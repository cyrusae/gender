"""P27 (exploratory, PI 2026-10-12): German neuter diminutives (-chen/-lein), where grammatical
gender (always neuter) and sex come apart. Does the German f - m direction, fitted on inanimate
nouns, still sort *das Schwesterchen* from *das Brüderchen*?

Items (data/stimuli/german_diminutives_v1.csv):
  person  PI-screened (docs/reports/german-diminutives-review.csv): Wiktionary neuter -chen/-lein
          nouns; sex from the English gloss (or the base noun's) via WordNet person senses.
          Stem = the base noun; "clean" = the stem is not m/f (the -mädchen compounds; Mädchen
          itself is historically Magd + -chen, reported with them but flagged).
  object  automatic: Wiktionary-tagged neuter diminutives of inanimate lexicon nouns (Zipf >= 2),
          stem gender from the lexicon (Würstchen < die Wurst, Häkchen < der Haken).
Statistics per layer (directions DE / ES / POOLED as Phase 4, kind="dom"):
  persons      AUC female vs male persons (stem gender confounded with sex)
  objects      AUC feminine-stem vs masculine-stem objects (the stem effect, no sex)
  excess       persons - objects (sex beyond what the stem explains)
  clean        AUC clean female persons vs neuter-stem objects (sex with no gendered stem);
               also without Mädchen
Bootstrap: persons and objects resampled independently. Same bare-word input as Phases 2-3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import activations as acts
from .models import git_state, model_slug, progress

SHEET = "docs/reports/german-diminutives-review.csv"
ITEMS = "data/stimuli/german_diminutives_v1.csv"
KAIKKI_DE = "data/raw/kaikki/kaikki.org-dictionary-German.jsonl"
MIN_ZIPF = 2.0
SEED = 0
N_BOOT = 1000


def _objects() -> pd.DataFrame:
    from wordfreq import zipf_frequency

    lex = pd.read_csv("data/lexicon/de_nouns.csv", keep_default_na=False).drop_duplicates("lemma")
    lex = lex.set_index("lemma")
    rows = []
    with open(KAIKKI_DE) as f:
        for line in f:
            if '"pos": "noun"' not in line[:40]:
                continue
            d = json.loads(line)
            w = d.get("word", "")
            if d.get("lang_code") != "de" or not w.endswith(("chen", "lein")):
                continue
            for s in d.get("senses", []):
                tags = set(s.get("tags", []))
                if not s.get("glosses") or tags & {"obsolete", "archaic"}:
                    continue
                bases = [x.get("word") for x in (s.get("form_of") or [])]
                b = bases[0] if bases else None
                if ({"diminutive", "neuter"} <= tags and b in lex.index
                        and lex.animacy[b] == "inanimate" and lex.gender[b] in ("m", "f", "n")):  # fmt: skip
                    z = zipf_frequency(w, "de")
                    if z >= MIN_ZIPF:
                        rows.append({"word": w, "set": "object", "sex": "", "stem": b,
                                     "stem_gender": lex.gender[b], "clean": "",
                                     "english": lex.concept_en[b], "zipf": round(z, 2)})  # fmt: skip
                break
    return pd.DataFrame(rows).drop_duplicates("word")


def build() -> pd.DataFrame:
    from .phase4 import P2_FINAL, P3_FINAL

    s = pd.read_csv(SHEET, keep_default_na=False, dtype=str)
    s = s[s["keep (PI: y/n)"].str.strip() == "y"]
    per = pd.DataFrame({"word": s.word, "set": "person", "sex": s.sex, "stem": s.stem,
                        "stem_gender": s.stem_gender, "clean": s["clean (stem not m/f)"],
                        "english": s.english_gloss, "zipf": s.zipf})  # fmt: skip
    obj = _objects()
    obj = obj[~obj.word.isin(per.word)]
    df = pd.concat([per, obj], ignore_index=True)
    train = set(pd.read_csv(P3_FINAL, keep_default_na=False).lemma) | set(
        pd.read_csv(P2_FINAL, keep_default_na=False).lemma)  # fmt: skip
    assert not set(df.word) & train, set(df.word) & train  # held out from direction training
    df.to_csv(ITEMS, index=False)
    print(df.groupby(["set", "sex", "stem_gender"]).size().to_string())
    return df


def extract(model_id: str, device=None, dtype=None) -> None:
    from datetime import UTC, datetime

    from .models import load_model, pick_device, pick_dtype, run_metadata

    words = list(pd.read_csv(ITEMS, keep_default_na=False).word)
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    X, _, toks = acts.states_at(model, tok, words, acts.AFTER, desc="diminutives")
    acts.save(model_id, "de_diminutives", X, words, toks, {**meta, "position": "last"})


def analyze(model_id: str, out_root: str = "results/diminutives", n_boot: int = N_BOOT):
    from .estimators import auc
    from .phase4 import _fit, de_train, es_train, pooled

    df = pd.read_csv(ITEMS, keep_default_na=False, dtype=str)
    X, meta = acts.load(model_id, "de_diminutives")
    assert list(meta["words"]) == list(df.word)
    es, de = es_train(model_id, "last"), de_train(model_id, "last")
    per, obj = df.set == "person", df.set == "object"
    pf, pm = (per & (df.sex == "f")).to_numpy(), (per & (df.sex == "m")).to_numpy()
    of, om = (obj & (df.stem_gender == "f")).to_numpy(), (obj & (df.stem_gender == "m")).to_numpy()
    on = (obj & (df.stem_gender == "n")).to_numpy()
    clean = (per & (df.clean == "True")).to_numpy()
    clean4 = clean & (df.word != "Mädchen").to_numpy()
    groups = {"pf": pf, "pm": pm, "of": of, "om": om, "on": on, "clean": clean, "clean4": clean4}
    idx = {k: np.where(v)[0] for k, v in groups.items()}
    rng = np.random.default_rng(SEED)

    def stats(s, ix):
        a = lambda p, q: auc(np.r_[np.ones(len(p)), np.zeros(len(q))].astype(int),
                             np.r_[s[p], s[q]])  # fmt: skip
        out = {"persons": a(ix["pf"], ix["pm"]), "objects": a(ix["of"], ix["om"]),
               "clean_vs_neuter_objects": a(ix["clean"], ix["on"]),
               "clean4_vs_neuter_objects": a(ix["clean4"], ix["on"]),
               "male_persons_vs_neuter_objects": a(ix["pm"], ix["on"])}  # fmt: skip
        out["excess"] = out["persons"] - out["objects"]
        return out

    rows = []
    for layer in progress(range(1, X.shape[1] - 1), desc="diminutives", unit="layer"):
        Xe, Xd = es["X"][:, layer].astype(np.float64), de["X"][:, layer].astype(np.float64)
        Xl = X[:, layer].astype(np.float64)
        dirs = {"de": _fit(Xd, de, kind="dom"), "es": _fit(Xe, es, kind="dom"),
                "pooled": _fit(np.r_[Xe, Xd], pooled(es, de), kind="dom")}  # fmt: skip
        r = {"layer": layer}
        for nm, v in dirs.items():
            s = Xl @ (v / np.linalg.norm(v))
            point = stats(s, idx)
            boots = {k: [] for k in point}
            for _ in range(n_boot):
                ib = {k: rng.choice(v_, len(v_)) for k, v_ in idx.items()}
                for k, x in stats(s, ib).items():
                    boots[k].append(x)
            for k, x in point.items():
                lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
                r |= {f"{nm}_{k}": x, f"{nm}_{k}_lo": float(lo), f"{nm}_{k}_hi": float(hi)}
        rows.append(r)
    res = pd.DataFrame(rows)
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "layers.csv", index=False)
    keys = [c for c in res.columns if c != "layer" and not c.endswith(("_lo", "_hi"))]
    summ = {"model": model_id, "analysis_git": git_state(),
            "n": {k: int(v.sum()) for k, v in groups.items()},
            "mean": res[keys].mean().round(3).to_dict(),
            "layers_lo_above": {k: int((res[f"{k}_lo"] > (0 if k.endswith("excess") else 0.5)).sum())
                                for k in keys}}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k != "analysis_git"}, indent=1))
    return summ
