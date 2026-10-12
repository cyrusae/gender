"""Personification sensitivity check (exploratory, PI 2026-10-12): are the Spanish/German
grammatical directions carried by nouns the culture personifies (la muerte, der Tod)?

Source: Iconclass (CC0; github.com/iconclass/data, English texts in data/raw/iconclass), the
art-subject classification, which includes Ripa's Iconologia. A concept counts as personified if
an Iconclass entry is a personification of it ("personification(s) of X", "X personified") or a
Ripa allegory whose English heading names it. Training nouns are matched by their English concept
(lexicon concept_en); over-matching (polysemous concepts: lot, will, end) only makes the check
stricter. Segel & Boroditsky (2011) used 243 personified entities but didn't publish the list.

Check: refit ES/DE/POOLED (Phase 4 `_fit`, kind="dom") without the matched nouns; per layer
report the cosine with the original direction and the P24 person-word / name AUCs both ways.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .models import git_state, model_slug, progress

ICONCLASS = "data/raw/iconclass"  # txt_en_*.txt from github.com/iconclass/data (txt/en)
OUT = "data/stimuli/personified_v1.csv"
N_NULL = 200  # random drops of the same size and gender mix (Spanish)


def subjects() -> dict:
    """{English concept: [Iconclass notations]} for personifications and Ripa allegories."""
    subj = defaultdict(set)
    for f in sorted(Path(ICONCLASS).glob("txt_en_*.txt")):
        for line in f.read_text().splitlines():
            if "|" not in line:
                continue
            key, txt = line.split("|", 1)
            low = txt.lower()
            if "personif" not in low and "(ripa)" not in low:
                continue
            head = re.sub(r"'[^']*'", "", txt.split(";")[0]).split(" - ")[0].lower()
            head = re.sub(r"\([^)]*\)", "", head).strip(" ,.")
            m = re.search(r"personifications? of (?:the )?([a-z][a-z ]*)", head) or re.search(
                r"^([a-z][a-z ]*?) personified", head)  # fmt: skip
            s = m.group(1).strip() if m else (head if "personif" not in head else "")
            for w in re.split(r",| and | or ", s):
                if w.strip():
                    subj[w.strip()].add(key)
    return {k: sorted(v) for k, v in subj.items()}


def build() -> pd.DataFrame:
    """Training nouns (Phase 2 ES, Phase 3 DE) whose English concept Iconclass personifies."""
    from .phase4 import P2_FINAL, P3_FINAL

    subj = subjects()
    rows = []
    for lang, f in (("es", P2_FINAL), ("de", P3_FINAL)):
        df = pd.read_csv(f, keep_default_na=False).drop_duplicates("lemma")
        for r in df.itertuples():
            if r.concept_en in subj:
                rows.append({"lang": lang, "lemma": r.lemma, "gender": r.gender,
                             "concept_en": r.concept_en,
                             "iconclass": " ".join(subj[r.concept_en][:5])})  # fmt: skip
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"{len(subj)} personified concepts; matched:", out.groupby("lang").size().to_dict())
    return out


def _drop(d: dict, lemmas: set) -> dict:
    keep = ~np.isin(d["lemma"], list(lemmas))
    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
           for k, v in d.items()}  # fmt: skip
    out["covs"] = [c[keep] for c in d["covs"]]
    return out


def analyze(model_id: str, position: str = "last", out_root: str = "results/personified"):
    from . import activations as acts
    from .english import NAME_MIN_PER_SEX, WORDS
    from .estimators import auc
    from .phase4 import _fit, de_train, es_train, pooled, within_cell_auc

    pers = pd.read_csv(OUT, keep_default_na=False)
    es, de = es_train(model_id, position), de_train(model_id, position)
    es2 = _drop(es, set(pers[pers.lang == "es"].lemma))
    de2 = _drop(de, set(pers[pers.lang == "de"].lemma))
    sets = {"orig": (es, de), "no_personified": (es2, de2)}
    words = pd.read_csv(WORDS, keep_default_na=False)
    X, meta = acts.load(model_id, "english" + ("" if position == "last" else "_after"))
    row = {w: i for i, w in enumerate(meta["words"])}
    p2 = words[words.set == "person2"]
    nm = words[words.set == "name"]
    cnt = nm.groupby(["cell", "sex"]).size().unstack(fill_value=0)
    nm = nm[nm.cell.isin(cnt[(cnt.f >= NAME_MIN_PER_SEX) & (cnt.m >= NAME_MIN_PER_SEX)].index)]
    have = p2.word.isin(row).all() and nm.word.isin(row).all()
    p2i, p2y = np.array([row[w] for w in p2.word], int), (p2.sex == "f").to_numpy(int)
    ni, ny = np.array([row[w] for w in nm.word], int), (nm.sex == "f").to_numpy(int)
    ncell = nm.cell.to_numpy()
    es_drop = set(pers[pers.lang == "es"].lemma)
    rng = np.random.default_rng(0)
    rows = []
    for layer in progress(range(1, X.shape[1] - 1), desc=f"personified {position}", unit="layer"):
        r = {"layer": layer}
        Xl = X[:, layer].astype(np.float64)
        dirs = {}
        for tag, (e, d) in sets.items():
            Xe, Xd = e["X"][:, layer].astype(np.float64), d["X"][:, layer].astype(np.float64)
            for nm_, v in (("es", _fit(Xe, e, kind="dom")), ("de", _fit(Xd, d, kind="dom")),
                           ("pooled", _fit(np.r_[Xe, Xd], pooled(e, d), kind="dom"))):  # fmt: skip
                dirs[tag, nm_] = v / np.linalg.norm(v)
        # null: drop as many random training nouns of each gender (same language), refit
        null = {k: [] for k in ("cos_es", "es_persons2", "es_names")}
        Xe = es["X"][:, layer].astype(np.float64)
        for _ in range(N_NULL):
            pick = set()
            for g in (0, 1):
                k = int((np.isin(es["lemma"], list(es_drop)) & (es["y"] == g)).sum())
                pick |= set(rng.choice(es["lemma"][es["y"] == g], k, replace=False))
            e3 = _drop(es, pick)
            v = _fit(Xe[~np.isin(es["lemma"], list(pick))], e3, kind="dom")
            v = v / np.linalg.norm(v)
            null["cos_es"].append(float(dirs["orig", "es"] @ v))
            if have:
                null["es_persons2"].append(auc(p2y, Xl[p2i] @ v))
                null["es_names"].append(within_cell_auc(ny, Xl[ni] @ v, ncell))
        r["cos_es_null_p05"] = float(np.percentile(null["cos_es"], 5))
        if have:
            for k in ("es_persons2", "es_names"):
                r[f"{k}_null_p05"], r[f"{k}_null_p95"] = (float(np.percentile(null[k], 5)),
                                                          float(np.percentile(null[k], 95)))  # fmt: skip
        for nm_ in ("es", "de", "pooled"):
            r[f"cos_{nm_}"] = float(dirs["orig", nm_] @ dirs["no_personified", nm_])
            if have:
                for tag in sets:
                    v = dirs[tag, nm_]
                    r[f"{nm_}_persons2_{tag}"] = auc(p2y, Xl[p2i] @ v)
                    r[f"{nm_}_names_{tag}"] = within_cell_auc(ny, Xl[ni] @ v, ncell)
        rows.append(r)
    res = pd.DataFrame(rows)
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"layers_{position}.csv", index=False)
    summ = {"model": model_id, "position": position, "analysis_git": git_state(),
            "n_dropped": {"es": int(len(es["y"]) - len(es2["y"])),
                          "de": int(len(de["y"]) - len(de2["y"]))},
            "mean": res.drop(columns="layer").mean().round(4).to_dict(),
            "min_cos": {k: float(res[f"cos_{k}"].min()) for k in ("es", "de", "pooled")}}  # fmt: skip
    (out / f"summary_{position}.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k != "analysis_git"}, indent=1))
    return summ
