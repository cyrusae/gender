"""Readout-position check: the word's LAST token vs a newline token AFTER it.

Gender-blind by construction: no gender labels are read anywhere in this module. Pre-registered
criteria and decision rule: docs/decisions.md (2026-10-06).

  C1 word identity      each word in two neutral contexts (bare; after "Ejemplo:"/"Beispiel:"),
                        each context mean-centred; retrieve the word's colon-context vector among
                        words sharing its final token (groups >= 3) by cosine to its bare vector.
                        Top-1 accuracy, higher = better.
  C2 tokenisation       5-fold CV balanced accuracy of a logistic probe decoding token count
                        (1/2/3/4+) and final-token identity (10 most common), from bare vectors;
                        mean of the two, lower = better.
  C3 sink check         share of vectors > 10x the median norm (should be 0).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
)  # fmt: skip

COLON = {"es": "Ejemplo:", "de": "Beispiel:"}
SOURCES = ["data/stimuli/phase0_v3.csv", "data/stimuli/phase2_pool_v2.csv"]


def nouns() -> pd.DataFrame:
    """Unique (lang, lemma) from the stimulus lists. Gender columns are dropped on purpose."""
    df = pd.concat([pd.read_csv(p, keep_default_na=False)[["lang", "lemma"]] for p in SOURCES])
    return df.drop_duplicates().reset_index(drop=True)


def extract(model_id: str, device=None, dtype=None) -> None:
    df = nouns()
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    meta["input_format"] = acts.INPUT_FORMAT + " (+ newline for AFTER)"
    for lang, g in df.groupby("lang"):
        words = list(g.lemma)
        for ctx, texts in (("bare", words), ("colon", [f"{COLON[lang]} {w}" for w in words])):
            stage(f"{model_id}: {lang} {ctx} ({len(texts)} texts), LAST + AFTER")
            X_last, X_after, toks = acts.states_at(
                model, tok, texts, acts.AFTER, desc=f"{lang}-{ctx}"
            )
            for pos, X in (("last", X_last), ("after", X_after)):
                acts.save(model_id, f"readout_{lang}_{ctx}_{pos}", X, words, toks, meta)


def _centre(X):
    return X - X.mean(0, keepdims=True)


def _identity(A, B, groups) -> float:
    """Top-1 retrieval of each word's B vector among same-final-token words, by cosine to A."""
    A, B = _centre(A), _centre(B)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    hits, n = 0, 0
    for idx in groups:
        sims = A[idx] @ B[idx].T
        hits += int((sims.argmax(1) == np.arange(len(idx))).sum())
        n += len(idx)
    return hits / n if n else float("nan")


def _decode(X, labels) -> float:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    pred = cross_val_predict(pipe, X, labels, cv=cv)
    return float(balanced_accuracy_score(labels, pred))


def analyze(model_id: str, out_root: str = "results/readout") -> dict:
    df = nouns()
    rows = []
    for lang, g in df.groupby("lang"):
        _, meta = acts.load(model_id, f"readout_{lang}_bare_last")
        toks = meta["tokens"]
        final = [t[-1] for t in toks]
        ntok = np.array([min(len(t), 4) for t in toks])
        by_final: dict[str, list[int]] = {}
        for i, f in enumerate(final):
            by_final.setdefault(f, []).append(i)
        groups = [np.array(v) for v in by_final.values() if len(v) >= 3]
        top10 = {f for f, _ in Counter(final).most_common(10)}
        keep10 = np.array([f in top10 for f in final])
        lab10 = np.array(final)[keep10]
        X = {(c, p): acts.load(model_id, f"readout_{lang}_{c}_{p}")[0]
             for c in ("bare", "colon") for p in ("last", "after")}  # fmt: skip
        L = X[("bare", "last")].shape[1]
        stage(f"{model_id}: {lang} readout criteria over {L - 2} inner layers "
              f"({sum(len(x) for x in groups)} words in {len(groups)} final-token groups)")  # fmt: skip
        for layer in progress(range(1, L - 1), desc=f"{lang} layers", unit="layer"):
            for pos in ("last", "after"):
                A = X[("bare", pos)][:, layer].astype(np.float64)
                B = X[("colon", pos)][:, layer].astype(np.float64)
                norms = np.linalg.norm(np.r_[A, B], axis=1)
                rows.append({
                    "lang": lang, "layer": layer, "position": pos,
                    "c1_identity": _identity(A, B, groups),
                    "c2_ntok": _decode(A, ntok),
                    "c2_final": _decode(A[keep10], lab10),
                    "c3_sink_share": float(np.mean(norms > 10 * np.median(norms))),
                })  # fmt: skip
    res = pd.DataFrame(rows)
    res["c2"] = (res.c2_ntok + res.c2_final) / 2
    cells = {}
    for lang, g in res.groupby("lang"):
        m = g.groupby("position")[
            ["c1_identity", "c2", "c2_ntok", "c2_final", "c3_sink_share"]
        ].mean()
        ok = (m.loc["after", "c1_identity"] >= m.loc["last", "c1_identity"] - 0.05
              and m.loc["after", "c2"] < m.loc["last", "c2"])  # fmt: skip
        cells[lang] = {"means": m.round(3).to_dict(orient="index"), "after_meets_rule": bool(ok)}
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "layers.csv", index=False)
    summary = {"model": model_id, "analysis_git": git_state(), "cells": cells,
               "after_sink_clean": bool((res[res.position == "after"].c3_sink_share == 0).all())}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def decide(model_ids: list[str], out_root: str = "results/readout") -> str:
    """Apply the pre-registered rule across all model x language cells."""
    cells, clean = [], True
    for m in model_ids:
        s = json.loads((Path(out_root) / model_slug(m) / "summary.json").read_text())
        clean &= s["after_sink_clean"]
        cells += [c["after_meets_rule"] for c in s["cells"].values()]
    verdict = "AFTER" if clean and sum(cells) > len(cells) / 2 else "LAST"
    return f"{sum(cells)}/{len(cells)} cells favour AFTER; AFTER sink-clean: {clean} -> primary readout: {verdict}"
