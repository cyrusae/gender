"""Gemma slim prereg: frozen lists, known-check as a filter only.

For models outside the original list-building family (Gemma), analyses keep only the nouns the
model passes on each list's own known-check (Spanish: the Phase 0 rule as run for Phase 2; German:
`phase3_known`, same 1-nat margin, same frames). Qwen lists were built from Qwen known-checks, so
for Qwen models the filter is the identity and outputs are unchanged.

A list that loses more than 30% of either gender is "not portable" (`portable`), and the analysis
on it is reported as such rather than run on a lopsided set.
"""

import json
from pathlib import Path

import pandas as pd

from .models import model_slug

FILTERED_PREFIXES = ("google/gemma",)
# instruct replication (docs/design/instruct-replication-prereg.md): the lists were built on the
# Qwen3 base models, so their instruct siblings are filtered like Gemma
FILTERED_MODELS = {"Qwen/Qwen3-8B", "Qwen/Qwen3-14B"}
ROOTS = {"es": "results/phase2_known", "de": "results/phase3_known"}
MAX_LOSS = 0.30


def applies(model_id: str) -> bool:
    return model_id.startswith(FILTERED_PREFIXES) or model_id in FILTERED_MODELS


def known_lemmas(model_id: str, lang: str) -> set[str] | None:
    """Lemmas the model knows on the list's known-check; None = no filter (Qwen)."""
    if not applies(model_id):
        return None
    p = Path(ROOTS[lang]) / model_slug(model_id)
    if lang == "es":
        it = pd.read_csv(p / "items.csv", keep_default_na=False)
        return set(it[(it.lang == "es") & (it.passed.astype(str) == "True")].lemma)
    s = pd.read_csv(p / "scores.csv", keep_default_na=False)
    return set(s[s.status == "known"].lemma)


PAIR_ROOTS = {"es": "results/phase4_known_es", "de": "results/phase4_known_de"}


def pair_mask(model_id: str, pairs: pd.DataFrame):
    """Pairs whose Spanish and German nouns the model both knows on the Phase 4 pair checks (the
    rule `phase4.finalize` applied to Qwen3 1.7B/4B); None = no filter (Qwen)."""
    if not applies(model_id):
        return None
    slug = model_slug(model_id)
    de = pd.read_csv(Path(PAIR_ROOTS["de"]) / slug / "scores.csv", keep_default_na=False)
    es = pd.read_csv(Path(PAIR_ROOTS["es"]) / slug / "items.csv", keep_default_na=False)
    kde = set(de[de.status == "known"].lemma)
    kes = set(es[es.passed.astype(str) == "True"].lemma)
    return (pairs.de_lemma.isin(kde) & pairs.es_lemma.isin(kes)).to_numpy()


def phase5_lemmas(model_id: str) -> dict[str, set[str]] | None:
    """Phase 5 nouns kept per language: each set filtered by its own list's check (steer_es by
    the Phase 2 check, steer_de by `phase3_known`, pair nouns only if both of the pair are known).
    Classics had no known-check in the Qwen pipeline and are kept. None = no filter (Qwen)."""
    if not applies(model_id):
        return None
    from .phase4 import PAIRS_FINAL

    n = pd.read_csv("data/stimuli/phase5_nouns_v1.csv", keep_default_na=False)
    pairs = pd.read_csv(PAIRS_FINAL, keep_default_na=False)
    kept = pairs[pair_mask(model_id, pairs)]
    out = {}
    for lang in ("es", "de"):
        d = n[n.lang == lang]
        k = known_lemmas(model_id, lang)
        keep = set(d[d.set == "classic"].lemma) | set(
            d[(d.set == f"steer_{lang}") & d.lemma.isin(k)].lemma
        )
        keep |= set(kept[f"{lang}_lemma"])
        out[lang] = keep
    return out


def loss_by_gender(model_id: str, lang: str, lemmas, genders) -> dict[str, float]:
    k = known_lemmas(model_id, lang)
    df = pd.DataFrame({"lemma": list(lemmas), "g": list(genders)})
    if k is None:
        return {g: 0.0 for g in sorted(set(df.g))}
    return {g: round(float(1 - sub.lemma.isin(k).mean()), 3) for g, sub in df.groupby("g")}


def portable(loss: dict[str, float]) -> bool:
    return all(v <= MAX_LOSS for v in loss.values())


def report(model_ids, out: str = "results/gemma_known/portability.json") -> dict:
    """Per model and training list: share of each gender lost to the known filter."""
    from .phase4 import P2_FINAL, P3_FINAL

    p2 = pd.read_csv(P2_FINAL, keep_default_na=False)
    p2 = p2[p2.set.isin(["strat", "matched"])]
    p3 = pd.read_csv(P3_FINAL, keep_default_na=False)
    p3 = p3[(p3.set == "strat3") & p3.gender.isin(["m", "f"])]
    res = {}
    for m in model_ids:
        r = {}
        for lang, d in (("es", p2), ("de", p3)):
            loss = loss_by_gender(m, lang, d.lemma, d.gender)
            r[lang] = {"loss": loss, "portable": portable(loss)}
        res[m] = r
        print(m, r, flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    return res
