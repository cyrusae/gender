"""P22b: the model's own woman/man lean of adjectives, scored in context with whole words, validated
against Glasgow (exploratory; design: phase5-design.md P22b, PI 2026-10-12). Fixes P22's
multi-token problem: P22 reads one output vector per word, so most Spanish/German translations
were scored on their first token only.

Score (simple difference) = mean over frame pairs of
    log P(adj | feminine person frame) - log P(adj | masculine person frame)
(full word, all its tokens; the adjective's own frequency cancels). Frames: woman/man and
mother/father in each language, predicative ("is very ___"). Sets: English adjectives (as P22),
German translations (predicative adjectives don't inflect), Spanish gender-invariant translations.

Spanish gendered translations (blanco/blanca) can't use the simple difference: the two sides score
different strings, so the form's own prior (and anything grammatical gender does to it) would
enter. Double difference: subtract the same gap after the Phase 5 Spanish steering nouns
(inanimate, known; feminine nouns with the feminine form, masculine with the masculine), in the
same frame shapes. What's left is woman vs man beyond what any feminine vs masculine noun does: a
conservative check (any grammatical-gender lean on adjectives is subtracted too). The double
difference is also computed on the invariant set, to compare the two scores on the same words.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

GENDERED = "data/stimuli/p22b_es_gendered_v1.csv"
KAIKKI_ES = "data/raw/kaikki/kaikki.org-dictionary-Spanish.jsonl"

# (feminine frame, masculine frame) pairs, fixed 2026-10-12 before any P22b output
FRAMES = {
    "en": [("The woman is very", "The man is very"), ("My mother is very", "My father is very")],
    "es": [("La mujer es muy", "El hombre es muy"), ("Mi madre es muy", "Mi padre es muy")],
    "de": [("Die Frau ist sehr", "Der Mann ist sehr"),
           ("Meine Mutter ist sehr", "Mein Vater ist sehr")],
}  # fmt: skip
# inanimate controls in the same frame shapes as the Spanish person frames (article, possessive)
CONTROL = [("La {N} es muy", "El {N} es muy"), ("Mi {N} es muy", "Mi {N} es muy")]


def build_gendered() -> pd.DataFrame:
    """Spanish gendered translations of the adjective list with their single Wiktionary feminine
    form (`forms` entry tagged exactly ["feminine"]); adjectives without exactly one are dropped."""
    from .phase5_stimuli import ADJ_TRANS, ES_ADJ_PI_DROP

    t = pd.read_csv(ADJ_TRANS, keep_default_na=False)
    t = t[(t.es != "") & (t.es_invariant.astype(str) != "True") & ~t.es.isin(ES_ADJ_PI_DROP)]
    want, fem = set(t.es), {}
    with open(KAIKKI_ES) as f:
        for line in f:
            d = json.loads(line)
            if d.get("pos") != "adj" or d.get("word") not in want:
                continue
            for fm in d.get("forms", []):
                if fm.get("tags") == ["feminine"]:
                    fem.setdefault(d["word"], set()).add(fm["form"])
    one = {w: next(iter(v)) for w, v in fem.items() if len(v) == 1}
    # a Spanish word glossing two English words has no single rating (salado: expensive, salty)
    t = t[t.es.isin(one) & ~t.es.duplicated(keep=False)]
    out = pd.DataFrame({"en": t.en, "es_m": t.es, "es_f": t.es.map(one), "GEND": t.GEND,
                        "VAL": t.VAL, "source": "kaikki es forms [feminine]"})  # fmt: skip
    out.to_csv(GENDERED, index=False)
    print(f"{len(out)} gendered Spanish adjectives ({len(want) - len(one)} without one form)")
    return out


def _sets(fam: str) -> dict:
    """name -> (fem words, masc words, GEND, VAL)."""
    from .phase5_stimuli import ADJ_TRANS, es_invariant_adjectives

    adj = pd.read_csv(f"data/stimuli/phase5_adjectives_{fam}_v1.csv", keep_default_na=False)
    tr = pd.read_csv(ADJ_TRANS, keep_default_na=False)
    de = tr[tr.de != ""]
    inv = es_invariant_adjectives()
    gen = pd.read_csv(GENDERED, keep_default_na=False)
    return {"en": (adj.word, adj.word, adj.GEND, adj.VAL),
            "de": (de.de, de.de, de.GEND, de.VAL),
            "es_invariant": (inv.es, inv.es, inv.GEND, inv.VAL),
            "es_gendered": (gen.es_f, gen.es_m, gen.GEND, gen.VAL)}  # fmt: skip


def _rho(s, gend, val) -> dict:
    from .p22 import _partial

    g, v = -pd.to_numeric(gend).to_numpy(), pd.to_numeric(val).to_numpy()  # > 0 = feminine
    return {"n": len(s), "rho": float(pd.Series(s).corr(pd.Series(g), "spearman")),
            "rho_partial_val": _partial(np.asarray(s), g, v)}  # fmt: skip


def run(model_id: str, out_root: str = "results/p22b", device=None, dtype=None) -> dict:
    from .models import load_model, model_slug, pick_device, pick_dtype, run_metadata
    from .phase5 import SMOKE, family
    from .phase5_stimuli import NOUNS
    from .scoring import continuation_logprobs_batch

    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    sets = _sets(family(model_id))
    if SMOKE:
        sets = {k: tuple(x.head(3) for x in v) for k, v in sets.items()}
    nouns = pd.read_csv(NOUNS, keep_default_na=False)
    nouns = nouns[nouns.set == "steer_es"]
    if SMOKE:
        nouns = nouns.groupby("gender").head(2)
    fN, mN = list(nouns[nouns.gender == "f"].lemma), list(nouns[nouns.gender == "m"].lemma)

    def lp(prompts_words):
        return np.array(continuation_logprobs_batch(
            model, tok, [(p, " " + w) for p, w in prompts_words], desc="P22b"))  # fmt: skip

    out, rows = {"meta": meta, "frames": FRAMES, "validation": {}}, []
    for nm, (fw, mw, gend, val) in sets.items():
        fw, mw = list(fw), list(mw)
        lang = nm[:2]
        simple = np.zeros(len(fw))
        for ff, mf in FRAMES[lang]:
            simple += lp([(ff, w) for w in fw]) - lp([(mf, w) for w in mw])
        simple /= len(FRAMES[lang])
        res = {"simple": _rho(simple, gend, val)}
        scores = {"simple": simple}
        if lang == "es":
            ctrl = np.zeros(len(fw))
            for fc, mc in CONTROL:
                cf = lp([(fc.format(N=n), w) for w in fw for n in fN]).reshape(len(fw), len(fN))
                cm = lp([(mc.format(N=n), w) for w in mw for n in mN]).reshape(len(mw), len(mN))
                ctrl += cf.mean(1) - cm.mean(1)
            double = simple - ctrl / len(CONTROL)
            res["double"] = _rho(double, gend, val)
            scores["double"] = double
        out["validation"][nm] = res
        for i in range(len(fw)):
            rows.append({"set": nm, "fem": fw[i], "masc": mw[i],
                         **{k: float(v[i]) for k, v in scores.items()}})  # fmt: skip
    d = Path(out_root) / model_slug(model_id)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "scores.csv", index=False)
    (d / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["validation"], indent=1))
    return out
