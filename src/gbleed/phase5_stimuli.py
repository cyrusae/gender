"""Phase 5 stimuli (design: docs/design/phase5-design.md, P1-P22).

Built here, all fixed before any Phase 5 model output:
  adjectives  Glasgow-rated English adjectives (Scott et al. 2019): WordNet senses >= 50%
              adjective, Zipf >= 3, a single token in the model family's tokenizer (P16); with
              gender, valence, arousal, size ratings and frequency (the R1 covariates).
  nouns       steering nouns: Spanish = Phase 2 regular test nouns; German = a fresh seeded draw
              of known single-root m/f nouns (no Phase 3 list, flipped pair, classic, suffix_ctrl
              noun or compound head); plus the final flipped pairs and the classics (frozen).
  gate        Spanish -o/-a adjective pairs for *Mi {noun} es muy ___* (forms differing only in
              one final token); German article/pronoun candidates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GLASGOW = "data/raw/glasgow/glasgow_norms_scott2019.csv"
SCALES = ["AROU", "VAL", "DOM", "CNC", "IMAG", "FAM", "AOA", "SIZE", "GEND"]
FAMILIES = {"qwen3": "Qwen/Qwen3-4B-Base", "eurollm": "utter-project/EuroLLM-1.7B"}
NOUNS = "data/stimuli/phase5_nouns_v1.csv"
SEED = 0
N_DE_PER_GENDER = 50  # about 100 German steering nouns (design: "about 100 per language")

# Gate, Spanish: common descriptive adjectives with regular -o/-a agreement that can describe
# objects (fixed on grammatical grounds; the frame *Mi {noun} es muy ___* is the design's).
ES_GATE_ADJ = ["blanco", "negro", "nuevo", "viejo", "bonito", "feo", "alto", "bajo", "barato",
               "caro", "largo", "corto", "ancho", "limpio", "sucio", "rojo", "redondo", "lleno",
               "vacío", "seco", "frío", "pequeño", "duro", "blando", "oscuro", "claro", "lento",
               "rápido", "pesado", "ligero"]  # fmt: skip
DE_GATE = {  # P4: dictionary frame primary, pronoun frame secondary
    "dict": ("Wörterbuch:\n{noun},", {"m": " der", "f": " die", "n": " das"}),
    "pron": ("Thema: {noun}.", {"m": " Er", "f": " Sie", "n": " Es"}),
}


def glasgow() -> pd.DataFrame:
    raw = pd.read_csv(GLASGOW, header=[0, 1])
    g = pd.DataFrame({"word": raw.iloc[:, 0].astype(str)})
    for sc in SCALES:
        g[sc] = raw[(sc, "M")].astype(float)
    return g


def _adj_dominant(w: str, wn) -> bool:
    ss = wn.synsets(w)
    return bool(ss) and sum(s.pos() in "as" for s in ss) / len(ss) >= 0.5


def adjectives(family: str) -> pd.DataFrame:
    """The R1/R1-EN adjective set for a model family, with ratings and token ids."""
    from transformers import AutoTokenizer
    from wordfreq import zipf_frequency

    from .lexicon import _wordnet

    wn = _wordnet()
    tok = AutoTokenizer.from_pretrained(FAMILIES[family])
    g = glasgow()
    g = g[g.word.map(lambda w: _adj_dominant(w, wn))].copy()
    g["zipf"] = g.word.map(lambda w: zipf_frequency(w, "en"))
    g = g[g.zipf >= 3]
    ids = g.word.map(lambda w: tok(" " + w, add_special_tokens=False)["input_ids"])
    g = g[ids.map(len) == 1].copy()
    g["token_id"] = ids[ids.map(len) == 1].map(lambda x: x[0])
    assert g.token_id.is_unique
    out = f"data/stimuli/phase5_adjectives_{family}_v1.csv"
    g.to_csv(out, index=False)
    r = np.corrcoef(g.GEND, g.VAL)[0, 1]
    print(f"{family}: {len(g)} adjectives (gender-valence r = {r:.2f}) -> {out}")
    return g


def gate_adjectives(family: str) -> list[tuple[str, str, list[int], int, int]]:
    """(masculine, feminine, shared stem token ids, last token m, last token f) for gate
    adjectives whose two forms tokenize identically except for one final token each. The
    margin log P(f) - log P(m) then equals the difference at that one position (the stem's
    probability is shared and cancels), so each adjective needs one next-token read."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(FAMILIES[family])
    out = []
    for m in ES_GATE_ADJ:
        f = m[:-1] + "a"
        im = tok(" " + m, add_special_tokens=False)["input_ids"]
        i_f = tok(" " + f, add_special_tokens=False)["input_ids"]
        if len(im) == len(i_f) and im[:-1] == i_f[:-1] and im[-1] != i_f[-1]:
            out.append((m, f, im[:-1], im[-1], i_f[-1]))
    return out


def build_nouns() -> pd.DataFrame:
    """Steering nouns (both languages), final flipped pairs and classics."""
    from .phase3_stimuli import held_out_phase45
    from .phase4 import PAIRS_FINAL, training_lemmas

    p2 = pd.read_csv("data/stimuli/phase2_final_v4.csv", keep_default_na=False)
    es = p2[(p2.set == "regular") & (p2.split == "test")]
    rows = [{"lang": "es", "lemma": r.lemma, "gender": r.gender, "concept_en": r.concept_en,
             "set": "steer_es", "pair_id": ""} for r in es.itertuples()]  # fmt: skip

    pool = pd.read_csv("data/stimuli/phase3_pool_v4.csv", keep_default_na=False)
    fin = pd.read_csv("data/stimuli/phase3_final_v3.csv", keep_default_na=False)
    sc = pd.read_csv("data/stimuli/phase3_suffixctrl_pool_v1.csv", keep_default_na=False)
    heads = set(fin[fin.set.str.startswith("compound")]["head"])
    used = set(fin.lemma) | held_out_phase45() | set(sc.lemma) | heads
    known = None
    for m in ("1.7B", "4B"):
        s = pd.read_csv(f"results/phase3_known/Qwen__Qwen3-{m}-Base/scores.csv",
                        keep_default_na=False)  # fmt: skip
        k = set(s[s.status == "known"].lemma)
        known = k if known is None else known & k
    de = pool[(pool.set == "simplex") & (pool.en_overlap.astype(str) != "True")
              & (pool.sex_assoc == "") & pool.gender.isin(["m", "f"])
              & pool.lemma.isin(known) & ~pool.lemma.isin(used)]  # fmt: skip
    draw = pd.concat([de[de.gender == g].sample(N_DE_PER_GENDER, random_state=SEED)
                      for g in ("m", "f")])  # fmt: skip
    rows += [{"lang": "de", "lemma": r.lemma, "gender": r.gender, "concept_en": r.concept_en,
              "set": "steer_de", "pair_id": ""} for r in draw.itertuples()]  # fmt: skip

    pairs = pd.read_csv(PAIRS_FINAL, keep_default_na=False)
    for r in pairs.itertuples():
        rows.append({"lang": "de", "lemma": r.de_lemma, "gender": r.de_gender,
                     "concept_en": r.concept_en, "set": "pair", "pair_id": r.pair_id})  # fmt: skip
        rows.append({"lang": "es", "lemma": r.es_lemma, "gender": r.es_gender,
                     "concept_en": r.concept_en, "set": "pair", "pair_id": r.pair_id})  # fmt: skip
    cl = pd.read_csv("data/stimuli/classics.csv", keep_default_na=False)
    es_tr, de_tr = training_lemmas()
    cl = cl[~cl.lemma.isin(es_tr | de_tr)]  # llave (Phase 2 training leak) is dropped
    have = {(r["lang"], r["lemma"]) for r in rows}
    rows += [{"lang": r.lang, "lemma": r.lemma, "gender": r.gender, "concept_en": r.concept_en,
              "set": "classic", "pair_id": ""} for r in cl.itertuples()
             if (r.lang, r.lemma) not in have]  # fmt: skip
    df = pd.DataFrame(rows)
    assert not set(df[df.lang == "es"].lemma) & es_tr, "Spanish steering noun in training"
    assert not set(df[df.lang == "de"].lemma) & de_tr, "German steering noun in training"
    df["split"] = "test"
    df.to_csv(NOUNS, index=False)
    print(df.groupby(["set", "lang", "gender"]).size().unstack(fill_value=0).to_string())
    return df
