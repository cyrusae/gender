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
# German gate candidates (2026-10-09, PI), written with their grammatical reasons before any
# test. Readout after the noun; zero-article contexts (any article before the noun gives the
# gender away). Margin = logsumexp(feminine forms) - logsumexp(masculine forms) (m/f only).
#   C1   dictionary headword + comma -> article (die is also plural: the earlier failure)
#   C2   dative pronoun after "mit": ihm (m/n) / ihr (f); plural would be ihnen
#   C2p  C2 pooled with dative articles (mit dem/einem vs der/einer may start a new noun phrase)
#   C3   relative pronoun forced by "Noun, mit": dem (m/n) / der (f); plural would be denen
#   C4   C1 minus the margin of a content-free headword ("N/A"): contextual calibration
#        (Zhao et al. 2021)
# Selection (pre-declared): candidates passing >= 70% per gender on Qwen3 (1.7B, 4B) and
# EuroLLM-1.7B; the highest worst-case accuracy wins; ties: C2 > C3 > C1 > C4 > C2p.
DE_GATE_CANDIDATES = {
    "C1": ("Wörterbuch:\n{noun},", {"m": [" der"], "f": [" die"]}, None),
    "C2": ("Thema: {noun}. Was macht man mit", {"m": [" ihm"], "f": [" ihr"]}, None),
    "C2p": ("Thema: {noun}. Was macht man mit",
            {"m": [" ihm", " dem", " einem"], "f": [" ihr", " der", " einer"]}, None),
    "C3": ("Wörterbuch:\n{noun}, mit", {"m": [" dem"], "f": [" der"]}, None),
    "C4": ("Wörterbuch:\n{noun},", {"m": [" der"], "f": [" die"]}, "N/A"),
}  # fmt: skip
DE_GATE_PRIMARY = "C4"  # PI 2026-10-09 (decisions.md): one frame for all models
DE_GATE_SECONDARY = "C1"


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


NUMBER = "data/stimuli/phase5_number_v1.csv"
KAIKKI = "data/raw/kaikki/kaikki.org-dictionary-{}.jsonl"


def _headword_plurals(lang: str, lemmas: set[str]) -> dict[str, set[str]]:
    """Headword-line plurals (kaikki `forms` tagged exactly ["plural"]) of the given nouns, over
    all of each word's noun entries."""
    import json

    out: dict[str, set[str]] = {w: set() for w in lemmas}
    with open(KAIKKI.format({"es": "Spanish", "de": "German"}[lang]), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            w = d.get("word")
            if w in out and d.get("pos") == "noun":
                out[w] |= {x["form"] for x in d.get("forms", []) if x.get("tags") == ["plural"]}
    return out


def build_number() -> pd.DataFrame:
    """Singular/plural pairs for the §5.4 number direction, on the nouns the gender directions
    are fitted on (Phase 2 `strat`+`matched`, Phase 3 `strat3`, all three genders), so number and gender
    vectors come from the same words. Kept only if Wiktionary's headword gives exactly one
    plural, letters only, different from the singular (dropped, not reviewed)."""
    import re

    from .phase4 import P2_FINAL, P3_FINAL

    p2 = pd.read_csv(P2_FINAL, keep_default_na=False)
    p3 = pd.read_csv(P3_FINAL, keep_default_na=False)
    src = {"es": p2[p2.set.isin(["strat", "matched"])],
           "de": p3[p3.set == "strat3"]}  # fmt: skip
    rows, counts = [], {}
    for lang, df in src.items():
        pl = _headword_plurals(lang, set(df.lemma))
        for r in df.itertuples():
            forms = pl[r.lemma]
            p = next(iter(forms)) if len(forms) == 1 else ""
            if p and re.fullmatch(r"[^\W\d_]+", p) and p != r.lemma:
                rows.append({"lang": lang, "lemma": r.lemma, "plural": p, "gender": r.gender,
                             "split": "train"})  # fmt: skip
        counts[lang] = (len(df), sum(1 for x in rows if x["lang"] == lang))
    out = pd.DataFrame(rows)
    out.to_csv(NUMBER, index=False)
    for lang, (n0, n1) in counts.items():
        g = out[out.lang == lang].gender.value_counts().to_dict()
        print(f"{lang}: {n1} of {n0} nouns kept {g}")
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


ADJ_TRANS = "data/stimuli/phase5_adj_translations_v1.csv"
SKIP_ADJ_TAGS = {"archaic", "obsolete", "rare", "dated"}


def _gloss_head(g: str, exact: bool = True) -> str:
    """exact: the whole first gloss minus parentheticals ('strong (intense)' -> 'strong'; 'good,
    great' -> 'good, great', which matches nothing). Else its first comma item ('good')."""
    import re

    g = re.sub(r"\([^)]*\)", "", g).strip().lower()
    return g if exact else re.split(r"[,;]", g)[0].strip()


def _adjectives_glossing(lang: str, english: set[str]) -> dict[str, list[dict]]:
    """Foreign adjectives whose first sense's first gloss head is one of the English words (first
    sense not archaic/obsolete/rare/dated). Spanish: also whether the form is gender-invariant
    (no feminine singular form listed: *fuerte*, not *blanco/blanca*)."""
    import json

    out: dict[str, list[dict]] = {}
    with open(KAIKKI.format({"es": "Spanish", "de": "German"}[lang]), encoding="utf-8") as f:
        for line in f:
            if '"pos": "adj"' not in line:
                continue
            d = json.loads(line)
            ss = [s for s in d.get("senses", []) if s.get("glosses")]
            if d.get("pos") != "adj" or not ss or set(ss[0].get("tags", [])) & SKIP_ADJ_TAGS:
                continue
            e = _gloss_head(ss[0]["glosses"][0])
            if e not in english or " " in d["word"] or not d["word"].isalpha():
                continue
            fem = {x["form"] for x in d.get("forms", [])
                   if "feminine" in x.get("tags", []) and "plural" not in x.get("tags", [])}  # fmt: skip
            out.setdefault(e, []).append({"word": d["word"], "invariant": not (fem - {d["word"]})})
    return out


def build_adj_translations() -> pd.DataFrame:
    """P20/R3 adjectives: the R1 Glasgow adjectives (qwen3 set) with one Spanish and one German
    translation from Wiktionary (kaikki): a foreign adjective (Zipf >= 3 in its language) whose
    first gloss is that English word; English words with two or more such translations are
    dropped (ambiguous; not reviewed). Spanish `invariant` marks the R3 set. Token counts with a
    leading space for both families (P16 applies to steered readouts)."""
    from transformers import AutoTokenizer
    from wordfreq import zipf_frequency

    g = pd.concat([pd.read_csv(f"data/stimuli/phase5_adjectives_{f}_v1.csv")
                   for f in ("qwen3", "eurollm")]).drop_duplicates("word")  # fmt: skip
    eng = set(g.word)
    toks = {f: AutoTokenizer.from_pretrained(m) for f, m in FAMILIES.items()}
    cols = {}
    for lang in ("es", "de"):
        cand = _adjectives_glossing(lang, eng)
        keep = {}
        for e, cs in cand.items():
            cs = [
                c
                for c in {c["word"]: c for c in cs}.values()
                if zipf_frequency(c["word"], lang) >= 3
            ]
            if len(cs) == 1:
                keep[e] = cs[0]
        cols[lang] = keep
        print(f"{lang}: {len(cand)} English adjectives with a candidate, {len(keep)} unambiguous")
    rows = []
    for r in g.itertuples():
        es, de = cols["es"].get(r.word), cols["de"].get(r.word)
        if not es and not de:
            continue
        row = {"en": r.word, "es": es["word"] if es else "", "de": de["word"] if de else "",
               "es_invariant": bool(es and es["invariant"])}  # fmt: skip
        row |= {sc: getattr(r, sc) for sc in SCALES}
        for f, t in toks.items():
            for lang in ("es", "de"):
                w = row[lang]
                row[f"{lang}_ntok_{f}"] = (
                    len(t(" " + w, add_special_tokens=False)["input_ids"]) if w else 0
                )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(ADJ_TRANS, index=False)
    inv = df[df.es_invariant]
    print(
        f"rows {len(df)}: es {int((df.es != '').sum())} (invariant {len(inv)}; single-token "
        f"qwen3 {int((inv.es_ntok_qwen3 == 1).sum())}), de {int((df.de != '').sum())}"
    )
    print(f"invariant gender-valence r = {np.corrcoef(inv.GEND, inv.VAL)[0, 1]:.2f}")
    return df
