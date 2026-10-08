"""Phase 5 nonce words (P15/P17, adopted 2026-10-08), generated with Wuggy.

Wuggy (Keuleers & Brysbaert 2010) makes pseudowords that follow a language's syllable structure
and letter-transition statistics, using real words as templates.

R-NONCE (Spanish-shaped, used in English frames): templates are common inanimate 2-3 syllable
Spanish nouns ending in -a/-o (equal numbers; never a test-set word; they only give shape). Stem =
pseudoword minus its final vowel; stems ending in c/g/z/q are dropped (a following e/i changes
their pronunciation). Four forms per stem:
  -a   Spanish strong feminine cue; English vowel-final names are judged female
  -o   Spanish strong masculine cue
  -e   weak masculine cue (16% of Spanish -e nouns are feminine)
  -iz  38% feminine (la nariz / el lapiz): the most even consonant ending that isn't an
       inflection (-s, 45%, is the plural marker); English consonant-final names are judged male
Every form must be absent from the Spanish Wiktionary (all words and inflected forms) and have
zero frequency in six languages. 150 stems.

English-style set (steered only): templates are common concrete 2-syllable English nouns; outputs
with zero frequency in six languages that aren't WordNet lemmas; ~100 words. The final-letter
class is recorded (English name phonology), though it cancels in steered-minus-unsteered shifts.
"""

from __future__ import annotations

import random
import re

import pandas as pd

from .lexicon import _wordnet, eligible, load_lexicon

SEED = 0
LANGS = ("es", "en", "pt", "it", "fr", "de")
ENDINGS = ["a", "o", "e", "iz"]
OUT_ES = "data/stimuli/phase5_nonce_es_v1.csv"
OUT_EN = "data/stimuli/phase5_nonce_en_v1.csv"
VOWEL_GROUP = re.compile(r"[aeiouáéíóúü]+")


def _syllables(w: str) -> int:
    return len(VOWEL_GROUP.findall(w.lower()))


def _generator(plugin: str):
    from wuggy import WuggyGenerator

    g = WuggyGenerator()
    if plugin not in getattr(g, "language_plugins", {}):
        g.download_language_plugin(plugin, auto_download=True)
    g.load(plugin)
    return g


def _zero_freq(w: str) -> bool:
    from wordfreq import zipf_frequency

    return all(zipf_frequency(w, lang) == 0 for lang in LANGS)


def _held_out_es() -> set[str]:
    pairs = pd.read_csv("data/lexicon/pairs_de_es.csv", keep_default_na=False)
    classics = pd.read_csv("data/stimuli/classics.csv", keep_default_na=False)
    p2 = pd.read_csv("data/stimuli/phase2_final_v4.csv", keep_default_na=False)
    return set(pairs.es_lemma) | set(classics.lemma) | set(p2[p2.split == "test"].lemma)


def spanish_templates(n_per: int = 150) -> list[str]:
    el = eligible(load_lexicon("es"))
    el = el[(el.es_regular == "yes") & (el.zipf >= 3.5) & ~el.lemma.isin(_held_out_es())]
    el = el[el.lemma.map(_syllables).between(2, 3) & el.lemma.str.fullmatch(r"[a-zñ]+")]
    by = {e: el[el.lemma.str.endswith(e)].sort_values("zipf", ascending=False) for e in "ao"}
    k = min(n_per, *(len(v) for v in by.values()))  # equal numbers of -a and -o templates
    return [w for v in by.values() for w in v.lemma.head(k)]


def build_spanish(n_stems: int = 150, ncand: int = 10) -> pd.DataFrame:
    from .phase1_stimuli import scan_spanish

    all_forms, _, _ = scan_spanish()
    g = _generator("orthographic_spanish")
    templates = spanish_templates()
    res = g.generate_classic(templates, ncandidates_per_sequence=ncand)
    stems = {}
    for r in res:
        pw = r["pseudoword"]
        if not pw or pw[-1] not in "ao" or not re.fullmatch(r"[a-zñ]+", pw):
            continue
        stem = pw[:-1]
        if stem[-1] in "aeiouc gzq" or _syllables(stem) < 1:
            continue
        stems.setdefault(stem, r["word"])
    rows = []
    for stem, tmpl in stems.items():
        forms = [stem + e for e in ENDINGS]
        if any(f in all_forms for f in forms) or not all(_zero_freq(f) for f in forms):
            continue
        rows.append(
            {
                "stem": stem,
                "template": tmpl,
                **{f"form_{e}": f for e, f in zip(ENDINGS, forms, strict=True)},
            }
        )
    rng = random.Random(SEED)
    rng.shuffle(rows)
    df = pd.DataFrame(rows[:n_stems])
    df["source"] = (
        "wuggy orthographic_spanish; absent from es Wiktionary + zero wordfreq in "
        + "/".join(LANGS)
    )
    df.to_csv(OUT_ES, index=False)
    print(
        f"Spanish-shaped: {len(stems)} candidate stems, {len(rows)} pass all checks, kept {len(df)}"
    )
    return df


def english_templates(n: int = 200) -> list[str]:
    from wordfreq import top_n_list

    wn = _wordnet()
    out = []
    for w in top_n_list("en", 20000):
        if not re.fullmatch(r"[a-z]{4,8}", w) or _syllables(w) != 2:
            continue
        ss = wn.synsets(w)
        concrete = {"noun.artifact", "noun.object", "noun.food", "noun.plant", "noun.substance"}
        # a noun and never an adjective (local, total, simple are out)
        if ss and ss[0].pos() == "n" and ss[0].lexname() in concrete and not wn.synsets(w, pos="a"):
            out.append(w)
        if len(out) == n:
            break
    return out


def build_english(n: int = 100, ncand: int = 5) -> pd.DataFrame:
    wn = _wordnet()
    g = _generator("orthographic_english")
    res = g.generate_classic(english_templates(), ncandidates_per_sequence=ncand)
    seen, rows = set(), []
    for r in res:
        pw = r["pseudoword"]
        if not pw or pw in seen or not re.fullmatch(r"[a-z]+", pw):
            continue
        seen.add(pw)
        if wn.synsets(pw) or not _zero_freq(pw):
            continue
        rows.append(
            {
                "word": pw,
                "template": r["word"],
                "final": "vowel" if pw[-1] in "aeiouy" else "consonant",
            }
        )
    rng = random.Random(SEED)
    rng.shuffle(rows)
    df = pd.DataFrame(rows[:n])
    df["source"] = "wuggy orthographic_english; not a WordNet lemma + zero wordfreq in " + "/".join(
        LANGS
    )
    df.to_csv(OUT_EN, index=False)
    print(
        f"English-shaped: {len(seen)} candidates, {len(rows)} pass, kept {len(df)}; final {df.final.value_counts().to_dict()}"
    )
    return df
