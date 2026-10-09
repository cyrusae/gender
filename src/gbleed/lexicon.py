"""Noun lexicons from Wiktionary (kaikki.org extracts of English Wiktionary).

Step 1, `build_lexicon`: parse the per-language JSONL dump into one row per
noun lemma with its gold gender(s), first English gloss, frequency, an
animacy judgement, and the morphological tags the design doc needs
(German suffixes; Spanish -o/-a regularity and exception type). Also writes
German-Spanish translation pairs matched on English gloss.

Step 2, `sample_phase0`: draw a Phase 0 stimulus list from the lexicons,
stratified by frequency bin and gender, plus flipped/control pairs.

Gold genders come straight from Wiktionary's gender tags. Animacy and
translation pairing are heuristics (see below): every row keeps its English
gloss and the reasons for its animacy call so it can be audited in English.
Wiktionary content is CC BY-SA; see https://kaikki.org for attribution.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
import re
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("data/raw/kaikki")
NLTK_DIR = Path("data/raw/nltk_data")
LEX_DIR = Path("data/lexicon")
LANG_NAMES = {"de": "German", "es": "Spanish", "ru": "Russian"}
URL = "https://kaikki.org/dictionary/{name}/kaikki.org-dictionary-{name}.jsonl"

GENDER_TAGS = {"masculine": "m", "feminine": "f", "neuter": "n"}
# Senses that aren't a real lexical use of this lemma.
SKIP_SENSE_TAGS = {
    "form-of", "alt-of", "abbreviation", "initialism", "acronym", "misspelling",
    "plural", "plural-only", "clipping-of",
}  # fmt: skip
# Register/usage marks: such a noun is a poor test of what the model "knows".
MARKED_TAGS = {
    "obsolete", "archaic", "dated", "historical", "rare", "slang", "vulgar",
    "derogatory", "offensive", "dialectal", "regional", "colloquial", "informal",
    "nonstandard", "poetic", "humorous", "euphemistic", "Internet", "jargon",
    "uncommon", "literary", "childish", "proscribed",
}  # fmt: skip
# Parts of speech that make a noun a homograph (and corrupt its frequency).
OTHER_POS = {
    "verb", "adj", "adv", "prep", "conj", "pron", "det", "intj", "num", "article",
    "contraction", "particle", "postp",
}  # fmt: skip
WORD_RE = {
    "de": re.compile(r"^[A-ZÄÖÜ][a-zäöüß]+$"),  # nouns are capitalised; no compounds w/ hyphens
    "es": re.compile(r"^[a-záéíóúüñ]+$"),
    "ru": re.compile(r"^[а-яё]+$"),  # page titles are unstressed and lowercase
}

# ---- animacy heuristics -----------------------------------------------------
PEOPLE_CAT = re.compile(
    r"\b(people|occupations|professions|family members|ethnonyms|nationalities|demonyms|"
    r"leaders|titles|given names|surnames|deities|gods|mythological creatures|"
    r"male|female|women|men|children)\b",
    re.IGNORECASE,
)
ANIMAL_CAT = re.compile(
    r"\b(animals?|mammals|birds|fish|insects|reptiles|amphibians|dogs|cats|horses|cattle|"
    r"pigs|flies|bees|ants|beetles|butterflies|moths|spiders|arachnids|snakes|lizards|"
    r"rodents|bovines|equids|livestock|poultry|crustaceans|molluscs|primates|bats|"
    r"cetaceans|whales|sharks|canids|felids|marsupials|parrots|birds of prey|corvids|"
    r"waterfowl|gamebirds|songbirds|owls|worms|vermin|pets|monsters)\b",
    re.IGNORECASE,
)
GLOSS_ANIMATE = re.compile(
    r"^(?:a |an |the )?(?:(?:young|old|little|male|female|small|large) )?"
    r"(person|people|man|woman|men|women|boy|girl|child|baby|male|female|one who|someone|"
    r"somebody|member|inhabitant|native|resident|citizen|follower|adherent|supporter|"
    r"practitioner|specialist|expert|player|worker|servant|husband|wife|son|daughter|"
    r"mother|father|brother|sister|king|queen|god|goddess|animal|bird|fish|insect|dog|"
    r"cat|horse|cow|pig)\b",
    re.IGNORECASE,
)
ANIMATE_LEX = {"noun.person", "noun.animal"}
CONCRETE_LEX = {
    "noun.artifact", "noun.object", "noun.food", "noun.plant", "noun.substance", "noun.body",
}  # fmt: skip
# Wiktionary marks of a person/animal noun: a gendered counterpart form
# (director -> directora, Hund -> Hündin), or a "female equivalent of" sense.
COUNTERPART_GLOSS = re.compile(r"\b(female|male|feminine|masculine) equivalent of\b", re.IGNORECASE)

# ---- morphology tags --------------------------------------------------------
DE_SUFFIXES = [  # reliable gender predictors -> keep as their own test subgroup
    ("schaft", "f"), ("heit", "f"), ("keit", "f"), ("ung", "f"), ("ion", "f"),
    ("tät", "f"), ("enz", "f"), ("anz", "f"), ("ik", "f"), ("ei", "f"), ("ur", "f"),
    ("ismus", "m"), ("ling", "m"),
]  # fmt: skip


def de_suffix(word: str) -> str:
    for suf, _ in DE_SUFFIXES:
        if word.endswith(suf) and len(word) >= len(suf) + 2:
            return "-" + suf
    return ""


def es_morphology(word: str, gender: str, etymology: str, sense_tags: set[str]) -> dict:
    ending = "ma" if word.endswith("ma") else word[-1] if word[-1] in "oa" else "other"
    if ending == "other" or gender not in ("m", "f"):
        regular, exc = "", ""
    else:
        expected = "m" if ending == "o" else "f"
        regular = "yes" if gender == expected else "no"
        exc = ""
        if regular == "no":
            if ending == "ma" and gender == "m" and "Greek" in etymology:
                exc = "greek_ma"
            elif "clipping" in sense_tags or re.search(r"\b[Cc]lipping of\b", etymology):
                exc = "clipping"
            else:
                exc = "other"
    return {
        "es_ending": ending,
        "es_regular": regular,
        "es_exception": exc,
        # el agua, el hacha: feminine nouns with stressed initial a- take "el".
        # Stress isn't in the data, so flag every feminine a-/ha- noun.
        "initial_a_f": gender == "f" and bool(re.match(r"^h?[aá]", word)),
    }


def ru_head(expansion: str) -> str:
    """Grammatical part of a Russian headword line, e.g. 'f anim' from
    'соба́ка • (sobáka) f anim (genitive ...)': the transliteration comes in parentheses
    *before* the gender, so the German/Spanish rule (text before the first '(') misses it."""
    m = re.match(r"[^(]*\([^)]*\)\s*([^(]*)", expansion)
    return m.group(1).strip() if m else ""


def ru_morphology(word: str, gender: str, tags: set[str], heads: list[str]) -> dict:
    """Ending class (the spelling cue Russian gender mostly follows) and grammatical animacy.
    a: -а/-я (mostly f); o: -о/-е/-ё (mostly n); mja: -мя (n); soft: -ь (m or f: the
    spelling-matched cell); cons: consonant or -й (mostly m)."""
    if word.endswith("мя"):
        ending = "mja"
    elif word[-1] in "ая":
        ending = "a"
    elif word[-1] in "оеё":
        ending = "o"
    elif word[-1] == "ь":
        ending = "soft"
    else:
        ending = "cons"
    head = " ".join(ru_head(h) for h in heads)  # heads: full headword lines
    anim = "anim" if "animate" in tags or re.search(r"\banim\b", head) else ""
    inan = "inan" if "inanimate" in tags or re.search(r"\binan\b", head) else ""
    return {
        "ru_ending": ending,
        "ru_animacy": "|".join(x for x in (anim, inan) if x),  # grammatical animacy tag
        "ru_indecl": "indeclinable" in tags or any("indecl" in h for h in heads),
    }


# ---- helpers ----------------------------------------------------------------
def dump_path(lang: str) -> Path:
    name = LANG_NAMES[lang]
    return RAW_DIR / f"kaikki.org-dictionary-{name}.jsonl"


def download(lang: str, force: bool = False) -> Path:
    path = dump_path(lang)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = URL.format(name=LANG_NAMES[lang])
    print(f"Downloading {url} (~1 GB) ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_gloss(gloss: str) -> str:
    """'bridge (structure built over a river)' -> 'bridge'."""
    g = re.sub(r"\([^)]*\)", "", gloss)
    g = re.split(r"[;,:]", g)[0].strip().lower().rstrip(".")
    g = re.sub(r"^(a|an|the|to)\s+", "", g)
    return re.sub(r"\s+", " ", g).strip()


def _wordnet():
    import nltk

    nltk.data.path.insert(0, str(NLTK_DIR))
    try:
        from nltk.corpus import wordnet as wn

        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet", download_dir=str(NLTK_DIR), quiet=True)
        from nltk.corpus import wordnet as wn
    return wn


def wordnet_animacy(wn, concept: str) -> tuple[str, float | None]:
    """(lexname of most frequent sense, share of tagged usage that is animate).

    Looks up the whole phrase, then the English head noun ('piece of
    furniture' -> 'piece', 'folding chair' -> 'chair')."""
    candidates = [concept.replace(" ", "_")]
    head = concept.split(" of ")[0].split()[-1] if concept else ""
    if head and head != candidates[0]:
        candidates.append(head)
    for c in candidates:
        lemmas = wn.lemmas(c, pos="n")
        if not lemmas:
            continue
        counts = [lem.count() for lem in lemmas]
        total = sum(counts)
        animate = sum(
            n for lem, n in zip(lemmas, counts, strict=True)
            if lem.synset().lexname() in ANIMATE_LEX
        )  # fmt: skip
        return lemmas[0].synset().lexname(), (animate / total if total else None)
    return "", None


def wordnet_concrete(wn, concept: str) -> bool:
    """Is the dominant (usage-weighted) sense a physical thing? 'table' is
    WordNet's data table first, but the furniture sense is far more frequent."""
    head = concept.split(" of ")[0].split()[-1] if concept else ""
    for c in (concept.replace(" ", "_"), head):
        lemmas = wn.lemmas(c, pos="n") if c else []
        if not lemmas:
            continue
        counts = [lem.count() for lem in lemmas]
        if sum(counts) == 0:
            return lemmas[0].synset().lexname() in CONCRETE_LEX
        conc = sum(
            n for lem, n in zip(lemmas, counts, strict=True)
            if lem.synset().lexname() in CONCRETE_LEX
        )  # fmt: skip
        return conc / sum(counts) >= 0.5
    return False


LETTER_GLOSS = re.compile(r"(name of the .*letter|letter of the .*alphabet)", re.IGNORECASE)
NUMBER_GLOSS = re.compile(
    r"^(the )?((natural|cardinal|ordinal) )?(number|numeral|digit)\b", re.IGNORECASE
)


def lexical_class(gloss: str, cats: set[str]) -> set[str]:
    """Nouns that are names of letters/numbers: not useful stimuli."""
    out = set()
    if LETTER_GLOSS.search(gloss) or any("letter names" in c.lower() for c in cats):
        out.add("letter-name")
    if NUMBER_GLOSS.search(gloss) or any(
        re.search(r"\bnumbers?\b", c, re.IGNORECASE) for c in cats
    ):
        out.add("number")
    return out


def judge_animacy(
    cats: set[str], tags: set[str], gloss: str, lexname: str, share, counterpart: bool = False
):
    reasons = []
    if counterpart:
        reasons.append("gendered-counterpart")
    if "agent" in tags:
        reasons.append("agent-noun")
    if any(PEOPLE_CAT.search(c) for c in cats):
        reasons.append("people-category")
    if any(ANIMAL_CAT.search(c) and "body parts" not in c.lower() for c in cats):
        reasons.append("animal-category")
    if "by-personal-gender" in tags:
        reasons.append("by-personal-gender")
    if GLOSS_ANIMATE.search(gloss):
        reasons.append("gloss")
    if lexname in ANIMATE_LEX:
        reasons.append(f"wordnet:{lexname}")
    if reasons:
        return "animate", ";".join(reasons)
    if not lexname:
        return "uncertain", "not-in-wordnet"
    if share is not None and share >= 0.25:
        return "uncertain", f"wordnet-animate-share:{share:.2f}"
    if share is None:
        # No usage counts: fall back to "first sense is inanimate" only.
        return "inanimate", f"wordnet-first:{lexname}"
    return "inanimate", f"wordnet:{lexname}"


def multi_gender_info(ents: list[dict]) -> dict:
    """For a noun with more than one gender, is the gender tied to meaning?

    meaning_split     every sense has one gender, and the genders' glosses
                      differ (der See "lake" / die See "sea")
    free_variation    some sense allows both genders (el/la mar "sea",
                      el/la azúcar "sugar")
    """
    senses = [sn for e in ents for sn in e["senses"]]
    by_gender: dict[str, list[str]] = defaultdict(list)
    shared = False
    for gs, gloss in senses:
        if len(gs) > 1:
            shared = True
        for g in gs:
            by_gender[g].append(gloss)
    single = {g: [gl for gs, gl in senses if gs == {g}] for g in by_gender}
    concepts = {g: {normalize_gloss(gl) for gl in gls} for g, gls in single.items()}
    overlap = any(concepts[a] & concepts[b] for a in concepts for b in concepts if a < b)
    kind = "free_variation" if shared or overlap else "meaning_split"
    info = {"multi_type": kind}
    for g in ("m", "f", "n"):
        gls = single.get(g) or by_gender.get(g) or []
        info[f"gloss_{g}"] = gls[0] if gls else ""
    return info


def write_multi_candidates(min_zipf: float = 2.5) -> pd.DataFrame:
    """Same spelling, masculine AND feminine (no neuter), not animate: the
    pool for hand-picking spelling-constant tests (der/die See, el/la mar)."""
    out = []
    for lang in ("de", "es"):
        df = load_lexicon(lang)
        # m AND f (a rare neuter use too is allowed, but flagged). Animate ones
        # are kept and flagged: der Leiter "leader" / die Leiter "ladder" mixes
        # in social gender, which matters for some uses and not others.
        g = df.genders.astype(str)
        m = g.str.contains("m") & g.str.contains("f") & (df.zipf >= min_zipf)
        m &= (df.marked == "") & (df.also_pos == "")
        sub = df.loc[m].assign(has_neuter=g[m].str.contains("n"))
        cols = ["lang", "lemma", "multi_type", "gloss_m", "gloss_f", "has_neuter", "animacy",
                "animacy_reason", "zipf", "regions"]  # fmt: skip
        out.append(sub[cols])
    res = pd.concat(out).sort_values(["lang", "multi_type", "zipf"], ascending=[True, True, False])
    res.to_csv(LEX_DIR / "multi_gender_candidates.csv", index=False)
    print(res.groupby(["lang", "multi_type"]).size().to_string())
    print(f"-> {LEX_DIR}/multi_gender_candidates.csv")
    return res


# ---- step 1: build ----------------------------------------------------------
def parse_dump(lang: str, path: Path):
    """Yield per-lemma aggregates and the set of words attested as other POS."""
    word_re = WORD_RE[lang]
    entries = defaultdict(list)
    other_pos = set()
    size = path.stat().st_size
    with open(path, encoding="utf-8") as f, tqdm(total=size, unit="B", unit_scale=True,
                                                 desc=f"parse {lang}") as bar:  # fmt: skip
        for line in f:
            bar.update(len(line.encode("utf-8")))
            r = json.loads(line)
            if r.get("lang_code") != lang:
                continue
            word, pos = r.get("word", ""), r.get("pos")
            if pos == "name":
                # Given names (Charlotte, Isa): name-homograph nouns carry social
                # gender and a contaminated frequency. Surnames aren't counted
                # (Stein, Berg are surnames too).
                if any(
                    "given name" in g for sn in r.get("senses", []) for g in sn.get("glosses", [])
                ):
                    other_pos.add((word.lower(), "given-name"))
                continue
            if pos != "noun":
                if pos in OTHER_POS:
                    # Lowercased: wordfreq lowercases too, so "Aber" shares "aber"'s count.
                    # Inflected forms (sonne < sonnen, mesa < mesar) are tagged
                    # "form", not excluded: they're the Phase 2 homograph pool.
                    senses = r.get("senses", [])
                    is_form = bool(senses) and all(
                        {"form-of", "alt-of"} & set(sn.get("tags", [])) for sn in senses
                    )
                    other_pos.add((word.lower(), f"{pos}-form" if is_form else pos))
                continue
            if not word_re.match(word):
                continue
            core = [
                s for s in r.get("senses", [])
                if s.get("glosses") and not (set(s.get("tags", [])) & SKIP_SENSE_TAGS)
            ]  # fmt: skip
            if not core:
                # An inflected form of a DIFFERENT noun (Plane = plural of Plan).
                targets = {
                    fo.get("word")
                    for sn in r.get("senses", [])
                    for fo in sn.get("form_of", []) or []
                }
                if targets - {word, None}:
                    other_pos.add((word.lower(), "noun-form"))
                continue
            genders = {GENDER_TAGS[t] for s in core for t in s.get("tags", []) if t in GENDER_TAGS}
            if not genders:  # fall back to the headword line, e.g. "Brücke f (...)"
                for h in r.get("head_templates", []):
                    exp = h.get("expansion", "")
                    head = ru_head(exp) if lang == "ru" else exp.split("(")[0]
                    genders |= set(re.findall(r"\b([mfn])\b", head))
            # Topical categories of the FIRST sense only: "Buch" has a minor sense
            # (omasum) filed under "Animal body parts".
            s0 = core[0]
            cats = {c["name"] for c in s0.get("categories", []) if isinstance(c, dict)}
            cats |= set(s0.get("topics", []) or [])
            entries[word].append(
                {
                    "genders": genders,
                    # (genders tagged on this sense, or the entry's if untagged; gloss)
                    "senses": [
                        (
                            frozenset(
                                GENDER_TAGS[t] for t in sn.get("tags", []) if t in GENDER_TAGS
                            )
                            or frozenset(genders),
                            sn["glosses"][0],
                        )
                        for sn in core
                    ],
                    "gloss": core[0]["glosses"][0],
                    "first_tags": set(core[0].get("tags", [])),
                    "tags": {t for s in core for t in s.get("tags", [])},
                    "cats": cats,
                    "etymology": r.get("etymology_text", ""),
                    "heads": [h.get("expansion", "") for h in r.get("head_templates", [])]
                    if lang == "ru"
                    else [],
                    # A *different singular word* of the other gender (director ->
                    # directora). Not plurals: "mares" is tagged with gender too.
                    "counterpart": any(
                        {"masculine", "feminine"} & set(fm.get("tags", []))
                        and not {"plural", "diminutive", "augmentative", "canonical"}
                        & set(fm.get("tags", []))
                        # Russian headwords carry stress marks (кни́га vs the title книга)
                        and fm.get("form", "").replace("\u0301", "").replace("\u0300", "") != word
                        for fm in r.get("forms", [])
                    )
                    or any(
                        COUNTERPART_GLOSS.search(g)
                        for sn in r.get("senses", [])
                        for g in sn.get("glosses", [])
                    ),
                }
            )
    return entries, other_pos


def build_lexicon(lang: str, force_download: bool = False) -> pd.DataFrame:
    from wordfreq import zipf_frequency

    path = download(lang, force_download)
    entries, other_pos = parse_dump(lang, path)
    other_words = defaultdict(set)
    for w, p in other_pos:
        other_words[w].add(p)
    wn = _wordnet()

    rows = []
    for word, ents in tqdm(entries.items(), desc=f"lexicon {lang}"):
        genders = set().union(*(e["genders"] for e in ents))
        if not genders:
            continue
        first = ents[0]
        tags = set().union(*(e["tags"] for e in ents))
        cats = first["cats"]
        concept = normalize_gloss(first["gloss"])
        lexname, share = wordnet_animacy(wn, concept)
        counterpart = any(e["counterpart"] for e in ents)
        animacy, why = judge_animacy(cats, tags, first["gloss"], lexname, share, counterpart)
        gender = next(iter(genders)) if len(genders) == 1 else "multi"
        row = {
            "lang": lang,
            "lemma": word,
            "gender": gender,
            "genders": "|".join(sorted(genders)),
            "concept_en": concept,
            "gloss": first["gloss"],
            "zipf": zipf_frequency(word, lang),
            # Loanwords/false friends whose frequency comes from English text (arcade, van).
            "zipf_en": zipf_frequency(word, "en"),
            "animacy": animacy,
            "animacy_reason": why,
            "wordnet": lexname,
            "concrete": wordnet_concrete(wn, concept),
            "marked": "|".join(
                sorted((first["first_tags"] & MARKED_TAGS) | lexical_class(first["gloss"], cats))
            ),
            # Region tags are the capitalised ones (Austria, Mexico, Latin-America...).
            "regions": "|".join(sorted(t for t in first["first_tags"] if t[:1].isupper())),
            # also_pos: same spelling is another word (aber, de, la): frequency is
            # contaminated -> excluded. also_form: same spelling is an inflected
            # form of another word (camino < caminar) -> kept, but tagged.
            "also_pos": "|".join(
                sorted(p for p in other_words.get(word.lower(), ()) if not p.endswith("-form"))
            ),
            "also_form": "|".join(
                sorted(p for p in other_words.get(word.lower(), ()) if p.endswith("-form"))
            ),
            "n_entries": len(ents),
            **(multi_gender_info(ents) if len(genders) > 1 else {}),
        }
        if lang == "de":
            row["de_suffix"] = de_suffix(word)
        elif lang == "ru":
            row.update(ru_morphology(word, gender, tags, [h for e in ents for h in e["heads"]]))
        else:
            row.update(es_morphology(word, gender, first["etymology"], tags))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("zipf", ascending=False).reset_index(drop=True)

    LEX_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LEX_DIR / f"{lang}_nouns.csv", index=False)
    _update_metadata(
        lang,
        {
            "url": URL.format(name=LANG_NAMES[lang]),
            "dump_mtime": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            "dump_sha256": _sha256(path),
            "built": datetime.now(UTC).isoformat(timespec="seconds"),
            "n_lemmas": len(df),
            "by_gender": df.gender.value_counts().to_dict(),
            "by_animacy": df.animacy.value_counts().to_dict(),
        },
    )
    print(f"{lang}: {len(df)} noun lemmas -> {LEX_DIR / f'{lang}_nouns.csv'}")
    return df


def _update_metadata(lang: str, info: dict) -> None:
    p = LEX_DIR / "metadata.json"
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta[lang] = info
    p.write_text(json.dumps(meta, indent=2))


def source_tag(lang: str) -> str:
    meta = json.loads((LEX_DIR / "metadata.json").read_text())
    return f"wiktionary:kaikki@{meta[lang]['dump_mtime'][:10]}"


def load_lexicon(lang: str) -> pd.DataFrame:
    return pd.read_csv(LEX_DIR / f"{lang}_nouns.csv", keep_default_na=False)


def eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, inanimate, single-gender m/f nouns usable as stimuli.

    Homograph filters are strict on purpose: excluded nouns stay in the
    lexicon with their tags (also_form is the Phase 2 noun/verb homograph pool).
    """
    m = (
        df.gender.isin(["m", "f"])
        & (df.animacy == "inanimate")
        & (df.marked == "")
        & (df.regions == "")
        & (df.also_pos == "")  # another word, incl. a given name (aber, de, Charlotte)
        & (df.also_form == "")  # an inflected form of another word (Tolle, van, Plane)
        & (df.zipf_en < df.zipf)  # not more common in English than in the language
    )
    if "initial_a_f" in df:
        m &= ~df.initial_a_f.astype(str).eq("True")
    return df[m]


DOMINANCE = 1.0  # zipf units: 1.0 = 10x more frequent
EN_SAME = 0.9  # string similarity to the English gloss (accent-insensitive)
EN_COGNATE = 0.7
EN_HOMOGRAPH_ZIPF = 3.0


def _plain(w: str) -> str:
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFD", str(w).lower()) if unicodedata.category(c) != "Mn"
    )


MAX_ZIPF_GAP = 1.0


def build_pairs(min_zipf: float = 2.5) -> pd.DataFrame:
    from wordfreq import zipf_frequency

    """German-Spanish translation pairs matched on normalised English gloss.

    For each concept, the most frequent eligible lemma in each language is
    taken. `strict` pairs are the ones that need no checking: each lemma is
    >= 10x more frequent than any other eligible noun with that first gloss
    in its language (so Zeit/vez "time" is out: tiempo is also "time" and as
    common), the two are within 10x of each other in frequency (Turmuhr/reloj
    "clock" is out), and they aren't cognates.
    `concrete`: the dominant WordNet sense is a physical thing."""
    best = {}
    for lang in ("de", "es"):
        el = eligible(load_lexicon(lang))
        el = el[el.zipf >= min_zipf].sort_values("zipf", ascending=False)
        grp = el.groupby("concept_en").zipf
        top = el.drop_duplicates("concept_en").set_index("concept_en")
        top["n_candidates"] = grp.size()
        # Second-highest frequency among nouns sharing this gloss (NaN if none).
        top["runner_up_zipf"] = grp.apply(lambda z: z.iloc[1] if len(z) > 1 else float("nan"))
        cols = ["lemma", "gender", "gloss", "zipf", "n_candidates", "runner_up_zipf", "concrete"]
        best[lang] = top[cols].add_prefix(f"{lang}_")
    pairs = best["de"].join(best["es"], how="inner").reset_index()
    pairs["flipped"] = pairs.de_gender != pairs.es_gender
    # Shared spelling (Apartheid/apartheid): the "translation" is just the same word.
    pairs["cognate"] = [
        difflib.SequenceMatcher(None, d.lower(), e.lower()).ratio() >= 0.8
        for d, e in zip(pairs.de_lemma, pairs.es_lemma, strict=True)
    ]
    # Dominant: >= 10x more frequent than any other noun with the same gloss.
    dominant = [
        pairs[f"{lang}_runner_up_zipf"].isna()
        | (pairs[f"{lang}_zipf"] - pairs[f"{lang}_runner_up_zipf"] >= DOMINANCE)
        for lang in ("de", "es")
    ]
    # Translation equivalents have similar frequency; Turmuhr/reloj don't.
    similar_freq = (pairs.de_zipf - pairs.es_zipf).abs() <= MAX_ZIPF_GAP
    # English overlap (Phases 4-5 are cross-lingual and prompt in English):
    #   en_same       a lemma IS the English word (Stagnation, Grill, melón~melon): dropped
    #   en_homograph  a lemma is a common English word (zipf >= 3; Devise, Stein): dropped
    #   en_cognate    similar to the English word (montaña~mountain-ish, Lunge~lung): kept,
    #                 flagged, so Phase 4 can compare cognate vs non-cognate pairs
    for lang in ("de", "es"):
        pairs[f"{lang}_en_sim"] = [
            difflib.SequenceMatcher(None, _plain(w), _plain(c)).ratio()
            for w, c in zip(pairs[f"{lang}_lemma"], pairs.concept_en, strict=True)
        ]
        pairs[f"{lang}_zipf_en"] = [zipf_frequency(w, "en") for w in pairs[f"{lang}_lemma"]]
    sim = pairs[["de_en_sim", "es_en_sim"]].max(axis=1)
    pairs["en_same"] = sim >= EN_SAME
    pairs["en_homograph"] = pairs[["de_zipf_en", "es_zipf_en"]].max(axis=1) >= EN_HOMOGRAPH_ZIPF
    pairs["en_cognate"] = (sim >= EN_COGNATE) & ~pairs.en_same
    pairs["strict"] = (
        dominant[0]
        & dominant[1]
        & similar_freq
        & ~pairs.cognate
        & ~pairs.en_same
        & ~pairs.en_homograph
    )
    pairs["concrete"] = pairs.de_concrete.astype(bool)  # same English concept on both sides
    pairs = pairs.drop(columns=["de_concrete", "es_concrete"])
    pairs = pairs.sort_values(["flipped", "strict", "concrete", "de_zipf"], ascending=False)
    pairs.to_csv(LEX_DIR / "pairs_de_es.csv", index=False)
    f = pairs[pairs.flipped]
    print(
        f"{len(pairs)} pairs, {len(f)} flipped; strict flipped: {int(f.strict.sum())}, "
        f"strict+concrete flipped: {int((f.strict & f.concrete).sum())} -> {LEX_DIR}/pairs_de_es.csv"
    )
    return pairs


# ---- hand-picked classics ---------------------------------------------------
def build_classics(
    spec: str = "data/stimuli/classics_spec.csv", out: str = "data/stimuli/classics.csv"
) -> pd.DataFrame:
    """Hand-picked flipped pairs, with genders looked up in Wiktionary.

    `check` lists anything that makes an item weaker than it looks: the noun
    has more than one gender, isn't flipped after all, has an animate sense,
    or is a homograph. Items are kept either way; the column says why to worry.
    """
    sp = pd.read_csv(spec, comment="#", dtype=str)
    lex = {lang: load_lexicon(lang).set_index("lemma") for lang in ("de", "es")}
    rows = []
    for r in sp.itertuples():
        info, checks = {}, []
        for lang, lemma in (("de", r.de), ("es", r.es)):
            if lemma not in lex[lang].index:
                raise ValueError(f"{lemma!r} not in {lang} lexicon")
            e = lex[lang].loc[lemma]
            if isinstance(e, pd.DataFrame):
                e = e.iloc[0]
            info[lang] = e
            if e.gender not in ("m", "f"):
                checks.append(f"{lang}:genders={e.genders}")
            if e.animacy != "inanimate":
                checks.append(f"{lang}:{e.animacy}({e.animacy_reason})")
            if e.also_pos:
                checks.append(f"{lang}:also_{e.also_pos}")
            if e.marked:
                checks.append(f"{lang}:marked={e.marked}")
            from wordfreq import zipf_frequency

            sim = difflib.SequenceMatcher(None, _plain(lemma), _plain(r.concept_en)).ratio()
            if sim >= EN_SAME:
                checks.append(f"{lang}:same-as-English")
            elif sim >= EN_COGNATE:
                checks.append(f"{lang}:English-cognate")
            if zipf_frequency(lemma, "en") >= EN_HOMOGRAPH_ZIPF:
                checks.append(f"{lang}:English-homograph(zipf {zipf_frequency(lemma, 'en'):.1f})")
        gd, ge = info["de"].gender, info["es"].gender
        if gd in ("m", "f") and ge in ("m", "f") and gd == ge:
            checks.append("not-flipped")
        for lang, lemma in (("de", r.de), ("es", r.es)):
            e = info[lang]
            # A multi-gender noun gets the gender of its sense matching the concept.
            g = e.gender
            if g not in ("m", "f"):
                for cand in ("m", "f"):
                    if re.search(
                        rf"\b{re.escape(r.concept_en)}\b",
                        str(e.get(f"gloss_{cand}", "")),
                        re.IGNORECASE,
                    ):
                        g = cand
            rows.append(
                {
                    "lang": lang,
                    "lemma": lemma,
                    "gender": g,
                    "concept_en": r.concept_en,
                    "set": "classic",
                    "source": source_tag(lang),
                    "zipf": e.zipf,
                    "gloss": e.gloss,
                    "cited_in": r.cited_in,
                    "check": "; ".join(checks),
                }
            )
    df = pd.DataFrame(rows)
    df["freq_bin"] = df.zipf.map(freq_bin)
    df.to_csv(out, index=False)
    bad = df[df.check != ""].drop_duplicates("concept_en")
    print(f"{len(sp)} classic pairs -> {out}; {len(bad)} with checks:")
    for b in bad.itertuples():
        print(f"  {b.concept_en}: {b.check}")
    return df


# ---- step 2: sample a Phase 0 list -------------------------------------------
FREQ_BINS = [(2.5, 3.5, "low"), (3.5, 4.5, "mid"), (4.5, 99.0, "high")]


def freq_bin(z: float) -> str:
    for lo, hi, name in FREQ_BINS:
        if lo <= z < hi:
            return name
    return "rare"


def sample_phase0(
    out: str,
    per_cell: int = 20,
    max_pairs: int = 60,
    seed: int = 0,
    exclude: set[tuple[str, str]] | None = None,
    concrete_pairs: bool = True,
    classics: str | None = "data/stimuli/classics.csv",
) -> pd.DataFrame:
    """per_cell nouns per (language, frequency bin, gender), plus up to
    max_pairs flipped and max_pairs control translation pairs."""
    rng = random.Random(seed)
    exclude = exclude or set()
    keep = ["lang", "lemma", "gender", "concept_en", "gloss", "zipf"]
    extra = {"de": ["de_suffix"], "es": ["es_ending", "es_regular", "es_exception"]}
    lex = {lang: eligible(load_lexicon(lang)) for lang in ("de", "es")}
    rows, used = [], set(exclude)

    if classics and Path(classics).exists():
        cl = pd.read_csv(classics, keep_default_na=False)
        cl = cl[cl.gender.isin(["m", "f"])]
        for row in cl.to_dict("records"):
            rows.append({k: row[k] for k in ["lang", "lemma", "gender", "concept_en", "gloss",
                                            "zipf", "set", "cited_in", "check"]})  # fmt: skip
            used.add((row["lang"], row["lemma"]))

    pairs = build_pairs()
    for flipped, name in ((True, "flipped"), (False, "control")):
        sub = pairs[(pairs.flipped == flipped) & pairs.strict]
        if concrete_pairs:
            sub = sub[sub.concrete]
        idx = rng.sample(range(len(sub)), min(max_pairs, len(sub)))
        for r in sub.iloc[sorted(idx)].itertuples():
            ks = [("de", r.de_lemma), ("es", r.es_lemma)]
            if any(k in used for k in ks):
                continue
            for lang, lemma in ks:
                row = lex[lang][lex[lang].lemma == lemma].iloc[0]
                rows.append({**row[keep + extra[lang]].to_dict(), "set": name,
                             "concept_en": r.concept_en})  # fmt: skip
                used.add((lang, lemma))

    for lang, df in lex.items():
        df = df.assign(freq_bin=df.zipf.map(freq_bin))
        for lo, hi, b in FREQ_BINS:
            for g in ("m", "f"):
                cell = df[(df.freq_bin == b) & (df.gender == g)]
                cell = cell[[(lang, w) not in used for w in cell.lemma]]
                idx = rng.sample(range(len(cell)), min(per_cell, len(cell)))
                for _, row in cell.iloc[sorted(idx)].iterrows():
                    rows.append({**row[keep + extra[lang]].to_dict(), "set": "single"})
                    used.add((lang, row.lemma))

    st = pd.DataFrame(rows)
    st["freq_bin"] = st.zipf.map(freq_bin)
    st["source"] = st.lang.map(source_tag)
    first = ["lang", "lemma", "gender", "concept_en", "set", "source", "freq_bin", "zipf"]
    st = st[first + [c for c in st.columns if c not in first]]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    st.to_csv(out, index=False)
    print(st.groupby(["lang", "set", "freq_bin", "gender"]).size().unstack(fill_value=0))
    print(f"-> {out}")
    return st
