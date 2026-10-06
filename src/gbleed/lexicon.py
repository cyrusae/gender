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
LANG_NAMES = {"de": "German", "es": "Spanish"}
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


def judge_animacy(cats: set[str], tags: set[str], gloss: str, lexname: str, share):
    reasons = []
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
    if lexname == "noun.group":
        return "uncertain", "wordnet:noun.group"
    if share is not None and share >= 0.25:
        return "uncertain", f"wordnet-animate-share:{share:.2f}"
    if share is None:
        # No usage counts: fall back to "first sense is inanimate" only.
        return "inanimate", f"wordnet-first:{lexname}"
    return "inanimate", f"wordnet:{lexname}"


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
            if pos != "noun":
                if pos in OTHER_POS:
                    # Lowercased: wordfreq lowercases too, so "Aber" shares "aber"'s count.
                    other_pos.add((word.lower(), pos))
                continue
            if not word_re.match(word):
                continue
            core = [
                s for s in r.get("senses", [])
                if s.get("glosses") and not (set(s.get("tags", [])) & SKIP_SENSE_TAGS)
            ]  # fmt: skip
            if not core:
                continue
            genders = {GENDER_TAGS[t] for s in core for t in s.get("tags", []) if t in GENDER_TAGS}
            if not genders:  # fall back to the headword line, e.g. "Brücke f (...)"
                for h in r.get("head_templates", []):
                    head = h.get("expansion", "").split("(")[0]
                    genders |= set(re.findall(r"\b([mfn])\b", head))
            # Topical categories of the FIRST sense only: "Buch" has a minor sense
            # (omasum) filed under "Animal body parts".
            s0 = core[0]
            cats = {c["name"] for c in s0.get("categories", []) if isinstance(c, dict)}
            cats |= set(s0.get("topics", []) or [])
            entries[word].append(
                {
                    "genders": genders,
                    "gloss": core[0]["glosses"][0],
                    "first_tags": set(core[0].get("tags", [])),
                    "tags": {t for s in core for t in s.get("tags", [])},
                    "cats": cats,
                    "etymology": r.get("etymology_text", ""),
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
        animacy, why = judge_animacy(cats, tags, first["gloss"], lexname, share)
        gender = next(iter(genders)) if len(genders) == 1 else "multi"
        row = {
            "lang": lang,
            "lemma": word,
            "gender": gender,
            "genders": "|".join(sorted(genders)),
            "concept_en": concept,
            "gloss": first["gloss"],
            "zipf": zipf_frequency(word, lang),
            "animacy": animacy,
            "animacy_reason": why,
            "wordnet": lexname,
            "marked": "|".join(
                sorted((first["first_tags"] & MARKED_TAGS) | lexical_class(first["gloss"], cats))
            ),
            # Region tags are the capitalised ones (Austria, Mexico, Latin-America...).
            "regions": "|".join(sorted(t for t in first["first_tags"] if t[:1].isupper())),
            "also_pos": "|".join(sorted(other_words.get(word.lower(), ()))),
            "n_entries": len(ents),
        }
        if lang == "de":
            row["de_suffix"] = de_suffix(word)
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
    """Clean, inanimate, single-gender m/f nouns usable as stimuli."""
    m = (
        df.gender.isin(["m", "f"])
        & (df.animacy == "inanimate")
        & (df.marked == "")
        & (df.regions == "")
        & (df.also_pos == "")
    )
    if "initial_a_f" in df:
        m &= ~df.initial_a_f.astype(str).eq("True")
    return df[m]


def build_pairs(min_zipf: float = 2.5) -> pd.DataFrame:
    """German-Spanish translation pairs matched on normalised English gloss.

    For each concept, the most frequent eligible lemma in each language is
    taken. This is a heuristic: check the two `gloss` columns before relying
    on a pair."""
    best = {}
    for lang in ("de", "es"):
        el = eligible(load_lexicon(lang))
        el = el[el.zipf >= min_zipf]
        n = el.groupby("concept_en").size().rename("n_candidates")
        top = el.sort_values("zipf", ascending=False).drop_duplicates("concept_en")
        top = top.set_index("concept_en").join(n)
        best[lang] = top[["lemma", "gender", "gloss", "zipf", "n_candidates"]].add_prefix(
            f"{lang}_"
        )
    pairs = best["de"].join(best["es"], how="inner").reset_index()
    pairs["flipped"] = pairs.de_gender != pairs.es_gender
    # Shared spelling (Apartheid/apartheid): the "translation" is just the same word.
    pairs["cognate"] = [
        difflib.SequenceMatcher(None, d.lower(), e.lower()).ratio() >= 0.8
        for d, e in zip(pairs.de_lemma, pairs.es_lemma, strict=True)
    ]
    pairs = pairs.sort_values(["flipped", "de_zipf"], ascending=False)
    pairs.to_csv(LEX_DIR / "pairs_de_es.csv", index=False)
    print(f"{len(pairs)} pairs ({int(pairs.flipped.sum())} flipped) -> {LEX_DIR}/pairs_de_es.csv")
    return pairs


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
) -> pd.DataFrame:
    """per_cell nouns per (language, frequency bin, gender), plus up to
    max_pairs flipped and max_pairs control translation pairs."""
    rng = random.Random(seed)
    exclude = exclude or set()
    keep = ["lang", "lemma", "gender", "concept_en", "gloss", "zipf"]
    extra = {"de": ["de_suffix"], "es": ["es_ending", "es_regular", "es_exception"]}
    lex = {lang: eligible(load_lexicon(lang)) for lang in ("de", "es")}
    rows, used = [], set(exclude)

    pairs = build_pairs()
    for flipped, name in ((True, "flipped"), (False, "control")):
        sub = pairs[(pairs.flipped == flipped) & ~pairs.cognate]
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
