"""Feasibility count for the multilingual noun list (no models).

How many English nouns have a clean, common translation with a single unambiguous gender in each
candidate language, and how much do the languages disagree? One pass over the English Wiktionary
dump's translation tables; genders here are the tables' tags (gold verification against each
language's own dump comes later, for the languages we keep: this script reports how often the
tags agree with our de/es/ru lexicons).

Rules (fixed before looking at counts):
- English side: single lowercase word, noun, English zipf >= 3.0, Brysbaert concreteness known;
  WordNet first sense not noun.person / noun.animal (inanimate); not a reserved pilot noun.
- Translation table: per English word, the noun sense group with the most translations (the main
  sense). Candidates per language: entries without dialect/dated/rare/regional/obsolete/plural
  tags, single word (no spaces).
- Gendered language: candidates with zipf >= 2.5 in that language (wordfreq) count; the concept's
  gender is clean if they all share one gender (m/f/n). No common candidate, or mixed genders ->
  not usable for that language.
- Genderless language: at least one candidate with zipf >= 2.5 (Japanese: wordfreq needs MeCab,
  not installed, so any candidate counts, flagged).

Usage: uv run python instruct/scripts/feasibility.py --out instruct/results/feasibility
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parents[2]
_MAIN = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT, text=True
    ).strip()
).parent  # untracked raw data lives in the main checkout


def data(rel: str) -> Path:
    return ROOT / rel if (ROOT / rel).exists() else _MAIN / rel


sys.path.insert(0, str(ROOT / "src"))
from gbleed import paths
from gbleed.lexicon import _wordnet

GENDERED = {"de": "German", "ru": "Russian", "es": "Spanish", "hi": "Hindi",
            "fr": "French", "it": "Italian", "pt": "Portuguese", "pl": "Polish", "cs": "Czech",
            "el": "Greek", "ar": "Arabic", "he": "Hebrew", "nl": "Dutch", "sv": "Swedish"}  # fmt: skip
GENDERLESS = {"fi": "Finnish", "tr": "Turkish", "ja": "Japanese", "hu": "Hungarian"}
SETS = {
    "core4": ["de", "ru", "es", "hi"],
    "core6": ["de", "ru", "es", "hi", "fi", "ja"],
    "core7": ["de", "ru", "es", "hi", "fi", "ja", "tr"],
}
BAD_TAGS = {"dated", "rare", "regional", "obsolete", "plural", "archaic", "dialectal", "colloquial",
            "slang", "vulgar", "informal", "humorous", "literary", "poetic", "diminutive",
            "derogatory", "figuratively", "uncommon"}  # fmt: skip
GTAG = {"masculine": "m", "feminine": "f", "neuter": "n", "common": "c"}
NO_FREQ = {"ja"}


def zipf(w: str, lang: str) -> float | None:
    if lang in NO_FREQ:
        return None
    if lang == "ru":  # Wiktionary marks stress (ло́жка); wordfreq needs the plain spelling
        w = w.replace("\u0301", "").replace("\u0300", "")
    return zipf_frequency(w, lang)


def main_sense_tables(dump: Path, keep: set[str]) -> dict[str, dict]:
    """{english word: {'sense': label, 'trans': [...]}} for the largest translation group."""
    groups: dict[str, Counter] = defaultdict(Counter)
    trans: dict[tuple, list] = defaultdict(list)
    with dump.open() as fh:
        for ln in fh:
            if '"translations"' not in ln or '"pos": "noun"' not in ln:
                continue
            d = json.loads(ln)
            w = d.get("word")
            if d.get("pos") != "noun" or w not in keep:
                continue
            for s in d.get("senses", []):
                for t in s.get("translations") or []:
                    lab = t.get("sense") or (s.get("glosses") or [""])[0]
                    groups[w][lab] += 1
                    trans[(w, lab)].append(t)
            for t in d.get("translations") or []:
                lab = t.get("sense") or ""
                groups[w][lab] += 1
                trans[(w, lab)].append(t)
    out = {}
    for w, c in groups.items():
        lab = c.most_common(1)[0][0]
        out[w] = {"sense": lab, "trans": trans[(w, lab)]}
    return out


def resolve(trans: list[dict], lang: str) -> dict:
    cands = []
    for t in trans:
        if t.get("lang_code") != lang:
            continue
        tags = set(t.get("tags") or [])
        w = (t.get("word") or "").strip()
        if not w or " " in w or tags & BAD_TAGS or any("-" in x and x[0].isupper() for x in tags):
            continue  # regional variants are tagged like "Alemannic-German", "Egyptian-Arabic"
        g = {GTAG[x] for x in tags if x in GTAG}
        cands.append((w, g))
    if not cands:
        return {"status": "none"}
    scored = [(w, g, zipf(w, lang)) for w, g in cands]
    common = [x for x in scored if x[2] is None or x[2] >= 2.5]
    if not common:
        return {"status": "rare"}
    best = max(common, key=lambda x: x[2] or 0)
    res = {"word": best[0], "zipf": best[2], "n_cands": len(common)}
    if lang in GENDERLESS:
        return res | {"status": "ok"}
    gs = set().union(*(g for _, g, _ in common))
    if any(not g for _, g, _ in common) or len(gs) != 1:
        return res | {
            "status": "mixed" if len(gs) > 1 else "untagged",
            "genders": "".join(sorted(gs)),
        }
    return res | {"status": "ok", "gender": gs.pop()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    conc = pd.read_csv(
        data("data/raw/norms/Concreteness_ratings_Brysbaert_et_al_BRM.txt"), sep="\t"
    )
    conc = conc[conc.Bigram == 0].set_index("Word")["Conc.M"]
    gl = pd.read_csv(data("data/raw/glasgow/glasgow_norms_scott2019.csv"), header=[0, 1])
    gl = pd.DataFrame({"word": gl.iloc[:, 0], "val": gl.iloc[:, 5], "gend": gl.iloc[:, 26]})
    gl = gl.drop_duplicates("word").set_index("word")
    reserved = set(
        pd.read_csv(Path(__file__).parents[1] / "stimuli/reserved_nouns.csv", comment="#").noun
    )

    wn = _wordnet()
    pool = {}
    for w, c in conc.items():
        if not isinstance(w, str) or not re.fullmatch(r"[a-z]+", w) or w in reserved:
            continue
        if zipf_frequency(w, "en") < 3.0:
            continue
        syn = wn.synsets(w, pos="n")
        if not syn or syn[0].lexname() in ("noun.person", "noun.animal"):
            continue
        pool[w] = {"conc": c, "lexname": syn[0].lexname(), "zipf_en": zipf_frequency(w, "en")}
    print(f"English pool: {len(pool)} nouns", flush=True)

    tables = main_sense_tables(paths.kaikki("kaikki.org-dictionary-English.jsonl"), set(pool))
    print(f"with a translation table: {len(tables)}", flush=True)
    rows = []
    for w, tb in tables.items():
        r = {"en": w, "sense": tb["sense"][:80]} | pool[w]
        if w in gl.index:
            r |= {"val": gl.loc[w, "val"], "gend_en": gl.loc[w, "gend"]}
        for lang in [*GENDERED, *GENDERLESS]:
            x = resolve(tb["trans"], lang)
            r[f"{lang}_status"] = x["status"]
            r[f"{lang}_word"] = x.get("word", "")
            if lang in GENDERED:
                r[f"{lang}_g"] = x.get("gender", "")
        rows.append(r)
    d = pd.DataFrame(rows)
    d["conc_bin"] = pd.cut(d.conc, [0, 2.5, 3.5, 5.01], labels=["abstract", "mid", "concrete"])
    d.to_csv(args.out / "candidates.csv", index=False)

    rep = []
    rep.append(
        f"# Feasibility count\n\nEnglish pool {len(pool)}; with a translation table {len(d)}.\n"
    )
    rep.append("## Usable per language (status ok)\n")
    rep.append("| lang | name | ok | none | rare | mixed gender | untagged | gender split |")
    rep.append("|---|---|---|---|---|---|---|---|")
    for lang, name in {**GENDERED, **GENDERLESS}.items():
        st = d[f"{lang}_status"].value_counts()
        split = ""
        if lang in GENDERED:
            gc = d.loc[d[f"{lang}_status"] == "ok", f"{lang}_g"].value_counts()
            split = " / ".join(f"{k} {v}" for k, v in gc.items())
        rep.append(f"| {lang} | {name} | {st.get('ok', 0)} | {st.get('none', 0)} | {st.get('rare', 0)} "
                   f"| {st.get('mixed', 0)} | {st.get('untagged', 0)} | {split} |")  # fmt: skip

    def usable(langs):
        return d[(d[[f"{lg}_status" for lg in langs]] == "ok").all(axis=1)]

    rep.append("\n## Nouns usable in every language of a set, by concreteness\n")
    rep.append(
        "| set | languages | total | abstract (<2.5) | mid | concrete (>=3.5) | with Glasgow valence |"
    )
    rep.append("|---|---|---|---|---|---|---|")
    extra_sets = dict(SETS)
    for add in ["fr", "it", "pt", "pl", "el", "ar", "he", "nl", "cs", "sv", "hu"]:
        extra_sets[f"core7+{add}"] = SETS["core7"] + [add]
    for nm, langs in extra_sets.items():
        u = usable(langs)
        cb = u.conc_bin.value_counts()
        rep.append(f"| {nm} | {' '.join(langs)} | {len(u)} | {cb.get('abstract', 0)} | "
                   f"{cb.get('mid', 0)} | {cb.get('concrete', 0)} | {u['val'].notna().sum()} |")  # fmt: skip

    u = usable(SETS["core7"])
    g = [lg for lg in SETS["core7"] if lg in GENDERED]
    rep.append(f"\n## Gender patterns on the core7 set ({len(u)} nouns)\n")
    pat = u[[f"{lg}_g" for lg in g]].agg("".join, axis=1).value_counts()
    rep.append(f"{len(pat)} distinct patterns (order {' '.join(g)}). Most common:\n")
    rep.append("| pattern | n |\n|---|---|")
    rep += [f"| {k} | {v} |" for k, v in pat.head(15).items()]
    rep.append(
        "\n## Pairwise gender agreement (share of nouns with the same gender), all usable pairs\n"
    )
    langs = list(GENDERED)
    m = pd.DataFrame(index=langs, columns=langs, dtype=float)
    for a, b in itertools.product(langs, langs):
        both = d[(d[f"{a}_status"] == "ok") & (d[f"{b}_status"] == "ok")]
        if len(both) >= 30:
            m.loc[a, b] = round((both[f"{a}_g"] == both[f"{b}_g"]).mean(), 2)
    rep.append(m.to_markdown())
    rep.append("\n(Three-gender vs two-gender pairs can't agree on neuter nouns, so their agreement is "
               "lower by construction.)\n")  # fmt: skip

    rep.append("\n## Tag check against our Wiktionary lexicons (de/es/ru)\n")
    for lang in ["de", "es", "ru"]:
        lex = pd.read_csv(data(f"data/lexicon/{lang}_nouns.csv"), usecols=["lemma", "gender"])
        lex = lex.drop_duplicates("lemma").set_index("lemma")["gender"]
        ok = d[d[f"{lang}_status"] == "ok"]
        words = ok[f"{lang}_word"].str.replace("́", "", regex=False)  # Russian stress marks
        hit = words.isin(lex.index)
        same = (lex.reindex(words[hit]).values == ok.loc[hit, f"{lang}_g"].values).mean()
        rep.append(f"- {lang}: {hit.mean():.0%} of words found in the lexicon; gender agrees "
                   f"for {same:.1%} of those")  # fmt: skip
    (args.out / "report.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
