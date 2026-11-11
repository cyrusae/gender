"""Candidate person-noun pairs for the non-occupation baseline (docs/wordpair-persons-prereg.md, DRAFT).
Builds a list by rule for PI review; nothing is scored here.

Question: is masculine intrusion in the word-pair test specific to occupations, or does it hold for
any gendered person noun (generic-masculine / lemma frequency)? Adapted from the non-blind Gemini
notebook's section 4, fixing its problems (sex-specific English glosses, homographs, 3 cues).

Rules (Spanish, kaikki.org Wiktionary dump, part of speech noun):
- headword = masculine singular, lowercase letters; feminine = form tagged exactly ["feminine"],
  different from the headword; both forms in the first 50k MUSE words.
- English gloss = first sense's first gloss, first comma/semicolon item, parentheses removed; one
  word in the first 200k MUSE English words.
- sex-neutral English: gloss not in Bolukbasi et al.'s gender-specific list (debiaswe
  gender_specific_full.json), so "brother", "son", "king" are out.
- person: the English gloss's first Wiktionary noun sense mentions a person (person, someone, one who,
  member, inhabitant…), or it is in Wiktionary's People category, or in Bolukbasi et al.'s
  professions list, or the Spanish entry is in a People/Occupations/Demonyms/Family category.
- not male-specific in English: first sense doesn't mention male/man/boy/husband… and the gloss doesn't
  end in -man (catches duke, frenchman, which the gender-specific list misses).
- group (by rule): occupation = Wiktionary category containing "occupation", or the pair is in Zhou
  et al.'s occupation list; demonym = category Demonyms or Nationalities; other = everything else
  (person-ness decided at PI review).
Flags for review: name_homograph (either form is a Spanish given name in Wiktionary), fem_also_noun
(the feminine form has a noun sense other than "female equivalent of ...", e.g. química 'chemistry'), also_adj (the masculine form
has an adjective entry; demonyms nearly always do), glasgow_gend of the English gloss.

    uv run --no-project --python 3.12 --with pandas python embeddings/scripts/wp_persons_build.py
"""

import json
import re
from pathlib import Path

import pandas as pd
from wp_cues_build import KK, gloss, glasgow, vocab

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"^[a-záéíóúñü]+$")


PERSON = re.compile(r"\b(person|persons|someone|somebody|one who|individual|people|member|human|inhabitant|native|resident|adherent|follower|player|child)\b")
MALE = re.compile(r"\b(male|man|men|boy|husband|father|son|brother|gentleman)\b")

# Claude's suggestions for PI review (claude-unverified), overriding the rule label. Keys are the
# masculine form. occupation / person (non-occupation) / identity (demonyms, religions, peoples) /
# drop: reason.
OCC = """académico bancario biólogo botánico astrofísico autónomo carrero carretero catedrático comisario
consultor crítico cómico diplomático directivo ejecutivo filósofo funcionario físico gestor gobernador
hotelero ilustrador impresor marinero ministro senador político presentador ranchero rapero publicitario
diputado parlamentario táctico nadador goleador jugador luchador narrador barbero conductor entrenador
locutor reportero productor coordinador misionero agregado presidente concesionario rector explorador casero criado hechicero""".split()
PERSON_ = """acogido acusado adicto adulto afiliado alcohólico alumno amigo antecesor asesino asociado bruto
burgués candidato cautivo ciudadano colaborador compañero competidor condenado conocido converso creador
defensor detenido devoto egresado graduado enamorado enano enemigo escapado esclavo evacuado experto
entendido famoso fanático fundador gemelo heredado huérfano héroe iniciado innovador invasor libertario
millonario motorizado mudo místico nieto niño nominado norteño oriundo patrono patrón peregrino pesado
pionero preso primogénito prisionero protector querido revolucionario rústico socio soltero sospechoso
subordinado sureño sustituto trepador usuario vago vecino vegetariano veterano villano voluntario
invitado ganador fracasado dueño propietario poseedor bebe aficionado adoptado anfitrión opositor partidario
preferido sucesor predecesor precursor profano campeón fallecido extranjero extraño guardián libertador
tirano soberano homólogo lector delegado enviado curioso guerrero olímpico""".split()
DROP = {"caballo": "animal", "cerdo": "animal", "loro": "animal", "perro": "animal", "gato": "animal",
        "lobo": "animal", "zorro": "animal", "oso": "animal", "pato": "animal", "mono": "animal",
        "gallo": "animal", "becerro": "animal", "ante": "animal", "paco": "animal", "muñeco": "object",
        "mago": "gloss wrong (magician, not farmer)", "superior": "gloss wrong (mother superior)",
        "interno": "gloss doubtful (intern/inmate)", "ecléctico": "not a person noun in practice",
        "míster": "male title", "señor": "male title", "virrey": "male title", "vizconde": "male title",
        "cartero": "postman (male-marked English)", "retrasado": "slur", "gitano": "slur sense",
        "mano": "colloquial", "lolo": "colloquial", "cuco": "name / bogeyman", "malo": "devil sense",
        "rojo": "colour sense", "rudo": "wrestling jargon", "divo": "star sense", "dios": "deity",
        "humano": "adjective sense dominates", "desaparecido": "adjective sense", "desconocido": "adjective sense",
        "designado": "gloss wrong", "habilitado": "gloss doubtful", "iluminado": "gloss wrong",
        "ingenuo": "adjective sense", "extremo": "sports position", "delantero": "sports position",
        "decano": "gloss doubtful", "galeno": "colloquial", "marciano": "fictional", "carnívoro": "animal sense",
        "bárbaro": "name homograph (Bárbara)", "sereno": "rare sense", "llanero": "regional", "cholo": "slur sense",
        "colector": "gloss doubtful", "distribuidor": "company sense", "portador": "carrier (thing sense)",
        "regulador": "thing sense", "trepador": "colloquial", "directivo": "adjective sense"}  # fmt: skip
IDENTITY_CATS = ("demonym", "nationalit", "religion", "christianity", "catholicism", "peoples", "ethnic")
IDENTITY = """anglicano católico copto judío musulmán luterano pagano salesiano republicano cristiano ateo
agustino benedictino celestino monárquico patricio mestizo""".split()


def english_info(words):
    """First noun-sense gloss and categories of each English word (kaikki English dump)."""
    out = {}
    with open(KK / "kaikki.org-dictionary-English.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"pos": "noun"' not in line:
                continue
            i = line.find('"word": "')
            w = line[i + 9:line.find('"', i + 9)]
            if w.lower() not in words or w.lower() in out:
                continue
            j = json.loads(line)
            if j.get("word") != w or not j.get("senses"):
                continue
            w = w.lower()
            if w in out:
                continue
            cats = {c.get("name", "") for c in j.get("categories", [])}
            cats |= {c.get("name", "") for s in j["senses"] for c in (s.get("categories") or [])}
            out[w] = (" ".join(j["senses"][0].get("glosses", [""])).lower(), cats)
    return out


def main():
    V, en = vocab("es", 50_000), vocab("en", 200_000)
    gl = glasgow()
    spec = {w.lower() for w in json.load(open(ROOT / "external/debiaswe/data/gender_specific_full.json"))}
    mp = pd.read_csv(ROOT / "data/occupation_mapping_v1.csv")
    zhou = set(mp[mp.lang == "es"].masc)
    nouns, adjs, names, own_sense, cands = set(), set(), set(), set(), {}
    with open(KK / "kaikki.org-dictionary-Spanish.jsonl", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            w, pos = j.get("word", ""), j.get("pos")
            if pos == "name":
                if any("given name" in " ".join(s.get("glosses", [])).lower() or "given-name" in s.get("tags", [])
                       for s in j.get("senses", [])):  # fmt: skip
                    names.add(w.lower())
                continue
            if pos == "adj":
                adjs.add(w)
                continue
            if pos != "noun":
                continue
            nouns.add(w)
            if any(not re.match(r"(female|feminine) (equivalent|form|version) of", " ".join(se.get("glosses", [""])).lower())
                   for se in j.get("senses", [])):  # fmt: skip
                own_sense.add(w)
            if w in cands or not WORD.match(w) or w not in V:
                continue
            fem = next((x["form"] for x in j.get("forms", []) if x.get("tags") == ["feminine"]), None)
            if not fem or fem == w or fem not in V:
                continue
            cats = {c.get("orig", c.get("name", "")) for c in j.get("categories", [])}
            cats |= {c.get("orig", c.get("name", "")) for s in j.get("senses", []) for c in (s.get("categories") or [])}
            cands[w] = (fem, gloss(j), {c for c in cats if c.startswith("es:")})
    prof = {p[0] for p in json.load(open(ROOT / "external/debiaswe/data/professions.json"))}
    info = english_info({e for _, e, _ in cands.values()})
    rows = []
    for m, (f, e, cats) in cands.items():
        if " " in e or e not in en or e in spec:
            continue
        lc = " ".join(cats).lower()
        eg, ecats = info.get(e, ("", set()))
        person = bool(PERSON.search(eg)) or "People" in ecats or e in prof or any(
            k in lc for k in ("people", "occupation", "demonym", "nationalit", "family", "children"))
        male_spec = bool(MALE.search(eg)) or (e.endswith("man") and e not in ("human", "woman", "german", "layman"))
        grp = ("occupation" if ("occupation" in lc or m in zhou) else
               "demonym" if ("demonym" in lc or "nationalit" in lc) else "other")  # fmt: skip
        rows.append({"lang": "es", "masc": m, "fem": f, "en": e, "group": grp, "rank_m": V[m], "rank_f": V[f],
                     "name_homograph": m in names or f in names, "fem_also_noun": f in own_sense,
                     "also_adj": m in adjs, "en_person": person, "en_male_specific": male_spec, "en_gloss": eg[:120], "glasgow_gend": gl.get(e), "categories": "; ".join(sorted(cats)),
                     "pi_keep": ""})  # fmt: skip
    d = pd.DataFrame(rows).sort_values(["group", "masc"])
    d["rule_keep"] = d.en_person & ~d.en_male_specific & ~d.name_homograph

    def suggest(r):
        if r.masc in DROP:
            return "drop: " + DROP[r.masc]
        if r.name_homograph:
            return "drop: name homograph"
        if r.masc in OCC or r.group == "occupation":
            return "occupation"
        if r.masc in PERSON_:
            return "person"
        if r.masc in IDENTITY or r.group == "demonym":
            return "identity"
        return "identity" if r.rule_keep and r.group == "other" else ("check: rule says not a person or male-specific (first English sense only)" if not r.rule_keep else "person")

    d["claude_suggest_unverified"] = d.apply(suggest, axis=1)
    d.to_csv(ROOT / "data/wp_persons_es_candidates_v0.csv", index=False)
    print(d.groupby("group").size().to_dict(), "names", int(d.name_homograph.sum()),
          "fem_also_noun", int(d.fem_also_noun.sum()))  # fmt: skip
    print(d.groupby(["group", "rule_keep"]).size().to_dict())
    print(d.claude_suggest_unverified.str.split(":").str[0].value_counts().to_dict())
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(d[d.rule_keep][["group", "masc", "fem", "en"]].to_string(index=False))
        print("DROPPED other:", " ".join(d[(d.group == "other") & ~d.rule_keep].en))


if __name__ == "__main__":
    main()
