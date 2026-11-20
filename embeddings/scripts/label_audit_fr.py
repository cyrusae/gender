"""Label audit of the upstream French noun lists (exploratory, 2026-11-20). French counterpart of the
label-audit half of label_audit.py (the Spanish ending-exception test has no French equivalent here).

Compares Zhou et al.'s (masculine_nouns_fr / feminine_nouns_fr) and Omrani Sabbaghi & Caliskan's
(fr-masc-v2.txt / fr-fem-v2.txt) French lists with our Wiktionary lexicon (data/lexicon/fr_nouns.csv):
agree / disagree / two genders in Wiktionary / listed under both genders / not a noun lemma there
(split into plurals of a lexicon noun, by a trailing -s/-x, and other). Among agreeing nouns: animacy as
the lexicon records it. Also: how often the label matches the spelling cue "ends in -e = feminine"
(fr_final_e), among agreeing vs disagreeing nouns.

    uv run --no-project --python 3.12 python embeddings/scripts/label_audit_fr.py
"""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "embeddings/external"
OUT = ROOT / "embeddings/results/label_audit_fr"


def read_list(path):
    return [w for w in (line.strip() for line in open(path, encoding="utf-8")) if w]


def audit(name, masc, fem, lex):
    rows, cat = [], Counter()
    both = set(masc) & set(fem)
    for label, words in (("m", masc), ("f", fem)):
        for w in dict.fromkeys(words):
            e = lex.get(w)
            if w in both:
                c = "listed_both"
            elif e is None:
                stem = w[:-1] if w[-1:] in ("s", "x") else None
                c = "not_a_lemma_plural" if stem and stem in lex else "not_a_lemma_other"
            elif e["gender"] == "multi":
                c = "two_gender_in_wiktionary"
            elif e["gender"] == label:
                c = "agree"
            else:
                c = "disagree"
            final_e = (e["fr_final_e"] == "True") if e else None
            rows.append({"list": name, "word": w, "label": label, "category": c,
                         "wiktionary": e["gender"] if e else "", "final_e": "" if final_e is None else final_e,
                         "spelling_cue_matches_label": "" if final_e is None else (final_e == (label == "f")),
                         "animacy": e["animacy"] if e else ""})  # fmt: skip
            cat[c] += 1
    dis = [r for r in rows if r["category"] == "disagree"]
    agr = [r for r in rows if r["category"] == "agree"]

    def cue(rs):
        rs = [r for r in rs if r["spelling_cue_matches_label"] != ""]
        return round(sum(r["spelling_cue_matches_label"] for r in rs) / len(rs), 3) if rs else None

    n = len(rows)
    summary = {
        "list_lines_masc_fem": [len(masc), len(fem)],
        "distinct_words": n,
        "categories": dict(cat),
        "share_not_a_lemma": round((cat["not_a_lemma_plural"] + cat["not_a_lemma_other"]) / n, 3),
        "disagree_share_of_lexicon_nouns": round(len(dis) / max(1, len(dis) + len(agr)), 4),
        "animacy_of_agree": dict(Counter(r["animacy"] for r in agr)),
        "spelling_cue_matches_label_among_agree": cue(agr),
        "spelling_cue_matches_label_among_disagree": cue(dis),
        "disagree_examples": [(r["word"], r["label"], r["wiktionary"]) for r in dis[:30]],
        "not_a_lemma_other_examples": [r["word"] for r in rows if r["category"] == "not_a_lemma_other"][:40],
        "animate_agree_examples": [r["word"] for r in agr if r["animacy"] == "animate"][:30],
    }
    return rows, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lex = {r["lemma"]: r for r in csv.DictReader(open(ROOT / "data/lexicon/fr_nouns.csv", encoding="utf-8"))}
    lists = {
        "zhou2019": (read_list(EXT / "Bias_in_Gendered_Languages/masculine_nouns_fr"),
                     read_list(EXT / "Bias_in_Gendered_Languages/feminine_nouns_fr")),
        "omrani2022": (read_list(EXT / "GG_Disentangling/data/nouns/fr-masc-v2.txt"),
                       read_list(EXT / "GG_Disentangling/data/nouns/fr-fem-v2.txt")),
    }  # fmt: skip
    out, allrows = {}, []
    for name, (m, f) in lists.items():
        rows, summ = audit(name, m, f, lex)
        allrows += rows
        out[name] = summ
    json.dump(out, open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False)
    with open(OUT / "word_labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
