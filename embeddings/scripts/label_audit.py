"""Exploratory CPU checks on the upstream Spanish noun lists (2026-11-08).

(1) Label audit: compare Zhou et al.'s and Omrani Sabbaghi & Caliskan's Spanish masculine/feminine
    noun lists with our Wiktionary lexicon (data/lexicon/es_nouns.csv): agree / disagree /
    two-gender in Wiktionary / not a noun lemma there; and whether disagreements cluster on nouns
    whose ending predicts the other gender.
(2) Ending-exception test: fit each paper's grammatical-gender direction the way they do (LinearSVC
    on 3,000 + 3,000 nouns from their lists, after dedup) and classify our Phase 2 sets
    (data/stimuli/phase2_final_v4.csv): regular -o/-a nouns vs exceptions (el problema, la mano,
    el día...). If the direction is mostly "ends in -o vs -a", exceptions fall below chance.

    uv run --no-project --python 3.12 --with numpy,scikit-learn python embeddings/scripts/label_audit.py
"""

import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "embeddings/external"
OUT = ROOT / "embeddings/results/label_audit_es"


def read_list(path, strip3=False):
    out = []
    for line in open(path, encoding="utf-8"):
        w = line.strip()
        if strip3:
            w = w[3:]
        if w:
            out.append(w)
    return out


def load_vec(path, nmax):
    vecs, w2i = [], {}
    with io.open(path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            w, v = line.rstrip().split(" ", 1)
            if w in w2i:
                continue
            vecs.append(np.asarray(v.split(" "), dtype=np.float32))
            w2i[w] = len(w2i)
            if len(w2i) == nmax:
                break
    return np.vstack(vecs), w2i


def audit(name, masc, fem, lex):
    rows, cat = [], Counter()
    both = set(masc) & set(fem)
    for label, words in (("m", masc), ("f", fem)):
        for w in dict.fromkeys(words):
            e = lex.get(w)
            if w in both:
                c = "listed_both"
            elif e is None:
                c = "not_a_lemma"
            elif e["gender"] == "multi":
                c = "two_gender_in_wiktionary"
            elif e["gender"] == label:
                c = "agree"
            else:
                c = "disagree"
            ending = e["es_ending"] if e else ""
            ending_says = {"o": "m", "a": "f"}.get(ending, "")
            rows.append({"list": name, "word": w, "label": label, "category": c,
                         "wiktionary": e["gender"] if e else "", "ending": ending,
                         "ending_predicts_label": ending_says == label if ending_says else "",
                         "animacy": e["animacy"] if e else ""})  # fmt: skip
            cat[c] += 1
    dis = [r for r in rows if r["category"] == "disagree"]
    agr = [r for r in rows if r["category"] == "agree"]

    def ending_rate(rs):
        rs = [r for r in rs if r["ending_predicts_label"] != ""]
        return round(sum(r["ending_predicts_label"] for r in rs) / len(rs), 3) if rs else None

    summary = {
        "distinct_words": len(rows),
        "categories": dict(cat),
        "disagree_share_of_lexicon_nouns": round(len(dis) / max(1, len(dis) + len(agr)), 4),
        "label_matches_ending_rule_among_agree": ending_rate(agr),
        "label_matches_ending_rule_among_disagree": ending_rate(dis),
        "animacy_of_agree": dict(Counter(r["animacy"] for r in agr)),
        "disagree_examples": [(r["word"], r["label"], r["wiktionary"]) for r in dis[:25]],
    }
    return rows, summary


def direction(E, w2i, masc, fem, colnorm):
    m = [w for w in dict.fromkeys(masc) if w in w2i]
    f = [w for w in dict.fromkeys(fem) if w in w2i]
    both = set(m) & set(f)
    pairs = [(a, b) for a, b in zip([w for w in f if w not in both], [w for w in m if w not in both])][:3000]
    X = np.array([E[w2i[w]] for p in pairs for w in p], dtype=np.float64)
    y = np.tile([1, 2], len(pairs))  # 1 = feminine, 2 = masculine
    scale = np.linalg.norm(X, axis=0) if colnorm else np.ones(X.shape[1])
    clf = LinearSVC(C=10).fit(X / scale, y)
    train_words = {w for p in pairs for w in p}
    return clf, scale, train_words


def exceptions(clf, scale, train_words, E, w2i, p2):
    res = {}
    for s in sorted({r["set"] for r in p2}):
        items = [r for r in p2 if r["set"] == s and r["lemma"] in w2i and r["lemma"] not in train_words]
        if not items:
            continue
        X = np.array([E[w2i[r["lemma"]]] for r in items], dtype=np.float64) / scale
        pred = clf.predict(X)
        gold = np.array([1 if r["gender"] == "f" else 2 for r in items])
        res[s] = {"n": len(items), "accuracy_vs_wiktionary_gender": round(float((pred == gold).mean()), 3)}
        if s.startswith("exception"):
            res[s]["items"] = [(r["lemma"], r["gender"], "f" if p == 1 else "m") for r, p in zip(items, pred)]
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lex = {r["lemma"]: r for r in csv.DictReader(open(ROOT / "data/lexicon/es_nouns.csv", encoding="utf-8"))}
    p2 = list(csv.DictReader(open(ROOT / "data/stimuli/phase2_final_v4.csv", encoding="utf-8")))
    lists = {
        "omrani2022": (read_list(EXT / "GG_Disentangling/data/nouns/es-masc-v2.txt"),
                       read_list(EXT / "GG_Disentangling/data/nouns/es-fem-v2.txt")),
        "zhou2019": (read_list(EXT / "Bias_in_Gendered_Languages/masculine_nouns_es"),
                     read_list(EXT / "Bias_in_Gendered_Languages/feminine_nouns_es")),
    }  # fmt: skip
    out, allrows = {}, []
    for name, (m, f) in lists.items():
        rows, summ = audit(name, m, f, lex)
        allrows += rows
        out[name] = {"label_audit": summ}
    E, w2i = load_vec(EXT / "GG_Disentangling/data/embeddings/cc.es.300.vec", 1_000_000)
    clf, scale, tw = direction(E, w2i, *lists["omrani2022"], colnorm=True)
    out["omrani2022"]["exception_test"] = exceptions(clf, scale, tw, E, w2i, p2)
    del E
    E, w2i = load_vec(EXT / "Bias_in_Gendered_Languages/data/wiki.multi.es.vec.txt", 50_000)
    clf, scale, tw = direction(E, w2i, *lists["zhou2019"], colnorm=False)
    out["zhou2019"]["exception_test"] = exceptions(clf, scale, tw, E, w2i, p2)
    json.dump(out, open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False)
    with open(OUT / "word_labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
