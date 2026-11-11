"""XWEAT sensitivity: how fragile are Lauscher & Glavas's (2019) cross-language gender scores?

Lauscher & Glavas (*SEM 2019, Table 5) report WEAT T7 (math vs arts words x male vs female words)
and T8 (science vs arts x male vs female) for seven languages on fastText Wikipedia vectors (200k
vocabulary, lowercased). Their translated lists live in `external/XWEAT` (umanlp/XWEAT, commit
4f95484, no licence file): the English lists are parsed from `weat.py` with `ast` (not imported),
the translations are read from their own pickles with an unpickler that refuses any code object.
Nothing from the clone is copied into tracked files.

Arms (rules fixed in `docs/xweat-sensitivity-prereg.md` before any score was computed):
  repro   their pipeline: translate (both gender forms when given; capitalised English words left
          untranslated), de-duplicate, lowercase, drop out-of-vocabulary words, then pop words from
          the longer set until sizes match. Which word is popped follows Python's hash-randomised
          set order, so every possible pop is scored and the range reported.
  clean   mechanical fixes only: drop words that are in both attribute (or both target) sets,
          multi-word items, and untranslated English proper names (non-English languages); no
          popping (unequal set sizes are fine for the effect size and the permutation test).
  loo     clean, minus one word at a time, for every word of every set.
  within  (languages with a Wiktionary lexicon: de, es, ru) clean target sets restricted to one
          grammatical gender on both sides; scored only when each side keeps >= 3 words.

Effect size and p-value follow their code: per-word association s(w) = mean sim(w, A1) - mean
sim(w, A2); effect = (mean s over T1 - mean s over T2) / population SD of s over T1 and T2;
one-sided p = share of re-splits of T1 + T2 whose statistic is strictly larger (exact, all splits).

    uv run --no-project --python 3.12 --with numpy,pandas \
        python embeddings/scripts/xweat_sensitivity.py [--langs en es it] [--similarity euclidean]
"""

import argparse
import ast
import itertools
import json
import pickle
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
XWEAT = ROOT / "embeddings/external/XWEAT"
MUSE = Path("/Volumes/Lexar/gbleed/shared/embeddings/muse")
OUT = ROOT / "embeddings/results/xweat_sensitivity"
LANGS = ["en", "de", "es", "it", "hr", "ru", "tr"]
LEXICON_LANGS = {"de", "es", "ru"}
TESTS = (7, 8)
SETS = ("T1", "T2", "A1", "A2")
# Lauscher & Glavas 2019, Table 5 (fastText Wikipedia; * = not significant at 0.05)
PAPER = {
    7: {"en": 1.30, "de": 0.46, "es": 1.47, "it": 1.00, "hr": 0.72, "ru": 0.59, "tr": -0.88},
    8: {"en": 1.30, "de": 0.05, "es": 1.16, "it": 0.10, "hr": 0.13, "ru": 0.37, "tr": 1.72},
}
PAPER_NS = {(7, "de"), (7, "hr"), (7, "ru"), (8, "de"), (8, "it"), (8, "hr"), (8, "ru")}
MIN_WITHIN = 3


class _NoCode(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing {module}.{name}")


def nfc(s):
    return unicodedata.normalize("NFC", s)


def english_lists(xweat):
    """T1, T2, A1, A2 of weat_7 / weat_8, read from their source without executing it."""
    tree = ast.parse((xweat / "weat.py").read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {f"weat_{t}" for t in TESTS}:
            names = {
                "targets_1": "T1",
                "targets_2": "T2",
                "attributes_1": "A1",
                "attributes_2": "A2",
            }
            lists = {}
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and stmt.targets[0].id in names:
                    lists[names[stmt.targets[0].id]] = ast.literal_eval(stmt.value)
            out[int(node.name.split("_")[1])] = lists
    assert set(out) == set(TESTS) and all(len(v) == 4 for v in out.values()), out
    return out


def translations(xweat, lang):
    if lang == "en":
        return None
    with open(xweat / f"data/vocab_dict_en_{lang}.p", "rb") as f:
        return _NoCode(f).load()


def translate(tdict, terms):
    """Their translate(): returns [(english source, form)], de-duplicated on form, then lowered."""
    pairs = []
    for t in terms:
        key = t.lower() if tdict and t.lower() in tdict else t if tdict and t in tdict else None
        if key is None:
            pairs.append((t, t))  # untranslated (always the case for English)
            continue
        male, female = tdict[key]
        pairs.append((t, male))
        if female:
            pairs.append((t, female))
    seen, out = set(), []
    for src, form in pairs:  # their list(set(...)) + .lower(); order irrelevant here
        word = nfc(form.lower())
        if word not in seen:
            seen.add(word)
            out.append({"en": src, "form": form, "word": word})
    return out


def load_vectors(path, want, nmax):
    vecs = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= nmax:
                break
            w, rest = line.rstrip("\n").split(" ", 1)
            w = nfc(w)
            if w in want and w not in vecs:
                v = np.array(rest.split(), dtype=float)
                vecs[w] = v / np.linalg.norm(v)
    return vecs


def sim(W, A, kind):
    if kind == "cosine":
        return W @ A.T
    d = np.sqrt(np.maximum(2 - 2 * (W @ A.T), 0))  # unit vectors: ||w - a||^2 = 2 - 2 cos
    return 1 / (1 + d)


def assoc(V, words, a1, a2, kind):
    W = np.array([V[w] for w in words])
    return sim(W, np.array([V[a] for a in a1]), kind).mean(1) - sim(
        W, np.array([V[a] for a in a2]), kind
    ).mean(1)


def effect(sx, sy):
    return (sx.mean() - sy.mean()) / np.concatenate([sx, sy]).std()


def p_exact(sx, sy):
    s = np.concatenate([sx, sy])
    idx = np.array(list(itertools.combinations(range(len(s)), len(sx))))
    stats = 2 * s[idx].sum(1) - s.sum()
    return float((stats > sx.sum() - sy.sum()).mean())


def score(V, sets, kind):
    s1 = assoc(V, sets["T1"], sets["A1"], sets["A2"], kind)
    s2 = assoc(V, sets["T2"], sets["A1"], sets["A2"], kind)
    return effect(s1, s2), p_exact(s1, s2)


def lexicon_gender(lang):
    path = ROOT / f"data/lexicon/{lang}_nouns.csv"
    if lang not in LEXICON_LANGS or not path.exists():
        return None
    lex = pd.read_csv(path, keep_default_na=False)
    exact = dict(zip(lex["lemma"], lex["gender"], strict=True))
    folded = {}
    for lemma, g in exact.items():
        folded.setdefault(nfc(lemma).casefold(), set()).add(g)

    def look(form):
        g = exact.get(form)
        if g is None:
            gs = folded.get(nfc(form).casefold(), set())
            g = next(iter(gs)) if len(gs) == 1 else None
        return g if g in {"m", "f", "n"} else "unknown"

    return look


def build(lang, eng, tdict, vocab_words):
    """Per-word records for one language and test, with repro and clean status."""
    trans = {k: translate(tdict, eng[k]) for k in SETS}
    words = {k: {r["word"] for r in trans[k]} for k in SETS}
    both_a, both_t = words["A1"] & words["A2"], words["T1"] & words["T2"]
    recs = []
    for k in SETS:
        for r in trans[k]:
            w = r["word"]
            oov = w not in vocab_words
            reason = ""
            if " " in w:
                reason = "multi-word"
            elif w in (both_a if k[0] == "A" else both_t):
                reason = "in both sets"
            elif lang != "en" and r["en"][:1].isupper() and r["form"] == r["en"]:
                reason = "untranslated English name"
            elif oov:
                reason = "not in vocabulary"
            recs.append({**r, "set": k, "in_vocab": not oov, "clean_kept": not reason,
                         "drop_reason": reason})  # fmt: skip
    return recs


def repro_options(kept):
    """Every way their popping could equalise T1/T2 and A1/A2 sizes."""

    def opts(a, b):
        if len(a) == len(b):
            return [(a, b)]
        long_, short = (a, b) if len(a) > len(b) else (b, a)
        out = [(list(c), short) for c in itertools.combinations(long_, len(short))]
        return out if len(a) > len(b) else [(s, c) for c, s in out]

    for (t1, t2), (a1, a2) in itertools.product(
        opts(kept["T1"], kept["T2"]), opts(kept["A1"], kept["A2"])
    ):
        yield {"T1": t1, "T2": t2, "A1": a1, "A2": a2}


def run_lang(lang, eng_all, args, rows, loo_rows, word_rows, info):
    vec = args.muse_dir / f"wiki.multi.{lang}.vec"
    if not vec.exists():
        info["missing"].append(str(vec))
        return
    tdict = translations(args.xweat_dir, lang)
    want = {r["word"] for t in TESTS for k in SETS for r in translate(tdict, eng_all[t][k])}
    V = load_vectors(vec, want, args.nmax)
    info["vectors"][lang] = {"path": str(vec), "bytes": vec.stat().st_size, "found": len(V)}
    gender = lexicon_gender(lang)
    for t in TESTS:
        recs = build(lang, eng_all[t], tdict, set(V))
        base = {"lang": lang, "test": t, "paper": PAPER[t][lang],
                "paper_sig": (t, lang) not in PAPER_NS}  # fmt: skip

        kept = {k: [r["word"] for r in recs if r["set"] == k and r["in_vocab"]] for k in SETS}
        res = [score(V, o, args.similarity) for o in repro_options(kept)]
        es = np.array([e for e, _ in res])
        ps = np.array([p for _, p in res])
        rows.append({**base, "arm": "repro", **{f"n_{k}": len(kept[k]) for k in SETS},
                     "effect": float(np.median(es)), "p": float(np.median(ps)),
                     "effect_min": es.min(), "effect_max": es.max(), "p_max": ps.max(),
                     "n_options": len(res)})  # fmt: skip

        clean = {k: [r["word"] for r in recs if r["set"] == k and r["clean_kept"]] for k in SETS}
        e, p = score(V, clean, args.similarity)
        rows.append({**base, "arm": "clean", **{f"n_{k}": len(clean[k]) for k in SETS},
                     "effect": e, "p": p, "n_options": 1})  # fmt: skip

        for k in SETS:
            for w in clean[k]:
                sub = {**clean, k: [x for x in clean[k] if x != w]}
                e2, p2 = score(V, sub, args.similarity)
                loo_rows.append({"lang": lang, "test": t, "set": k, "dropped": w,
                                 "effect": e2, "p": p2, "delta": e2 - e})  # fmt: skip

        s_clean = {}
        for k in ("T1", "T2"):
            if clean[k]:
                s = assoc(V, clean[k], clean["A1"], clean["A2"], args.similarity)
                s_clean.update(zip(clean[k], s, strict=True))
        for r in recs:
            g = "none" if lang == "en" else gender(r["form"]) if gender else "no lexicon"
            word_rows.append({"lang": lang, "test": t, **r, "gram_gender": g,
                              "s_clean": s_clean.get(r["word"]) if r["clean_kept"] else None})  # fmt: skip

        if gender:
            g_of = {r["word"]: gender(r["form"]) for r in recs}
            for g in ("m", "f", "n"):
                sub = {**clean, **{k: [w for w in clean[k] if g_of[w] == g] for k in ("T1", "T2")}}
                n1, n2 = len(sub["T1"]), len(sub["T2"])
                ok = min(n1, n2) >= MIN_WITHIN
                e, p = score(V, sub, args.similarity) if ok else (None, None)
                rows.append({**base, "arm": f"within_{g}", "n_T1": n1, "n_T2": n2,
                             "n_A1": len(sub["A1"]), "n_A2": len(sub["A2"]),
                             "effect": e, "p": p, "n_options": 1})  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--langs", nargs="+", default=LANGS)
    ap.add_argument("--muse-dir", type=Path, default=MUSE)
    ap.add_argument("--xweat-dir", type=Path, default=XWEAT)
    ap.add_argument("--similarity", choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--nmax", type=int, default=200_000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    eng_all = english_lists(args.xweat_dir)
    commit = subprocess.run(["git", "-C", str(args.xweat_dir), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()  # fmt: skip
    info = {"xweat_commit": commit, "similarity": args.similarity, "nmax": args.nmax,
            "vectors": {}, "missing": []}  # fmt: skip
    rows, loo_rows, word_rows = [], [], []
    for lang in args.langs:
        run_lang(lang, eng_all, args, rows, loo_rows, word_rows, info)

    out = args.out if args.similarity == "cosine" else args.out / args.similarity
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(loo_rows).to_csv(out / "leave_one_out.csv", index=False)
    pd.DataFrame(word_rows).to_csv(out / "per_word.csv", index=False)
    (out / "info.json").write_text(json.dumps(info, indent=1, default=str))

    if info["missing"]:
        print("missing vectors (https://dl.fbaipublicfiles.com/arrival/vectors/<file>):")
        print("  " + ", ".join(Path(m).name for m in info["missing"]))
    if summary.empty:
        return
    wide = summary[summary["arm"].isin(["repro", "clean"])].pivot_table(
        index=["test", "lang"], columns="arm", values="effect"
    )
    wide["paper"] = [PAPER[t][lg] for t, lg in wide.index]
    print(wide.round(2).to_string())
    print(f"-> {out}/ (summary, leave_one_out, per_word, info)")


if __name__ == "__main__":
    main()
