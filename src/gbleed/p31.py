"""P31: the German version of P30 (exploratory; `docs/design/p31-german-object-context-prereg.md`).

Does *die* (vs *der*) on a made-up German noun pull toward feminine-rated adjectives when the noun
is clearly a thing? German predicative adjectives don't inflect (*ist sehr groß* for every
gender), so no agreement can leak and every listed adjective can be read.

Stimuli (built once, versioned):
  - nonce nouns: Wuggy (orthographic_german) on Phase 3 training nouns (shape only), keeping
    pseudowords whose last 1, 2 and 3 letters each have no gender majority in the lexicon
    (classes with >= 50 nouns), absent from German Wiktionary, zero frequency in six languages;
  - person nouns: adjectival-declension nouns (*der/die Angestellte*) by rule from Wiktionary.
Readout: first tokens of the German translation adjectives, kept when >= 3 letters and not shared
with another listed adjective (sensitivity: single-token adjectives only).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import paths
from .models import model_slug
from .p29 import _boot

OUT_ROOT = "results/p31"
NONCE_DE = "data/stimuli/p31_nonce_de_v1.csv"
PERSON_DE = "data/stimuli/p31_person_nouns_de_v1.csv"
N_NONCE = 300
SEED = 31
# past-participle shape (ge-...-t, or an inseparable prefix ...-t: gekauft, verkauft): could read
# as a verb form or adjective rather than a plain noun
PARTICIPLE = re.compile(r"(?i)(?:ge|be|ver|er|zer|ent|emp)\w*t$")
ENDING_MIN_NOUNS = 50  # an ending class needs this many lexicon nouns to count as evidence
ENDING_MAX_SHARE = 0.5  # ... and then no gender may hold more than this share
# German articles and relative pronouns by gender: nominative article; relative pronoun in the
# accusative (object of kaufen/legen/kennenlernen) and in the dative (after mit)
ART = {"m": "Der", "f": "Die", "n": "Das"}
REL_ACC = {"m": "den", "f": "die", "n": "das"}
REL_DAT = {"m": "dem", "f": "der", "n": "dem"}
CONTEXTS = {  # wording: (context, template); fixed on grammatical grounds
    "O1": ("object", "{art} {N}, {acc} ich gestern gekauft habe, ist sehr"),
    "O2": ("object", "{art} {N}, {acc} ich in die Schublade gelegt habe, ist sehr"),
    "N1": ("neutral", "{art} {N} ist sehr"),
    "P1": ("person", "{art} {N}, mit {dat} ich gestern gesprochen habe, ist sehr"),
    "P2": ("person", "{art} {N}, {acc} ich gestern kennengelernt habe, ist sehr"),
}
PROBE_PERSON = (" ein Mensch", " eine Person", " ein Mann", " eine Frau")
PROBE_THING = (" ein Ding", " ein Gegenstand", " ein Objekt", " eine Sache")
PROBE_NARROW = (" eine Person", " ein Ding")  # comparable with P30's Spanish probe
C1_MARGIN = 1.0
SHARE_MIN = 0.05


# ---- stimuli ---------------------------------------------------------------------------------
def _german_forms() -> set[str]:
    """Every German Wiktionary word and inflected form, lowercased."""
    out = set()
    with open(paths.kaikki("kaikki.org-dictionary-German.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("lang_code") != "de":
                continue
            out.add(r.get("word", "").lower())
            out |= {fm.get("form", "").lower() for fm in r.get("forms", [])}
    return out


def ending_table() -> pd.DataFrame:
    """Gender shares by last 1-3 letters over the single-gender German lexicon nouns."""
    lex = pd.read_csv("data/lexicon/de_nouns.csv", keep_default_na=False)
    lex = lex[lex.gender.isin(["m", "f", "n"])]
    rows = []
    for k in (1, 2, 3):
        e = lex.lemma.str[-k:].str.lower()
        g = lex.groupby(e).gender.value_counts(normalize=True).unstack(fill_value=0)
        g["count"] = lex.groupby(e).size()
        g["top"] = g[["f", "m", "n"]].max(axis=1)
        g["k"] = k
        rows.append(g.reset_index(names="ending"))
    return pd.concat(rows, ignore_index=True)


def ending_ok(word: str, table: dict) -> bool:
    """No gender majority for the last 1, 2 and 3 letters (where the lexicon has evidence).
    `table`: (k, ending) -> (count, top share), from `ending_index`."""
    for k in (1, 2, 3):
        count, top = table.get((k, word[-k:].lower()), (0, 0.0))
        if count >= ENDING_MIN_NOUNS and top > ENDING_MAX_SHARE:
            return False
    return True


def ending_index() -> dict:
    t = ending_table()
    return {(int(r.k), r.ending): (int(r.count), float(r.top)) for r in t.itertuples()}


def german_templates(n: int = 600) -> list[str]:
    """Shape templates: common inanimate single-gender German lexicon nouns (Zipf >= 3.5, letters
    only), never a held-out test word of any phase (they only give Wuggy a shape)."""
    lex = pd.read_csv("data/lexicon/de_nouns.csv", keep_default_na=False)
    lex = lex[lex.gender.isin(["m", "f", "n"]) & (lex.animacy == "inanimate") & (lex.zipf >= 3.5)
              & lex.lemma.str.fullmatch(r"[A-ZÄÖÜ][a-zäöüß]{2,9}")]  # fmt: skip
    test = set()
    for f in ("data/stimuli/phase3_final_v3.csv", "data/stimuli/phase3_pool_v4.csv"):
        s = pd.read_csv(f, keep_default_na=False)
        test |= set(s[s.split == "test"].lemma) if "split" in s else set()
    pairs = pd.read_csv("data/lexicon/pairs_de_es.csv", keep_default_na=False)
    test |= set(pairs.de_lemma)
    lex = lex[~lex.lemma.isin(test)].sort_values("zipf", ascending=False)
    return lex.lemma.head(n).tolist()


def build_nonce_de(n: int = N_NONCE, ncand: int = 20) -> pd.DataFrame:
    import random

    from .nonce5 import _generate, _generator, _zero_freq

    templates = german_templates()
    g = _generator("orthographic_german")
    res, skipped = _generate(g, templates, ncand)
    print(f"German templates: {len(templates)}, not in Wuggy's lexicon: {len(skipped)}")
    forms, table = _german_forms(), ending_index()
    seen, rows = set(), []
    for r in res:
        pw = r["pseudoword"]
        if not pw or pw in seen or not re.fullmatch(r"[A-ZÄÖÜ][a-zäöüß]{2,9}", pw):
            continue
        seen.add(pw)
        if (
            PARTICIPLE.match(pw)
            or pw.lower() in forms
            or not _zero_freq(pw)
            or not ending_ok(pw, table)
        ):
            continue
        rows.append({"word": pw, "template": r["word"]})
    rng = random.Random(SEED)
    rng.shuffle(rows)
    df = pd.DataFrame(rows[:n])
    df["source"] = ("wuggy orthographic_german; no participle shape; endings without a gender majority (last 1-3 "
                    "letters); absent from de Wiktionary + zero wordfreq in es/en/pt/it/fr/de")  # fmt: skip
    df.to_csv(NONCE_DE, index=False)
    print(f"German nonce: {len(seen)} candidates, {len(rows)} pass, kept {len(df)}")
    return df


def build_person_de() -> pd.DataFrame:
    """Adjectival-declension person nouns (*der/die Angestellte*) by rule: Wiktionary de-noun with
    adjectival declension, feminine entry in -e with a masculine form, gloss 'female equivalent
    of X: female Y' with Y a WordNet person noun; no digits/hyphens; not a readout adjective's
    form; Zipf >= 3."""
    from wordfreq import zipf_frequency

    from .lexicon import _wordnet
    from .phase5_stimuli import ADJ_TRANS

    wn = _wordnet()
    adj = set(pd.read_csv(ADJ_TRANS, keep_default_na=False).de.str.lower()) - {""}
    rows = []
    with open(paths.kaikki("kaikki.org-dictionary-German.jsonl"), encoding="utf-8") as f:
        for line in f:
            if '"+"' not in line:
                continue
            d = json.loads(line)
            if d.get("pos") != "noun":
                continue
            ht = [h for h in d.get("head_templates", []) if h.get("name") == "de-noun"]
            if not ht or ht[0].get("args", {}).get("1") != "+":
                continue
            w = d["word"]
            masc = [x["form"] for x in d.get("forms", []) if x.get("tags") == ["masculine"]]
            gl = [g for s in d.get("senses", []) for g in s.get("glosses", [])]
            if not w.endswith("e") or not masc or not gl:
                continue
            m = re.match(
                r"female equivalent of \S+: (?:a )?female ([a-z][a-z ]*?)(?:[;,/(]|$)", gl[0]
            )
            if not m or re.search(r"[\d-]", w) or w.lower()[:-1] in adj or w.lower() in adj:
                continue
            en = m.group(1).strip()
            if not any(s.lexname() == "noun.person" for s in wn.synsets(en.replace(" ", "_"), "n")):
                continue
            z = zipf_frequency(w, "de")
            if z >= 3.0:
                rows.append({"lemma": w, "masculine": masc[0], "gloss_en": en, "zipf": round(z, 2)})
    df = pd.DataFrame(rows).drop_duplicates("lemma").sort_values("zipf", ascending=False)
    df["source"] = "wiktionary de (adjectival declension) rule, p31.build_person_de"
    df.to_csv(PERSON_DE, index=False)
    print(len(df), "person nouns")
    return df


def readout(tok, fam: str) -> pd.DataFrame:
    """German adjectives read by first token: >= 3 letters and unique within the list."""
    from wordfreq import zipf_frequency

    from .phase5_stimuli import ADJ_TRANS

    t = pd.read_csv(ADJ_TRANS, keep_default_na=False)
    t = t[t.de != ""].reset_index(drop=True)
    ids = [tok(" " + w, add_special_tokens=False)["input_ids"] for w in t.de]
    t["first"] = [i[0] for i in ids]
    t["single"] = [len(i) == 1 for i in ids]
    txt = [tok.decode([i]).strip() for i in t["first"]]
    counts = t["first"].map(t["first"].value_counts())
    t = t[(counts == 1).to_numpy() & np.array([len(x) >= 3 for x in txt])].reset_index(drop=True)
    t["zipf"] = [zipf_frequency(w, "de") for w in t.de]
    return t.rename(columns={"en": "word"})


# ---- run -------------------------------------------------------------------------------------
def prompt_keys(person_drop: set[str] | None = None) -> pd.DataFrame:
    from .phase5 import SMOKE

    nw = pd.read_csv(NONCE_DE, keep_default_na=False)
    pn = pd.read_csv(PERSON_DE, keep_default_na=False)
    pn = pn[~pn.lemma.isin(person_drop or set())]
    if SMOKE:
        nw, pn = nw.head(8), pn.head(4)
    rows = []
    for kind, words, genders in (
        ("nonce", nw.word, ("m", "f", "n")),
        ("person", pn.lemma, ("m", "f")),
    ):
        for w in words:
            for wd, (ctx, t) in CONTEXTS.items():
                if kind == "person" and ctx == "object":
                    continue
                for g in genders:
                    rows.append({"kind": kind, "word": w, "stem": w, "context": ctx,
                                 "wording": wd, "gender": g,
                                 "prompt": t.format(art=ART[g], N=w, acc=REL_ACC[g], dat=REL_DAT[g])})  # fmt: skip
    return pd.DataFrame(rows)


# PI review 2026-10-21 (English glosses): ordinal, ambiguous with "big"/colour/party, dated slur,
# sex/sexuality-linked, appearance
PERSON_DROP = {"Erste", "Große", "Grüne", "Farbige", "Homosexuelle", "Heterosexuelle",
               "Prostituierte", "Geliebte", "Rothaarige"}  # fmt: skip


@torch.no_grad()
def run(model_id: str, out_root: str = OUT_ROOT, device=None, dtype=None, model_tok=None):
    from datetime import UTC, datetime

    from .models import load_model, pick_device, pick_dtype, run_metadata
    from .phase5 import Prompt, family
    from .phase5 import run as fwd
    from .scoring import continuation_logprobs_batch

    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = model_tok or load_model(model_id, dev, dt)
    fam = family(model_id)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    keys = prompt_keys(PERSON_DROP)
    ro = readout(tok, fam)
    read = ro["first"].astype(int).tolist()
    V = torch.zeros((1, model.config.hidden_size))
    ps = [Prompt(p, w) for p, w in zip(keys.prompt, keys.word, strict=True)]
    r = fwd(model, tok, ps, 0, [(i, 0, 0.0) for i in range(len(ps))], V, read)
    stems = [p.removesuffix(" sehr") for p in keys.prompt]
    assert all(s.endswith(" ist") for s in stems)
    conts = list(dict.fromkeys(PROBE_PERSON + PROBE_THING + PROBE_NARROW))
    lp = np.asarray(continuation_logprobs_batch(model, tok, [(s, c) for s in stems for c in conts],
                                                desc="P31 probe")).reshape(len(stems), len(conts))  # fmt: skip
    for j, c in enumerate(conts):
        keys[f"lp{c.replace(' ', '_')}"] = lp[:, j]
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    keys.drop(columns=["prompt"]).to_csv(out / "prompts.csv.gz", index=False)
    ro.to_csv(out / "readout.csv", index=False)
    np.savez_compressed(out / "next.npz", lp=r["lp"], read=np.asarray(read))
    (out / "meta.json").write_text(json.dumps({"meta": meta, "n_prompts": len(keys),
                                               "n_adjectives": len(ro)}, indent=1, default=str))  # fmt: skip
    print(model_id, "P31 scored:", len(keys), "prompts,", len(ro), "adjectives")
    return keys


# ---- analysis --------------------------------------------------------------------------------
def _pair(keys: pd.DataFrame, X: np.ndarray, a: str, b: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Per word: gender-a row minus gender-b row, averaged over the wordings present."""
    idx = keys.pivot_table(
        index=["word", "wording"], columns="gender", values="row", aggfunc="first"
    )
    d = X[idx[a].to_numpy()] - X[idx[b].to_numpy()]
    per = idx.reset_index()[["word"]]
    words_ = per.word.unique()
    wi = {w: i for i, w in enumerate(words_)}
    acc = np.zeros((len(words_), X.shape[1]))
    np.add.at(acc, per.word.map(wi).to_numpy(), d)
    acc /= per.groupby("word").size().reindex(words_).to_numpy()[:, None]
    return pd.DataFrame({"word": words_, "stem": words_}), acc


def _lse(keys: pd.DataFrame, conts) -> np.ndarray:
    return np.logaddexp.reduce(
        np.stack([keys[f"lp{c.replace(' ', '_')}"] for c in conts], 1), axis=1
    )


def analyze(model_id: str, out_root: str = OUT_ROOT) -> dict:
    from .phase5_analysis import graded

    d = Path(out_root) / model_slug(model_id)
    keys = pd.read_csv(d / "prompts.csv.gz", keep_default_na=False)
    keys["row"] = np.arange(len(keys))
    ro = pd.read_csv(d / "readout.csv", keep_default_na=False)
    lp = np.load(d / "next.npz")["lp"]
    keys["share"] = np.exp(lp).sum(1)
    keys["probe"] = _lse(keys, PROBE_PERSON) - _lse(keys, PROBE_THING)
    keys["probe_narrow"] = _lse(keys, PROBE_NARROW[:1]) - _lse(keys, PROBE_NARROW[1:])
    non = keys[keys.kind == "nonce"]
    res: dict = {"model": model_id, "n_adjectives": len(ro), "n_single": int(ro.single.astype(str).eq("True").sum()),
                 "wordings": {}}  # fmt: skip
    med = non.groupby("wording").probe.median()
    medn = non.groupby("wording").probe_narrow.median()
    real_n1 = float(keys[(keys.kind == "person") & (keys.wording == "N1")].probe.median())
    for w, (ctx, _) in CONTEXTS.items():
        share = float(non[non.wording == w].share.median())
        # broad probe judged only relative to N1 (its zero point carries the probe words' base
        # rates); the matched narrow pair carries the "reads as a thing" sign (PI, 2026-10-21)
        c1 = {"object": medn[w] < 0 and med[w] <= med["N1"] - C1_MARGIN, "neutral": True,
              "person": med[w] >= med["N1"] + C1_MARGIN}[ctx]  # fmt: skip
        res["wordings"][w] = {"context": ctx, "probe_median": float(med[w]),
                              "probe_narrow_median": float(medn[w]), "share": share,
                              "C1": bool(c1), "C2": share >= SHARE_MIN,
                              "valid": bool(c1 and share >= SHARE_MIN)}  # fmt: skip
    res["probe_real_person_N1"] = real_n1
    ok = {w for w, v in res["wordings"].items() if v["valid"]}
    res["valid"] = bool([w for w in ("O1", "O2") if w in ok] and "N1" in ok and real_n1 > 0)
    rng = np.random.default_rng(SEED)
    single = ro.single.astype(str).eq("True").to_numpy()
    scores = {}
    for ctx in ("object", "neutral", "person"):
        ws = [w for w, (c, _) in CONTEXTS.items() if c == ctx and w in ok]
        if not ws:
            continue
        ks = non[non.wording.isin(ws)]
        for a, tag in (("f", ""), ("n", "_das")):  # die - der (primary), das - der (reported)
            m, dd = _pair(ks, lp, a, "m")
            s = graded(dd, ro)[0]
            lo, hi = _boot(s, m.stem.to_numpy(), rng)
            res[f"S_{ctx}{tag}"] = {"S": float(s.mean()), "lo95": lo, "hi95": hi, "wordings": ws,
                                   "n_words": len(m)}  # fmt: skip
            if tag == "":
                scores[ctx] = s
                ss = graded(dd[:, single], ro[single].reset_index(drop=True))[0]
                res[f"S_{ctx}_single_token"] = float(ss.mean())
        pk = keys[(keys.kind == "person") & keys.wording.isin(ws)]
        if len(pk):
            mp, dp = _pair(pk, lp, "f", "m")
            sp = graded(dp, ro)[0]
            lo, hi = _boot(sp, mp.stem.to_numpy(), rng)
            res[f"S_real_{ctx}"] = {"S": float(sp.mean()), "lo95": lo, "hi95": hi}
    if "object" in scores and "neutral" in scores:
        so, sn = scores["object"], scores["neutral"]
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(so), len(so))
            bs.append(so[i].mean() / sn[i].mean())
        res["ratio_object_neutral"] = {"ratio": float(so.mean() / sn.mean()),
                                       "lo95": float(np.quantile(bs, 0.025)),
                                       "hi95": float(np.quantile(bs, 0.975))}  # fmt: skip
    if res["valid"] and res["S_neutral"]["lo95"] <= 0:
        res["verdict"] = "no neutral pull to explain (descriptive only)"
    elif res["valid"]:
        s, sn = res["S_object"], res["S_neutral"]["S"]
        res["verdict"] = ("pull survives an object reading" if s["lo95"] > 0
                          else "mostly the 'a woman' guess" if s["hi95"] < 0.5 * sn
                          else "partial")  # fmt: skip
    else:
        res["verdict"] = "no valid test"
    (d / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res
