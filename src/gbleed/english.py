"""P23 (English base case) and P24 (cross-domain transfer, §5.1 cosine); exploratory, adopted
2026-10-09 (PI). Design: docs/design/phase5-design.md P23/P24.

Words (bare, same input format as Phases 2-3; LAST and AFTER from one pass):
  social pairs   the fixed female/male English pairs that define the social direction
  person nouns   held-out sex-specific English person nouns: first WordNet noun sense under
                 person.n.01 whose gloss has female words or male words (not both), Zipf >= 3,
                 one word, not a social-pair word
  concepts       the English concepts of the Phase 5 steering nouns and flipped pairs
  calibration    Glasgow-rated nouns: WordNet senses >= 50% noun, Zipf >= 3, one word, not under
                 person/animal/people or group roots in the first two senses
Directions per layer (unit, female/feminine positive): SOCIAL = mean over pairs of (female -
male); ES/DE = the Phase 5 steering vectors (stratified difference of means); POOLED as Phase 4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import activations as acts
from .estimators import auc, shuffle_within
from .models import git_state, model_slug, progress, stage

# (female, male); sex labels checked against WordNet in build() (pronouns: grammatical person)
SOCIAL_PAIRS = [("woman", "man"), ("girl", "boy"), ("mother", "father"), ("daughter", "son"),
                ("sister", "brother"), ("queen", "king"), ("wife", "husband"), ("aunt", "uncle"),
                ("she", "he")]  # fmt: skip
WORDS = "data/stimuli/english_v1.csv"
SEED = 0
N_BOOT = 1000
N_SPLITS = 200


FEMALE_WORDS = {"woman", "women", "female", "girl", "wife", "mother", "sister", "daughter",
                "lady"}  # fmt: skip
MALE_WORDS = {"man", "men", "male", "boy", "husband", "father", "brother", "son", "gentleman"}  # fmt: skip


def morph_marked(w: str) -> bool:
    """Person words whose form carries sex (P24 robustness, PI 2026-10-10): a compound containing
    a sex word (salesman, schoolgirl, gentleman), an English feminine suffix (-ess, -ette, -ine,
    -ina), or a final -a/-o (Spanish-like gender endings: ballerina, soprano, hero)."""
    sex = (FEMALE_WORDS | MALE_WORDS) - {w}
    return any(x in w for x in sex) or w.endswith(("ess", "ette", "ine", "ina", "a", "o"))


REGISTER = {"offensive", "slang", "informal", "disparaging", "derogatory", "vulgar", "term"}


def _gloss_words(wn, w: str):
    import re

    ss = wn.synsets(w, pos="n")[:1]
    return (ss[0], re.findall(r"[a-z]+", ss[0].definition().lower())) if ss else (None, [])


def _social_ok(wn, w: str, sx: str) -> bool:
    """Social-pair check: the first-sense gloss contains a word of the expected sex (definitional
    glosses may also mention the other sex: "an adult female person (as opposed to a man)")."""
    _, words = _gloss_words(wn, w)
    return bool(set(words) & (FEMALE_WORDS if sx == "f" else MALE_WORDS))


def _sex(wn, w: str) -> str:
    """Held-out person nouns: first noun sense under person.n.01 and at least half of the word's
    tagged WordNet usage; a female or male word among the gloss's first four words and no word
    of the other sex anywhere; no register label (offensive, slang, informal, "term for")."""
    syn, words = _gloss_words(wn, w)
    if syn is None or set(words) & REGISTER:
        return ""
    if wn.synset("person.n.01") not in {h for path in syn.hypernym_paths() for h in path}:
        return ""
    # the person sense must be the word's dominant one in WordNet's tagged usage (drops rip,
    # antique, private, whose first noun sense is a rare person sense)
    counts = {x.synset(): x.count() for ss in wn.synsets(w) for x in ss.lemmas() if x.name() == w}
    if not counts.get(syn) or counts[syn] < 0.5 * sum(counts.values()):
        return ""
    head, allw = set(words[:4]), set(words)
    if head & FEMALE_WORDS and not allw & MALE_WORDS:
        return "f"
    if head & MALE_WORDS and not allw & FEMALE_WORDS:
        return "m"
    return ""


def build() -> pd.DataFrame:
    from wordfreq import zipf_frequency

    from .lexicon import _wordnet
    from .phase3_stimuli import ANIMATE_ROOTS, GROUP_ROOTS, _under
    from .phase5_stimuli import NOUNS, glasgow

    wn = _wordnet()
    rows = []
    for f, m in SOCIAL_PAIRS:
        for w, sx in ((f, "f"), (m, "m")):
            if w not in ("she", "he") and not _social_ok(wn, w, sx):
                print(f"social pair ({f}, {m}) dropped: WordNet gloss of {w!r} lacks a {sx} word")
                break
        else:
            rows += [{"word": f, "set": "social", "sex": "f", "pair": f"{f}/{m}"},
                     {"word": m, "set": "social", "sex": "m", "pair": f"{f}/{m}"}]  # fmt: skip
    social = {r["word"] for r in rows}
    seen = set()
    for syn in wn.all_synsets("n"):
        for lem in syn.lemmas():
            w = lem.name()
            if w in seen or not w.isalpha() or not w.islower() or w in social:
                continue
            seen.add(w)
            sx = _sex(wn, w)
            if sx and zipf_frequency(w, "en") >= 3:
                rows.append({"word": w, "set": "person", "sex": sx, "pair": ""})
    nouns = pd.read_csv(NOUNS, keep_default_na=False)
    for c in sorted(set(nouns.concept_en)):
        rows.append({"word": c, "set": "concept", "sex": "", "pair": ""})
    animate = {wn.synset(r) for r in ANIMATE_ROOTS}
    group = {wn.synset(r) for r in GROUP_ROOTS}
    g = glasgow()
    for r in g.itertuples():
        w = r.word
        ss = wn.synsets(w)
        if (not w.isalpha() or not ss or sum(s.pos() == "n" for s in ss) / len(ss) < 0.5
                or zipf_frequency(w, "en") < 3 or _under(w, wn, animate, 2)
                or _under(w, wn, group, 2)):  # fmt: skip
            continue
        rows.append({"word": w, "set": "calibration", "sex": "", "pair": "", "GEND": r.GEND,
                     "VAL": r.VAL, "AROU": r.AROU, "SIZE": r.SIZE})  # fmt: skip
    df = pd.DataFrame(rows).drop_duplicates(["word", "set"])
    df.to_csv(WORDS, index=False)
    print(df.groupby(["set", "sex"]).size().to_string())
    return df


def extract(model_id: str, device=None, dtype=None) -> None:
    from datetime import UTC, datetime

    from .models import load_model, pick_device, pick_dtype, run_metadata

    df = pd.read_csv(WORDS, keep_default_na=False)
    words = list(dict.fromkeys(df.word))
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    X_last, X_after, toks = acts.states_at(model, tok, words, acts.AFTER, desc="english")
    for pos, X in (("last", X_last), ("after", X_after)):
        acts.save(model_id, "english" + ("_after" if pos == "after" else ""), X, words, toks,
                  {**meta, "position": pos})  # fmt: skip


def _unit(v):
    return v / np.linalg.norm(v)


def _rank(x):
    return pd.Series(x).rank().to_numpy()


def _partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y, controlling for z (ranks residualised on z's ranks)."""
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    res = [r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0] for r in (rx, ry)]
    return float(np.corrcoef(*res)[0, 1])


N_SHUFFLE = 100
N_NULL_SPLITS = 20


def _ratio_cos(terms) -> float:
    num, a, b = np.mean(terms, axis=0)
    return float(num / np.sqrt(a * b)) if a > 0 and b > 0 else float("nan")


def _split_cos(Xl, fi, mi, Xg, y, cells, covs, rng, n_splits) -> float:
    """Split-half cosine (ratio of means) between the social direction (pairs halved) and a
    grammatical class vector (halves within cell x label)."""
    from .estimators import _halves, class_betas, nuisance

    terms = []
    for _ in range(n_splits):
        h = rng.permutation(len(fi)) < len(fi) // 2
        sA, sB = ((Xl[fi[m]] - Xl[mi[m]]).mean(0) for m in (h, ~h))
        A = _halves(cells, y, rng)
        gA, gB = (class_betas(Xg[m], y[m][:, None].astype(float),
                              nuisance(cells[m], [c[m] for c in covs]))[0] for m in (A, ~A))  # fmt: skip
        terms.append(((sA @ gB + sB @ gA) / 2, sA @ sB, gA @ gB))
    return _ratio_cos(terms)


def _split_cos_pairs(Xl, fi, mi, Xp, Xs, rng, n_splits) -> float:
    """Split-half cosine between the social direction and a paired-difference direction
    (plural - singular), both halved by pairs."""
    terms = []
    for _ in range(n_splits):
        h = rng.permutation(len(fi)) < len(fi) // 2
        sA, sB = ((Xl[fi[m]] - Xl[mi[m]]).mean(0) for m in (h, ~h))
        g = rng.permutation(len(Xp)) < len(Xp) // 2
        nA, nB = ((Xp[m] - Xs[m]).mean(0) for m in (g, ~g))
        terms.append(((sA @ nB + sB @ nA) / 2, sA @ sB, nA @ nB))
    return _ratio_cos(terms)


def _number_acts(model_id: str) -> dict:
    """{lang: (plural rows, singular rows)} of the phase5_number pairs, if extracted."""
    from .phase5_stimuli import NUMBER

    try:
        Xp, mp = acts.load(model_id, "phase5_number")
    except FileNotFoundError:
        return {}
    df = pd.read_csv(NUMBER, keep_default_na=False)
    rp = {w: i for i, w in enumerate(mp["words"])}
    out = {}
    for lang, name in (("es", "phase2_bare"), ("de", "phase3_bare")):
        Xs, ms = acts.load(model_id, name)
        rs = {w: i for i, w in enumerate(ms["words"])}
        d = df[df.lang == lang]
        out[lang] = (Xp[[rp[w] for w in d.plural]], Xs[[rs[w] for w in d.lemma]])
    return out


def analyze(model_id: str, position: str = "last", out_root: str | None = None,
            n_boot: int = N_BOOT) -> dict:  # fmt: skip
    from .estimators import _halves, class_betas, nuisance
    from .phase4 import PAIRS_FINAL, _fit, de_train, es_train, pooled, within_cell_auc

    sx = "" if position == "last" else "_after"
    df = pd.read_csv(WORDS, keep_default_na=False)
    X, meta = acts.load(model_id, "english" + sx)
    row = {w: i for i, w in enumerate(meta["words"])}
    es, de = es_train(model_id, position), de_train(model_id, position)
    num_acts = _number_acts(model_id) if position == "last" else {}
    po = pooled(es, de)
    soc = df[df.set == "social"]
    fi = np.array([row[w] for w in soc[soc.sex == "f"].word])
    mi = np.array([row[w] for w in soc[soc.sex == "m"].word])
    per = df[df.set == "person"]
    pi_ = np.array([row[w] for w in per.word])
    py = (per.sex == "f").to_numpy(int)
    cal = df[df.set == "calibration"].copy()
    ci_ = np.array([row[w] for w in cal.word])
    gend, val = pd.to_numeric(cal.GEND).to_numpy(), pd.to_numeric(cal.VAL).to_numpy()
    pairs = pd.read_csv(PAIRS_FINAL, keep_default_na=False)
    pc = np.array([row[c] for c in pairs.concept_en])
    es_f = (pairs.es_gender == "f").to_numpy(int)
    de_f = (pairs.de_gender == "f").to_numpy(int)
    rng = np.random.default_rng(SEED)
    L = X.shape[1]
    rows = []
    stage(f"{model_id} [{position}]: P23/P24 over {L - 2} inner layers; {len(fi)} social pairs, "
          f"{len(pi_)} person nouns, {len(ci_)} calibration nouns, {len(pc)} pair concepts")  # fmt: skip
    for layer in progress(range(1, L - 1), desc=f"english {position}", unit="layer"):
        Xl = X[:, layer].astype(np.float64)
        Xe, Xd = es["X"][:, layer].astype(np.float64), de["X"][:, layer].astype(np.float64)
        s_vec = _unit((Xl[fi] - Xl[mi]).mean(0))
        dirs = {"social": s_vec, "es": _unit(_fit(Xe, es, kind="dom")),
                "de": _unit(_fit(Xd, de, kind="dom")),
                "pooled": _unit(_fit(np.r_[Xe, Xd], po, kind="dom"))}  # fmt: skip
        r = {"layer": layer}
        # P23 (1): social direction on Glasgow-rated inanimate nouns (feminine = low GEND)
        proj = Xl[ci_] @ s_vec
        r["p23_social_glasgow_rho"] = float(pd.Series(proj).corr(pd.Series(-gend), "spearman"))
        r["p23_social_glasgow_rho_partial_val"] = _partial_spearman(proj, -gend, val)
        # P23 (2): English pair concepts on ES/DE/POOLED vs the translations' genders
        for nm in ("es", "de", "pooled"):
            pj = Xl[pc] @ dirs[nm]
            r[f"p23_{nm}_vs_es_gender"] = auc(es_f, pj)
            r[f"p23_{nm}_vs_de_gender"] = auc(de_f, pj)
        # P24: social -> grammatical (within cells), grammatical -> social (person nouns)
        r["p24_social_on_es"] = within_cell_auc(es["y"], Xe @ s_vec, es["cells"])
        r["p24_social_on_de"] = within_cell_auc(de["y"], Xd @ s_vec, de["cells"])
        for nm in ("es", "de", "pooled"):
            r[f"p24_{nm}_on_persons"] = auc(py, Xl[pi_] @ dirs[nm])
        r["social_on_persons"] = auc(py, Xl[pi_] @ s_vec)  # sanity: should be high
        um = ~per.word.map(morph_marked).to_numpy()  # robustness: unmarked person words only
        unmarked = {f"p24_{nm}_on_persons_unmarked": auc(py[um], Xl[pi_[um]] @ dirs[nm])
                    for nm in ("es", "de", "pooled")}  # fmt: skip
        unmarked["social_on_persons_unmarked"] = auc(py[um], Xl[pi_[um]] @ s_vec)
        for nm in ("es", "de", "pooled"):
            r[f"cos_plain_social_{nm}"] = float(s_vec @ dirs[nm])
        # bootstrap CIs (items resampled; directions refitted where they depend on the items)
        bt = {k: [] for k in r if k != "layer" and not k.startswith("cos")}
        npair = len(fi)
        for _ in range(n_boot):
            ip = rng.choice(npair, npair)
            sv = _unit((Xl[fi[ip]] - Xl[mi[ip]]).mean(0))
            ic = rng.choice(len(ci_), len(ci_))
            pr = Xl[ci_[ic]] @ sv
            bt["p23_social_glasgow_rho"].append(
                pd.Series(pr).corr(pd.Series(-gend[ic]), "spearman")
            )
            bt["p23_social_glasgow_rho_partial_val"].append(
                _partial_spearman(pr, -gend[ic], val[ic])
            )
            jp = rng.choice(len(pc), len(pc))
            for nm in ("es", "de", "pooled"):
                pj = Xl[pc[jp]] @ dirs[nm]
                bt[f"p23_{nm}_vs_es_gender"].append(auc(es_f[jp], pj))
                bt[f"p23_{nm}_vs_de_gender"].append(auc(de_f[jp], pj))
            ie = np.concatenate([rng.choice(np.where(es["cells"] == c)[0], (es["cells"] == c).sum())
                                 for c in np.unique(es["cells"])])  # fmt: skip
            idd = np.concatenate([rng.choice(np.where(de["cells"] == c)[0], (de["cells"] == c).sum())
                                  for c in np.unique(de["cells"])])  # fmt: skip
            bt["p24_social_on_es"].append(
                within_cell_auc(es["y"][ie], Xe[ie] @ sv, es["cells"][ie])
            )
            bt["p24_social_on_de"].append(
                within_cell_auc(de["y"][idd], Xd[idd] @ sv, de["cells"][idd])
            )
            jq = rng.choice(len(pi_), len(pi_))
            for nm in ("es", "de", "pooled"):
                bt[f"p24_{nm}_on_persons"].append(auc(py[jq], Xl[pi_[jq]] @ dirs[nm]))
            bt["social_on_persons"].append(auc(py[jq], Xl[pi_[jq]] @ sv))
        rng_u = np.random.default_rng(SEED + 1)  # separate stream: other CIs stay as before
        iu = np.where(um)[0]
        bu = {k: [] for k in unmarked}
        for _ in range(n_boot):
            jj = iu[rng_u.choice(len(iu), len(iu))]
            for nm in ("es", "de", "pooled"):
                bu[f"p24_{nm}_on_persons_unmarked"].append(auc(py[jj], Xl[pi_[jj]] @ dirs[nm]))
            bu["social_on_persons_unmarked"].append(auc(py[jj], Xl[pi_[jj]] @ s_vec))
        r |= unmarked
        bt |= bu
        for k, v in bt.items():
            v = np.asarray(v, float)
            v = v[np.isfinite(v)]
            r[f"{k}_lo"], r[f"{k}_hi"] = (
                (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)
            )
        # §5.1 split-half cosine: social halves (pairs) x grammatical halves (within cell x gender)
        cs = {nm: [] for nm in ("es", "de")}  # per split: (cross term, social len2, gram len2)
        for _ in range(N_SPLITS):
            h = rng.permutation(npair) < npair // 2
            sA, sB = ((Xl[fi[m]] - Xl[mi[m]]).mean(0) for m in (h, ~h))
            for nm, d, Xg in (("es", es, Xe), ("de", de, Xd)):
                A = _halves(d["cells"], d["y"], rng)
                gA, gB = (class_betas(Xg[m], d["y"][m][:, None].astype(float),
                                      nuisance(d["cells"][m], [c[m] for c in d["covs"]]))[0]
                          for m in (A, ~A))  # fmt: skip
                cs[nm].append(((sA @ gB + sB @ gA) / 2, sA @ sB, gA @ gB))
        for nm in ("es", "de"):
            # ratio of means over splits, as estimators.split_half_geometry (a per-split ratio
            # explodes when a half's noise-corrected length is near zero); NaN if a mean
            # length is not positive (direction not reproducible across halves)
            num, l_s, l_g = np.mean(cs[nm], axis=0)
            r[f"cos_split_social_{nm}"] = (float(num / np.sqrt(l_s * l_g)) if l_s > 0 and l_g > 0
                                           else float("nan"))  # fmt: skip
            r[f"len2_split_social_{nm}"], r[f"len2_split_gram_{nm}"] = float(l_s), float(l_g)
        r["cos_random_sd"] = 1 / np.sqrt(Xl.shape[1])
        # null (as Phase 4 T2): grammatical labels shuffled within cells; separate stream so the
        # estimates above are unchanged
        rng_n = np.random.default_rng(SEED + 2)
        for nm, d, Xg in (("es", es, Xe), ("de", de, Xd)):
            null = [_split_cos(Xl, fi, mi, Xg, shuffle_within(d["y"], d["cells"], rng_n),
                               d["cells"], d["covs"], rng_n, N_NULL_SPLITS)
                    for _ in range(N_SHUFFLE)]  # fmt: skip
            r[f"cos_split_social_{nm}_null_hi"] = float(np.nanpercentile(null, 97.5))
        # reference: a real but unrelated direction built the same way (plural - singular on
        # the same training nouns): how much do real difference-of-means directions overlap?
        for nm, (Xp, Xs) in num_acts.items():
            r[f"cos_plain_social_number_{nm}"] = float(
                s_vec @ _unit((Xp[:, layer] - Xs[:, layer]).mean(0))
            )
            r[f"cos_split_social_number_{nm}"] = _split_cos_pairs(
                Xl,
                fi,
                mi,
                Xp[:, layer].astype(np.float64),
                Xs[:, layer].astype(np.float64),
                np.random.default_rng(SEED + 3),
                N_SPLITS,
            )
        rows.append(r)
    res = pd.DataFrame(rows)
    out = Path(out_root or f"results/english{sx}") / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "layers.csv", index=False)
    keys = [c for c in res.columns if c != "layer" and not c.endswith(("_lo", "_hi"))]
    summ = {"model": model_id, "position": position, "analysis_git": git_state(),
            "mean": res[keys].mean().round(3).to_dict(),
            "layers_lo_above_0.5": {k: int((res[f"{k}_lo"] > 0.5).sum()) for k in keys
                                    if f"{k}_lo" in res and "rho" not in k},
            "layers_cos_above_shuffle_null": {
                nm: int((res[f"cos_split_social_{nm}"] > res[f"cos_split_social_{nm}_null_hi"]).sum())
                for nm in ("es", "de")},
            "layers_rho_lo_above_0": {k: int((res[f"{k}_lo"] > 0).sum()) for k in keys
                                      if "rho" in k}}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(
        json.dumps({k: v for k, v in summ.items() if k != "analysis_git"}, indent=1, default=float)
    )
    return summ
