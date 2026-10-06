"""Phase 0: does the model know the genders?

Per noun, three behavioural measures plus a tokenization log.

Primary: sentence scoring in two frames. The same sentence with the masculine
vs the feminine article; compare log P(noun + rest of sentence | lead-in +
article) (margin, in nats). The article's own probability is left out: "ein"
and "dem" also serve neuter nouns, so they're a priori likelier than "eine"
and "der", which biased margins toward masculine (on both Qwen3 and EuroLLM).
Frames are chosen so the wrong article is never a valid reading:
  frame 1, dative singular
    de: "Das hat etwas mit dem/der X zu tun."  the plural would be "den X-n",
        so "der" can't be read as plural
    es: "Esto tiene que ver con el/la X."  "con el" doesn't contract
        (unlike "a el" -> "al", "de el" -> "del")
  frame 2, indefinite article after a lead-in
    de: "Hier ist ein/eine X."  indefinite articles have no plural, so unlike
        "die X" a masculine noun can't be read as plural (die Schlüssel);
        neuter "ein" is irrelevant since neuter nouns are excluded
    es: "Aquí hay un/una X."
  Frame 2 was fixed on these grammatical grounds and then checked on two
  model families (Qwen3, EuroLLM), not tuned on scores. Earlier versions:
  sentence-initial "Der/Die X ist hier." (article with no preceding context:
  ~75% on Qwen3-0.6B, which adds no BOS token), then "Ich weiß, dass der/die
  X hier ist." (plural reading of "die X" survives until "ist", weakening
  margins for nouns like Stiefel, Besen). Also avoid frames where the other
  article has another reading ("Sé que la X ..." : la = "her"), where later
  words agree in gender ("... es nuevo"), or where the article contracts.

Each frame is "right" / "wrong" when its margin points the right / wrong way
by at least `min_margin` nats, else "unsure". A noun's status:
  known     both frames right         -> usable downstream (`passed`)
  wrong     a frame is confidently wrong and the other isn't right
  conflict  one frame right, the other confidently wrong
  unsure    otherwise (near-ties): dropped, not counted as wrong

Diagnostic only: the metalinguistic quiz (few-shot "noun: article" list, next
token P(der) vs P(die), averaged over two shot orderings). In small models it
is format-sensitive (answers flip with shot order), so it doesn't gate
anything; it's kept to see whether a model can *report* gender as well as use it.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .models import load_model, model_slug, pick_device, pick_dtype, run_metadata
from .scoring import candidate_logprobs, continuation_logprob, tokens_of
from .stimuli import flipped_pairs, load_stimuli

# A frame counts only if the two sentences differ by >= this many nats
# (e^1 ~ 2.7x more likely). Well above fp16 rounding noise (~0.05).
DEFAULT_MIN_MARGIN = 1.0

LANG_CONFIG = {
    "de": {
        "header": "Bestimmter Artikel:\n",
        # Few-shot examples: balanced m/f; must not appear in the test lists.
        "shots": [("Teppich", "der"), ("Tasche", "die"), ("Garten", "der"), ("Kerze", "die")],
        "meta": {"m": " der", "f": " die"},
        "meta_other": [" das"],
        "frames": [
            ("Das hat etwas mit {art} {noun} zu tun.", {"m": "dem", "f": "der"}),
            ("Hier ist {art} {noun}.", {"m": "ein", "f": "eine"}),
        ],
    },
    "es": {
        "header": "Artículo definido:\n",
        "shots": [("cuaderno", "el"), ("ventana", "la"), ("zapato", "el"), ("camisa", "la")],
        "meta": {"m": " el", "f": " la"},
        "meta_other": [],
        "frames": [
            ("Esto tiene que ver con {art} {noun}.", {"m": "el", "f": "la"}),
            ("Aquí hay {art} {noun}.", {"m": "un", "f": "una"}),
        ],
    },
}


def shot_orderings(shots):
    """Two orderings ending on different genders."""
    return [list(shots), list(reversed(shots))]


def meta_prompt(cfg, shots, noun: str) -> str:
    body = "".join(f"{n}: {a}\n" for n, a in shots)
    return f"{cfg['header']}{body}{noun}:"


def _logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def score_noun(model, tok, lang: str, noun: str) -> dict:
    cfg = LANG_CONFIG[lang]
    cands = [cfg["meta"]["m"], cfg["meta"]["f"], *cfg["meta_other"]]
    per_order = [
        candidate_logprobs(model, tok, meta_prompt(cfg, s, noun), cands)
        for s in shot_orderings(cfg["shots"])
    ]
    lp_m = sum(o[0] for o in per_order) / len(per_order)
    lp_f = sum(o[1] for o in per_order) / len(per_order)
    lp_other = (
        sum(_logsumexp(o[2:]) for o in per_order) / len(per_order)
        if cfg["meta_other"]
        else float("nan")
    )
    mass = sum(math.exp(_logsumexp(o)) for o in per_order) / len(per_order)

    frames = {}
    for i, (frame, arts) in enumerate(cfg["frames"], start=1):
        # Score the noun and what follows GIVEN the article, not the article
        # itself: German "ein"/"dem" also serve neuter nouns, so their prior is
        # higher than "eine"/"der" and would bias every noun toward masculine.
        prefix, rest = frame.split("{noun}")
        lp = {
            g: continuation_logprob(model, tok, prefix.format(art=a).rstrip(), " " + noun + rest)
            for g, a in arts.items()
        }
        frames[f"ctx{i}_margin"] = lp["m"] - lp["f"]
    toks = tokens_of(tok, " " + noun)
    return {
        **frames,
        "meta_lp_m": lp_m,
        "meta_lp_f": lp_f,
        "meta_lp_other": lp_other,
        "meta_mass": mass,
        "meta_margin": lp_m - lp_f,
        "meta_order_agree": (per_order[0][0] > per_order[0][1])
        == (per_order[1][0] > per_order[1][1]),
        "n_tokens": len(toks),
        "tokens": "|".join(toks),
    }


def check_shots_disjoint(df: pd.DataFrame) -> None:
    for lang, cfg in LANG_CONFIG.items():
        shot_words = {n.lower() for n, _ in cfg["shots"]}
        clash = df[(df.lang == lang) & df.lemma.str.lower().isin(shot_words)]
        if len(clash):
            raise ValueError(f"Few-shot words appear in stimuli: {clash.lemma.tolist()}")


def _balanced_acc(sub: pd.DataFrame, col: str) -> float:
    accs = [sub[sub.gender == g][col].mean() for g in ("m", "f") if (sub.gender == g).any()]
    return float(sum(accs) / len(accs)) if accs else float("nan")


def frame_verdict(margin: float, gender: str, min_margin: float) -> str:
    if abs(margin) < min_margin:
        return "unsure"
    return "right" if (margin > 0) == (gender == "m") else "wrong"


def classify(items: pd.DataFrame, min_margin: float) -> pd.DataFrame:
    items = items.copy()
    frame_cols = sorted(c for c in items if c.startswith("ctx") and c.endswith("_margin"))
    verdicts = []
    for c in frame_cols:
        v = c.replace("_margin", "_verdict")
        items[v] = [
            frame_verdict(m, g, min_margin) for m, g in zip(items[c], items.gender, strict=True)
        ]
        verdicts.append(v)

    def status(row) -> str:
        vs = [row[v] for v in verdicts]
        if all(x == "right" for x in vs):
            return "known"
        if "right" in vs and "wrong" in vs:
            return "conflict"
        if "wrong" in vs:
            return "wrong"
        return "unsure"

    items["status"] = items.apply(status, axis=1)
    items["passed"] = items.status == "known"
    items["ctx_margin"] = items[frame_cols].mean(axis=1)
    items["ctx_pred"] = (items.ctx_margin > 0).map({True: "m", False: "f"})
    items["ctx_correct"] = items.ctx_pred == items.gender  # sign only, no threshold
    items["frames_agree"] = (items[frame_cols].gt(0)).nunique(axis=1).eq(1)
    items["meta_pred"] = (items.meta_margin > 0).map({True: "m", False: "f"})
    items["meta_correct"] = items.meta_pred == items.gender
    return items


def _rates(sub: pd.DataFrame) -> dict:
    return {
        "n": len(sub),
        "known_bal": _balanced_acc(sub, "passed"),
        "ctx_bal_acc": _balanced_acc(sub, "ctx_correct"),
        "meta_bal_acc": _balanced_acc(sub, "meta_correct"),
    }


def summarize(items: pd.DataFrame, min_margin: float) -> dict:
    out = {"min_margin": min_margin}
    for lang, sub in items.groupby("lang"):
        st = sub.status.value_counts(normalize=True)
        out[lang] = {
            **_rates(sub),
            "n_m": int((sub.gender == "m").sum()),
            "n_f": int((sub.gender == "f").sum()),
            "known_m": float(sub[sub.gender == "m"].passed.mean()),
            "known_f": float(sub[sub.gender == "f"].passed.mean()),
            **{
                f"rate_{k}": float(st.get(k, 0.0)) for k in ("known", "wrong", "conflict", "unsure")
            },
            "frames_agree": float(sub.frames_agree.mean()),
            "quiz_agrees_with_ctx": float((sub.meta_pred == sub.ctx_pred).mean()),
            "meta_order_agree": float(sub.meta_order_agree.mean()),
            "mean_meta_mass": float(sub.meta_mass.mean()),
            "frac_multitoken": float((sub.n_tokens > 1).mean()),
        }
    # Flipped pairs usable downstream = both halves known.
    passed = items[items.passed]
    pairs = flipped_pairs(items)
    ok = sum(
        ((passed.lang == "de") & (passed.lemma == r.de_lemma)).any()
        and ((passed.lang == "es") & (passed.lemma == r.es_lemma)).any()
        for r in pairs.itertuples()
    )
    out["flipped_pairs"] = {"n": len(pairs), "both_pass": int(ok)}
    # Breakdowns: frequency bins separate models better than the overall score.
    for col in ("freq_bin", "set", "de_suffix", "es_exception"):
        if col in items and items[col].astype(str).str.len().gt(0).any():
            out[f"by_{col}"] = _breakdown(items[items[col].astype(str).str.len() > 0], col)
    return out


def _breakdown(items: pd.DataFrame, col: str) -> dict:
    res = {}
    for (lang, key), sub in items.groupby(["lang", items[col].astype(str)]):
        res.setdefault(lang, {})[key] = _rates(sub)
    return res


def run(
    model_id: str,
    stimuli_path: str,
    out_root: str = "results/phase0",
    device: str | None = None,
    dtype: str | None = None,
    langs: list[str] | None = None,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> Path:
    df = load_stimuli(stimuli_path)
    check_shots_disjoint(df)
    if langs:
        df = df[df.lang.isin(langs)].reset_index(drop=True)

    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    print(f"Loading {model_id} on {dev} ({dt}) ...")
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["stimuli"] = str(stimuli_path)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")

    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc=model_id):
        rows.append({**r._asdict(), **score_noun(model, tok, r.lang, r.lemma)})
    items = classify(pd.DataFrame(rows).drop(columns="Index"), min_margin)

    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "items.csv", index=False)
    summary = {"meta": meta, "results": summarize(items, min_margin)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["results"], indent=2))
    print(f"Wrote {out}/")
    return out


def compare(out_root: str = "results/phase0", by: str | None = None) -> pd.DataFrame:
    if by:
        return _compare_by(out_root, by)
    rows = []
    for f in sorted(Path(out_root).glob("*/summary.json")):
        s = json.loads(f.read_text())
        for lang, r in s["results"].items():
            if lang not in LANG_CONFIG:
                continue
            rows.append(
                {
                    "model": s["meta"]["model_id"],
                    "params_B": round(s["meta"]["n_params"] / 1e9, 2),
                    "device": s["meta"]["device"],
                    "dtype": s["meta"]["dtype"],
                    "lang": lang,
                    "n": r["n"],
                    "known_bal": _r(r.get("known_bal")),
                    "known_m": _r(r.get("known_m")),
                    "known_f": _r(r.get("known_f")),
                    "wrong": _r(r.get("rate_wrong")),
                    "unsure": _r(r.get("rate_unsure")),
                    "conflict": _r(r.get("rate_conflict")),
                    "ctx_bal_acc": _r(r.get("ctx_bal_acc")),
                    "quiz_bal_acc": _r(r.get("meta_bal_acc")),
                    "flipped_ok": f"{s['results']['flipped_pairs']['both_pass']}"
                    f"/{s['results']['flipped_pairs']['n']}",
                }
            )
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(["lang", "known_bal"], ascending=[True, False])
        table.to_csv(Path(out_root) / "comparison.csv", index=False)
    return table


def _compare_by(out_root: str, by: str) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(out_root).glob("*/summary.json")):
        s = json.loads(f.read_text())
        for lang, groups in s["results"].get(f"by_{by}", {}).items():
            for key, r in groups.items():
                rows.append({"model": s["meta"]["model_id"], "lang": lang, by: key,
                             "n": r["n"], "known_bal": _r(r.get("known_bal"))})  # fmt: skip
    if not rows:
        return pd.DataFrame()
    t = (
        pd.DataFrame(rows)
        .pivot_table(index=["lang", "model"], columns=by, values="known_bal")
        .reset_index()
    )
    t.to_csv(Path(out_root) / f"comparison_by_{by}.csv", index=False)
    return t


def _r(x):
    return None if x is None else round(x, 3)
