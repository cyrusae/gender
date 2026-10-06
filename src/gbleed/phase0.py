"""Phase 0: does the model know the genders?

Two independent behavioural measures per noun, plus a tokenization log.

1. Metalinguistic (next-token): a few-shot "noun: article" list, so the next
   token after "{noun}:" is the article. Compare log P(der) vs log P(die)
   (or el vs la). Run with two shot orderings (one ending on a masculine
   example, one on a feminine example) and average, to cancel recency bias.
   P(das) and total probability mass on the candidates are logged as
   diagnostics: low mass means the model isn't following the format.

2. Contextual (sentence scoring): the same sentence with the masculine vs
   feminine article, compare total log-probability. Frames are chosen so the
   wrong article is never a valid reading:
     de: "Das hat etwas mit dem/der X zu tun."  dative singular; the plural
         would be "den X-n", so "der" can't be read as plural.
     es: "Esto tiene que ver con el/la X."  "con el" doesn't contract
         (unlike "a el" -> "al", "de el" -> "del").

A noun "passes" when both measures are correct.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .models import load_model, model_slug, pick_device, pick_dtype, run_metadata
from .scoring import candidate_logprobs, sentence_logprob, tokens_of
from .stimuli import flipped_pairs, load_stimuli

LANG_CONFIG = {
    "de": {
        "header": "Bestimmter Artikel:\n",
        # Few-shot examples: balanced m/f; must not appear in the test lists.
        "shots": [("Teppich", "der"), ("Tasche", "die"), ("Garten", "der"), ("Kerze", "die")],
        "meta": {"m": " der", "f": " die"},
        "meta_other": [" das"],
        "ctx_frame": "Das hat etwas mit {art} {noun} zu tun.",
        "ctx_art": {"m": "dem", "f": "der"},
    },
    "es": {
        "header": "Artículo definido:\n",
        "shots": [("cuaderno", "el"), ("ventana", "la"), ("zapato", "el"), ("camisa", "la")],
        "meta": {"m": " el", "f": " la"},
        "meta_other": [],
        "ctx_frame": "Esto tiene que ver con {art} {noun}.",
        "ctx_art": {"m": "el", "f": "la"},
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

    ctx = {
        g: sentence_logprob(model, tok, cfg["ctx_frame"].format(art=art, noun=noun))
        for g, art in cfg["ctx_art"].items()
    }
    toks = tokens_of(tok, " " + noun)
    return {
        "meta_lp_m": lp_m,
        "meta_lp_f": lp_f,
        "meta_lp_other": lp_other,
        "meta_mass": mass,
        "meta_margin": lp_m - lp_f,
        "meta_order_agree": (per_order[0][0] > per_order[0][1])
        == (per_order[1][0] > per_order[1][1]),
        "ctx_lp_m": ctx["m"],
        "ctx_lp_f": ctx["f"],
        "ctx_margin": ctx["m"] - ctx["f"],
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


def summarize(items: pd.DataFrame) -> dict:
    out = {}
    for lang, sub in items.groupby("lang"):
        out[lang] = {
            "n": len(sub),
            "n_m": int((sub.gender == "m").sum()),
            "n_f": int((sub.gender == "f").sum()),
            "meta_acc": float(sub.meta_correct.mean()),
            "meta_bal_acc": _balanced_acc(sub, "meta_correct"),
            "ctx_acc": float(sub.ctx_correct.mean()),
            "ctx_bal_acc": _balanced_acc(sub, "ctx_correct"),
            "both_acc": float(sub.both_correct.mean()),
            "both_bal_acc": _balanced_acc(sub, "both_correct"),
            "acc_m": float(sub[sub.gender == "m"].both_correct.mean()),
            "acc_f": float(sub[sub.gender == "f"].both_correct.mean()),
            "measures_agree": float((sub.meta_pred == sub.ctx_pred).mean()),
            "meta_order_agree": float(sub.meta_order_agree.mean()),
            "mean_meta_mass": float(sub.meta_mass.mean()),
            "frac_multitoken": float((sub.n_tokens > 1).mean()),
        }
    # Flipped pairs usable downstream = both halves pass.
    passed = items[items.both_correct]
    pairs = flipped_pairs(items)
    ok = sum(
        ((passed.lang == "de") & (passed.lemma == r.de_lemma)).any()
        and ((passed.lang == "es") & (passed.lemma == r.es_lemma)).any()
        for r in pairs.itertuples()
    )
    out["flipped_pairs"] = {"n": len(pairs), "both_pass": int(ok)}
    return out


def run(
    model_id: str,
    stimuli_path: str,
    out_root: str = "results/phase0",
    device: str | None = None,
    dtype: str | None = None,
    langs: list[str] | None = None,
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
    items = pd.DataFrame(rows).drop(columns="Index")
    items["meta_pred"] = (items.meta_margin > 0).map({True: "m", False: "f"})
    items["ctx_pred"] = (items.ctx_margin > 0).map({True: "m", False: "f"})
    items["meta_correct"] = items.meta_pred == items.gender
    items["ctx_correct"] = items.ctx_pred == items.gender
    items["both_correct"] = items.meta_correct & items.ctx_correct

    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "items.csv", index=False)
    summary = {"meta": meta, "results": summarize(items)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["results"], indent=2))
    print(f"Wrote {out}/")
    return out


def compare(out_root: str = "results/phase0") -> pd.DataFrame:
    rows = []
    for f in sorted(Path(out_root).glob("*/summary.json")):
        s = json.loads(f.read_text())
        for lang, r in s["results"].items():
            if lang == "flipped_pairs":
                continue
            rows.append(
                {
                    "model": s["meta"]["model_id"],
                    "params_B": round(s["meta"]["n_params"] / 1e9, 2),
                    "device": s["meta"]["device"],
                    "dtype": s["meta"]["dtype"],
                    "lang": lang,
                    "n": r["n"],
                    "both_bal_acc": round(r["both_bal_acc"], 3),
                    "meta_bal_acc": round(r["meta_bal_acc"], 3),
                    "ctx_bal_acc": round(r["ctx_bal_acc"], 3),
                    "acc_m": round(r["acc_m"], 3),
                    "acc_f": round(r["acc_f"], 3),
                    "agree": round(r["measures_agree"], 3),
                    "mass": round(r["mean_meta_mass"], 3),
                    "flipped_ok": f"{s['results']['flipped_pairs']['both_pass']}"
                    f"/{s['results']['flipped_pairs']['n']}",
                }
            )
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values(["lang", "both_bal_acc"], ascending=[True, False])
        table.to_csv(Path(out_root) / "comparison.csv", index=False)
    return table
