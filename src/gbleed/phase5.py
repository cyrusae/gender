"""Phase 5: does steering grammatical gender move social-gender content? (design:
docs/design/phase5-design.md, P1-P22). This module runs the steering; analysis is separate.

Steering vectors (P2): per language and layer, the stratified difference of means (Phase 4's
`class_betas` on the Phase 2 `strat` / Phase 3 `strat3` m/f sets), unit length, pointing
feminine. Doses are in units of the median residual-stream norm at the steered (noun)
positions of the readout prompts at that layer. Steering edits only the noun's tokens (P13).
Layers: about 25/40/55/70% of depth (boundary index k as in `activations.py`).

Readouts (each frame is fixed text; every scored word is a single token, P16):
  R1      foreign noun in an English frame, 4 wordings (P18: W1-W3 averaged, W4 the consensus
          contrast) + W1 without "very"
  R1-EN   English noun only, 4 wordings + W1 without "very"
  R2      forced choice man/woman, male/female, boy/girl (both orders) + narrative pronoun
  R2-EN   the same with the English noun
Gate (§5.2): Spanish *Mi {noun} es muy ___* agreement (adjective endings given their stem);
German dictionary frame (der/die/das) and pronoun frame (Er/Sie/Es).
Damage (§5.3): KL(unsteered || steered) of the next-token distribution over the 20 tokens after
the noun in a neutral sentence.
P21: the top-50 next tokens of every scored prompt are saved alongside.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .estimators import class_betas, nuisance
from .models import model_slug, progress, stage
from .scoring import _head, head_dtype
from .steering import lower, upper

LAYER_FRACS = (0.25, 0.40, 0.55, 0.70)
DOSE_MULTS = (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
GATE_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)  # x median norm
N_RANDOM_GATE = 20
N_RANDOM = 100  # P9: >= 40; 20 across doses, all 100 at the working dose
TOPK = 50
LANG_NAME = {"es": "Spanish", "de": "German"}
FAMILY = {"Qwen/": "qwen3", "utter-project/EuroLLM": "eurollm"}

# ---- frames ----------------------------------------------------------------------
R1 = {
    "W1": 'The {L} word "{N}" means {G}. Described in one word, it is very',
    "W2": 'In {L}, "{G}" is "{N}". Described in one word, it is very',
    "W3": 'The {L} word "{N}" means {G}. I think it is very',
    "W4": 'The {L} word "{N}" means {G}. Everyone says it is very',
    "W1nv": 'The {L} word "{N}" means {G}. Described in one word, it is',
}
R1_EN = {
    "W1": "The {N} is very",
    "W2": "The {N} was very",
    "W3": "I think the {N} is very",
    "W4": "Everyone says the {N} is very",
    "W1nv": "The {N} is",
}
R2_CHOICES = {  # (options as asked, answer frame, scored tokens in the asked order)
    "man_woman": ("a man or a woman", " It would be a", (" man", " woman")),
    "woman_man": ("a woman or a man", " It would be a", (" woman", " man")),
    "male_female": ("male or female", " It would be", (" male", " female")),
    "female_male": ("female or male", " It would be", (" female", " male")),
    "boy_girl": ("a boy or a girl", " It would be a", (" boy", " girl")),
    "girl_boy": ("a girl or a boy", " It would be a", (" girl", " boy")),
}
R2_NARR = (
    "In the story, the {G} ({N}) came to life. Every morning,",
    "The {N} came to life. Every morning,",
)
PRONOUNS = (" she", " he", " it", " they")
NEUTRAL = {
    "es": "Ayer hablamos de mi {N} durante la cena con unos amigos de la familia y luego volvimos a casa en coche.",
    "de": "Gestern stand das Wort {N} an der Tafel, und danach haben wir mit Freunden der Familie zu Abend gegessen.",
    "en": "Yesterday we talked about the {N} over dinner with some family friends, and then we drove home.",
}
ES_GATE = "Mi {N} es muy"
KL_TOKENS = 20


@dataclass
class Prompt:
    text: str
    noun: str  # the steered substring (its first occurrence after `noun_from`)
    key: dict = field(default_factory=dict)  # readout, wording, item, ...
    extra: list[int] = field(default_factory=list)  # token ids appended after the text


def _encode(tok, p: Prompt) -> tuple[list[int], list[int]]:
    """Token ids of the prompt and the positions of the noun's tokens (from character offsets,
    so the noun is tokenized as in the running text)."""
    enc = tok(p.text, add_special_tokens=False, return_offsets_mapping=True)
    start = p.text.index(p.noun)
    end = start + len(p.noun)
    pos = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
    assert pos, (p.text, p.noun)
    return enc["input_ids"] + list(p.extra), pos


def _pad(tok, seqs):
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask


# ---- vectors ----------------------------------------------------------------------
def dom_vector(model_id: str, lang: str, k: int) -> np.ndarray:
    """Unit stratified difference of means (f - m) at boundary k, from the training activations
    on the same device (Phase 2 `strat` for es, Phase 3 `strat3` m/f for de)."""
    from .phase4 import de_train, es_train

    d = es_train(model_id, "last") if lang == "es" else de_train(model_id, "last")
    X = d["X"][:, k].astype(np.float64)
    v = class_betas(X, d["y"][:, None].astype(float), nuisance(d["cells"], d["covs"]))[0]
    return v / np.linalg.norm(v)


def random_vectors(d: int, n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---- the batched runner -------------------------------------------------------------
@torch.no_grad()
def run(model, tok, prompts: list[Prompt], k: int, conds: list[tuple[int, int, float]],
        vecs: torch.Tensor, read_ids: list[int] | None, prompt_batch: int = 16,
        row_batch: int = 64, topk: int = TOPK, unit: float = 1.0) -> dict:  # fmt: skip
    """conds: (prompt index, vector index, dose in units); vecs [n_vec, d] unit vectors.
    Returns per condition: log-probs at read_ids [n_cond, len(read_ids)] and top-k ids/logprobs."""
    enc = [_encode(tok, p) for p in prompts]
    by_prompt: dict[int, list[int]] = {}
    for ci, (pi, _, _) in enumerate(conds):
        by_prompt.setdefault(pi, []).append(ci)
    order = sorted(by_prompt, key=lambda i: len(enc[i][0]))
    dev = model.device
    n = len(conds)
    out_lp = np.zeros((n, len(read_ids or [])), np.float32)
    out_top = np.zeros((n, topk), np.int32)
    out_toplp = np.zeros((n, topk), np.float32)
    rid = torch.tensor(read_ids or [0], device=dev)
    vecs = vecs.to(dev)
    for b0 in range(0, len(order), prompt_batch):
        pis = order[b0 : b0 + prompt_batch]
        ids, mask = _pad(tok, [enc[i][0] for i in pis])
        where = torch.zeros_like(ids, dtype=torch.bool)
        for r, i in enumerate(pis):
            where[r, enc[i][1]] = True
        ids, mask, where = ids.to(dev), mask.to(dev), where.to(dev)
        slot = mask.sum(1) - 1
        h = lower(model, ids, mask, k)
        cis = [c for i in pis for c in by_prompt[i]]
        local = {i: r for r, i in enumerate(pis)}
        for c0 in range(0, len(cis), row_batch):
            chunk = cis[c0 : c0 + row_batch]
            item = torch.tensor([local[conds[c][0]] for c in chunk], device=dev)
            vi = torch.tensor([conds[c][1] for c in chunk], device=dev)
            al = torch.tensor([conds[c][2] * unit for c in chunk], device=dev, dtype=h.dtype)
            hc = h[item].clone()
            hc += (al[:, None, None] * vecs[vi].to(h.dtype)[:, None, :]) * where[item][..., None]
            lp = upper(model, hc, mask[item], k, slot[item])
            if read_ids:
                out_lp[chunk] = lp[:, rid].cpu().numpy()
            t = lp.topk(topk, dim=-1)
            out_top[chunk] = t.indices.cpu().numpy()
            out_toplp[chunk] = t.values.cpu().numpy()
    return {"lp": out_lp, "top": out_top, "toplp": out_toplp}


@torch.no_grad()
def noun_norm(model, tok, prompts: list[Prompt], k: int, batch: int = 32) -> float:
    """Median residual-stream norm at the noun positions at boundary k (the dose unit)."""
    norms = []
    enc = [_encode(tok, p) for p in prompts]
    for b0 in range(0, len(enc), batch):
        e = enc[b0 : b0 + batch]
        ids, mask = _pad(tok, [x[0] for x in e])
        h = lower(model, ids.to(model.device), mask.to(model.device), k)
        for r, (_, pos) in enumerate(e):
            norms += h[r, pos].float().norm(dim=-1).cpu().tolist()
    return float(np.median(norms))


@torch.no_grad()
def damage(model, tok, prompts: list[Prompt], k: int, conds, vecs, unit: float,
           batch_rows: int = 16) -> np.ndarray:  # fmt: skip
    """Mean KL(unsteered || steered) over the KL_TOKENS positions after the noun, per condition."""
    from .steering import _only_layers_from

    enc = [_encode(tok, p) for p in prompts]
    dev = model.device
    W, bias = _head(model, head_dtype(model))
    body = model.get_decoder()
    vecs = vecs.to(dev)
    out = np.zeros(len(conds), np.float32)
    by_prompt: dict[int, list[int]] = {}
    for ci, (pi, _, _) in enumerate(conds):
        by_prompt.setdefault(pi, []).append(ci)
    for pi, cis in by_prompt.items():
        ids_l, pos = enc[pi]
        ids, mask = _pad(tok, [ids_l])
        ids, mask = ids.to(dev), mask.to(dev)
        last = pos[-1]
        span = list(range(last, min(last + KL_TOKENS, len(ids_l) - 1)))
        h = lower(model, ids, mask, k)

        def logp(hh, mm, span=span):
            with _only_layers_from(body, k):
                x = body(inputs_embeds=hh, attention_mask=mm, use_cache=False).last_hidden_state
            lg = x[:, span].to(W.dtype) @ W.T
            if bias is not None:
                lg = lg + bias
            return torch.log_softmax(lg.float(), -1)

        base = logp(h, mask)[0]
        for c0 in range(0, len(cis), batch_rows):
            chunk = cis[c0 : c0 + batch_rows]
            hc = h.repeat(len(chunk), 1, 1)
            al = torch.tensor([conds[c][2] * unit for c in chunk], device=dev, dtype=h.dtype)
            vi = torch.tensor([conds[c][1] for c in chunk], device=dev)
            add = al[:, None] * vecs[vi].to(h.dtype)
            hc[:, pos] += add[:, None, :]
            lp = logp(hc, mask.repeat(len(chunk), 1))
            kl = (base.exp()[None] * (base[None] - lp)).sum(-1).mean(-1)
            out[chunk] = kl.cpu().numpy()
    return out


# ---- prompts per readout ------------------------------------------------------------
def r1_prompts(nouns: pd.DataFrame) -> list[Prompt]:
    out = []
    for r in nouns.itertuples():
        for w, f in R1.items():
            t = f.format(L=LANG_NAME[r.lang], N=r.lemma, G=r.concept_en)
            out.append(Prompt(t, f'"{r.lemma}"', {"readout": "R1", "wording": w, "lemma": r.lemma,
                                                    "lang": r.lang}))  # fmt: skip
    return out


def r1en_prompts(concepts: list[str]) -> list[Prompt]:
    return [Prompt(f.format(N=c), c, {"readout": "R1-EN", "wording": w, "lemma": c, "lang": "en"})
            for c in concepts for w, f in R1_EN.items()]  # fmt: skip


def r2_prompts(nouns: pd.DataFrame | None, concepts: list[str] | None) -> list[Prompt]:
    out = []
    items = ([(r.lemma, r.concept_en, r.lang) for r in nouns.itertuples()] if nouns is not None
             else [(c, c, "en") for c in concepts])  # fmt: skip
    for n, g, lang in items:
        subj = f"the {g} ({n})" if lang != "en" else f"the {n}"
        steer = f"({n})" if lang != "en" else n
        for name, (opts, ans, _) in R2_CHOICES.items():
            t = f"If {subj} were a person, would it be {opts}?{ans}"
            out.append(Prompt(t, steer, {"readout": "R2" if lang != "en" else "R2-EN",
                                         "wording": name, "lemma": n, "lang": lang}))  # fmt: skip
        nf = R2_NARR[0].format(G=g, N=n) if lang != "en" else R2_NARR[1].format(N=n)
        out.append(Prompt(nf, steer, {"readout": "R2" if lang != "en" else "R2-EN",
                                      "wording": "narrative", "lemma": n, "lang": lang}))  # fmt: skip
    return out


def single_ids(tok, words) -> list[int]:
    ids = [tok(w, add_special_tokens=False)["input_ids"] for w in words]
    assert all(len(i) == 1 for i in ids), [
        w for w, i in zip(words, ids, strict=True) if len(i) != 1
    ]
    return [i[0] for i in ids]


def family(model_id: str) -> str:
    return next(v for k, v in FAMILY.items() if model_id.startswith(k))


# ---- gate -----------------------------------------------------------------------------
def gate_prompts(nouns: pd.DataFrame, lang: str, fam: str):
    """Spanish: one prompt per noun x gate adjective (stem appended), read the two endings.
    German: one prompt per noun x frame, read der/die/das (Er/Sie/Es)."""
    from .phase5_stimuli import DE_GATE, gate_adjectives

    out = []
    if lang == "es":
        for r in nouns.itertuples():
            for m, f, stem, im, i_f in gate_adjectives(fam):
                out.append((Prompt(ES_GATE.format(N=r.lemma), r.lemma,
                                   {"lemma": r.lemma, "gender": r.gender, "adj": m}, extra=stem),
                            [im, i_f]))  # fmt: skip
    else:
        for r in nouns.itertuples():
            for fr, (t, _) in DE_GATE.items():
                out.append((Prompt(t.format(noun=r.lemma), r.lemma,
                                   {"lemma": r.lemma, "gender": r.gender, "frame": fr}), None))  # fmt: skip
    return out


def gate(model, tok, model_id: str, k: int, nouns: pd.DataFrame, lang: str, v: np.ndarray,
         n_random: int = N_RANDOM_GATE) -> dict:  # fmt: skip
    """Agreement margin (feminine minus masculine log-prob; Spanish: mean over adjectives; German:
    dictionary frame) per noun across the gate grid, for v and random vectors (the random ones at
    the flip dose only). Feminine nouns are pushed with -dose, masculine with +dose."""
    from .phase5_stimuli import DE_GATE

    fam = family(model_id)
    gp = gate_prompts(nouns, lang, fam)
    prompts = [p for p, _ in gp]
    if lang == "es":
        read = sorted({i for _, r in gp for i in r})
    else:
        read = single_ids(tok, [*DE_GATE["dict"][1].values(), *DE_GATE["pron"][1].values()])
    unit = noun_norm(model, tok, prompts[:: max(1, len(prompts) // 200)], k)
    d = model.config.hidden_size
    V = torch.tensor(np.r_[v[None], random_vectors(d, n_random, seed=k)], dtype=torch.float32)
    sign = np.array([-1.0 if p.key["gender"] == "f" else 1.0 for p in prompts])
    grid = (0.0, *GATE_GRID)
    conds = [(i, 0, s * a) for a in grid for i, s in enumerate(sign)]
    res = run(model, tok, prompts, k, conds, V, read, unit=unit)

    def margins(lp, idx_prompts):
        if lang == "es":
            pos = {t: j for j, t in enumerate(read)}
            m = np.array([lp[j, pos[gp[i][1][1]]] - lp[j, pos[gp[i][1][0]]]
                          for j, i in enumerate(idx_prompts)])  # fmt: skip
        else:
            m = lp[:, 1] - lp[:, 0]  # dictionary frame: die - der
            fr = np.array([prompts[i].key["frame"] for i in idx_prompts])
            m = np.where(fr == "dict", m, lp[:, 4] - lp[:, 3])  # pronoun frame: Sie - Er
        return m

    rows = []
    n_p = len(prompts)
    for gi, a in enumerate(grid):
        sl = slice(gi * n_p, (gi + 1) * n_p)
        m = margins(res["lp"][sl], range(n_p))
        df = pd.DataFrame({"lemma": [p.key["lemma"] for p in prompts],
                           "gender": [p.key["gender"] for p in prompts], "m": m,
                           "frame": [p.key.get("frame", "adj") for p in prompts]})  # fmt: skip
        if lang == "de":
            df = df[df.frame == "dict"]
        per = df.groupby(["lemma", "gender"]).m.mean().reset_index()
        rows.append({"dose": a, "median_f": float(per[per.gender == "f"].m.median()),
                     "median_m": float(per[per.gender == "m"].m.median()),
                     "flipped": float(np.mean(np.where(per.gender == "f", per.m < 0, per.m > 0)))})  # fmt: skip
    tab = pd.DataFrame(rows)
    crossed = tab[(tab.dose > 0) & (tab.median_f < 0) & (tab.median_m > 0)]
    a_star = float(crossed.dose.min()) if len(crossed) else float("nan")
    rand_flip = []
    if np.isfinite(a_star):
        conds_r = [(i, 1 + j, s * a_star) for j in range(n_random) for i, s in enumerate(sign)]
        rr = run(model, tok, prompts, k, conds_r, V, read, unit=unit)
        for j in range(n_random):
            sl = slice(j * n_p, (j + 1) * n_p)
            m = margins(rr["lp"][sl], range(n_p))
            df = pd.DataFrame({"lemma": [p.key["lemma"] for p in prompts],
                               "gender": [p.key["gender"] for p in prompts], "m": m,
                               "frame": [p.key.get("frame", "adj") for p in prompts]})  # fmt: skip
            if lang == "de":
                df = df[df.frame == "dict"]
            per = df.groupby(["lemma", "gender"]).m.mean().reset_index()
            rand_flip.append(float(np.mean(np.where(per.gender == "f", per.m < 0, per.m > 0))))
    passes = bool(np.isfinite(a_star) and np.mean(rand_flip) < 0.10)
    return {"layer": k, "lang": lang, "unit": unit, "alpha_star": a_star, "table": tab,
            "random_flip": rand_flip, "passes": passes}  # fmt: skip


def layers_for(model) -> list[int]:
    n = model.config.num_hidden_layers
    return sorted({max(1, round(f * n)) for f in LAYER_FRACS})


# ---- timing ----------------------------------------------------------------------------
def timed(fn, *a, **kw):
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    r = fn(*a, **kw)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return r, time.time() - t0


def run_gate(model_id: str, out_root: str = "results/phase5_gate", device=None, dtype=None,
             max_nouns: int | None = None) -> list[dict]:  # fmt: skip
    """§5.2 gate at every grid layer, both languages, on the steering nouns (steer_es /
    steer_de). Writes per-layer tables and a summary with timings."""
    from datetime import UTC, datetime

    from .models import load_model, pick_device, pick_dtype, run_metadata
    from .phase5_stimuli import NOUNS

    nouns = pd.read_csv(NOUNS, keep_default_na=False)
    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    meta["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for k in progress(layers_for(model), desc="gate layers", unit="layer"):
        for lang in ("es", "de"):
            nn = nouns[nouns.set == f"steer_{lang}"]
            if max_nouns:
                nn = nn.groupby("gender").head(max_nouns // 2)
            v = dom_vector(model_id, lang, k)
            g, sec = timed(gate, model, tok, model_id, k, nn, lang, v)
            g["table"].to_csv(out / f"gate_{lang}_k{k}.csv", index=False)
            s = {kk: vv for kk, vv in g.items() if kk != "table"}
            s |= {"seconds": sec, "n_nouns": len(nn)}
            summary.append(s)
            stage(f"k={k} {lang}: alpha*={g['alpha_star']} random flip "
                  f"{np.mean(g['random_flip']) if g['random_flip'] else float('nan'):.2f} "
                  f"pass={g['passes']} ({sec:.0f} s)")  # fmt: skip
    (out / "summary.json").write_text(json.dumps({"meta": meta, "gate": summary}, indent=1,
                                                 default=float))  # fmt: skip
    return summary


# ---- dose-response sweep (§5.3) with specificity controls -------------------------------
def conditions(n_prompts: int, n_random: int, n_random_sweep: int, extra_vecs: int = 0):
    """(prompt, vector, dose multiple) for: v (index 0) and the first n_random_sweep random
    vectors at every DOSE_MULTS; the remaining random vectors at the working dose (1.0) only;
    extra vectors (number, social; indices after the random ones) at every dose."""
    conds = []
    sweep_vecs = [0, *range(1, 1 + n_random_sweep)]
    sweep_vecs += list(range(1 + n_random, 1 + n_random + extra_vecs))
    for i in range(n_prompts):
        for vi in sweep_vecs:
            conds += [(i, vi, m) for m in DOSE_MULTS]
        conds += [(i, vi, 1.0) for vi in range(1 + n_random_sweep, 1 + n_random)]
    return conds


def readout_sets(tok, nouns: pd.DataFrame, lang: str, fam: str):
    """(name, prompts, read ids) for one steering language: R1/R2 on that language's nouns,
    R1-EN/R2-EN on their English concepts."""
    adj = pd.read_csv(f"data/stimuli/phase5_adjectives_{fam}_v1.csv", keep_default_na=False)
    adj_ids = adj.token_id.astype(int).tolist()
    r2_ids = single_ids(
        tok, sorted({w for _, _, ws in R2_CHOICES.values() for w in ws}) + list(PRONOUNS)
    )
    nl = nouns[nouns.lang == lang]
    concepts = sorted(set(nl.concept_en))
    return [("R1", r1_prompts(nl), adj_ids), ("R1-EN", r1en_prompts(concepts), adj_ids),
            ("R2", r2_prompts(nl, None), r2_ids), ("R2-EN", r2_prompts(None, concepts), r2_ids)]  # fmt: skip


def sweep(model_id: str, k: int, lang: str, alpha_star: float, out_root: str = "results/phase5",
          device=None, dtype=None, nouns_filter=None, n_random: int = N_RANDOM,
          n_random_sweep: int = N_RANDOM_GATE, readouts=None, model_tok=None) -> dict:  # fmt: skip
    """Run every readout for one steering language at boundary k. Saves per readout an .npz
    (log-probs at the read ids, top-50) and a condition table; returns timings."""
    from .phase5_stimuli import NOUNS

    if model_tok is None:
        from .models import load_model, pick_device, pick_dtype

        dev = pick_device(device)
        model, tok = load_model(model_id, dev, pick_dtype(dtype, dev))
    else:
        model, tok = model_tok
    nouns = pd.read_csv(NOUNS, keep_default_na=False)
    if nouns_filter is not None:
        nouns = nouns_filter(nouns)
    fam = family(model_id)
    v = dom_vector(model_id, lang, k)
    d = model.config.hidden_size
    V = torch.tensor(
        np.r_[v[None], random_vectors(d, n_random, seed=1000 + k)], dtype=torch.float32
    )
    out = Path(out_root) / model_slug(model_id) / f"{lang}_k{k}"
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    for name, prompts, read in readout_sets(tok, nouns, lang, fam):
        if readouts and name not in readouts:
            continue
        unit = noun_norm(model, tok, prompts[:: max(1, len(prompts) // 200)], k)
        conds = conditions(len(prompts), n_random, n_random_sweep)
        dose = alpha_star  # multiples of alpha* (in norm units)
        cc = [(i, vi, m * dose) for i, vi, m in conds]
        res, sec = timed(run, model, tok, prompts, k, cc, V, read, unit=unit)
        keys = pd.DataFrame([prompts[i].key for i, _, _ in conds])
        keys["vec"] = [vi for _, vi, _ in conds]
        keys["mult"] = [m for _, _, m in conds]
        keys.to_csv(out / f"{name}_conds.csv.gz", index=False)
        np.savez_compressed(out / f"{name}.npz", lp=res["lp"], top=res["top"],
                            toplp=res["toplp"], read=np.array(read))  # fmt: skip
        timings[name] = {"prompts": len(prompts), "conditions": len(conds), "seconds": sec,
                         "rows_per_s": len(conds) / sec, "unit": unit}  # fmt: skip
        stage(f"{model_id} k={k} {lang} {name}: {len(conds)} rows in {sec:.0f} s "
              f"({len(conds) / sec:.0f}/s)")  # fmt: skip
    (out / "timings.json").write_text(json.dumps({"alpha_star": alpha_star, **timings}, indent=1))
    return timings
