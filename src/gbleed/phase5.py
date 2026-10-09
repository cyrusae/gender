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
German: the frame selected from DE_GATE_CANDIDATES by the pre-declared frame-check rule
(phase5_stimuli; revised 2026-10-09 after the dictionary and pronoun frames failed).
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
DOSE_SUBSET_FRAC = 0.25  # random directions across doses run on this share of nouns
DOSE_SUBSET_SEED = 0
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


def first_ids(tok, words) -> list[int]:
    """First token of each word (the German gate labels: the first tokens differ)."""
    ids = [tok(w, add_special_tokens=False)["input_ids"][0] for w in words]
    assert len(set(ids)) == len(ids), words
    return ids


def family(model_id: str) -> str:
    return next(v for k, v in FAMILY.items() if model_id.startswith(k))


# ---- gate -----------------------------------------------------------------------------
def de_reader(tok, frame: str):
    """(read ids, margin function lp[:, read] -> f - m) for a German candidate frame."""
    from .phase5_stimuli import DE_GATE_CANDIDATES

    _, forms, _ = DE_GATE_CANDIDATES[frame]
    im, i_f = single_ids(tok, forms["m"]), single_ids(tok, forms["f"])
    read = im + i_f

    def margin(lp):
        lp = torch.as_tensor(lp)
        return (torch.logsumexp(lp[:, len(im) :], 1) - torch.logsumexp(lp[:, : len(im)], 1)).numpy()

    return read, margin


@torch.no_grad()
def de_calibration(model, tok, frame: str) -> float:
    """Contextual calibration: the margin of the content-free headword (0 if the frame has none)."""
    from .phase5_stimuli import DE_GATE_CANDIDATES

    t, _, cf = DE_GATE_CANDIDATES[frame]
    if cf is None:
        return 0.0
    read, margin = de_reader(tok, frame)
    V = torch.zeros((1, model.config.hidden_size))
    r = run(model, tok, [Prompt(t.format(noun=cf), cf)], 0, [(0, 0, 0.0)], V, read)
    return float(margin(r["lp"])[0])


def gate_prompts(nouns: pd.DataFrame, lang: str, fam: str, de_frame: str | None = None):
    """Spanish: one prompt per noun x gate adjective (stem appended), read the two endings.
    German: one prompt per noun in the selected candidate frame (DE_GATE_CANDIDATES)."""
    from .phase5_stimuli import DE_GATE_CANDIDATES, gate_adjectives

    out = []
    if lang == "es":
        for r in nouns.itertuples():
            for m, f, stem, im, i_f in gate_adjectives(fam):
                out.append((Prompt(ES_GATE.format(N=r.lemma), r.lemma,
                                   {"lemma": r.lemma, "gender": r.gender, "adj": m}, extra=stem),
                            [im, i_f]))  # fmt: skip
    else:
        t = DE_GATE_CANDIDATES[de_frame][0]
        for r in nouns.itertuples():
            out.append((Prompt(t.format(noun=r.lemma), r.lemma,
                               {"lemma": r.lemma, "gender": r.gender, "frame": de_frame}), None))  # fmt: skip
    return out


def gate(model, tok, model_id: str, k: int, nouns: pd.DataFrame, lang: str, v: np.ndarray,
         n_random: int = N_RANDOM_GATE, de_frame: str | None = None) -> dict:  # fmt: skip
    """Agreement margin (feminine minus masculine log-prob; Spanish: mean over adjectives; German:
    dictionary frame) per noun across the gate grid, for v and random vectors (the random ones at
    the flip dose only). Feminine nouns are pushed with -dose, masculine with +dose."""
    from .phase5_stimuli import DE_GATE_PRIMARY

    fam = family(model_id)
    de_frame = de_frame or DE_GATE_PRIMARY
    gp = gate_prompts(nouns, lang, fam, de_frame)
    prompts = [p for p, _ in gp]
    if lang == "es":
        read = sorted({i for _, r in gp for i in r})
    else:
        assert de_frame, "no German gate frame selected (DE_GATE_PRIMARY)"
        read, de_margin = de_reader(tok, de_frame)
        cal = de_calibration(model, tok, de_frame)
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
            m = de_margin(lp) - cal
        return m

    rows = []
    n_p = len(prompts)
    for gi, a in enumerate(grid):
        sl = slice(gi * n_p, (gi + 1) * n_p)
        m = margins(res["lp"][sl], range(n_p))
        df = pd.DataFrame({"lemma": [p.key["lemma"] for p in prompts],
                           "gender": [p.key["gender"] for p in prompts], "m": m,
                           "frame": [p.key.get("frame", "adj") for p in prompts]})  # fmt: skip
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
    from .phase5_stimuli import DE_GATE_PRIMARY, DE_GATE_SECONDARY, NOUNS

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
            frames = [None] if lang == "es" else [DE_GATE_PRIMARY, DE_GATE_SECONDARY]
            for fr in frames:
                g, sec = timed(gate, model, tok, model_id, k, nn, lang, v, de_frame=fr)
                tag = lang if fr is None else f"{lang}_{fr}"
                g["table"].to_csv(out / f"gate_{tag}_k{k}.csv", index=False)
                s = {kk: vv for kk, vv in g.items() if kk != "table"}
                s |= {"frame": fr, "seconds": sec, "n_nouns": len(nn)}
                summary.append(s)
                stage(f"k={k} {tag}: alpha*={g['alpha_star']} random flip "
                      f"{np.mean(g['random_flip']) if g['random_flip'] else float('nan'):.2f} "
                      f"pass={g['passes']} ({sec:.0f} s)")  # fmt: skip
    (out / "summary.json").write_text(json.dumps({"meta": meta, "gate": summary}, indent=1,
                                                 default=float))  # fmt: skip
    return summary


# ---- dose-response sweep (§5.3) with specificity controls -------------------------------
def conditions(n_prompts: int, n_random: int, n_random_sweep: int, extra_vecs: int = 0,
               in_subset: list[bool] | None = None):  # fmt: skip
    """(prompt, vector, dose multiple) for: v (index 0) at every DOSE_MULTS, including 0 (the
    unsteered baseline, shared by every vector); the first n_random_sweep random vectors at every
    nonzero dose on the dose subset's prompts (trim, PI 2026-10-09), else at the working dose
    (1.0) only; the remaining random vectors at the working dose only; extra vectors (number,
    social; indices after the random ones) at every nonzero dose. A zero dose is the same for
    every vector, so it is run once (with v)."""
    nonzero = [m for m in DOSE_MULTS if m != 0.0]
    extras = list(range(1 + n_random, 1 + n_random + extra_vecs))
    conds = []
    for i in range(n_prompts):
        conds += [(i, 0, m) for m in DOSE_MULTS]
        for vi in extras:
            conds += [(i, vi, m) for m in nonzero]
        full = in_subset is None or in_subset[i]
        for vi in range(1, 1 + n_random_sweep):
            conds += [(i, vi, m) for m in nonzero] if full else [(i, vi, 1.0)]
        conds += [(i, vi, 1.0) for vi in range(1 + n_random_sweep, 1 + n_random)]
    return conds


def dose_subset(nouns: pd.DataFrame, frac: float = DOSE_SUBSET_FRAC,
                seed: int = DOSE_SUBSET_SEED) -> set[str]:  # fmt: skip
    """The fixed prompt subset on which random directions run across doses (trim, PI
    2026-10-09): a seeded share of nouns per (language, set, gender), with their English
    concepts (R1-EN/R2-EN prompts are keyed by concept). Every wording of a chosen noun is in."""
    pick = pd.concat([g.sample(max(1, round(frac * len(g))), random_state=seed)
                      for _, g in nouns.groupby(["lang", "set", "gender"])])  # fmt: skip
    return set(pick.lemma) | set(pick.concept_en)


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
        sub = dose_subset(nouns[nouns.lang == lang])
        conds = conditions(len(prompts), n_random, n_random_sweep,
                           in_subset=[p.key["lemma"] in sub for p in prompts])  # fmt: skip
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


def frame_check(model_id: str, out_root: str = "results/phase5_framecheck", device=None,
                dtype=None) -> dict:  # fmt: skip
    """Pre-registered gate frame check (§5.2): unsteered, >= 70% of this model's known Zipf >= 4
    Phase 0 nouns must get the right gender, per gender, in each gate frame."""
    from .models import load_model, pick_device, pick_dtype, run_metadata
    from .phase5_stimuli import DE_GATE_CANDIDATES, gate_adjectives

    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta = run_metadata(model, model_id, dev, dt)
    it = pd.read_csv(f"results/phase0/{model_slug(model_id)}/items.csv", keep_default_na=False)
    it = it[(it.status == "known") & (pd.to_numeric(it.zipf) >= 4)]
    res = {}
    V = torch.zeros((1, model.config.hidden_size))
    for lang in ("es", "de"):
        nn = it[it.lang == lang][["lemma", "gender"]]
        if lang == "es":
            adj = gate_adjectives(family(model_id))
            prompts = [Prompt(ES_GATE.format(N=w), w, {"lemma": w, "gender": g}, extra=st)
                       for w, g in zip(nn.lemma, nn.gender, strict=True) for _, _, st, _, _ in adj]  # fmt: skip
            read = sorted({i for *_, im, i_f in adj for i in (im, i_f)})
            r = run(model, tok, prompts, 0, [(i, 0, 0.0) for i in range(len(prompts))], V, read)
            pos = {t: j for j, t in enumerate(read)}
            ends = [(im, i_f) for w in nn.lemma for *_, im, i_f in adj]
            m = np.array(
                [r["lp"][j, pos[f]] - r["lp"][j, pos[mm]] for j, (mm, f) in enumerate(ends)]
            )
            df = pd.DataFrame({"lemma": [p.key["lemma"] for p in prompts],
                               "gender": [p.key["gender"] for p in prompts], "m": m})  # fmt: skip
            per = df.groupby(["lemma", "gender"]).m.mean().reset_index()
            res["es_adj"] = {g: float(np.mean((per[per.gender == g].m > 0) == (g == "f")))
                             for g in ("m", "f")}  # fmt: skip
        else:
            g = nn.gender.to_numpy()
            for fr, (t, _, _) in DE_GATE_CANDIDATES.items():
                prompts = [Prompt(t.format(noun=w), w, {"lemma": w, "gender": x})
                           for w, x in zip(nn.lemma, g, strict=True)]  # fmt: skip
                read, margin = de_reader(tok, fr)
                r = run(model, tok, prompts, 0, [(i, 0, 0.0) for i in range(len(prompts))], V, read)
                m = margin(r["lp"]) - de_calibration(model, tok, fr)
                right = np.where(g == "f", m > 0, m < 0)
                res[f"de_{fr}"] = {x: float(right[g == x].mean()) for x in ("m", "f")}
    res["n"] = it.groupby("lang").size().to_dict()
    res["passes"] = {k: all(v >= 0.7 for v in d.values()) for k, d in res.items() if k != "n"}
    out = Path(out_root) / model_slug(model_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"meta": meta, **res}, indent=1, default=float))
    print(model_id, json.dumps(res, default=float))
    return res
