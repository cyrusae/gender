"""Gemma slim prereg G0: technical gates before any Gemma 3 result is computed.

Per model (`docs/design/gemma-slim-prereg.md`, G0 table):
  dtype     parameters are bf16
  loading   20 fixed German/Spanish prompts give finite next-token log-probs; top-1 continuations
            of "Der Hund" / "El perro" are recorded for a plausibility read
  softcap   the batch scorer equals the model's own forward log-softmax (max abs diff < 1e-2)
  bos       exactly one <bos> in both the activation format and the scorer's prompts
  sink      per-position residual norm at every layer, 20 prompts in our exact formats: any
            non-BOS position > 10x its layer's median fails
  sae       Gemma Scope 2 layer L reconstructs our hidden_states[L + 1] best, at every published
            layer (four; the prereg's minimum was three)

Writes results/gemma_g0/<slug>.json; `passed` is the AND of the mechanical checks (the
plausibility read is reported, not scored).
"""

from __future__ import annotations

import json
import math
import os

import pandas as pd
import torch

from .activations import last_token_states, prefix_ids
from .models import load_model, model_slug
from .scoring import _ids, continuation_logprobs_batch

MODELS = ("google/gemma-3-4b-pt", "google/gemma-3-12b-pt")
OUT = "results/gemma_g0"
PROMPTS = [
    "Der Hund", "Die Katze schläft auf dem", "Ich gehe morgen in die", "Das Wetter ist heute",
    "Berlin ist die Hauptstadt von", "Er trinkt jeden Morgen einen", "Die Kinder spielen im",
    "Wir fahren mit dem Zug nach", "Das Buch liegt auf dem", "Sie hat eine neue",
    "El perro", "La casa está en la", "Mañana voy a la", "El tiempo hoy es muy",
    "Madrid es la capital de", "Ella bebe cada mañana un", "Los niños juegan en el",
    "Vamos en tren a", "El libro está sobre la", "Él tiene una nueva",
]  # fmt: skip
SINK_RATIO = 10.0
SOFTCAP_TOL = 1e-2


def _words(n: int = 200) -> list[str]:
    s = pd.read_csv("data/stimuli/phase3_final_v3.csv", keep_default_na=False)
    s = s[s.lang.isin(["de", "es"])].sample(n=n, random_state=0)
    return list(s.lemma)


@torch.no_grad()
def _forward(model, ids: list[int]):
    t = torch.tensor([ids], device=model.device)
    return model(input_ids=t, output_hidden_states=True, use_cache=False)


def run(model_id: str) -> dict:
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, tok = load_model(model_id, dev, torch.bfloat16)
    r: dict = {"model": model_id, "softcap_config": model.config.final_logit_softcapping}
    bos = tok.bos_token_id

    # dtype
    dts = {str(p.dtype) for p in model.parameters()}
    r["dtype"] = {"dtypes": sorted(dts), "pass": dts == {"torch.bfloat16"}}

    # loading + softcap: scorer vs the model's own forward on the SAME input (prompt + top-1
    # token, read at the prompt's last position). Comparing against a forward on the prompt
    # alone mixes in bf16 shape noise (~0.07 nat on Qwen3-1.7B and Gemma 4B alike, 2026-10-22).
    top, finite, reqs = {}, True, []
    for p in PROMPTS:
        pi = _ids(tok, p, special=True)
        lp = torch.log_softmax(_forward(model, pi).logits[0, -1].float(), -1)
        finite &= bool(torch.isfinite(lp).all())
        i = int(lp.argmax())
        top[p] = tok.decode([i])
        if _ids(tok, top[p], special=False) == [i]:  # decode/encode round-trip is one token
            same = torch.log_softmax(_forward(model, pi + [i]).logits[0, -2].float(), -1)
            reqs.append((p, top[p], float(same[i])))
    pairs = [(p, c) for p, c, _ in reqs]
    one = continuation_logprobs_batch(model, tok, pairs, batch_size=1, desc="g0 scorer")
    batched = continuation_logprobs_batch(model, tok, pairs, batch_size=8, desc="g0 scorer b8")
    diffs = [abs(g - w) for g, (_, _, w) in zip(one, reqs, strict=True)]
    r["loading"] = {"finite": finite, "top1": top, "pass": finite}
    r["softcap"] = {"n_compared": len(diffs), "max_abs_diff": max(diffs) if diffs else None,
                    "batched_vs_unbatched_max": max(abs(a - b) for a, b in zip(one, batched)),
                    "pass": bool(diffs) and max(diffs) < SOFTCAP_TOL}  # fmt: skip

    # bos
    act = prefix_ids(tok) + _ids(tok, " Hund", special=False)
    sc = _ids(tok, PROMPTS[0], special=True)
    n_act, n_sc = act.count(bos), sc.count(bos)
    r["bos"] = {"activation_format": n_act, "scorer": n_sc, "act_first": act[0] == bos,
                "pass": n_act == 1 and n_sc == 1 and act[0] == bos and sc[0] == bos}  # fmt: skip

    # sink scan: activation format (10 words) and scorer format (10 prompts)
    seqs = [prefix_ids(tok) + _ids(tok, " " + w, special=False) for w in _words(10)]
    seqs += [_ids(tok, p, special=True) for p in PROMPTS[::2]]
    norms = [torch.stack([h[0].float().norm(dim=-1) for h in _forward(model, s).hidden_states])
             for s in seqs]  # each [L+1, t]  # fmt: skip
    worst = []
    for layer in range(norms[0].shape[0]):
        med = float(torch.cat([n[layer, 1:] for n in norms]).median())
        for k, n in enumerate(norms):
            for pos in range(1, n.shape[1]):
                if float(n[layer, pos]) > SINK_RATIO * med:
                    worst.append({"layer": layer, "seq": k, "pos": pos,
                                  "ratio": float(n[layer, pos]) / med})  # fmt: skip
    r["sink"] = {"n_flagged": len(worst), "flagged": worst[:20], "pass": not worst}

    # SAE index alignment
    from . import sae as S

    X, _ = last_token_states(model, tok, _words(200), batch_size=32, desc="g0 sae")
    al = {}
    for L in S.GEMMA_LAYERS[model_id]:  # all published layers (prereg asked for three)
        fv = S.alignment(S.load(model_id, L), X, L)
        best = min(fv, key=fv.get)
        others = sorted(v for o, v in fv.items() if o != 1)
        al[L] = {"fvu": {str(o): v for o, v in fv.items()}, "best_offset": best,
                 "margin": others[0] / fv[1] if 1 in fv and others else None}  # fmt: skip
    r["sae"] = {"layers": al, "pass": all(a["best_offset"] == 1 for a in al.values())}

    r["passed"] = all(r[k]["pass"] for k in ("dtype", "loading", "softcap", "bos", "sink", "sae"))
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{model_slug(model_id)}.json", "w") as f:
        json.dump(r, f, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v)
                      for k, v in r.items() if k not in ("model",)}, indent=1))  # fmt: skip
    del model
    return r


def main(models=MODELS) -> None:
    for m in models:
        print(f"=== G0 {m}")
        run(m)
