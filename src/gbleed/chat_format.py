"""Chat-template inputs for the instruct replication (docs/design/instruct-replication-prereg.md).

Formats (stage A, sensitivity to the raw primary):
  A-en         user "Say one word."   / assistant: the noun, then end-of-turn
  A-something  user "Say something."  / assistant: the noun, then end-of-turn
  A-lang       user "say something" in the noun's language / assistant: the noun, end-of-turn
  U            user: the noun / the template's generation prompt
LAST = the noun's last token; AFTER = the first template token after the noun (A: the
end-of-turn marker). The template is rendered once with a sentinel in place of the noun and
split there, so every noun sits in identical scaffolding; nouns whose tokens merge with the
scaffolding when the whole text is tokenised jointly are flagged (`boundary_ok`) and dropped.

Activations go to activations_chat/<format>/<model slug>/<name>{,_after}.npy, rows in the
order of the raw file of the same name, so the unchanged analyses run with
GBLEED_ACT_DIR=activations_chat/<format>.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SENTINEL = "⁣XNOUNX⁣"
LANG_PROMPT = {"es": "Dime algo.", "de": "Sag etwas.", "ru": "Скажи что-нибудь."}
FORMATS = {"A-en": "Say one word.", "A-something": "Say something.", "A-lang": None, "U": None}
NAMES = {"phase2_bare": "es", "phase3_bare": "de", "phase4_pairs_es": "es",
         "phase4_pairs_de": "de", "phase4_ru": "ru", "phase5_number": None}  # fmt: skip
OUT = Path("activations_chat")


def _template_kwargs(tok) -> dict:
    """Qwen3 hybrid models: non-thinking mode (prereg)."""
    t = tok.chat_template or ""
    return {"enable_thinking": False} if "enable_thinking" in t else {}


def scaffold(tok, fmt: str, lang: str | None) -> tuple[str, str]:
    """(text before the noun, text after it) for one format and language."""
    kw = _template_kwargs(tok)
    if fmt == "U":
        msgs = [{"role": "user", "content": SENTINEL}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    else:
        prompt = LANG_PROMPT[lang] if fmt == "A-lang" else FORMATS[fmt]
        msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": SENTINEL}]
        text = tok.apply_chat_template(msgs, tokenize=False, **kw)
    assert text.count(SENTINEL) == 1, (fmt, text)
    pre, post = text.split(SENTINEL)
    assert post, "nothing follows the noun: no AFTER position"
    return pre, post


def encode(tok, word: str, pre: str, post: str) -> tuple[list[int], int, int, bool]:
    """(ids, LAST index, AFTER index, boundary_ok)."""
    p = tok(pre, add_special_tokens=False)["input_ids"]
    w = tok(word, add_special_tokens=False)["input_ids"]
    q = tok(post, add_special_tokens=False)["input_ids"]
    ids = p + w + q
    ok = tok(pre + word + post, add_special_tokens=False)["input_ids"] == ids
    return ids, len(p) + len(w) - 1, len(p) + len(w), ok


def _langs(name: str, words: list[str]) -> list[str]:
    if NAMES[name] is not None:
        return [NAMES[name]] * len(words)
    num = pd.read_csv("data/stimuli/phase5_number_v1.csv", keep_default_na=False)
    m = {
        **dict(zip(num.lemma, num.lang, strict=True)),
        **dict(zip(num.plural, num.lang, strict=True)),
    }
    return [m[w] for w in words]


@torch.no_grad()
def states(model, ids_list, last, after, batch_size: int = 32):
    pad = 0
    out_l, out_a = [], []
    for i in range(0, len(ids_list), batch_size):
        b = ids_list[i : i + batch_size]
        width = max(map(len, b))
        ids = torch.full((len(b), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, s in enumerate(b):
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        hs = model.get_decoder()(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                                 output_hidden_states=True, use_cache=False).hidden_states  # fmt: skip
        rows = torch.arange(len(b), device=model.device)
        li = torch.tensor(last[i : i + batch_size], device=model.device)
        ai = torch.tensor(after[i : i + batch_size], device=model.device)
        out_l.append(torch.stack([h[rows, li] for h in hs], 1).float().cpu().numpy())
        out_a.append(torch.stack([h[rows, ai] for h in hs], 1).float().cpu().numpy())
    return np.concatenate(out_l), np.concatenate(out_a)


def extract(model_id: str, formats=tuple(FORMATS), names=tuple(NAMES), device=None, dtype=None):
    """Chat-format activations for every raw file already extracted for this model."""
    from datetime import UTC, datetime

    from . import activations as acts
    from .models import load_model, model_slug, pick_device, pick_dtype, run_metadata, stage

    dev = pick_device(device)
    dt = pick_dtype(dtype, dev)
    model, tok = load_model(model_id, dev, dt)
    meta0 = run_metadata(model, model_id, dev, dt)
    report = {}
    for fmt in formats:
        d = OUT / fmt / model_slug(model_id)
        d.mkdir(parents=True, exist_ok=True)
        for name in names:
            try:
                _, raw = acts.load(model_id, name)
            except FileNotFoundError:
                continue
            words = list(raw["words"])
            langs = _langs(name, words)
            enc = [
                encode(tok, w, *scaffold(tok, fmt, lg)) for w, lg in zip(words, langs, strict=True)
            ]
            ok = [e[3] for e in enc]
            stage(f"{model_id} {fmt} {name}: {len(words)} nouns, boundary merges {ok.count(False)}")
            XL, XA = states(model, [e[0] for e in enc], [e[1] for e in enc], [e[2] for e in enc])
            pre, post = scaffold(tok, fmt, langs[0])
            meta = {**meta0, "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                    "format": fmt, "scaffold_pre": pre, "scaffold_post": post,
                    "boundary_ok": ok, "raw_source": name}  # fmt: skip
            toks = [tok.convert_ids_to_tokens(tok(w, add_special_tokens=False)["input_ids"])
                    for w in words]  # fmt: skip
            for suffix, X, pos in (("", XL, "last"), ("_after", XA, "after")):
                np.save(d / f"{name}{suffix}.npy", X)
                (d / f"{name}{suffix}.json").write_text(json.dumps(
                    {**meta, "position": pos, "words": words, "tokens": toks,
                     "shape": list(X.shape)}, ensure_ascii=False, indent=1))  # fmt: skip
            report[f"{fmt}/{name}"] = {"n": len(words), "boundary_merges": ok.count(False)}
    (OUT / f"report_{model_slug(model_id)}.json").write_text(json.dumps(report, indent=1))
    return report
