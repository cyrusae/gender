"""Generate character paragraphs from instruct models.

  python instruct/gen.py --config instruct/configs/pilot0_v1.json \
      --nouns instruct/stimuli/pilot0_nouns_v1.csv --model Qwen/Qwen3-8B --backend vllm \
      --out instruct/results/pilot0/Qwen__Qwen3-8B.jsonl
Smoke (Mac): --backend hf --model Qwen/Qwen3-0.6B --n 2 --only-nouns pot --only-mods plain,best
OpenRouter: --backend openrouter (model needs an "openrouter" block in the config: id + provider pin;
  key from $OPENROUTER_API_KEY or ~/.config/openrouter/gender-key)

One jsonl row per completion with full provenance. Reasoning ('<think>') is split from the answer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from code_outputs import split_think


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def load_nouns(path: Path) -> list[str]:
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    return [r["noun"] for r in csv.DictReader(lines)]


def check_reserved(nouns: list[str], cfg_name: str) -> None:
    """Burned pilot nouns may only be used by the list that reserved them."""
    res = Path(__file__).parent / "stimuli" / "reserved_nouns.csv"
    for r in csv.DictReader(res.read_text().splitlines()):
        if r["noun"] in nouns and not r["reserved_by"].startswith(cfg_name.split("_v")[0]):
            raise SystemExit(f"noun {r['noun']!r} is reserved by {r['reserved_by']}")


LANG_NAMES = {"de": "German", "ru": "Russian", "es": "Spanish", "hi": "Hindi", "ar": "Arabic",
              "fi": "Finnish", "ja": "Japanese", "tr": "Turkish"}  # fmt: skip


def plain_spelling(w: str, lang: str) -> str:
    """Everyday spelling: Wiktionary headwords mark Russian stress (клавиату́ра) and Arabic short
    vowels (مِطْرَقَة); ordinary text does not."""
    import re
    import unicodedata

    w = unicodedata.normalize("NFC", w)
    if lang == "ru":
        w = unicodedata.normalize(
            "NFC", unicodedata.normalize("NFD", w).replace("\u0301", "").replace("\u0300", "")
        )
    elif lang == "ar":
        w = re.sub("[\u064b-\u065f\u0670]", "", w)
    return w


def load_translations(path: Path) -> dict[str, dict[str, str]]:
    """{english noun: {lang: foreign word}} from a *_translations_*.csv (blank = arm skipped)."""
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    out = {}
    for r in csv.DictReader(lines):
        out[r["en"]] = {lg: plain_spelling(r[f"{lg}_word"], lg) for lg in LANG_NAMES
                        if r.get(f"{lg}_word")}  # fmt: skip
    return out


def build_writer(cfg: dict, nouns: list[str], trans: dict) -> list[dict]:
    """Writer arms (2026-10-26): per language, the foreign word and the English word, both in
    quotes and without an article; plus one plain-English arm with no persona. Arms whose
    translation failed are skipped. 'modifier' holds the arm id (lang:foreign|english|plain)."""
    instr = cfg["name_conditions"][cfg["writer_name_condition"]]
    out = []
    for noun in nouns:
        arms = [("plain", None, noun)]
        for lg, fw in trans.get(noun, {}).items():
            arms += [(f"{lg}:foreign", lg, fw), (f"{lg}:english", lg, noun)]
        for arm, lg, word in arms:
            t = cfg["template_plain"] if lg is None else cfg["template"]
            p = t.replace("{LANGUAGE}", LANG_NAMES.get(lg, "")).replace("{WORD}", f'"{word}"')
            p = p.replace("{NOUN}", noun).replace("{NAME_INSTR}", instr)
            out.append({"noun": noun, "modifier": arm, "mod_noun": word, "lang": lg or "",
                        "foreign_word": trans.get(noun, {}).get(lg, "") if lg else "",
                        "prompt": p, "name_condition": cfg["writer_name_condition"]})  # fmt: skip
    return out


def build_casey(cfg: dict, nouns: list[str], n: int, seed: int) -> list[dict]:
    """Casey/Riley frames: name x pronoun crossed; role and genre drawn at random per sample
    (seeded by noun/name/pronoun/sample), so each sample is its own prompt."""
    out = []
    for noun in nouns:
        for mk, mt in cfg["modifiers"].items():
            mod_noun = mt.format(a=article(noun), noun=noun)
            for name in cfg["names"]:
                for pk, (subj, poss) in cfg["pronouns"].items():
                    for j in range(n):
                        rng = random.Random(f"{seed}|{noun}|{mk}|{name}|{pk}|{j}")
                        role, genre = rng.choice(cfg["roles"]), rng.choice(cfg["genres"])
                        p = cfg["template"].replace("{SUBJ}", subj).replace("{POSS}", poss)
                        p = p.replace("{NAME}", name).replace("{ROLE}", role)
                        p = p.replace("{GENRE_NOVEL}", f"{genre} novel" if genre else "novel")
                        p = p.replace("{MOD_NOUN}", mod_noun).replace("{NOUN}", noun)
                        p = p.replace("{A_GENRE}", "an" if genre and genre[0] in "aeiou" else "a")
                        out.append({"noun": noun, "modifier": mk, "mod_noun": mod_noun,
                                    "name_condition": f"{name}|{pk}", "given_name": name,
                                    "given_pron": pk, "role": role, "genre": genre or "none",
                                    "prompt": p, "fixed_sample": j})  # fmt: skip
    return out


def build_prompts(cfg: dict, nouns: list[str], mods: list[str] | None) -> list[dict]:
    """noun x modifier (x name_condition when the config has them: the template's {NAME_INSTR}
    is filled per condition)."""
    conds = cfg.get("name_conditions") or {None: ""}
    out = []
    for nc, instr in conds.items():
        for noun in nouns:
            for mk, mt in cfg["modifiers"].items():
                if mods and mk not in mods:
                    continue
                mod_noun = mt.format(a=article(noun), noun=noun)
                p = cfg["template"].replace("{NAME_INSTR}", instr)
                p = p.replace("{MOD_NOUN}", mod_noun).replace("{NOUN}", noun)
                out.append({"noun": noun, "modifier": mk, "mod_noun": mod_noun, "prompt": p}
                       | ({"name_condition": nc} if nc else {}))  # fmt: skip
    return out


def sampling_key(family: str, thinking: bool) -> str:
    if family == "qwen3":
        return "qwen3_thinking" if thinking else "qwen3_nothinking"
    return family


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def render(tok, prompt: str, thinking: bool, family: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    # Qwen3 and Gemma 4 both take enable_thinking (Gemma 4: <|think|> in the system turn; off =
    # an empty <|channel>thought\n<channel|> prefilled). Gemma 3 has no thinking mode.
    kw = {"enable_thinking": thinking} if family in ("qwen3", "gemma4") else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def run_vllm(model, texts, sp, n, seed):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="bfloat16", seed=seed, max_model_len=sp["max_tokens"] + 2048)
    params = SamplingParams(n=n, temperature=sp["temperature"], top_p=sp.get("top_p", 1.0),
                            top_k=sp.get("top_k", -1), min_p=sp.get("min_p", 0.0),
                            max_tokens=sp["max_tokens"], seed=seed,
                            skip_special_tokens=False)  # fmt: skip
    res = llm.generate(texts, params)
    return [[(o.text, o.finish_reason, len(o.token_ids)) for o in r.outputs] for r in res]


def run_hf(model, texts, sp, n, seed):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.bfloat16 if dev == "cuda" else torch.float16 if dev == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(dev).eval()
    torch.manual_seed(seed)
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).to(dev)
        g = m.generate(**ids, do_sample=True, temperature=sp["temperature"], top_p=sp["top_p"],
                       top_k=sp["top_k"], max_new_tokens=sp["max_tokens"], num_return_sequences=n)  # fmt: skip
        L = ids["input_ids"].shape[1]
        row = []
        for seq in g:
            new = seq[L:]
            eos = (new == tok.eos_token_id) | (new == tok.pad_token_id)
            k = int(eos.nonzero()[0]) if eos.any() else len(new)
            row.append(
                (
                    tok.decode(new[:k], skip_special_tokens=False),
                    "stop" if eos.any() else "length",
                    k,
                )
            )
        out.append(row)
    return out


OR_URL = "https://openrouter.ai/api/v1/chat/completions"


def or_key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        f = Path.home() / ".config" / "openrouter" / "gender-key"
        k = f.read_text().strip() if f.exists() else ""
    if not k:
        raise SystemExit(
            "no OpenRouter key: set OPENROUTER_API_KEY or ~/.config/openrouter/gender-key"
        )
    return k


def or_call(body: dict, key: str, tries: int = 9) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(OR_URL, data=data, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})  # fmt: skip
    err = ""
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read())
            if "choices" in d:
                return d
            err = json.dumps(d.get("error", d))[:500]
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:500]!r}"
            if e.code not in (408, 429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError) as e:
            err = repr(e)
        time.sleep(min(60, 2**i) + random.random() * 2)
    return {"error": err}


def run_openrouter(ocfg: dict, prompts: list[str], sp: dict, n: int, seed: int, thinking: bool,
                   skip=frozenset(), on_result=None):  # fmt: skip
    """One request per sample (n>1 is not portable across providers). Each result
    (text, finish_reason, n_tokens, extra), extra = provider, usage, reasoning, error, is passed to
    on_result(i, j, result) as soon as it arrives (so an interrupted run keeps what it has);
    (i, j) pairs in `skip` are not requested (resume)."""
    key = or_key()
    jobs = [(i, j) for i in range(len(prompts)) for j in range(n) if (i, j) not in skip]

    def one(ij):
        i, j = ij
        body = {"model": ocfg["id"], "messages": [{"role": "user", "content": prompts[i]}],
                "max_tokens": sp["max_tokens"], "seed": seed * 1000 + j,
                "provider": ocfg["provider"], "usage": {"include": True}}  # fmt: skip
        # only the sampling settings the vendor specifies (closed reasoning models take none)
        body |= {k: sp[k] for k in ("temperature", "top_p", "top_k", "min_p") if k in sp}
        if ocfg.get("reasoning_param"):
            body["reasoning"] = {"enabled": thinking}
        if ocfg.get("service_tier"):
            body["service_tier"] = ocfg["service_tier"]
        d = or_call(body, key)
        if "error" in d:
            return ("", "error", 0, {"error": d["error"]})
        ch = d["choices"][0]
        msg = ch.get("message") or {}
        u = d.get("usage") or {}
        return (msg.get("content") or "", ch.get("finish_reason"), u.get("completion_tokens", 0),
                {"provider": d.get("provider"), "or_model": d.get("model"), "usage": u,
                 "reasoning": msg.get("reasoning"), "native_finish_reason":
                 ch.get("native_finish_reason"), "service_tier": d.get("service_tier")})  # fmt: skip

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one, ij): ij for ij in jobs}
        for f in as_completed(futs):
            on_result(*futs[f], f.result())


def row_key(r: dict) -> tuple:
    return (r.get("name_condition"), r["noun"], r["modifier"], bool(r["thinking"]), r["sample"],
            r.get("fixed_sample"))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--nouns", type=Path, required=True)
    ap.add_argument("--translations", type=Path, help="writer configs: pilot translations csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", help="override (qwen3|gemma3); else from config")
    ap.add_argument("--backend", choices=["vllm", "hf", "openrouter"], default="vllm")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, help="samples per cell (default: config)")
    ap.add_argument("--thinking", choices=["config", "on", "off"], default="config")
    ap.add_argument("--only-nouns")
    ap.add_argument("--only-mods")
    ap.add_argument("--only-conds", help="comma-separated name conditions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="overwrite --out instead of resuming")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    mcfg = cfg["models"].get(args.model, {})
    family = args.family or mcfg["family"]
    thinks = {"on": [True], "off": [False]}.get(args.thinking) or mcfg.get("thinking", [False])
    nouns = load_nouns(args.nouns)
    if args.only_nouns:
        nouns = [x for x in nouns if x in args.only_nouns.split(",")]
    check_reserved(nouns, cfg["name"])
    kind = cfg.get("kind", "character")
    if kind == "writer":
        items = build_writer(cfg, nouns, load_translations(args.translations))
    elif kind == "casey":
        items = build_casey(cfg, nouns, args.n or cfg["samples_per_cell"], args.seed)
    else:
        items = build_prompts(cfg, nouns, args.only_mods.split(",") if args.only_mods else None)
    if args.only_conds:
        items = [x for x in items if x.get("name_condition") in args.only_conds.split(",")]
    n = args.n or cfg["samples_per_cell"]
    if kind == "casey":
        n = 1  # each sample is its own prompt (role/genre drawn per sample)
    if args.backend != "openrouter":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
    meta = {"model": args.model, "backend": args.backend, "config": cfg["name"],
            "config_sha": hashlib.sha256(args.config.read_bytes()).hexdigest()[:12],
            "git_commit": git_commit(), "seed": args.seed}  # fmt: skip
    if args.backend == "vllm":
        import vllm

        meta["backend_version"] = vllm.__version__
    if args.backend == "openrouter":
        meta["openrouter"] = mcfg["openrouter"]
    tot = {"cost": 0.0, "errors": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Resume: rows already in --out that are not errors are kept and not re-requested.
    done = set()
    if args.out.exists() and not args.fresh:
        for ln in args.out.read_text().splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:  # half-written last line from an interruption
                continue
            if r.get("finish_reason") != "error":
                done.add(row_key(r))
        print(f"resume: {len(done)} completed rows in {args.out}", flush=True)
    fh = args.out.open("w" if args.fresh else "a")
    lock = threading.Lock()

    def write(it, th, sp, i, res):
        raw, fin, ntok, *ex = res
        extra = ex[0] if ex else {}
        think, ans, closed = split_think(
            raw.replace("<end_of_turn>", "").replace("<|im_end|>", "").replace("<turn|>", "")
        )
        if extra.get("reasoning") and think is None:
            think = extra["reasoning"]
        row = meta | it | {"thinking": th, "sampling": sp, "sample": i, "raw": raw, "think": think,
               "answer": ans, "think_closed": closed, "finish_reason": fin, "n_tokens": ntok} | extra  # fmt: skip
        with lock:
            tot["cost"] += float((extra.get("usage") or {}).get("cost") or 0)
            tot["errors"] += fin == "error"
            fh.write(json.dumps(row) + "\n")
            fh.flush()

    for th in thinks:
        sk = sampling_key(family, th)
        if th and f"{family}_thinking" in cfg["sampling"]:
            sk = f"{family}_thinking"
        sp = cfg["sampling"][sk] | mcfg.get("sampling_override", {})  # e.g. a short context window
        t0 = time.time()
        skip = {(i, j) for i, it in enumerate(items) for j in range(n)
                if row_key(it | {"thinking": th, "sample": j}) in done}  # fmt: skip
        todo = len(items) * n - len(skip)
        if args.backend == "openrouter":
            run_openrouter(mcfg["openrouter"], [it["prompt"] for it in items], sp, n, args.seed, th,
                           skip, lambda i, j, r, th=th, sp=sp: write(items[i], th, sp, j, r))  # fmt: skip
        elif todo:
            texts = [render(tok, it["prompt"], th, family) for it in items]
            res = (run_vllm if args.backend == "vllm" else run_hf)(
                args.model, texts, sp, n, args.seed
            )
            for i, outs in enumerate(res):
                for j, r in enumerate(outs):
                    if (i, j) not in skip:
                        write(items[i], th, sp, j, r)
        print(f"thinking={th}: {todo} of {len(items) * n} requested in {time.time() - t0:.0f} s",
              flush=True)  # fmt: skip
    fh.close()
    if args.backend == "openrouter":
        print(f"OpenRouter cost ${tot['cost']:.4f}, errors {tot['errors']}", flush=True)
    print(f"=== GEN DONE {args.model} ===", flush=True)


if __name__ == "__main__":
    main()
