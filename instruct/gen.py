"""Generate character paragraphs from instruct models.

  python instruct/gen.py --config instruct/configs/pilot0_v1.json \
      --nouns instruct/stimuli/pilot0_nouns_v1.csv --model Qwen/Qwen3-8B --backend vllm \
      --out instruct/results/pilot0/Qwen__Qwen3-8B.jsonl
Smoke (Mac): --backend hf --model Qwen/Qwen3-0.6B --n 2 --only-nouns pot --only-mods plain,best

One jsonl row per completion with full provenance. Reasoning ('<think>') is split from the answer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
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


def build_prompts(cfg: dict, nouns: list[str], mods: list[str] | None) -> list[dict]:
    out = []
    for noun in nouns:
        for mk, mt in cfg["modifiers"].items():
            if mods and mk not in mods:
                continue
            mod_noun = mt.format(a=article(noun), noun=noun)
            out.append({"noun": noun, "modifier": mk, "mod_noun": mod_noun,
                        "prompt": cfg["template"].replace("{MOD_NOUN}", mod_noun)})  # fmt: skip
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
    kw = {"enable_thinking": thinking} if family == "qwen3" else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def run_vllm(model, texts, sp, n, seed):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="bfloat16", seed=seed, max_model_len=8192)
    params = SamplingParams(n=n, temperature=sp["temperature"], top_p=sp["top_p"],
                            top_k=sp["top_k"], min_p=sp["min_p"], max_tokens=sp["max_tokens"],
                            seed=seed, skip_special_tokens=False)  # fmt: skip
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--nouns", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", help="override (qwen3|gemma3); else from config")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, help="samples per cell (default: config)")
    ap.add_argument("--thinking", choices=["config", "on", "off"], default="config")
    ap.add_argument("--only-nouns")
    ap.add_argument("--only-mods")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    cfg = json.loads(args.config.read_text())
    mcfg = cfg["models"].get(args.model, {})
    family = args.family or mcfg["family"]
    thinks = {"on": [True], "off": [False]}.get(args.thinking) or mcfg.get("thinking", [False])
    nouns = load_nouns(args.nouns)
    if args.only_nouns:
        nouns = [x for x in nouns if x in args.only_nouns.split(",")]
    check_reserved(nouns, cfg["name"])
    items = build_prompts(cfg, nouns, args.only_mods.split(",") if args.only_mods else None)
    n = args.n or cfg["samples_per_cell"]
    tok = AutoTokenizer.from_pretrained(args.model)
    meta = {"model": args.model, "backend": args.backend, "config": cfg["name"],
            "config_sha": hashlib.sha256(args.config.read_bytes()).hexdigest()[:12],
            "git_commit": git_commit(), "seed": args.seed}  # fmt: skip
    if args.backend == "vllm":
        import vllm

        meta["backend_version"] = vllm.__version__
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for th in thinks:
            sp = cfg["sampling"][sampling_key(family, th)]
            texts = [render(tok, it["prompt"], th, family) for it in items]
            t0 = time.time()
            res = (run_vllm if args.backend == "vllm" else run_hf)(
                args.model, texts, sp, n, args.seed
            )
            print(
                f"thinking={th}: {len(texts)} prompts x {n} in {time.time() - t0:.0f} s", flush=True
            )
            for it, outs in zip(items, res):
                for i, (raw, fin, ntok) in enumerate(outs):
                    think, ans, closed = split_think(
                        raw.replace("<end_of_turn>", "").replace("<|im_end|>", "")
                    )
                    fh.write(json.dumps(meta | it | {"thinking": th, "sampling": sp, "sample": i,
                             "raw": raw, "think": think, "answer": ans, "think_closed": closed,
                             "finish_reason": fin, "n_tokens": ntok}) + "\n")  # fmt: skip
    print(f"=== GEN DONE {args.model} ===", flush=True)


if __name__ == "__main__":
    main()
