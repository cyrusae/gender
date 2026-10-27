"""Embedding-based content score (2026-10-26; set frozen in docs/decisions.md, tentative until
prereg). Model-based measure of OUTPUTS: never the primary gender outcome.

Score per paragraph and embedder = mean cosine to FEMININE anchors - mean cosine to MASCULINE
anchors (positive = reads feminine). Anchors are fixed on principle, no model involved: the 50
lowest- and 50 highest-rated single-sense Glasgow gender words (1 = very feminine, 7 = very
masculine). A shuffle null re-splits the 100 anchors at random (same sizes) to show how large a
score difference arises from an arbitrary split.

Masking before embedding (so the score reflects the description, not the pronoun choice):
he/she/they and neopronouns -> "the person" (it/its only when the coder classes the character
as "it"; otherwise "it" refers to objects), his/her (possessive) -> "the person's", him/her (object) -> "the
person" (they-forms are left as written: replacing he/she with "they" broke verb agreement); gendered person
nouns -> "person"; the character's name and other capitalised words after the first -> "the
character"; the study noun and plural -> "thing"; a trailing Name line is removed.

Embedders: bge-m3 (dense), multilingual-e5-large-instruct, LaBSE, Qwen3-Embedding-0.6B,
EmbeddingGemma-300m (sentence-transformers), plus static aligned fastText (mean word vector).
All fp32 on one device (CPU). Instruction prefix where the model expects one: fixed text
"Represent this character description", never tuned on results.
Run: uv run --with sentence-transformers python character/content_embed.py FILE.jsonl --out X.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from code_outputs import code, split_name
from content_glasgow import gend_table

INSTR = "Represent this character description"
EMBEDDERS = {  # name: (hf id, query prefix for paragraphs, prefix for anchor words)
    "bge-m3": ("BAAI/bge-m3", "", ""),
    "e5-instruct": ("intfloat/multilingual-e5-large-instruct", f"Instruct: {INSTR}\nQuery: ", ""),
    "labse": ("sentence-transformers/LaBSE", "", ""),
    "qwen3-emb-0.6b": ("Qwen/Qwen3-Embedding-0.6B", f"Instruct: {INSTR}\nQuery:", ""),
    "embeddinggemma": ("google/embeddinggemma-300m", "task: clustering | query: ", ""),
}
FASTTEXT = Path("/Volumes/Lexar/gbleed/shared/fasttext/wiki.en.align.vec")
PERSON = r"\b(woman|women|man|men|girl|girls|boy|boys|lady|ladies|gentleman|gentlemen|mother|father|wife|husband|queen|king|sister|brother|daughter|son|heroine|hero)\b"
K = 50
FUNC = "a|an|the|some|this|that|these|those|no|every|and|or|but|to|with|as|in|on|at|for|from|by|of|into|like|than|that|when|while|if|again|too|away|up|down|out|back|so|because"


def is_person_or_body(w: str) -> bool:
    """WordNet: any noun sense in noun.person or noun.body (woman, beard, womb, ...)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gbleed.lexicon import _wordnet

    return any(s.lexname() in ("noun.person", "noun.body", "noun.animal")
               for s in _wordnet().synsets(w, "n"))  # fmt: skip


def study_nouns() -> set[str]:
    """Every noun the study prompts about (main pool, pilot, reserved): never an anchor."""
    import pandas as pd

    st = Path(__file__).resolve().parent / "stimuli"
    rs = Path(__file__).resolve().parent / "results/confound/pool_191_DRAFT.csv"
    out = set(pd.read_csv(rs).en)
    out |= set(pd.read_csv(st / "reserved_nouns.csv", comment="#").noun)
    return out | {w + "s" for w in out}


def anchors(kind: str = "properties") -> tuple[list[str], list[str]]:
    """'properties' (primary, PI 2026-10-26): person, body and animal words excluded (WordNet), so
    the anchors are gender-stereotyped things and qualities. 'people' (secondary): all words.
    Study nouns are never anchors."""
    g = sorted(gend_table().items(), key=lambda kv: kv[1])
    nouns = study_nouns()
    g = [(w, v) for w, v in g if re.fullmatch(r"[a-z]+", w) and w not in nouns]
    if kind == "properties":
        g = [(w, v) for w, v in g if not is_person_or_body(w)]
    return [w for w, _ in g[:K]], [w for w, _ in g[-K:]]


def mask(
    answer: str, noun: str, name: str | None, char_is_it: bool = False, mask_they: bool = True
) -> str:
    para, nm = split_name(answer or "")
    name = name or nm or ""
    t = re.sub(PERSON, "person", para, flags=re.IGNORECASE)
    for p in {noun, noun + "s", noun + "es"} | set(name.split()):
        if p:
            t = re.sub(rf"\b{re.escape(p)}\b", "thing" if p.startswith(noun) else "the character",
                       t, flags=re.IGNORECASE)  # fmt: skip

    t = re.sub(r"\b(?:xemself|xirself|zirself|hirself|emself|faeself|verself|thonself)\b",
               "themselves", t, flags=re.IGNORECASE)  # fmt: skip
    t = re.sub(r"\b(?:xyr|xyrs|xirs|zirs|hirs|eir|eirs|faer|faers|thons)\b", "the person's", t,
               flags=re.IGNORECASE)  # fmt: skip
    # xir/zir/hir are object or possessive, like "her"
    t = re.sub(rf"\b(?:xir|zir|hir)(?=\s+(?!(?:{FUNC})\b)[a-z])", "the person's", t,
               flags=re.IGNORECASE)  # fmt: skip
    subj = (
        "he|she|they|xe|xem|xir|ze|zie|zir|hir|ey|thon"
        if mask_they
        else "he|she|xe|xem|xir|ze|zie|zir|hir|ey|thon"
    )
    t = re.sub(rf"\b(?:{subj})\b", "the person", t, flags=re.IGNORECASE)
    if mask_they:  # off only for the they-masking check (he/she paragraphs)
        t = re.sub(r"\b(?:themselves|themself)\b", "themselves", t, flags=re.IGNORECASE)
        t = re.sub(r"\b(?:their|theirs)\b", "the person's", t, flags=re.IGNORECASE)
        t = re.sub(r"\bthem\b", "the person", t, flags=re.IGNORECASE)
    if char_is_it:  # only when the coder classed the character as "it"
        t = re.sub(r"\bitself\b", "themselves", t, flags=re.IGNORECASE)
        t = re.sub(r"\bits\b", "the person's", t, flags=re.IGNORECASE)
        t = re.sub(r"\bit\b", "the person", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:himself|herself)\b", "themselves", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:his|hers)\b", "the person's", t, flags=re.IGNORECASE)
    # "her": possessive before a content word, else object
    t = re.sub(rf"\bher(?=\s+(?!(?:{FUNC})\b)[a-z])", "the person's", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:him|her)\b", "the person", t, flags=re.IGNORECASE)
    t = re.sub(
        r"(^|[.!?]\s+)(the (?:person|character))", lambda m: m.group(1) + m.group(2).capitalize(), t
    )
    # remaining capitalised non-initial words (invented names, places) -> neutral
    return re.sub(r"(?<=[a-z,;] )[A-Z][a-z]+", "the character", t)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def score_matrix(P: np.ndarray, A: np.ndarray, nf: int, rng, n_null: int = 200):
    """P paragraphs x d, A anchors (first nf feminine) x d, unit vectors."""
    S = P @ A.T
    obs = S[:, :nf].mean(1) - S[:, nf:].mean(1)
    null = []
    for _ in range(n_null):
        idx = rng.permutation(A.shape[0])
        null.append(S[:, idx[:nf]].mean(1) - S[:, idx[nf:]].mean(1))
    return obs, np.stack(null, 1)


def fasttext_vectors(words: set[str]) -> dict[str, np.ndarray]:
    out = {}
    with FASTTEXT.open(encoding="utf-8", errors="ignore") as fh:
        next(fh)
        for ln in fh:
            w, _, rest = ln.partition(" ")
            if w in words:
                out[w] = np.array(rest.split(), dtype=np.float32)
                if len(out) == len(words):
                    break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", help="comma-separated embedder names")
    ap.add_argument("--anchors", choices=["properties", "people"], default="properties")
    ap.add_argument("--keep-they", action="store_true",
                    help="check: score he/she paragraphs with plural they/them/their unmasked")  # fmt: skip
    args = ap.parse_args()
    import pandas as pd

    rows = [json.loads(ln) for f in args.files for ln in f.read_text().splitlines()]
    rows = [r for r in rows if r.get("finish_reason") != "error" and r.get("answer")]
    prons = [code(split_name(r["answer"])[0])["pron"] for r in rows]
    if args.keep_they:  # they-masking check: he/she paragraphs only, plural they left intact
        rows = [r for r, p in zip(rows, prons) if p in ("she", "he")]
        prons = [p for p in prons if p in ("she", "he")]
    texts = [mask(r["answer"], r["noun"], r.get("given_name"), p == "it", not args.keep_they)
             for r, p in zip(rows, prons)]  # fmt: skip
    fem, masc = anchors(args.anchors)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({k: [r.get(k) for r in rows] for k in
                       ("model", "name_condition", "noun", "modifier", "thinking", "sample",
                        "fixed_sample", "given_pron", "lang")})  # fmt: skip
    df["pron"] = prons
    names = args.only.split(",") if args.only else [*EMBEDDERS, "fasttext"]
    for nm in names:
        if nm == "fasttext":
            toks = [re.findall(r"[a-z]+", t.lower()) for t in texts]
            vec = fasttext_vectors(set(fem + masc) | {w for ts in toks for w in ts})
            P = _unit(np.stack([np.mean([vec[w] for w in ts if w in vec] or [np.zeros(300)], 0)
                                for ts in toks]))  # fmt: skip
            fem_ft, masc_ft = [w for w in fem if w in vec], [w for w in masc if w in vec]
            A = _unit(np.stack([vec[w] for w in fem_ft + masc_ft]))
            print(f"fasttext anchors present: {len(fem_ft)} f / {len(masc_ft)} m", flush=True)
        else:
            from sentence_transformers import SentenceTransformer

            hf, qp, ap_ = EMBEDDERS[nm]
            import torch

            # fp32 for every embedder (one precision); e5's config defaults to fp16, which
            # gave NaNs on CPU in the first check
            m = SentenceTransformer(hf, device="cpu", model_kwargs={"torch_dtype": torch.float32})
            P = m.encode([qp + t for t in texts], batch_size=16, normalize_embeddings=True)
            A = m.encode([ap_ + w for w in fem + masc], normalize_embeddings=True)
            del m
        nf = len(fem_ft) if nm == "fasttext" else len(fem)
        obs, null = score_matrix(P, A, nf, rng)
        df[f"emb_{nm}"] = obs.round(5)
        df[f"emb_{nm}_null_sd"] = null.std(1).round(5)
        print(f"{nm}: mean {obs.mean():+.4f}, sd {obs.std():.4f}", flush=True)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
