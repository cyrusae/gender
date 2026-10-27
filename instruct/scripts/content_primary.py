"""Assemble primary content scores for pilot0_v9 (decision 2026-10-27) and apply the they-masking
check criterion. Primary rule = character-only masking: he/she paragraphs take the --keep-they
scores (plural they intact); all other paragraphs keep the mask-all scores (identical under both
rules for them). Sensitivity = mask-all for everyone.
Criterion: |mean shift| (he/she paragraphs, kept - masked) < 0.25 x the scorer's shuffle-null SD.
Also adds the model-free Glasgow word average. Writes results/pilot0_v9/content_primary.csv."""

import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code_outputs import split_name
from content_glasgow import glasgow_score

R = Path(__file__).resolve().parents[1] / "results/pilot0_v9"
K = ["model", "name_condition", "noun", "modifier", "thinking", "sample", "fixed_sample"]
EMB = ["bge-m3", "e5-instruct", "labse", "qwen3-emb-0.6b", "embeddinggemma", "fasttext"]
out, crit = [], []
for f in sorted(glob.glob(str(R / "*_emb.csv"))):
    tag = Path(f).name.replace("_emb.csv", "")
    a = pd.read_csv(f, keep_default_na=False)
    b = pd.read_csv(f.replace("_emb.csv", "_emb_kept.csv"), keep_default_na=False)
    for d in (a, b):
        d["fixed_sample"] = d["fixed_sample"].astype(str)
        d["sample"] = d["sample"].astype(str)
    m = a.merge(b[K + [f"emb_{e}" for e in EMB]], on=K, how="left", suffixes=("", "_kept"))
    for e in EMB:
        c = f"emb_{e}"
        hs = m.pron.isin(["she", "he"]) & m[c + "_kept"].ne("") & m[c + "_kept"].notna()
        sh = pd.to_numeric(m.loc[hs, c + "_kept"]) - pd.to_numeric(m.loc[hs, c])
        nsd = pd.to_numeric(m[c + "_null_sd"]).mean()
        crit.append({"file": tag, "embedder": e, "n_hs": int(hs.sum()), "mean_shift": sh.mean(),
                     "null_sd": nsd, "ratio": abs(sh.mean()) / nsd, "pass": abs(sh.mean()) < 0.25 * nsd})  # fmt: skip
        m[c + "_mask_all"] = pd.to_numeric(m[c])
        m[c] = pd.to_numeric(m[c]).where(~hs, pd.to_numeric(m[c + "_kept"]))
    m["file"] = tag
    out.append(m)
d = pd.concat(out, ignore_index=True)
raw = {}


def key(*v) -> tuple:
    """Normalise None/''/NaN and 0/'0'/'0.0' so jsonl rows and csv rows match."""
    out = []
    for x in v:
        x = "" if x is None or str(x) in ("", "nan", "None") else str(x)
        out.append(x.removesuffix(".0"))
    return tuple(out)


for f in glob.glob(str(R / "*.jsonl")):
    for ln in Path(f).read_text().splitlines():
        r = json.loads(ln)
        raw[key(r["model"], r.get("name_condition"), r["noun"], r["modifier"], r.get("sample"),
                r.get("fixed_sample"))] = r  # fmt: skip
gl = []
for _, x in d.iterrows():
    r = raw.get(
        (x.model, str(x.name_condition), x.noun, x.modifier, str(x["sample"]), str(x.fixed_sample))
    )
    gl.append(
        glasgow_score(split_name(r["answer"])[0], r["noun"], r.get("given_name") or "") if r else {}
    )
d = pd.concat([d, pd.DataFrame(gl)], axis=1)
d.to_csv(R / "content_primary.csv", index=False)
c = pd.DataFrame(crit)
c.to_csv(R / "they_mask_check.csv", index=False)
print(c.groupby("embedder").agg(mean_ratio=("ratio", "mean"), max_ratio=("ratio", "max"),
                                all_pass=("pass", "all")).round(3).to_string())  # fmt: skip
