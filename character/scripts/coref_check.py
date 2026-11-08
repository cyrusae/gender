"""FastCoref agreement check on the regex pronoun coder (prereg v1 amendment 3, 2026-11-07).

Validation only, never the outcome. Random sample (seed 20261107) of 300 valid outputs per model ×
thinking cell from the main run. FastCoref (FCoref, default checkpoint) finds the largest
coreference cluster that contains a third-person personal pronoun; its pronoun class
(she/he/they/it/mixed) is compared with the regex class. Reports agreement %, Cohen's kappa and the
confusion table per cell. Usage:
  uv run --with fastcoref python character/scripts/coref_check.py RESULTS_CODED.csv RAW.jsonl --out X.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from code_outputs import split_name

CLS = {"she": "she", "her": "she", "hers": "she", "herself": "she", "he": "he", "him": "he",
       "his": "he", "himself": "he", "they": "they", "them": "they", "their": "they",
       "theirs": "they", "themselves": "they", "themself": "they", "it": "it", "its": "it",
       "itself": "it"}  # fmt: skip


def cluster_class(spans: list[str]) -> str | None:
    cs = {CLS[s.lower()] for s in spans if s.lower() in CLS}
    return None if not cs else (cs.pop() if len(cs) == 1 else "mixed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("coded", type=Path)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    from fastcoref import FCoref
    from sklearn.metrics import cohen_kappa_score

    d = pd.read_csv(a.coded, keep_default_na=False)
    d = d[d.valid.astype(str) == "True"]
    raw = {}
    for ln in a.raw.read_text().splitlines():
        r = json.loads(ln)
        raw[(r["noun"], r["modifier"], str(r.get("sample")), str(r.get("thinking")))] = r["answer"]
    rows = []
    for cell, g in d.groupby("thinking"):
        s = g.sample(min(a.n, len(g)), random_state=20261107)
        texts = [
            split_name(raw[(x.noun, x.modifier, str(x["sample"]), str(cell))])[0]
            for _, x in s.iterrows()
        ]
        preds = FCoref(device="cpu").predict(texts=texts)
        for (_, x), p in zip(s.iterrows(), preds):
            cl = [c for c in p.get_clusters() if any(w.lower() in CLS for w in c)]
            best = max(cl, key=len) if cl else []
            rows.append({"thinking": cell, "regex": x.pron, "coref": cluster_class(best) or "none"})
    o = pd.DataFrame(rows)
    o.to_csv(a.out, index=False)
    for cell, g in o.groupby("thinking"):
        print(
            cell,
            f"agree {(g.regex == g.coref).mean():.1%}",
            f"kappa {cohen_kappa_score(g.regex, g.coref):.2f}",
        )
        print(pd.crosstab(g.regex, g.coref).to_string())


if __name__ == "__main__":
    main()
