"""German rarity-lean held-out test (docs/design/rarity-lean-heldout-prereg.md).

Stage 1 (this file so far): freeze the held-out simplex list H. Extraction and the
within-gender rank test come after the list hash is recorded in the prereg.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .frequency_axis import complexity_rule
from .models import model_slug
from .phase2_stimuli import _clean
from .phase3_known import FINAL, KNOWN_MODELS
from .phase3_stimuli import POOL

HELDOUT = "data/stimuli/rarity_heldout_v1.csv"


def build(known_root: str = "results/phase3_known", out: str = HELDOUT) -> pd.DataFrame:
    """Known (Phase 3 filter), clean, unused, non-complex simplex nouns; m/f/n only."""
    pool = pd.read_csv(POOL, keep_default_na=False)
    pool["en_overlap"] = pool.en_overlap.astype(str).eq("True")
    pool["zipf"] = pd.to_numeric(pool.zipf, errors="coerce")
    known = None
    for m in KNOWN_MODELS:
        s = pd.read_csv(Path(known_root) / model_slug(m) / "scores.csv", keep_default_na=False)
        k = set(s[s.status == "known"].lemma)
        known = k if known is None else known & k
    used = set(pd.read_csv(FINAL, keep_default_na=False).lemma)
    h = _clean(pool[(pool.set == "simplex") & pool.lemma.isin(known)])
    h = h[~h.lemma.isin(used) & h.gender.isin(["m", "f", "n"])]
    why = complexity_rule()
    h = h[h.lemma.map(why) == ""]
    h = h.sort_values(["gender", "lemma"]).reset_index(drop=True)
    h.to_csv(out, index=False)
    digest = hashlib.sha256(Path(out).read_bytes()).hexdigest()
    print(h.gender.value_counts().to_dict(), "-ie f:", int(((h.gender == "f") & (h.ending == "ie")).sum()))
    print(f"{out} sha256 {digest}")
    return h
