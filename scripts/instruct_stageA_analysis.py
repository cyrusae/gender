"""Instruct replication stage A analyses (docs/design/instruct-replication-prereg.md), one job.

    uv run python scripts/instruct_stageA_analysis.py MODEL FORMAT JOB
      FORMAT: raw | A-en | A-something | A-lang | U
      JOB:    outlier | phase3-last | phase3-after | phase4-last | phase4-after | t2 | c

Activations: raw from GBLEED_ACT_DIR_RAW (default the stage A extract), chat formats from
<root>/activations_chat/<FORMAT>. Outputs: results/instruct_stageA/<FORMAT>/<job>/...
Phase 3/4 verdicts run unfixed, as for Gemma base; T2 cosine values and the frequency cosine (c)
run with the outlier-dimension fix and the validity gate (gemma-slim amendment).
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("STAGEA_ROOT", "/Volumes/Lexar/gbleed/instruct-stageA/extract"))


def main(model: str, fmt: str, job: str) -> None:
    act = ROOT / ("activations" if fmt == "raw" else f"activations_chat/{fmt}")
    if fmt == "base":  # the base sibling, for like-for-like intervals
        act = Path(os.environ.get("BASE_ACT", "/Volumes/Lexar/gbleed/gemma-pod/extract/activations"))
    os.environ["GBLEED_ACT_DIR"] = str(act)
    os.environ.setdefault("GBLEED_STORE", "local")
    out = Path("results/instruct_stageA") / fmt
    out.mkdir(parents=True, exist_ok=True)
    from gbleed import activations as acts
    from gbleed import outlier_dims as od

    if job == "outlier":
        import numpy as np

        X, _ = acts.load(model, od.REFERENCE)
        share = []
        for k in range(1, X.shape[1] - 1):
            v = X[:, k].astype(np.float64).var(0)
            share.append(float(np.sort(v)[-1] / v.sum()))
        dims = {int(k): v.tolist() for k, v in od.outlier_dims(model).items()}
        (out / "outlier.json").write_text(json.dumps(
            {"model": model, "top_share_by_layer": share, "zeroed": dims}, indent=1))
        print(fmt, "max top-dim share", round(max(share), 3), "layers with a zeroed dim", len(dims))
        return
    kind, _, pos = job.partition("-")
    if kind == "phase3":
        from gbleed import phase3

        phase3.analyze(model, pos, out_root=str(out / f"phase3_{pos}"))
    elif kind == "phase4":
        from gbleed import phase4

        phase4.analyze(model, pos, out_root=str(out / f"phase4_{pos}"))
    elif job == "t2":
        from gbleed import phase4

        with od.dropped(model):
            phase4.analyze(model, "last", out_root=str(out / "phase4_fixed"))
    elif job == "c":
        from gbleed import confidence_ctl as cc

        with od.dropped(model):
            r = cc.run_c(model)
        r["valid"] = r.cos_gender_zipf.abs().le(1) & r.cos_gender_zipf.notna()
        r["overlap"] = r.overlap & r.valid
        r.to_csv(out / "c_frequency_cosine.csv", index=False)
    elif job == "cboot":  # amendment 1: bootstrap intervals, outlier fix applied
        from gbleed import cosine_boot as cb

        with od.dropped(model):
            cb.run(model).to_csv(out / "c_boot.csv", index=False)
    else:
        raise SystemExit(f"unknown job {job}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
