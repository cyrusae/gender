# Centring sensitivity: C1 and Gender ValNorm with the corrected PCA centring (2026-11-19)

Exploratory sensitivity check (upstream Z3/Z33). Zhou et al.'s released code centres definitional pairs at
`a + b/2`; the paper describes the ordinary midpoint `(a + b)/2`. Our rebuilt Shift/Hybrid spaces use the
released code's centring (pre-registered, it reproduces their tables). This reruns the studies that use those
spaces with `ZHOU_CENTRE=midpoint`, in a sandbox copy so the frozen results aren't overwritten
(`scripts/c1_run.py`, `gvalnorm_run.py`, `gvalnorm_orth.py`, corrected attribute lists = the default).
Comparison baseline: `results/attrfix_sensitivity/` (same corrected lists, released-code centring).
The crime probe and the zero point's primary verdicts use only the original space and are unaffected; the
word-pair test is in `results/wordpair_cues_midpoint/`.

| Study | Released-code centring | Midpoint centring |
|---|---|---|
| C1 ES (8 arm × attribute cells) | all "survives"; ES Shift keeps 84.2% | all "survives"; 84.3% (largest ratio change 0.007) |
| C1 FR | all "survives"; Shift 1.09× | all "survives"; 1.08× (largest change 0.08, English-attribute Hybrid 2.87 → 2.95) |
| Gender ValNorm, main (ES, FR) | — | no verdict changes; largest β_H change 0.035 (ES), FR β_G up to 0.18 |
| Gender ValNorm, orthogonal arms ES | Hybrid, English attributes, off-axis β_H −0.035 [−0.117, 0.047] "removed" | −0.040 [−0.121, 0.042] "reduced" (crosses the 25%-of-original threshold; interval still includes 0) |
| Gender ValNorm, orthogonal arms FR | — | no verdict changes |

**Reading.** C1 and Gender ValNorm don't depend on the centring bug; the one verdict label that changes is a
threshold crossing on an estimate whose interval includes zero in both versions.
