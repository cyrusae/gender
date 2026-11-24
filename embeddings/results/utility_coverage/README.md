# Utility-test coverage (exploratory, 2026-11-23)

`scripts/utility_coverage.py` (committed before running, f4545e9).

**Translation.** MUSE test queries (1,500 per direction) whose source word or a gold target is one of the
shifted occupation forms: EN→ES 12, ES→EN 4, EN→FR 6, FR→EN 3 (0.2–0.8%). Shift moves about 116 of 50,000
Spanish words (0.2%), so the translation test can barely register it.

**Spanish word similarity by vocabulary size (original, unmitigated space).**

| vocab | pairs covered | Spearman | Pearson |
|---|---|---|---|
| 50k | 317 | **0.7363** | 0.7077 |
| 75k | 340 | 0.7378 | 0.7112 |
| 100k | 351 | 0.7376 | 0.7125 |
| 150k | 364 | 0.7411 | 0.7171 |
| 200k | 369 | 0.7399 | 0.7172 |

Zhou et al. Table 1: Original 0.7392, Shift Ori 0.7363, Hybrid Ori 0.7358. Their Shift Ori value equals the
*unmitigated* space at 50k exactly (our Shift Ori at 50k is 0.7367; Shift touches 15 pairs). Their Original
value matches no vocabulary size exactly (closest 0.7399 at 175–200k). Reading: the Shift Ori row is
consistent with an unmitigated 50k score, so the reported drop is plausibly vocabulary size, as for
translation (Z18); not established, since the Original value is not recovered.
