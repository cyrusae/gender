# Reproduction: Omrani Sabbaghi & Caliskan (2022) notebook, Spanish (2026-11-08)

`main.ipynb` from `external/GG_Disentangling` (commit 725d813) executed unchanged with `lang = "es"`,
fastText Common Crawl `cc.es.300.vec` (store, checksummed). Executed copy with all outputs: `main_es_executed.ipynb` (the xlsx/txt files it
wrote into the clone's `Results/` were discarded when the upstream originals were restored). Environment: Python 3.12, current numpy/scipy/
scikit-learn/pandas (uv, no pins; the repo has no requirements file). Mac CPU, ~20 min.

| output | upstream saved | rerun |
|---|---|---|
| nouns in vocabulary (m / f) | 5,381 / 4,837 | same |
| SVC accuracy train / held-out | 0.9923 / 0.9850 | same |
| Gonen same- / different-gender similarity | 0.4958 / 0.3701 | same |
| SC-GG-WEAT masculine nouns, mean d before → after | 0.839 → −0.244 | 0.839 → −0.247 |
| masculine nouns weakened | 998/1000 | 998/1000 |
| SC-GG-WEAT feminine nouns, mean d before → after | −0.902 → −0.294 | −0.902 → −0.296 |
| feminine nouns weakened | 969/1000 | 968/1000 |
| ValNorm r before / after removal | 0.807 / 0.818 | same |

Everything before the removal step matches exactly. After removal the numbers differ in the third
decimal: the iterative removal refits LinearSVC (no random_state) each round and the WEAT nulls are
permutation-based, so small run-to-run variation is expected. Reproduces.
