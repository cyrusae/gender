# Gender ValNorm, English pilot (2026-11-08)

`embeddings/scripts/gvalnorm_en.py`. Embedding gender lean of each Glasgow word (male-positive)
correlated with Glasgow GEND (1 feminine – 7 masculine; homonym senses averaged → 4,675 words,
attribute words excluded). Attributes as Valentini et al. 2023 / Lewis & Lupyan 2020.

| embeddings | words | mean-cos difference r | SC-WEAT effect size r | published reference |
|---|---|---|---|---|
| MUSE fastText Wikipedia (200k) | 4,641 | **0.584** (ρ 0.478) | 0.533 | Lewis & Lupyan, fastText Wikipedia: r = 0.59 |
| fastText Common Crawl (2M) | 4,674 | **0.639** (ρ 0.539) | 0.575 | (subtitles: 0.63; Valentini 0.44–0.58) |

The Wikipedia number reproduces Lewis & Lupyan's 0.59; the pipeline is valid. The SC-WEAT
normalisation (ValNorm's score) correlates a little lower than the plain mean difference.
