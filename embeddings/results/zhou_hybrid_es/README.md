# Zhou et al. (2019) Spanish De-Align and Hybrid Ori, rebuilt (2026-11-08)

`embeddings/scripts/zhou_hybrid.py`. English MUSE (200k) hard-debiased with Bolukbasi et al.'s
procedure (debiaswe 6bb03ee data: 10 definitional pairs, 1,441 gender-specific words, 52 equalize
pairs → 43 present with case variants), then Spanish MUSE mapped onto it by orthogonal Procrustes
on 5,000 es-en dictionary pairs (MUSE supervised alignment's standard method; the paper doesn't
say which alignment it used). Alignment check: mean dictionary-pair cosine 0.633 → 0.662.

| | Table 1 | rebuilt |
|---|---|---|
| Original MWEAT–Diff | 3.6918 (p 0.000) | 3.6918 (p 0.000) |
| De-Align | **3.5748** (p 0.001) | 3.6918 (p 0.000) |
| Shift Ori | 0.3090 (p 0.113) | 0.3087 (p 0.757) |
| Hybrid Ori | 0.3090 (p 0.733) | 0.3087 (p **0.012**) |

**1. Hybrid Ori and Shift Ori are the same number by construction.** A rotation preserves every
dot product among Spanish words, and Table 1's statistic, the PCA direction, the SVC direction and
the shift use only Spanish words. So Hybrid Ori must equal Shift Ori (identical here to 1e-10),
which is why Table 1 prints 0.3090 twice. For the same reason De-Align should equal Original;
Table 1's 3.5748 means their alignment wasn't a pure rotation (e.g. MUSE refinement with centring,
or re-normalisation), which we can't recover. On a Spanish-only measure De-Align and Hybrid add
nothing; they can only matter cross-lingually (English words in the shared space).

**2. The post-mitigation p-values are tie artefacts.** MWEAT–Diff = ‖Σ_m| − |Σ_f‖. When a random
re-partition gives the two sums opposite signs, the statistic equals |Σ_m + Σ_f|: the total over
all 116 occupation words, the same for every partition. After Shift Ori **757 of 1,000
permutations reproduce the observed statistic exactly** (original: 1 of 1,000). Counting
"permuted > observed" then depends on floating-point noise: 0.757 in our Spanish space, 0.012 in a
rotated copy of the same space, 0.113 and 0.733 in Table 1 for the same 0.3090. The "not
significant after mitigation" results in Table 1 are not tests. Evidence: tie count from the
permutation null with `random.seed(520)`, 1,000 draws (as their `_weat_pval`).
