# Zhou et al. (2019) Figures 2–4 replicated, with additions (exploratory, 2026-11-23)

`scripts/paperA_zhou_figs.py`. Figure 1 is `results/zhou_figures/zhou_fig1_redrawn.png` (`scripts/zhou_figures.py`).
Alignment (De-Align, Hybrid) is the Procrustes rebuild (`scripts/zhou_hybrid.py`; not released, B9).

- `zhou_fig2_redrawn.png` — A: their five occupations on the semantic direction (their directions, origin as
  neutral). B: all 58 pairs with the epicene zero point (−0.031): feminine forms further than masculine in 52/58
  pairs from the origin, 46/58 from the epicene zero. C: full space, English word closer to the masculine form
  in 54/58 pairs.
- `zhou_fig3_redrawn.png` — after Hybrid Ori. B: every pair symmetric about the origin to 1e-15 — by
  construction of Shift, not a finding; not symmetric about the epicene zero. C: English word still closer to
  the masculine form in 54/58 (same as before; De-Align alone also 54/58).
- `zhou_fig4_redrawn.png` — A: their ten labelled occupations, Spearman 0.45 over 58 (|s(m)| − |s(f)| form).
  B: all 58; clean list (54) ρ 0.42. "adviser" appears twice: two Spanish pairs share the gloss.

Their Figs. 2–3 include doctor/doctora, which is not among the 58 scored pairs (B12); it is drawn in panel A
only (and shifted for Fig. 3, as in theirs). Hard-debiased English in Fig. 2/3 A is projected on the Spanish
semantic direction carried into the aligned frame, so it is not exactly 0.
