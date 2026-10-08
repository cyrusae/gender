# Grammatical gender in language models: methods summary (one page)

*[Name], 2026-10. Research design mine; implementation with an AI coding assistant (Claude Code).
Details: `docs/explainers/`, `docs/screening/`, `docs/decisions.md`.*

**Question.** In German *die Brücke* ("bridge") is feminine; in Spanish *el puente* is masculine.
For objects this is pure grammar. Does a language model's representation of grammatical gender,
learned from inanimate nouns, overlap with its representation of *social* gender? (An AI version
of the contested Boroditsky "bridge" effect in humans.)

**Setup.** Qwen3 base models (0.6B–14B parameters, plus a 30B mixture-of-experts model). Each noun
is fed bare; we read its hidden state x ∈ ℝ^d (d ≈ 1,000–5,000) at every layer. Nouns are
filtered so spelling can't give gender away (Spanish *-a/-o*, German suffixes, compounds), and the
model must demonstrably know each noun's gender (article-choice probabilities). Gold labels come
from Wiktionary, never from a model.

**Directions.** A gender direction is either a difference of class means or an L2-regularised
logistic-regression probe. Nuisance variables (spelling cell, frequency, token count, concreteness,
loan status) are removed first by **residualising**: X ↦ (I − Z Z⁺) X, the Frisch–Waugh–Lovell
idea, so the direction can only use within-cell variation. With n ≈ 100 nouns and d ≈ 5,000, the
probe is fitted in the training rows' span (Z = U S Vᵀ ⇒ fit on U S, map back with V): same
optimum, n-dimensional problem.

**Tests.** A direction trained on regular nouns is scored on held-out nouns where spelling and
gender disagree (Spanish *el día* "day", masculine despite *-a*), noun/verb homographs, and German
compounds whose first part and head differ in gender. Effects are ROC AUCs with bootstrap
intervals (training nouns resampled within strata and directions refitted).

**The estimator problem.** Is the masculine vector shorter than the feminine one (masculine as
the grammatical default)? The plug-in estimate of a squared length is biased:
E‖v̂‖² = ‖v‖² + tr(Σ)(1/n_g + 1/n_ref), and more so for the smaller group (27 feminine vs 64
masculine nouns). Two vectors sharing a reference mean also have upward-biased cosines. The fix is
**cross-fitting**: split the nouns into halves A and B within strata; E[v̂_A · v̂_B] = ‖v‖², since
the halves' noise is independent. Averaged over 200 splits, computed exactly through the Gram
matrix (every v̂ is P X, so all dot products are P_A (X Xᵀ) P_Bᵀ). Calibrated against a
**within-cell label-shuffle null** (no real class difference, same group sizes).

**Result of the fix.** The plug-in estimator "confirmed" the hypothesis in 25/27 and 31/35 layers
(two models). The split-half estimator supported it in **0 layers**: the apparent effect was
noise bias. A second subtlety: bootstrap resamples repeat items, and copies split across halves
share noise. Copies must stay in one half (found from an interval that didn't contain its own
point estimate).

**Findings so far.**
- **Spanish:** with spelling controlled, the direction tracks grammatical gender, not the ending
  (stratified probe reads gender in 22/27, 26/35 layers exploratory; 28/35 in the confirmatory
  8B run).
- **German:** a direction trained on single-root nouns sorts unseen compounds by their *head*'s
  gender, even when the first part has the other gender (all layers, AUC ≈ 0.8).
- **Markedness:** not supported, once the noise bias is removed.
- Both gender results hold at the noun's own last token; at a following neutral token they mostly
  vanish (a pre-registered robustness check).

**Next.**
- **Cross-language sharing:** train on Spanish, test on German; flipped pairs (*Brücke*/*puente*):
  a shared grammatical direction should follow each language's gender.
- **Steering:** push a noun along the gender direction and measure shifts in English adjectives
  rated for gender association by humans (Glasgow Norms). The shift is regressed on the gender
  rating with valence, size and frequency as covariates, because gender and valence correlate
  (r = −0.47).
- **Controls:** random directions, a number (singular/plural) direction, a social-gender
  direction, and model-damage (KL) limits.
- **Multiple comparisons:** Holm correction across co-primary tests.
- **Practice:** everything is pre-registered before data; deviations are logged.
