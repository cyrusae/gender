# Methods and results: first draft (2026-10-22)

*Status: draft for the PI. Framing, introduction and discussion wait on the literature review.
Assembled from `docs/methods-summary.md`, `docs/explainers/00–07`, the preregistrations in
`docs/design/` and `docs/decisions.md`. Numbers marked **[check]** were carried over from
summaries and must be re-read from `results/` before anything leaves the repo.*

---

## 2. Methods

### 2.1 Ground rules

- **Pre-registration.** Every confirmatory analysis was written down and committed before its
  data were produced (`docs/design/*-prereg.md`; deviations logged in `docs/decisions.md`).
  Exploratory analyses are labelled as such wherever they appear.
- **Gold labels** come from Wiktionary (via kaikki.org dumps), never from a language model.
- **Drop, don't review.** Items failing an automatic filter are dropped; nobody hand-edits a
  list to rescue items. Lists that have produced results are frozen; flaws found later are
  logged and tested, not patched (e.g. §2.8, complex German training nouns).
- **Held-out sets** are fixed before the analyses that use them and asserted in code.
- **Same chip** for any comparison across runs.

### 2.2 Models

Qwen3 base models, 0.6B–14B dense plus Qwen3-30B-A3B (mixture of experts); EuroLLM-1.7B as a
second family for the behavioural tests. Base models only (instruct-tuned models are a planned
follow-up). Confirmatory runs in bf16 on a single A100/L40S; small models also on an M-series Mac.

### 2.3 Stimuli

- **Lexicons** built from Wiktionary nouns with a single gender, frequency from `wordfreq`
  (Zipf), concreteness from published norms (English-gloss norms where no native norms exist).
- **Spelling control by design.** Phase 1 showed that erasing spelling information with LEACE
  does not generalise (nonce endings stay readable, accuracy ≈ 0.99–1.00), so spelling is
  controlled by stratifying nouns into ending × loan-status cells and comparing genders
  *within* cells.
- **Test sets:** Spanish exceptions (*el día*), noun/verb homographs, German compounds whose
  first part and head differ in gender, Spanish–German flipped pairs (*die Brücke* / *el
  puente*; ≈10% gloss-error caveat), Russian soft-sign nouns, English person words and names.
- **Phase 5 readouts** use English adjectives rated for gender association in the Glasgow Norms
  (and Spanish invariant adjectives); gender ratings correlate with valence (r ≈ −0.47), so
  valence, arousal, size and frequency are covariates throughout.

### 2.4 Extraction

Each noun is fed bare after a fixed prefix (`<|endoftext|>\n `, which keeps the first content
token from becoming an attention sink). Hidden states are read at every layer either at the
noun's last token (LAST) or at a following neutral token (AFTER). Which position is primary was
decided by a **gender-blind** pre-registered check, before any gender result was seen.

### 2.5 Directions and the estimator

- A gender direction is a difference of class means or an L2 logistic probe, fitted after
  residualising nuisance variables (cell dummies, Zipf, token count, concreteness):
  X ↦ (I − Z Z⁺) X, so only within-cell variation is used.
- **Split-half (cross-fitted) estimator.** Plug-in squared lengths and cosines are biased
  upwards (E‖v̂‖² = ‖v‖² + tr Σ (1/n_g + 1/n_ref)), worst for the smaller class. We split the
  nouns into halves within strata and use E[v̂_A · v̂_B] = ‖v‖², averaged over 200 splits and
  computed exactly through the Gram matrix. Inference uses a **within-cell label-shuffle null**
  and a bootstrap in which copies of a resampled item stay in one half.
- This matters: the plug-in estimator supported "masculine is the shorter, default vector" in
  25/27 and 31/35 layers; the split-half estimator supported it in 0.

### 2.6 Behavioural gates

Phase 0 "known" checks (balanced article-choice rate; cross-size comparisons only on nouns every
model knows), frame checks on two families, and in Phase 5 an agreement-flip gate (a dose α*
must flip article agreement beyond a random-direction limit) plus readout frame checks (C1: the
probe must read the intended referent type with a 1-nat margin).

### 2.7 Steering and readouts (Phase 5, P29–P31)

- **Steering:** push the noun's state along the grammatical gender direction (dose in units of
  the state norm) and measure the shift in English (R1) or Spanish (R1-ES/R2/R3) adjective
  log-probabilities, regressed on the adjectives' gender ratings with the covariates above.
- **Controls:** 100 random directions (rank test), a number direction, a social-gender
  direction, damage (KL) matching.
- **Unsteered article tests (P30 Spanish, P31 German):** nonce nouns given a feminine vs
  masculine article in a sentence that makes them a *thing* (object reading) vs neutral vs a
  *person*; S = slope of the die−der (la−el) shift on gender rating.
- **Verdict rules:** a majority of the pre-registered layers; Holm across co-primary tests.

### 2.8 Controls added late

- **Confidence control** (prereg `confidence-ctl-prereg.md`): (a) midpoint log-prob covariate,
  (b) Spearman rank check, (c) cross-fitted cosine between the f−m coefficient and the Zipf
  coefficient with a within-cell × gender permutation null; "overlap" = cross term beyond the
  null's 95th percentile **and** |cos| ≥ 0.3.
- **Frequency axis** (prereg `frequency-axis-prereg.md`): A flexible frequency control,
  B frequency axis projected out and Phase 4 rerun, C frequency-matched pairs,
  D complex-noun drop/covariate.
- **Register / age of acquisition** (prereg `register-axis-prereg.md`).
- **Stimulus flaw found late:** 12/64 masculine German training nouns are complex (numerals such
  as *Vierer*, compounds such as *Feuerlöscher*), 0/27 feminine; the simplex filter only checked
  whole-word *-er* agent nouns. Lists frozen; tested by frequency-axis D.

### 2.9 Table of experiments

*(To build from `docs/glossary.md`: phase/P-number, question, prereg status, verdict.)*

---

## 3. Results

### 3.1 Phase 0–1: models know the genders; spelling can't be erased

- Known rates (balanced, 257 nouns per language; `results/phase0/comparison.csv`) are high and
  rise with Qwen3 size: German 0.89 / 0.90 / 0.94 and Spanish 0.89 / 0.92 / 0.94 at 0.6B / 1.7B /
  4B; EuroLLM-1.7B 0.97 / 0.98. (Phase 0 was not run on 8B/14B.) The article quiz is diagnostic
  only: reordering the options changed 20% (DE) and 45% (ES) of 0.6B answers
  (`meta_order_agree` 0.80 / 0.55).
- A verb-trained LEACE eraser leaves nonce endings readable at all sizes (≈ 0.99–1.00); unseen
  verbs AUC 0.68–0.81. Hence spelling control by design.

### 3.2 Phases 2–3: a gender direction exists, at the noun's own token

| Model | Spanish LAST | Spanish AFTER |
|---|---|---|
| 0.6B | 24/27 | 0/27 |
| 1.7B | 22/27 | 2/27 |
| 4B | 26/35 | 0/35 |
| 8B | 28/35 | 3/35 |
| 14B | 26/39 | 0/39 |
| 30B-A3B | 45/47 | 38/47 |

*Layers where the stratified direction reads gender (confirmatory, bf16).*

- German compounds follow their **head's** gender at LAST in every layer at every size (AUC
  0.76–0.87); the first part's gender: 0 layers. Markedness: 0 layers once the estimator bias is
  removed.
- Both effects are **position-dependent**: at AFTER they largely vanish (except 30B-A3B).

### 3.3 Phase 4: partly shared across languages (8B, 14B)

- T1 (transfer in both directions): fails. T3 flipped pairs: passes, AUC 0.83–0.93. T4 Spanish →
  Russian soft-sign nouns (spelling-controlled): AUC 0.82 / 0.85. T2 overlap cosine 0.43 / 0.35,
  above the shuffle null. At AFTER, everything fails. **Reading: partly shared.**
- Robust to removing the German frequency axis (§3.7, test B). A DE→ES transfer gain seen after
  removal was a sample-composition artefact and is retracted.

### 3.4 Phase 5: steering grammar into social readouts

| | 8B | 14B |
|---|---|---|
| R1 grammatical push | +0.020 (3/4 layers) | 0.000 (0/3) |
| R1 random 95th pct | 0.012 | 0.008 |
| R1 social push (control) | 0.064 | 0.047 |

- German: **no valid test** (8B frame 62% < 70%; 14B gate failed); verdicts are Spanish only.
- 8B: small, language-specific bleed; 14B: none, with a sensitive readout (the social control
  moves).
- R2 moves both readouts but the noun is visible, so it can't separate routes.
- R3-POOL / P20-POOL looked positive, but **P29** shows an article-only flip moves the pooled
  readout by +0.096 on 8B, at least as much as steering: pooling is not agreement-neutral, so
  these readouts are not evidence. Reported in full as a cautionary result.
- Confidence control (a)–(c): every P29–P31 and Phase 5 verdict is unchanged.

### 3.5 English social gender (P22b, P23, P24, P27)

- Glasgow gender ratings predict model behaviour (EN ρ 0.51–0.59, DE 0.28–0.43), so the readout
  is meaningful.
- The Spanish grammatical direction separates English person words by sex (0.87 / 0.85; German
  direction weaker, 0.67 / 0.72). Its overlap with a social direction falls with size (ES 0.44 →
  0.22; DE 0.18 → 0.08).
- German neuter diminutives show no sex signal: the person effect tracks persons, not sex.

### 3.6 Unsteered article pull (P30 Spanish, P31 German)

- **P30:** a feminine article on a nonce noun read as a *thing* pulls toward feminine-rated
  adjectives in every Qwen3 size (1.7B +0.038, 4B +0.026, 8B +0.080, 14B +0.124; verified against
  `results/p30/*/analysis.json`); EuroLLM's object pull is small (+0.008) and it shows no neutral
  pull (−0.000). Neutral and person wordings pull too, so a social reading of
  the object effect is not established.
- **P31 (German, *die* vs *der*, *das* reported alongside):**

  | Model | Verdict | S_object | object / neutral |
  |---|---|---|---|
  | Qwen3-1.7B | pull survives object reading | 0.035 | 1.95 |
  | Qwen3-4B | pull survives object reading | 0.043 | 2.14 |
  | Qwen3-8B | no object pull | 0.002 | (neutral 0.002; person 0.081) |
  | Qwen3-14B | no valid test (C1 margin) | 0.025 *(exploratory)* | 0.73 [0.63, 0.85] |
  | EuroLLM-1.7B | pull survives object reading | 0.081 | 2.32 |

  Verified against `results/p31/*/analysis.json`. The *das* object effect is ≈ 0 on Qwen3
  (≤ 0.001) but not on EuroLLM (+0.045).

### 3.7 The German gender direction is aligned with word frequency

- Control (c): in German the f−m direction points toward *rarer* words (8B cos −0.34, −0.56,
  −0.62, −0.69 at the four Phase 5 layers; 14B −0.19 to −0.52); Spanish |cos| ≤ 0.14.
- This is not a list-frequency imbalance. In both training lists the feminine nouns are the
  *more* frequent (wordfreq Zipf, within spelling cells: Spanish d = +0.52, German d = +0.45;
  raw d ≈ +0.7), and frequency is a covariate in the fit; yet only German shows the alignment.
  Across the full Wiktionary lexicons the genders barely differ in frequency (Spanish f − m
  +0.17 Zipf; German −0.05), so the imbalance is a property of our samples, not of the
  languages.
- It survives a flexible frequency control (A), frequency-matched pairs (C, 14 German pairs)
  and dropping or covarying the complex nouns (D): the overlap disappears at only 0–1 of 3–4
  layers, never the pre-registered majority.
- Projecting the frequency axis out (B) leaves the Phase 4 verdicts unchanged.
- Register / age of acquisition does not explain it (8B overlap −0.62 → −0.61).
- SAE-feature test (G2, Qwen3-8B, pre-registered in `gemma-slim-prereg.md`): **no valid test.**
  The held-out compound nouns (632) do not show the overlap (cos +0.11 to +0.26 at the four
  layers, opposite sign and under the 0.3 bar), so there was nothing to remove. Descriptively,
  this means the alignment does not carry over to a different German noun sample, which fits it
  being partly sample-specific.
- **Open:** what the axis encodes, and why it appears in the training sample but not the
  held-out compounds. Gemma 3 is the cross-family check.

### 3.8 Claims, evidence and status

| Claim | Status |
|---|---|
| Directions read grammatical gender, not spelling | strong |
| Gender is read at the noun's own token | strong; must be stated |
| Partly shared across languages | strong-ish ("partly") |
| Steering bleeds grammar into social readouts at 8B | small; survives confidence control |
| No bleed at 14B | good null |
| Article pull on thing-nouns (P30/P31) | descriptive; not shown to be social |
| R3-POOL / P20-POOL | **not evidence** (P29) |
| German direction aligned with frequency | robust in training sample; absent in held-out compounds; mechanism open |

---

## 5. Limitations (to expand from the explainers' "Limits worth stating")

- One model family for the mechanistic results so far (Gemma 3 slim replication planned).
- Base models only.
- German Phase 5 had no valid test; Phase 5 verdicts are Spanish-only.
- Position dependence: the effects are read at the noun's own token.
- The German training list's complex-noun imbalance (tested, not fixed).
- Concreteness norms for German/Russian are English-gloss approximations.
