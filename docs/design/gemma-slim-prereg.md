# Gemma 3 slim replication: pre-registration (ADOPTED, PI 2026-10-22, with the tweaks below)

Written 2026-10-22, before any Gemma number. Plan approved by the PI 2026-10-22 ("slim + Gemma
Scope step"); this document fixes the details. Follows `phases-3-5-plan.md` ("After the Qwen
RunPod session: Gemma 3 + Gemma Scope 2").

## Question

Do the Qwen3 conclusions hold in a second model family? Specifically:

1. **R1 (Phases 3–5):** the same pre-registered verdicts, from the same frozen noun lists, on
   Gemma 3 4B and 12B base (`google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`; 27B only if the
   whole session is under budget and the PI agrees at pod time; the call rests on whether 4B/12B
   were interesting, since 27B has no Qwen3 peer, but Qwen3's 14B behaved differently from 8B).
2. **R2 (German frequency overlap):** does Gemma's German f−m direction also point toward
   "rarer word" (Qwen3 8B cos −0.26 to −0.69, 14B −0.19 to −0.52; Spanish ≤ 0.14)? If yes, the
   overlap is about German stimuli/German, not a Qwen quirk.
3. **R3 (Gemma Scope 2):** which SAE features carry the German frequency-aligned component? This
   is the only route we have into *what* that axis encodes. Exploratory-confirmatory split below.

We compare **conclusions** with Qwen, never vectors.

## What "slim" means (and doesn't)

- **Frozen lists, no rebuild.** Phase 3 `data/stimuli/phase3_final_v3.csv`, Phase 4
  `data/stimuli/phase4_pairs_final_v1.csv`, Phase 5 nouns `data/stimuli/phase5_nouns_v1.csv`,
  the Phase 2 list the Qwen results used. No Phase 0–2 rebuild with Gemma's tokenizer.
- **Known-check, filter only.** Each list's existing known-check (Phase 0 rule, `phase3_known`,
  same 1-nat margin, same frames) is run on Gemma; analyses use the nouns Gemma knows, exactly as
  for Qwen. Nothing is added. If a list loses more than 30% of either gender, that analysis is
  reported as "list not portable" rather than run on a lopsided set.
- **The one tokenizer-dependent set.** Phase 5's adjective readout requires adjectives whose m/f
  forms differ in exactly one final token (`phase5_stimuli`, `FAMILIES`). It gets a `gemma3`
  entry built by the existing rule, unchanged. This is a rule-built set, not hand-picked; the PI
  sees the list before the steering run.
- **Not in scope:** P29–P31 reruns, Russian, Phase 1 nonce work. Full replication only if slim
  disagrees with Qwen on a primary verdict.

## G0. Preconditions (on the pod, before any measured run; any failure stops the session)

| Check | Pass rule |
|---|---|
| dtype | bf16 only; a 20-prompt fp16/bf16 comparison is *not* run (fp16 is known to overflow) |
| loading | the text-only model loads from the multimodal 4B/12B checkpoint; next-token log-probs on 20 fixed German/Spanish prompts are finite and the top-1 continuation of "Der Hund" / "El perro" is a plausible word |
| softcap | read `final_logit_softcapping` from the config. If set, implement `c·tanh(x/c)` in `scoring.py` (currently raises `NotImplementedError`) and unit-test against the model's own forward logits (max abs diff < 1e-2 in bf16). If unset, the guard stays and is documented |
| BOS / sink guard | Gemma's tokenizer adds `<bos>`; confirm one BOS, then the existing newline buffer; the sink guard is quiet on measured positions |
| sink scan (fresh) | repeat the diagnostic that found the Qwen3 sinks, not just the guard: per-position residual norm at every layer on 20 prompts in our exact format. Any position other than BOS with norm > 10× the median of its layer (as Qwen3's newline separator was on 1.7B) is a failure; fix the format before measuring. The guard alone compares measured vectors with each other, so it would miss a format that puts *every* word on a sink |
| SAE index | Gemma Scope 2 "resid_post after layer L" ↔ our `hidden_states[L+1]`: the `sae.alignment()` check (matching index reconstructs best, by a clear margin over L and L+2) at three layers |

### G0 implementation notes (fixed 2026-10-22, before any Gemma result; code `src/gbleed/gemma_g0.py`, `gbleed gemma-g0`)

- **Text-only model.** The 4B/12B checkpoints are multimodal (`Gemma3ForConditionalGeneration`).
  `Gemma3ForCausalLM.from_pretrained` on them silently random-initialises the decoder (key
  prefixes differ), so `models.load_model` wraps the loaded multimodal model's language model and
  head in a `Gemma3ForCausalLM` (`models.text_only`, unit-tested to match the multimodal logits).
- **Softcap.** Both checkpoints have `final_logit_softcapping: null` (and the multimodal class
  never applies one), so `scoring.softcap` is a no-op on them; the check still compares the batch
  scorer with the model's own forward. Comparison is on the **same input** (prompt + top-1 token,
  read at the prompt's last position): a forward on the prompt alone differs by bf16 shape noise
  (~0.07 nat on Qwen3-1.7B and Gemma 4B alike), which is not what the check is about. Batched vs
  unbatched scorer difference is reported, not scored (Gemma 4B 0.13, Qwen3-1.7B 0.09 max, bf16).
- **SAE layers.** Gemma Scope 2 publishes `resid_post` 16k SAEs at four layers only (4B: 9, 17, 22,
  29; 12B: 12, 24, 31, 41). The index check runs at all four (the rule asked for three). G2 is
  restricted to these layers.
- **Adjective lists.** `gemma3` columns were *appended* to `phase5_adj_translations_v1.csv` and
  `phase5_es_pooled_v1.csv` by the existing rules (`phase5_stimuli.add_late_family`; every existing
  byte unchanged, checked), and `phase5_adjectives_gemma3_v1.csv` was built by `adjectives()`:
  721 adjectives (gender-valence r = −0.48; Qwen3 656), 23 gate adjectives (Qwen3 21), 76 single-
  token R3 invariants (Qwen3 51), pooled kinds 44 both-single / 84 shared-first / 29 excluded.
- **Mac smoke (2026-10-22, MPS bf16, 4B):** every check passes (`results/gemma_g0/mac_smoke/`):
  one BOS in both formats, 0 sink positions, SAE offset +1 best at all four layers (FVU 0.19–0.56;
  margin over the next index 1.04–1.85, smallest at layer 29), scorer = forward to 4e-7. Top-1
  continuations plausible (*Die Katze schläft auf dem* → *Sofa*; *Madrid es la capital de* →
  *España*). The pod run is the gate of record.

## G1 (R1). Frozen pipeline: what counts as "replicates"

For each primary Phase 3–5 verdict already reported for Qwen3, Gemma gets the same analysis with
the same thresholds, layers chosen by the same rule (relative depth, as in the Qwen runs — never
by looking at Gemma results). Per verdict, outcomes:

- **replicates** — same verdict on both Gemma sizes;
- **partly** — same verdict on one size;
- **does not replicate** — opposite verdict on both;
- **no valid test** — a gate (known-check portability, Phase 5 C1/gate) fails.

The write-up reports a table of Qwen verdict × Gemma verdict; no pooled "replication rate".

## G1 (R2). German frequency overlap

Confidence control (c) as pre-registered (`confidence-ctl-prereg.md`): cross-fitted cosine
between the f−m coefficient and the Zipf coefficient, with the within-cell×gender permutation
null, at the four Phase 5 layers. Overlap rule unchanged: cross term beats the null's 95th
percentile **and** |cos| ≥ 0.3. Then frequency-axis A, C and D (`frequency-axis-prereg.md`),
same rules. Readings:

- **German-general:** overlap at a majority of layers on both Gemma sizes, Spanish clean.
- **Qwen-specific:** no overlap on either Gemma size.
- **mixed:** anything else; reported as such.

## G2 (R3). Gemma Scope 2 features (German only)

Discovery on **training nouns only** (Phase 3 strat3 train split), per Phase 5 layer:

1. Encode the LAST-position activations. Score each feature by (i) its f−m effect (mean
   difference, within cells) and (ii) its Zipf slope, both in the same regression as the
   direction fit.
2. **Candidate set:** the top 20 features by |f−m effect| whose Zipf slope has the sign of the
   overlap. Selection uses training nouns only.
3. **Confirmatory test (held out):** on the held-out German nouns, does removing (zeroing in the
   SAE reconstruction) the candidate features reduce the |cos| of the refit direction with the
   frequency axis by at least half, while held-out gender decoding drops by less than 5 points?
   Control: 100 random sets of 20 features matched on activation frequency; the candidate effect
   must beat the 95th percentile.
4. **Descriptive only:** for each candidate feature, its top-activating nouns across the whole
   German lexicon (`data/lexicon/de_nouns.csv`), with English glosses and Wiktionary structure
   labels (compound / derived / loan), so we can see whether it looks like frequency,
   technical-register, compound-head or something else. Interpretation is labelled exploratory;
   no feature label comes from an LLM.

Also (descriptive, from `phases-3-5-plan.md` G2): whether separate "feminine" and "masculine"
features exist, or one feature with masculine nearer the default.

Phase 5 social-overlap via SAE features (plan G2 item 4) is deferred to its own prereg, if R1
gives a reason.

### G2 implementation notes (fixed 2026-10-22, before any G2 number)

The prereg left these open; fixed here before computing anything:

- **Held-out nouns:** the Phase 3 `compound_test` m/f set (632 nouns; 425 f / 207 m), the only
  sizeable German m/f set disjoint from the strat3 training nouns (the matched sets reuse strat3
  nouns; `suffix` is 78 f / 3 m). Same covariates and cells (ending|loan) as `de_train`.
- **Baseline validity:** the held-out set must itself show the overlap (confidence-ctl (c) rule:
  cross term beats the permutation 95th percentile and |cos| ≥ 0.3) on the unedited activations.
  At layers where it doesn't, that layer is "no valid test".
- **Removal:** x' = x − Σ_{i∈set} f_i(x) · W_dec[:, i] (the SAE error term is kept, so only the
  chosen features change). Features are computed per noun on the held-out activations.
- **|cos|:** the (c) cross-fitted cos(f−m coefficient, Zipf coefficient) refit on the held-out
  nouns, before and after removal (200 splits, the (c) seed).
- **Decoding:** 5-fold stratified logistic-regression accuracy (standardised, L2, C = 1) on the
  held-out nouns, before vs after; "drop" = percentage points.
- **Random control:** 100 sets of 20 features; each candidate is replaced by a random feature
  with the nearest training-noun firing count (ties broken at random; none from the candidate
  set). Candidate effect = |cos| reduction; must exceed the random sets' 95th percentile.
- **Pass (per layer):** |cos| at most half of baseline, decoding drop < 5 points, and beats the
  random 95th percentile. **Reading:** "features found" if it passes at a majority of the valid
  layers; otherwise "not found".
- **Layers:** the four Phase 5 layers (hidden_states 9, 14, 20, 25 for 8B), i.e. SAE layers
  8, 13, 19, 24 (SAE L = hidden_states[L + 1]); the mapping is checked by reconstruction error
  first (`sae.alignment`) and the run stops if the +1 offset doesn't win.
- **Sign:** the overlap is negative (feminine ↔ rarer), so candidates have
  sign(f−m effect) ≠ sign(Zipf slope).

### G2 result on Qwen3-8B (2026-10-22): no valid test

| hidden layer (SAE layer) | held-out cos | after removal | decoding before → after |
|---|---|---|---|
| 9 (8) | +0.17 | +0.185 | 99.1% → 98.3% |
| 14 (13) | +0.26 | +0.248 | 99.4% → 99.2% |
| 20 (19) | +0.17 | +0.151 | 98.9% → 98.9% |
| 25 (24) | +0.11 | +0.105 | 99.1% → 98.9% |

The SAE layer mapping passed (`results/sae_freq/alignment.csv`). At every layer the held-out
`compound_test` nouns fail the baseline-validity rule: their overlap is *positive* and below 0.3,
so there is no feminine ↔ rarer alignment to remove (pass 0/0 valid layers). The test is therefore
uninformative about the SAE; it does not say the features are absent. Descriptive only: the
held-out compound nouns do not show the training-set overlap (−0.34 to −0.69), which fits the
sample-specific reading in draft §3.7. Outputs: `results/sae_freq/`.

## Compute and logistics

- One chip type for all sizes: A100 80GB (~$1.59/h at last check; price stated again at pod
  time). Estimate 2–4 h total → about $3–6. 27B only with explicit PI OK.
- `HF_TOKEN` passed at pod time only, never on the volume. Official templates via
  `runpodctl template`. Attended session, anchored end markers.
- Activations saved and copied back; R2's A/C/D and all of R3 run on the Mac afterwards (CPU).
- Commit this prereg, the `gemma3` Phase 5 adjective list and the G0 code before the pod starts.

## Qwen3-8B first (R3 on the Mac, before the pod)

Qwen-Scope SAEs exist for Qwen3-8B Base (`Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50`, every layer;
`src/gbleed/sae.py`), and 8B shows the German overlap. G2 steps 1–4 run first on Qwen3-8B from the
saved activations, same rules; Gemma then becomes the cross-family check of whatever it finds.
No Qwen-Scope SAE exists for 14B.

## Decisions (PI 2026-10-22)

1. Adopted 2026-10-22 with the sink-scan, 27B and Qwen3-8B-first tweaks.
2. 27B: decided at pod time, by whether 4B/12B were interesting; default off.
3. 30% "list not portable" threshold: accepted as a rule of thumb, fixed before any Gemma data.

## Addendum (PI 2026-10-22): noun-sense sensitivity check

Fixed before any Gemma Phase 5 result. Some English readout adjectives are mostly nouns
(*expert*, *assistant*, *consonant*), so their Glasgow gender rating may describe the noun.

- **Rule:** drop a word when its dominant part of speech in SUBTLEX-US (Brysbaert, New &
  Keuleers 2012, `Dom_PoS_SUBTLEX`) is *Noun*; words SUBTLEX lacks fall back to WordNet SemCor
  counts (noun uses > adjective uses). The same rule on every family's list. Mechanical: genuine
  adjectives with noun homographs (*firm*, *stern*, *kind*, *light*) are dropped too; accepted.
- **Check:** rerun the Phase 5 analysis with those words removed from the English graded
  readouts (R1, R1-EN, RN-R1); compare each verdict (majority of passing layers) with the
  primary one. Primary verdicts stay those on the full approved list.
- **Reading:** *robust* if no verdict changes on that model; otherwise each change is reported.
- **Status:** pre-registered for Gemma 3; **exploratory** on Qwen3 (its results already exist).
- Code: `src/gbleed/noun_sense.py`, CLI `gbleed noun-sense`; output `results/noun_sense/`.
- **Qwen3 result (exploratory, 2026-10-22):** *robust* — 0 of 8 English graded verdicts change
  on 8B or 14B (8B es/all R1 3/4 layers both ways, effect 0.0649 → 0.0645; 14B stays 0/3).
  The unmodified rerun reproduces the published verdicts exactly. `results/noun_sense/compare.csv`.

## Bug fix before any measured Gemma data (2026-10-23): Phase 5 prompts had no `<bos>`

The Mac smoke run (Gemma-3-4B, not a measured run) failed every Phase 5 frame check, Spanish
included (feminine nouns right 0.61 in the adjective frame). Cause: Phase 5 builds its prompts
with `add_special_tokens=False` (`phase5._encode`), so Gemma saw no `<bos>`. G0's BOS check
covered the activation format and the Phase 0/known-check scorer, not this path. The G0 rule
above ("exactly one `<bos>`") already applies, so this is a code fix, not a design change:
`phase5._encode` now prepends one `<bos>` for Gemma only (noun positions shifted by one), and G0's
BOS and sink checks now include a Phase 5 prompt. Qwen3 and EuroLLM prompts are unchanged
(tested), so no frozen result moves. The smoke frame-check numbers above are void; the pod run
remains the gate of record.

## Known-check filter: wiring completed before any Gemma analysis (2026-10-24)

Found before any Gemma Phase 3–5 analysis was run or looked at: the rule above ("analyses use the
nouns Gemma knows") had been wired into the Phase 4 training sets and the R2 code (confidence
control, frequency axis) only. Now also applied, each list filtered by its own check, identity for
Qwen (`known_filter.py`, tests in `tests/test_known_filter.py`):

- **Phase 3** analysis: every set of `phase3_final_v3` by `phase3_known` (`multi`, both-gender
  items with no known-check, kept).
- **Phase 4** test pairs: a pair is kept only if both nouns are known on the Phase 4 pair checks,
  the rule `phase4.finalize` applied to Qwen3 1.7B/4B.
- **Phase 5** analysis: steering rows of steer_es / steer_de nouns by the Phase 2 / Phase 3
  checks, pair nouns by the pair rule, classics kept (no known-check existed for them); P20 and
  P25 pairs by the pair rule. Nonce (R-NONCE) rows are untouched.

**Not re-run:** the Phase 5 gate and frame check ran on the pod over the unfiltered noun list.
They are pass/fail checks on behaviour, and the filtered nouns are a small share of each list.
This is recorded as a deviation, not corrected.

**Phase 2 is outside R1's scope** ("Phases 3–5"). Its Spanish probe would also need Phase 1
activations, which the Gemma driver did not extract.

## G1 results (2026-10-24)

Pod runs: 4B commit 88cc069, 12B commit ecbbeea (both L40S, bf16). Analysis on the Mac at commits
1cd5d6e + c66819f (known-check filter wiring; P20 pair fix). Tables: `results/gemma_slim/`.
Raw archives: `/Volumes/Lexar/gbleed/gemma-pod{,-12b}/`.

**R1 verdict table (Qwen × Gemma), primary position (LAST)**

| verdict | Qwen3-8B | Qwen3-14B | Gemma-4B | Gemma-12B | outcome |
|---|---|---|---|---|---|
| known-check portability | (lists built on Qwen) | | es f .03/m .06, de f .11/m .09 | es f .12/m .08, de f .04/m .06 | portable |
| Phase 3: compounds follow the head | yes (all layers) | yes | yes, 33/33 | yes, 47/47 | **replicates** |
| Phase 3: markedness | not supported | not supported | not supported (0) | not supported (0) | **replicates** |
| Phase 4 T1: direction transfers ES↔DE | no (6/35, 5/35) | no (7/39, 22/39) | no (7/33, 26/33) | no (11/47, 27/47) | **replicates** |
| Phase 4 T3: pairs follow each language's gender | yes | yes | yes (33/33) | yes (47/47) | **replicates** |
| Phase 4 T2: cosine above null | 0 layers | 0 layers | 0 layers | 0 layers | replicates (rule); cosine values not interpretable, see below |
| Phase 5 Spanish, noun push, R1 majority | yes (3/4, +0.020) | no (0/3) | no (0/3, −0.004) | no (1/3, +0.029) | **does not replicate** the 8B positive; matches 14B |
| Phase 5 Spanish, all positions (P14), R1 | yes (3/4, +0.065) | no (0/3) | no (0/3, +0.035) | no (0/3, +0.100) | **does not replicate** the 8B positive; matches 14B |
| Phase 5 German | no valid test (frame) | no valid test | no valid test (C4 frame m 0.67) | no valid test (C4 frame passes; gate fails: random directions flip 0–50%) | no valid test |

The 12B all-position push moves R1 by +0.100 (more than Qwen3-8B's passing +0.065), but random
directions pushed the same way move it about as much, so it fails at every layer. The social
direction (positive control) moves R1 +0.29 to +0.34 on both sizes, so the readout can detect
a gender push. Noun-sense check: 0/8 Gemma verdicts change (`results/gemma_slim/noun_sense/`).

**Robustness (AFTER position).** Phase 3 at AFTER: compounds "neither" (4B 6/33, 12B 18/47
layers), as on Qwen (8B 7/35): gender sits on the noun's own token in Gemma too. **Phase 4 at AFTER
was not run:** the pod extracted Spanish training activations at LAST only
(`phase2_bare_after` missing). This is an extraction gap, not a result.

**Cross-fitted cosines are not interpretable on Gemma (R2, Phase 4 T2).** The split-half cosine
estimator gives values outside [−1, 1] (−3.97 to +1.11) or none (negative squared-length estimate)
at about half of the Gemma layers. The Spanish control and the number direction are affected too
(number cos up to +1.51), and Phase 3's own interpretability check finds 0 interpretable layers.
Gemma's residual squared lengths are about 10³ times Qwen's (outlier dimensions), so the signal is
small relative to the noise the estimator removes. **Resolved by the amendment below** (outlier
fix + validity gate): see "Amendment results".

## Amendment: outlier-dimension fix with a validity gate (PI, 2026-10-24; before any rerun)

**Why.** The cross-fitted cosines (R2, Phase 4 T2) gave impossible values on Gemma (outside
[−1, 1], or undefined). Outcome-blind diagnostics (`docs/design/gemma-outlier-dimension.md`;
`scripts/outlier_fix_diagnostics.py`, `results/outlier_fix/diagnostics.csv`) traced it to one
residual dimension carrying 59–92% of the variance across nouns (dimension 443 in 4B, 2339 in
12B, at every layer from 3 on; Qwen3 ≤ 1.6%). The fix was chosen on those diagnostics only, with
no frequency or cross-language quantity computed under any candidate.

**Fix (B).** Per model and layer, zero every dimension with more than 10% of the variance across
all Phase 3 bare nouns (`src/gbleed/outlier_dims.py`; reference file `phase3_bare`). Rejected:
per-dimension standardisation (damages Qwen: lower gender AUC and split-half agreement, invalid
Qwen cosines); top-PC removal and RMS normalisation performed the same as B in the diagnostics
but are less direct. On Qwen3 no dimension qualifies, so Qwen results are unchanged by
construction; R2 is rerun on all four models anyway as the check.

**Validity gate.** A layer whose cross-fitted cosine is non-finite or outside [−1, 1] is
**invalid** and shows no overlap. Per model × language over the four Phase 5 layers:
- **no valid test** if 2 or more layers are invalid;
- **overlap** if at least 3 valid layers show the overlap (cross term beats the null's 95th
  percentile and |cos| ≥ 0.3, unchanged);
- **no overlap** if at least 3 valid layers do not;
- **mixed** otherwise.

The R2 readings across sizes (German-general / Qwen-specific / mixed) are unchanged, applied to
these verdicts. The Spanish control must not show "overlap" for "German-general".

**Frequency-axis A/C/D** are run only on a Gemma size whose German verdict is "overlap", with the
same fix. A variant layer whose cosine fails the gate counts as "overlap not gone" (conservative).

**Phase 4 T2.** Rerun with the fix for its cosine values; reported only at valid layers. T1/T3
verdicts stay those of the unfixed run (primary); the fixed run's T1/T3 are descriptive.

**No second fix.** If the gate leaves "no valid test", that is the result.

### Amendment results (2026-10-24)

All layers valid on all four models. **R2: Qwen-specific.** German overlap on Qwen3-8B (4/4) and
14B (3/4), reproduced under the fix; **no overlap on Gemma-4B (1/4) or 12B (0/4)**. Spanish:
no overlap except Gemma-12B "mixed" (2/4). A/C/D not run. Phase 4 T2 mean cosine 4B 0.47, 12B
0.55; T2 verdict unchanged (0 layers on all models). Details: `gemma-outlier-dimension.md` §6.
