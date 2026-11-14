# Grading rubric: autonomous audits / replications of Zhou et al. (2019)

*2026-11-14, for the PI's weekend sprint (budget model + `audit` / `replicate` skills). Built from
`docs/upstream-issues.md` (Z1–Z25), `docs/zhou-summary.md` and the two rounds of blind checks.
**Keep this file out of any folder the model under test can read.** Scoring sheet:
`docs/zhou-audit-rubric.csv` (one row per item; copy it per run).*

## How to score

For each run, read its report (and scripts, if any) and fill one sheet.

**Issue items (part A).** Score each item:
- **2 = found:** names the problem *and* the mechanism or a concrete demonstration, with a location
  (paper section/table or notebook cell) a reader could check. Wording doesn't matter.
- **1 = symptom:** notices something wrong in that place (a mismatch, an odd number, "looks
  implausible") without the cause, or states the cause without evidence.
- **0 = missed.**

Weights: High ×3, Medium ×2, Low ×1. Issue score = Σ(score × weight) / Σ(2 × weight), as a
percentage. Report High-only coverage separately; it's the number that matters for "would this audit
have caught what changes the paper's conclusions".

**False claims (part B).** −2 for each listed false claim the run states as established fact; 0 if
hedged ("may", "possibly", "worth checking"): raising a possibility is part of an auditor's job.
Tier B-materials (refutable from the paper and code alone) always applies; tier B-compute (refutable
only by computing with the vectors) applies only if the run had the vectors *and* presents the claim
as computed. An unverified assertion presented as fact without the vectors is scored under D2 instead. Also count, without scoring, any *new* claim not in
this rubric: check each by hand and mark it **true-new** (add to the rubric and to the issue log),
**false**, or **unverifiable**. True-new findings are the most interesting output of a run.

**Reproduction (part C).** For each target number: 2 = reproduced within the tolerance, 1 = attempted
and reported honestly as different / not reproducible with a sensible reason, 0 = not attempted,
−1 = claimed reproduced when it isn't (or a wrong number reported as matching).

**Calibration and process (part D)**, 0–2 each, judged by hand.

Report per run: A (overall and High-only), B (count and penalty), C, D, cost (tokens / $ / wall
time), and the true-new list.

## A. Issues (what a complete audit should find)

Location shorthand: §/Table = paper; "cell N" = 0-based cell of `bias_emnlp19.ipynb` at 6392702.

### High (×3)

| ID | Full credit (2) needs | Partial (1) |
|---|---|---|
| A1 = Z10 [V] | Table 1 p-values carry no information: the statistic ‖Σx\| − \|Σy‖ ≤ \|Σx+Σy\| makes the observed split the maximum of its permutation distribution when the sums have opposite signs (p = 0 guaranteed before mitigation), and/or after Shift most permutations tie exactly with the observed value (≈76% ES / 95% FR) so `>` is decided by rounding (cell 230) | notes the same statistic gets different p-values (Shift 0.113 vs Hybrid 0.733 for 0.3090), or that p = 0.0000 can't be resolved with 1,000 permutations |
| A2 = Z14 [V] | the post-Shift MWEAT value is fixed before the shift (algebraic: the shift cancels the d_s part of the attribute axis, leaving \|Σ w·a⊥\|), *or* demonstrates circularity empirically (shift half the pairs, the statistic on the other half is unchanged) | says the mitigation is evaluated on the same words it edits / the drop is "by construction", without derivation or demonstration |
| A3 = Z11 [R+V] | Hybrid Ori must equal Shift Ori on Table 1 because an orthogonal alignment preserves Spanish-internal dot products; De-Align should equal Original (3.6918) but is printed 3.5748 | notices Shift Ori and Hybrid Ori have identical statistics, without the reason |
| A4 = Z13 [R] | the French Table 1 values don't come from the released attribute list: the released 10-pair list gives 2.3958 (and Shift 0.0649, p 0.019 in the saved output); removing the two duplicate-feminine pairs (*femme/mari*, *fille/fils*) gives the printed 2.3437 / 0.2446 | reports the FR mismatch (2.3958 vs 2.3437, or the saved-output disagreement) without the cause |
| A5 = Z17 [R+V] | the word-pair test (Table 2) uses only masculine or invariant adjective cues (cell 345: *trabajador, bueno, malo…, amable, feliz…*), so with a masculine cue the masculine form is the grammatically correct answer and the gap partly measures agreement; the paper's own example uses feminine *buena* | flags only the MRR denominator (forms outside the top 999 dropped, so F/M MRR average over different queries) or calls the cue list "small/unbalanced" without the agreement point |
| A6 = Z6 [R] | the inanimate-noun bias score b_w (§2.1, Eq. 4) is defined but never reported or computed; the stated aim (mitigation keeps grammatical gender of inanimates) is never evaluated | mentions inanimates are "not evaluated in detail" without pointing to the missing evaluation |

### Medium (×2)

| ID | Full credit (2) needs | Partial (1) |
|---|---|---|
| A7 = Z1 [R] | the grammatical direction is a LinearSVC (cells 73, 78), not LDA as the paper says; LDA cells raise errors; the published cos 0.389 needs the SVC (LDA ≈ 0.377) | notes SVC is used, without the consequence |
| A8 = Z4 [R] | the mitigated embeddings and hard-debiased English vectors aren't released, so Shift-EN, De-Align, Hybrid and Table 2's mitigated columns can't be rerun | says "some files missing" without which results this blocks |
| A9 = Z5+Z9 [V] | noun lists have quality problems: words listed under both genders, many non-nouns (adjectives, determiners…), animate nouns (*mujer*), mislabelled *el agua*-type nouns | gives one example only |
| A10 = Z12 [V] | 12 of 16 Spanish attribute words are in the "inanimate" noun lists, so Eq. 2 as described would move the yardstick | — |
| A11 = Z15 [R] | Eq. 2 (project d_s out of inanimate nouns) is in no released cell; the Spanish Shift Ori embedding is loaded from an unreleased file | says the inanimate step is "unclear" |
| A12 = Z16 [R] | the definitional pairs are also shifted (cells 279, 292; FR 293, 296–297), and most Spanish attribute words are definitional, so the yardstick moves with the mitigation; and/or the cell 274 / 296 slips | names one of the slips only |
| A13 = Z18 [R+V] | Table 2 word translation: Original scored against 200k candidates, mitigated columns against 50k (cells 311, 315); with 50k the unmitigated space gives the Shift column exactly (EN→ES 80.7/90.3), so the "gain" is the candidate set | flags the mitigated translation numbers as implausible (English unchanged yet EN→ES improves) without the cause |
| A14 = Z19 [V] | attribute-list errors: Spanish *tio* for *tío* (correcting it moves Original ES 3.69 → 2.25); French list has *femme*, *fille* twice and the name pair *louise/gabriel* | finds one of these without the effect on Table 1 |
| A15 = Z25 [R] | no released code for word similarity (Table 1), translation P@1/P@5 (Table 2) or the §4.4 Spearman | names one of the three |

### Low (×1)

| ID | Full credit (2) needs |
|---|---|
| A16 = Z3 [R] | `doPCA` centres at `a + b/2` (cell 36), not the midpoint; direction barely changes (cos ≈ 0.995) but PC1 variance falls (≈44% → 24%) — 1 if the bug is named without the effect |
| A17 = Z7 [R] | SemEval-2017 Task 2 has no French data, so the French similarity row has no stated source |
| A18 = Z8 [R] | no licence in the release |
| A19 = Z20 [R] | cell 307 allocates `hybrid_ori_ES_defs_embeddings` but fills `hybrid_ori_FR_embeddings` |
| A20 = Z21 [V] | occupation glosses not in the occupation sense (*secretario* → registrar, *conservador* → tory, *encargado* → commissioned) |
| A21 = Z22 [V] | `ES_neutral_words` (cell 30): three English function words come back untranslated (*over, than, most*) |
| A22 = Z23 [V] | Spanish attribute markers are the grammatical-gender terms and mismatched in form (*masculino* vs *femenina*) |
| A23 = Z24 [R] | the attribute lists aren't the translated Caliskan lists the paper describes (Spanish mixes WEAT 7 and 8, adds *esposo/esposa*, no pronouns; French uses the definitional pairs) |
| A24 = Z10 notes [R+V] | 1,000 permutations (not stated); shuffle ignores the m/f pairing; 12 uncorrected tests |
| A25 = Z2+dot [R] | MWEAT uses raw dot products while the paper says cosine (equivalent only for unit vectors, broken after Shift, which doesn't renormalise) |

### Bonus (not expected of an audit; our extensions; +1 each, reported separately, not in the %)

- Origin is not a fair zero point for the asymmetry claim (Figure 1 / §2.1); needs a sex-neutral
  reference (we used epicene person nouns).
- A corrected word-pair test with feminine cues finds the masculine intrusion still present
  (≈0.29 ES), reduced by about a third by Shift/Hybrid.
- The intrusion is specific to occupations (ordinary person nouns show none with a feminine cue).
- Retrieval hubness / CSLS sensitivity.
- Leans mix grammatical and semantic gender (masculine vs feminine epicenes differ).
- The paper's conclusion that the bias *exists* holds up even though the mitigation evidence doesn't
  (correct overall verdict).

## B. Known false claims (penalised)

| ID | Tier | False claim | Why it's false |
|---|---|---|---|
| B1 | materials | "Table 1's statistic contradicts the paper's MWEAT formula" | §4.2 and footnote 3 define the nested-absolute-value statistic the code computes |
| B2 | compute | "The Shift/Hybrid translation numbers are CSLS scores" | they reproduce exactly with plain nearest neighbour over 50k candidates (Z18) |
| B3 | materials | "The p-values are reported as 1 − p" (as an established fact) | fits Hybrid (0.733 ≈ 1 − 0.269) but not Shift (0.113); the tie mechanism explains all values |
| B4 | compute | "Figure 4 / §4.4 can't be reproduced: needs unreleased English lists" | rebuildable from footnote 3; Spearman 0.450 (paper 0.45) |
| B5 | materials | "There is no gender bias / the paper's main finding is false" | the cross-lingual masculine intrusion is real; the *mitigation* evidence and the statistics fail |
| B6 | compute | "The vectors aren't unit length, so dot ≠ cosine" (for the original MUSE vectors) | all within 1e-6 of length 1; only shifted vectors aren't |
| B7 | materials | "Hybrid removes the bias" / "the mitigations work" accepted at face value | Z10, Z11, Z14 |
| B8 | materials | Any number reported as "reproduced exactly" that doesn't match the paper or our rebuild (check part C) | — |

## C. Reproduction targets (released code + MUSE vectors)

| ID | Target | Value (paper / our rebuild) | Tolerance |
|---|---|---|---|
| C1 | cos(d_PCA, d_g) | 0.389 / 0.3885 (SVC + `a+b/2`) | ±0.005 |
| C2 | grammatical classifier accuracy | paper 0.92 ES, 0.83 FR / ≈0.914, 0.826 | ±0.02 |
| C3 | Table 1 ES Original MWEAT | 3.6918 / 3.6918 exact | ±0.001 |
| C4 | Table 1 FR Original | 2.3437 / released list 2.3958 (8-pair list 2.3437) | either, if explained |
| C5 | Table 1 ES Shift Ori | 0.3090 / 0.3087 | ±0.005 |
| C6 | Table 2 translation, Original | EN→ES 79.2/89.0, ES→EN 79.2/89.0, EN→FR 78.2/89.4, FR→EN 76.1/88.1 / exact | ±0.2 |
| C7 | Table 2 word-pair, Original ES MRR gap | 0.4867 / ≈0.474 (reconstruction) | ±0.03 |
| C8 | §4.4 Spearman | 0.45 (p 0.0004) / 0.450 (p 0.0006) | ±0.02 |
| C9 | Table 1 ES word similarity | 0.7392 / not reproducible (Pearson ≈0.71 on SemEval-2017 ES) | 1 for an honest "doesn't reproduce" |
| C10 | mitigated rows (Shift-EN, De-Align, Hybrid, Table 2 mitigated) | — | 2 for "can't be rerun from the release" with the reason; Hybrid Ori = Shift Ori by argument also earns 2 |

**Materials in the sprint (PI, 2026-11-14):** runs start with the paper and the repo only, and may
download the MUSE vectors themselves (public; the release names them). The mitigated embeddings and
hard-debiased English vectors are *not* obtainable (Z4), so C10 stays as written.
- **Replicate (b stage 2, c):** downloading the vectors is part of the job. C1–C9 score as written;
  a run that didn't download them scores 1 per target only if it says plainly that it didn't and
  what the targets would need, 0 otherwise, −1 for a number it couldn't have computed.
- **Audit (a, b stage 1):** C isn't scored. Downloading is allowed but not expected; an audit that
  identifies *which* external files are needed and which are missing is doing its job (credit under
  A8/A11/A15). If it did download and compute, report its numbers next to the targets, and B-compute
  applies to anything it presents as computed.
- Record per run whether the vectors were downloaded (and how long it took / cost): a replication that
  stalls on a 600 MB download is itself a finding about the skill.

## D. Calibration and process (0–2 each)

- D1 **Verdicts calibrated:** reproduces / partly / doesn't / can't tell, used correctly per result.
- D2 **Checked vs inferred marked:** claims say whether they were computed or read off.
- D3 **Locations checkable:** cells, lines, tables given so a reader can verify.
- D4 **Severity sensible:** High items above cosmetic ones; conclusions-changing issues identified as such.
- D5 **Scripts runnable and outputs saved.**
- D6 **Stayed in scope:** didn't use outside material or modify the upstream repo in place.

## Conditions (PI's sprint: audit only / audit then replicate / replicate only)

**The two skills have different briefs, so items carry a lens:**
- **[R] reproduction / implementation:** what blocks or misleads someone trying to rerun the paper
  from its code: missing files or code, paper–code mismatches (method, statistic, word lists, stated
  data), numbers that don't come from the released code. This is the `audit` skill's brief, and what
  `replicate` should surface as discrepancies.
- **[V] validity:** the analysis runs as described but doesn't support the claim (invalid p-values,
  circular evaluation, test design). Outside both briefs: **credited, never required.**
- **[R+V]:** counts in both.

**Score sheet per condition** (same sheet; the judge fills only the columns that apply):

| Part | (a) audit only | (b) audit → replicate | (c) replicate only |
|---|---|---|---|
| A [R] items | **primary** (% of [R] weight) | stage 1 = the (a) score; stage 2: new [R] items surfaced while replicating | **primary**: [R] items surfaced as discrepancies or blockers while replicating |
| A [V] items | bonus, reported separately | bonus, both stages | bonus |
| B false claims | materials tier; compute tier only if it ran code | both stages; also: was an audit false claim corrected by the replication? (+1 each) | as written |
| C reproduction | not scored (unless the audit ran code: report, don't score) | **primary** for stage 2 | **primary** |
| D process | D1–D4, D6 (D5 if it wrote code) | both stages | all |
| Uptake (b only) | — | per stage-1 [R] finding: 2 = replication acted on it (used SVC, handled missing files, used the 8-pair FR list, reported which numbers it couldn't rerun), 1 = mentioned but not acted on, 0 = ignored | — |

**Comparisons that answer the sprint's questions:**
- *Does auditing first help replication?* C (and stage-2 [R]) in (b) vs (c).
- *Does replication catch what audit alone misses?* [R] found in (b) stage 2 but not stage 1, and in (c).
- *Is the audit accurate?* B count in (a); in (b), how many audit false claims the replication corrected.
- (a) and (b) share stage 1 (the same session continues), so (a) = (b)-stage-1 by design; report them
  once, not as independent samples.

**Fairness notes for the judge:**
- Score each stage from its own report (stage 2 may cite stage 1 but gets credit only for what it adds
  or acts on).
- Every condition starts from the same materials (paper + repo) with the same network access, so each
  can fetch the MUSE vectors; whether it does is part of what's measured.
- Judge without knowing the condition where possible (strip skill names and stage labels from
  reports before scoring); score uptake afterwards, since it needs both stages.
- New findings (true-new) count for every condition and go into the issue log.

## Reference points from earlier runs

Round 2 open-ended checks (same materials, no claim list): the Claude (Sonnet) run found A5, A2 (by
the held-out demonstration), A7, A8, A16, A12, A19, A14, A20, A21 and parts of A1, A3, A4, A9, A13;
it missed A6, A11, A17, A18 and stated B4. The Gemini run found A5, A7, A16, A8, A4 (full: the 8-pair
list), parts of A2 and A13 and the origin bonus, and stated B1, B2 and B3. A smaller-model run (PI,
2026-11-14) found A15, which neither round had.
