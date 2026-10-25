# Methods (paper-style draft, 2026-10-23)

*Adjunct to `2026-10-22-methods-results-draft.md`, not a replacement. That draft explains each step for a reader new to the field. This one follows a conventional methods layout and assumes a reader who knows what a residual stream, a linear probe and a bootstrap are. It still states every procedure, stimulus and decision rule we actually used. Example prompts are given in full; full lists are cited by file.*

---

## 2. Methods

### 2.1 Design overview

We asked whether a language model's representation of grammatical gender, learned from inanimate nouns, shares structure with its representation of social gender. The study ran in six gated phases. Each phase's output was a precondition for the next:

| Phase | Purpose | Key output |
|---|---|---|
| 0 | Behavioural known-check: does the model assign each noun its dictionary gender? | per-model lists of known nouns |
| 1 | Can spelling (*-o/-a*) be linearly erased? | negative; spelling controlled by stimulus design instead |
| 2 | Spanish gender direction, tested on spelling–gender dissociations | Spanish direction |
| 3 | German gender direction, tested on compounds; markedness | German direction |
| 4 | Cross-language sharing (T1–T4) | evidence for partial sharing |
| 5 | Steering along the grammatical direction; social readouts (R1–R3) | the primary bleed test |

Follow-up studies are numbered P*n* in order of proposal. Each was pre-registered separately. The main ones are P22b–P27 (English social gender), P29 (pooling neutrality), P30–P31 (unsteered article pull), and the late controls (confidence, frequency axis, register, SAE features).

**Pre-registration and deviations.** For every confirmatory analysis, the hypothesis, statistic, sample and verdict rule were committed to the repository before data collection (`docs/design/*-prereg.md`). Deviations are logged with rationale in `docs/decisions.md`. Analyses specified after seeing data are labelled exploratory and never used to rescue a confirmatory verdict. Stimulus lists are frozen once results exist; flaws discovered later are logged and their impact tested (§2.13), not edited away.

### 2.2 Models

- **Primary family:** Qwen3 base models at 0.6B, 1.7B, 4B, 8B and 14B parameters (28–40 layers, hidden width 1,024–5,120). Also Qwen3-30B-A3B-Base, a mixture-of-experts model (48 layers, width 2,048, ~3B active parameters), run once as exploratory. 8B and 14B are the confirmatory models. Smaller sizes were used for development and to check size trends.
- **Second family (behavioural tests):** EuroLLM-1.7B.
- **Replication:** Gemma 3 4B and 12B pretrained, under a separate "slim" pre-registration (frozen lists, known-check filter only, loss ≤ 12% per gender; `docs/design/gemma-slim-prereg.md`). Results in §3.9.
- Base (pretrained-only) checkpoints throughout; no instruction-tuned models.
- **Compute:** confirmatory runs in bf16 on single A100 80GB or L40S GPUs (RunPod). Development on Apple-silicon MPS in fp16. Quantities that are compared with each other always come from the same hardware and precision.

### 2.3 Materials

#### 2.3.1 Lexical source and annotation

All nouns and gold gender labels come from Wiktionary via the kaikki.org JSONL dumps (snapshot 2026-10-03). No label comes from a model. Eligibility for every list:
- a single gender;
- no archaic, slang or regional tag;
- primary part of speech is noun, and the word is not an inflected form;
- not more frequent as an English word.

Every noun is annotated by rule with:
- **frequency:** Zipf, from `wordfreq`;
- **token count** per tokenizer;
- **concreteness:** whether the dominant WordNet sense of the English gloss falls in a physical-entity lexicographer file;
- **loan status:** Wiktionary etymology templates.

Inanimacy is enforced twice: once by gloss rules, and once by a WordNet check on the first two senses (person, animal or group ⇒ drop). Russian adds a morphological animacy check. Inclusion is by rule only ("drop, don't review"). The single hand-picked set is the 19 classic literature pairs (§2.3.4), which are always reported separately. Filter counts at each step are in `docs/screening/01–07`.

#### 2.3.2 Spanish

Exclusions, in order:
1. nouns with gender-predictive endings: *-o, -a, -ción, -dad, -umbre, -ez, -itis, -or, -aje, -ón, -ista, -ante, -ud* and others;
2. English-near spellings;
3. sex-associated meanings (clothing, jewellery, cosmetics, body parts);
4. persons, animals, groups, places and chemicals;
5. nouns with Zipf < 2;
6. nouns not known to the model (Phase 0, §2.5.1).

Remaining nouns are stratified into **cells**: final two letters × loan status.

| Set | Use | n |
|---|---|---|
| `strat` | training | 33 f, 132 m |
| `matched` | training, equal f/m within cells | 27 f, 27 m |
| regular test | held-out | 247 |
| exceptions | masc. *-a* (*el día*), fem. *-o* (*la mano*), Greek *-ma* (*el problema*), clippings (*la foto*) | 7 + 20 + 2 |
| homographs | noun/verb homographs (*camino* "path" / "I walk") | 72 |

Word-initial *a-* was imbalanced (0/33 f, 17/132 m), so it was added as a nuisance covariate. 37 training nouns that later became flipped-pair nouns were excluded from the Phase 4 tests.

#### 2.3.3 German

Exclusions, in order:
1. **compounds:** any noun whose final ≥3 letters, after ≥3 letters, form another noun (≈11,000 removed);
2. names, diminutives, chemicals and *Ge-* collectives;
3. animates and groups;
4. *-er* agent nouns derived from verbs, other deverbal nouns, and sex-typed clothing;
5. gender-fixing suffixes (*-ung*, *-chen*, …);
6. nouns not known to the model (three-article known-check).

Cells are final two letters × loan status, plus shared-suffix cells (*-nis*, *-tum*). Loan status matters here because loans are disproportionately neuter (44% vs 28%).

| Set | Use | n |
|---|---|---|
| `strat3` | training | 64 m, 27 f, 31 n |
| matched | training | 14 per gender |
| compound test | held-out compounds; **conflict** items have a first constituent whose gender differs from the head's (*der Türrahmen* = *die Tür* + *der Rahmen*) | 720 (219 conflict) |
| compound_test (G2) | further held-out compounds | 632 |

German compounds are right-headed: the final constituent determines both the gender and the semantic category. The conflict items therefore dissociate the head's gender from the first constituent's gender and from the word's initial tokens.

#### 2.3.4 Cross-language flipped pairs

- German and Spanish eligible nouns with Zipf ≥ 2.5 were paired via a shared English gloss, keeping the most frequent noun per gloss on each side. This gave 2,217 pairs, of which 519 have opposite genders.
- Rule-based exclusions were then applied: animates, sex-associated meanings, English-near forms, and training-set overlap. The PI also excluded 51 pairs for sense mismatches; a spot check had estimated a ~10% gloss-mismatch rate.
- Adopted list: 389 pairs (239 de-f/es-m, 150 de-m/es-f) in `data/stimuli/phase4_pairs_pool_v1.csv`. The analysed set is the 271 pairs whose German and Spanish nouns are both known by Qwen3-1.7B *and* 4B (`phase4.finalize`; 166 de-f / 105 de-m), frozen for every model (`data/stimuli/phase4_pairs_final_v1.csv`).
- The 19 classic literature pairs (*Brücke/puente*, *Mond/luna*, *Sonne/sol*, …) are reported separately.

#### 2.3.5 Russian (prepared; T4 run, main Russian study not yet run)

Soft-sign (*-ь*) nouns only, the one Russian ending class shared by masculine and feminine. Gender-fixing sub-suffixes inside *-ь* are cell-separated (`ru:spell` cells: *-ость*, other *-сть*, hushing + *ь*, *-знь/-снь*, *-тель*, other). After filtering: 245 nouns (123 m, 122 f) in 10 cells, with all within-cell confound effect sizes |d| < 0.3.

#### 2.3.6 Readout vocabularies and norms

- **English adjectives.** From the Glasgow Norms (Scott et al., 2019; 1–7 ratings of gender association, valence, arousal, size), keeping words that are:
  - adjective in ≥ 50% of their WordNet senses;
  - Zipf ≥ 3;
  - a single token for the model family.

  This leaves 656 adjectives for Qwen3 and 721 for Gemma 3. In this set, gender rating correlates with valence at r = −0.47, so valence, arousal, size and Zipf are covariates in every readout model.
- **Spanish gate adjectives.** Masculine/feminine pairs whose tokenizations differ only in the final token (21 for Qwen3, 23 for Gemma 3).
- **Spanish invariant adjectives** for R3/P30: single-token, gender-invariant (*fuerte*, *elegante*, *frágil*). There are 51 for Qwen3 and 72 for EuroLLM; each was PI-screened for drop only.
- **Nonce nouns:**
  - Spanish-shaped Wuggy stems × endings *-a/-o/-e/-iz* (150 stems × 4 = 600; P19, P30);
  - German nonce nouns (300), filtered so that no ending is > 50% one gender in the lexicon (P31).

  All are absent from Wiktionary in six languages and have zero `wordfreq` frequency.
- **English person words and names (P23–P24):**
  - unmarked sex-specific person nouns (GN-GloVe lists + WordNet, sex from the WordNet gloss; 16 f / 14 m);
  - SSA first names (≥ 10,000 births, ≥ 95% one sex, not a WordNet lemma; 1,779 names, within-final-letter cells).
- **German neuter diminutives (P27):**
  - persons, sex from the gloss: 11 f / 8 m;
  - inanimate diminutives: 165, stems 68 f / 62 m / 35 n.

### 2.4 Representation extraction

- **Input format.** Each noun is presented in isolation as `<|endoftext|>` `\n` `␣noun`. Gemma uses `<bos>` in place of `<|endoftext|>`.
  - The newline buffer exists because Qwen3 makes the first content position after the document token an attention sink: its residual norm is ~180× the median on 0.6B.
  - A guard aborts the run if any measured position's norm exceeds 10× its layer median.
- **Recording.** Residual-stream activations are recorded at every layer. Analyses use inner layers 1 to L−1, since the final stored layer is post-norm.
- **Positions:**
  - **LAST:** the noun's final token;
  - **AFTER:** an appended `\n` token, which does not affect LAST under causal attention.
- **Choice of primary position.** This was made before any gender analysis, by a pre-registered gender-blind rule. AFTER would be primary only if it preserved word identity (cross-context retrieval against *Beispiel: X*) *and* reduced dependence on tokenization. It met both criteria in 1 of 4 model × language combinations (1.7B, 4B), so **LAST is primary** and AFTER a pre-registered robustness check.

### 2.5 Behavioural measures

All behavioural quantities are log-probabilities from the unmodified model under teacher forcing. Prompts are scored with the model's own BOS convention: none for Qwen3, `<bos>` for Gemma (see §2.13 for EuroLLM).

#### 2.5.1 Known-check (Phase 0; Phase 3 three-article variant)

Each noun is scored in two frames, once with each gender's article:

| | frame 1 | frame 2 |
|---|---|---|
| German | *Das hat etwas mit dem/der X zu tun.* | *Hier ist ein/eine X.* |
| Spanish | *Esto tiene que ver con el/la X.* | *Aquí hay un/una X.* |

The score is the log-probability of the continuation (noun plus following text), excluding the article itself, because some articles are shared across genders (*ein*). The margin is the correct-gender score minus the other gender's.

| Status | Condition |
|---|---|
| known | correct gender wins by ≥ 1 nat in both frames |
| wrong | other gender wins by ≥ 1 nat |
| conflict | right by ≥ 1 nat in one frame, wrong by ≥ 1 nat in the other |
| unsure | otherwise |

German Phase 3 uses *Hier ist der/die/das X* and *Dort gibt es einen/eine/ein X*, with the margin taken against the strongest rival. A few-shot "definite article:" quiz was also run. It was order-sensitive (0.6B: 20% DE / 45% ES of answers changed under reordering), so it is diagnostic only. Phase 0 balanced known rate = mean over genders of the share known, on 257 nouns per language.

#### 2.5.2 Phase 5 gate frame checks

These are run unsteered on each model's known nouns with Zipf ≥ 4. A frame passes if ≥ 70% of nouns of *each* gender receive the correct agreement.
- **Spanish:** *Mi X es muy*, read as log P(f form) − log P(m form) at the final differing token of each gate adjective, averaged over adjectives.
- **German candidates, fixed in advance:**
  - C1 *Wörterbuch:\n X,* + *der/die*;
  - C2 *Thema: X. Was macht man mit* + *ihm/ihr*;
  - C2p (variant of C2);
  - C3 *Wörterbuch:\n X, mit* + *dem/der*;
  - C4 the dictionary frame with contextual calibration.

  C4 was adopted as primary and C1 as secondary for all models (decision 2026-10-09).
- A failing frame is never replaced post hoc; the model gets "no valid test" for that language.

### 2.6 Direction estimation

#### 2.6.1 Estimators

- **Difference of means:** v = μ_f − μ_m over training nouns.
- **Logistic probe:** standardised features, L2 penalty, C = 1, not tuned. It is fitted in the row space of the training data (X = USVᵀ, fit on US, map back by V), which gives the same optimum in an n-dimensional problem. The probe was selected once, before any 8B/14B data, by a pre-registered rule applied to development-model AUCs (probe 0.70, shrinkage LDA 0.69, standardised DoM 0.665, raw DoM 0.65).
- Both estimators are reported; the probe is primary for Phase 2–3 tests, and stratified DoM is the steering vector in Phase 5.

#### 2.6.2 Residualisation

Before fitting, nuisance variables are projected out of the activations: X ↦ (I − ZZ⁺)X, where Z contains:
- cell indicator columns;
- Zipf;
- token count;
- concreteness;
- the Spanish *a-* indicator.

By Frisch–Waugh–Lovell, the direction can then use only within-cell variation orthogonal to the covariates. The covariates were chosen before gender results from the screening imbalance reports; additions follow the same rule and are logged.

#### 2.6.3 Unbiased geometry: split-half estimation

The plug-in squared norm of an estimated class-mean difference is biased upward: E‖v̂‖² = ‖v‖² + tr Σ (1/n_g + 1/n_ref). With unequal class sizes, this bias mimics markedness. All lengths, markedness contrasts and cosines are therefore estimated split-half:
- nouns are split into halves A and B, stratified by cell;
- E[v̂_A · v̂_B] = ‖v‖² is unbiased;
- cosines are cross-fitted from such terms;
- results are averaged over 200 random splits, computed exactly through the Gram matrix, since every v̂ is a linear map of X.

Under the bootstrap, duplicated items are constrained to fall in the same half; otherwise shared noise reintroduces the bias.

#### 2.6.4 Inference

- **Bootstrap:** 1,000 resamples, stratified within cells. The direction is refitted in each resample. Test nouns are resampled too where the statistic depends on them.
- **AUCs** are computed within the test language's cells and pooled over within-cell f–m comparisons.
- **Null:** within-cell label shuffle (cell × gender counts preserved). For split-half geometry, the null is applied to the cross term rather than the cosine, because a shuffled class vector has noise-corrected length ≈ 0 and its cosine is undefined (decision 2026-10-10).
- **Multiple comparisons:** co-primary tests use Holm via interval width. The stronger test must exclude chance at 97.5%; the second is then judged at 95%.

### 2.7 Phase 2–3 tests

- **Phase 2 (Spanish).** For each layer, the direction scores two contrasts:
  - masculine *-a* exceptions vs regular feminine *-a* nouns (A_f);
  - the exceptions vs regular masculine *-o* nouns (A_m).

  A layer is **gender** if A_f's 95% lower bound is > 0.5 and A_m's is not; **spelling** if the reverse; **mixed** if both; **neither** otherwise. Secondary tests: noun- vs verb-frame homograph sorting, *el/la mar*, feminine *-o* exceptions.
- **Phase 3 (German).** On the 219 conflict compounds, does the direction sort by head gender or by first-constituent gender? Each layer is classified by whether the 95% interval of each AUC excludes 0.5. Markedness (masculine mean closer to the grand mean than feminine) is tested with split-half lengths against the shuffle null.
- **Verdicts** are counts of layers with each label. The pre-registered summary is the majority label across inner layers, reported at LAST and AFTER.

### 2.8 Phase 4: cross-language tests (8B, 14B; LAST primary)

- **T1, transfer (co-primary).** The Spanish direction scores German `strat3` m/f nouns, and the German direction scores Spanish `strat` nouns. Scoring uses within-cell AUC on nouns held out from the direction under test. "Transfers" means the lower bound is > 0.5 in a majority of inner layers. T1 requires both directions to transfer.
- **T2, geometry (descriptive).** Split-half cosine between the Spanish and German directions, against the shuffle null.
- **T3, flipped pairs (co-primary).** A single direction (Spanish, German or pooled) scores *both* nouns of each pair. The statistic is the paired AUC and the share of pairs in which score(de) − score(es) has the sign of gender(de) − gender(es). Bootstrap over pairs. Same-language scoring would replicate Phases 2–3 without testing sharing, so it is reported descriptively only. Run with and without cognates.
- **T4, Russian (secondary).** Spanish, German and pooled directions score the soft-sign set, with AUC within cells.
- **Robustness.** All tests were rerun at AFTER, and with the German frequency direction projected out (§2.11.2, B).

### 2.9 Phase 5: steering and social readouts

#### 2.9.1 Intervention

- At layer ℓ and the noun's token positions, we add α·u·n̄_ℓ to the residual stream, where:
  - u is the unit grammatical direction (stratified DoM);
  - n̄_ℓ is the median residual norm at noun positions in layer ℓ.
- **Layers:** four pre-registered layers per model, at ~25/40/55/70% depth (8B: 9, 14, 20, 25; 14B: 10, 16, 22, 28).
- **Steering nouns:** ~360 per model (Spanish training and pair nouns known to the model).

#### 2.9.2 Gate (dose selection)

- α is swept over 0.05–3 in the gate frame, with each noun pushed toward the opposite gender.
- α\* is the smallest α at which the median noun of *both* genders flips agreement.
- **Specificity:** 20 random directions of equal norm must flip < 10% of nouns at α\*.
- Layers without a valid α\* drop out. Readouts are evaluated at +α\* (the "working dose"). A descriptive dose–response slope is fitted over the doses whose median damage (below) does not exceed the random directions' median damage at α\*.

#### 2.9.3 Readouts

| Readout | Prompt (example noun *puente*) | Scored vocabulary |
|---|---|---|
| R1 | *The Spanish word "puente" means bridge. Described in one word, it is very* | 656 Glasgow adjectives |
| R1-EN | *The bridge is very* (push on the English noun's tokens) | same |
| R2 | *If the bridge (puente) were a person, would it be a man or a woman? It would be a* | *woman/man*, *female/male*, *girl/boy*, both orders |
| R3 | *Mi puente es muy* | Spanish invariant adjectives |

Three wordings per co-primary readout were averaged (P18). Wordings failing the readout check (scored vocabulary < 5% of next-token mass) are excluded by rule.

#### 2.9.4 Bleed score

For each adjective j, Δ_j = log P(j | steered) − log P(j | unsteered). We fit

Δ_j = b₀ + b_g·gender_j + b_v·valence_j + b_a·arousal_j + b_s·size_j + b_f·Zipf_j + ε,

and b_g, oriented so that positive means toward feminine-rated words, is the **bleed score**. Every noun is pushed toward feminine (+α\* along u, whatever its own gender), and the sign is flipped so that b_g > 0 means a shift toward feminine-rated adjectives (Glasgow GEND runs 1 = feminine to 7 = masculine). Row scores are averaged per noun over the primary wordings, then over nouns.

#### 2.9.5 Comparison vectors and decision rule

At the same norm, the bleed score is compared with:
- 100 random directions (the working-dose 95th percentile), and 20 for slopes;
- a number direction (singular − plural, the same training nouns);
- a social-gender direction (positive control: DoM over 9 fixed English female/male person pairs, WordNet-checked);
- for exploratory P26, the other language's grammatical direction.

R1-EN additionally uses damage matching: random directions are rescaled to match the KL divergence the gender push causes, measured as mean KL(unsteered ‖ steered) over the 20 positions after the noun in a fixed neutral sentence per language (English: *Yesterday we talked about the N over dinner with some family friends, and then we drove home.*; Spanish and German equivalents in `phase5.NEUTRAL`).

A layer **passes** if the real direction's mean noun score beats the 100 random directions (one-sided rank test toward feminine) and the noun-bootstrap interval of (real − mean random) excludes 0. A readout **passes** at a majority of the valid layers. R1 and R1-EN are co-primary (Holm). The verdict labels are:

| Verdict | Condition |
|---|---|
| bleed | Spanish and German directions both pass, with the same sign |
| language-specific | only one passes |
| no effect | neither beats random, while the social control moves the readout |

R2 is secondary because the noun is the queried entity, so grammatical agreement and social meaning are confounded in that prompt. R3 is secondary.

### 2.10 Unsteered article experiments (P29–P31)

- **P29, pooling neutrality.** For gendered Spanish adjectives read as P(m form) + P(f form) (R3-POOL/P20-POOL), we tested whether the pooled readout is invariant to an article-only flip (*el → la*) with no steering. The pre-registered failure condition was an article-only shift at least as large as the steered effect.
- **P30 (Spanish) and P31 (German).** Each nonce noun appears in each context with each article (German also *das*):

| Context | Spanish | German |
|---|---|---|
| O1 object | *La N que compré ayer es muy* | *Die N, die ich gestern gekauft habe, ist sehr* |
| O2 object | *La N que guardé en el cajón es muy* | *Die N, die ich in die Schublade gelegt habe, ist sehr* |
| N1 neutral | *La N es muy* | *Die N ist sehr* |
| P1 person | *La N con quien hablé ayer es muy* | *Die N, mit der ich gestern gesprochen habe, ist sehr* |
| P2 person | *La N que conocí ayer es muy* | *Die N, die ich gestern kennengelernt habe, ist sehr* |

  - S(context) is the bleed-score regression (§2.9.4) applied to Δ_j = log P(j | *la*/*die*) − log P(j | *el*/*der*). The vocabulary is the invariant adjectives (Spanish) or all single-first-token Glasgow translations (German, where predicate adjectives do not inflect; read at the first token after *sehr*).
  - Bootstrap over nouns.
  - Real article-alternating person nouns (*el/la presidente*; 41 German adjectivally declined nouns) give a social yardstick.
- **Checks.**
  - **C1 (reading manipulation).** In the same frame with the adjective slot replaced by *es …*/*ist …*, compare log P(person word) − log P(thing word). Spanish uses *una persona* vs *una cosa*. German uses *Mensch, Person, Mann, Frau* vs *Ding, Gegenstand, Objekt, Sache* (log-sum over each set).
    - Object wordings must favour "thing" on the median noun and by ≥ 1 nat more than N1.
    - Person wordings must favour "person" by ≥ 1 nat more than N1.
    - Failing wordings are dropped; no surviving object wording means "no valid test".
  - **C2 (coverage).** Scored vocabulary ≥ 5% of next-token mass.
- **Verdicts.** **Pull survives an object reading** if the 95% interval of S(object) is above 0. **Mostly the "a woman" inference** if its upper bound is below ½·S(neutral). Descriptive only; outside the Holm family.

### 2.11 Supplementary controls

#### 2.11.1 Confidence/flattening control (`confidence-ctl-prereg.md`)

- **(a)** Add the midpoint log-probability ½[log P(j|A) + log P(j|B)] as a covariate in every bleed regression (Phase 5, P29–P31). The midpoint is used rather than the baseline, to avoid regression-to-the-mean artefacts.
- **(b)** Spearman correlation of Δ_j with the gender rating.
- **(c)** Cross-fitted cosine between the f−m coefficient and the Zipf coefficient in a joint regression of activations on gender and Zipf (plus cell covariates), at LAST and the four Phase 5 layers. **Overlap** means the cross term exceeds the 95th percentile of the within-cell × gender permutation null *and* |cos| ≥ 0.3.

#### 2.11.2 German frequency axis (`frequency-axis-prereg.md`)

- **A:** cubic Zipf + token-count indicators + Zipf × tokens.
- **B:** project the frequency axis f̂ out of all activations, then rerun Phase 4. Robust if T1 and T3 verdicts are unchanged and the T2 cosine moves by < 0.05.
- **C:** f/m pairs matched within cells at |ΔZipf| ≤ 0.3 with no regression; fewer than 10 pairs means "too few".
- **D1:** drop the complex `strat3` nouns (by rule, §2.13). **D2:** add a complexity covariate instead.
- For A and D, the overlap is "explained" if it disappears at a majority of the layers where (c) found it.

#### 2.11.3 Register (`register-axis-prereg.md`)

Age-of-acquisition norms for the English gloss (80 rated German nouns) are added as a covariate to (c). The primary statistic is E2, the change in cross-fitted cos(gender, frequency).

#### 2.11.4 SAE features (G2; `gemma-slim-prereg.md`)

On Qwen3-8B (Qwen-Scope residual SAEs at layers 8/13/19/24, read at hidden_states[L+1] after an offset test), the overlap is computed on the 632 held-out compounds. The top-20 features most aligned with frequency are then ablated. Pass requires all of:
- |cos| with frequency at least halves;
- gender decoding drops < 5 points;
- the ablation beats 95% of random feature sets.

A precondition is that the held-out overlap itself meets the |cos| ≥ 0.3 bar; otherwise the result is "no valid test".

### 2.12 English social-gender analyses (P22b–P27; exploratory)

- **P22b.** The difference in log P(adjective) between *The woman / The man is very*, and between *My mother / My father is very*, gives each adjective's in-context woman/man lean. Its Spearman ρ with the Glasgow gender rating is reported raw and partialled on valence. The same is done for German (*Die Frau / Der Mann ist sehr*) and Spanish translations, using the English ratings.
- **P23–P24.** Grammatical directions (Spanish, German, pooled) score the English person words, names (within-final-letter AUC) and female/male word pairs. Split-half cosine with the social direction is computed, against the shuffle null and a number-direction reference.
- **P27.** Grammatical and social directions score the neuter person diminutives:
  - female vs male persons;
  - persons vs inanimate diminutives (the person effect);
  - inanimate diminutives by stem gender (the stem effect).

### 2.13 Known issues and deviations

- **German training-list complexity imbalance.** 12 of the 64 masculine `strat3` nouns are complex: numerals (*Vierer*) and compounds with deverbal *-er* heads (*Wasserkocher*). None of the feminine nouns are. Cause: the simplex filter checked only whole-word agent nouns. The list is frozen; the impact is tested by D (§2.11.2).
- **Prompt BOS.** Phase 5–style prompts (Phase 5, P29–P31) were built without special tokens.
  - This is correct for Qwen3, which has no BOS.
  - It was caught for Gemma before any measured run: one `<bos>` is now added, and G0 checks this path.
  - EuroLLM's P30/P31 prompts lacked its default `<s>`. A labelled sensitivity rerun with `<s>` (`GBLEED_BOS_ALL=1`) is reported alongside the original (§3.6). EuroLLM's contribution to the 2026-10-09 German gate-frame selection carries the same caveat.
  - Known-check scoring and activation extraction used each model's BOS convention throughout.
- **Not run.** Phase 0 was not part of the original Gemma driver (added before measured runs). P29–P31 and Russian are out of scope for the Gemma slim replication.

---

## 3. Results

Unless stated otherwise: LAST position, inner layers, 95% stratified bootstrap intervals, verdicts by the rules of §2.7–2.10. Layer counts for co-primary tests are Holm-corrected. Confirmatory models are Qwen3-8B and 14B (bf16, pod); smaller sizes and EuroLLM are development or exploratory.

### 3.1 Known-check and spelling erasure (Phases 0–1)

**Table 1.** Phase 0 balanced known rate (257 nouns per language).

| Model | German | Spanish |
|---|---|---|
| Qwen3-0.6B | 0.89 | 0.89 |
| Qwen3-1.7B | 0.90 | 0.92 |
| Qwen3-4B | 0.94 | 0.94 |
| Qwen3-8B | 0.97 | 0.94 |
| Qwen3-14B | 0.95 | 0.95 |
| Qwen3-30B-A3B | 0.98 | 0.97 |
| EuroLLM-1.7B | 0.97 | 0.98 |

- Confidently wrong items are ≤ 1.2% for every model; nearly all non-known items are "unsure".
- Mac (fp16) and pod (bf16) runs of the same model differ by ≤ 0.01.
- The few-shot quiz was order-sensitive: reordering changed 20% (DE) and 45% (ES) of 0.6B's answers, so it is diagnostic only.
- **LEACE erasure**, fitted on Spanish verb person endings (*hablo/habla*), did not erase the ending as a form. Nonce *-o/-a* words stayed linearly separable after erasure (accuracy 0.99–1.00 at every size), and erasure on held-out verbs was partial (AUC 0.68–0.81). Spelling is therefore controlled by stimulus design (§2.3) rather than by erasure.

### 3.2 Grammatical gender directions (Phases 2–3)

**Table 2.** Layers at which the stratified direction reads gender rather than spelling (Spanish masculine *-a* exceptions; §2.7).

| Model | LAST | AFTER |
|---|---|---|
| 0.6B | 24/27 | 0/27 |
| 1.7B | 22/27 | 2/27 |
| 4B | 26/35 | 0/35 |
| 8B | 28/35 | 3/35 |
| 14B | 26/39 | 0/39 |
| 30B-A3B | 45/47 | 38/47 |

- **German.** On the 219 conflict compounds, the direction trained on single-root nouns sorts by head gender at every layer of every size (AUC 0.76–0.87). It sorts by first-constituent gender at no layer.
- **Markedness** is supported at 0 layers by the split-half estimator. The plug-in estimator supported it at 25/27 (1.7B) and 31/35 (4B) layers, which is an artefact of the bias in §2.6.3.
- **Position.** Both gender effects are LAST-position effects and largely vanish at AFTER, except in the 30B-A3B mixture-of-experts model.

### 3.3 Cross-language sharing (Phase 4)

**Table 3.** Phase 4 tests (271 flipped pairs; layer-averaged AUC and cosine).

| Test | 8B | 14B | Verdict |
|---|---|---|---|
| T1 ES→DE | 6/35 layers | 7/39 | fails |
| T1 DE→ES | 5/35 | 22/39 | 14B only |
| T2 cosine (split-half) | 0.43 | 0.35 | above shuffle null |
| T3 flipped pairs, ES / DE direction | 0.93 / 0.83 | 0.91 / 0.89 | passes, all layers |
| T4 Russian (ES direction) | 0.82 | 0.85 | passes |

- T1 requires transfer in both directions and fails on both models.
- T3 and T4 pass. The reading is **partial sharing**: a common component sorts flipped pairs by each language's own gender and generalises to Russian, but does not give full Spanish↔German transfer.
- Every test fails at AFTER.
- Projecting out the German frequency axis (§2.11.2, B) leaves all T1/T3 verdicts unchanged, and T2 moves by < 0.05. An apparent DE→ES transfer gain under projection was traced to a change in sample composition, and the claim was retracted.

### 3.4 Steering (Phase 5)

**German: no valid test.**
- 8B: frame C4 failed its frame check (feminine 0.62 < 0.70).
- 14B: C4 passed the frame check but failed the gate at every layer. Feminine medians flipped while masculine medians did not, so the push collapsed both genders toward zero rather than swapping them.
- By rule, no substitute frame was used. All Phase 5 verdicts are Spanish-only and therefore at most "language-specific".

**Table 4.** Spanish R1 bleed score at α\*, averaged over passing layers (noun-position push).

| | 8B | 14B |
|---|---|---|
| Grammatical push | +0.020 (3/4 layers) | +0.000 (0/3) |
| Random, 95th pct | 0.012 | 0.008 |
| Number push | −0.003 | +0.002 |
| Social push (positive control) | +0.064 | +0.047 |
| R1-EN, layers passing | 3/4 | 0/3 |

- **8B:** R1 and R1-EN pass at a majority of layers. The verdict is **language-specific** (Spanish only). The effect is about one third of the social control's.
- **14B:** **no effect.** The positive control moves the readout, so the null is informative.
- **P26 (exploratory, 14B).** The German direction applied to the Spanish nouns moves R1 above random at 2 of 3 layers (k10, k16). The 14B null therefore holds for the Spanish direction only.
- **P14, all-positions steering (secondary; vector-level overlap).** The same vector added at every prompt position, readout included. 8B: R1 passes at 3/4 layers (effect +0.065, about three times the noun-position effect), R1-EN at 2/4 (+0.034; not a majority). 14B: 0/3 for both (R1 +0.001). The confound specific to P14 (the push acts directly on next-token prediction) is addressed by the same random- and number-direction comparisons; the number direction moves R1 by −0.004 on 8B.
- **R2** passes at every layer (8B 4/4, 14B 3/3). It is non-diagnostic because agreement and social meaning are confounded in that prompt (§2.9.5).
- **R3** had no valid test on 8B: its readout check failed (median share 0.029 < 0.05).
- **R3-POOL and P20-POOL** are not evidence. P29 found an article-only shift in the pooled readout of +0.096 (8B) and +0.038 (14B), against failure bounds of 0.026 and 0.019. Agreement leaks into the pooled readout.
- **Confidence control (a)/(b)** changes no Phase 5 verdict and no P29–P31 verdict.

### 3.5 English social gender (exploratory)

- **P22b.** The in-context woman/man lean correlates with the Glasgow gender rating:
  - English ρ = 0.51–0.59 across sizes (partial on valence: similar);
  - German 0.28–0.43;
  - in Spanish (4B), the lean is largely explained by valence.
- **P23/P24, Table 5.** Grammatical directions on English person words and names (AUC; layers passing in parentheses).

| | 8B | 14B |
|---|---|---|
| ES direction, person words | 0.87 (31) | 0.85 (35) |
| DE direction, person words | 0.67 (9) | 0.72 (20) |
| ES direction, names (within final letter) | 0.75 (31) | 0.70 (34) |
| Social direction, person words | 0.93 | 0.85 |
| Split-half cos with social: ES / DE / number | 0.44 / 0.18 / −0.01 | 0.22 / 0.08 / 0.00 |

  The grammatical–social overlap roughly halves from 8B to 14B, in line with the steering null at 14B.
- **P27 (1.7B, 4B).** The German grammatical direction does not separate female from male neuter diminutives (AUC 0.38–0.39; no layer passes). It does separate persons from inanimate diminutives. The social direction separates female from male diminutives (0.89–0.95).

### 3.6 Unsteered article pull (P30, P31)

**Table 6.** P30 (Spanish): S by context. All models pass C1/C2 on at least one object wording; the *que conocí* person wording failed C1 on most models.

| Model | Object | Neutral | Person | Real person nouns | Verdict |
|---|---|---|---|---|---|
| Qwen3-1.7B | +0.038 | +0.009 | +0.025 | +0.043 | survives |
| Qwen3-4B | +0.026 | +0.019 | +0.065 | +0.066 | survives |
| Qwen3-8B | +0.080 | +0.020 | +0.152 | +0.130 | survives |
| Qwen3-14B | +0.124 | +0.068 | +0.186 | +0.192 | survives |
| EuroLLM-1.7B | +0.008 (with `<s>`: +0.009) | −0.000 (−0.002) | – | +0.052 (−0.000) | no neutral pull |

**Table 7.** P31 (German, *die* − *der*).

| Model | Verdict | S(object) | Object / neutral |
|---|---|---|---|
| Qwen3-1.7B | survives | 0.035 | 1.95 |
| Qwen3-4B | survives | 0.043 | 2.14 |
| Qwen3-8B | no neutral pull | 0.002 | (neutral 0.002; person 0.081) |
| Qwen3-14B | no valid test (C1) | 0.025 (exploratory, C1 waived) | 0.73 [0.63, 0.85] |
| EuroLLM-1.7B | survives | 0.081 (with `<s>`: 0.073) | 2.32 (with `<s>`: 1.29 [1.22, 1.37]) |

- *das* − *der* object effects are ≈ 0 on Qwen3 (≤ 0.001) and +0.045 on EuroLLM (+0.026 with `<s>`).
- The cross-language pattern is model-specific:
  - large Qwen3: object pull in Spanish only;
  - small Qwen3: in both languages;
  - EuroLLM: in German only.
- EuroLLM's P30/P31 prompts lacked its default `<s>` (§2.13). With `<s>` the verdicts are unchanged, but:
  - in Spanish the real-person yardstick vanishes (+0.052 → −0.000), so EuroLLM's Spanish readout detects nothing;
  - in German the object pull falls to about the size of the neutral pull (ratio 1.29), and real person nouns now show the largest pull (0.074).

  The "EuroLLM: German only" contrast therefore rests on a German object pull that is real but not larger than the neutral one.
- The design rules out the "a woman" inference as the source of the object pull. It does not establish that the pull is social rather than distributional (article–adjective co-occurrence).

### 3.7 German gender–frequency alignment

- **Confidence control (c).** The cross-fitted cosine between the f−m and Zipf coefficients:
  - Spanish: |cos| ≤ 0.14 at all layers (no overlap);
  - German: overlap on both models, 8B −0.34 / −0.56 / −0.62 / −0.69 at the four Phase 5 layers, 14B −0.19 to −0.52.

  The sign means "feminine ↔ rarer". In both training lists the feminine nouns are in fact the *more* frequent ones within cells (Spanish d = +0.52, German +0.45). Across each lexicon the gender difference in frequency is small (Spanish +0.17 Zipf, German −0.05).
- **Follow-ups (Table 8).**

| Test | Result |
|---|---|
| A flexible frequency control | overlap persists |
| B axis projected out, Phase 4 rerun | all Phase 4 verdicts unchanged; T2 moves < 0.05 |
| C matched pairs (14 German pairs; Spanish 8, "too few") | overlap persists |
| D1/D2 complex nouns dropped / covariate | overlap persists |
| Register (AoA covariate, 80 nouns) | 8B −0.62 → −0.61 at 55% depth; 14B unchanged; E1 no AoA difference by gender |
| G2 SAE (Qwen3-8B, 632 held-out compounds) | no valid test: held-out overlap absent (cos +0.11 to +0.26) |

  Under A, C and D, the overlap disappears at ≤ 1 of 3–4 layers, never the majority required for "explained".
- **Status.** The alignment is robust in the German training sample of Qwen3 8B/14B and absent in the held-out compound sample. Its mechanism is unexplained. A pre-registered held-out test on new simplex German nouns (`rarity-lean-heldout-prereg.md`) is pending. The Gemma replication (R2, §3.9) finds no overlap on either size, so the alignment is Qwen-specific.

### 3.9 Replication in Gemma 3 (G1)

Gemma 3 4B and 12B pretrained were run in bf16 (L40S; commits 88cc069 and ecbbeea) on the frozen Qwen-derived lists, filtered by each list's own known-check. Known-check loss was ≤ 0.12 per gender per list, so every list is portable. Analysis code was unchanged except for wiring in the known filter (1cd5d6e), which was done before any Gemma analysis.

**Table 9.** Primary verdicts, LAST position.

| Verdict | Qwen3-8B | Qwen3-14B | Gemma-4B | Gemma-12B | Outcome |
|---|---|---|---|---|---|
| Compounds follow the head | 35/35 | 39/39 | 33/33 | 47/47 | replicates |
| Markedness | not supported | not supported | not supported | not supported | replicates |
| T1 transfer (ES→DE, DE→ES; Holm) | no (6, 5/35) | no (7, 22/39) | no (7, 26/33) | no (11, 27/47) | replicates |
| T3 pairs follow gender | yes | yes | yes | yes | replicates |
| T2 cosine above null | 0 | 0 | 0 | 0 | replicates |
| Phase 5 ES noun-position R1 | 3/4 (+0.020) | 0/3 | 0/3 (−0.004) | 1/3 (+0.029) | 8B positive not replicated |
| Phase 5 ES all-position R1 (P14) | 3/4 (+0.065) | 0/3 | 0/3 (+0.035) | 0/3 (+0.100) | 8B positive not replicated |
| Phase 5 DE | no valid test | no valid test | no valid test (C4 m 0.67) | no valid test (gate: random flip ≤ 0.50) | – |
| R2 German f−m × Zipf overlap | 4/4 | 3/4 | 1/4 (opposite sign) | 0/4 | Qwen-specific |

- **Phase 5 controls.** The social direction moves R1 by +0.29 / +0.34, so the readout is sensitive. On 12B the all-position effect (+0.100) does not exceed the random directions at any layer, and the number direction is +0.006. The noun-sense sensitivity check changes 0/8 verdicts.
- **AFTER position.** Compounds "neither" (4B 6/33, 12B 18/47 layers), as for Qwen3-8B (7/35). Phase 4 at AFTER is pending for 4B and not run for 12B, because the Spanish AFTER activations were not extracted on the pod.
- **Outlier dimension (amendment; `docs/design/gemma-outlier-dimension.md`).** A single residual channel (index 443 in 4B, 2339 in 12B) carries 59–92% of the variance across nouns at the tested layers; for Qwen3 the figure is ≤ 1.6%. On raw activations the cross-fitted cosines left [−1, 1] at about half the layers. After outcome-blind diagnostics, we pre-registered two things: zero any dimension above 10% of variance (1 per Gemma model, 0 per Qwen), and a validity gate that counts invalid layers against overlap. Under the fix all layers are valid. Qwen verdicts reproduce, with cosines within about 0.03 of the originals. T2 means are 0.47 (4B) and 0.55 (12B), against 1.13 and 0.61 raw. Raw-activation gender AUC on Gemma training nouns rises from about 0.58 to about 0.70 once the channel is removed. Raw Gemma probes therefore understate decodability; the Phase 3–4 verdicts replicated despite this.
- **Not run.** Phase 2, P29–P31 and Russian were out of scope. Frequency-axis A/C/D were not run, because no German overlap was found on Gemma.

### 3.8 Summary of claims

| Claim | Evidence | Status |
|---|---|---|
| Directions encode grammatical gender, not spelling | Tables 2, 3; compound conflicts | strong |
| Encoded at the noun's own token | LAST vs AFTER throughout | strong (scope condition) |
| Partly shared across Spanish and German | T2, T3, T4 pass; T1 fails | moderate |
| Steering moves social readouts (8B) | R1, R1-EN 3/4 layers | small; Spanish-only; survives confidence control; absent in Qwen3-14B and Gemma 3 4B/12B |
| No steering effect (14B) | R1 0/3, control works | null for the Spanish direction (P26 caveat) |
| Unsteered object pull (P30/P31) | Tables 6–7 | descriptive; not shown social; EuroLLM verdicts robust to `<s>`, sizes not |
| Pooled readouts | P29 | not evidence |
| German direction aligned with low frequency | §3.7, §3.9 | Qwen3 only (absent in Gemma 3 and held-out compounds); mechanism open |
| Representation results generalise across families | §3.9, Table 9 | replicate on Gemma 3 4B and 12B |
