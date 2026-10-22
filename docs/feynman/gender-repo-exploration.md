# Exploration brief: `~/GitHere/gender` — "Grammatical Gender Bleedthrough in LLMs"

**Prepared:** 2026-10-21 · **Method:** read of README, CLAUDE.md, docs/ (decisions, outline, methods summary, explainers 00–06, design docs), results summaries, git history (262 commits), and src/ maps. All numbers below are quoted from the repo's own documents; nothing re-computed.

---

## 1. What this project is (one paragraph)

Does an LLM's representation of **grammatical** gender — learned from inanimate nouns like
*el puente*/*die Brücke* where gender is pure grammar — overlap with its representation of
**social** gender (woman/man)? The AI version of the contested Boroditsky bridge effect, with a
causal twist: instead of correlating, you extract a gender direction from hidden states, push a
noun along it, and watch whether English adjective choices rated feminine by humans move. Model
family: **Qwen3 base** (0.6B–14B, plus 30B-A3B exploratory; EuroLLM-1.7B as a cross-family
check). Bulgarian levels of pre-registration discipline: gold labels from Wiktionary only,
measurement choices fixed before data, every decision dated in `docs/decisions.md`.

## 2. Where everything lives

```
~/GitHere/gender                     main checkout (branch main)
├── README.md                        status table, setup, pipeline, lexicon docs
├── CLAUDE.md                        working rules + current-state log (updated Oct 17)
├── Grammatical Gender Bleedthrough in LLMs Project Design.pdf
├── src/gbleed/                      cli.py, lexicon, stimuli, models, activations, scoring,
│                                   phase0–phase5, steering, estimators, sae, p22, p29, p30, p31…
├── tests/                           stimulus validation, heuristics, analysis-logic guards
├── data/stimuli, lexicon, raw       versioned stimulus CSVs; bulky data elsewhere
├── results/                         phase0–phase5 + side tests (small tables only)
├── results/runpod, runpod2/         two cloud-session summaries (README + subfolders)
├── docs/
│   ├── decisions.md                 THE decision log (141 lines, every choice + why)
│   ├── project-outline.md           chronological narrative (blog skeleton already exists)
│   ├── methods-summary.md           one-page methods
│   ├── explainers/00–06             per-phase method + results explainers (PI-facing)
│   ├── design/                      pre-registrations: phases 3–5, phase4, phase5 (P1–P27),
│   │   │                            pooling-neutrality (P29), p30, p31, suffix follow-up
│   ├── screening/, reports/         word-list audits and reviews (CSVs + markdown)
│   ├── writing tips not included: blog/series-outline.md, writeups/progress
├── runpod/                          pod scripts (session1/2, p29, p30, p31…)
├── sources/                         PDFs + SUMMARIES.md
└── worktrees elsewhere:             ../gender-pod (branch pod-analysis, session 1)
                                    ../gender-s2 (branch s2-analysis, session 2)
                                    ../gender-ru (branch russian)
External store (default /Volumes/Lexar/gbleed): activations, phase5 sweep arrays, kaikki dumps,
pod archives — via src/gbleed/paths.py (GBLEED_STORE; exFAT sidecar gotcha).
```

## 3. The design in brief

- **Phase 0 — do the models know the genders?** Sentence-scoring with each article in two fixed
  frames; "known" = both frames ≥ 1 nat toward the correct article. Qwen3 0.6B/1.7B/4B know
  ~89–94% of de/es nouns; EuroLLM-1.7B 97/98%.
- **Phase 1 — can spelling be erased?** LEACE eraser on -o/-a (fit on *verbs* where the ending
  marks person, not gender). Outcome (b): the eraser does not generalise to new words
  (nonce AUC stays ≈ 0.8). Consequence: **control spelling by design** (stratified, within-
  ending-cell comparisons) instead of by erasure.
- **Phase 2 — Spanish gender direction.** Directions = stratified difference of means or
  logistic probe, residualised for frequency/length/concreteness; readout at the noun's own
  last token (LAST primary, AFTER secondary, chosen by a gender-blind check — a flagship
  piece of discipline). Confirmatory 8B: stratified probe reads gender in **28/35** layers;
  14B **26/39**; **not** at AFTER (position-dependent).
- **Phase 3 — German, three genders.** Compound test: a direction trained on simplex nouns
  sorts unseen compounds by their **head**, even against conflict items (35/35, 39/39 layers).
  **Markedness: null** — the naive "masculine vector is shorter" result was pure estimator
  noise (see §5).
- **Phase 4 — shared across languages?** T3 flipped pairs: one direction scores each pair by
  *each language's own gender*, everywhere (paired AUC ≈ 0.9). T4: both directions sort
  **Russian** *-ь* nouns (spelling-controlled by construction) — the cleanest transfer
  evidence. T1 (strict within-ending transfer) fails; T2 overlap beats the shuffle null
  (split-half cosine ≈ 0.4). Verdict: "**partly shared**".
- **Phase 5 — does it bleed into social gender?** Steering: push *el puente* along feminine at
  dose α (measured in units of the state norm), gate = agreement flip, then score fixed
  adjective lists with Glasgow gender ratings, regressing the probability shift on the rating
  with valence/size/frequency as covariates. Controls: 100 random directions (rank p-value),
  damage-matched working window, number direction, social-gender positive control,
  cross-language push (P26). **Verdict: 8B language-specific positive (R1 +0.020 vs random 95%
  0.012; ~1/3 of the social push), 14B null (0.000; 0/3 layers).** German: no valid test on
  either size (pre-registered frame failed; PI declined to switch post hoc).

## 4. Results ledger (the write-up's spine)

| Item | Result | Where |
|---|---|---|
| Phase 0 known (de/es) | 89/89 · 90/92 · 94/94 (0.6/1.7/4B); EuroLLM 97/98 | explainer 00 |
| Phase 1 | outcome (b): eraser doesn't generalise (nonce AUC ≈ 0.8) | explainer 01 |
| Phase 2 confirmatory | gender read at LAST 8B 28/35, 14B 26/39; gone at AFTER | explainer 02 |
| Phase 3 | compounds follow head 35/35 · 39/39; markedness null | explainer 03 |
| Phase 4 | T3 pass everywhere (AUC ~0.9), T4 Russian ~0.8, T1 fail, T2 overlap ~0.4 | explainer 04 |
| Phase 5 R1 (ES) | 8B +0.020 (3/4 layers) vs random-95% 0.012; 14B 0.000 (0/3) | explainer 05 |
| Phase 5 social control | 8B 0.064 / 14B 0.047 (readout is sensitive) | explainer 05 |
| R2 man/woman | moves on both sizes (but noun is visible → "grammar → man/woman words") | explainer 05 |
| R3-POOL → P29 | pooled-agreement readout positive at both sizes… then **killed**: bare *La* vs *El* on meaningless nouns reproduces it (+0.096 8B / +0.038 14B ≥ steering effects) | explainer 05 |
| P20/P20-POOL | 8B +0.103 pooled; reinterpreted (same article-flip magnitude) | explainer 05 |
| **P30 (unsteered, object reading)** | *La siante que compré…* as a *thing*: Qwen3 8B +0.080, 14B +0.124 toward feminine-rated adjectives; **EuroLLM ≈ 0**; smaller at 1.7B +0.038 / 4B +0.026. "The closest thing yet to a direct, in-language yes" — mechanism (social vs text-class statistics) open | explainer 05 + outline §16 |
| P22b Glasgow sanity | English ρ 0.51–0.59 (validates EN ratings), German 0.28–0.43, Spanish invariant ≈ 0 after pleasantness (caveat that attaches to R3/P20) | explainer 06 |
| P24/P24b/P24c | ES direction (inanimate-only training) sorts English person words 8B 0.87 / 14B 0.85, names 0.75/0.70, female-male pairs 0.92/0.87; overlap with social cosine 0.44 (8B) → 0.22 (14B) | explainer 06 |
| P27 diminutives | grammatical directions read no sex in neuter person diminutives; social control near-perfect → informative null | explainer 06 |
| Costs | session 1 A100 ≈ $1.80; session 2 L40S ≈ $28.67 (~$17 idle); P29 $0.51; P30 $0.12 | outline |

## 5. Methods gold (write-up/blog material — all documented in your own repo)

1. **The estimator trap that would have confirmed markedness.** Plug-in length/cosine estimates
   are noise-biased; the naive test "confirmed" the hypothesis in 25/27 and 31/35 layers, the
   split-half estimator in **0**. Calibrated against a within-cell label-shuffle null.
2. **A test that couldn't fail.** The shuffle-null cosine exploded (>1) because shuffled-label
   directions have ~zero length; fixed by testing the split-half **cross term** (no division).
3. **Attention sinks:** single-token words at position 0 had one dimension ≈ 2,500 holding
   72–93% of variance; the first Phase 2 run was invalid and superseded — an input-format
   artifact demonstrated independently of gender.
4. **Recurring contamination sources, each caught:** pronoun confound in the homograph verb
   frame (*yo* vs *usted* = person = gender); English gloss mismatches (~10% of pairs); a
   training leak (37 flipped pairs); bootstrap duplicates sharing noise; the RAE filter
   artifact (*el agua*: word-initial *a-* = perfect masculine cue).
5. **The P29 episode** — a pre-registered check that sank two results (R3-POOL, P20-POOL) by
   showing a *bare article flip on meaningless nouns* reproduced them. This is the strongest
   "we killed our own result" story in the project, and exactly the kind of honesty that makes
   exploratory write-ups credible.

## 6. SAE state of play (good news for Phase 2 of our roadmap)

`src/gbleed/sae.py` (67 lines) **already loads Qwen-Scope SAEs** (`Qwen/SAE-Res-Qwen3-*`)
for 1.7B/8B/30B-A3B, with:
- `load/encode/decode` (TopK-50 residual-stream features, plain tensors),
- an FVU evaluator, and
- an **`alignment()`** function that empirically finds the correct hidden-state↔SAE-layer
  index mapping (SAE layer L = output of block L = `hidden_states[L+1]`) — precisely the
  bookkeeping that trips people up.

CMake sure to note: Qwen-Scope SAEs cover **Qwen3-8B-Base** (your core confirmatory model).
What's missing for the SAE phase: feature *scoring* on your templates (Δf = P(fire|fem noun) −
P(fire|masc noun), Qwen-Scope's own recipe), cross-lingual feature-sharing analysis (the
Brinkmann et al. overlap method), SAE-feature steering into the Phase 5 readouts, and the Gemma
side (Gemma Scope 2 is loadable via SAELens v6.24+ — your repo has no SAELens dependency yet,
but `sae.py` shows you don't need it: raw tensors suffice).

## 7. Write-up implications (checking against the deep dives in `outputs/paper-deep-dives.md`)

**Already strong (no new work needed to claim):**
- Phase 5 already implements most of the 2610.08559 control battery *by design*: dose sweeps,
  100-random-direction rank tests, matching disruption levels, a number-direction control, a
  social positive control, and probability-scored readouts. The steering claims are the most
  controlled in the literature you'll cite.
- Position robustness, from a gender-blind pre-registered decision — compare favorably with
  prompt-level work (e.g., 2407.09704, which never touches internal state).
- The exploratory framing is built into the repo: decisions.md timestamps everything,
  explainers separate pre-registered from exploratory ("interpretation, not a verdict").

**Gaps worth one cheap pass before or during write-up:**
1. **Confidence-direction check (2610.08559) for the steering/P30 results:** compute
   `cos(v_gender, v_confidence)` (highest-vs-lowest-probability completions on neutral text) in
   the same layers, and report answer entropy at α* vs baseline. P30 is unsteered so the
   confound is different there (already discussed as "la ⇒ woman?" + text-class statistics),
   but the steering results should show the confidence cosine to pre-empt the critique.
2. **P30 mechanism test** (already listed as next in your outline): the text-statistics reading
   vs. social-gender reading is *the* open question of the whole project; a corpus-statistic
   baseline (do Spanish feminine nouns as a class attract feminine-rated adjectives in *your
   model's* pretraining distribution, or in an n-gram model / smaller EuroLLM family?) would
   sharpen the claim.
3. **The stale README status table** says Phases 3–5 "not started" — update before sharing the
   repo with anyone.
4. **Model-family replication:** Gemma 3 is confirmed as the replication family; EuroLLM's
   P30 null and 8B-vs-14B divergence are your two best "what explains this?" hooks.

## 8. Suggested next steps (pick any)

1. **Draft the exploratory paper skeleton** with the claims-status table from the deep-dives
   doc, mapped onto your actual results (phases/paragraphs per result column in §4).
2. **Write the confidence-control addendum** (gap 1 above) — small, decisive, pre-empts review.
3. **SAE extension:** score Qwen-Scope features on the Phase 2/5 templates, compare the
   top-Δf features with your centroid/probe directions, then try feature-steering the Phase 5
   readouts on 8B (reuses `sae.py`, `steering.py`, `phase5.py`).
4. **Methods essay** (blog/paper): the split-half estimator + "test that couldn't fail" + P29
   are a cohesive "how measurement artifacts masquerade as Whorfian results" narrative.

## Sources (in-repo)
- `README.md`, `CLAUDE.md`
- `docs/project-outline.md` (chronological narrative, Oct 5–17)
- `docs/decisions.md` (decision log)
- `docs/methods-summary.md`
- `docs/explainers/00…06` (phase explainers)
- `docs/design/phase5-design.md`, `pooling-neutrality-prereg.md`, `p30-object-context-prereg.md`, `p31-german-object-context-prereg.md`
- `results/runpod/README.md`, `results/runpod2/README.md`
- `src/gbleed/sae.py` (existing Qwen-Scope loader)
- git history, 262 commits, branches: `main`, `batched-solver`, `phase5-build`, `pod-analysis`, `s2-analysis`, `russian`
