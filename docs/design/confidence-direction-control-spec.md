# Confidence-direction cosine & entropy: what it is, why it matters, and how to implement it

**Prepared:** 2026-10-21 · **Two audiences:** Part A is for the PI (concepts, no math beyond
what's explained); Part B is the implementation spec to hand to the coding agent (Claude Code in
`~/GitHere/gender`). The spec follows the repo's own rules (pre-registration, decisions.md,
tests, paths.py, Mac-dev → RunPod-confirmatory).

---

## TL;DR

We already suspect the steering results in Phase 5 might be explained in more than one way. A
2026 paper (Buttigieg et al., arXiv 2610.08559) showed that for **contrast-derived directions** in
LLM activation space, a large part of what looks like "the concept direction" is actually a
**confidence direction** — pointing from regions of high-probability to low-probability tokens.
Two cheap measurements tell us whether that's true of *our* gender direction:

1. **The confidence cosine:** build a direction `v_conf` that provably separates "expected"
   from "unexpected" completions on gender-irrelevant text, and report `cos(v_gender, v_conf)`
   per steering layer. If the cosine is large (comparable to the overlap we see between
   Spanish and German directions, ~0.4), part of our gender direction is "surprisingness", not
   gender, and the paper must say so.
2. **Entropy & rank checks at the readout:** report the entropy of the scored-adjective
   distribution at α=0 vs α* vs −α*, and whether steering *reorders* adjectives or just
   flattens the distribution (Kendall τ between steered and unsteered adjective rankings).
   Pure confidence effects flatten without reordering; a genuine gender shift should reorder
   the gender-polarized adjectives specifically.

Plus one regression tweak: add **baseline log-probability (α=0) as a covariate** in the
per-adjective readout regression — the single most direct defense against the
"flattening artifact" at the statistical level.

**Why for us specifically (and why it's not a debunking exercise):** our gender vector is
*not* derived from a bias-contrast the way the paper's was, so the worry is smaller — but real.
Feminine and masculine Spanish nouns in our templates can differ in frequency, length, or
predictability, which can leak a "rare vs. common noun" axis into the gender direction. Also,
steering at dose α inevitably perturbs the output distribution; if it mostly *flattens* it, the
gender coefficient could move for reasons that have nothing to do with gender. These checks
tell us *which* of the candidate explanations survives. Either outcome is publishable.

---

# Part A — For the PI (the concepts, in plain words)

## A1. What "a direction" and "a cosine" are

Think of a model's internal state at a layer as a **point in a very high-dimensional space**
(d ≈ 4,096 for Qwen3-8B). A "gender direction" is an arrow through that space: the average
"feminine noun" point minus the average "masculine noun" point. Pushing a noun's state along
that arrow is our steering.

A **cosine** measures how parallel two arrows are: 1 = exactly the same direction, 0 =
perpendicular, −1 = opposite. In 4,096 dimensions, two *random* arrows are almost always
nearly perpendicular (cosine ≈ 0.00–0.05), so a cosine of 0.3 is already notable, 0.4–0.7 is
strong overlap, and 0.9 is "essentially the same direction". *Cosines are comparable only
within the same layer and the same model.*

Reference cosines from your own results (Phase 4/6 explainers): Spanish×German gender
directions ≈ 0.3–0.44; Spanish×**social** direction ≈ 0.3 (8B) / 0.22 (14B); **number
direction × social** ≈ 0.01 — that last one is your "what unrelated looks like" baseline.

## A2. The confidence confound (what the 2026 paper found)

Buttigieg et al. (arXiv 2610.08559) trained 8 open-weight models (including Qwen3.5-9B — your
family) and asked: when a "bias direction" is computed the standard way (contrast biased vs.
anti-biased prompts), what does it *actually* separate? Two findings:

1. A classifier trained to separate "biased vs. anti-biased" hidden states also separates
   **high-probability vs. low-probability answers on completely unrelated, bias-free QA** with
   AUROC up to 0.95. I.e., the linear structure it found was mostly likelihood.
2. The contrast direction itself had cosine **0.29–0.91** (mean ≈ 0.5–0.7) with a
   **confidence direction** built from high-vs-low probability completions on gender-free
   datasets. And steering *along the "debiasing" direction* mostly didn't correct bias — it
   increased abstention and flattened answer probabilities, like temperature scaling.

The intuition: in a transformer, "how likely is this next token" is not stored in one place —
it's a real, geometrically organized property of the hidden state. Any contrast you build that
correlates with likelihood (and almost any meaningful contrast does a little) picks up a bit of
that axis. If the axis you *wanted* (gender) is smaller than the likelihood axis, you can be
measuring mostly likelihood without knowing it.

**Can this bite *us*?** Our gender direction comes from contrasting feminine vs. masculine
***inanimate nouns* in templates**. Two ways likelihood leaks in:
- feminine and masculine noun sets can differ in frequency, length, or rarity (a "rare vs.
  common noun" axis), and
- the template `La {noun}` vs `El {noun}` can make feminine-noun sentences more or less
  predictable on average.

That's a much weaker correlation than the paper's bias-contrast had — but it's nonzero, and
the whole point of your project is that you *check* things. The checks below are cheap.

## A3. The three measurements, in plain words

**① The confidence cosine.** Build a direction that provably encodes "expected vs. surprising",
with no gender in it: take ~150 neutral English sentence prefixes ("The capital of France
is ___"), run the model, take its top-ranked completion (high probability) and a low-ranked
one (low probability), extract the hidden states for both, and take the mean difference per
layer → `v_conf`. Then compute `cos(v_gender, v_conf)` at each steering layer, and compare it
with (a) the shuffle-null distribution (gender labels permuted → what chance looks like) and
(b) `cos(v_number, v_conf)` (a real direction that has no reason to be about confidence — the
"unrelated" baseline).

Reading the outcome:
- `cos(v_gender, v_conf)` ≈ 0.0–0.1 and below the number-direction baseline → clean; gender is
  not likelihood.
- `cos ≈ 0.3+` (comparable to the gender×social overlap) → part of our gender direction is
  "surprisingness"; we calibrate how much and say so. This doesn't kill the project — it makes
  the *quantitative* claim harder and the *qualitative* claim (grammar and social gender are
  entangled in these models) still supported.

**② Entropy & rank checks at the readout.** Confidence-flattening has a signature: the
distribution gets *flatter* (entropy goes up) but the **ranking of the options is mostly
preserved**. A genuine gender shift should **reorder** the feminine-rated adjectives relative to
masculine-rated ones. So, per sentence:
- entropy of the distribution over the scored adjectives at α=0, α*, −α*, and random-at-α*;
- **Kendall τ** between the α=0 adjective ranking and the α* ranking (and vs random);
- total probability mass on the scored list (you already track this in the readout check — now
  report it at every dose);

and a small worked intuition for the entropy: a distribution over 3 adjectives of
[0.90, 0.06, 0.04] has entropy ≈ 0.43 bits; [0.34, 0.33, 0.33] has ≈ 1.58 bits. Flattening
shows up as a jump toward the second number. (Shannon entropy = −Σ p·log₂p; 0 = all mass on one
word.)

**③ The regression tweak (the sharpest fix).** The readout regression is
Δlog p(a) = γ·gender_rating(a) + β·valence + … + ε. If steering *flattens* the distribution,
then Δlog p(a) ≈ −(log p₀(a) − mean), i.e., proportional to how far each adjective was from
its baseline probability — and *baseline log-probability correlates with frequency and with
rating* (rarer words have lower p and different ratings). Adding **log p₀(a) as a covariate**
absorbs exactly that channel. Report γ with and without it. If γ barely moves, the effect is
not a flattening artifact; if it collapses, we've found the mechanism.

## A4. What we will and won't claim (draft pre-registration language)

> *Pre-registered (before any confidence-ctl data): for each steering layer k, compute
> v_conf(k) from 150 gender-neutral English cloze items (fixed list) and report
> cos(v_gender(k), v_conf(k)) against a within-cell label-shuffle null and the number
> direction. If cos(v_gender, v_conf) exceeds the 95th percentile of the shuffle-null at a
> **majority** of steering layers, the Phase 5 steering verdicts for that model are reported
> with the caveat "consistent with a substantial confidence overlap". Independently, we
> report list entropy, list mass and Kendall τ (α=0 vs α*) for every readout × wording, and
> the readout regression with and without the baseline log-probability covariate. Both
> measurements are diagnostics; neither can *rescue* a null nor *by itself* invalidate a
> positive — the steering verdict rules from Phase 5 stand unchanged.*

That's the spirit. The coding agent writes the dated decisions.md entry in this style.

---

# Part B — Implementation spec (hand this to the coding agent)

## B0. Context & required reading

- Repo: `/Users/watcher/GitHere/gender` (branch `main`). Follow `CLAUDE.md` hard rules:
  pre-register before data; log every decision in `docs/decisions.md`; add terms to
  `docs/glossary.md`; commit before runs (`git_commit` in result meta); tests for builders;
  use `paths.py` for any bulky output (arrays → external store; only small tables in
  `results/`); one device + precision per model; Qwen3 has no BOS token; never measure a
  single-token item without the `<|endoftext|>\n ` prefix (attention sinks; `activations.py`
  has `prefix_ids` and `check_no_sink`).
- Papers: **Buttigieg et al. 2026, arXiv 2610.08559** — read §3 (setup), §4.2–4.3 (the
  classifier that transfers to confidence), §5 (CAA direction + cosine with confidence
  direction; their Eq. 4 is the mean-difference form we mimic).
- Existing code to extend: `src/gbleed/activations.py` (extraction conventions, layer indexing:
  k = 0 embeddings, k = 1..L−1 output of block k, LAST position), `src/gbleed/estimators.py`
  (`shuffle_within`, `boot_within`, `split_half_geometry` patterns), `src/gbleed/phase5.py`
  (steering layers at ~25/40/55/70% depth — compute the boundary index k the same way the gate
  does; readout regression in `phase5_analysis` — find the per-adjective regression and add the
  covariate), `src/gbleed/cli.py` (subcommand registration style), and `runpod/p30.sh` as the
  template for any pod run.
- Gender directions: already on disk from Phase 5 (`v_gender` per layer, per model). Number and
  social directions exist too (Phases 4–5). Reuse them; do not refit.

## B1. Stimulus design (fixed before any run — no tuning on model scores)

File: `data/stimuli/confidence_ctl_v1.csv`, columns: `item_id`, `prefix`, `hi`, `lo`, `source`.

- **150 items.** Prefixes are neutral English cloze sentences about facts/objects with a
  clearly expected completion: *"The capital of France is"*, *"A bicycle has two"*, *"Water
  boils at 100 degrees"*, *"The opposite of hot is"*. Rules: no gendered pronouns, no person
  nouns, no sex-associated words; each prefix must not end in a preposition/auxiliary that
  forces a specific closed class (avoid "the" alone). Build from a fixed list of ~10 templates
  × 15 fillers; define `hi` = the canonical expected completion (e.g., *Paris*), `lo` = a
  low-frequency English noun matched in length/register that is *wrong for the slot* (taken
  from the Wiktionary-derived lexicons or wordfreq, Zipf < 2, inanimate). The `hi`/`lo` pairs
  are fixed **before** extraction and never changed.
- **Gender-blind check:** assert `item_id`s contain none of the Phase 0–5 stimulus nouns (like
  Phase 1–5 builders do — add the assertion in `tests/`).
- The agent must verify that for at least ~120/150 items the model actually ranks `hi` above
  `lo` in probability at α=0 **on the development model only (Qwen3-1.7B)**; if fewer, expand
  items (still fixed list, new version v2) — do **not** curate per model.

## B2. Extraction (one module: `src/gbleed/confidence_ctl.py`)

- For each item, two forward passes: `prefix + hi` and `prefix + lo`. Read the hidden state at
  the **last token of the continuation** (`hi`/`lo`), every inner layer — same convention as
  `activations.py` (LAST). Continuations are multi-token as needed; keep the sink guard.
- Save: `results/confidence_ctl/<model_slug>/states.npz` (n_items × 2 × n_layers × d) or,
  better, per-condition means and a downsample for bootstrap. Arrays are small (150×2×~35×4096
  × 4 bytes ≈ 170 MB for 8B — beyond the 5 MB tracked limit; keep the npz on the store via
  `paths`, store only a small `summary.json` + table in `results/`).
- Models: dev **Qwen3-1.7B and 4B** (Mac, MPS fp16); confirmatory **8B** (and 14B if cheap)
  via a RunPod script modeled on `runpod/p30.sh` (bf16, one GPU type). Log `git_commit`.

## B3. Analysis (same module or `phase5_analysis` addendum)

Per model, per steering layer k:

1. `v_conf(k) = mean(h(hi)) − mean(h(lo))` (mean over items). Unit-normalize.
2. Cosines: `cos(v_gender(k), v_conf(k))`, `cos(v_number(k), v_conf(k))`, `cos(v_social(k),
   v_conf(k))`, plus the **within-cell label-shuffle null**: shuffle the `hi`/`lo` labels
   within items 500×, recompute `v_conf`, cosine with `v_gender` — report the 95th percentile.
   (Mirrors `estimators.shuffle_within`; the split-half cross-term trick isn't needed for a
   diagnostic cosine, but state that noise biases the *signed* cosine toward +, and the shuffle
   null absorbs that.)
3. Verdict per the pre-registered rule in §A4 (majority of steering layers above the null 95th
   percentile ⇒ "confidence overlap" caveat).
4. **Readout-side (Phase 5 data already on disk):** recompute, per readout × wording × noun,
   at doses {−α*, 0, +α*}: (i) entropy of the normalized scored-adjective distribution;
   (ii) total mass on the scored list; (iii) Kendall τ between the α=0 and α* **rankings** and
   between α=0 and each random-direction ranking; then aggregate: mean entropy and mean τ,
   real direction vs. the random-direction distribution (mean ± 95% band, 100 random
   directions). Do the same for P30's conditions (El vs La, object/neutral/person sentences)
   as optional-cheap.
5. **Regression covariate:** add `log p₀(a)` (per sentence: the α=0 probability of adjective a,
   which already exists in the Phase 5 score tables) as an extra covariate in the Phase 5
   per-adjective regression; re-report γ with vs. without it for R1/R1-EN at α* (8B, 14B).

## B4. Pre-registration & reporting requirements

- Before running anything: `docs/design/confidence-ctl-prereg.md` (what, why, the §A4 rule,
  fixed item rules, thresholds) + the dated entry in `docs/decisions.md`.
- After results: add new terms to `docs/glossary.md` (confidence direction, list entropy,
  Kendall τ); optionally a short section in `docs/explainers/05-phase5-steering.md`; write the
  result table to `results/confidence_ctl/` (small CSVs; npz on the store).
- Anything slightly off-spec, or any post-hoc exploration, must be labelled exploratory
  (project convention).

## B5. Deliverables checklist

- [ ] `data/stimuli/confidence_ctl_v1.csv` (+ gender-blind assertion test)
- [ ] `src/gbleed/confidence_ctl.py`; CLI subcommand `confidence-ctl`
- [ ] `tests/test_confidence_ctl.py` (builder invariants: 150 items, no phase nouns, hi≠lo,
      lo rare/inanimate; analysis smoke on a toy vector)
- [ ] Pre-registration doc + decisions.md entry (before data)
- [ ] Dev runs 1.7B/4B; then pod 8B (14B optional)
- [ ] `results/confidence_ctl/*` tables; explainer/glossary updates
- [ ] Reported to the PI as: per-layer cosine table vs null + number direction; entropy/tau
      table for the real vs random directions at α*; γ with/without `log p₀`.

## B6. How to read the results (PI cheat sheet)

| Pattern | Meaning | Action |
|---|---|---|
| cos with v_conf below null-95th AND below number-direction | gender ≠ confidence | report as a clean control; done |
| cos ≥ ~0.3, above null | substantial confidence overlap | report the number; state the steering claim with the calibration caveat; do not overclaim "gender-only" |
| entropy(α*) within random band AND τ ≈ random | effect is flattening-like even if γ > 0 | the γ-with-covariate comparison decides; if it holds, γ survives; if not, the γ was flattening |
| entropy(α*) near α=0 AND τ higher than random | steering reordered the list specifically | strong evidence the effect is content (gender), not confidence |
| γ unchanged by adding log p₀ | regression-level artifact ruled out | cleanest single sentence for the paper |
