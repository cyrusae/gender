# Suffix follow-up: pre-registration (adopted 2026-10-08, PI)

*Status: adopted 2026-10-08 (PI: D1–D4 below). Written after seeing the exploratory
1.7B/4B result (suffix_p ≈ 0.33) and before any data on the new test sets, which don't exist yet.
Relates to: Phase 3 pre-registration (`phases-3-5-plan.md`), explainer 03, `decisions.md`
2026-10-08.*

## What prompted it

A masculine/feminine direction trained on single-root German nouns places feminine-suffix nouns
(*-heit*, *-ung*, *-tät*, *-enz*…, all reliably feminine, all known to the models behaviourally)
only about a third of the way from the masculine to the feminine training mean (`suffix_p` 0.33
at 1.7B and 4B, in no layer reliably above 0.5). Yet the same direction sorts compounds by their
head, including suffix-headed ones (head AUC 0.81 with them, 0.79 without).

## The two readings to tell apart

1. **Offset, not failure.** `suffix_p` is a *position* measure: it asks where the nouns sit
   relative to the training means. Derived abstract nouns may differ from single-root nouns on the
   direction for reasons unrelated to gender (abstractness, length, being derived), which shifts
   all of them by the same amount. The direction could still *rank* them correctly by gender. The
   compound result hints at this: AUC is a ranking measure and ignores offsets.
2. **Two routes.** The model may encode gender for suffix nouns differently from single-root
   nouns (via the suffix's own representation), so a single-root direction carries little of it.

Right now these can't be separated, because the suffix test set is all feminine (3 masculine
nouns survived the filters), so there is nothing to rank against.

## Design

### New test set: derived nouns of every gender

Built with the existing Phase 3 filters and frozen before any extraction. **Version** `phase3_final_v4`:
v3 unchanged (same training sets, so the single-root direction is the one already analysed) plus
a new test set `suffix_ctrl`:

| gender | suffixes | in the German lexicon | loan status |
|---|---|---|---|
| masculine | *-ismus* (*Tourismus*); ~~*-ling*~~ (dropped, D2) | 414 | borrowed |
| neuter | *-tum* (*Eigentum*), *-ment* (*Instrument*) | 113 / 121 | *-tum* native, *-ment* borrowed |
| feminine (existing) | *-heit/-keit/-ung/-ei* · *-tät/-enz/-anz/-ik/-ur* | | native · borrowed |

Up to 20 known nouns per suffix (Zipf ≥ 2.5, known by 1.7B and 4B with the three-way frames),
plus top-ups for *-ung* and *-keit* (only 1 known noun each in v3). *-ling* will mostly be removed
by the animacy filter (*Lehrling*, "apprentice"). It's kept if ≥ 8 nouns survive and dropped
otherwise; the expected loss is stated in advance.

Native and borrowed suffixes exist in each gender, which matters because loanwords look different.
Comparisons are made **within loan status**: borrowed (*-ismus*, *-ment* vs *-tät/-enz/-anz/-ik/-ur*)
and native (*-tum* vs *-heit/-keit/-ung/-ei*).

### S1 (primary): does the single-root direction rank derived nouns by gender?

- **Measure:** the single-root m/f direction (unchanged, from `strat3`) scores every derived noun.
  AUC of feminine-suffix vs masculine-suffix nouns, computed within loan status and then averaged.
  The 95% CI comes from a bootstrap that resamples **suffixes, then nouns within suffixes**: the
  suffix is the real unit, since all nouns sharing a suffix share its spelling.
- **"Ranks correctly"** if the lower bound is > 0.5 in a majority of inner layers.
- **Spelling check (required for interpretation):** since suffix and gender coincide perfectly
  here, a gap could be spelling alone. The reference is the spread *between feminine suffixes*:
  how far apart two feminine suffix groups sit on the direction, with gender held constant. The
  gender gap (mean of masculine-suffix groups minus mean of feminine-suffix groups) is
  interpreted only if it exceeds the 95th percentile of the between-feminine-suffix gaps.
  Otherwise the result is "not separable from spelling".
- **Readings:** ranks correctly and passes the spelling check: reading 1 (offset). Fails: reading 2
  (two routes) is supported. Neuter suffixes are reported descriptively (where they fall between
  the masculine and feminine groups).

### S2 (secondary, descriptive): covariate-adjusted position

`suffix_p` recomputed after removing the covariate part (frequency, token count, concreteness)
from the test nouns, using the coefficients from the training fit. It is computed for feminine,
masculine and neuter suffix nouns alike. If all three shift by a similar amount, the 0.33 was an
offset. This is a cheap complement to S1 and can run on the existing data.

### S3 (exploratory, PI's question): a direction that has seen suffix nouns

This tests "train with suffix nouns included" without stacking the deck. Stacking the deck would
mean adding only feminine suffix nouns: the direction could then learn "derived abstract noun =
feminine". Here, derived nouns of **all three genders** go into training, so being derived
doesn't predict any gender.

- **Leave one suffix out:** train an m/f probe on single-root `strat3` nouns plus every derived
  noun except one suffix. Cells are (single-root ending cells) + (derived × loan status); a cell
  per suffix would remove all of the information.
- Score the held-out suffix as the share classified correctly at the training threshold, and as
  its position between the derived training means.
- **Generalising to a held-out suffix** requires something beyond that suffix's spelling, because
  the model never saw it in training. It is reported per suffix.
- **Cost check:** does the augmented direction still sort conflict compounds by head (AUC on
  compounds without a suffix head)? What is its cosine with the single-root direction (bootstrap
  CI)? A low cosine together with intact compound sorting would be another sign of two routes.
- **Exploratory only:** a positive result shows that some direction covers both kinds of noun,
  not that the model uses one.

### Models, positions, stages

- **Position:** LAST only. At AFTER, Phase 3 showed no gender information.
- **Exploratory (Mac):** 1.7B/4B. Extraction of `suffix_ctrl` takes a few minutes. S2 needs no new
  data and can run on the pod's activations for every size.
- **Confirmatory:** 8B/14B. `suffix_ctrl` extraction rides along with the next A100 round
  (Phase 4/5).

## Decisions for the PI

- **D1. Adopted (PI, 2026-10-08).** Adopt S1 as the confirmatory test, with the suffix-level bootstrap and the spelling
  check (recommended)? Or treat the whole follow-up as exploratory?
- **D2. Adopted (PI, 2026-10-08): drop *-ling* up front.** Of 133 masculine *-ling* nouns,
  only ~3 common inanimate ones survive the filters (*Frühling*, *Schilling*, *Riesling*), below
  the 8-noun threshold anyway. Consequence: *-ling* was the only native masculine suffix, so
  **S1's confirmatory comparison is within borrowed suffixes only** (*-ismus* vs
  *-tät/-enz/-anz/-ik/-ur*); native feminine suffixes (*-heit/-keit/-ung/-ei*) are placed
  against neuter *-tum* descriptively. (German has no other productive native masculine
  suffix without people: *-er* is agent nouns, already excluded.) Original question: *-ling*: keep it if ≥ 8 survive, or drop it up front (most are people)?
- **D3. Adopted (PI, 2026-10-08).** S3 as specified (derived nouns of all genders, leave one suffix out)?
- **D4. Adopted (PI, 2026-10-08)** (to run after the corrected Phase 3 reruns). Run S2 on the existing pod data now? It's descriptive and needs no new stimuli.

## Limits to state

The follow-up was designed after seeing the 0.33. The new test set and S1 are untouched by that,
but the question isn't. There are few suffix groups (≈ 9 feminine, 2 masculine, 2 neuter), so
the spelling check has little resolution. Suffixes differ in productivity and meaning (abstract
*-heit* vs concrete *-ment*), not only in gender.
