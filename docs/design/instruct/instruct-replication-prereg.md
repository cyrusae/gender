# The base-model study on instruct-tuned models, stage A (no steering). Pre-registration (ADOPTED, PI 2026-10-26)

Phase 2 queue item 3 (`docs/decisions.md` 2026-10-26). Distinct from the `instruct/` character
study: same pipeline, frozen lists and verdict rules as the base-model study, run on each base
model's instruct sibling, so every verdict is compared like for like. **Stage A** is everything
except Phase 5 steering; steering (stage B) is decided afterwards by a rule fixed here.

## Rule exception

CLAUDE.md's "base (pretrained) models only" gets a scoped exception for this study (models listed
below, the procedures below). Everything else in the hard rules stands.

## Models (instruct sibling of each base model)

| base (phase 1) | instruct sibling | relation |
|---|---|---|
| google/gemma-3-4b-pt | google/gemma-3-4b-it | post-trained from the pt checkpoint |
| google/gemma-3-12b-pt | google/gemma-3-12b-it | post-trained from the pt checkpoint |
| Qwen/Qwen3-8B-Base | Qwen/Qwen3-8B | post-trained from Qwen3-8B-Base by strong-to-weak distillation (Qwen3 Technical Report, arXiv 2505.09388) |
| Qwen/Qwen3-14B-Base | Qwen/Qwen3-14B | same |

The Qwen3 8B/14B releases are hybrid thinking models; the chat formats below use the template's
`enable_thinking=False` mode (it inserts an empty `<think></think>` block before the assistant's
text; that block is part of the fixed scaffolding). bf16 on the pod (L40S), as phase 1; Gemma
with the BOS rule and the outlier-dimension fix as in `gemma-slim-prereg.md`.

## Input formats

1. **Raw (primary):** exactly the phase-1 inputs (`activations.INPUT_FORMAT`; Gemma with its BOS).
   LAST = the noun's last token; AFTER = the appended newline.
2. **A, assistant says the word (sensitivity, PI 2026-10-26):** chat template, user turn = a fixed
   neutral prompt, assistant turn = the noun, then the template's end-of-turn marker. LAST = the
   noun's last token; **AFTER = the end-of-turn token** (`<end_of_turn>` for Gemma, `<|im_end|>`
   for Qwen): the point where the assistant has finished saying the word.
   - **A-en (A's primary wording):** user turn *"Say one word."* for every language.
   - **A-en-something (sensitivity wording, PI 2026-10-26):** *"Say something."* for every
     language. It differs from A-lang only in language, and from A-en only in wording, so the two
     contrasts can be told apart.
   - **A-lang (A's sensitivity wording):** a noun-free "say something" in the noun's language, so
     the prompt carries no grammatical gender of its own: Spanish *Dime algo.* "Tell me
     something." (PI-checkable), German *Sag etwas.* "Say something." (claude-unverified),
     Russian *Скажи что-нибудь.* "Say something." (claude-unverified). A direct translation of
     "Say one word" was rejected: *una palabra* is feminine, *Wort*/*слово* neuter, so it would
     prime a gender right before the noun.
3. **U, user says the word (sensitivity):** user turn = the noun, assistant turn opened by the
   template's generation prompt. LAST = the noun's last token; AFTER = the next template token.

## Gates (per model, before analysis; same rules as phase 1)

- Phase 0 article-choice gate (raw), known-check filter by the existing rule (list loses > 30%
  of either gender = "not portable" for that list).
- Outlier-dimension check: top dimension's share of variance on `phase3_bare` per format; the
  amendment's fix (zero dimensions > 10%) applied wherever it qualifies.
- A format whose LAST token is not the noun's last token for some noun (template tokenisation
  merging across the boundary) drops those nouns, counted and reported.

## Analyses (CPU, unchanged code and rules)

Per model × format × position: Phase 3 (compounds follow the head; markedness), Phase 4 (T1, T3,
T2 with the validity gate), confidence control (c) German frequency cosine (R2 rule with the
validity gate), Russian direction R-A/R-B (raw only; the Russian list's chat versions added if
cheap). Phase 2 (Spanish gender vs spelling) on raw.

**Reading:** per verdict, **same** (instruct verdict = its base sibling's), **differs**, or
**no valid test**, tabulated base × instruct × format. The comparison is between conclusions, not
vectors (different checkpoints). A format effect is reported where raw and A/U verdicts differ.

## Stage B (steering) decision rule, fixed now

Phase 5 is run on an instruct model only if (1) its Spanish raw frame check and gate pass by the
phase-1 rules, and (2) the PI decides to before any steering number exists. Formats for steering:
raw only unless the PI adds A. Cost ~$3.40–6.90 per model (phase-1 timings).

## Cost and order

Pod: L40S secure ($1.09/h), image direct, no volume. Per model ~1 h (download, Phase 0, known
check, extraction × 3 formats (raw; A in three wordings; U)). Order: Gemma 4B-it first as a pilot (if any gate fails
unexpectedly, stop and report), then 12B-it, Qwen3-8B, Qwen3-14B. Estimated $5–6 total.

## Not decided here

- Instruct-model frame checks for German Phase 5 (only relevant to stage B).
- Whether to add the chat formats to Phase 0 (raw Phase 0 gates the noun lists).

## Decisions (PI 2026-10-26)

Adopted with *"Say something."* added as a third A wording (A-en stays primary: *"Say one
word."* matches a one-word assistant turn). RunPod credit at adoption: about $12.

## Amendment 1 (PI 2026-10-27): bootstrap intervals and clipped reporting for cross-fitted cosines

Written after the 4B-it frequency cosines (c) were seen (German +0.3 to +1.34, some out of range),
before any interval was computed. **Verdict rules are unchanged** (the validity gate still decides
"overlap"); this adds descriptive reporting only.

- **Bootstrap interval** for every cross-fitted cosine reported in this study (R2 frequency cosine;
  Phase 4 T2): 200 resamples of nouns within cell × gender; in each, the cosine from 50 item-aware
  half splits (copies of a resampled noun stay in one half, `estimators._halves_items`); the
  2.5th–97.5th percentiles. Seed 27. Applied to the base sibling too, so the comparison is like
  for like.
- **Clipped reporting:** a point estimate outside [−1, 1] is reported as "> 1 (estimator limit)"
  or "< −1 (estimator limit)" together with its interval, not dropped. A layer whose interval lies
  entirely above 0.8 (or below −0.8) is described as "nearly parallel"; descriptive only.
- Code: `src/gbleed/cosine_boot.py`; outputs `results/instruct_stageA/<format>/c_boot.csv`.

## Exploratory: knowledge-check flips (PI 2026-10-27)

Which nouns change known-check status from base to instruct (known ↔ conflict / wrong / unsure),
and how they differ (Zipf, loan, ending, token count, concreteness, gender, which frame, which
wrong article). Uses the existing `results/phase{2,3}_known` scores only. Labelled exploratory.

### Amendment 1 results: bootstrap intervals (2026-10-27)

German cos(f−m, Zipf) 95% bootstrap intervals (200 resamples × 50 item-aware splits; undefined
resamples in brackets). Qwen3-8B base: [−0.65, 0.06], **[−0.88, −0.27], [−0.96, −0.35], [−1.17,
−0.35]** (0–3). Qwen3-14B base: [−0.52, 0.00], **[−0.60, −0.01], [−0.74, −0.12], [−0.65, −0.17]**
(0–1). Gemma-4B base: [−0.15, 1.45] … [−0.62, 1.87] (6–23). Gemma-4B-it raw: [0.01, 4.2] …
[−0.51, 3.4] (26–44); A-en / A-something / U wider still (35–132 undefined); A-lang [−0.61, 1.37]
at its widest, centred near 0 (2–4). Spanish intervals contain 0 everywhere. No layer is "nearly
parallel". **Reading (descriptive):** the Qwen3 German alignment is well determined (interval
excludes 0 at 3/4 layers on both sizes); the Gemma-4B-it positive point estimates are not (the
verdict rule's "overlap" on raw/A-en/U rests on cosines whose intervals include 0). Note: a
bootstrap resample keeps ≈63% of distinct nouns, so intervals are somewhat conservative.
`results/instruct_stageA/*/c_boot.csv`.

### Exploratory: knowledge-check flips, Gemma-4B → 4B-it (2026-10-27)

German (4,605 nouns): 760 known → not known, 301 the reverse. Loss rises with rarity (Zipf
quartiles 30%, 26%, 19%, 9%) and token count (1 token 7% … 6+ tokens 32%); equal across genders
(f 21%, m 20%, n 21%). Wrong-article choices among lost nouns: masculine → *die* 276 / *das*
113; neuter → *die* 124 / *der* 134; feminine → *der* 86 / *das* 27. Both frames fail about
equally (58%, 63% below 1 nat). Margins are larger on average (median 3.97 vs 2.95) and more often
wrong (13% vs 7%): sharper, not uniformly less sure. Spanish (1,013): 82 lost, 52 gained; loss
by Zipf quartile 18%, 8%, 7%, 3%. `results/instruct_stageA/flips/`.

## Stage A results: Gemma-4B-it pilot (2026-10-27)

Pod run 2026-10-27 (L40S, bf16, commit 600748e, ~$0.15; archive `/Volumes/Lexar/gbleed/instruct-stageA/`,
sha256 d6666846…fabf4). Analyses on the Mac (`scripts/instruct_stageA_analysis.py`,
`results/instruct_stageA/<format>/`), 30 jobs, none failed. Gates: known-check loss es f 0.21 /
m 0.17, de f 0.22 / m 0.20 (all lists portable; base 4B: ≤ 0.11); 0 boundary merges in every
format; outlier dimension 443, as base (fix applied to T2 and the frequency cosine).

| verdict (LAST unless noted) | base 4B | raw | A-en | A-something | A-lang | U | same as base? |
|---|---|---|---|---|---|---|---|
| compounds follow the head (layers /33) | 33 | 32 | 31 | 23 | 33 | 33 | same (all formats) |
| markedness | 0 | 0 | 0 | 0 | 0 | 0 | same |
| T1 transfer ES→DE / DE→ES (Holm) | 7 / 26 | 1 / 25 | 1 / 2 | 2 / 2 | 7 / 17 | 5 / 20 | same (no) |
| T3 pairs follow gender | yes | yes | yes | yes | yes | yes | same |
| T4-ES on Russian | 30 | 31 | 31 | 31 | 31 | 31 | same |
| T2 cross term > null (fixed; descriptive) | 31 | 24 | 3 | 3 | 27 | 18 | — |
| T2 cosine, median valid layer | 0.47 | 0.50 | −0.17 | 0.00 | 0.56 | 0.43 | — |
| R2 German frequency overlap (rule) | no | overlap | overlap | no valid test | no | overlap | differs on raw/A-en/U; intervals include 0 (amendment 1) |
| AFTER: compounds follow head | 6 | 4 | 1 | 1 | 8 | 3 | same ("neither") |
| AFTER: T1 / T3 | fail / fail | fail / fail | fail / fail | fail / fail | fail / fail | fail / fail | same |

**Reading.** Every Phase 3–4 verdict is the same as the base sibling's in every format. The only
rule-level difference is R2 (German frequency overlap) on raw, A-en and U, and its bootstrap
intervals include 0 there (amendment 1), so it is not treated as a finding. **Exploratory:** with
an English instruction before the noun (A-en, A-something) the descriptive cross-language measures
collapse (DE→ES transfer 25 → 2 layers; T2 cross term 24 → 3; cosine ≈ 0), while within-language
gender (T3, T4) is intact; the in-language prompt (A-lang) and the user-turn format (U) look like
raw and base. Known-check flips: see "Exploratory" above. Next per the prereg: 12B-it and the Qwen3
instruct siblings (~$1 each), PI's call.
