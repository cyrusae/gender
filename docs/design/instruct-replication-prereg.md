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
