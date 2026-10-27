# The base-model study on instruct-tuned models, stage A (no steering). Pre-registration (DRAFT for PI sign-off, 2026-10-26)

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
check, extraction × 3 formats). Order: Gemma 4B-it first as a pilot (if any gate fails
unexpectedly, stop and report), then 12B-it, Qwen3-8B, Qwen3-14B. Estimated $5–6 total.

## Not decided here

- Instruct-model frame checks for German Phase 5 (only relevant to stage B).
- Whether to add the chat formats to Phase 0 (raw Phase 0 gates the noun lists).
