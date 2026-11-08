# Is the German "frequency" axis partly register? Pre-registration (ADOPTED, PI 2026-10-22)

Drafted 2026-10-22. Diagnostic, CPU only, existing activations (Qwen3 8B/14B, LAST, the four
Phase 5 steering layers). Follows `frequency-axis-prereg.md`: the German gender direction aligns
with low frequency (cos −0.3 to −0.7). That isn't explained by the frequency control, the
regression or compound nouns. Thread 3's noun list (exploratory) suggested the axis, beyond
Zipf, separates short concrete words from technical/written ones. This tests that idea with a
rule-built label instead of a reading of the list.

## The label: age of acquisition of the English gloss

- **Source:** Kuperman, Stadthagen-Gonzalez & Brysbaert (2012), age-of-acquisition ratings for
  30,121 English words (the same lab as the concreteness norms already used). Downloaded
  2026-10-22 via the Internet Archive copy of `crr.ugent.be/papers/AoA_ratings_Kuperman_et_al_BRM.zip`;
  converted to `data/raw/norms/kuperman_aoa/kuperman_aoa.csv` (gitignored, like the other raw
  norms).
- **Why AoA:** words learned late are mostly technical, learned or written vocabulary
  (*catheter*, *inscription*). Words learned early are everyday concrete ones (*juice*, *hill*).
  It's a standard, well-validated proxy for register/learnedness that doesn't come from a model.
- **Matching rule (fixed now):** the lexicon's `concept_en` gloss, lowercased. Exact match;
  otherwise, for a two-word gloss, its last word (the English head: *blood vessel* → *vessel*,
  *operating lever* → *lever*). Anything else is unrated and left out of the AoA analyses (count
  reported). Exact matching covers 70 of the 91 German and 137 of the 165 Spanish training
  nouns.
- **Limit:** the AoA of the English gloss is a proxy for the German word's register; it isn't
  the same thing.

## Tests

- **E1 (no model): do German feminine and masculine training nouns differ in AoA?** The mean AoA
  difference within ending cells, with a bootstrap CI. Spanish alongside. A lexical fact about
  our stimuli.
- **E2 (primary): does an AoA control remove the overlap?** On the AoA-rated subset only:
  - the cross-fitted cos(gender, frequency) without and with AoA as an extra covariate (as D2);
  - both fits on the same nouns, so the subset itself can't make a difference.
  - Same machinery and overlap rule as `frequency-axis-prereg.md` (200 splits; permutation
    null on the cross term; |cos| ≥ 0.3).
  - **Reading:** **"register (as AoA) explains it"** if, with AoA, the overlap disappears at a
    majority of the layers where the same-subset baseline shows it. Otherwise **"not register as
    measured by AoA"**.
- **E3: is the frequency axis aligned with an AoA axis?** The AoA coefficient of the same fit
  (with AoA in it). Cross-fitted cosines cos(frequency, AoA) and cos(gender, AoA), German and
  Spanish. Descriptive.
- **E4 (exploratory): does thread 3's "reads commoner than its Zipf" track late AoA?** Spearman
  between each German noun's thread-3 residual position and its AoA. Exploratory, because thread
  3's list has already been seen.

## Secondary: Wiktionary subject tags

Present for only 32 of 91 German nouns, and attached to any sense (*Winkel* "angle/corner" →
"mathematics"). Reported descriptively: the thread-3 position of tagged vs untagged nouns. Not
used for any reading.

## What this can and can't show

E2 can show that AoA *accounts for* the overlap. It can't show that the model represents
"register", only that the axis lines up with a register proxy. A null leaves register open,
since AoA is one proxy measured through English. A later SAE look (Qwen-Scope SAEs for 8B) could
suggest other properties to label, each of which would then need its own rule-built test.

Code: `src/gbleed/register_axis.py`. Results go to `results/register_axis/`. Committed before it's
run.

## Decisions (PI, 2026-10-22)

Adopted as written (AoA as the register proxy, the matching rule, E2 as primary).

## Results (2026-10-22; Qwen3 8B/14B pod activations; `results/register_axis/`)

**E1 (lexical, no model): no AoA difference by gender.** Feminine − masculine at equal ending:
German −0.27 years [−1.39, 0.71] (80 rated nouns: 70 exact, 10 by head word), Spanish −0.21
[−1.09, 0.65] (155 rated).

**E2 (primary): an AoA control doesn't remove the overlap.** Same 80 German nouns, cross-fitted
cos(gender, frequency), without → with AoA:

| layer | 8B | 14B |
|---|---|---|
| 25% | −0.31 → −0.30 | −0.22 → −0.22 |
| 40% | −0.55 → −0.53 | −0.29 → −0.27 |
| 55% | −0.62 → −0.61 | −0.35 → −0.34 |
| 70% | −0.74 → −0.72 | −0.38 → −0.37 |

Reading (as pre-registered): **not register as measured by AoA.** 8B loses the overlap call at 1
of 4 layers, only because −0.298 falls just under the 0.3 bar. 14B loses it at 0 of 2. Spanish shows no
overlap either way.

**E3:** in Spanish the AoA and frequency coefficients align (cos 0.39 on 8B; 0.35–0.80 on 14B),
as expected, since late-learned words are rarer. In German the AoA coefficient isn't
reproducible across halves (its cross-fitted squared length is ≤ 0), so its cosines are
undefined. With 80 nouns the model's German activations show no stable AoA direction to compare
with.

**E4 (exploratory): thread 3's positions don't track AoA** (Spearman −0.08 on 8B, −0.02 on 14B;
n = 80). Claude's "register" reading of the noun list isn't supported by the rule-built label.

**Where this leaves it:** the German gender–frequency overlap survives the frequency control,
matched pairs, the complex nouns and now AoA. It remains unexplained. Candidates not yet
tested: a register property AoA doesn't capture; properties of the German word itself rather
than its English gloss (the German word's own learnedness or domain); or a genuine German
feminine/low-frequency association in the model. The next step would be an SAE look (Qwen-Scope,
8B) to suggest candidate properties, then a rule-built test for each. A third language would
show whether the overlap is German-specific.
