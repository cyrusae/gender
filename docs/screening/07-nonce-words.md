# 07 · Nonce (invented) words

*Code: `src/gbleed/phase1_stimuli.py` (`build_nonce`). File: `data/stimuli/phase1_nonce_v1.csv`.
Planned: Wuggy for Phase 5 (P15, adopted 2026-10-08).*

## Phase 1: Spanish-looking minimal pairs (frozen)

**Purpose:** test whether a spelling eraser removes the *-a*/*-o* ending in general, on words the
model can't know.

**How they're made:**
1. Stems are built from Spanish-plausible pieces: onset (*b, br, bl, cr, fl, tr*…) + vowel +
   optional coda (*l, n, r, s*) + a final consonant or cluster (*d, t, ch, ll, br, pl*…), with
   only consonant sequences that occur inside Spanish words (*al-ba*, *car-ta*, *is-la*).
   No final *m* (keeps the Greek *-ma* pattern out), no *c*/*g* (spelling changes).
2. Each stem gets both endings: *flitr-a* / *flitr-o*. **Minimal pairs**: only the ending differs,
   so nothing else can carry the difference (decided 2026-10-06, instead of the design doc's
   *brelda*/*brelpo*).
3. **Checks:** neither form appears anywhere in the Spanish Wiktionary (every word and inflected
   form), and both have zero frequency in Spanish, English, Portuguese, Italian, French and
   German.
4. 300 stems, split 50/50 into train and test **by stem** (seed fixed). The test half was never
   used for training anything.

**Status:** frozen with Phase 1's result. They're pronounceable but sometimes unusual for Spanish
(*flitr-*, *trubr-*), which didn't matter for a spelling test.

## Phase 5: Wuggy (planned, P15)

**Purpose:** R-NONCE: does pushing a meaningless word along the gender direction move social-gender
readouts (no concept to interact with), and do *-a* vs *-o* (and other) endings carry social
gender on their own?

**How they'll be made:**
- **Wuggy** (Keuleers & Brysbaert 2010, *Behavior Research Methods*): the standard pseudoword
  generator in psycholinguistics. It produces words that follow a language's syllable structure
  and letter-transition statistics, modelled on real words. Python package `wuggy` 1.1.2 (MIT);
  English, German and Spanish modules (README; confirm on install).
- **Graded endings on the same stems:** *-a*, *-o*, *-e*, and one consonant ending (the ending
  whose Spanish nouns split closest to 50/50, chosen before any model output). Each ending has
  stated predictions, since none is neutral: Spanish grammar (*-a* f, *-o* m, *-e* mixed leaning
  m) and English name sounds (vowel-final names judged female, consonant-final male; Cassidy,
  Kelly & Sharoni 1999) are separated by the design.
- Same safety checks as Phase 1, plus logged tokenisation; held out from all training by
  construction.
- **English-style nonce words** (*fleebleglorp*), if used for the English-only readouts: Wuggy's
  English module (alternative: ARC Nonword Database, Rastle, Harrington & Coltheart 2002).

**Candidate (not adopted):** German nonce words with gender-predicting suffixes
(*Flitr-ung* → *die*?), to test the suffix follow-up's "two routes" reading without real-word
confounds. Pre-register only after the suffix follow-up's main result.
