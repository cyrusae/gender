# 07 · Nonce (invented) words

*Code: `src/gbleed/phase1_stimuli.py` (`build_nonce`), `src/gbleed/nonce5.py` (Phase 5, Wuggy).
Files: `data/stimuli/phase1_nonce_v1.csv`, `phase5_nonce_es_v1.csv`, `phase5_nonce_en_v1.csv`.*

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

## Where the nonce sets go

| set | shape | used in | purpose | status |
|---|---|---|---|---|
| Phase 1 nonces | Spanish-looking *-a/-o* pairs (*flitra/flitro*) | bare words | spelling-eraser test | done, frozen |
| R-NONCE | Spanish-looking, four endings (*robeda/robedo/robede/robediz*) | **English** frames (*If the robedo were a person…*, *The robedo is very ___*) | unsteered: does a Spanish gender *ending* alone carry social gender? steered: a meaningless word pushed along the gender direction | built (v1), not yet used |
| English-style | English-looking (*boddle*, *mupper*) | English frames | steered only: a push with no gender cue in the word at all | built (v1), not yet used |
| German suffix nonces | German stem + gender suffix (*Flitr-ung*) | German article frames | suffix follow-up: does the model say *die Flitrung*, and does the direction read it? | candidate only |

## Phase 5: Wuggy (built 2026-10-08, P15/P17)

*Code: `src/gbleed/nonce5.py`. Files: `data/stimuli/phase5_nonce_es_v1.csv`,
`data/stimuli/phase5_nonce_en_v1.csv`. Decisions: `docs/decisions.md` (P15, P17).*

**Wuggy** (Keuleers & Brysbaert 2010, *Behavior Research Methods*) is the standard pseudoword
generator in psycholinguistics: given a real word as a template, it produces pseudowords with the
same syllable structure and letter-transition statistics. Python package `wuggy` 1.1.2 (MIT),
installed as a project dependency; language modules `orthographic_spanish` and
`orthographic_english` (also available: German, French, Italian, Dutch, Polish, Basque, Serbian,
Estonian, Vietnamese). Quirks found while building: language data downloads on first use (a
prompt, answered with `auto_download=True`), and **templates must be in Wuggy's own lexicon**
(unknown ones are skipped and counted: 7 of 230 Spanish, 12 of 200 English).

### R-NONCE: Spanish-shaped, four endings (150 stems = 600 words)

1. **Templates:** common inanimate Spanish nouns with a regular *-a*/*-o* ending, 2–3 syllables,
   Zipf ≥ 3.5, never a test-set word; **equal numbers of *-a* and *-o*** (115 each), so the
   template's gender can't tilt the stems. Templates only give shape.
2. **Generation:** up to 10 pseudowords per template.
3. **Stem** = pseudoword minus its final vowel (Wuggy copies the template's final vowel, so it is
   always removed), deduplicated. Stems ending in *c*, *g*, *z* or *q* are dropped, because a
   following *e*/*i* changes their pronunciation.
4. **Four endings per stem**, each with a stated prediction (no ending is neutral):

   | ending | Spanish cue | English name-sound cue |
   |---|---|---|
   | *-a* | strong feminine | vowel-final → female |
   | *-o* | strong masculine | vowel-final → female-ish (weaker) |
   | *-e* | weak masculine: only 16% of Spanish *-e* nouns are feminine (*el puente*; *la muerte*) | vowel-final |
   | *-iz* | even: 38% feminine (*la nariz* / *el lápiz*) | consonant-final → male |

   The pre-set rule for the consonant ending ("closest to 50/50") picked *-s* (45% feminine), but
   *-s* is also the plural marker in Spanish and English, so it was excluded on that ground; *-iz*
   is the most even consonant ending that isn't an inflection. (English name phonology: Cassidy,
   Kelly & Sharoni 1999.)
5. **Checks:** every one of the four forms is absent from the Spanish Wiktionary (every word and
   inflected form) and has zero frequency in Spanish, English, Portuguese, Italian, French and
   German. Of 1,389 candidate stems, 837 passed; 150 were drawn (seed 0).
6. Examples: *robeda/robedo/robede/robediz*, *tenceja/tencejo/tenceje/tencejiz*,
   *sañera/sañero/sañere/sañeriz*. Test-only (never in any training).

### English-style set (100 words, steered only)

1. **Templates:** common English nouns, 2 syllables, 4–8 letters, whose first WordNet sense is a
   physical thing (artifact, object, food, plant, substance) and which are never adjectives (so
   not *local*, *total*), and which don't look inflected.
2. **Generation:** up to 5 pseudowords per template.
3. **Checks:** zero frequency in the six languages and not a WordNet lemma; **no endings that
   look like English inflections**, *-s* (plural), *-ing* and *-ed* (verb forms), in templates
   or outputs (found on the first run: *predurts*, *bassing*, *hoved*). Same principle as Spanish
   *-s*: an inflection would add a number or part-of-speech cue.
4. **Final sound recorded** (English name phonology), by sound rather than spelling: a silent
   final *-e* after a consonant counts as consonant-final (*caggle*, *phove*). 77 consonant-final,
   23 vowel-final. It cancels in steered-minus-unsteered shifts, but is kept for analysis.
5. Of 864 candidates, 597 passed; 100 drawn (seed 0). Examples: *boddle, mupper, shatug, capple,
   muttle, frince, sacket, imboke, pomia, knolo*.

**Tokenisation** is recorded when activations are extracted (every saved activation file stores
each word's tokens), as for all other stimuli.

## German suffix nonces (candidate, not built)

Wuggy stems + a gender-predicting suffix (*-ung*/*-heit* f, *-ismus* m, *-tum*/*-chen* n). Would
test, without real-word confounds, whether the model assigns the suffix's gender (*die Flitrung*)
and whether the single-root gender direction reads it. Its case depends on the suffix follow-up:
the exploratory S1 (2026-10-08) was mixed (1.7B: "offset"; 4B: "two routes", with a middle-layer
reversal), so the decision waits for the confirmatory 8B/14B S1.
