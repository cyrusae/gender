# 03 · Spanish word lists (Phase 2)

*Code: `src/gbleed/phase2_stimuli.py` (pool), `phase2.finalize` (final sets). Pool:
`data/stimuli/phase2_pool_v4.csv`; frozen final list: `phase2_final_v4.csv`. Decisions:
`docs/decisions.md` 2026-10-06/07.*

## The problem these lists solve

In Spanish, *-o* nouns are mostly masculine and *-a* nouns mostly feminine. A "gender direction"
trained on regular nouns could just be an *ends-in-a* detector. So the **training** nouns must
be ones where spelling gives the gender away as little as possible, and the **test** nouns are
cases where spelling and gender disagree.

## The sets

| set | role | what it is | final count |
|---|---|---|---|
| `strat` + `matched` | **training (primary)** | nouns *without* *-o/-a* or any gender-predicting ending, in spelling cells with both genders | 33 f / 132 m (matched: 27 + 27, nested inside) |
| `regular` | training (comparison) + test | regular *-o* masculine / *-a* feminine nouns | 247 (fixed train/test split) |
| `exception_ma` | test | masculine *-ma* nouns from Greek (*el problema*, *el tema*) | 20 |
| `exception_clipping` | test | shortened words keeping the long word's gender (*la foto* ← *fotografía*) | 2 |
| `exception_true` | test | genuine exceptions (*el día* "day", *la mano* "hand") | 7 |
| `homograph` | test | nouns that are also verb forms (*camino* "path" / "I walk"), read in a noun frame (*mi camino*) and a verb frame (*siempre camino*) | 72 |
| `multi` | test | same word, both genders (*el/la mar* "sea") | 15 (separate file) |

## Filters, in order

**Starting point:** `eligible()` (doc 01): single gender, inanimate, no register/region marks,
not another part of speech or inflected form, not more common in English, no feminine
*a-*/*ha-* nouns (*el agua*). Phase 0 example words excluded.

**Training candidates (`stratum`):**
1. **No gender-predicting ending**: *-o*, *-a*, *-ción*/*-sión* (f), *-dad*/*-tad*/*-tud* (f),
   *-umbre* (f), *-ez*/*-eza* (f), *-itis*/*-sis* (f), *-or* (m), *-aje* (m), *-án*/*-ón*/*-ín*,
   *-ista*, *-ante*/*-ente*, *-ie*, *-ud*. What's left: *-e*, *-l*, *-r*, *-z*, *-s*… nouns
   (*el puente* "bridge", *la llave* "key", *el árbol* "tree", *la cárcel* "prison").
2. **English overlap**: drop if the noun ≈ its English gloss (string similarity ≥ 0.9: *hotel*)
   or is a common English word (Zipf ≥ 3 in English).
3. **Sex-associated concepts** (they carry *social* gender, which would make Phase 5 circular):
   gloss words (woman, female, beard, skirt, lipstick, sexual anatomy…) or WordNet's first sense
   under genitalia, reproductive organ, garment, undergarment, jewellery, cosmetic.
4. **Second semantic screen** (carried over from German, 2026-10-07): WordNet person/animal on
   the first two senses; groups (*instituto*, *ayuntamiento* "town council", *dinastía*);
   animal/person gloss words; **chemicals** (Spanish elements are all masculine, *el sodio*:
   a meaning class with one gender); place names; sex-typed garments.
5. Zipf ≥ 2.0.

**Spelling cells** for the stratified comparison: the last two letters (*-te*, *-al*, *-ce*…),
except derivational suffixes hiding inside them, which get their own cell (*-ete*/*-ote*:
*billete* "ticket", *camarote* "cabin"; *-men*: *examen*). Only cells containing both genders
are kept. Loan status (from Wiktionary etymologies) is a **covariate**, not a cell key: half of
Spanish nouns have no etymology and matching on it would empty most feminine cells.

**Known check** (doc 02): known by Qwen3-1.7B and 4B.

**Final sets** (`phase2.finalize`): every known noun in a both-gender cell → `strat` (compared
*within* cells; this is the primary, adopted 2026-10-07 for power). A seeded shuffle picks equal
masculine and feminine per cell → `matched` (nested; the strict secondary).

**Test sets:** regular and homograph sets are **carried over** from v3 (never redrawn, so
versions can be compared), minus anything the new filters flag. Exceptions come from a
hand-written spec (`exceptions_spec.csv`) and are checked against Wiktionary; any that turn out
regular, animate, regional or homographic are dropped and listed. Homographs: masculine *-o*
nouns that are also a 1st-person verb form, feminine *-a* nouns that are also a 3rd-person form,
with a common verb reading (Zipf ≥ 3).

## Amendments and known issues

- **Initial *a-* (amendment 2026-10-08):** dropping feminine *el agua*-type nouns left word-initial
  *a-* as a perfectly masculine cue in training (0 of 33 feminine vs 17 of 132 masculine start
  with *a-*). Found by a sparse-autoencoder feature; fixed by residualising initial *a-* in the
  analysis (lists unchanged).
- **Residual spelling cue:** within cells, the last 3–4 letters still predict gender slightly
  (balanced accuracy 0.55, chance 0.50) through many small sub-endings (*-ste*, *-che*, *-bre*).
  Stated as a limitation.
- **Held-out leak (found 2026-10-08):** the Phase 2 builder never excluded the Phase 4–5 flipped
  pairs and classics; 37 pairs' Spanish nouns and the classic *llave* are in training. Phase 2's
  results are unaffected; those pairs are dropped from Phase 4–5 (doc 05).
