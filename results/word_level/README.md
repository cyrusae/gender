# Word-level Phase 5 tables (for exploration; not CC0)

Built by `scripts/export_words.py` from the Phase 5 steering sweeps of Qwen3-8B, Qwen3-14B,
Gemma-3-4B and Gemma-3-12B (Spanish steering only; German had no valid test). Descriptive:
the verdicts are the pre-registered ones in `verdicts.csv` / the preregs. Contains real words and
Glasgow ratings (CC BY 4.0), so this is not part of the CC0 `eda/` set.

All scores are at the working dose (+1 × α*), averaged over the primary prompt wordings W1–W3
that passed the frame check. Gemma rows use only nouns Gemma knows (known-check filter).

## `noun_steering.csv` (one row per run × readout × noun)

| column | meaning |
|---|---|
| `model`, `layer`, `positions` | which steering run (`noun` = push on the noun only; `all` = every word) |
| `readout` | `R1` (Spanish sentence, English adjective) or `R1-EN` (all-English prompt with the noun's English concept) |
| `lemma` | the noun as prompted (for R1-EN this is the English concept) |
| `source_lemma`, `gender`, `concept_en`, `set` | the Spanish noun, its grammatical gender, English gloss, and list (`steer_es`, `pair`, `classic`). An R1-EN concept shared by two nouns appears once per noun |
| `real` | graded score for the grammatical-gender direction: > 0 = shifted toward feminine-rated adjectives (−β_GEND, controls for valence, arousal, size, frequency) |
| `random_mean`, `random_p95` | the same for 100 random directions of equal size: mean and 95th percentile |
| `real_minus_random` | `real − random_mean`: the gender-specific part for this noun |
| `social`, `number` | the social-gender direction (positive control) and the singular/plural direction |

## `adjective_shifts.csv` (one row per run × readout × English adjective)

| column | meaning |
|---|---|
| `word`, `GEND`, `VAL`, `AROU`, `SIZE`, `zipf` | adjective and its Glasgow ratings (GEND 1 = feminine … 7 = masculine) and frequency |
| `shift_real`, `shift_random_mean`, `shift_social`, `shift_number` | mean change in log-probability (nats) vs the unsteered prompt, over nouns and wordings |

A "bleed" shows up as `shift_real` sloping downward with `GEND` (feminine-rated words up). Compare
with `shift_random_mean` (the generic effect of any push) and `shift_social` (what a real social
push looks like).

## `adjectives_{qwen3,gemma3}.csv`

The readout vocabularies (656 / 721 words): Glasgow ratings, Zipf and the token id each model reads.

## Cautions

- One noun's score averages 3 prompts: top/bottom lists are noisy. Look for patterns across many nouns (by set, gender, concept type) rather than single winners.
- Effects are small (Qwen3-8B noun push: mean `real` about +0.02), so most per-noun variation is noise.
