# C2. Character study: confirmatory results (batch 1, final)

Date 2026-11-11. Data: batch 1 = final (prereg amendment 4). Code `character/analysis_main.py`,
`character/scripts/run_confirmatory.py` (frozen at 4599ca9 before any estimate, tested on synthetic
data with planted effects). Full table: `character/results/confirmatory/confirmatory.csv`.

**Unit:** percentage points of *she* among *she + he* characters. Four cells: GPT-6-Luna with
reasoning medium / none, Qwen3-14B thinking off / on. Headline = four-cell average (gatekeeping,
amendment 2); cells are Holm-corrected behind it. Frames (writer persona, consider) averaged
within noun unless stated.

## Q1 writer: does the foreign word's grammatical gender move the pronoun?

Shift = she-share with the foreign word minus with the English word, same persona. Effect = how
much larger the shift is for feminine than masculine nouns.

| | effect (points) | 95% CI | decision |
|---|---|---|---|
| **Headline (4-cell average)** | **+26.1** | 22.9 to 29.2 | effect |
| Luna medium | +17.8 | 14.9 to 20.6 | effect |
| Luna none | +33.2 | 29.6 to 36.9 | effect |
| Qwen off | +32.7 | 26.6 to 38.8 | effect |
| Qwen on | +19.6 | 15.6 to 23.7 | effect |

- Five times the smallest effect of interest (5 points). Permutation tests agree (p < 0.001 in
  every cell). Same in both frames (e.g. Luna medium writer 18.2, consider 17.4).
- Every language shows it in every cell (per-language estimates 10 to 58 points; largest for
  German with Luna, *der/die/das* citation forms).
- Neuter nouns: no consistent shift (−4.9 to +5.6; one interval just excludes 0).
- **Thinking/reasoning shrinks it in both families** (Luna 33 → 18 with reasoning on; Qwen 33 →
  20 with thinking on). Exploratory (Q4, no prediction pre-registered); the Qwen part mixes mode
  with sampling settings (control cell not yet analysed).

**Check that cuts against it (pre-registered control).** In Finnish, Japanese and Turkish (no
grammatical gender) the foreign-word shift should not depend on the German/Spanish/Russian gender
of the same noun. For Luna it doesn't (all intervals include 0). **For Qwen it partly does**:
coded by German gender, +10.2 (1.3 to 19.2) thinking off, +6.2 (0.7 to 11.7) on; by Russian, +6.9
(1.0 to 12.8) on. So part of Qwen's Q1 effect may come from the nouns themselves (feminine-in-
German nouns read as more feminine whatever the language) rather than from the grammatical gender
of the word shown. Even subtracting it, Qwen's effect stays well above 5 points. A noun-level
control (e.g. adding the plain-arm she-share as a covariate) is the obvious exploratory follow-up.

## Q2 character: do adjectives change the pronoun?

Each modifier vs its article-matched plain form ("the best door" vs "the door"; else vs "a door").

- **57 of 70 modifiers change it** (30 toward *he*, 27 toward *she*); 12 "no effect of interest";
  1 inconclusive. **13 of 15 antonym pairs differ.**
- Biggest moves toward *he*: *malevolent* −76, *mediocre* −47, *boring* −47, *the worst* −41,
  *bad* −36, *old* −33, *terrible* −27, *your least favorite* −24.
- Pairs: interesting − boring +57, best − worst +52, new − old +45, favorite − least favorite +25,
  popular − unpopular +16; traditional − modern −24, large − small −21, rough − smooth −20.
  No difference: deliberate − accidental, and *an ideal* − *the ideal* (the article contrast).
- **Read with the baseline in mind:** plain "a door" gives *she* 79% (Luna) and 90% (Qwen), so
  there is little room to move toward *she* and lots toward *he*. Negative-valence words pulling
  toward *he* is the clearest pattern; grouping by taxonomy is exploratory (PI 2026-10-26).
- **Wugs** (nonce adjectives): any wug vs plain +2.7 to +10.5 (small, mostly toward *she*).
  Sound symbolism, *miryal* (soft) − *sputul* (hard): +7 to +17 toward *she* in every cell
  (estimate only: one word per side).

## Q5 frame: writer persona vs the model's own voice

- Writer experiment: the persona frame gives **fewer** *she* (headline −9.4); small for Luna (−2,
  −3), large for Qwen (−16 in both modes).
- Character experiment: small (headline +1.8; two cells "no effect of interest", two +3 to +4).

## Limits worth stating

- Two models; noun list fixed (191); precision is set by noun-to-noun spread (amendment 4).
- Qwen's genderless-language control is not clean (above).
- Outcomes are regex pronoun classes; the FastCoref agreement check (amendment 3) is still to run.
- Q4 (thinking) and the style measures are exploratory and not yet analysed.
