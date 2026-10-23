# Does the German "rarer = more feminine" lean reach new simple nouns? Pre-registration (DRAFT, awaiting PI)

Written 2026-10-22. No activations have been extracted for any noun in the held-out set, and no
number below comes from it. This follows `frequency-axis-prereg.md` (A–D: the overlap survives
every control) and `gemma-slim-prereg.md` G2 (no valid test: held-out *compounds* show no
overlap, cos +0.11 to +0.26 against −0.26 to −0.69 on the training nouns).

## The question

On Qwen3 8B/14B the German feminine-minus-masculine direction, fitted on the 91 `strat3` m/f
training nouns, points toward "rarer word". G2's held-out compounds don't show this. Two readings:

- **Training-set property.** The lean belongs to the 91 training nouns (a few dozen words; a
  sample quirk). Then it says nothing about how the model encodes gender.
- **Model property, not compounds.** The direction really does carry frequency, but compounds
  are read at their last token (roughly the head), whose frequency isn't the compound's. Then
  simple nouns the direction has never seen should still show it.

New simple nouns separate these.

## Why not just rerun the (c) cosine on new nouns

The (c) cosine needs masculine and feminine nouns compared *within* the same ending × loan cell,
because German endings predict gender. The remaining Phase 3 pool has no such nouns: every
unused known simplex noun in a cell shared by m and f was already taken for training (0 nouns
in shared ending × loan cells; 8 in shared ending-only cells). The rest are gender-pure cells
(*-ie*, *-te*, *-se*, … all feminine). A gender × frequency fit on them would mostly measure
endings. So this test uses a **within-gender** prediction that needs no shared cells.

## Held-out set H (frozen before extraction)

Every noun in the Phase 3 pool (`phase3_known.POOL`) with set = simplex, known to both Qwen3
1.7B and 4B (the frozen Phase 3 known filter), passing `_clean` (no English overlap, no
sex association), not in `phase3_final_v3.csv`, and **not complex** by the frequency-axis
complexity rule (D). Counts: f 502, m 190, n 196 (the rule removes 5 masculine nouns:
*Hubschrauber* "helicopter", *Rauchmelder* "smoke detector", *Zweisitzer* "two-seater",
*Gaskocher* "gas stove", *Pfefferstreuer* "pepper shaker"). Frozen to
`data/stimuli/rarity_heldout_v1.csv` with a hash in this file before any extraction. Neuter is
kept for a descriptive check only. A list the PI wants to review gets an English review sheet.

## Measure

For each model (Qwen3-8B-Base, Qwen3-14B-Base), each of the four Phase 5 layers, LAST position,
the Phase 3 frames:

1. **d** = the frozen German f−m direction from the training fit (the same one (c) and A–D used;
   unit norm). Nothing is refitted on H.
2. Score each held-out noun: p = projection of its activation onto d.
3. **Within-gender rank correlation** ρ_g = partial Spearman(p, Zipf | token count, ending), for g
   in {m, f}. Ending enters as fixed effects (residualise both on ending dummies within gender).

**Prediction if the lean is a model property:** ρ_m < 0 and ρ_f < 0 (rarer nouns of either gender
project further toward feminine).

**Null:** permute Zipf within gender × ending cells, 2,000 times; the observed ρ_g must be below
the 2.5th percentile (one-sided, the predicted direction). **Size floor:** |ρ_g| ≥ 0.1, so a tiny
effect on hundreds of nouns doesn't count.

**Control direction:** the same statistic with d replaced by the German number (plural)
direction, whose own frequency alignment is ≤ 0.17. If the number direction shows the same ρ,
the result is about frequency in all directions, not gender.

**Sensitivity (stated in advance).** With 190 masculine nouns the standard error of a rank
correlation is about 0.07, so passing the one-sided null needs roughly ρ_m ≤ −0.14. The
masculine test can detect a moderate lean, not a weak one; a "training-set property" reading
rules out only |ρ| of about 0.14 or more, not the 0.1 floor.

**Feminine robustness row (*-ie*).** *-ie* loanwords are 211 of the 502 feminine nouns. Ending
fixed effects stop *-ie* as a group from differing in mean, but ρ_f is still mostly *-ie*. So
ρ_f is also computed without the *-ie* nouns (same null, same floor).

## Readings (per model)

- **Lean generalises:** ρ_m and ρ_f both pass (null and floor) at a majority of the four layers,
  ρ_f without *-ie* also passes at that majority, and the number-direction control does not
  pass at that majority. If ρ_f passes only with the *-ie* nouns included, the reading is
  **mixed**, not "lean generalises".
- **Training-set property:** neither passes at a majority of layers.
- **Mixed:** anything else; reported descriptively, no conclusion drawn.
- **No valid test:** fewer than 100 masculine nouns remain after the complexity rule.

Neuter: ρ_n reported descriptively. Gender decoding accuracy of d on H (m vs f, sign of p) is
reported as a sanity check: if it's near chance, d isn't reading gender on these nouns and the
reading is "no valid test".

## Cost and order

Extraction only (no steering). 8B on the Mac, as Phase 3 was. 14B on the next pod (it can share
the Gemma pod; about 10 minutes). CPU analysis afterwards. Code: `src/gbleed/rarity_heldout.py`,
`gbleed rarity-heldout` CLI; outputs `results/rarity_heldout/`.

## Known limits

- Known-filter models (1.7B/4B) aren't the tested models, as in Phase 3.
- Feminine dominates H (about 2.5 : 1). The within-gender design doesn't need balance, but the
  masculine estimate is the noisier one.
- Zipf is from wordfreq. It measures the word's corpus frequency, not the model's exposure.
