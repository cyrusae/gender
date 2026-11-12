# Hubness, CSLS retrieval and the epicene gap (exploratory, 2026-11-11)

Raised by an outside reviewer (another Claude session, via the PI). Script `scripts/hub_csls_check.py`
(committed before running, f0440c4). Nothing here is pre-registered.

## 1. Epicene gap: grammatical gender leaks into the attribute yardstick

Mean lean of masculine minus feminine strict-tier epicenes (*individuo, genio, bebé…* vs *persona,
víctima, criatura…*), which have fixed grammatical gender and no sex reference:
native (corrected) **+0.061 [0.025, 0.094]**; English attributes **+0.046 [0.013, 0.081]** (17 + 17).

So grammatical gender alone moves a word's lean toward the matching sex. For scale, Zhou's occupation
forms differ by about 0.17 in mean lean (masculine +0.067, feminine −0.106, corrected lists): roughly a
third of that difference is the size of a pure grammatical-gender effect. The zero-point study is not
affected: its zero point averages the masculine-epicene and feminine-epicene means with equal weight,
so this gap cancels by design. It is direct evidence that MWEAT-style leans mix grammatical and
semantic gender, even with English attribute words.

## 2. Hubness

Hub count N10 = how often a Spanish word is among the 10 nearest neighbours of the first 50k English
words (11.6% of Spanish words never are; the largest hub is the markup token *lat_seg*, 428). The
masculine form is the bigger hub for 64% of person nouns, 74% of occupations, 77% of identity nouns
(medians 7 vs 4, 12 vs 6, 9 vs 4). The masculine/feminine hub ratio correlates with intrusion
(Spearman 0.49 persons, 0.36 occupations, 0.62 identity). Hubness is largely frequency, so this
restates the frequency-ratio result (β 0.35) in retrieval terms.

## 3. CSLS retrieval (hubness-corrected), person-noun conditions

| Condition | Persons (cos → CSLS) | Occupations (cos → CSLS) | Difference (CSLS) |
|---|---|---|---|
| Feminine cues | −0.005 → −0.080 [−0.150, −0.009] | 0.290 → **0.194 [0.122, 0.266]** | −0.275 [−0.367, −0.181] |
| Masculine cues | 0.41 → 0.41 | 0.60 → 0.58 | −0.16 |
| Invariant cues | 0.26 → 0.24 | 0.49 → 0.46 | −0.22 |
| Plain translation | 0.28 → 0.34 | 0.51 → 0.52 | −0.18 |

**Reading.** Correcting for hubness removes about a third of the occupational intrusion (0.29 → 0.19)
but not the rest, and the occupation-specific gap is unchanged (−0.29 → −0.27). Under CSLS a feminine
cue slightly over-corrects for ordinary person nouns (the feminine form wins). The reviewer's specific
worry (frequent feminine forms such as *enfermera* winning by frequency) runs against the overall
pattern: masculine forms are usually the bigger hubs, so hubness inflates masculine intrusion, and the
correction lowers it. Per-occupation hub checks for female-dominated jobs not done.
