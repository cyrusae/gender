# Word-pair intrusion: intervals and mechanism (DRAFT pre-registration)

*Status: DRAFT, 2026-11-10, written by Claude for the PI's review. Nothing in Part B has been
scored. Part A puts intervals on numbers already seen (exploratory run, commit cd7693c), so its
point estimates are not confirmatory; its value is the uncertainty.*

## Background

Zhou et al.'s Table 2 test ("good:doctor = buena:?") used only masculine or invariant adjective
cues (Z17). Our rerun with feminine cues added (`scripts/zhou_wordpair.py`, `results/zhou_wordpair/`)
found the masculine occupation form is retrieved better than the feminine one **even when the cue
is feminine** (ES original: M MRR 0.58 vs F 0.37; FR 0.54 vs 0.33). Call that gap the
**intrusion**. Three accounts predict it, and they predict different patterns:

| account | what produces the intrusion | predicts |
|---|---|---|
| **Dictionary anchoring** | MUSE's alignment is fit on a bilingual dictionary; if English *doctor* is paired only with *doctor* (masc.), the rotation pulls English toward the masculine form | larger intrusion where the dictionary lists only the masculine form; shrinks when occupation entries are removed from the alignment's training pairs |
| **Generic masculine in usage** | Spanish/French text uses the masculine for sex-unspecified people, so the masculine vector sits closest to unmarked English | intrusion across the board, including female-stereotyped jobs; tracks how much more often the masculine form is used; unaffected by alignment changes; also visible in monolingual geometry (already seen: Zhou Fig. 1 skew, which a rotation cannot create) |
| **Occupational stereotype** | the job itself is associated with men | larger intrusion for male-stereotyped jobs (*ingeniera*), smaller or reversed for female-stereotyped ones (*enfermera*), over and above frequency |

A fact checked before drafting (descriptive, not an outcome): the MUSE dictionary does **not**
pair English occupations only with masculine forms. ES: 29 of 58 pairs list both forms, 29 list
only the masculine, none only the feminine. FR: 12 both, 9 masculine only, 2 neither. So
dictionary coverage varies across occupations and can be used as a predictor.

## Part A: intervals on the word-pair measures

Same queries, spaces and fixed scoring rules as `zhou_wordpair.py` version (b)/(c).

- Measures per language × space: intrusion under feminine cues (M MRR − F MRR | feminine cue);
  agreement gap (M MRR | masc. cue − F MRR | fem. cue); invariant-cue gap (M − F | invariant);
  and the change from Original for Shift Ori and Hybrid Ori on each.
- 95% bootstrap, 10,000 draws, seed 0, resampling occupations and, independently, cues within
  each cue gender (ES has 3 feminine cues, so intervals will be wide; reported as such).
- Reading: a measure is "present" if its CI excludes 0; a mitigation "reduces" it if the
  change's CI is below 0.

## Part B: mechanism (Spanish primary; French descriptive)

Per occupation o (original MUSE space): **I_o** = mean over feminine cues of (1/rank_m − 1/rank_f),
misses scored 0. Secondary: the same over invariant cues.

Predictors:
- **D_o** = 1 if the MUSE en–es dictionary lists only the masculine form for the English word.
- **R_o** = log(rank_f / rank_m) in MUSE frequency order (> 0: masculine more frequent).
- **S_o** = Glasgow GEND rating of the English occupation word (1 feminine … 7 masculine).
  Coverage is 32 of 58 Spanish occupations; S analyses use that subset.

Tests:
1. **Regression** I ~ D + R (all occupations) and I ~ D + R + S (Glasgow subset); standardised
   coefficients, 95% bootstrap over occupations (10,000). Report the correlation between R and
   S first: if |r| > 0.6, their separate effects are labelled not identifiable.
2. **Female-stereotyped check:** mean I over occupations with S ≤ 3.5. Generic-masculine usage
   predicts > 0; stereotype predicts ≤ 0 (CI).
3. **Alignment ablation:** refit the Procrustes alignment (as `zhou_hybrid.py`, onto the
   *original* English, no debiasing) three ways: dictionary as is; with every occupation form
   and its English word removed from the training pairs; with each English occupation paired
   only to the feminine form. Recompute I for all occupations. Anchoring predicts the mean
   intrusion falls in the second and third versions (CI of the change below 0); the other two
   accounts predict no change.

Readings, fixed now:
- **Anchoring supported** if D's coefficient CI is above 0 **and** the ablation lowers I.
- **Usage supported** if mean I > 0 in the female-stereotyped subset, R's coefficient CI is
  above 0, and the ablation changes nothing (change CI includes 0 and |change| < 0.05).
- **Stereotype supported** if S's coefficient CI is above 0 with R in the model.
- These are not exclusive; each is reported separately.

## PI decisions (2026-11-10)

- Cleaned occupation list is primary (non-occupations and duplicates dropped); Zhou's list as
  published is a secondary replication of their setup.
- French: descriptive only.
- Stereotype measure: labor statistics from several countries (scope being settled), alongside
  Glasgow GEND.

- Mapping `data/occupation_mapping_v1.csv` approved: generic *trabajador, empleado* (FR
  *employé, ouvrier*) use whole-workforce shares; *jefe* uses all managers.
- Feminine homographs (*química* 'chemistry', etc.; column `fem_also_nonperson`): kept in, entered
  as a predictor H_o in the regression (primary); excluding them is a sensitivity check.

## Earlier open decisions (kept for the record)

1. Zhou's ES list includes entries that are not occupations or are duplicates (*tory*,
   *commissioned*, *reader*, *recorder*; *employee* and *adviser* appear twice). Drop them
   before scoring? (Recommended; they'd go in a sensitivity check.)
2. Stereotype measure: Glasgow GEND covers 32/58. An alternative is the share of women in each
   occupation from US labor statistics (BLS), which covers more jobs but is American and about
   who does the job, not the word. Glasgow only, BLS only, or both?
3. French: 23 pairs is too few for the regression; report it descriptively only?
