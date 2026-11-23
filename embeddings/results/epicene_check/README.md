# Epicene-anchor check (exploratory, 2026-11-23, PI question while writing Appendix E)

Writing out the frozen strict-tier epicene lists showed entries that fail the zero-point prereg's own
strict-tier criteria. Frozen lists untouched; the zero point (`scripts/zhou_zero_point.py`, corrected
attributes, Zhou's occupation lists) rerun in a sandbox without them.

- **V1 (rule):** remove anchors whose opposite-gender form is a Wiktionary noun with the same meaning
  (ES *cría*; FR *adjoint*, *adjointe*, *partisane*, *historienne*) and the mythic beings the prereg names
  as excluded (ES *hada*, FR *sirène*).
- **V2 (V1 + judgement):** also doubtful person senses (ES *pauta*, *procuraduría*, *defensoría*;
  FR *parution*, *capteur*, *bistrot*). Spanish then has 12 strict feminine epicenes, below the prereg's
  floor of 15, so by its rule the inanimate anchor becomes primary.
- **V3 (French only, V1 + morphology):** also French masculine anchors whose regular feminine (-e, -euse,
  -rice, -ère) is in the first 200k MUSE French words as the same word (*élu, associé, ingénieur, tenant,
  habitué, régent, soignant, intrus*; claude-unverified judgement excluded spurious forms such as *exile*,
  *titane*, *croisée*). A lower bound: feminines outside the vocabulary (*penseuse*, *recruteuse*) are not caught.

| | Frozen | V1 | V2 | V3 |
|---|---|---|---|---|
| ES native | −1.90 [−5.51, 1.39] inconcl. | −2.22 [−5.86, 1.10] inconcl. | −2.00 (epicene) / −1.74 (inanimate, primary by floor rule) inconcl. | — |
| ES English | −3.15 [−6.35, 0.17] inconcl. | **−3.63 [−6.93, −0.38] feminine further** | −3.42 [−6.77, −0.26] (epicene) / −3.38 [−7.14, 0.60] (inanimate, primary) inconcl. | — |
| FR native | −1.53 [−3.25, 0.41] inconcl. | −1.83 [−3.71, 0.28] inconcl. | −1.85 [−3.65, 0.25] inconcl. | −1.76 [−3.57, 0.33] inconcl. |
| FR English | −2.03 [−3.39, −0.63] fem. further | −2.37 [−3.76, −0.95] | −2.35 [−3.74, −0.90] | −2.33 [−3.72, −0.86] |

Reading: French verdicts unchanged. The Spanish English-attribute cell is borderline: inconclusive with the
frozen list, "feminine further" without *hada* and *cría*, inconclusive again when the floor rule applies.
