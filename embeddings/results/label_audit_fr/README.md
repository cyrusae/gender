# Label audit, French (exploratory, 2026-11-20)

`scripts/label_audit_fr.py` (committed before running, fc7791c). Gold = our French Wiktionary lexicon
(`data/lexicon/fr_nouns.csv`). French counterpart of the label audit in `label_audit_es/` (no
ending-exception test). Prompted by the PI asking whether appendix B13 (noun-list quality) covers French.

| | Zhou et al. 2019 | Omrani Sabbaghi & Caliskan 2022 |
|---|---|---|
| distinct words | 10,111 | 11,776 |
| agree with Wiktionary | 4,435 | 5,969 |
| disagree | **52 (1.2% of lexicon nouns)** | 29 (0.5%) |
| two genders in Wiktionary | 118 | 183 |
| listed under both genders | **1,598** | 174 |
| not a noun lemma: plural of a lexicon noun | 63 | 3,483 |
| not a noun lemma: other | **3,845** (verbs *constater, prétendre, savez, voulez*; adjectives *seul*; country names *portugal*; abbreviations *pib*) | 1,938 (compounds, abbreviations *km, hd*, inflected plurals) |
| share not a noun lemma | **38.7%** | 46.0% (mostly plurals) |
| animate among agreeing nouns | 378 (8.5%; *citoyen, directeur, père, professeur*) | 529 (8.9%) |

Reading: Zhou's French lists have the same problems as the Spanish ones, in some respects worse:
1,598 words appear under both genders (Spanish 860), and many non-nouns are verb forms. Where an entry is a
real noun, its gender label agrees with Wiktionary 98.8% of the time. Several disagreements are verb forms
labelled masculine (*conteste, gouverne, joue*), consistent with labels taken from a tagger rather than a
dictionary. The spelling cue "ends in -e = feminine" matches agreeing labels 60% of the time and
disagreeing ones 39%.
