# Grammatical vs semantic gender in attribute leans (exploratory, 2026-11-12)

PI question: the epicene gap shows grammatical gender moves a word's lean; is the lean *only* grammatical
gender? Script `scripts/common_gender_check.py` (committed before running, 46b2689). Not pre-registered.

## 1. Gap between masculine and feminine forms, compared with the pure-grammar gap

Gap = mean lean of the masculine forms − mean lean of the feminine forms (corrected attribute words).
Strict epicenes (*individuo/persona*: fixed grammatical gender, no sex) give the gap that grammatical
gender alone produces. Spanish:

| Group | Native gap | Excess over epicenes | English-words gap | Excess |
|---|---|---|---|---|
| Epicenes (grammar only, 17+17) | 0.061 [0.025, 0.093] | — | 0.046 [0.013, 0.081] | — |
| Zhou occupations (58) | 0.173 [0.131, 0.211] | **0.112 [0.072, 0.155]** | 0.147 | **0.101 [0.058, 0.142]** |
| Rule-built occupations (115) | 0.167 | 0.107 [0.069, 0.148] | 0.140 | 0.094 [0.054, 0.134] |
| Ordinary person nouns (107) | 0.149 | 0.088 [0.053, 0.128] | 0.121 | 0.074 [0.038, 0.111] |

French epicene gap (74 m / 23 f): native 0.061 [0.038, 0.087], English 0.048 [0.019, 0.077].

**Reading.** Gendered person nouns differ by about 2.5–3× the pure-grammar gap; the excess (≈ 0.09–0.11)
is semantic gender (the forms refer to a female vs a male person). So the occupation leans are roughly
one third grammatical gender and two thirds semantic gender. Occupations carry a little more excess
than ordinary person nouns (0.11 vs 0.09; not tested formally).

## 2. Common-gender nouns (el/la periodista): one vector, no fixed grammatical gender

Correlation of each noun's lean with the Glasgow GEND rating of its English concept (human rating,
1 feminine – 7 masculine), nouns with a rating only:

| Language | n | Native words r [95%] | English words r [95%] |
|---|---|---|---|
| Spanish | 27 | 0.62 [−0.05, 0.83] | **0.69 [0.10, 0.88]** |
| French | 33 | 0.43 [−0.01, 0.66] | **0.53 [0.10, 0.73]** |

**Reading.** Where grammatical gender can't act (one vector for both sexes), the lean still tracks how
masculine people judge the concept to be: semantic/stereotype gender reaches the vectors without any
grammatical help. Intervals are wide (27–33 rated nouns); with the native 8+8 words they touch 0. The
coverage is low because Glasgow rates English words and many concepts aren't in it.
