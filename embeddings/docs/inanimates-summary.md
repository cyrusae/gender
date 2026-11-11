# What we learned about inanimate nouns

*Plain-language summary, 2026-11-10, written by Claude for the PI. Spanish and French static
embeddings (MUSE for Zhou's methods, Common Crawl fastText for Omrani's). Numbers point to the
preregs named in each section.*

## The short version

An inanimate noun's vector carries two kinds of "gender" at once:

1. **Grammatical gender**: *la mesa* is feminine, *el libro* masculine. This is large, and it is
   not just spelling.
2. **Social gender**: how people judge the thing (lipstick feels female, a hammer male). This is
   smaller, but it's really there, beyond grammatical gender.

The two mitigations we studied each go after one of these and leave the other:

| | grammatical gender | social gender (stereotype) |
|---|---|---|
| **Zhou's Shift** | survives | removed, but only as measured by Spanish/French gender words, and partly by construction |
| **Zhou's Hybrid** | survives | removed with English gender words, via Bolukbasi's English debiasing |
| **Omrani's erasure** | mostly removed (ES −72%, FR −95%) | untouched |

So "debiasing" means different things in the two papers: Zhou removes stereotype and keeps
grammar, Omrani removes grammar and keeps stereotype. Neither paper measured the half it wasn't
aiming at.

## 1. Grammatical gender in the vectors is real and not only spelling

- Both papers' grammatical directions classify exception nouns correctly: *el problema*,
  *el mapa* (masculine in -a), *la mano* (feminine in -o), 93–100%. A pure spelling detector
  would get these wrong. (Label audit, `upstream-issues.md` O17, Z9.)
- The gap is bigger for regular *-o/-a* nouns but present for every ending (Spanish 0.038 vs
  0.021; French suffixed 0.027 vs unsuffixed 0.026). (C1 spelling control.)
- Omrani's result generalises: on 1,000 + 1,000 nouns the direction never saw, erasure collapses
  the masculine–feminine gap the same way it does on training nouns (1.52 → −0.03). (Held-out
  check, `upstream-issues.md` O1.)

## 2. Zhou's mitigation keeps it (C1)

Zhou says Shift should remove stereotype from occupations while preserving inanimate nouns'
grammatical gender, but never measures inanimate nouns (Z6), and the projection step for them
(Eq. 2) is in no released cell (Z15). C1 measured it on 2,459 Spanish and 3,505 French
Wiktionary-labelled nouns that are in none of Zhou's lists. (`c1-prereg.md`)

- **The gap survives every version of the mitigation, 16 of 16 verdicts.** Spanish with Spanish
  gender words keeps 84–86% of it; French keeps all of it (1.11–1.16×).
- **Hybrid makes the gap bigger when measured with English gender words** (Spanish 1.4–1.8×,
  French 2.5–2.9×). Aligning to debiased English moves Spanish things toward the English gender
  words that hard debiasing deliberately leaves gendered (*woman, mother*).
- Inanimate nouns also sit near zero on Zhou's *semantic* axis (masc −0.011, fem −0.012), so
  Zhou's Figure 1 claim that things aren't socially gendered *on that axis* holds.

## 3. Things also carry social gender beyond grammar (gender ValNorm)

Question: does a noun's lean track how people rate the thing (Glasgow GEND of its English
translation), after accounting for grammatical gender and frequency? (`gvalnorm-prereg.md`)

- **Yes, in both languages.** β (human rating) is 0.25 (Spanish words) / 0.34 (English words) in
  Spanish, 0.17 / 0.28 in French; all CIs above 0. Grammatical gender is much bigger (β ≈ 1).
  English vectors track the same ratings more strongly (r 0.58).
- **Where it lives depends on the yardstick.** With Spanish/French gender words, the stereotype
  signal sits entirely along Zhou's semantic axis. With English gender words, about half (ES) to
  three quarters (FR) lies off it. (Addendum B.)

## 4. What each mitigation does to the social part

- **Shift (Zhou):** removes it as measured by Spanish/French gender words, but that is close to
  built in, because those words lie mostly along the axis Shift edits (cos 0.82 / 0.88; the same
  reason as the Table 1 identity, Z14). With English gender words, the off-axis part is
  untouched, as the method implies.
- **Hybrid (Zhou):** removes the off-axis part too, and it stays removed when we swap back to
  the original, un-debiased English yardstick (Addendum C). Since Hybrid only rotates Spanish,
  the change is in how Spanish things relate to English, not inside Spanish. The work is done by
  Bolukbasi's English debiasing; Table 1 can't see it (Z11).
- **Omrani's erasure:** leaves the social part unchanged in raw size (Spanish 0.0080 → 0.0074,
  French 0.0064 → 0.0066) while removing most of the grammatical gap. (Its standardised β looks
  like it grows only because the grammatical variance beside it shrank.)

## 5. Side findings about the noun lists

- **Zhou's "inanimate" list** is not really a list of inanimate singular nouns: about 40% of
  entries aren't singular nouns, about 9% are animate (*mujer*), 860 words are under both
  genders, and 18 *el agua*-type feminines are labelled masculine (tagger-derived labels; Z5, Z9).
  12 of the 16 Spanish gender attribute words are in it (Z12), so applying Eq. 2 to their list
  also edits the yardstick.
- **Omrani's lists** are 99.8% right where the entry is a real noun, but also ~40% plurals and
  adjectives, with duplicates (O2, O3, O17). None of this changed a conclusion.
- **Our lists** (C1, frozen) come from Wiktionary by rule; a few person or group nouns slipped
  into the frozen inanimate sets (*eminencia*, *voyer*, *philharmonie*), logged, not edited.

## Caveats

- Social gender ratings are British judgments of English translations (Glasgow), not Spanish or
  French speakers' ratings of the Spanish/French words.
- Hybrid is our rebuild (Procrustes onto hard-debiased English); Zhou didn't release it.
- English gender words in an aligned space may import English stereotype; Addendum C rules out
  the yardstick moving, not that.
- Effects are small in cosine terms (Spanish gap ≈ 0.03) but consistent across thousands of nouns.

## Where this connects to the main LLM project

The same split appears there: grammatical gender is strong and robust, and the question is how
much social gender rides along with it. In static embeddings the answer is "a small but real
amount, separable by method", which is a useful baseline for the transformer results.
