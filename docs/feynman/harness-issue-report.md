# Issue report: fabricated content in `feynman-to-sources` reading-library summaries

*Drafted by Claude for the project PI to send to the harness developer (2026-11-07). Everything below was checked against the local PDFs or arXiv records; the per-paper evidence is in two ledgers the PI can share on request.*

## Summary

The reading library generated in one run of the `feynman-to-sources` workflow (Stage 4 "Triage & Reading Guides": per-folder `README.md` files and the master `SUMMARIES.md`) contained widespread fabricated content: wrong authors, wrong titles and venues, invented statistics, models and datasets, and in several cases findings that are the **opposite** of what the paper reports. The PDFs themselves were mostly the right papers. The upstream research outputs (the Feynman literature reviews the library was built from) were much more accurate, in one case explicitly correcting an author ("Finkley, Li, Walsh (not 'Pucci')"), so most errors were
introduced at the library-generation step. But the research outputs aren't error-free either (see "Follow-up: the research outputs themselves" below).

## Scale

Two independent checking passes read each paper's local PDF (first pages for metadata, full text where a claim needed it), falling back to arXiv or the DOI page.

| file set | papers | already correct | corrected | couldn't check |
|---|---|---|---|---|
| `SUMMARIES.md` (detailed entries) | 28 | 6 | 21 | 1 |
| folder guides 04–08 | 95 | 3 | 90 | 2 |
| folder guides 01–03 | ~40 | ~13 | ~27 | 0 |
| master index citation lines | ~135 | — | ~85 lines rewritten | 1 |

Three PDFs are the wrong paper entirely (e.g. a file named for a statistics paper contains an unrelated gesture-recognition preprint; another contains an unrelated reinforcement-learning paper), and the summaries describe the intended paper anyway.

## Kinds of error, with examples

**Wrong attribution, correct title.** The title usually matched the real paper; the surrounding metadata was filled in.
- "Neutrality Bites" (arXiv 2606.07969): listed as *Pucci, Flavio et al.*; actually Finkley, Li & Walsh (FAccT 2026). "Pucci" is a real author of a *different* paper in the same folder.
- "GRUFF" (2605.30214): *Gautam, Vasu et al.*; actually Mewes, Lauscher & **Vagrant** Gautam (third author, wrong first name).
- "Does Thinking Help Fairness?" (2609.30768): *Zheng et al.*; actually Pan, Germino, Ma, Daly, Moniz, Hua & Chawla. One guide lists the correct authors and still labels it "[Zheng et al.]".
- "Thinking Inertia" (2610.11765): *Gao et al.*, subtitle "Overthinking in Test-Time Compute"; actually Lei et al., "…LLMs Keep Thinking When Told Not To".
- Several file names carry the wrong first author, and later text keeps using that label.

**Invented specifics.** Model lists, dataset names, sample sizes and headline numbers that appear nowhere in the paper (e.g. a replication summarised with "BF01 = 47.6" and a "complete null"; the paper reports registered Bayes factors of 37.42 and 2.97 and a mixed result; a benchmark paper given "22 models and 96%"; tabular-fairness work (Adult/COMPAS/Credit) described as BBQ/StereoSet).

**Reversed findings.** At least five summaries state the opposite of the paper:
- "reasoning models stay unfaithful": the paper finds them *more* faithful (59% vs 7%);
- "chain-of-thought fails to reduce gender bias": the paper finds it *reduces* bias;
- "stated reasoning steps carry near-zero causal weight": 76.9% are causally load-bearing;
- "reasoning raises bias": it lowered a bias rate from 0.64 to 0.40;
- "heavily biased thoughts drive output bias": the correlation is weak (< 0.6).

**Invented relevance.** "Relevance to the project" lines linked papers to the user's own pilot results ("explains why our pilot showed early commitment…") on the strength of the invented content.

## Pattern

- Correct entries were mostly well-known, older papers (LEACE, ITI, Gonen & Goldberg). But age doesn't protect: a 2011 and a 2022 paper also had invented designs and statistics.
- The errors look like summaries written from the **title, file name and prior knowledge**, not from the PDF text: a plausible paper with that title, aimed at the user's research interests. The reversed findings lean the same way, toward one narrative ("thinking doesn't help").

## Where in the skill, and suggested fixes

`SKILL.md` Stage 4 asks for "detailed per-paper analytical entries (Hypotheses, Datasets/Models, Methodology, Findings…)" but never says the entry must come from the PDF, and nothing checks it. Stage 2 doesn't confirm a downloaded PDF is the requested paper; Stage 3 names files from the review's citation, not from the PDF.

1. **Stage 2/3: confirm identity before naming.** After download, read the PDF's first page and compare title (fuzzy) and first-author surname with the requested citation; on mismatch, log it to `MISSING_SOURCES.md` instead of filing it. Build the `[Author Year]` name from the PDF's author line, not the review's.
2. **Stage 4: ground every entry in the PDF.** Require each entry to include a verbatim sentence from the abstract, and have a script check the quote occurs in the PDF's extracted text. Require numbers, models and datasets to be copied from the text with a page reference; anything not found is written as "not stated" rather than filled in.
3. **Stage 4: separate summary from relevance.** Relevance-to-project lines should cite which finding they depend on, so a wrong finding visibly breaks the link.
4. **Stage 6: don't write Creators from file names.** Take author metadata from the PDF / DOI / arXiv record (CrossRef or arXiv API), not from the `[Author Year]` pattern or `SUMMARIES.md`.
5. **A verification pass as a final stage**, writing a per-paper ledger (VERIFIED / CORRECTED / UNVERIFIABLE + evidence), so the user can see what was checked.

## Follow-up: the research outputs themselves (2026-11-08)

Two later Feynman outputs (a reading guide to a paper's 11 cited works, and a forward-citation review) were fact-checked the same way, against full texts. They were better than the library, but not clean:

| output | claims checked | correct | partly | wrong | unverifiable |
|---|---|---|---|---|---|
| bibliography reading guide | 65 | 40 | 12 | 8 | 5 |
| forward-citation review (+ a planning handoff) | ~67 | 48 | 12 | 4 | 3 |

The errors are a different kind from the library's: metadata was right (every venue, DOI and ID for the 11 cited papers checked out), but specifics were wrong or invented:
- **an invented statistic**: "balanced 95.12% ± 1.46% accuracy" for Bolukbasi et al. 2016; the string doesn't occur in the paper, which reports an F-score of .627 ± .102;
- **a wrong arXiv ID** for a key paper (it resolves to an unrelated physics paper), even though the same output says it read that paper's full text from Zotero;
- **a doubled sample size** (2,000 per gender instead of 2,000 per language);
- **a guessed method** presented as a differentiator ("GloVe-style setups (check their embeddings)", but they used the same fastText vectors);
- **misattributed provenance** (which paper a word list came from), a mislabelled statistic, wrong coverage claims (a paper said to cover a language it doesn't), and conflated definitions (a per-word score vs a set-level statistic).

The pattern fits the same mechanism at smaller scale: where the output quotes the paper it's right; where it summarises from memory or fills a gap, it invents plausible specifics. The fixes above apply here too: numbers should be quoted with a page/section reference and checked against the extracted text, and IDs should come from the record actually read (the PDF header shows the arXiv ID).

## What we did locally

Both passes' fixes are applied in place (PDF file names left unchanged so Zotero links don't break; mismatched names are listed in the ledgers). Entries that couldn't be checked are marked *[unverified]*.
