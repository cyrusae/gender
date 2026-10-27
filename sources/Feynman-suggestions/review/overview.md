# Review of the Feynman reading list and deep dives (2026-10-22)

*Part 1: overall quality, where I agree and disagree. Part 2 (per-paper accuracy checks) is in
`batch-*.md` in this folder, from subagents that read the papers; treat their verdicts as the
evidence and this file as opinion.*

## Verdict in one paragraph

A useful map of the right literatures (human linguistic-relativity work, LLM gender
representations, cross-lingual sharing, steering, the confidence critique), with three of the
most relevant papers correctly picked out for deep dives. But it was written for the experiment
we **planned** in the spring (Spanish templates → contrast/centroid direction → steer English
WinoGender-style pronoun completions), not the one we **built**. So its "what this means for your
write-up" advice is partly already done, partly inapplicable, and in two places overconfident.
Use it as a reading map, not as a framing.

## Where I agree

- **Elpers et al. 2022 is essential.** The correlational vs. induction contrast (Exp 1 null,
  Exp 2 replicates, with a demand-characteristics caveat) is the right human backdrop, and "scope
  claims to models, not to Whorf" (synthesis table, row 4) is exactly right.
- **The confidence critique (Buttigieg 2610.08559) is a real threat model.** We took it seriously
  and pre-registered our own control (a)–(c); every Phase 5 / P29–P31 verdict survived. Its
  emphasis on forced-choice readouts and random-direction controls matches what we did.
- **Report pre-registered vs. exploratory separately and publish the nulls.** We already do this;
  the advice is sound.
- **"Single-family unless replicated" in the abstract.** Agreed; that is what the Gemma-slim
  replication is for.

## Where I disagree or would soften

1. **"Elegant Bridge predicts your headline result" (deep dive §3) is too strong.** Shtedritski &
   Mihaylov is a prompting-level, correlational result (adjective distributions, ~55–69% accuracy,
   chance 50%). It shows output-level adjective associations transfer across languages. It does not
   predict that a hidden-state grammatical direction, pushed at the noun, moves English adjective
   ratings. Our results (small language-specific bleed at 8B, null at 14B) are consistent with
   their finding *and* with weak causal routing. Frame it as closest prior art, not a prediction.
2. **"Your pipeline is the Experiment-2 analog" (deep dive §1) is a loose analogy.** Elpers Exp 2
   teaches *people* a new category and then measures similarity. We neither train the model nor
   use a similarity task; we intervene on an existing representation. The useful parallel is
   narrower: in both, an effect appears only when the gender category is made active, and both
   raise the "is it a sex-association heuristic?" question. Our person/thing contrasts (P30/P31)
   and the neuter-diminutive check are our answer to that question.
3. **The Buttigieg "mandatory controls" don't map one-to-one.** Their confidence direction is built
   from high- vs low-probability QA answers. On our inputs that direction is mostly a *frequency*
   direction (our earlier analysis), which is why we replaced it with the log-prob covariate,
   Spearman check and gender×Zipf cosine. The unexpected finding that came out of it, the German
   f−m direction aligning with "rarer", is ours, not something the list anticipates.
4. **The design checklist (§8) assumes centroid directions, mean-centring and WinoGender pronoun
   steering with an α sweep.** We used within-cell residualised directions with a split-half
   estimator, Glasgow-norm adjective readouts and dose units of the state norm. The list misses the
   estimator-bias issue entirely, which turned out to matter more than anything on the checklist
   (plug-in markedness 25/27 layers → 0 under split-half).
5. **It recommends a social-gender direction from occupation nouns via 2606.30152.** We already have
   a social-gender control direction; the citation is only useful if the paper exists and says what
   is claimed (see batch files: several 2026 IDs could not be verified when the list was written).
6. **Missing literatures we will need:** the estimator-bias / cross-fitting literature (the
   split-half trick), LEACE and concept-erasure limits (we use LEACE in Phase 1), and the
   frequency/anisotropy literature on embedding geometry (relevant to the German frequency axis).
   I'll suggest specific papers once the checks are back rather than add unverified IDs.

## Reading priority (draft; will be revised after the accuracy checks)

**Read fully (shapes framing):**
1. Elpers, Jensen & Holmes 2022 (JML; open PDF).
2. Samuel, Cole & Eacott 2019 (systematic review; sets the human base rate).
3. Shtedritski & Mihaylov 2024 (closest LLM prior art).
4. Brinkmann et al. 2025 (2501.06346; shared grammatical features across languages, the
   mechanism-level parallel to our Phase 4).

**Read the abstract and the key figure:**
- Buttigieg et al. 2026 (we have already engaged with it in detail).
- Stanczak et al. 2023 (2311.18567; grammatical gender vs meaning confound, close to our
  within-cell design).
- Phillips & Boroditsky 2003 (the original; mostly needed for citation).

**Skim / cite only if needed:** the steering-method papers (§5), evaluation benchmarks (§7), and
any 2026 ID the batch checks could not verify.

## Part 2: accuracy checks (summary of `batch-*.md`, 2026-10-22)

43 references checked (abstracts for most; full text where open). **None fabricated; none NOT FOUND.**

- **Wrong (fix before citing):** the escholarship `0jt9w8zf` link is the 2000 CogSci paper, not the 2003
  chapter; Boroditsky & Schmidt 2000 has no adjective-generation task.
- **Partly right, numbers/scope to correct:**
  - Shtedritski & Mihaylov 2407.09704: within-language 54.3–69.0% (not 56–69%); Mihaylov is first author.
  - Brinkmann 2501.06346: ablation result is averaged over all concepts; "mid-layers best" unsupported (one layer used).
  - Stanczak 2311.18567: quoted MI values are word2vec only.
  - Buttigieg 2610.08559: layers 13–30, not 13–22.
  - 2606.30152: frozen encoders, Spanish/French only, so weak support for a social-gender direction in our models.
  - Xiao et al. 2026 is published (Cognitive Science, doi 10.1111/cogs.70176); the claimed person-noun control was not found.
  - Mahajan & Durgin 2025 is a potency-rating task; Kurinski & Sera 2011 is a classroom longitudinal study.
  - CLAS 2601.16390, 2312.03813 (mean-centring), 2502.13120 (FR→EN/DE), MORPHOGEN 2604.18914 (none of our languages): descriptions loose.
- **Unchecked:** venues for ITI/WinoBias/WinoGender; Brinkmann steering numbers; Samuel 2019 primary figures; Elpers Exp 2 n (189 recruited vs "~150").

Reading priority above stands; cite only the corrected versions.
