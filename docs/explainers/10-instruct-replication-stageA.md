# 10: Do the results hold in chat models? (instruct replication, stage A)

Work from 2026-10-26/27. Pre-registered in `docs/design/instruct/instruct-replication-prereg.md` (adopted
by the PI 2026-10-26, amendment 1 on 2026-10-27). So far only the Gemma-3-4B-it pilot has run;
Gemma-3-12B-it and Qwen3-8B/14B (non-thinking) await the PI.

This is separate from the character study (`character/`, explainers C0–C1), which asks what chat
models *say*. Here the question is about representations, as in phase 1.

## 1. The question

Everything in phase 1 used base (pretrained) models. People talk to chat-tuned ("instruct")
versions, which start from the same checkpoint and are then trained further to follow
instructions. Does that further training change how grammatical gender is stored? And does it
matter how the noun is presented: bare, or inside a chat conversation?

The project's rule "base models only" got a scoped exception for this study.

## 2. Design

**Models.** The instruct sibling of each phase-1 base model: Gemma-3-4B-it and 12B-it, Qwen3-8B
and 14B (Qwen run with thinking switched off). Same pod, same precision (bf16), same frozen noun
lists, filtered by the same known-check.

**Five input formats.** The same noun, wrapped differently:

| format | what the model sees | "AFTER" position |
|---|---|---|
| raw (primary) | the bare noun, exactly as in phase 1 | the next newline |
| A-en | chat: user says *"Say one word."*, assistant says the noun | the end-of-turn token |
| A-something | the same with *"Say something."* | end-of-turn |
| A-lang | the same in the noun's language: *Dime algo.* / *Sag etwas.* / *Скажи что-нибудь.* | end-of-turn |
| U | the noun as the user's own message | the next template token |

The A-lang wordings avoid "one word" on purpose: *una palabra* is feminine and *Wort* neuter, so
they would prime a gender right before the noun. A-en vs A-something differ only in wording;
A-something vs A-lang differ only in language, so the two contrasts can be told apart.

**Gates.** The article test and known-check (list portable if ≤ 30% lost per gender); the
outlier-dimension check and fix (explainer 08); and a **boundary check**: if the chat template
fused the noun's last piece with the next token, that noun would be dropped. None were.

**Analyses.** Unchanged code and rules: Phase 3 (compounds follow the head, markedness), Phase 4
(transfer T1, pair test T3, Russian T4, overlap T2), and the German frequency cosine. Each verdict
is labelled **same** as the base sibling, **differs**, or **no valid test**. Steering (stage B)
runs only if the PI decides so before any steering number exists.

## 3. The Gemma-3-4B-it pilot

**Every pre-registered Phase 3–4 verdict is the same as the base model's, in all five formats.**
German compounds follow their head (23–33 of 33 layers), there's no markedness, Spanish↔German
transfer is partial, and pairs follow each language's gender. One token after the noun, the
effects are gone, as in the base model.

**The one rule-level difference isn't a finding.** On raw, A-en and U the frequency-overlap rule
says "overlap". Amendment 1 added bootstrap intervals to every such cosine, and on Gemma-4B-it
those intervals include 0 (and many resamples are undefined). By contrast Qwen3's German
frequency lean has intervals excluding 0 at 3 of 4 layers on both sizes, so it is well determined.

**Exploratory: an English instruction pulls the languages apart.** With *"Say one word."* or
*"Say something."* before the noun, the cross-language measures collapse:

| measure | base | raw | A-en | A-something | A-lang | U |
|---|---|---|---|---|---|---|
| German→Spanish transfer (layers of 33) | 26 | 25 | 2 | 2 | 17 | 20 |
| T2 overlap above null (layers) | 31 | 24 | 3 | 3 | 27 | 18 |
| T2 cosine, median | 0.47 | 0.50 | −0.17 | 0.00 | 0.56 | 0.43 |

Within each language gender still works (T3, T4 intact). With an in-language prompt or the noun as
the user's turn, everything looks like the base model. One possible reading: an English frame puts
the model "in English", and the Spanish and German gender directions stop lining up there. It's
one model and exploratory, so it's a lead, not a result.

**Exploratory: instruct tuning forgets rare words' genders.** On the German known-check, 760
nouns the base model knew became unknown (301 the reverse). Loss rises with rarity (30% of the
rarest quarter vs 9% of the commonest) and with token count (7% for one-token words, 32% for six
or more), equally across genders. Masculine nouns that went wrong mostly got *die* (276) rather
than *das* (113). The instruct model is also *more confident* on average (median margin 3.97 vs
2.95 nats) and more often confidently wrong (13% vs 7%).

## 4. Limits worth stating

- One pilot model so far; the format effects need the other three siblings before they mean much.
- A-lang wordings for German and Russian are unverified translations (marked claude-unverified).
- Representations, not behaviour: whether a chat model *says* gendered things about objects is
  the character study's question.
