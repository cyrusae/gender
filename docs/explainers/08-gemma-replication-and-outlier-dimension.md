# 08: Replicating in a second model family (Gemma 3), and an outlier dimension

Work from 2026-10-22 to 10-25. Pre-registered in `docs/design/gemma-slim-prereg.md`, with an
amendment (outlier fix + validity gate) added before the affected analyses were rerun. The full
account of the measurement problem is in `docs/design/gemma-outlier-dimension.md`.

## 1. Why a second family, and why "slim"

Every result up to here came from one model family (Qwen3, plus EuroLLM-1.7B for checks). A
result that holds in one family might be a quirk of how that family was trained. Gemma 3 (Google)
was trained by a different team on different data, and it has public sparse autoencoders (Gemma
Scope 2), which kept SAE follow-ups possible.

**"Slim"** means: no new stimulus building and no re-tuning. The noun lists were frozen from the
Qwen work; the only change allowed was to drop nouns Gemma doesn't know, by the same article
check used before (the **known-check**). A list stays usable ("portable") if it loses at most 30%
of either gender. It lost at most 12%. Every comparison is between *conclusions* (verdicts), never
between the two families' vectors, which live in unrelated spaces.

Sizes: Gemma 3 4B and 12B base ("pt"), run in bf16 on a rented GPU. Phases 3–5 were in scope;
Phase 2 and the unsteered article tests (P29–P31) were not.

## 2. Technical gates (G0)

Gemma needs things Qwen didn't: exactly one `<bos>` token at the start (a doubled one silently
changes results), an attention-sink check, and a check that the "softcap" setting is handled. All
passed on both sizes before anything else ran.

## 3. What replicated, and what didn't

| result | Qwen3-8B | Qwen3-14B | Gemma-4B | Gemma-12B |
|---|---|---|---|---|
| German compounds take the head's gender | yes | yes | yes (33/33 layers) | yes (47/47) |
| masculine as a "default" (markedness) | no | no | no | no |
| Spanish ↔ German transfer (T1) | partial | partial | partial | partial |
| each language's direction sorts its pair nouns (T3) | yes | yes | yes | yes |
| Spanish steering moves social-gender adjectives | **yes** | no | **no** | **no** |
| German steering | no valid test | no valid test | no valid test | no valid test |
| German direction leans toward rarer words | **yes** | **yes** | **no** | **no** |

**How to read it.** Everything about how grammatical gender is *stored* replicates. Two things
don't:

- **Steering.** Qwen3-8B is now the only one of four models with a Spanish steering effect, so it
  looks like a property of that model rather than of language models in general. On Gemma-12B,
  pushing every word moved the readout by +0.10, but random directions moved it about as much: a
  push, not a gender-specific one. The positive control (a social-gender direction) moved it
  strongly (+0.29, +0.34), so the test could have seen an effect.
- **The German frequency lean.** The German lists are identical across families, so the lean in
  Qwen must come from Qwen's representations, not from which German nouns were chosen.

**German steering, two different failures.** On 4B the sentence frame fails its check (masculine
nouns 0.67, against a 0.70 bar). On 12B the frame passes, but random directions also flip German
agreement in up to half the nouns, so there's no gender-specific dose to steer with.

## 4. The outlier dimension

**What happened.** Several geometric measures came out impossible on raw Gemma activations:
cosines like −3.97 or +1.11, where a cosine must lie in [−1, 1].

**Why that's possible at all.** The project's cosines are *cross-fitted* (explainer 02): the
"length" and "overlap" pieces are each estimated from separate halves of the data, to remove a
noise bias. That makes each piece unbiased, but the ratio of three noisy pieces isn't bound to
[−1, 1], in the same way a reliability-corrected correlation can exceed 1.

**Why Gemma.** One coordinate of Gemma's hidden state carries 59–92% of the variation across
nouns at the tested layers (Qwen: under 2% for any coordinate), and the activations' overall size
is far larger (norms of about 7,000–66,000, against 70–240 for Qwen). That one coordinate swamps the pieces' noise.

**What was done, in order.**
1. Diagnosed using only quantities that don't involve any result being tested (variance shares,
   held-out gender decoding, a cosine between two gender quantities). A first version of this
   diagnostic had its own bug (it mixed the two halves) and was corrected before use.
2. Compared fixes: rescaling every dimension (rejected: damaged Qwen), dropping the dominant
   dimension(s), removing the top principal component, the model's own normalisation. The simple
   rule won: **drop any dimension carrying more than 10% of the variance.** It removes exactly
   one dimension in each Gemma model and none in Qwen, so Qwen results are untouched.
3. Pre-registered the fix plus a **validity gate**: a layer whose cosine is still impossible
   counts *against* overlap.
4. Reran. Every layer was valid on all four models; Qwen's verdicts came out the same.

**Practical lesson for anyone using Gemma.** Check how concentrated the variance is before
trusting any distance or angle. Raw Gemma probes also understate how well gender can be read out
(AUC about 0.58 raw, about 0.70 with the dimension removed).

## 5. Limits worth stating

- Two families is better than one but still few; both are recent decoder-only models.
- Gemma ran a reduced set (no Phase 2, P29–P31, Russian), and the 12B one-token-later check ran on
  a separate small pod afterwards.
- The outlier fix was chosen after seeing that raw values were impossible (not after seeing any
  verdict), and it was pre-registered before the rerun. That is a disclosed amendment, not a
  blind choice.
