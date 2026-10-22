# Gemma 3 slim replication: pre-registration (DRAFT, awaiting PI)

Written 2026-10-22, before any Gemma number. Plan approved by the PI 2026-10-22 ("slim + Gemma
Scope step"); this document fixes the details. Follows `phases-3-5-plan.md` ("After the Qwen
RunPod session: Gemma 3 + Gemma Scope 2").

## Question

Do the Qwen3 conclusions hold in a second model family? Specifically:

1. **R1 (Phases 3–5):** the same pre-registered verdicts, from the same frozen noun lists, on
   Gemma 3 4B and 12B base (`google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`; 27B only if the
   whole session is under budget and the PI agrees at pod time).
2. **R2 (German frequency overlap):** does Gemma's German f−m direction also point toward
   "rarer word" (Qwen3 8B cos −0.26 to −0.69, 14B −0.19 to −0.52; Spanish ≤ 0.14)? If yes, the
   overlap is about German stimuli/German, not a Qwen quirk.
3. **R3 (Gemma Scope 2):** which SAE features carry the German frequency-aligned component? This
   is the only route we have into *what* that axis encodes. Exploratory-confirmatory split below.

We compare **conclusions** with Qwen, never vectors.

## What "slim" means (and doesn't)

- **Frozen lists, no rebuild.** Phase 3 `data/stimuli/phase3_final_v3.csv`, Phase 4
  `data/stimuli/phase4_pairs_final_v1.csv`, Phase 5 nouns `data/stimuli/phase5_nouns_v1.csv`,
  the Phase 2 list the Qwen results used. No Phase 0–2 rebuild with Gemma's tokenizer.
- **Known-check, filter only.** Each list's existing known-check (Phase 0 rule, `phase3_known`,
  same 1-nat margin, same frames) is run on Gemma; analyses use the nouns Gemma knows, exactly as
  for Qwen. Nothing is added. If a list loses more than 30% of either gender, that analysis is
  reported as "list not portable" rather than run on a lopsided set.
- **The one tokenizer-dependent set.** Phase 5's adjective readout requires adjectives whose m/f
  forms differ in exactly one final token (`phase5_stimuli`, `FAMILIES`). It gets a `gemma3`
  entry built by the existing rule, unchanged. This is a rule-built set, not hand-picked; the PI
  sees the list before the steering run.
- **Not in scope:** P29–P31 reruns, Russian, Phase 1 nonce work. Full replication only if slim
  disagrees with Qwen on a primary verdict.

## G0. Preconditions (on the pod, before any measured run; any failure stops the session)

| Check | Pass rule |
|---|---|
| dtype | bf16 only; a 20-prompt fp16/bf16 comparison is *not* run (fp16 is known to overflow) |
| loading | the text-only model loads from the multimodal 4B/12B checkpoint; next-token log-probs on 20 fixed German/Spanish prompts are finite and the top-1 continuation of "Der Hund" / "El perro" is a plausible word |
| softcap | read `final_logit_softcapping` from the config. If set, implement `c·tanh(x/c)` in `scoring.py` (currently raises `NotImplementedError`) and unit-test against the model's own forward logits (max abs diff < 1e-2 in bf16). If unset, the guard stays and is documented |
| BOS / sink guard | Gemma's tokenizer adds `<bos>`; confirm one BOS, then the existing newline buffer; the sink guard is quiet on measured positions |
| SAE index | Gemma Scope 2 "resid_post after layer L" ↔ our `hidden_states[L+1]`: the `sae.alignment()` check (matching index reconstructs best, by a clear margin over L and L+2) at three layers |

## G1 (R1). Frozen pipeline: what counts as "replicates"

For each primary Phase 3–5 verdict already reported for Qwen3, Gemma gets the same analysis with
the same thresholds, layers chosen by the same rule (relative depth, as in the Qwen runs — never
by looking at Gemma results). Per verdict, outcomes:

- **replicates** — same verdict on both Gemma sizes;
- **partly** — same verdict on one size;
- **does not replicate** — opposite verdict on both;
- **no valid test** — a gate (known-check portability, Phase 5 C1/gate) fails.

The write-up reports a table of Qwen verdict × Gemma verdict; no pooled "replication rate".

## G1 (R2). German frequency overlap

Confidence control (c) as pre-registered (`confidence-ctl-prereg.md`): cross-fitted cosine
between the f−m coefficient and the Zipf coefficient, with the within-cell×gender permutation
null, at the four Phase 5 layers. Overlap rule unchanged: cross term beats the null's 95th
percentile **and** |cos| ≥ 0.3. Then frequency-axis A, C and D (`frequency-axis-prereg.md`),
same rules. Readings:

- **German-general:** overlap at a majority of layers on both Gemma sizes, Spanish clean.
- **Qwen-specific:** no overlap on either Gemma size.
- **mixed:** anything else; reported as such.

## G2 (R3). Gemma Scope 2 features (German only)

Discovery on **training nouns only** (Phase 3 strat3 train split), per Phase 5 layer:

1. Encode the LAST-position activations. Score each feature by (i) its f−m effect (mean
   difference, within cells) and (ii) its Zipf slope, both in the same regression as the
   direction fit.
2. **Candidate set:** the top 20 features by |f−m effect| whose Zipf slope has the sign of the
   overlap. Selection uses training nouns only.
3. **Confirmatory test (held out):** on the held-out German nouns, does removing (zeroing in the
   SAE reconstruction) the candidate features reduce the |cos| of the refit direction with the
   frequency axis by at least half, while held-out gender decoding drops by less than 5 points?
   Control: 100 random sets of 20 features matched on activation frequency; the candidate effect
   must beat the 95th percentile.
4. **Descriptive only:** for each candidate feature, its top-activating nouns across the whole
   German lexicon (`data/lexicon/de_nouns.csv`), with English glosses and Wiktionary structure
   labels (compound / derived / loan), so we can see whether it looks like frequency,
   technical-register, compound-head or something else. Interpretation is labelled exploratory;
   no feature label comes from an LLM.

Also (descriptive, from `phases-3-5-plan.md` G2): whether separate "feminine" and "masculine"
features exist, or one feature with masculine nearer the default.

Phase 5 social-overlap via SAE features (plan G2 item 4) is deferred to its own prereg, if R1
gives a reason.

## Compute and logistics

- One chip type for all sizes: A100 80GB (~$1.59/h at last check; price stated again at pod
  time). Estimate 2–4 h total → about $3–6. 27B only with explicit PI OK.
- `HF_TOKEN` passed at pod time only, never on the volume. Official templates via
  `runpodctl template`. Attended session, anchored end markers.
- Activations saved and copied back; R2's A/C/D and all of R3 run on the Mac afterwards (CPU).
- Commit this prereg, the `gemma3` Phase 5 adjective list and the G0 code before the pod starts.

## Decisions for the PI

1. Adopt as drafted?
2. 27B: off unless budget allows (default), or never?
3. The 30% "list not portable" threshold OK?
