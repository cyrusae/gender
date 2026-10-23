# P31: the German version of P30 (ADOPTED, PI 2026-10-21)

Drafted 2026-10-17 (PI asked to start; first-token readout, *das* reported alongside, broad probe).
Only a 0.6B smoke test (8 words) has run. Code: `src/gbleed/p31.py`. Exploratory.

## The question in plain words

P30 found that on Qwen3, a feminine Spanish article on a made-up noun read as a *thing* still
pulls toward English-feminine-rated adjectives. Is that specific to Spanish? German makes a clean
second test:
- German adjectives after *ist* don't change form for gender (*ist sehr groß* for every noun),
  so nothing can leak through word forms;
- German has three genders, so *das* (neuter) gives a third point.

## Stimuli (built by rule; the PI doesn't read German, so English glosses only)

**Made-up nouns.** Wuggy (German) on common inanimate lexicon nouns, which only lend their shape
(never a held-out test word of any phase). Kept if:
- **the ending doesn't give gender away**: for each of the word's last 1, 2 and 3 letters, no
  gender holds more than 50% of the lexicon nouns with that ending (where at least 50 such nouns
  exist). This removes *-e* (mostly feminine), *-ung*/*-heit*/*-keit*/*-schaft* (feminine) and
  *-chen* (neuter) automatically;
- absent from German Wiktionary (all words and forms), zero frequency in six languages;
- not shaped like a past participle (*ge-...-t*, or an inseparable prefix *be-/ver-/er-/zer-/ent-/emp-* ...*-t*,
  as in *gekauft*, *verkauft*), which could read as a verb form or adjective;
- capitalised, 3–10 letters. Target 300; count: see build output.

**Real person nouns (the yardstick).** German nouns declined like adjectives, which take *der*
for a man and *die* for a woman with the same form: *der/die Angestellte* "employee". From
Wiktionary by rule:
- adjectival declension;
- a feminine *-e* entry with a masculine form, glossed "female equivalent of ...: female Y", where
  Y is a WordNet person noun;
- no digits or hyphens; not a form of a readout adjective; Zipf ≥ 3.

That gives 41 nouns; the PI dropped 9 (below), leaving **32** (`PERSON_DROP` in the code).

## Design

Articles: *der* / *die* / *das* for the made-up nouns; *der* / *die* only for the real person nouns.

| Context | Wording (masculine shown) | Gloss |
|---|---|---|
| object | O1 *Der Flitt, den ich gestern gekauft habe, ist sehr ___* | "The flitt I bought yesterday is very ___" |
| object | O2 *Der Flitt, den ich in die Schublade gelegt habe, ist sehr ___* | "...that I put in the drawer..." |
| neutral | N1 *Der Flitt ist sehr ___* | P20's German frame |
| person | P1 *Der Flitt, mit dem ich gestern gesprochen habe, ist sehr ___* | "...that I talked with yesterday..." |
| person | P2 *Der Flitt, den ich gestern kennengelernt habe, ist sehr ___* | "...that I met yesterday..." |

The relative pronoun agrees in gender and case: accusative *den/die/das*, dative after *mit*
*dem/der/dem*. Getting that wrong would make one gender's sentences ungrammatical. *Ist*
(singular) rules out reading *die* as plural.

**Readout.** Next token after *sehr*: the first token of each German translation adjective,
kept when it's at least 3 letters and not shared with another listed adjective. That's 147
adjectives in Qwen3, 177 in EuroLLM. Sensitivity: single-token adjectives only (58 / 67). Ratings
as P30 (Glasgow, via the English gloss), with German word frequency as the frequency covariate.

## Checks (as P30)

**C1, manipulation check.** On the stem with *ist* in place of *ist sehr*:
- person words: *ein Mensch, eine Person, ein Mann, eine Frau* ("a human, a person, a man, a
  woman"), probabilities summed;
- thing words: *ein Ding, ein Gegenstand, ein Objekt, eine Sache* ("a thing, an item, an object,
  a thing/matter"), probabilities summed.

*Mann*/*Frau* are sexed, but the probe is a median over all articles, and summing the two
largely cancels the article's sex bias (PI, 2026-10-17).

**The broad probe's zero point isn't meaningful** (found in the 0.6B smoke test, fixed on
principle 2026-10-21, PI agreed): the person words include very common ones (*Mann*, *Frau*) and
the thing words rarer ones (*Gegenstand*, *Objekt*), so the summed probe leans "person" in every
sentence. Smoke (0.6B): object O1/O2 broad +0.08/+0.45 vs narrow −2.23/−0.97. So the broad probe
is judged only *relative to N1*, where base rates cancel, and the matched narrow pair (*eine
Person* vs *ein Ding*, as P30's *una persona*/*una cosa*) carries the sign:
- object wordings: **narrow** median < 0, and **broad** median at least 1 nat below N1's;
- person wordings: broad median at least 1 nat above N1's;
- real person nouns' N1 broad median > 0.

0.6B fails the object rule either way (O1 is 0.8 nats below N1), so the fix isn't fitted to pass.

**C2.** Combined share of the read tokens ≥ 5% per wording.

## Measures (as P30)

- S(context): graded score of the *die* − *der* shift (primary); *das* − *der* reported.
- Precondition: S(neutral)'s lower 95% bound > 0.
- Primary S(object):
  - lower 95% bound > 0 → **the pull survives an object reading**;
  - upper bound < ½ S(neutral) → **mostly the "a woman" guess**;
  - otherwise **partial**.
- Reported: per context, the real-person scale, the single-token sensitivity, the *das*
  contrasts.

## Models and cost

As P30: Qwen3 8B/14B on an L40S (about 15 min, roughly $0.30); EuroLLM-1.7B and Qwen3 1.7B/4B on
the Mac (free). Smoke test first.

## Decisions (PI, 2026-10-21)

Adopted (PI, 2026-10-21), with the C1 fix above and these 9 drops from the 41 person nouns:
   - *Erste* "the first", an ordinal;
   - *Große* "adult", also "the big one";
   - *Grüne* "Green" party member, a colour;
   - *Farbige* "coloured person", dated and offensive;
   - *Homosexuelle*, *Heterosexuelle*, *Prostituierte*, *Geliebte* "beloved/mistress": tied
     directly to sex or sexuality;
   - *Rothaarige* "redhead", appearance.

   Glosses are the extracted Wiktionary ones, except three marked * where the extraction picked a
   neighbouring sense and Claude substituted the usual meaning (`claude-unverified`; the analysis
   doesn't use glosses). Full list: Erste (winner), Große (adult), Linke (leftist), Grüne (Green), Bekannte
   (acquaintance), Erwachsene (adult), Unbekannte (stranger), Älteste (elder), Freiwillige
   (volunteer), Angehörige (relative), Verwandte (relative), Angestellte (employee),
   Professionelle (professional), Geliebte (beloved), Reisende (traveller*), Verdächtige
   (suspect), Studierende (student), Geistliche (cleric), Homosexuelle (homosexual), Beauftragte
   (commissioner), Angeklagte (accused), Beteiligte (participant), Prostituierte (prostitute),
   Überlebende (survivor), Obdachlose (homeless person), Farbige (coloured person),
   Auszubildende (apprentice), Gelehrte (scholar), Sachverständige (expert), Befragte
   (respondent*), Vorgesetzte (superior), Rothaarige (redhead), Delegierte (delegate), Begabte
   (gifted person*), Ungläubige (unbeliever), Außenstehende (outsider), Gesandte (envoy),
   Heterosexuelle (heterosexual), Staatsangehörige (citizen), Bedienstete (employee),
   Begünstigte (beneficiary).

## Results (2026-10-21/22; Qwen3 8B/14B on L40S bf16 at commit 0f25e03; EuroLLM-1.7B and Qwen3 1.7B/4B on the Mac fp16; `results/p31/`)

Graded score of the *die* − *der* shift, made-up nouns, 95% CI.

| | EuroLLM-1.7B | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Qwen3-14B |
|---|---|---|---|---|---|
| object wordings valid (C1) | O1, O2 | O1 (O2 fails) | O1, O2 | O1, O2 | **none** (broad probe 0.1–0.3 nat below N1) |
| person wordings valid | none | P1, P2 | P1 | P1 | P1, P2 |
| S(neutral) | +0.035 [0.029, 0.040] | +0.018 [0.013, 0.023] | +0.020 [0.015, 0.024] | +0.002 [−0.004, 0.006] | +0.034 [0.030, 0.039] |
| **S(object)** | **+0.081** [0.077, 0.085] | **+0.035** [0.032, 0.039] | **+0.043** [0.039, 0.046] | +0.002 [−0.001, 0.005] | (+0.025, exploratory) |
| S(person) | n/a | +0.042 | +0.054 | +0.081 | +0.074 |
| real person nouns, neutral | +0.014 (CI includes 0) | +0.009 (CI includes 0) | +0.059 | +0.059 | +0.090 |
| *das* − *der*, object | +0.045 | 0.000 | +0.001 | 0.000 | (+0.002) |
| **verdict** | **pull survives** | **pull survives** | **pull survives** | **no neutral pull to explain** | **no valid test** |

**Reading (as pre-registered):**
- **The object pull replicates in German on three of five models.** On EuroLLM-1.7B and Qwen3
  1.7B/4B, the made-up noun reads as a thing (narrow probe −0.8 to −2.7 nats), and *die* still
  draws feminine-rated adjectives more than *der* does, by about twice the neutral sentence's
  shift. That is the same "object > neutral" pattern as Spanish.
- **Qwen3-8B shows no pull outside person sentences.** **14B** failed the probe check.
  Descriptively, with the margin waived, it shows +0.025, below its neutral +0.034.
- **Neuter *das* is ≈ 0 against *der* in objects** on every Qwen size, so the pull is
  feminine-specific, not "any non-masculine article". EuroLLM is the exception (+0.045).

**Compared with Spanish (P30):**

| | EuroLLM-1.7B | Qwen3-1.7B | 4B | 8B | 14B |
|---|---|---|---|---|---|
| Spanish S(object) | +0.008 (no neutral pull) | +0.038 | +0.026 | +0.080 | +0.124 |
| German S(object) | +0.081 | +0.035 | +0.043 | +0.002 | invalid |

The small Qwen3 models agree across languages. The large ones show it in Spanish only, and
EuroLLM in German only. It isn't a single language-general effect that grows with size.

**Limits to state:**
- **Hardware is mixed.** The Mac runs are fp16 and the pod runs bf16, so cross-size comparisons are rough.
- **The scale check is weak on two models.** On EuroLLM and Qwen3-1.7B, real person nouns barely
  shift in the neutral sentence (CI includes 0). So "how big is a real woman's shift" is poorly
  measured there.
- **Frequency-axis caveat.** The German gender *direction*
  (`frequency-axis-prereg.md`) aligns with low frequency. P31 uses no direction, but if
  feminine-rated adjectives differ in frequency, an article that shifts a frequency-related
  property could mimic the pull. The confidence control (a) re-scores P31 with a baseline-
  probability covariate.
- **Possible routes.** First-token readout; the gender ratings are of the English glosses.

## Sensitivity (2026-10-23): EuroLLM with its `<s>` BOS token

P30/P31 prompts were built without special tokens, which is correct for Qwen3 (no BOS) but drops EuroLLM's default `<s>`. Rerun with `GBLEED_BOS_ALL=1`, same code and lists (`results/p31_bos/`). The verdict is unchanged ("pull survives an object reading"), but its size moves:

| EuroLLM-1.7B | without `<s>` (as run) | with `<s>` |
|---|---|---|
| S(object) | 0.081 | 0.073 |
| S(neutral) | 0.035 | 0.057 |
| object / neutral | 2.32 [2.06, 2.69] | 1.29 [1.22, 1.37] |
| *das* − *der*, object | 0.045 | 0.026 |
| real person nouns, neutral | 0.014 | 0.074 |

With `<s>`, the object pull is about as large as the neutral pull, not twice it. The real-person yardstick also becomes clearly larger than either, which is the pattern a working social readout should show. The as-run numbers remain the pre-registered result; the `<s>` version is the more trustworthy description of EuroLLM.
