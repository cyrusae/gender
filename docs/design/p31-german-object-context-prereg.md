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
