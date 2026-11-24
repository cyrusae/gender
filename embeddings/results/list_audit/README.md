# List audit after the 2026-11-11 findings (exploratory)

Question (PI, 2026-11-11): do issues found since each study was frozen change any result? New issues:
Z19 (attribute-word errors), Z21 (occupation glosses: secretario → registrar, maestro → maestro), and
the masculine-side flags built for the person-noun study (masculine form identical to English: director,
editor, constructor; masculine form with a non-person noun sense: trabajador, conductor, autor,
consejero, bailarín, grabador, animador, cazador, maestro). Each study was rerun in a sandbox with those
13 Spanish occupations removed (French: the French flag rules did not exist yet; see the 2026-11-23
addendum below). Frozen results untouched; outputs here.

| Study | Change | Verdicts |
|---|---|---|
| Word-pair mechanism (Spanish, 42 of 54 occupations) | S_bls β 0.32 [0.05, 0.54] (was 0.30, above 0) | unchanged |
| Expanded cues (Spanish) | intrusion 0.33 [0.20, 0.44] (was 0.29); Shift −0.12, Hybrid −0.11 | unchanged (one descriptive invariant-gap cell "no clear change" → "reduces") |
| Generic masculine (flagged pairs dropped AND name pairs maría/juan, louise/gabriel and French duplicates removed from the direction) | ES supported, FR inconclusive | unchanged (descriptive no-homograph ES subset: "at neutral" → "not shown") |
| Zero point, Spanish (corrected attributes, 42 clean occupations) | native −1.22 [−4.05, 1.29]; English −2.08 [−4.58, 0.37] | unchanged (inconclusive) |
| **Zero point, French (corrected attributes, 22 clean occupations)** | **native −1.91 [−3.34, −0.39]**; English −1.99 [−3.33, −0.64] | **native flips back to "asymmetry"** |

Not rerun: C1, gender ValNorm, crime probe (occupations only enter as Zhou's Shift input or as an
exclusion list; their targets are other noun lists); person-noun baseline (built with these flags).

Reading: everything holds except the French native zero point, which is borderline: it reads
"asymmetry" with Zhou's lists (−1.91) and with corrected attributes on the clean occupation list
(−1.91), and "inconclusive" with corrected attributes on Zhou's 23 occupations (−1.53; dropping only
*conservateur* 'tory', not an occupation, moves it back). It should be reported as borderline, not as
either verdict.

## Addendum 2026-11-23: French flagged occupations (PI question)

The French masculine-side flags were built later (person-noun study, 2026-11-19). Applied to Zhou's 23 French
occupations they flag three: *conservateur* ('conservative; preservative', already out of the clean list),
*serveur* ('server') and *ouvrier* ('operator'). Rerun in a sandbox without *serveur* and *ouvrier* (20
occupations; outputs in `fr_flagged/`, released centring as frozen):

| Study | Frozen (22 or 23 occupations) | Without serveur, ouvrier (20) | Verdicts |
|---|---|---|---|
| Zero point, French, native attributes | −1.53 [−3.25, 0.41] (23); −1.91 [−3.34, −0.39] (22 clean) | **−1.44 [−3.03, 0.33] inconclusive** | borderline confirmed: flips with one or two occupations |
| Zero point, French, English attributes | −2.03 [−3.39, −0.63] | −1.83 [−3.14, −0.45] | unchanged (feminine further) |
| Expanded cues, French intrusion | 0.191 [0.007, 0.371] present | 0.233 [0.043, 0.411] present | unchanged |
| — after Shift / Hybrid (descriptive) | 0.083 / 0.130 inconclusive | 0.120 inconclusive / 0.172 [0.001, 0.335] present | Hybrid level inconclusive → present |
| — change after Shift / Hybrid | −0.108 / −0.062 reduces | −0.113 / −0.060 reduces | unchanged |
| Generic masculine, French | Δρ 0.13, inconclusive | Δρ 0.10, inconclusive | unchanged |

Feminine-side homographs (*cuisinière* 'stove') are not part of this set in either language; see Z21.

## Addendum 2026-11-24: expanded-cue test with the described (midpoint) centring, flagged occupations removed

The paper presents the midpoint centring as primary, so the robustness check was rerun with it
(`ZHOU_CENTRE=midpoint`; Spanish 42 occupations, French 20; outputs in `midpoint_flagged/`).

| | Primary, midpoint (54 / 22) | Flagged removed, midpoint (42 / 20) |
|---|---|---|
| ES intrusion, original | 0.291 [0.186, 0.396] present | 0.326 [0.202, 0.444] present |
| ES change after Shift / Hybrid | −0.103 / −0.086, both reduce | −0.113 [−0.161, −0.069] / −0.107 [−0.175, −0.049], both reduce |
| FR intrusion, original | 0.191 [0.002, 0.370] present | 0.233 [0.040, 0.419] present |
| FR change after Shift / Hybrid | −0.096 reduces / −0.047 [−0.100, 0.002] no clear change | −0.098 [−0.134, −0.065] reduces / −0.043 [−0.095, 0.002] no clear change |
| FR level after Shift / Hybrid (descriptive) | 0.096 / 0.144, inconclusive | 0.135 inconclusive / 0.189 [0.014, 0.364] present |

Verdicts unchanged in both languages.
