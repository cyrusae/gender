# List audit after the 2026-11-11 findings (exploratory)

Question (PI, 2026-11-11): do issues found since each study was frozen change any result? New issues:
Z19 (attribute-word errors), Z21 (occupation glosses: secretario → registrar, maestro → maestro), and
the masculine-side flags built for the person-noun study (masculine form identical to English: director,
editor, constructor; masculine form with a non-person noun sense: trabajador, conductor, autor,
consejero, bailarín, grabador, animador, cazador, maestro). Each study was rerun in a sandbox with those
13 Spanish occupations removed (French: no masculine-side flags exist; *conservateur* already out of the
clean list). Frozen results untouched; outputs here.

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
