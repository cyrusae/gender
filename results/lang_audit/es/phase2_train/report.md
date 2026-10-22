# Language audit: es / phase2_train

165 nouns; by gender {'m': 132, 'f': 33}.

## Flags

- initial 'a' (n=17): gender mix departs by 0.20; no f ({'m': 17})
- tokens: d_f_vs_m = -0.41 (within cells)
- zipf: d_f_vs_m = +0.71 (within cells)
- no word-structure rule for this language yet (playbook Stage 3)

## Spelling

- Gender from the last 1/2/3 letters (leave-one-out accuracy): 0.81 / 0.76 / 0.81 (majority class 0.80).
- Within cells: cell majority 0.76; last 3/4 letters add +0.00 / -0.01.

## Balance (within-cell standardised differences vs the first gender)

- **length**: coverage 1.00, mean_m 6.85, mean_f 6.58, d_f_vs_m -0.08
- **tokens**: coverage 1.00, mean_m 2.42, mean_f 2.21, d_f_vs_m -0.41
- **zipf**: coverage 1.00, mean_m 3.13, mean_f 3.71, d_f_vs_m 0.71
- **aoa**: coverage 0.94, mean_m 8.20, mean_f 8.57, d_f_vs_m -0.07
- **concreteness**: coverage 0.96, mean_m 4.09, mean_f 3.79, d_f_vs_m -0.27

## Word structure

- complex share by gender: None
