# Language audit: de / phase3_train_mf

91 nouns; by gender {'m': 64, 'f': 27}.

## Flags

- zipf: d_f_vs_m = +0.59 (within cells)
- word structure: complex share by gender {'f': 0.0, 'm': 0.1875}
- word structure (Wiktionary): complex share by gender {'f': 0.1111111111111111, 'm': 0.265625}

## Spelling

- Gender from the last 1/2/3 letters (leave-one-out accuracy): 0.77 / 0.74 / 0.77 (majority class 0.70).
- Within cells: cell majority 0.74; last 3/4 letters add +0.01 / -0.04.

## Balance (within-cell standardised differences vs the first gender)

- **length**: coverage 1.00, mean_m 6.94, mean_f 6.19, d_f_vs_m -0.18
- **tokens**: coverage 1.00, mean_m 2.62, mean_f 2.33, d_f_vs_m -0.21
- **zipf**: coverage 1.00, mean_m 3.28, mean_f 3.77, d_f_vs_m 0.59
- **aoa**: coverage 0.88, mean_m 7.89, mean_f 7.44, d_f_vs_m -0.07
- **concreteness**: coverage 0.95, mean_m 4.26, mean_f 4.00, d_f_vs_m -0.15

## Word structure

- hand rule, complex share by gender: {'f': 0.0, 'm': 0.1875}
- Wiktionary templates (transparent), complex share by gender: {'f': 0.1111111111111111, 'm': 0.265625}
