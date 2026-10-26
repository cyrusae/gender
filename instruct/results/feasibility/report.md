# Feasibility count

English pool 8289; with a translation table 7474.

## Usable per language (status ok)

| lang | name | ok | none | rare | mixed gender | untagged | gender split |
|---|---|---|---|---|---|---|---|
| de | German | 4741 | 773 | 497 | 1305 | 158 | f 2307 / m 1401 / n 1033 |
| ru | Russian | 4144 | 1130 | 535 | 1494 | 171 | m 1732 / f 1545 / n 867 |
| es | Spanish | 4656 | 1052 | 433 | 996 | 337 | f 2414 / m 2241 / n 1 |
| hi | Hindi | 2129 | 3619 | 858 | 482 | 386 | m 1377 / f 752 |
| fr | French | 4930 | 1274 | 258 | 586 | 426 | f 2552 / m 2378 |
| it | Italian | 4143 | 1763 | 274 | 671 | 623 | f 2124 / m 2017 / n 2 |
| pt | Portuguese | 4652 | 1336 | 342 | 584 | 560 | f 2470 / m 2181 / n 1 |
| pl | Polish | 3916 | 1821 | 854 | 701 | 182 | f 1629 / m 1601 / n 686 |
| cs | Czech | 3884 | 2404 | 458 | 557 | 171 | f 1695 / m 1469 / n 720 |
| el | Greek | 3382 | 2714 | 933 | 259 | 186 | f 1841 / n 1164 / m 377 |
| ar | Arabic | 2943 | 3347 | 390 | 505 | 289 | m 1908 / f 1035 |
| he | Hebrew | 2568 | 4080 | 238 | 165 | 423 | m 1477 / f 1088 / n 3 |
| nl | Dutch | 3417 | 1923 | 509 | 723 | 902 | f 1419 / n 1050 / m 948 |
| sv | Swedish | 951 | 1884 | 790 | 0 | 3849 | n 948 / m 2 / f 1 |
| fi | Finnish | 5393 | 628 | 1453 | 0 | 0 |  |
| tr | Turkish | 3837 | 3075 | 562 | 0 | 0 |  |
| ja | Japanese | 5241 | 1968 | 265 | 0 | 0 |  |
| hu | Hungarian | 3843 | 2182 | 1449 | 0 | 0 |  |

## Nouns usable in every language of a set, by concreteness

| set | languages | total | abstract (<2.5) | mid | concrete (>=3.5) | with Glasgow valence |
|---|---|---|---|---|---|---|
| core4 | de ru es hi | 809 | 178 | 182 | 449 | 418 |
| core6 | de ru es hi fi ja | 704 | 149 | 154 | 401 | 384 |
| core7 | de ru es hi fi ja tr | 613 | 116 | 129 | 368 | 352 |
| core8 | de ru es hi ar fi ja tr | 465 | 86 | 94 | 285 | 264 |
| core7+fr | de ru es hi fi ja tr fr | 559 | 109 | 121 | 329 | 322 |
| core7+it | de ru es hi fi ja tr it | 524 | 97 | 110 | 317 | 311 |
| core7+pt | de ru es hi fi ja tr pt | 552 | 108 | 114 | 330 | 321 |
| core7+pl | de ru es hi fi ja tr pl | 512 | 93 | 105 | 314 | 297 |
| core7+el | de ru es hi fi ja tr el | 487 | 95 | 105 | 287 | 278 |
| core7+ar | de ru es hi fi ja tr ar | 465 | 86 | 94 | 285 | 264 |
| core7+he | de ru es hi fi ja tr he | 424 | 73 | 90 | 261 | 253 |
| core7+nl | de ru es hi fi ja tr nl | 485 | 89 | 104 | 292 | 280 |
| core7+cs | de ru es hi fi ja tr cs | 545 | 101 | 113 | 331 | 314 |
| core7+sv | de ru es hi fi ja tr sv | 137 | 18 | 34 | 85 | 79 |
| core7+hu | de ru es hi fi ja tr hu | 543 | 99 | 111 | 333 | 330 |

## Gender patterns on the core8 set (465 nouns)

66 distinct patterns (order de ru es hi ar). Most common:

| pattern | n |
|---|---|
| mmmmm | 49 |
| ffffm | 30 |
| fffmf | 28 |
| nmmmm | 25 |
| fffff | 25 |
| fffmm | 21 |
| mmmmf | 21 |
| fnfmm | 14 |
| fnffm | 10 |
| mfmmm | 10 |
| mmmfm | 10 |
| nnmmm | 9 |
| nmmfm | 8 |
| fmmmm | 8 |
| fmffm | 8 |

## Pairwise gender agreement (share of nouns with the same gender), all usable pairs

|    |   de |   ru |   es |   hi |   fr |   it |   pt |   pl |   cs |   el |   ar |   he |   nl |   sv |
|:---|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| de | 1    | 0.51 | 0.59 | 0.4  | 0.59 | 0.58 | 0.59 | 0.56 | 0.57 | 0.58 | 0.38 | 0.51 | 0.71 | 0.48 |
| ru | 0.51 | 1    | 0.57 | 0.45 | 0.57 | 0.57 | 0.55 | 0.72 | 0.68 | 0.37 | 0.46 | 0.57 | 0.47 | 0.21 |
| es | 0.59 | 0.57 | 1    | 0.54 | 0.85 | 0.88 | 0.91 | 0.62 | 0.59 | 0.49 | 0.54 | 0.63 | 0.52 | 0    |
| hi | 0.4  | 0.45 | 0.54 | 1    | 0.56 | 0.57 | 0.55 | 0.47 | 0.47 | 0.31 | 0.57 | 0.57 | 0.35 | 0    |
| fr | 0.59 | 0.57 | 0.85 | 0.56 | 1    | 0.86 | 0.85 | 0.62 | 0.58 | 0.49 | 0.53 | 0.65 | 0.52 | 0    |
| it | 0.58 | 0.57 | 0.88 | 0.57 | 0.86 | 1    | 0.88 | 0.61 | 0.6  | 0.48 | 0.54 | 0.63 | 0.51 | 0    |
| pt | 0.59 | 0.55 | 0.91 | 0.55 | 0.85 | 0.88 | 1    | 0.61 | 0.58 | 0.5  | 0.53 | 0.63 | 0.51 | 0    |
| pl | 0.56 | 0.72 | 0.62 | 0.47 | 0.62 | 0.61 | 0.61 | 1    | 0.76 | 0.42 | 0.48 | 0.57 | 0.51 | 0.2  |
| cs | 0.57 | 0.68 | 0.59 | 0.47 | 0.58 | 0.6  | 0.58 | 0.76 | 1    | 0.42 | 0.46 | 0.55 | 0.5  | 0.22 |
| el | 0.58 | 0.37 | 0.49 | 0.31 | 0.49 | 0.48 | 0.5  | 0.42 | 0.42 | 1    | 0.28 | 0.39 | 0.55 | 0.51 |
| ar | 0.38 | 0.46 | 0.54 | 0.57 | 0.53 | 0.54 | 0.53 | 0.48 | 0.46 | 0.28 | 1    | 0.61 | 0.35 | 0    |
| he | 0.51 | 0.57 | 0.63 | 0.57 | 0.65 | 0.63 | 0.63 | 0.57 | 0.55 | 0.39 | 0.61 | 1    | 0.44 | 0    |
| nl | 0.71 | 0.47 | 0.52 | 0.35 | 0.52 | 0.51 | 0.51 | 0.51 | 0.5  | 0.55 | 0.35 | 0.44 | 1    | 0.56 |
| sv | 0.48 | 0.21 | 0    | 0    | 0    | 0    | 0    | 0.2  | 0.22 | 0.51 | 0    | 0    | 0.56 | 1    |

(Three-gender vs two-gender pairs can't agree on neuter nouns, so their agreement is lower by construction.)


## Tag check against our Wiktionary lexicons (de/es/ru)

- de: 95% of words found in the lexicon; gender agrees for 96.0% of those
- es: 99% of words found in the lexicon; gender agrees for 97.6% of those
- ru: 97% of words found in the lexicon; gender agrees for 99.2% of those
