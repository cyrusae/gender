# Grammatical gender in nouns and language models: EDA dataset

Four small tables from a study of how language models represent grammatical gender
(Spanish *el/la*, German *der/die/das*). Released under **CC0 1.0**.

**What's not included, and why.** There are no words, dictionary definitions, word frequencies
or rating norms, because those come from sources that aren't CC0 (Wiktionary is CC BY-SA,
`wordfreq` is CC BY-SA, the Glasgow Norms are CC BY). Every value here was computed in this study:
- **Labels:** grammatical gender, animacy, why a noun was excluded.
- **Spelling facts:** word endings and length.
- **Tokenizer counts.**
- **Model measurements.**

## Files

### `noun_funnel.csv`: how a dictionary of ~50,000 nouns per language became ~100 training nouns

| column | meaning |
|---|---|
| `lang` | Spanish or German |
| `step_order` | order the filters were applied (1 = the full dictionary) |
| `step` | the filter; a noun is counted under the **first** filter it fails |
| `n_left` | nouns remaining after this step |
| `n_removed` | nouns this step removed |

The last step keeps nouns that a test model reliably knows the gender of, in spelling groups that
contain more than one gender. That step is what makes the training set unreadable from spelling alone.

### `filters.csv`: which filters exist for which language

| column | meaning |
|---|---|
| `step`, `lang` | a filter name from the funnel, crossed with both languages |
| `filter_applies` | `TRUE` if that language's pipeline has this filter |

The funnel lists only steps that removed something. Joining it to this table is how you tell
"this filter removed 0" apart from "this language has no such filter" (Spanish has no compound
filter; German has no *-o/-a* ending filter).

### `nouns.csv`: one row per dictionary noun (106,465 rows), with no words

| column | meaning |
|---|---|
| `noun_id` | random id; rows are shuffled, so they can't be matched back to dictionary order |
| `lang` | Spanish or German |
| `gender` | `m`, `f`, `n` (German only) or `several` (nouns with more than one gender) |
| `ending1`, `ending2` | last letter / last two letters |
| `length` | number of letters |
| `n_tokens` | pieces the Qwen3 tokenizer splits the word into (rarer words get more pieces) |
| `animate` | refers to a person or animal |
| `drop_reason` | the first filter it failed (same names as in the funnel), or `kept` |
| `in_training_set` | made the final training set |

### `layers_wide.csv`: where in each model the noun's gender can be read off

**Models:** six sizes of Qwen3 Base. The `-A3B` model is a mixture-of-experts model.

**How it was measured:** a direction fitted on training nouns is used to sort *unseen* test nouns.
AUC 0.5 = chance; 1.0 = perfect.

**Each test is built so that spelling points the wrong way:**
- **Spanish exceptions:** *el día* "day" is masculine despite ending in *-a*.
- **Spanish homographs:** words that are also verbs, read as nouns.
- **German compounds:** the gender comes from the last part (*der Tür·rahmen*), even when the first part has another gender.

| column | meaning |
|---|---|
| `model`, `params_b` | model name, size in billions of parameters |
| `lang`, `test` | which test |
| `layer`, `depth` | layer index; `depth` = layer / last layer (0–1), so models of different size line up |
| `auc_last`, `lo_last`, `hi_last` | AUC read at the noun's own last token, with a 95% bootstrap interval |
| `auc_after` etc. | the same, read one token *after* the noun |

The file is deliberately **wide**, with `last`/`after` as separate columns. Reshaping it is the
first transformation.

**Caveat:** the exceptions test has only about 27 nouns, so its intervals are wide. That is worth showing,
not hiding.

## Episodes: questions, transformations, checks, outputs

Each episode is one kind of analytical move. In each, the choice changes the *conclusion*, not only
the look. The suggested charts are in the next section.

| # | kind | question | the move | what changes |
|---|---|---|---|---|
| E1 | **missingness** | Is a blank a zero, or not applicable? | `count(lang, step)` wide, then `complete(..., fill = list(n = 0))` vs a `left_join` to `filters.csv` | Filling with 0 claims Spanish ran a compound filter that removed nothing. The join shows every blank is "no such filter". Spanish neuter is structurally absent too. |
| E2 | **missingness** | Do the models line up on a shared depth axis? | `count(model)` on `layers_wide`; bin `depth` with `cut_width(depth, .1)` and `count(model, bin)` | The models have 29–49 layer readings (embeddings + each layer), so bins hold unequal numbers of points per model. Averaging per bin weights models unevenly. |
| E3 | **aggregation** | What share of nouns is feminine; did a filter hit one gender harder? | the same `count(gender)`, with and without `lang` (and `drop_reason`) in the grouping | Pooled shares mix two languages with different gender systems and different filters. The per-language answer can differ from the pooled one. |
| E4 | **aggregation** | Which model reads German compound gender best? | three summaries per model: `mean(auc_last)`, `max(auc_last)`, `sum(lo_last > .5)` | The rankings can disagree. The last summary (the number of layers reliably above chance) is the one the study used for verdicts. Say why. |
| E5 | **transformation** | How destructive was each filter? | `n_removed / first(n_left)` (share of the dictionary) vs `n_removed / lag(n_left)` (share of what was left) | Same counts, different story. Spanish's ending filter removes about 14% of the dictionary but 89% of the nouns still in play. |
| E6 | **check** | Does the funnel add up? | the last `n_left` equals `sum(nouns$in_training_set)` per language; `sum(n_removed)` + final = start | A 30-second assert before any chart. A mismatch means a step is double-counted or missing. |
| E7 | **check** | Which lines are too thin to read? | `mutate(width = hi_last - lo_last)`, then a boxplot of `width` by `test` | The Spanish exceptions test (about 27 nouns) has much wider intervals, so its layer-to-layer wiggles are noise. |
| E8 | **check** | How much does "first reason only" shape the funnel? | narrate it; the export records one reason per noun | Reordering the filters would move counts between steps (animacy before frequency, for example). A limit of the summary, stated, not plotted. |
| E9 | **output** | What is the right form for the funnel finding? | a chart; a table (`slice_max(n_removed, n = 5)` per language); one sentence | "German keeps 0.23% of its dictionary": the same fact for three audiences. |

**A balanced set** for one session: E1, E3, E4, E6. That is one missingness check, one
aggregation, one choice of summary statistic and one sanity check.

## Questions, with alternative charts

Syntax lines are reminders, not scripts. Load with `d <- readr::read_csv("eda/<file>.csv")`.

**1. How many nouns did each filter remove?** (`noun_funnel`)
- Horizontal bars, with the steps in order: `geom_col(aes(n_left, reorder(step, -step_order))) + facet_wrap(~lang, scales = "free_y")`.
  Easy to read the long step names.
- A step or line chart of `step_order` against `n_left` shows the *shape* of the attrition: where the cliffs are.
- Add `scale_x_log10()`. German's compound filter (−11,272) dwarfs everything on a linear scale.
  The log scale makes the small filters visible. That's the comparison to narrate.
- Transform: `mutate(pct_left = n_left / first(n_left))`, then plot percentages so the two languages share one axis.

**2. How well does a word's ending predict its gender?** (`nouns`)
- Transform: `count(lang, ending1, gender)`, keep the commonest endings, then `group_by(lang, ending1)` and `mutate(share = n / sum(n))`.
- Filled bars: `geom_col(aes(ending1, n, fill = gender), position = "fill")`. Spanish *-a/-o* is almost one colour each.
- A heatmap, `geom_tile(aes(ending2, gender, fill = share))`, scales better to many endings and three German genders.
- Dot plot of the feminine share: `geom_point(aes(share, reorder(ending2, share))) + geom_vline(xintercept = .5)`. The most precise to read.

**3. What did each filter remove, and was it balanced by gender?** (`nouns`)
- `filter(drop_reason != "kept")`, then dodged bars: `geom_bar(aes(y = drop_reason, fill = gender), position = "dodge")`.
- Or proportions within each reason (`position = "fill"`). This answers "did a filter hit one gender harder?"
  For example, animacy removes a lot, and people nouns come in both genders.

**4. Are the genders different in length or tokenization?** (`nouns`)
- `geom_boxplot(aes(gender, n_tokens))` against `geom_violin(...)`. The boxplot compares medians; the violin shows shape.
- `geom_histogram(aes(length, fill = gender), position = "identity", alpha = .4, binwidth = 1)`.
  Integers want `binwidth = 1`, a nice detail to point out.
- Facet by `in_training_set` to see what the filters did to the distribution.

**5. Where in the model does gender live?** (`layers_wide`)
- Transform: `pivot_longer(c(auc_last, auc_after), names_to = "position", values_to = "auc")`.
  The intervals need either a second pivot or a `names_pattern`.
- Lines: `geom_line(aes(depth, auc, colour = model)) + geom_hline(yintercept = .5, linetype = 2) + facet_wrap(~test)`.
- Add uncertainty with `geom_ribbon(aes(ymin = lo_last, ymax = hi_last, fill = model), alpha = .15)`.
  Ribbons are better than error bars with 30–50 points per line.
- `facet_grid(test ~ position)`: the noun's own token against the token after.
- Alternative: a heatmap, `geom_tile(aes(depth, model, fill = auc))`, with no lines and every size at once.

**6. Do bigger models do better?** (`layers_wide`)
- `group_by(model, params_b, test) |> summarise(auc = mean(auc_last))`, then `geom_point(aes(params_b, auc)) + scale_x_log10()`.
- Comparing this summary with question 5's full lines is a point about what averaging hides.

## Provenance

- **How it was made:** built by `scripts/export_eda.py` from the study's frozen stimulus lists and result files.
- **Filter definitions:** they follow the study's documented screening rules.
- **Reason assignment:** when a noun fails several filters, only the first one is recorded.
