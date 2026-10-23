# Grammatical gender in nouns and language models: EDA dataset

Three small tables from a study of how language models represent grammatical gender
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
| `auc_after` etc. | the same, read one token *after* the noun (missing for some models/tests) |

The file is deliberately **wide**, with `last`/`after` as separate columns. Reshaping it is the
first transformation.

**Caveat:** the exceptions test has only about 27 nouns, so its intervals are wide. That is worth showing,
not hiding.

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
