**Raking** is a way of fixing a lopsided sample by **weighting items instead of throwing them away**. Pollsters use it all the time. If a survey got too many older respondents, each younger respondent's answer gets counted a bit more so the totals match the population.

**The problem here:** in the 50 ending-matched training nouns, the masculine half is mostly concrete things (60%) and the feminine half mostly abstract ones (28% concrete). The masculine nouns are also somewhat rarer. A plain "average feminine vector minus average masculine vector" therefore mixes gender with "abstract vs concrete" and with frequency.

**What raking does:** it gives each noun a weight. A concrete feminine noun, being scarce, gets a weight above 1. A concrete masculine noun, being plentiful, gets a weight below 1. The weights are chosen so that, once applied, the masculine and feminine groups have the **same mix** of:
- final two letters (the ending balance the set was built for)
- concrete vs abstract
- frequency (low / mid / high)

**Why it's called raking:** you can't solve all three balances at once directly, so the procedure takes turns. Adjust the weights to fix the ending mix, then the concreteness mix, then frequency, then back to endings, and so on. Each pass undoes a little of the previous ones, so it repeats until all three settle. Combing through the data repeatedly, one variable at a time, is the "raking". The formal name is iterative proportional fitting.

**Then:** the gender direction becomes the *weighted* average feminine vector minus the *weighted* average masculine vector. If a difference remains after the groups are balanced on endings, concreteness and frequency, those three can't be what produces it.

**The trade-off:** no nouns are dropped, but items with large weights count for more, so the estimate gets noisier. That's one reason the weights are recomputed inside every bootstrap round: the confidence intervals then include that extra noise.

(The "failed" notification for the Phase 2 run is just the stopped run, which was killed deliberately before the amendment. That's expected.)