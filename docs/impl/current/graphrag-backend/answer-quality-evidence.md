# Answer-Quality Evidence

Part of the [GraphRAG backend](../graphrag-backend.md) area of the
[current implementation index](../../current.md).

## Answer-quality evidence

The sweep above is model-independent: it measures what the context CARRIES, never what the model
does with it. `llb compare-answer-quality` (`make compare-answer-quality`) closes that gap. It
scores the identical item set END TO END under two retrieval lanes with the standard `run-eval`,
then compares the ANSWERS per question-type slice with the same paired bootstrap the sweep uses.

Both this lane and the fusion sweep read the calibrated paired sign-flip p, so both state how far
the deciding row sits from the cut: every paired delta carries `randomization_p`, diagnostic
`p_positive`, and a `(borderline)` flag; the reason gains a shared clause when a neighbouring
confidence convention would read it differently, and `report.md` renders a boundary table over
the focus slice. Two of the first three
recorded answer-quality comparisons and three of the six recorded fusion sweeps are now qualified
that way; see
[how settled a paired reading is](../rag-core/paired-verdicts.md#how-settled-a-paired-reading-is----p_positive-and-the-borderline-flag).

Three properties make the comparison readable:

- **The lanes are named by sweep row label.** `vector`, `graph/<strategy>`,
  `fused/<strategy>@<weight>[/d<depth>][/i<identity>][/r<ratio>]`, and
  `routed/<strategy>@<weight>[/d<depth>][/i<identity>][/r<ratio>]` parse back into retrieval knobs,
  and `--from-comparison <sweep>/comparison.json` reads the baseline plus the row that sweep's
  verdict named best -- so
  the scored lane is the row the retrieval sweep actually recommended, not a retyped approximation.
- **One shared item set, ordinary bundles.** The selection happens once and every lane is a plain
  `run-eval` bundle under `$DATA_DIR/run-eval/`, so any lane's number is reproducible with a bare
  `run-eval` and the per-item pairing is legitimate. Lanes that scored different item sets are a
  hard error, never a silent intersection. A comma-separated `--split` scores one bundle per split
  and pools them into one compared set.
- **Multi-span coverage beside the objective.** `retrieval_hit` in a score row is `recall@k`, which
  a two-hop item satisfies with one hop, so the lane recomputes `span_coverage` and `all_spans_at_k`
  from each bundle's `retrieval.jsonl` and reports them next to the objective. The verdict states a
  coverage claim on `span_coverage` (graded) rather than the `all_spans_at_k` gate, because on a
  hard multi-hop slice the gate can be near-zero for every lane and therefore blind to a lane that
  nonetheless carried more evidence.

The verdict is one of `answer_quality_gain` (the objective randomization test separates),
`retrieval_only` (the coverage test separates while the objective's does not),
`inconclusive`, or `no_gain`. `retrieval_only` is checked BEFORE `inconclusive` on purpose: a
measured coverage gain paired with a noisy objective is a result about retrieval, and reporting it
as merely inconclusive would drop the half that was measured.

More than two lanes are allowed, and then EVERY candidate keeps its own decision in the verdict's
`lane_decisions` (rendered as a "Per-lane decisions" list): the headline verdict names only the
strongest candidate, so a three-lane comparison that collapsed to one sentence would silently drop
the others.

```bash
make compare-answer-quality CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  FUSION_COMPARISON=<sweep-dir>/comparison.json SPLIT=final,tuning,calibration INCLUDE_DRAFTED=1
llb compare-answer-quality --config <cfg> --lanes vector,fused/global_community@0.10 --split final
```

Artifacts per run: `report.md` and `comparison.json` under
`$DATA_DIR/graph-vector-fusion-multihop/<run>/answer-quality/`.

### The retrieval-budget dimension

`ANSWER_QUALITY_BUDGETS=10,50` (`--budgets`) adds the second axis the lane needs to answer a
BUDGET question rather than a ranking one. Every selected lane is scored at every named `top_k`, so
the run is a `(lane x budget)` grid: the cells are ordinary lanes labelled `<row>#k<budget>`
(`vector#k50`), one run bundle each, one row in every table, and the label parses back into the
retrieval knobs plus the budget. `lanes[0]` at the FIRST budget stays the report baseline, so the
existing tables keep meaning "against the shipped configuration".

What the grid adds is a second pairing the single-baseline table cannot express -- the SAME row at
two budgets:

- **The conversion reading** (`budgets.py` + `conversion.py`). Each raised cell is additionally
  read against its own smallest-budget cell, on the identical resample draw (common random
  numbers), and judged by the SAME `judge_lane` the lane verdict uses. A budget outcome and a lane
  outcome therefore mean the same thing by construction: `answer_quality_gain`, `retrieval_only`,
  `inconclusive`, or `no_gain`, carrying the same borderline and minimum-evidence clauses. The
  sweep headline maps them to the operator's terms -- `converted`, `stalled`, `inconclusive`,
  `no_gain` -- naming the strongest row reached.
- **The cost scan.** A conversion bought by regressing another question type is not a conversion an
  operator should buy, so every non-focus slice whose objective the raised budget LOWERED is named
  in `cost_slices` and appended to the row's reason. The calibrated sign-flip test is one-sided by
  construction ("candidate ahead") and so cannot state a loss; the cost is read off the paired
  interval instead (`llb.rag.fusion_evidence.paired.regresses`) and carries the same
  minimum-evidence gate, so a loss resting on three differing items is not reported as one.
- **The context bill**, in two units, both reported with paired intervals like any other metric and
  neither decided on. `context_chars` (`coverage.py`) is the characters the lane laid into the
  prompt, summed from the retrieval sidecar's offsets rather than from the truncated
  `text_preview`. `prompt_tokens` is what the backend actually CONSUMED: `run-eval` persists it per
  case whenever the backend reports one (`llb.executor.cases`), and the comparison picks it up only
  when every lane carried it, so a bundle from a backend that reports no usage simply drops the
  column instead of comparing a measurement against a zero. Coverage at five times the budget is a
  different result from the same coverage for free, and these are the columns that price it.

Each cell reads its coverage and its bill at ITS OWN `top_k`; reading every cell at the base
config's budget would erase exactly the thing the sweep measures.

```bash
make compare-answer-quality CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  FUSION_COMPARISON=<sweep-dir>/comparison.json ANSWER_QUALITY_BUDGETS=10,50 \
  SPLIT=final,tuning,calibration INCLUDE_DRAFTED=1
```

One operator obligation comes with the second budget: the served context window must fit the
LARGER one, and it must be the same window at both. Ollama silently truncates a prompt past
`num_ctx`, so a config that leaves it at the model default would measure truncation rather than the
budget. Set `max_model_len` in the run config (it becomes the Ollama `num_ctx`), confirm
`ollama ps` still reports `100% GPU`, and read the run's own `prompt tokens` column back against
that window -- a maximum sitting AT the window is the signature of a truncated context, and the
comparison would then be measuring the truncation.

### Measured result: the multi-hop coverage gain does not reach the answer

Scored 2026-07-22 on the RTX 4060 Ti 16 GB CUDA host. The same
95-item drafted goods ledger, matched stores, and k=10 as the sweep above were scored end to end by
`MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` over Ollama under two lanes -- `vector` and the sweep's
best row `fused/global_community@0.10/d10` -- across all three splits (one run bundle per lane and
split, pooled), 2,000 bootstrap resamples, seed 13.

Multi-hop slice (n=35), fused minus vector, 95% paired bootstrap CI:

| metric | delta | interval | w/l/t | sign p |
| --- | ---: | ---: | :-: | ---: |
| objective | -0.005 | [-0.071, +0.072] | 9/8/18 | 1.000 |
| span coverage | **+0.057** | **[+0.014, +0.114]** | 4/0/31 | 0.125 |
| recall@10 | +0.086 | [+0.000, +0.200] | 3/0/32 | 0.250 |
| all-spans@10 | +0.029 | [+0.000, +0.086] | 1/0/34 | 1.000 |

Verdict: **`retrieval_only`**. The fused lane carries measurably more of the multi-hop evidence --
span coverage 0.429 versus 0.371, the only interval in the table that clears zero -- and the model
turns none of it into better answers: the objective is 0.321 versus 0.326, an interval straddling
zero with 9 wins against 8 losses. Paying for a graph build buys multi-hop RETRIEVAL on this
corpus, not multi-hop ANSWERS.

Under the shipped minimum-evidence gate that verdict is **`no_gain`**, and the same downgrade
applies to the two three-lane runs below: the coverage half of every `retrieval_only` here rests on
4-5 differing items (`4/0/31` above), fewer than the 6 the exact sign test needs to reach 95% ([the
gate](../rag-core/paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading)) -- which the
`sign p` column of these tables already showed at 0.062-0.125. No number in any of the tables moves;
the retrieval-only READING is what the 35-item slice cannot support. The answer-side conclusion is
unaffected: the objective was flat before the gate and is flat after it.

That withdrawn coverage reading is priced with the same rule as the sweep's: at 4 of 35 differing
items it needs **53 multi-hop items** to be readable at 95%, and the routed run's 5 of 35 needs
**42** ([the
re-decision](../rag-core/paired-verdicts.md#the-re-decision-what-a-withdrawn-reading-needs)). All
three comparisons ride on the one drafted slice, so a single wider accepted ledger settles all of
them or none.

Two things corroborate the measurement:

- **The retrieval columns reproduce the sweep exactly.** Scored through `run-eval` rather than
  through the sweep's replay wrappers, the fused lane still reports multi-hop recall 0.771 vs
  0.686, all-spans 0.086 vs 0.057, and span coverage 0.429 vs 0.371 -- every figure identical to
  the swept row. The two lanes are measuring the same retrieval through independent code paths.
- **Overall answer quality is flat to slightly negative**: objective -0.027 [-0.062, +0.009]
  (15 wins, 22 losses), with the `procedural` slice at -0.021 [-0.050, -0.001] on n=14 -- the only
  answer-side interval anywhere in the run that excludes zero, and it points DOWN. Extra graph
  candidates displace vector chunks the model was using, and on a 12B model that costs a little
  more than the multi-hop coverage gains back.

Boundaries, both recorded in the artifact rather than inferred:

- **The ledger is drafted.** No reviewer has accepted these 95 items, so the objective is
  diagnostic, not a leaderboard result. Scoring them at all required `--include-drafted`, and every
  bundle manifest carries `config.item_grounding: drafted` (see [RAG
  core](../rag-core/persistence-and-execution.md#executor)). Re-running on the accepted ledger is
  tracked in [`plan.md`](../../plan.md) (`multihop-ledger-human-acceptance`).
- **The answer-side metric cannot see hops.** `objective_score` is reference-answer token F1, so an
  answer stating one fact fluently and omitting the other scores about the same as a vague answer
  touching both. The retrieval side distinguishes partial from complete evidence; the answer side
  does not, which bounds how sharply this verdict can be read. Building the answer-side counterpart
  is tracked in [`plan.md`](../../plan.md) (`answer-side-span-coverage-metric`). Whether "did the
  model use the extra hop" is a property of this MODEL rather than of the corpus is settled below:
  [the second-model reading](#measured-result-the-verdict-is-model-invariant-the-cost-slice-is-not)
  reproduces this verdict on an independently tuned Ukrainian family.

### Measured result: the overlap span identity carries more evidence and costs factoid answers

Scored 2026-07-22 on the RTX 4060 Ti 16 GB CUDA host.
THREE lanes were scored end to end over the identical 95-item drafted ledger, same model, splits,
bootstrap, and seed as the two-lane run above: `vector`, the best `exact` row
(`fused/global_community@0.10/d10`), and the best `overlap` row
(`fused/global_community@0.30/d50/ioverlap`) named by the span-identity sweep's verdict.

Both shared lanes reproduced the earlier run EXACTLY -- every metric mean of `vector` and of the
`exact` row is identical to the two-lane comparison, overall and on the multi-hop slice. Generation
is deterministic for these grounded lanes, so the three-lane table and the earlier two-lane table
are one measurement, not two.

Multi-hop slice (n=35), lane minus vector, 95% paired bootstrap CI:

| lane | objective | span coverage | recall@10 | all-spans@10 |
| --- | ---: | ---: | ---: | ---: |
| exact `@0.10/d10` | -0.005 [-0.071, +0.072] | **+0.057 [+0.014, +0.114]** | +0.086 [+0.000, +0.200] | +0.029 [+0.000, +0.086] |
| overlap `@0.30/d50` | -0.000 [-0.075, +0.078] | **+0.071 [+0.014, +0.129]** | +0.114 [+0.029, +0.229] | +0.029 [+0.000, +0.086] |

Verdict: **`retrieval_only` for BOTH lanes** (**`no_gain`** under the minimum-evidence gate, for
the reason given in the two-lane run above -- 4 and 5 differing items). The overlap row carries the
most multi-hop evidence of any lane measured (span coverage 0.443 versus the vector lane's 0.371,
5 wins and 0 losses) and converts none of it into better answers: its multi-hop objective is 0.326
against the vector lane's 0.326 -- a delta of -0.000 with 12 wins against 7 losses, pure churn.

The new finding is on the other side of the ledger:

- **The overlap lane measurably HURTS factoid answers.** On the 40 factoid items the objective
  falls -0.053 [-0.111, -0.001] (4 wins, 13 losses, sign test p=0.049) -- the only interval in the
  run that clears zero, and it points down. The `exact` row costs less and does not clear zero
  (-0.040 [-0.096, +0.005]). Factoid retrieval itself is flat (span coverage -0.025 [-0.100,
  +0.050]), so this is the CONTEXT changing under a single-span question, not evidence being lost:
  a stronger graph vote re-ranks the chunk the model was already answering from.
- **Overall answer quality stays slightly negative for both**: -0.027 [-0.062, +0.009] for `exact`
  and -0.029 [-0.067, +0.008] for `overlap`, both straddling zero.
- **The retrieval columns reproduce the sweep exactly through a second code path.** Scored through
  `run-eval` rather than the sweep's replay wrappers, the overlap lane reports multi-hop recall
  0.800, all-spans 0.086, and span coverage 0.443 -- every figure identical to the swept row.

What this means for the recommendation: on this corpus and this model, `graph_fusion_span_identity=
overlap` buys strictly more multi-hop RETRIEVAL than `exact` and pays for it with a measured
factoid ANSWER cost, so the fixed-weight row stays opt-in. The question-type route below removes
that cost by never fusing a factoid. The factoid cost itself turns out to be specific to this
model: [the second-model reading](#measured-result-the-verdict-is-model-invariant-the-cost-slice-is-not)
finds it absent on Lapa, which pays on the multi-hop slice instead. The ledger is DRAFTED and the
answer-side metric still cannot see hops; both boundaries above apply unchanged.

### Measured result: question-type routing keeps the gain and clears the factoid loss

Scored 2026-07-22 on the RTX 4060 Ti 16 GB CUDA host; the sweep and the end-to-end answer
comparison are two lanes of the one run. The same 95-item drafted goods ledger, stores, k=10,
split pool,
2,000 bootstrap resamples, seed 13, and 12B MamayLM model as the fixed overlap result were used.
The sweep reports the routed rows beside the complete fixed grid; the end-to-end comparison scores
`vector` against `routed/global_community@0.30/d50/ioverlap`.

All 95 questions had sidecar labels. The router sent 37 to graph fusion -- all 35 `multi-hop` and
both `comparative` items -- and sent 58 to vector -- all 40 `factoid`, 4 `numeric`, and 14
`procedural` items. No heuristic decision contributes to this measurement.

Multi-hop slice (n=35), routed minus vector, 95% paired bootstrap CI:

| metric | delta | interval | w/l/t | sign p |
| --- | ---: | ---: | :-: | ---: |
| objective | -0.000 | [-0.075, +0.078] | 12/7/16 | 0.359 |
| span coverage | **+0.071** | **[+0.014, +0.129]** | 5/0/30 | 0.062 |
| recall@10 | **+0.114** | **[+0.029, +0.229]** | 4/0/31 | 0.125 |
| all-spans@10 | +0.029 | [+0.000, +0.086] | 1/0/34 | 1.000 |

The routed row is retrieval-identical to the best fixed overlap row on every multi-hop metric:
recall 0.800, all-spans 0.086, and span coverage 0.443. It therefore keeps the full measured
multi-hop gain, including both intervals that clear zero. The answer verdict remains
**`retrieval_only`** because objective 0.326 is unchanged from vector despite the extra evidence
(**`no_gain`** under the minimum-evidence gate: that coverage delta is 5 differing items).

The safety result is exact on the slice that motivated routing:

- **All 40 factoid retrieval and answer rows are vector ties.** Objective delta is 0.000
  [0.000, 0.000], 0/0/40 wins/losses/ties; recall, all-spans, and span coverage are also exact
  40-item ties. The fixed overlap row's -0.053 [-0.111, -0.001] factoid answer loss is absent.
- **Overall retrieval improves while answer quality stays flat.** Recall rises +0.063
  [+0.021, +0.116] and span coverage +0.047 [+0.016, +0.089]. Objective is -0.001
  [-0.029, +0.027], compared with -0.029 [-0.067, +0.008] for the fixed overlap row.
- **The exact endpoint is production behavior.** Each routed lane manifest records
  `graph_fusion_router: question_type`; CI verifies that a vector route does not query the graph
  lane, while the live factoid results reproduce vector generation exactly.

Recommendation: use the routed overlap row when the bundle has the documented question-type
sidecar and multi-hop coverage is the goal. Keep `fixed` as the shipped default: the evidence is
drafted and the multi-hop answer gain is still absent. Read that recommendation against [the
second-model reading](#measured-result-the-verdict-is-model-invariant-the-cost-slice-is-not): the
route clears the factoid cost, which is the slice THIS model paid on, and a second model pays on
multi-hop instead -- which routing sends to fusion by design. The sidecar-free policy has a separate
held-out result below and does not support changing its defaults.

### Measured result: the diagnosed budget buys retrieval, not answers

Scored 2026-08-16 on the RTX 4060 Ti 16 GB CUDA host. The
[budget diagnosis](retrieval-budget-evidence.md#is-the-both-hops-ceiling-a-budget-or-a-query-problem)
found the multi-hop `all-spans@k` ceiling to be a property of k=10 rather than of the ranking, with
large headroom at k=50. This run scores that headroom END TO END. The same 95-item drafted goods
ledger, the same rebuilt 1099-chunk store as the k-sweep, and the two lanes the k=50 sweep's own
verdict names (`vector` and `fused/global_community@0.70/d50/ioverlap`) were each scored at k=10
and at k=50 by `MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` over Ollama across all three splits
(pooled), 2,000 bootstrap resamples, seed 13, at a fixed `num_ctx` of 24,576 for every cell.

The reproduction check passed first: `vector#k10` reports multi-hop recall 0.686, `all-spans@10`
0.057, span coverage 0.371; `vector#k50` reports `all-spans@50` 0.229; and the fused row at k=50
reports 0.657 -- every figure identical to the corresponding k-sweep row, now through the
`run-eval` path instead of the sweep's replay wrappers, at BOTH budgets.

Multi-hop slice (n=35), each row at k=50 minus ITSELF at k=10, 95% paired bootstrap CI:

| row | all-spans@k | span coverage | objective | context chars |
| --- | :-: | ---: | ---: | :-: |
| `vector` | 0.057 -> 0.229 | **+0.171 [+0.086, +0.257]** | +0.001 [-0.102, +0.091] | 6,854 -> 33,366 |
| `fused@0.70/d50/ioverlap` | 0.057 -> 0.657 | **+0.443 [+0.343, +0.557]** | +0.030 [-0.076, +0.134] | 3,217 -> 15,188 |

Verdict: **`stalled`** -- `retrieval_only` on both rows. The budget delivers exactly the coverage
the probe predicted, including an eleven-fold lift in `all-spans@k` on the fused row, and the model
turns none of it into better answers.

Unlike the three fixed-budget comparisons above, this reading SURVIVES the minimum-evidence gate on
both halves, which is what makes it a result rather than an open question:

- **The coverage half separates decisively.** Randomization p 0.0005 on both rows, resting on 11
  and 25 differing items against the 6 the exact sign test needs. The ranking knobs never produced
  a coverage reading this far past the floor -- theirs rested on 4-5 items and were withdrawn
  ([the gate](../rag-core/paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading)).
- **The objective half is a MEASURED flat, not an unreachable one.** Its randomization p is 0.494
  and 0.297 on 19 and 22 differing items, so the lanes did differ item by item and the differences
  simply did not favour the larger budget. Both halves of the sentence "more evidence arrived and
  the answers did not move" therefore rest on readings that clear the gate here, which is what the
  earlier fixed-budget comparisons could not say: their objective was equally flat, but the
  coverage gain it was being weighed against had been withdrawn.
- **Nothing is measurably paid for it.** No slice's objective interval clears zero downward, so
  `cost_slices` is empty for both rows; the factoid slice on the vector lane comes closest at
  -0.043 [-0.100, +0.004]. Overall objective is -0.030 [-0.082, +0.022] for `vector#k50` and
  -0.010 [-0.047, +0.030] for the fused row.

The bill for that non-result is the other half of the finding:

| row | prompt tokens (median) | generation latency (median) | lane wall time |
| --- | :-: | :-: | ---: |
| `vector` k10 -> k50 | 2,815 -> 13,171 | 4.8s -> 70.4s | 9 -> 113 min |
| `fused@0.70/d50/ioverlap` k10 -> k50 | 1,444 -> 6,493 | 2.9s -> 18.4s | 5 -> 36 min |

A 4.9x context costs ~15x the generation latency on this host, because the whole increase is
prefill. The graph-fused lane is the notable asymmetry: at the same k=50 it carries MORE multi-hop
evidence than the vector lane (span coverage 0.814 versus 0.543) on LESS THAN HALF the context
(15,188 versus 33,366 characters), because much of its candidate pool is short entity mentions
rather than 800-character chunks -- the same shape the [covering-record
measurement](retrieval-budget-evidence.md#what-the-k50-coverage-is-actually-made-of) found. Cheaper
coverage, and it converts no better.

What this licenses: the budget diagnosis stands as a RETRIEVAL result and does not extend to
answers. `top_k` stays 10; raising it on this corpus and this model buys a measurably fuller
context, five times the prompt, fifteen times the latency, and no better answer. The negative
result points where the task predicted it would -- at compression or a two-stage context step,
which can put a hop in front of the generator without putting 33,000 characters there, rather than
at a larger k.

Boundaries, beyond the drafted ledger and the hop-blind token-F1 objective that bound every result
on this page:

- **One case of 95 timed out and is scored as a zero.** `pdf-6d8c2128b330.md-onto-41` exceeded the
  120s request timeout on `vector#k50` -- itself a consequence of the budget, since only that lane's
  prompts are large enough to approach it. It is a `comparative` item, a context slice of n=2, and
  it is the whole reason that slice shows -0.369; it is not in the focus slice and touches no
  verdict. The per-lane `not_ok` accounting that now surfaces such a case in the report was added
  AFTER this run, so for this artifact the count was established by hand from the bundles.
- **The served window was verified, not assumed.** The largest prompt any cell sent was 14,855
  tokens against the 24,576 served, so no context was truncated into the measurement.

### Measured result: the verdict is model-invariant, the cost slice is not

Scored 2026-08-22 on the RTX 4060 Ti 16 GB CUDA host, one comparison per model. Whether extra
retrieved
evidence converts into a better answer is a property of the MODEL as much as of the lane, so every
result above -- all measured with one generator -- could not tell "fusion does not help answers on
this corpus" from "this tune does not use the extra hop". This run separates them by scoring the
SAME three lanes with the roster's two Ukrainian-specialized families
([model families](../../../reference/model-families.md#ukrainian-specialized-families)):
`MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M` and `lapa-v0.1.2-instruct-GGUF:Q4_K_M`. Same 95-item
drafted goods ledger, same stores, k=10, all three splits pooled, 2,000 bootstrap resamples, seed
13, and a `num_ctx` of 8,192 served to both.

Both models were re-scored rather than pairing the new one against the recorded MamayLM numbers.
The July fixed-budget runs read a store whose lexical index was written by the `bm25-uk-v1`
tokenizer, which the current build refuses, so their retrieval columns are not reproducible under
today's code; this run uses the duplicate-collapsed 1,099-chunk store that the k-sweep and the
budget evidence both ride on. The re-scored MamayLM lane reproduces every July multi-hop
retrieval figure exactly anyway --
recall 0.686 / 0.771 / 0.800, all-spans 0.057 / 0.086 / 0.086, span coverage 0.371 / 0.429 / 0.443
-- so the new pair is readable against the three fixed-budget results above.

**The two models read identical context, by measurement rather than by assumption.** Every
retrieval delta, its win/loss/tie ledger, its randomization p, and the `context_chars` column are
BYTE-identical between the two comparisons on every slice. The only thing that differs between them
is the generator.

Multi-hop slice (n=35), lane minus vector, 95% paired bootstrap CI:

| lane | metric | MamayLM 12B | Lapa 12B |
| --- | --- | ---: | ---: |
| exact `@0.10/d10` | objective | -0.002 [-0.059, +0.072] 7/10/18 | -0.024 [-0.074, +0.016] 10/7/18 |
| exact `@0.10/d10` | span coverage | +0.057 [+0.014, +0.114] 4/0/31 | +0.057 [+0.014, +0.114] 4/0/31 |
| overlap `@0.30/d50` | objective | -0.010 [-0.087, +0.070] 11/9/15 | **-0.081 [-0.157, -0.017] 7/14/14** |
| overlap `@0.30/d50` | span coverage | +0.071 [+0.014, +0.129] 5/0/30 | +0.071 [+0.014, +0.129] 5/0/30 |

Verdict: **`no_gain` on both lanes for BOTH models** -- the same headline, the same per-lane
decisions, and the same reasons, because the coverage half that would have made either lane
`retrieval_only` rests on the same 4 and 5 differing items in both and is withdrawn by [the
gate](../rag-core/paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading) either way. The
answer-side finding is therefore NOT an artifact of the one model that was scored: two independently
tuned Ukrainian families, reading the identical context, both decline to turn the fused lane's extra
multi-hop evidence into better multi-hop answers.

That much reproduces. What does not is where the strongest-coverage row sends its bill:

- **The overlap row's factoid cost is a MamayLM property.** The [three-lane
  result](#measured-result-the-overlap-span-identity-carries-more-evidence-and-costs-factoid-answers)
  above found the overlap identity lowering the 40-item factoid objective by an interval clear of
  zero; it reproduces here on MamayLM at -0.054 [-0.108, -0.009] (4/10/26, 14 differing items, past
  the gate). On Lapa the same row over the same 40 items is -0.004 [-0.048, +0.049] (3/8/29, 11
  differing items) -- a MEASURED flat, not an unreachable one. The extra graph vote re-ranks the
  chunk each model was answering from; only one of the two tunes was answering worse for it.
- **The bill moves to multi-hop instead.** On Lapa the overlap row lowers the objective on the very
  slice it retrieves more evidence for: -0.081 [-0.157, -0.017], 7 wins against 14 losses on 21
  differing items, `p_positive` 0.004. That clears the minimum-evidence gate comfortably, which
  none of the coverage readings on this ledger do. So on Lapa the fused lane carries measurably
  more multi-hop evidence AND answers multi-hop measurably worse with it.
- **What is invariant is that the row pays, not what it pays with.** Across both models the
  strongest-coverage lane loses on exactly one question-type slice by an interval clear of zero --
  factoid for MamayLM, multi-hop for Lapa -- while overall answer quality stays flat for both
  (MamayLM -0.020 [-0.060, +0.019]; Lapa -0.023 [-0.062, +0.013]). The `exact` row tells the same
  story one step weaker: it costs MamayLM the same factoid slice (-0.045 [-0.099, -0.001] on 12
  differing items) and costs Lapa nothing on any slice.

Two corollaries an operator should read off this:

- **"A stronger answerer would use the extra hop" is disconfirmed here, not merely untested.** Lapa
  is much the better multi-hop answerer on this corpus -- its vector-lane multi-hop objective is
  0.543 against MamayLM's 0.333, on the identical context -- and it converts LESS of the coverage,
  not more.
- **The question-type route was validated against one model's failure mode.** [Routing](#measured-result-question-type-routing-keeps-the-gain-and-clears-the-factoid-loss)
  clears the factoid cost by never fusing a factoid, which is exactly the slice MamayLM paid on. It
  would not touch Lapa's cost, which lands on the multi-hop slice the router deliberately sends TO
  fusion. Re-reading the routed row on a second model is forward work in
  [`plan.md`](../../plan.md) (`routed-lane-second-model-cost-check`).

Boundaries, beyond the drafted ledger and the hop-blind token-F1 objective that bound every result
on this page:

- **Cross-model LEVELS of the objective are confounded; the deltas are not.** `objective_score` is
  reference-answer token F1, so a terser model scores better on it for the same facts -- Lapa's
  median answer is 22 completion tokens against MamayLM's 32, and their overall levels (0.510 and
  0.516) are not a ranking. Every claim above is a within-model paired delta against that model's
  own vector lane, which the length difference cannot move.
- **One reading on Lapa is thin rather than flat.** Its factoid objective on the `exact` row,
  -0.018 [-0.043, +0.002], differs on 5 of 40 items and is `insufficient evidence`; the factoid
  claim above is made on the `overlap` row, which differs on 11 and clears the gate. Across the run
  the gate relabels 8 of 126 paired readings for MamayLM and 6 of 126 for Lapa.
- **Nothing was truncated and nothing timed out.** The largest prompt any lane sent was 3,357
  tokens (MamayLM) and 2,509 (Lapa) against the 8,192 served, and all 95 cases in all six
  lane-model bundles are `ok`.

```bash
make compare-answer-quality CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  MODEL=<second-roster-model> SPLIT=final,tuning,calibration \
  ANSWER_QUALITY_LANES=vector,<best-exact-row>,<best-overlap-row> INCLUDE_DRAFTED=1 \
  ANSWER_QUALITY_OUT_DIR=<run>/<model-slug>/answer-quality
```
