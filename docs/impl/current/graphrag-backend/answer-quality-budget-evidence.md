# Answer-Quality Budget Evidence

Part of the [GraphRAG backend](../graphrag-backend.md) area of the
[current implementation index](../../current.md).

The lane itself -- what it scores, how it pairs, and what its four verdicts mean -- is
[answer-quality evidence](answer-quality-evidence.md#answer-quality-evidence). This page owns its
SECOND axis: scoring every lane at more than one retrieval budget, which is the only way to ask
whether a budget-driven coverage gain converts into answers and what the extra context costs.

## The retrieval-budget dimension

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

Unlike the three fixed-budget comparisons on the [lane
page](answer-quality-evidence.md#answer-quality-evidence), this reading SURVIVES the
minimum-evidence gate on both halves, which is what makes it a result rather than an open question:

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
  verdict. The per-lane `not_ok` accounting that surfaces such a case in the report was added AFTER
  this run, so the count was first established by hand from the bundles; re-rendering the recorded
  comparison from those same bundles now prints it -- `vector#k50`, 1 of 95 not `ok` -- with no
  generation re-run and no other number moved ([re-rendering a recorded
  comparison](answer-quality-rerender.md#measured-result-the-current-report-reaches-four-recorded-runs-unchanged-in-substance)).
- **The served window was verified, not assumed.** The largest prompt any cell sent was 14,855
  tokens against the 24,576 served, so no context was truncated into the measurement.
