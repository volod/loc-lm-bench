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

A routed lane adds one section to that report. `Routing outcome` states, per routed row, the
focus-slice coverage it keeps, every slice it reproduces the baseline on EXACTLY, every slice
whose objective it STILL lowers by an interval clear of zero (each marked with whether it clears
the minimum-evidence gate), and which of its fixed twin's cost slices it therefore clears,
retains, or adds. That last reading is what makes a routed row's bill readable without deriving
it from six tables, and it is model-dependent -- see [the routed per-model
result](#measured-result-what-the-route-clears-is-model-conditioned).

Scoring every lane at more than one retrieval budget is a second axis with its own page:
[answer-quality budget evidence](answer-quality-budget-evidence.md#the-retrieval-budget-dimension).

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
sidecar, multi-hop coverage is the goal, AND the generator's own cost slice is one the router
excludes. Keep `fixed` as the shipped default: the evidence is drafted and the multi-hop answer
gain is still absent. That last condition is not decoration -- the route clears the factoid cost,
which is the slice THIS model paid on, while a second model pays on multi-hop instead, which
routing sends to fusion by design and therefore cannot clear ([the routed per-model
result](#measured-result-what-the-route-clears-is-model-conditioned)). The sidecar-free policy has
a separate held-out result below and does not support changing its defaults.

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
  cannot touch Lapa's cost, which lands on the multi-hop slice the router deliberately sends TO
  fusion -- scored directly in [the routed per-model
  result](#measured-result-what-the-route-clears-is-model-conditioned), which confirms the routed
  row carries that loss unchanged.

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

### Measured result: what the route clears is model-conditioned

Scored 2026-08-22 on the RTX 4060 Ti 16 GB CUDA host, one comparison per model. The question-type
route is recommended for REMOVING a measured per-slice cost, and the reading above found that cost
slice moving between generators. This run scores the routed row itself under both families:
`vector`, the fixed overlap row `fused/global_community@0.30/d50/ioverlap`, and its routed twin
`routed/global_community@0.30/d50/ioverlap`, three lanes per model over the same 95-item drafted
goods ledger, the same duplicate-collapsed 1,099-chunk store, k=10, all three splits pooled, 2,000
bootstrap resamples, seed 13, and a `num_ctx` of 8,192 served to both.

Both shared lanes reproduce the [second-model
reading](#measured-result-the-verdict-is-model-invariant-the-cost-slice-is-not) exactly -- the
fixed row's multi-hop objective is -0.010 [-0.087, +0.070] on MamayLM and -0.081 [-0.157, -0.017]
on Lapa, its span coverage +0.071 [+0.014, +0.129] on both, and its factoid objective -0.054
[-0.108, -0.009] and -0.004 [-0.048, +0.049] -- so the routed row is readable against it rather
than against a re-measured baseline. Every retrieval delta is again byte-identical between the two
models: only the generator differs.

Lane minus vector, 95% paired bootstrap CI, each routed row printed beside its fixed twin:

| slice | lane | MamayLM 12B | Lapa 12B |
| --- | --- | ---: | ---: |
| multi-hop (n=35) objective | fixed | -0.010 [-0.087, +0.070] 11/9/15 | **-0.081 [-0.157, -0.017] 7/14/14** |
| multi-hop (n=35) objective | routed | -0.010 [-0.087, +0.070] 11/9/15 | **-0.081 [-0.157, -0.017] 7/14/14** |
| multi-hop (n=35) span coverage | routed | +0.071 [+0.014, +0.129] 5/0/30 | +0.071 [+0.014, +0.129] 5/0/30 |
| factoid (n=40) objective | fixed | **-0.054 [-0.108, -0.009] 4/10/26** | -0.004 [-0.048, +0.049] 3/8/29 |
| factoid (n=40) objective | routed | +0.000 [0.000, 0.000] 0/0/40 | +0.000 [0.000, 0.000] 0/0/40 |
| overall (n=95) objective | fixed | -0.020 [-0.060, +0.019] 22/25/48 | -0.023 [-0.062, +0.013] 17/25/53 |
| overall (n=95) objective | routed | -0.002 [-0.032, +0.027] 13/9/73 | **-0.032 [-0.064, -0.007] 7/15/73** |

Verdict: **`no_gain` for the routed row on BOTH models**, for the same reason every k=10
comparison on this ledger reaches it -- the coverage half that would make it `retrieval_only` rests
on 5 differing items and is withdrawn by [the
gate](../rag-core/paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading). What the run
settles is not that verdict but the per-slice bill:

- **The safety property is model-invariant and exact.** On both models all 40 `factoid`, 4
  `numeric`, and 14 `procedural` items are 0/0/n ties against vector on the objective AND on every
  retrieval column -- the router sent them to the zero endpoint, which does not query the graph
  lane, so the answers are the vector lane's own. MamayLM's factoid cost is therefore cleared
  exactly, reproducing [the routing
  result](#measured-result-question-type-routing-keeps-the-gain-and-clears-the-factoid-loss) on the
  rebuilt store.
- **The multi-hop gain survives the route in full,** identical to the fixed row on every retrieval
  column (recall 0.800, all-spans 0.086, span coverage 0.443) for both models.
- **What the route CLEARS is model-conditioned; what it PAYS is not.** Lapa's cost lands on
  `multi-hop`, the slice the router sends TO fusion by construction, so the routed row carries it
  unchanged: -0.081 [-0.157, -0.017] on 21 differing items, past the gate. The route removed the
  slice MamayLM paid on and could not touch the slice Lapa pays on.
- **On Lapa the route makes that loss MORE visible, not less.** Routing turns the untouched slices
  into 73 exact ties, so the multi-hop loss is no longer diluted by them: the routed row's overall
  objective is -0.032 [-0.064, -0.007] on 22 differing items -- an interval clear of zero, where
  the fixed row's -0.023 [-0.062, +0.013] over 42 differing items is not. On MamayLM the same
  arithmetic runs the other way, -0.020 to -0.002. Routing does not change how a model answers a
  fused question; it changes how much of the run is fused.

What this licenses: the recommendation is about the TUNE, not about the corpus. "Use the routed
overlap row when the bundle has a question-type sidecar" buys what it is recommended for only when
the generator's cost slice is one the router excludes; on a generator paying on the focus slice it
delivers the same loss and promotes it to the dominant overall term. `fixed` stays the shipped
default, and an operator adopting the route must first read WHICH slice their own model pays on --
which is the reading the artifact now prints.

Boundaries, beyond the drafted ledger and the hop-blind token-F1 objective that bound every result
on this page:

- **The `comparative` slice decides nothing.** It is n=2, it routes to fusion, and Lapa's -0.108
  [-0.216, +0.000] there rests on ONE differing item; it is not a cost by the gate and is not read
  as one.
- **The gate relabels 14 of 126 paired readings for MamayLM and 10 of 126 for Lapa.** An exact
  0/0/n tie is not among them: it claims no difference, so it reads `flat` rather than
  `insufficient evidence`.
- **Nothing was truncated and nothing timed out.** The largest prompt any lane sent was 3,357
  tokens (MamayLM) and 2,509 (Lapa) against the 8,192 served, and all 95 cases of every one of the
  six lane-model pairings (three split bundles each) are `ok`.

```bash
make compare-answer-quality CONFIG=<run-config.yaml> GOLDSET=<goldset-jsonl> \
  MODEL=<roster-model> SPLIT=final,tuning,calibration \
  ANSWER_QUALITY_LANES=vector,<best-overlap-row>,<routed-overlap-row> INCLUDE_DRAFTED=1 \
  ANSWER_QUALITY_OUT_DIR=<run>/<model-slug>/answer-quality
```
