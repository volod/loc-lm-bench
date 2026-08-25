# Context Ablation: Does RAG Pay For Itself? (rag-vs-long-context-ablation)

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

A leaderboard row says how well a model answers WITH retrieval; it never says how much of that
score retrieval bought. `llb compare-context-strategies` (`make compare-context-strategies`)
scores ONE item set end to end under four context lanes and reports the differences.

`RunConfig.context_strategy` selects the lane and is recorded in the manifest fingerprint like
every other knob, so a lane's bundle is reproducible from its own config
(`make run-eval CONTEXT_STRATEGY=<lane>`):

- `rag` (default) -- retrieve as configured. This is the leaderboard row; `closed_book` and
  `long_context` are DIAGNOSTICS and never rank a model.
- `closed_book` -- no context at all. `src/llb/eval/context_ablation/sources.py` supplies an empty
  context and swaps in the `eval.rag.closed_book` prompt, which asks the model to answer from its
  own knowledge (the RAG system prompt would push it to abstain). The empty context deliberately
  does NOT raise `retrieval_miss`: that status short-circuits generation, and a lane that never
  calls the model measures nothing.
- `long_context` -- the item's whole gold source document(s) laid into the prompt as one
  offset-exact chunk per document, with the SAME generation prompt as `rag`, so the delta is
  attributable to the context and not to prompt wording. The lane is oracle-grounded (it reads the
  item's own gold `doc_id`s), which makes it a ceiling, not a shippable retrieval policy.
- `retrieved_document` -- the shippable sibling of `long_context`, described in full
  [below](#the-shippable-sibling-retrieved_document). Same whole-document context, same generation
  prompt, same skip rule; the documents are chosen by RETRIEVAL rather than by the gold label.

Budget and skips: the document lanes resolve the model's usable window ONCE per run and check
each item against it. An item whose documents do not fit terminates as `context_overflow`, a
pre-generation status in the shared taxonomy: no model call, no truncation. A truncated document
is a different and unstated retrieval policy, so crediting its answer to a document lane would
measure whichever slice survived the cut. Both document lanes share one `document_context` helper,
so they can differ only in how the documents were chosen.

**The window is the MINIMUM of the declared one and the one the backend is actually serving.** The
run owns it, not the lane: `runner_setup` builds one `PromptWindow`
(`src/llb/backends/prompt_window.py`) per run and hands the lane its `fits` predicate, so a lane's
skips, the `rag` prompt check, and the agent-loop prompt guard are all bound by the same number on
one host ([the usable prompt
window](../extended-workflows/agent-context-policies.md#agent-context-management-policies)). The
declared side
is the host planner cap, the roster entry's `max_context`, `--max-model-len`, and `--context-budget`;
`resolve_model_spec` (`src/llb/backends/context_budget.py`) looks the served artifact up through
`candidate_sources`, so an Ollama GGUF tag resolves to its roster entry priced at the right quant.
The served side is
`launcher_served_window` (`src/llb/backends/served_window.py`), which asks the started launcher what
it is serving. Taking only the declared side is what makes a skip promise hollow: Ollama serves
`num_ctx` 4096 unless `--max-model-len` / `--context-budget` pins it, however large a window the
GGUF advertises, so the backend would truncate a document the report counts as fully delivered --
silently, leaving the lane to read as a measured long-context result.

Two details make that probe land. It is **lazy**: the graph is wired before `launcher.start()`, so
the window resolves on the first item's fit check instead of at build time, and every later item
reuses it. And it **warm-loads** on Ollama, which reports no window at all over `/api/ps` until
some request has loaded the model -- reading "unknown" there is exactly the case the probe exists
for, so `ensure_num_ctx` sends a one-token request first.

Which side bound the run is recorded, never inferred: `declared_max_model_len`,
`served_max_model_len`, and `budget_source` go into the lane's own `manifest.json` under
`context_window`, the comparison reads them back off those manifests (`lane_context_windows`), and
`report.md` names them beside the lane's skip count -- `long_context -- window 4096 tokens (served,
declared 131072)`. A lane that never checks a document against a window (`closed_book`, `rag`)
records `null`. Without a manifest entry for the model AND without a probe, only an explicit
`context_budget` / `max_model_len` can bound the prompt, so an unlisted model on an unreachable
backend skips nothing rather than everything.

The comparison (`src/llb/eval/context_ablation/`) is pure and file-driven: it consumes canonical
`scores.jsonl` rows, aligns them with `llb.eval.paired_cases` (shared with
`compare-answer-quality`), and reuses the fusion-evidence paired bootstrap and per-slice reporting,
so the artifact reads beside the retrieval sweep. It reports:

- `retrieval_uplift` = `rag - closed_book`, paired per item -- how much of the RAG score retrieval
  paid for.
- `long_context_delta` = `long_context - rag` -- whole-document stuffing versus chunked retrieval.
- `long_context_delta_fitting` -- the same delta over items the lane did not skip, emitted only
  when something WAS skipped. A skipped item scores zero, so the all-items delta would otherwise
  read a document that never reached the model as a long-context loss; the VERDICT reads the
  fitting delta when it exists.
- A per-item contamination flag: the closed-book answer already matches the reference (`exact` or
  `contains` is 1.0). Items the model answers with no evidence were never a retrieval problem, and
  a corpus full of them makes any uplift look small for reasons unrelated to retrieval.
- `retrieved_document_delta` = `retrieved_document - rag` and `oracle_document_gap` =
  `long_context - retrieved_document`, the two halves the oracle gap splits into. They sum to
  `long_context_delta` by construction, which is the whole point of measuring them together.

Each document-lane delta also gets a `_fitting` cut when either of ITS OWN two lanes skipped an
item. The population is scoped to the pair, not to the run: with two document lanes present, an
item only `retrieved_document` skipped says nothing about `long_context - rag`, and pooling every
lane's skips would silently shrink a delta that was fully measured.

Every one of those numbers is ALSO stated per question type, read from the gold set's
`needle_items.jsonl` / `item_provenance.jsonl` sidecar (`src/llb/rag/question_types.py` joins the
two). `src/llb/eval/context_ablation/per_slice.py` rebuilds the same derived table over each
slice's items, with that slice's own bootstrap draw, its own fitting cut, and its own contamination
report, and judges it through the same `decide_population` the pooled verdict uses
(`verdict.py`) -- so a slice reading and the corpus reading differ only in which items they saw.
The artifact leads the section with one summary row per question type (n, closed-book matches,
`retrieval_uplift`, `long_context_delta`, and the reading that slice reached) and follows it with
each slice's own per-lane metrics and derived deltas (`report_slices.py`, `report_tables.py`).

Three rules keep a slice honest:

- **A slice is diagnostic.** The pooled verdict stays the corpus decision, and no slice carries the
  `retrieved_document` adoption call: a shippable configuration picked off a dozen items of one
  question type is what the minimum-evidence gate exists to refuse.
- **A fitting cut is scoped to the slice.** A slice no lane skipped carries no fitting row at all;
  a slice a lane skipped ENTIRELY reports `not measurable` rather than the `0.000` that a mean over
  no items formats as.
- **No sidecar, no slices.** A gold set that labels nothing reports the pooled number and says why,
  instead of inventing a slice label to fill the table.

Verdicts, in check order: `long_context_wins` | `rag_pays_off` | `retrieval_inconclusive` |
`no_retrieval_gain` | `no_evidence`. Every gate reads the paired INTERVAL, never the point
estimate. Artifacts: `$DATA_DIR/context-ablation/<run>/{report.md,comparison.json}`, plus one
ordinary `run-eval` bundle per (lane, split) under `$DATA_DIR/run-eval/`. CI drives every lane over
fake bundles and the committed fixtures (`tests/llb/eval/context_ablation/`), no backend or GPU.

An optional a priori power contract keeps a borderline row from being "resolved" by choosing a
different confidence convention. Its context adapter delegates arithmetic and realized sensitivity
to the [shared comparison-lane
contract](paired-verdicts.md#paired-power-contract-for-comparison-lanes). Pass the earlier
`comparison.json` as `CONTEXT_POWER_REFERENCE=<comparison-json>`, predeclare the smallest material
objective delta as `CONTEXT_MDE=<delta>`, and optionally set `CONTEXT_TARGET_POWER=<share>` (default
0.80). Before the first model call, `src/llb/eval/context_ablation/power.py` reads the earlier
per-item `long_context - rag` differences, estimates their paired sample SD, and writes
`power-plan.json`. The required count is the two-sided paired normal approximation
`ceil(((z_(1-alpha/2) + z_power) * sample_sd / MDE)^2)`, where alpha follows the report confidence.
The completed `comparison.json` and `report.md` re-check the item target against realized SD, report
the resolvable MDE at the reached item count, and record:

- `separated` when the new paired interval is wholly on one side of zero;
- `flat` when the interval is wholly inside the predeclared `[-MDE, +MDE]` detectable-effect band;
- `undecidable` otherwise, explicitly saying whether the run's realized variance reached the
  target.

The power options are additive: omitting them preserves the original artifact schema and behavior.
`tests/llb/eval/test_context_ablation_power.py` covers the item-count calculation, paired-reference
loading, resolution states, and the guarantee that `power-plan.json` exists before lane scoring.

## Decoding stability: how far a re-run moves the number

Every interval in the artifact is a paired bootstrap over the ITEM SAMPLE -- "would another draw of
questions have said this?". None of them can see the other source of uncertainty: scoring the
identical configuration on the identical items again and getting a different answer. Greedy
decoding is not bit-reproducible on a GGUF runtime, and the flip is not equally likely in every
lane. A grounded prompt carries the answer in its context, so the next-token distribution is
sharply peaked and a bit of numeric drift changes nothing; a closed-book prompt leaves a much
flatter distribution, so the same drift rewrites the answer.

`--repeats <n>` (`make compare-context-strategies CONTEXT_REPEATS=<n>`) scores every selected lane
`n` times with the identical config on the identical items and adds a `decoding_stability` block to
`comparison.json` and `report.md` (`src/llb/eval/context_ablation/decoding_stability.py`). The
comparison itself is still taken over the FIRST repeat -- a repeat is not more evidence and never
enters a mean, an interval, a slice, or a verdict -- so the artifact keeps its schema and its
numbers, and the block says how far a re-run moves them. Omitting `--repeats` preserves the
artifact exactly, like the power options above.

Per lane the block reports the band (`min-max (+/-half-width)`) of that lane's own mean objective,
mean token F1, and reference-match rate across the repeats, plus the number of items whose
objective moved and whose recorded answer moved. The match rate IS the contamination rate on the
baseline lane and the found-rate on a grounded one, so the header's closed-book match line carries
its own `+/-` whenever repeats were scored. The report then re-reads every derived delta against a
DECODING FLOOR -- the SUM of its two lanes' half-widths, the conservative bound rather than their
quadrature, because the repeats are few -- and says whether the delta clears it. A delta inside its
floor was observed rather than measured.

Beside the band each lane carries its `repeat groups`: the sizes of the groups of repeats that
produced the IDENTICAL per-item objective vector, in first-appearance order. This is the SHAPE of
the drift where the half-width is only its size, and the two say different things. `4` is a lane
that reproduced throughout. `1+3` is one that answered differently on its first pass and then
settled -- a warm-up transient, and since the artifact quotes the FIRST repeat, the report says
outright that the numbers above are the odd one out and names the band's other end as the settled
value. `1+1+1+1` is a lane that never repeated itself, which is the only one of the three that is
irreducible noise. All three can print the same half-width.

The reading is stated in both directions on purpose. The hypothesis behind this measurement -- an
ungrounded prompt leaves a flatter distribution, so the closed-book lane should be the noisier one
-- is a hypothesis, so a run where the baseline lane is the QUIETER one says that rather than
printing the same sentence with a small multiple in it. On the first host it was run on, it was
([context-ablation evidence](context-ablation-evidence.md#the-lanes-reproduce-the-closed-book-lane-is-not-the-noisier-one-2026-08-25)).

Four rules keep the block honest:

- **The quoted value is not a summary of the repeats.** `base` is the first repeat, the one the
  tables above it print; collapsing the two into a pooled mean would hide exactly the question the
  block exists to answer.
- **A delta whose lanes were not both repeated is absent, not floored at zero.** An unmeasured
  floor is not a floor of zero, and printing one would read as "this delta is decode-stable".
- **A repeat that scored a different item set fails loudly** -- the same rule `shared_item_ids`
  applies across lanes. A band drawn over two different item sets is not a band.
- **The answer-divergence count is a lower bound.** It is read off the persisted answer PREVIEW
  (280 characters), so two answers that agree that far and diverge afterwards count as identical.
  The objective divergence beside it is exact.

The statistic is the shared `ValueSpread` (`src/llb/rag/fusion_evidence/spread.py`), the same band
the retrieval [measurement floor](retrieval-metrics.md) reports over its jitter replicates, so a
reader comparing the two floors is reading identical columns. CI pins the whole vertical over
committed fixture rows in
`tests/llb/eval/context_ablation/test_context_ablation_decoding_stability.py` -- the band
arithmetic, the contamination-rate band, the ungrounded-versus-grounded reading, the delta floors,
and the guarantee that a single pass leaves the artifact untouched.

## The shippable sibling: `retrieved_document`

`long_context` sizes a CEILING and can never be adopted -- it reads the item's gold `doc_id`s, and
a real query arrives without one ([product decisions](../scope-boundaries.md#context-ablation-lanes-stay-diagnostic)).
`retrieved_document` asks the operator's version of the same question: send the whole document,
but the one RETRIEVAL picked. Nothing in its path sees a gold label, so whatever it gains over
`rag` is a gain reachable by changing a config value.

**How it differs from the other lanes, mechanically.** `closed_book` and `long_context` do not
retrieve, so they install a context SOURCE that replaces the retrieve node outright.
`retrieved_document` DOES retrieve -- identically to `rag`, including query prep, the ACL chunk
filter, reranking, and the per-stage latency accounting -- so it installs a context REFINER
(`ContextRefiner` in `src/llb/eval/graph_contracts.py`) that runs after the retrieve node and
rewrites what the prompt carries. Replacing the node instead would have meant reimplementing the
whole retrieval path inside the lane, and the first divergence would have made the delta
unattributable.

The refiner (`retrieved_document_refiner`, `src/llb/eval/context_ablation/sources.py`):

- **selects** the first `retrieved_document_top_n` DISTINCT `doc_id`s off the ranked chunk list,
  best-first (default 1 -- the top-ranked chunk's document). De-duplicating by document makes the
  knob a document budget rather than a chunk budget: three chunks of one document are one
  document. It is recorded in the manifest fingerprint like every other knob
  (`make run-eval CONTEXT_STRATEGY=retrieved_document RETRIEVED_DOCUMENT_TOP_N=<n>`).
- **lays them in whole** through the same `document_context` helper `long_context` uses, so the two
  lanes differ ONLY in how the documents were chosen, and with the same generation prompt as `rag`,
  so the delta is attributable to the context and not to prompt wording.
- **skips, never truncates**, on the same served-window budget and the same `context_overflow`
  status.
- **keeps the retrieval it paid for**: `retrieve_latency_s`, `rerank_latency_s`, and the query-prep
  provenance survive the rewrite, because that retrieval is what chose the document.
- **fails loudly** when a retrieved `doc_id` is not in `--corpus`. Falling back to the chunk
  context would quietly turn the lane back into `rag` and report the result as a document lane.
- **claims no header restoration.** Whole documents replace the chunk block outright, so the
  table-header accounting is zeroed and the stale `prompt_chunks` copy dropped.

A retrieval miss short-circuits BEFORE the refiner: nothing was retrieved, so there is no document
to widen to, and the case terminates as `retrieval_miss` exactly as it would under `rag`.

**Reading its recall@k.** Under `closed_book` recall@k is 0.0 by construction and under
`long_context` it is 1.0 by construction; under `retrieved_document` it is the one lane column that
is measured. The prompt carries whole documents, so `retrieved` is rewritten to those documents and
recall@k reads DOCUMENT-level: how often the selection rule picked a document that actually holds
the answer. That is the lane's own ceiling, and it is not comparable to the `rag` row's chunk-level
recall@k.

**The verdict.** Because this lane is a configuration rather than a diagnostic, it carries its own
adopt-or-reject call (`src/llb/eval/context_ablation/verdict_adoption.py`), stated beside the
ablation verdict rather than folded into it: `adopt_retrieved_document` when the paired
`retrieved_document - rag` interval separates above zero, `reject_retrieved_document` when it
separates below (read off the mirrored `regresses` gate, since the calibrated sign-flip p is
one-sided by construction), `retrieved_document_inconclusive` when it straddles, and
`retrieved_document_not_measured` when the lane was not scored. The reason line also states the
captured share -- `retrieved_document_delta / long_context_delta`, defined only when the oracle
lane actually gained -- so an operator sees how much of the ceiling the shippable lane reached.

## Context-ablation evidence

Every measured result this lane has produced -- the roster cohort, the powered long-context
verdict, the shippable-document reject, the served-window binding, and the decoding-stability
bands -- lives on its own page:
[Context-ablation evidence](context-ablation-evidence.md).
