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

Each derived delta carries `p_positive` and a `(borderline)` flag, and the verdict names both the
rows it was decided on -- the retrieval uplift AND the long-context delta, because `_judge` checks
the long-context lane first
([how settled a paired reading is](paired-verdicts.md#how-settled-a-paired-reading-is----p_positive-and-the-borderline-flag)).
The original `qwen3.6-35b` final-only row was the one exception: its `rag_pays_off` rested on a
settled uplift (`p_positive` 1.000) but a long-context delta at `p_positive` 0.960 that a 90%
interval read as separated. The power-resolved run below removes that exception; every recorded
context-ablation verdict is now settled at the neighbouring 90%, 95%, and 97.5% conventions.

### The served window is 32x smaller than the declared one on this host (2026-08-24)

A four-lane run on the RTX 4060 Ti 16 GB CUDA host with Ollama, MamayLM-Gemma-3-12B-IT v2.0 GGUF
Q4_K_M, the committed UA fixture `samples/goldsets/ua_squad_postedited_v1/` (8 verified `final`
items, `top_k=5`, `max_tokens=512`), scored to check what the document lanes were measuring their
skips against. Both document-lane manifests recorded
`{"declared_max_model_len": 131072, "served_max_model_len": 4096, "budget_source": "served"}`; both
`closed_book` and `rag` recorded `null`.

**The declared window was 32x the served one.** At `max_tokens=512` that is a document budget of
9,216 usable characters, not the 390,144 the declared 131072 implies -- a 42x drop in what the
lanes will accept. No item was skipped in either lane, before or after: the fixture's largest
corpus document is 1,671 characters and its median is 615, three orders of magnitude inside both
budgets, which is why this corpus never exposed the gap and why the numbers are unchanged. The
finding is the gap itself: on any corpus whose documents run past ~9k characters, this host would
have handed Ollama a document it truncated at 4096 tokens and reported the answer as a fully
delivered long-context result.

This run is a binding check, not a quality reading: 8 items is far below every evidence floor this
page's verdicts are held to, so its lane means are deliberately not recorded here.

What would overturn it: an Ollama build whose `/api/ps` reports the GGUF window rather than the
served `num_ctx`, or a host with `OLLAMA_CONTEXT_LENGTH` raised, in which case `budget_source` reads
`declared` and the two windows agree. The skip behaviour itself is pinned deterministically in both
binding directions by `tests/llb/eval/context_ablation/test_context_ablation_window.py` rather than
by this run, which could not produce a skip on a corpus this small.

### The shippable document lane does not pay: reject (2026-08-23)

Four runs on the RTX 4060 Ti 16 GB CUDA host with Ollama, the committed UA fixture
`samples/goldsets/ua_squad_postedited_v1/` (82 verified `final` items, 250-document corpus, 311
chunks at 800/120, `intfloat/multilingual-e5-base`, `top_k=5`), scoring all four lanes on the SAME
item set: the two roster models whose oracle gap priced this work (MamayLM-Gemma-3-12B-IT v2.0 GGUF
Q4_K_M and Lapa v0.1.2-instruct GGUF Q4_K_M), each at both ends of the document-selection rule --
`retrieved_document_top_n=1` (the top-ranked chunk's document alone) and `=5` (every distinct
document in the retrieved top-5, which holds the retrieved SET fixed and changes only the unit).

| model | top_n | rag | retrieved_document | long_context | `retrieved_document - rag` | w/l/t | `long_context - retrieved_document` | adoption |
| --- | ---: | ---: | ---: | ---: | ---: | :-: | ---: | --- |
| MamayLM 12B | 1 | 0.509 | 0.498 | 0.633 | -0.011 [-0.077, +0.052] | 14/13/55 | +0.135 [+0.076, +0.207] | inconclusive |
| MamayLM 12B | 5 | 0.515 | 0.498 | 0.621 | -0.017 [-0.034, -0.004] | 0/6/76 | +0.122 [+0.060, +0.186] | **reject** |
| Lapa 0.1.2 | 1 | 0.485 | 0.446 | 0.556 | -0.039 [-0.099, +0.018] | 11/13/58 | +0.110 [+0.056, +0.172] | inconclusive |
| Lapa 0.1.2 | 5 | 0.485 | 0.487 | 0.556 | +0.002 [-0.027, +0.035] | 4/3/75 | +0.069 [+0.016, +0.132] | inconclusive |

**The verdict is do not adopt, on both models and at both settings.** Not one of the four
`retrieved_document - rag` intervals lies above zero; the one interval that separates at all
separates BELOW it. Retrieval still pays decisively in every run (+0.359 to +0.393 over
closed-book, 49-58 item wins against 3-4 losses), and `long_context` still wins the ablation
(+0.071 to +0.124) -- but `oracle_document_gap` is separated above zero in all four runs, which is
the finding: on this corpus the whole long-context gain was the gold LABEL, not the document size.
Widening the unit of retrieval recovers none of it.

The two settings fail for different and instructive reasons, and the lane's own document-level
recall@k is what separates them:

- **`top_n=1` narrows retrieval as well as widening the unit.** One document is document-level
  recall@1 = 0.768 against the `rag` lane's chunk-level recall@5 = 0.951, identical on both models
  because retrieval is pinned. The answer follows the coverage: the found-rate (`contains`) falls
  0.646 -> 0.561 on MamayLM (1 win / 8 losses / 73 ties) and 0.610 -> 0.524 on Lapa (0/7/75). It is
  not a document-versus-chunk reading at all -- it is depth 1 versus depth 5.
- **`top_n=5` is the clean test, and it is flat to slightly negative.** Document-level recall is
  0.9512, exactly the `rag` lane's, so the retrieved set is held fixed and only the unit changes.
  On MamayLM the found-rate is then IDENTICAL (0.6463 both, 0 wins / 0 losses / 82 ties) and so is
  answer-side span coverage (0.8293 both); the entire -0.017 is token precision, 0.470 -> 0.452, on
  answers that got longer (mean 16.7 -> 17.5 completion tokens). All six "losses" are correct
  answers stated more fully -- `1169 році.` becoming `У 1169 році.`. The reject is real at the cut
  but it prices VERBOSITY, not knowledge, and it rests on exactly 6 discordant items, the minimum
  the sign test needs at 95%. On Lapa the same setting is +0.002 (4/3/75) with the found-rate
  slightly up (0.610 -> 0.622) -- indistinguishable from `rag` in either direction.

Because `separates` is one-sided by construction ("candidate ahead"), the derived table prints the
rejected MamayLM row's reading as `flat` while the adoption verdict says `reject`: the loss is read
off the mirrored interval gate, and the interval `[-0.034, -0.004]` in the same row is what the
verdict quotes.

**Operationally,** `top_n=1` is a cost lane rather than a quality lane: the prompt shrinks from
1202 to 328 tokens on MamayLM (872 -> 235 on Lapa) and throughput nearly doubles (10.0 -> 17.8
tok/s; 10.7 -> 19.3), which buys roughly 8-9 points of found-rate away. `top_n=5` costs a little of
both (1202 -> 1310 prompt tokens, 10.0 -> 9.2 tok/s) and buys nothing. No item was skipped in any
of the four runs: SQuAD-derived documents are ~1.5k characters, so the budget path never fired and
this reading says nothing about how the lane behaves on a corpus of long documents.

Contamination was 10-11/82 (12-13%) in every run, unchanged from the earlier cohort, so the uplift
is measured against the same non-zero baseline as before.

Reproducibility, measured, and one correction to the earlier claim: Lapa's `closed_book`, `rag`,
and `long_context` lanes were IDENTICAL across its two back-to-back runs (0.092 / 0.485 / 0.556,
same intervals), so the lane machinery adds nothing. Against the 2026-07-24 MamayLM artifact,
however, the grounded lanes did NOT reproduce exactly this time: retrieval is byte-identical
(recall@5 0.951, same ranked chunks), but 5 of 82 `rag` answers and 7 of 82 `long_context` answers
differ in phrasing, moving `rag` 0.5005 -> 0.5089 and `long_context` 0.6428 -> 0.6330. That is the
same GGUF decode nondeterminism previously recorded only for `closed_book`, now visible in the
grounded lanes across a month of host and roster change; it is far inside every interval above and
changes no verdict, but the earlier "the `rag` and `long_context` lanes reproduce exactly across
runs" claim holds only within a host state, not across one.

Adding the fourth lane changes nothing the other three report. Recomputing the comparison from the
same four persisted run bundles with and without `retrieved_document` gives byte-identical
`retrieval_uplift` and `long_context_delta` entries, byte-identical per-lane reports for
`closed_book` / `rag` / `long_context`, and the same ablation verdict and reason string -- the
paired bootstrap index sets depend only on the item count and the seed, and each fitting population
is scoped to its own pair.

Artifacts: `$DATA_DIR/context-ablation/<run>/` for the four runs above, each with its own
`report.md` and `comparison.json`, plus one ordinary `run-eval` bundle per (lane, run).

### Power-resolved Qwen3.6 long-context verdict (2026-07-25)

The target was declared from the earlier 82-item `final` artifact BEFORE new inference:
minimum detectable delta +0.060 objective, 80% power, two-sided alpha 0.05. Its per-item paired SD
was 0.3078, pricing the run at 207 items. Pooling all three verified splits of the same committed
fixture supplied 250 items (`final,tuning,calibration`), above target; this is a diagnostic
ablation, not a leaderboard or tuning result.

| model | n | closed_book | rag | long_context | retrieval uplift | long-context delta | p_positive | resolution | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `batiai/qwen3.6-35b:iq3` | 250 | 0.121 | 0.534 | 0.593 | +0.413 [+0.363, +0.461] | +0.058 [+0.025, +0.096] | 0.9995 | separated | `long_context_wins` |

The long-context row has 51 wins / 23 losses / 176 ties (two-sided sign p=0.0015), no context
skips, and is separated at all three reported confidence conventions. The extra power therefore
changes the earlier final-only `rag_pays_off` reading to a settled `long_context_wins`: for this
model and corpus, chunked retrieval loses a small but real amount to whole-document context.
Retrieval itself still pays decisively over closed-book (+0.413, `p_positive` 1.000), and 33/250
closed-book answers match the reference (13.2%).

The operator boundary remains important: `long_context` is oracle-grounded on each item's gold
document, so this result supports sending the whole document AFTER a source is known; it does not
remove the need to retrieve or route to that source.
[`retrieved_document`](#the-shippable-sibling-retrieved_document) is the lane that closes that
distance, and its own evidence below says how much of this gap it reaches.

Artifact:
`$DATA_DIR/context-ablation/20260725T-power-resolution/{power-plan.json,comparison.json,report.md}`.
The `final` split inside the pooled run independently reproduces the earlier grounded rows exactly
(`rag` 0.554, `long_context` 0.615), so the changed verdict comes from added items rather than a
changed lane.

### MamayLM 12B rerun on 12 GiB Blackwell (2026-07-28)

The full 82-item `final` comparison used the fitting Ukrainian MamayLM Gemma 3 12B Q4_K_M model on
the RTX PRO 3000 Blackwell and the same 311-chunk e5-base store. Closed-book scored 0.155, RAG
0.510, and oracle whole-document context 0.624. Retrieval uplift was +0.356
`[+0.272, +0.438]` with 49/4/29 wins/losses/ties; long-context minus RAG was +0.114
`[+0.051, +0.180]`. Both readings are separated at the neighbouring confidence conventions, no
item was skipped, 11/82 closed-book answers matched, and the verdict remained
`long_context_wins`. A new powered run was unnecessary for this host/model pair because the
82-item long-context reading is not borderline. Artifact:
`$DATA_DIR/context-ablation/20260728T113000Z-blackwell12-mamaylm12b/`.

Durable evidence (2026-07-22, CUDA host, Ollama, committed UA fixture
`samples/goldsets/ua_squad_postedited_v1/` -- 82 verified `final` items, 250-document corpus,
311 chunks at 800/120, `top_k=5`, `DATA_DIR=.data/context-ablation-host`):

| model | closed_book | rag | long_context | retrieval uplift | long-context delta | closed-book matches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MamayLM-Gemma-3-12B-IT v2.0 GGUF Q4_K_M | 0.160 | 0.501 | 0.643 | +0.340 [+0.262, +0.423] | +0.142 [+0.083, +0.206] | 10/82 (12.2%) |
| Lapa v0.1.2-instruct GGUF Q4_K_M | 0.100 | 0.496 | 0.576 | +0.396 [+0.314, +0.484] | +0.080 [+0.036, +0.133] | 12/82 (14.6%) |

Both models return `long_context_wins`, and both agree on the shape of the result:

- Retrieval pays for itself, decisively. The uplift interval is far clear of zero for both models
  (sign-test p<0.001, 50/6/26 and 59/3/20 item wins/losses/ties). RAG is not decoration on this
  corpus.
- Whole-document stuffing still beats chunked retrieval, by a smaller but separable margin. That
  is expected here and is NOT an argument to ship long context: SQuAD-derived documents are ~1.5k
  characters, the lane is oracle-grounded on the item's own gold document, and `rag` retrieval was
  already near-ceiling (`recall@5=0.951`). The measured gap is what the retrieval layer still
  loses to chunk boundaries when the right document is known for free.
- Roughly one item in eight is answered correctly with no context at all -- parametric knowledge
  or contamination of a public post-edited SQuAD set. Any uplift on this fixture is therefore
  measured against a baseline that is not zero.

Skip path, measured (same model and item set, `context_budget: 1250` to force overflow):
28/82 items skipped, and the two populations diverge exactly as designed -- all-items
`long_context_delta` reads `-0.085 [-0.188, +0.018]` (the 28 skips score zero) while
`long_context_delta_fitting` over the remaining 54 reads `+0.165 [+0.091, +0.250]`. The verdict
reads the fitting delta, and the report carries both.

Reproducibility, measured: the `rag` lane's bundle is byte-identical to a plain `run-eval` of the
same configuration (all 82 items: same answers, same per-case scores), which is the check that the
lane machinery adds nothing to the leaderboard path. The `rag` and `long_context` lanes reproduce
exactly across runs WITHIN one host state -- across a month they drift too, by 5-7 of 82 answers
([2026-08-23](#the-shippable-document-lane-does-not-pay-reject-2026-08-23)); the `closed_book` lane
does not reproduce even back to back -- 11/82 answers differed between two
identical invocations (lane mean 0.160 vs 0.153), because an ungrounded prompt leaves a much
flatter next-token distribution for GGUF kernel nondeterminism to flip. The drift is well inside
the uplift interval half-width (~0.08) and changed no verdict, but a closed-book number is a
noisier measurement than a grounded one and should be quoted with that in mind.

Reports: `$DATA_DIR/context-ablation/20260722T142639Z/` (MamayLM),
`.../20260722T143030Z/` (Lapa), `.../20260722T143459Z/` (the budget-constrained skip run).

### Roster-wide ablation cohort (2026-07-24)

The same lane, host, index fingerprint, and item set extended to the Gemma 4, MamayLM v2.0, and
Qwen3.6 rosters. `rag` recall@5 is 0.951 for every row (retrieval is pinned), so all differences
are answer-side. Throughput is the `rag` lane's measured tokens/s.

| model | closed_book | rag | long_context | retrieval uplift | long-context delta | closed-book matches | rag tok/s | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `batiai/qwen3.6-35b:iq3` | 0.133 | 0.554 | 0.615 | +0.421 [+0.333, +0.503] | +0.060 [-0.008, +0.130] | 10/82 (12.2%) | 6.1 | `rag_pays_off` |
| MamayLM-Gemma-3-27B-IT v2.0 GGUF Q4_K_M | 0.193 | 0.546 | 0.609 | +0.353 [+0.269, +0.436] | +0.063 [+0.014, +0.124] | 12/82 (14.6%) | 3.3 | `long_context_wins` |
| MamayLM-Gemma-3-12B-IT v2.0 GGUF Q4_K_M | 0.153 | 0.501 | 0.643 | +0.348 [+0.268, +0.429] | +0.142 [+0.083, +0.206] | 10/82 (12.2%) | 11.8 | `long_context_wins` |
| `gemma4:e4b` | 0.062 | 0.365 | 0.470 | +0.303 [+0.242, +0.364] | +0.105 [+0.056, +0.163] | 5/82 (6.1%) | 31.8 | `long_context_wins` |
| `gemma4:26b` | 0.097 | 0.288 | 0.410 | +0.190 [+0.138, +0.240] | +0.122 [+0.081, +0.169] | 11/82 (13.4%) | 12.1 | `long_context_wins` |

Reports: `$DATA_DIR/context-ablation/20260724T0{65410,70414,73659,74544,75718}Z/` (gemma4:e4b,
gemma4:26b, MamayLM-12B, Qwen3.6-35B-A3B, MamayLM-27B).

What the wider cohort adds beyond the two-model result:

- **The original final-only Qwen3.6-35B-A3B row is the cohort's only `rag_pays_off`.** Its
  `long_context_delta` is +0.060 [-0.008, +0.130] (sign p=0.210), the one 82-item interval that
  straddles zero, and it posts the largest retrieval uplift measured (+0.421). The powered
  250-item run above resolves that near-miss as `long_context_wins`, so the final-only row is kept
  here as the reference observation that priced the larger run, not as the current operator
  verdict.
- **A tie at the top, at very different cost.** Qwen3.6-35B and MamayLM-27B are statistically
  indistinguishable on `rag` (0.554 vs 0.546) and on the context-position probe (paired
  +0.006 [-0.048, +0.059]), but Qwen serves from VRAM at 13 GB with ~3B active parameters while
  the 27B's 18 GB artifact runs at 23%/77% CPU offload -- 6.1 vs 3.3 tok/s on this lane, and
  18.5 vs 6.5 on closed-book. Quality-first ranking calls this a tie; the tiebreak is throughput.
- **Closed-book tracks Ukrainian specialization, not size.** MamayLM-27B leads the cohort at
  0.193 and the Gemma 4 rows sit at 0.062-0.097, so the contamination/parametric baseline a given
  uplift is measured against is model-specific and must be quoted with the uplift.
- The `long_context` lane skipped nothing for any model, so no fitting-population split applies.

Reproducibility, measured: MamayLM-12B reproduced its 2026-07-22 grounded lanes exactly
(`rag` 0.501, `long_context` 0.643, `long_context_delta` +0.142 [+0.083, +0.206]) while its
closed-book lane again landed at 0.153 against the original 0.160 -- an independent confirmation
of the closed-book nondeterminism documented above, on a re-run 2 days later.
