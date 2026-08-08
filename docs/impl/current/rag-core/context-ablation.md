# Context Ablation: Does RAG Pay For Itself? (rag-vs-long-context-ablation)

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

A leaderboard row says how well a model answers WITH retrieval; it never says how much of that
score retrieval bought. `llb compare-context-strategies` (`make compare-context-strategies`)
scores ONE item set end to end under three context lanes and reports the differences.

`RunConfig.context_strategy` selects the lane and is recorded in the manifest fingerprint like
every other knob, so a lane's bundle is reproducible from its own config
(`make run-eval CONTEXT_STRATEGY=<lane>`):

- `rag` (default) -- retrieve as configured. This is the leaderboard row; the other two are
  DIAGNOSTICS and never rank a model.
- `closed_book` -- no context at all. `src/llb/eval/context_ablation/sources.py` supplies an empty
  context and swaps in the `eval.rag.closed_book` prompt, which asks the model to answer from its
  own knowledge (the RAG system prompt would push it to abstain). The empty context deliberately
  does NOT raise `retrieval_miss`: that status short-circuits generation, and a lane that never
  calls the model measures nothing.
- `long_context` -- the item's whole gold source document(s) laid into the prompt as one
  offset-exact chunk per document, with the SAME generation prompt as `rag`, so the delta is
  attributable to the context and not to prompt wording. The lane is oracle-grounded (it reads the
  item's own gold `doc_id`s), which makes it a ceiling, not a shippable retrieval policy.

Budget and skips: the lane resolves the model's usable window ONCE per run --
`resolve_model_spec` looks the served artifact up through `candidate_sources`, so an Ollama GGUF
tag resolves to its roster entry priced at the right quant -- and each item is checked with
`fits_context_chars` (`src/llb/optimize/tuning_space.py`, the same arithmetic as `fits_context`).
An item whose document does not fit terminates as `context_overflow`, a new pre-generation status
in the shared taxonomy: no model call, no truncation. A truncated document is a different and
unstated retrieval policy, so crediting its answer to "long context" would measure whichever slice
survived the cut. Without a manifest entry for the model, only an explicit `context_budget` /
`max_model_len` can bound the prompt, so an unlisted model skips nothing rather than everything.

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

Verdicts, in check order: `long_context_wins` | `rag_pays_off` | `retrieval_inconclusive` |
`no_retrieval_gain` | `no_evidence`. Every gate reads the paired INTERVAL, never the point
estimate. Artifacts: `$DATA_DIR/context-ablation/<run>/{report.md,comparison.json}`, plus one
ordinary `run-eval` bundle per (lane, split) under `$DATA_DIR/run-eval/`. CI drives all three
lanes over fake bundles and the committed fixtures
(`tests/llb/eval/test_context_ablation.py`), no backend or GPU.

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

## Context-ablation evidence

Each derived delta carries `p_positive` and a `(borderline)` flag, and the verdict names both the
rows it was decided on -- the retrieval uplift AND the long-context delta, because `_judge` checks
the long-context lane first
([how settled a paired reading is](paired-verdicts.md#how-settled-a-paired-reading-is----p_positive-and-the-borderline-flag)).
The original `qwen3.6-35b` final-only row was the one exception: its `rag_pays_off` rested on a
settled uplift (`p_positive` 1.000) but a long-context delta at `p_positive` 0.960 that a 90%
interval read as separated. The power-resolved run below removes that exception; every recorded
context-ablation verdict is now settled at the neighbouring 90%, 95%, and 97.5% conventions.

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
remove the need to retrieve or route to that source. The forward
[`retrieved-document-long-context-lane`](../../plan.md#retrieved-document-long-context-lane) task
owns that shippable bridge.

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
exactly across runs; the `closed_book` lane does NOT -- 11/82 answers differed between two
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
