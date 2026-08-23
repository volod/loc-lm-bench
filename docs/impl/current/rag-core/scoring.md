# Generation Graph And Scoring

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

## Generation Graph

`src/llb/eval/graph.py` builds the retrieve-generate flow. LangGraph is imported only when the
graph is built. The graph records one status per case: `ok`, `empty`, `malformed`, `schema_invalid`,
`refusal`, `timeout`, `backend_error`, `retrieval_miss`, `context_overflow`, or another typed
failure from the shared taxonomy. `schema_invalid` is reachable only on a declared-answer-format
run (see [the typed answer envelope](#typed-rag-answer-envelope-typed-rag-answer-envelope)).
`retrieval_miss` and `context_overflow` are the pre-generation statuses
(`eval_common.PRE_GENERATION_STATUSES`): the prompt is never sent, so the answer stays empty and
the case scores zero rather than being quietly repaired into a different prompt.

Two optional seams keep diagnostic lanes out of the retrieval path itself. `context_source`
replaces the retrieve node's store lookup with a `RagState -> RagState` closure that supplies its
own context, and `template_id` overrides the generation prompt; both are resolved from
`RunConfig.context_strategy` in `_default_runner_fn` and are how the context ablation runs without
special-casing anything downstream (see [context ablation](context-ablation.md)).

`src/llb/backends/openai_client.py` normalizes endpoint failures. Backend launchers own process
lifecycle and readiness checks.

## Scoring

`src/llb/scoring/correctness.py` computes objective correctness using normalized token F1 and
persists its token precision and token recall components beside exact match and the strict
all-reference-tokens `contains` signal. `--score-semantic` records a pinned-embedder cosine signal
for paraphrases and morphology; it is kept separate from the objective unless a ranking policy
explicitly uses it.

`src/llb/scoring/judge/model.py` owns the calibration gate and outcome policy;
`src/llb/scoring/judge/scorer.py` normalizes scores and handles empty answers; and
`src/llb/scoring/judge/deepeval_adapter.py` runs the optional local DeepEval integration. The judge
enters ranking only when the caller supplies a calibration rho that clears the trust threshold.
Otherwise it is diagnostic and the declared base quality ranks alone.

`src/llb/scoring/aggregate.py` produces leaderboard rows. RAG base quality is 75% token recall
(fact coverage) plus 25% token precision (answer-format adherence); `objective` remains token F1
for continuity. The policy favors base quality first, then throughput, then lower VRAM when
telemetry is available.

### Measured: the headline objective is partly a verbosity ranking

2026-07-26. Token F1 is a single number over two different things -- whether the answer carries the reference
fact, and how much else it carries -- and the recorded evidence base contains the one comparison
that separates them: the context-ablation `rag` lanes score the SAME 82-item committed fixture under
PINNED retrieval (recall@5 = 0.951 for every row), so all differences are answer-side. Read-only
audit over those bundles:

| model | median completion tokens | objective (token F1) | contains | exact | r(len, objective) | items found | implied token precision when found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `batiai/qwen3.6-35b:iq3` | 10 | 0.554 | 0.659 | 0.329 | -0.305 | 54 | 0.709 |
| MamayLM-Gemma-3-27B v2.0 | 9 | 0.546 | 0.622 | 0.329 | -0.219 | 51 | 0.712 |
| MamayLM-Gemma-3-12B v2.0 | 10 | 0.501 | 0.634 | 0.293 | -0.363 | 52 | 0.650 |
| `lapa-v0.1.2-instruct` | 10 | 0.496 | 0.610 | 0.329 | -0.541 | 50 | 0.655 |
| `gemma4:e4b` | 25 | 0.365 | 0.756 | 0.122 | -0.331 | 62 | 0.339 |
| `gemma4:26b` | 33 | 0.288 | 0.720 | 0.012 | -0.367 | 59 | 0.230 |

On an item where `contains` is 1.0 the reference tokens are all present, so token recall is 1.0 and
`F1 = 2P/(1+P)` inverts to the token PRECISION the answer paid; that is the last column, computed
only over each model's own found items.

- **The two readings rank the roster in nearly opposite orders.** `gemma4:26b` is LAST by the
  headline objective (0.288) and SECOND by found-rate (0.720); it states the reference on 59 of 82
  items where the objective leader states it on 54. `gemma4:e4b` finds the most (62) and ranks 5th.
- **Length is the mechanism, and it is measured.** The two Gemma 4 rows emit 2.5x-3.3x the
  completion tokens of the rest (median 25 and 33 against 9-10) and pay for it in precision on
  exactly the items they got right (0.339 and 0.230 against 0.650-0.712). Answer length correlates
  negatively with the objective for EVERY model (r -0.22 to -0.54).
- **Neither number is the right headline on its own.** `contains` rewards the same verbosity the
  objective punishes -- an answer that repeats the whole context would score 1.0 -- and the shipped
  `eval.rag` system prompt does ask for a short answer ("Відповідай стисло"), so part of the Gemma 4
  penalty is a real instruction-following failure. What the single number cannot do is say WHICH of
  the two happened, and the leaderboard currently ranks on it alone.

### Headline decomposition and declared ranking policy

Shipped 2026-07-28. Each new `scores.jsonl` row carries `token_precision`, `token_recall`, and
`ranking_score` beside the unchanged `objective_score` / `token_f1`, `contains`, and
`completion_tokens`. The run manifest and leaderboard aggregate those into precision, recall,
found-rate, and mean-completion-token columns. `quality` is now the declared
`recall_75_precision_25` score for decomposed RAG rows:

```text
quality = 0.75 * token_recall + 0.25 * token_precision
```

Fact coverage is primary because a RAG answer that omits the reference fact has failed its main
job. Format remains a material quarter of the score because `eval.rag` explicitly asks for a
short answer. Token F1 remains the stable `objective` column, so the change does not rewrite
historical correctness values. Legacy bundles without decomposition columns continue to rank on
their objective rather than receiving fabricated components. A trusted judge, when available,
blends with the declared base quality; all category tiers retain their existing objectives.
`quality_per_watt` uses the same declared quality.

`make analyze-verbosity RUN_DIRS="<bundle> <bundle> ..."` runs the maintained fixed-item study in
`src/llb/eval/verbosity_sensitivity.py` and writes `report.{json,md}` under
`$DATA_DIR/verbosity-sensitivity/<run>/`. It refuses bundles with different ordered item IDs,
duplicate models, missing decomposition columns, or an `objective_score` that is not
bit-identical to `token_f1`. The JSON contains each model's rank under token F1, recall-only,
found-rate, and the selected policy, plus per-model and roster length correlations.

CUDA-host evidence over three locally installed models on the same 82-item final fixture, flat
recursive retrieval, and pinned recall@5 = 0.951 is under
`$DATA_DIR/verbosity-sensitivity/20260728T142750.517338Z-c0d8009a807d/`. Candidate inference ran
on the RTX PRO 3000 Blackwell GPU while the pinned embedder stayed on CPU for VRAM headroom:

| model | precision | recall | token F1 | found-rate | policy quality | mean completion tokens | r(length, F1) | F1 rank | policy rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemma4:e4b` | 0.274 | 0.814 | 0.358 | 0.768 | 0.679 | 29.9 | -0.335 | 3 | 1 |
| `qwen3:14b` | 0.380 | 0.727 | 0.440 | 0.622 | 0.640 | 34.8 | -0.462 | 2 | 2 |
| MamayLM-Gemma-3-12B v2.0 | 0.467 | 0.688 | 0.510 | 0.634 | 0.633 | 17.3 | -0.385 | 1 | 3 |

Length versus token F1 is negative within every model and across the three-row roster
(`r=-0.665`). Length versus the selected policy is positive across this small roster (`r=0.381`),
which makes the policy tradeoff visible instead of silently treating brevity as correctness.
The named rank changes are MamayLM from F1 rank 1 to policy rank 3 and Gemma 4 E4B from F1 rank 3
to policy rank 1. Qwen3 stays rank 2. The raw run bundles are:

- `$DATA_DIR/run-eval/20260728T141559.740730Z-2cda419dc495/` (MamayLM)
- `$DATA_DIR/run-eval/20260728T141903.934343Z-8f357a77ea1a/` (Qwen3)
- `$DATA_DIR/run-eval/20260728T142339.216084Z-ddc692ec71d8/` (Gemma 4 E4B)

The per-case decomposition, verbose-correct / terse-partial fixture, rank reversal, legacy-bundle
refusal, ASCII report, aggregate rendering, board reload, and unchanged token-F1 objective are
covered by `tests/llb/scoring/test_correctness.py`, `tests/llb/scoring/test_aggregate.py`,
`tests/llb/eval/test_verbosity_sensitivity.py`, and the executor / board suites.

### Groundedness and citation metrics (groundedness-citation-metrics)

Shipped: four answer-side signals that go beyond reference-answer overlap, all deterministic and
additive -- they never change the headline objective (they stay separate columns until a ranking
policy explicitly adopts them). `src/llb/scoring/groundedness.py` is a pure, dependency-free scorer
(no RAGAS, no frontier judge); the calibration-gated judge's faithfulness stays the optional
secondary groundedness signal.

- Groundedness fraction (`--score-groundedness`): the share of the answer's sentence-ish claims
  SUPPORTED by any retrieved chunk via token-overlap matching (a claim is supported when
  `GROUNDEDNESS_SUPPORT_THRESHOLD`=0.6 of its content tokens appear in a chunk). A fully-supported
  answer scores 1.0; an answer whose claims are absent from the context scores near 0.0.
- Citation validity + hallucinated-citation rate (`--cited-answers`): swaps in the
  `eval.rag.cited_answer` generation prompt (requires `[i]` chunk citations, reusing the numbered
  format `format_context` emits) and validates each citation against the chunk it points at, in
  PROMPT-LAYOUT order (so `reverse_rank` renumbering is respected). A citation whose in-range chunk
  lacks the claim is flagged invalid (lowers validity); a citation whose index is out of range is
  hallucinated.
- Citation coverage (`--cited-answers`, citation-coverage-metric): the share of countable claims
  (>= `MIN_CLAIM_TOKENS` content tokens, the same rule groundedness counts by) that carry ANY
  `[i]` citation, right or wrong. Validity alone collapses two failures into one low number -- a
  model that emits NO citations and a model that cites the WRONG chunk both score 0.0 validity
  (the durable llama3.2:3b run below made that concrete). Coverage separates them: coverage 0.0 =
  an instruction-following gap (does not cite); coverage high with validity low = a grounding gap
  (cites, but points at the wrong chunks). Reported beside validity, fully independent of it.
- Insufficient-context abstention probe (`--insufficient-context-probes <n>`,
  `src/llb/eval/insufficient_context.py`): re-runs a seeded sample of gold items with every chunk
  overlapping their gold spans EXCLUDED from retrieval (through the shipped chunk-metadata filter
  seam). Correct behavior is an explicit abstention (`llb.eval.common.is_abstention` = refusal OR an
  insufficient-context marker), scored as abstention accuracy. Probe rows live in `probes.jsonl` (+
  `insufficient_context_report.md`), NEVER in `scores.jsonl`, so they cannot enter the plain
  correctness aggregates.

Per-case fields land in `scores.jsonl` (`groundedness`, `citation_validity`, `citation_coverage`,
`hallucinated_citation_rate`, `n_citations`); their means plus `abstention_accuracy` / `n_probes`
land in the manifest `metrics`, echoed as the run's `answer-side:` summary line. Config knobs
(`cited_answers`, `score_groundedness`, `insufficient_context_probes`) are recorded in the manifest
fingerprint. `RunConfig` toggles are off by default, so pre-existing bundles keep their shape.

Modules/tests: `src/llb/scoring/groundedness.py`, `src/llb/eval/insufficient_context.py`, the
`eval.rag.cited_answer` template, `ScoreOptions` in `src/llb/executor/cases.py`;
`tests/llb/scoring/test_groundedness.py` (fully/partially/unsupported groundedness with zero
cross-class leakage, valid/flagged-invalid/hallucinated citations, coverage separating no-citation
from wrong-citation at equal validity, abstention markers, cited-answer prompt wiring, per-case
scoring + context-order-aware citation numbering, manifest mean coverage) and
`tests/llb/eval/test_insufficient_context.py` (gold exclusion, seeded sampling, abstention
accuracy, transport-error exclusion).

Durable evidence (2026-07-09, `llama3.2:3b` on Ollama, `intfloat/multilingual-e5-base` flat FAISS
over `samples/goldsets/ip_regulation_uk`, final split n=4, `--cited-answers --score-groundedness
--insufficient-context-probes 4`): mean groundedness 0.625 (per-case 1.0 / 0.5 / 1.0 / 0.0);
citation validity 0.000 with hallucinated-citation rate 0.000 -- the 3B model largely IGNORED the
`[i]` citation instruction (mostly emitted no citations), so validity is dominated by "did not cite"
rather than "cited wrongly"; abstention accuracy 0.000 -- on all four probes the model FABRICATED an
answer (even citing non-existent chunks) instead of abstaining when its gold evidence was removed.
Honest, unflattering evidence that a small model's answer-side grounding discipline is weak -- exactly
the axis these metrics expose beyond a passing recall@k.

### Answer-side gold-span coverage (answer-side-span-coverage-metric)

Shipped: `answer_span_coverage`, `answer_all_spans`, and `answer_spans_measured` on every
`scores.jsonl` row, from `src/llb/scoring/answer_spans.py`. They are the ANSWER-side twins of the
retrieval pair `span_coverage_at_k` / `all_spans_at_k` ([retrieval
metrics](retrieval-metrics.md#retrieval-metrics)): the retrieval pair says whether the
CONTEXT carried each labeled span, these say whether the ANSWER states it.

Why the objective cannot answer that question: `objective_score` is reference-answer token F1 over
the whole answer, so a two-hop answer that states one fact in the reference's own words scores the
same as a terse answer that states both -- the unit tests pin a pair that earns byte-identical
token F1 and differ 1.0 against 0.0 here. Every multi-hop answer-quality verdict rested on that
single number ([answer-quality
evidence](../graphrag-backend/answer-quality-evidence.md#answer-quality-evidence)).

Per gold span, what the answer must carry is not the span's text but the fact it contributes:

```text
grounded(span)    = content of (span text) AND content of (reference answer)
distinctive(span) = grounded(span) MINUS question content MINUS every OTHER span's content
required(span)    = distinctive(span), or grounded(span) when that leaves nothing to judge
```

Terms are Ukrainian lemmas (the same pinned pymorphy3 lemmatizer the lexical index uses, injectable
for tests) plus numerals matched literally, over the scoring tokenizer in
`src/llb/scoring/correctness.py`, with a scoring-owned function-word table dropped from both sides.
A span counts as carried when the answer holds at least `SPAN_CARRIED_MIN_SHARE` = 0.5 of its
required terms AND every required numeral -- a table fact restated with the wrong number is not
carried, however much of the wording around it is reproduced. `answer_span_coverage` is the share
of judgeable spans carried, `answer_all_spans` the all-or-nothing gate over them, and
`answer_spans_measured` the count behind both, so a vacuous 1.0 (nothing judgeable) never reads as
a carried one.

Each subtraction removes a way to score a fact the model never supplied, and the fallback is what
keeps them safe on a literal ledger:

- Without the reference intersection a correct one-line answer reads 0.05 against the registry
  paragraph grounding it, because a labeled span is routinely several times longer than its answer.
- Without the question subtraction, naming a hop's subject carries the hop ("the trademark
  certificate - unknown" reproduces most of that span's wording and none of its fact).
- Without the sibling-span subtraction, vocabulary the two hops share -- units, dates, the shared
  subject -- lets one hop's answer satisfy the other.
- The fallback fires on 42% of the spans of the drafted goods ledger, where the reference restates
  its question almost verbatim and nothing distinctive is left. Deleting given terms outright
  instead of falling back leaves 43% of the labeled spans unjudgeable, which is why the fallback
  exists rather than a stricter rule.

Read the pair BESIDE the objective, never instead of it. Coverage is a recall-side reading, so a
model that dumps its whole context scores 1.0 by construction; `token_precision` and
`ranking_score` in the same row are what price that, and the two together are what separate a terse
complete answer from a verbose one ([the verbosity
reading](#measured-the-headline-objective-is-partly-a-verbosity-ranking)). The reading is also NOT
bounded by the retrieval pair: an item whose context carried one hop can still answer both from
what the model already knows, and that gap is now visible rather than invisible.

Calibration, over the 588 span readings of the 14 recorded multi-hop bundles (read-only audit, no
model call): 76% of readings are exactly 0.0 or 1.0 and 10% fall anywhere near the cut, so the
threshold sits in an empty middle rather than on a slope. Where the strict `contains` signal fires
on a single-span item the span is carried in 49 of 49 cases, and where `exact` fires in 28 of 28;
no item reads "not carried" with an objective above 0.8, while 11 read "carried" with an objective
below 0.2 -- the answers stating the fact in words the reference did not use, which is the case the
objective cannot express.

Modules/tests: `src/llb/scoring/answer_spans.py` (with its function-word table in
`src/llb/scoring/function_words.py`), the columns in `src/llb/executor/cases.py` and
`llb.core.contracts.rag.AnswerSpanScores`; `tests/llb/scoring/test_answer_spans.py` (both facts,
one fact, neither, paraphrase in other grammatical forms, the wrong-numeral gate, the
question-echo case, the fallback, an unjudgeable span, and the token-F1 tie the pair separates) and
`tests/llb/executor/test_runner_backend.py` (the per-case columns). The answer-quality lane's use of
them is in [answer-quality
evidence](../graphrag-backend/answer-quality-evidence.md#answer-quality-evidence), and what they
measured on the recorded multi-hop comparison is [answer-side coverage
evidence](../graphrag-backend/answer-side-coverage-evidence.md#measured-result-the-answers-do-not-state-the-evidence-the-fused-lane-adds).

## Validation architecture: where a completion becomes typed

Two lanes now parse a model completion into a typed object, and they are deliberately different
things at different places:

| Lane | Where | What it validates | On failure |
| --- | --- | --- | --- |
| Structured-output BENCHMARK (`src/llb/scoring/structured/schema.py`) | inside one benchmark tier | a PER-CASE field schema compiled by `build_model` | `is_conformant` records False; the case is scored not-conformant |
| RAG ANSWER contract (`src/llb/eval/answer_envelope/`) | the generation boundary of every RAG run | ONE fixed contract, the same for every case | a typed terminal status (`malformed` / `schema_invalid`), after one bounded repair |

The answer boundary is `llb.eval.answer_envelope.boundary.parse_envelope`: the single place a RAG
completion becomes an `AnswerEnvelope`. Nothing downstream re-parses model text -- the scorers read
declared fields -- so the question "is this answer well-formed?" has exactly one answer per case,
recorded rather than re-derived. The JSON extraction itself is the benchmark lane's own
`parse_output`, so a fenced or prose-wrapped object is recovered identically on both sides.

## Typed RAG answer envelope (typed-rag-answer-envelope)

Shipped: `--answer-format envelope` (`make run-eval ANSWER_FORMAT=envelope`) asks the model for a
declared answer instead of prose, validates it at the generation boundary, and reads every
answer-side signal off the declared fields. Off by default (`free_text`), so an existing bundle and
an existing command record exactly what they recorded before.

### The contract

`src/llb/eval/answer_envelope/models.py` is the whole contract, as Pydantic models:

- `answer` -- the Ukrainian answer text. It is scored EXACTLY as the free-text answer of the same
  string would be; the envelope changes where the string comes from, never how correctness is
  computed.
- `abstained` -- a required, explicit flag. "The context does not carry it" stops being a regex over
  apology stems (`llb.eval.common.is_abstention`) and becomes something the model said.
- `claims[]` -- each factual statement with `citations`, the prompt-position indices it rests on,
  and an optional `triple` (subject / relation / object) whose two type fields are normalized into
  the CLOSED 13-type entity vocabulary (`llb.prep.ontology.extraction.entity_types`): a synonym
  canonicalizes, an invented type collapses to `MISC`, so the schema cannot silently expand.
- `evidence[]` -- optional verbatim quotes per chunk. Requested in the prompt and useful to a
  reviewer, but the citations already carry what the metrics need, so omitting them is not a
  contract failure.

Unknown extra keys are ignored rather than rejected: a model that adds `"confidence"` still emitted
the contract, and conformance should measure the declared fields, not decoration around them. The
prompt's worked example is the model instance `ENVELOPE_EXAMPLE` serialized by
`envelope_schema_block()`, so what the model is ASKED for and what it is CHECKED against cannot
drift apart.

### The two statuses, and the one repair

`malformed` keeps its meaning -- the completion is not JSON at all. `schema_invalid` is new: the
completion IS JSON and does not satisfy the contract. They are separate because they call for
different fixes (a decoding or prompt problem versus specific fields the model got wrong), and one
number cannot say which happened.

On a failure the boundary spends exactly ONE repair reprompt (`eval.rag.envelope_repair`), carrying
the validator's own complaint back to the model -- the same bounded policy shape the
[agent loop-policy lane](../extended-workflows/loop-policy-recommendation.md#agent-loop-policy-recommendation)
measures for tool calls. It is bounded on purpose: an unbounded repair loop converts a formatting
failure into an unmeasured token cost, and measuring that failure is the point. A transport failure
(`timeout` / `backend_error`) is never repaired -- the run's own retry policy owns that. A repaired
case is charged for BOTH generations in its usage accounting, so the format's real token cost is
visible.

A valid envelope's terminal status is then `classify_response` over its DECLARED answer text, so an
empty answer is `empty` and a declined one is `refusal` exactly as on the free-text path.

### What is recorded

Per case (`scores.jsonl`), present only on an envelope run: `envelope_status`, `repaired`,
`n_claims`, `envelope_abstained`. `repaired` means the reprompt was ISSUED, so first-attempt
conformance reads off the bundle as `1 - repair_rate` and the repair's contribution is the gap up to
final conformance. Run metrics add `envelope_conformance`, `envelope_schema_invalid_rate`,
`envelope_malformed_rate`, `envelope_repair_rate`, and `mean_claims`, echoed on the run's
`answer-side:` line. `answer_format` is recorded in the manifest fingerprint like every other knob.

`--score-groundedness` and `--cited-answers` still decide WHICH answer-side columns exist; the
envelope decides where they are READ FROM. Declared claims and citations replace punctuation
splitting and `[i]` scraping, under the same support threshold and the same countable-claim floor
(`llb.eval.answer_envelope.metrics` reuses `chunk_supports_claim` / `content_tokens` from
`llb.scoring.groundedness`), so an envelope run and a free-text run stay comparable column by
column. Citations are validated in PROMPT-LAYOUT order, so `reverse_rank` renumbering is respected
exactly as the scraped `[i]` validation already respects it. The
insufficient-context abstention probe is deliberately NOT converted: it is a separate lane with its
own prompt and its own artifact, so it keeps scoring abstention by marker
(`llb.eval.common.is_abstention`) and its numbers stay comparable with every probe run recorded
before the envelope existed.

### Reading the conformance study

`make analyze-answer-envelope RUN_DIRS="<bundle> <bundle> ..."`
(`src/llb/eval/answer_envelope/study.py`) compares roster models over ONE item set and writes
`report.{json,md}` under `$DATA_DIR/answer-envelope/<run>/`. It refuses a free-text bundle, a
duplicate model, a single bundle, or bundles over different item sets. The report keeps three
things apart on purpose: conformance from correctness (a repair gain is a FORMATTING gain), first
attempt from final, and truncation from non-conformance -- `truncated` is the share of the
NON-conformant cases whose completion reached the run's token cap, because a cut-off completion is
not JSON either. The envelope is several times longer than a short free-text answer, so raise
`MAX_TOKENS` for the lane; a run that does not will measure its own budget.

Modules/tests: `src/llb/eval/answer_envelope/` (`models`, `boundary`, `lane`, `metrics`, `study`),
the `eval.rag.envelope` / `eval.rag.envelope_repair` templates, `answer_format` on `RunConfig`,
`ScoreOptions` in `src/llb/executor/cases.py`, `_attach_envelope_metrics` in
`src/llb/executor/runner_metrics.py`, and the CLI `analyze-answer-envelope`;
`tests/llb/eval/test_answer_envelope.py` (contract-versus-prompt drift, closed-vocabulary
normalization, fenced JSON, the malformed/schema_invalid split, every terminal status, the bounded
repair and its two-generation cost, the untouched free-text update, journal coverage),
`tests/llb/eval/test_answer_envelope_scoring.py` (declared citation validity / coverage /
hallucination, prompt-layout order, declared groundedness, objective equality with the same free
text, the columns present only on an envelope run, the whole vertical through the runner), and
`tests/llb/eval/test_answer_envelope_study.py` (the study's separations and its four refusals).

### Measured: conformance is a model property, and one reprompt can be worth 46 points

2026-08-23, RTX 4060 Ti 16 GB CUDA host, ollama backend. Three UA-capable instruct models, each
over the SAME committed 82-item final split of `ua_squad_postedited_v1`, against the same flat
`intfloat/multilingual-e5-base` store (`top_k=5`, recall@5 = 0.951 and MRR 0.835 for every row --
identical retrieval, so every difference below is answer-side). Command per model:
`make run-eval ANSWER_FORMAT=envelope MAX_TOKENS=768 LIMIT=82 SCORE_GROUNDEDNESS=1
CITED_ANSWERS=1 MODEL=<model> BACKEND=ollama`, then `make analyze-answer-envelope` over the three
bundles.

| model | conformance | first attempt | repaired | rescued | schema_invalid | malformed | objective | found |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MamayLM-Gemma-3-12B v2.0 | 1.000 | 1.000 | 0.000 | 0 | 0.000 | 0.000 | 0.495 | 0.671 |
| `lapa-v0.1.2-instruct` | 0.890 | 0.890 | 0.110 | 0 | 0.049 | 0.061 | 0.512 | 0.500 |
| `aya-expanse:8b` | 0.854 | 0.390 | 0.610 | 38 | 0.146 | 0.000 | 0.422 | 0.549 |

- **The format is a property of the weights, not of the harness.** One prompt, one validator, one
  item set: 82 of 82 conformant on MamayLM, 73 of 82 on lapa, and only 32 of 82 on Aya's first
  attempt. This is exactly why the envelope is opt-in per model rather than switched on by
  construction.
- **The bounded repair is worth 46.4 points on one model and nothing on another.** Aya emitted the
  contract on 32 first attempts and, told what the validator rejected, on 38 of the remaining 50 --
  0.390 to 0.854. Lapa was reprompted on all 9 of its failures and recovered NONE of them. The
  reprompt is a FORMATTING intervention, and the table keeps it in its own columns: neither model's
  objective moved because of it.
- **The two failure statuses genuinely separate.** Aya never once failed to emit JSON (malformed
  0.000); all 12 residual failures were JSON of the wrong shape. Lapa's 9 failures split 5 / 4 the
  other way. Collapsing these into one "malformed" number, as the free-text path must, would have
  pointed both models at the wrong fix.
- **Conformance does not rank the roster the way correctness does.** Lapa has the best objective
  (0.512) and the middle conformance; MamayLM has perfect conformance and the middle objective
  (0.495). A model that cannot emit the shape is not thereby a model that does not know the answer,
  and the report prints the two orders side by side so the distinction cannot be lost.
- **The citation gap is now readable, which was the point.** Under declared claims MamayLM cites on
  0.976 of countable claims with 0.774 validity and ZERO hallucinated citations; lapa 0.890 / 0.632;
  Aya 0.842 / 0.657. The durable free-text 3B run recorded citation validity 0.000 -- a number that
  meant "did not cite", not "cited wrongly". With coverage near 1.0 by construction, validity now
  measures grounding alone.
- **A format failure is a typed status, not a silent zero.** Because a non-conformant case ends in
  `malformed` / `schema_invalid`, it lowers `reliability` (0.976 / 0.890 / 0.841) instead of
  entering the correctness mean as a wrong answer.
- **The format costs roughly fifteen times the completion tokens.** Mean completion tokens were
  258.9 / 217.8 / 227.4 against the 17.3 the same MamayLM tag recorded on this same fixture in the
  free-text verbosity study above. That is the price of the declaration and it is not small; at
  `MAX_TOKENS=768` truncation was not the story (one of lapa's nine failures reached the cap, none
  of Aya's), but a run left at a short free-text budget would have measured its own cap.

What would overturn this: a different prompt (the contract's worked example is part of what is
being measured, not a neutral instrument), a larger completion budget for the lapa truncation case,
or a roster with a reasoning model whose thinking preamble the JSON extractor has to survive --
none of the three models here emits one. The conformance numbers are per (model, prompt, budget)
and do not transfer to another quantization or another serving stack.
