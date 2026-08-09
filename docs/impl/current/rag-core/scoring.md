# Generation Graph And Scoring

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

## Generation Graph

`src/llb/eval/graph.py` builds the retrieve-generate flow. LangGraph is imported only when the
graph is built. The graph records one status per case: `ok`, `empty`, `malformed`, `refusal`,
`timeout`, `backend_error`, `retrieval_miss`, `context_overflow`, or another typed failure from the
shared taxonomy. `retrieval_miss` and `context_overflow` are the pre-generation statuses
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
audit over those bundles, harness and output under
`$DATA_DIR/paired-reading-audit/20260726T100856Z/verbosity{_probe.py,.txt}`:

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
