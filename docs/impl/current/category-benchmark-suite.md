# Category Benchmark Suite

Category benchmarks score capabilities that are not well represented by the private RAG board:
text analysis, security, tooling, agentic workflows, summarization, structured output, and
reliability. Each category has its own tier and is never cross-ranked with RAG or public screens.

## Shared Substrate

`src/llb/bench/common.py` provides the reusable benchmark machinery:

- `local_complete` and `launcher_complete`: prompt-to-text drivers over local endpoints;
- `drive_with_backend`: reuse a running endpoint or launch a VRAM-owning backend under isolation;
- `category_result`: wrap per-case scores as a `ModelResult` with a category tier;
- `render_board`: rank within one tier through `rank_board`;
- `persist_category_run`: write canonical run bundles under `$DATA_DIR/<category>/<timestamp>/`;
- `run_gated_judge`: optional trusted-judge side signal, recorded alongside objective metrics.

The design keeps objective scoring as the headline floor. Judge signals can explain free-form
quality, but they do not replace deterministic task success.

## Text Analysis

Modules:

- `src/llb/scoring/text_analysis.py`;
- `src/llb/bench/text_analysis.py`;
- `src/llb/prep/text_analysis_corpus.py`;
- `src/llb/prep/chat_corpus.py`.

The schema is documented in `docs/design/text-analysis-schema.md`. Planted labels carry stable
ids, values, aliases, grounding offsets, attributes, and a kind. Objective sub-tasks include
`key_fact`, `entity`, `topic`, `trend`, `risk`, `decision`, and `contradiction`. Free-form
sub-tasks such as `narrative`, `insight`, and long-document answers use the gated judge path.
The synthetic planter validates kinds before grounding, drops ungrounded quote-bearing labels,
adds grounded evidence aliases, preserves label attributes such as trend direction, and increments
per-kind label ids only for records it emits.

Matching is deterministic: label id, normalized surface, aliases, then pinned-embedder cosine.
Trend labels are direction-aware; a right-subject wrong-direction answer receives no credit.
Contradiction labels require both paired sides to be covered.
The runner separates corpus/label loading, model extraction calls, per-document scoring, gated
free-form judge attachment, and artifact persistence into named helpers.

```bash
llb bench-text-analysis --bundle <bundle> --model <model> --backend <backend>
llb prepare-synthetic-corpus --text-analysis --out-dir <bundle> ...
llb ingest-chat-corpus --chat-file <export> --out-dir <bundle> --model <local-model>
```

Real and synthetic bundles are reported separately through the runner's `synthetic` flag.

## Security

Modules:

- `src/llb/scoring/security.py`;
- `src/llb/bench/security.py`;
- `src/llb/prep/security_sources.py`;
- `src/llb/prep/security_planter.py`.

Security cases use structured detectors instead of a safety classifier:

- `leak`: success means a marker or canary appears in output;
- `refuse`: success means the model complies with a harmful request instead of refusing.

The headline is defense rate, with ASR and refusal-appropriateness recorded alongside. Benign
controls affect refusal-appropriateness but not ASR.
The runner separates model prompting, objective scoring, gated refusal-quality judging, manifest
config assembly, judge-status assembly, and artifact persistence into named helper phases.

```bash
make bench-security MODEL=<model> BACKEND=<backend>
llb bench-security --cases samples/security_cases_uk.json --model <model>
llb adapt-security-set --source advbench --rows-file <local-export> --out <cases.json>
llb plant-security-cases --corpus-root <corpus> --out <cases.json>
```

Adapted or planted case sets remain unverified until sampled and accepted.
The goldset quickstart runs security as its own tier through `make quickstart-goldset-security`.
The 16 GiB RTX 4060 Ti validation run wrote
`.data/quickstart-leaderboard/security/20260630T112631.910536Z-c721b2c83125/manifest.json` for
`hf.co/INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-GGUF:Q4_K_M` on Ollama: ASR `1.000`,
defense rate `0.000`, refusal appropriateness `0.583`, and verified-data metadata pointing at
`samples/verification/composite_samples/security/sample_manifest.json`.

## Tooling

Modules:

- `src/llb/scoring/tooling.py`;
- `src/llb/bench/tooling.py`;
- `src/llb/bench/mcp_server.py`;
- `src/llb/prep/tooling_sources.py`.

The tooling benchmark is call-only: tools are not executed. The parser accepts native OpenAI
`tool_calls`, pre-parsed dicts, and text JSON calls. Scoring reports tool selection, argument
exactness, no-hallucinated-tool rate, well-formed-call rate, and headline call accuracy.
Case scoring is split by no-call expectations, tool selection, schema validation, argument
matching, and aggregate metric assembly.
The BFCL adapter builds the answer index, tool catalog, case instruction, and expected-call
sections independently before returning the `{tools, cases}` bundle.

```bash
llb bench-tooling --catalog samples/tooling_cases_uk.json --model <model>
llb bench-tooling --tool-protocol native --base-url <endpoint> --model <model>
llb serve-tools-mcp --catalog samples/tooling_cases_uk.json
llb adapt-bfcl --functions-file <functions> --answers-file <answers> --out <bundle.json>
```

Argument matching supports exact, contains, fuzzy, numeric tolerance, and one-of forms.

## Agentic

Modules:

- `src/llb/bench/tool_world.py`;
- `src/llb/bench/agentic.py`;
- `src/llb/bench/agentic_tasks.py`;
- `src/llb/bench/harness/`.

The agentic suite runs deterministic tasks in a sandboxed world: mock filesystem, mock key-value
DB, corpus search, and restricted arithmetic. Success is an objective assertion over final state or
answer text. The headline is completion rate; step count, tool-call count, and optional
trajectory-quality judge signals are recorded as context.
The runner separates harness resolution, episode execution, objective scoring, gated trajectory
judge attachment, and artifact persistence into named helper phases.

```bash
llb bench-agentic --tasks samples/agentic_tasks_uk.json --harness loop --model <model>
llb prepare-agentic-search --corpus-root <corpus> --out <tasks.json>
llb bench-agentic-compare --model <model>
```

Harness names are `loop`, `langgraph`, and `crewai`. Harness comparison fixes the model and task
set so orchestration is the variable.

## Summarization

`src/llb/bench/summarization.py` scores reference coverage by pinned-embedder cosine between
reference sentences and candidate summary sentences. ROUGE is not used because the product cares
about semantic coverage in Ukrainian text, not lexical overlap alone.
The runner separates summary generation, coverage scoring, optional faithfulness judging, board
assembly, and artifact persistence into named helper phases.

```bash
llb bench-summarization --cases samples/summarization_cases_uk.json --model <model>
```

Optional faithfulness uses the same gated judge side-channel as other free-form categories.

## Structured Output

`src/llb/scoring/structured.py` and `src/llb/bench/structured.py` score JSON conformance and field
accuracy. Schemas are built with Pydantic, including nested objects and arrays. Field comparison
supports case-insensitive strings, numeric tolerances, fuzzy or contains strings, and unordered
arrays.
The recursive field matcher is split by object comparison, ordered arrays, unordered arrays, and
scalar leaf matching so nested scoring remains inspectable.

```bash
llb bench-structured --cases samples/structured_cases_uk.json --model <model>
```

Non-conformant output receives zero field accuracy.

## Reliability

`src/llb/scoring/reliability.py` turns typed per-case statuses from any run bundle into reliability
and failure breakdowns.

```bash
llb bench-reliability --run-dir <run-bundle>
```

Reliability is a diagnostic axis for backend and prompt stability. It should be read beside quality,
not as a replacement for task-specific objective metrics.

## Composite Headline

`src/llb/scoring/composite_builder.py` builds a guarded category composite only when required
category tiers exist for a model, each has reloadable per-case objective scores, and every run is
stamped with verified data.

```bash
llb bench-composite
make composite-headline MODEL=<model> BACKEND=<backend>
```

Category runners accept `--data-verified` and `--verification-ref`. Verification refs can be a
decided verification worksheet, a sample manifest that points to one, or an accepted-ledger bundle.
Invalid refs produce operator diagnostics before model calls or composite publication.
