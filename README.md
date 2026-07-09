# loc-lm-bench -- Production Local LLM Benchmark

loc-lm-bench is a benchmark for selecting open-weight LLMs on local Ukrainian RAG
and text-analysis workloads. It evaluates candidate models on your corpus and hardware,
records the full run bundle, and ranks results with reproducible telemetry,
confidence intervals, and reviewable data gates.

## Quick Start

Requires [`uv`](https://docs.astral.sh/uv/) and a running local backend such as Ollama.
CUDA and `HF_TOKEN` are only needed for GPU serving or gated model weights.

### Goldset Leaderboard Quickstart

Use the committed, already verified UA-SQuAD fixture to build a leaderboard for the current CUDA
host. This is the fastest path for model and inference-backend comparison because it skips corpus
creation and human verification.

All quickstart wrapper targets write timestamped logs under
`$DATA_DIR/llb/logs/quickstart/`. Each log contains step headings, the exact `make` commands,
important metrics from the underlying tools, and `[result]` lines with artifact paths.

```sh
# Purpose: run the committed-goldset leaderboard flow end to end.
# Default input: committed UA-SQuAD goldset/corpus, samples/models_uk.yaml, detected CUDA tier.
# Output/result: RAG metrics, serving configs, model-prep output, sweep runs, platform-matrix
# runs, security ASR/defense metrics, prompt candidates, and a debug log under
# $DATA_DIR/llb/logs/quickstart/.
# Note: goldset quickstart skips apt provisioning by default;
#       set QUICKSTART_SKIP_APT=0 to include it.
# Note: QUICKSTART_SETUP_VENV=auto reuses .venv when present;
#       set QUICKSTART_SETUP_VENV=1 to sync.
# Note: wrapper dependency cache defaults to $DATA_DIR/uv-cache for self-contained artifacts.
make quickstart-goldset

# Purpose: run the same flow in reviewable groups for experiments and debugging.
# Default input/output/result: same as quickstart-goldset, split by pipeline stage.
make quickstart-goldset-setup
make quickstart-goldset-rag
make quickstart-goldset-models
make quickstart-goldset-eval
make quickstart-goldset-security

# Purpose: prepare prompt candidates, then pin and score one reviewed prompt id.
# Default input: committed fixture corpus; 
#                reviewer selects QUICKSTART_PROMPT_ID from the summary.
# Output/result: prompt package and prompt-system comparison runs under 
#                quickstart leaderboard data.
make quickstart-goldset-prompt
make quickstart-goldset-prompt QUICKSTART_PROMPT_ID=<prompt-id>
```

[See granular commands without the wrapper orchestration](docs/guides/quickstart/quickstart-goldset-commands.md)

The default candidate-family intent is to compare the largest runnable MamayLM, Lapa, Gemma 4,
Qwen 3.6, and Mistral variants for the detected 12/16/24/32 GiB CUDA tier. The candidate and
serving manifests now cover all five families: the Mistral default is Mistral Small 3.1 24B
(Apache-2.0, ungated), served via vLLM FP8 on the 32 GiB tier, vLLM w4a16 on the 24 GiB tier, and
the curated `mistral-small3.1:24b` GGUF on Ollama (CPU offload) on the 12/16 GiB tiers.

### PDF Corpus To Goldset And Graph Quickstart

Use this track when you start from PDFs, text, or markdown. It prepares corpus artifacts, drafts a
reviewable gold set and ontology, builds a knowledge graph, and then hands the accepted ledger to
the goldset scoring flow.

The all-in-one PDF corpus target intentionally stops before model scoring because drafted rows are
`verified=false`. Continue with the review, acceptance, and score targets only after a human review.

```sh
# Purpose: run PDF corpus prep end to end up to the verification gate.
# Default input: .data/quickstart-pdf-corpus PDFs and all converted markdown documents.
# Model selection: QUICKSTART_DRAFT_MODEL=auto uses the host-fit CUDA Gemma 4 tier target;
# override with benchmark, a pinned local model, or a frontier litellm route when needed.
# Output/result: converted markdown, full RAG index, draft goldset, ontology, graph,
# validation metrics, and a debug log under $DATA_DIR/llb/logs/quickstart/.
# approve the multi hour draft gate with QUICKSTART_ASSUME_YES=1
QUICKSTART_ASSUME_YES=1 make quickstart-pdf-corpus

# Purpose: run the same corpus flow in reviewable groups for experiments and debugging.
# Default input/output/result: same as quickstart-pdf-corpus, split by pipeline stage.
make quickstart-pdf-corpus-convert
make quickstart-pdf-corpus-index
make quickstart-pdf-corpus-draft
make quickstart-pdf-corpus-graph
make quickstart-pdf-corpus-validate

# Purpose: complete the human gate, then score only the accepted ledger.
# Default input: .data/quickstart-pdf-corpus-draft/verify_sample.csv and accepted ledger.
# Output/result: accepted verified goldset/corpus, then sweep artifacts under
# .data/quickstart-pdf-corpus-leaderboard/.
make quickstart-pdf-corpus-review
make quickstart-pdf-corpus-accept
make quickstart-pdf-corpus-score
```

Common model-selection overrides:

```sh
# Default: resolve the most capable Gemma 4 target for this CUDA tier.
QUICKSTART_MODEL_SELECTION=gemma4 make quickstart-pdf-corpus

# On the 16 GiB CUDA tier this selects Gemma 4 12B vLLM with a 16k context,
# 16 GiB CPU weight offload, and a 32 GiB CPU KV offload buffer.

# Approve the full-draft confirmation gate in the logged all-in-one wrapper.
QUICKSTART_ASSUME_YES=1 make quickstart-pdf-corpus

# Use benchmark evidence from the committed-goldset quickstart,
# then draft the full PDF corpus.
QUICKSTART_MODEL_SELECTION=benchmark make quickstart-pdf-corpus

# Pin a known local model and skip the model-selection prompt.
QUICKSTART_DRAFT_MODEL=hf.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF:Q4_K_M \
  make quickstart-pdf-corpus

# Opt into an external provider through litellm. This sends corpus text
# off-box and needs the provider API key in the environment.
QUICKSTART_DRAFT_ENDPOINT=frontier QUICKSTART_DRAFT_MODEL=<litellm-model-id> \
  make quickstart-pdf-corpus
```

[See granular commands without the wrapper orchestration](docs/guides/quickstart/quickstart-pdf-corpus-commands.md)

## PDF Corpus Prep

Use citation-preserving conversion before indexing, ontology drafting, or GraphRAG runs:

```sh
make pdf-to-markdown
make pdf-to-markdown PDF_DIR=<pdf-dir> PDF_OUT_DIR=<out-dir> PDF_MIN_CHARS=500 PDF_PARSER=auto
```

`PDF_DIR` defaults to `$DATA_DIR/quickstart-pdf-corpus`, which is
`.data/quickstart-pdf-corpus` unless `.env` overrides `DATA_DIR`. When `PDF_OUT_DIR` is omitted,
markdown files, page citation sidecars,
`pdf_corpus_manifest.json`, and `pdf_corpus_quality.json` are written to `<pdf-dir>/_md`; for
example, `.data/quickstart-pdf-corpus/_md`. `PDF_PARSER=auto` uses PyMuPDF4LLM for born-digital
PDFs and Docling OCR for image-only scans when the `pdf-quality` extra and OCR apt packages are
installed.
For a full `.data/quickstart-pdf-corpus` example with Gemma 4 and corpus-specific artifact paths,
see
[Quickstart PDF corpus](docs/guides/quickstart/quickstart-pdf-corpus.md).

## Documentation

Start at the [documentation index](docs/README.md). For task-oriented workflows -- benchmark my
PDFs, build a gold set, verify drafted data, compare backends -- use the
[guides index](docs/guides/README.md) and its
["Choose a scenario"](docs/guides/README.md#choose-a-scenario) table. The main implementation
reference is [current.md](docs/impl/current.md), and contributor guardrails live in
[AGENTS.md](AGENTS.md).

## Core Capabilities

| Capability | Functional use case | Pipeline commands |
|---|---|---|
| Corpus-grounded gold sets | Convert local PDFs to markdown, then build or ingest Ukrainian eval data with exact source spans, verified splits, and reusable corpus bundles. See [Gold-set guide](docs/guides/data-prep/goldset-from-scratch.md) and [data prep](docs/guides/data-prep/data-prep.md). | `make pdf-to-markdown PDF_DIR=<pdf-dir>` -> `make ingest-uk-squad` -> `make validate-goldset` |
| Human verification gates | Cross-check AI-drafted data, review a stratified sample, and emit accepted ledgers before real model scoring. See [verification tooling](docs/guides/human-tooling/verification-tooling.md) and [human evaluation](docs/guides/human-tooling/human-in-the-loop-evaluation.md). | `make verify-sample` -> `make verify-review` -> `make verify-accept` |
| FAISS and GraphRAG retrieval | Build vector and graph stores, validate recall/MRR, and compare retrieval strategies before blaming the model. See [retrieval comparison](docs/guides/benchmarking/graph-vs-faiss-comparison.md). | `make build-index` -> `make build-graph` -> `make validate-retrieval` -> `make compare-retrieval` |
| Ukrainian query-side processing | Improve Ukrainian queries before retrieval without touching the corpus: casefold/apostrophe/transliteration normalization, corpus-vocabulary typo tolerance, alias/glossary expansion, and an opt-in logged LLM rewrite -- with an A/B report proving each step's recall/MRR delta. See [RAG core](docs/impl/current/rag-core.md) query-side processing. | `make build-query-glossary BUNDLE=<draft>` -> `make validate-retrieval QUERY_PREP=normalize,typos,glossary QUERY_PREP_AB=1` -> `make run-eval QUERY_PREP=normalize,typos,glossary` |
| Local serving and model planning | Resolve which candidate models fit the host, prepare weights, and run through Ollama, vLLM, or llama.cpp. See [vLLM backend guide](docs/guides/benchmarking/vllm-backend.md) and [inference config](docs/inference/config-example.md). | `make list-models` -> `make prep-models` |
| Private model leaderboards | Evaluate candidates on your corpus, isolate sweep cells, tune finalists, and inspect ranked boards with CIs. Then `recommend` distills the sweep into host-adaptive picks (best accuracy, best quality/watt, best model for this GPU tier) plus a comparison chart. See [RAG core](docs/guides/benchmarking/run-rag-core.md) and [MLflow analysis](docs/guides/benchmarking/mlflow-analysis.md). | `make run-eval` -> `make sweep` -> `make pipeline` -> `make recommend` -> `make board` -> `make mlflow` |
| Calibrated judge gates | Use a local DeepEval judge only after human-rated Ukrainian calibration clears the Spearman threshold. See [calibration tooling](docs/guides/human-tooling/calibration-tooling.md) and [judge experiments](docs/guides/human-tooling/judge-experiments.md). | `make calibration-run` -> `make calibration-rate` -> `make calibration-score` -> `make judge-experiment` |
| Prompt-system tuning | Generate reviewable prompt packages, tune on one split, and verify generalization on held-out final data. See [prompt-system guide](docs/guides/benchmarking/prompt-system-rag.md) and [RAG core](docs/guides/benchmarking/run-rag-core.md). | `make prompt-system-prepare PROMPT_SYSTEM_CORPUS=<dir>` -> `make prompt-system-review PROMPT_SYSTEM_RUN_DIR=<dir> PROMPT_SYSTEM_ACTION=pin PROMPT_SYSTEM_ID=<id>` -> `make run-eval PROMPT_SYSTEM_ID=<id> PROMPT_PACKAGE=<dir>` -> `make prompt-system-compare` |
| Security robustness | Score jailbreak, prompt-injection, RAG-injection, exfiltration, and benign-control cases as a separate security tier. See [security learning path](docs/guides/learning-path/learning-path-security.md). | `make bench-security MODEL=<model> BACKEND=<backend>` |
| Category benchmark suites | Score security, tooling, agentic, summarization, structured output, and text-analysis categories, then publish a guarded composite headline. See [composite headline guide](docs/guides/benchmarking/composite-headline.md) and [category learning path](docs/guides/learning-path/learning-path-evaluation-categories.md). | `make composite-headline` |
| Agentic harness comparison | Run the same task set through loop, LangGraph, and CrewAI harnesses to separate model quality from orchestration effects. See [CrewAI harness guide](docs/guides/benchmarking/crewai-harness.md) and [category learning path](docs/guides/learning-path/learning-path-evaluation-categories.md). | `make agentic-harness-compare` |
| Platform matrix telemetry | Compare a logical model base across serving backends with VRAM, throughput, power, and quality-per-watt telemetry. See [platform matrix guide](docs/guides/benchmarking/platform-matrix.md). | `make platform-matrix` |

## Model Families and Licenses

The default candidate sweep compares five open-weight families -- two Ukrainian-specialized and
three multilingual baselines. Each links to its upstream weights; comply with the listed license
when serving or redistributing.

| Family | Focus | Default weights | License |
| --- | --- | --- | --- |
| MamayLM v2 (INSAIT) | Ukrainian-specialized | [MamayLM v2.0 (Gemma 3) collection][mamay-col] | [Gemma Terms][gemma-lic] |
| Lapa v0.1.2 (lang-uk) | Ukrainian-specialized | [lapa-llm/lapa-v0.1.2-instruct][lapa-repo] | [Gemma Terms][gemma-lic] |
| Gemma 4 (Google) | Multilingual baseline | [google/gemma-4 collection][gemma-col] | [Gemma Terms][gemma-lic] |
| Qwen 3.6 (Alibaba) | Multilingual baseline | [Qwen/Qwen3.6-35B-A3B][qwen-repo] | [Apache 2.0][apache-lic] |
| Mistral Small 3.1 (Mistral AI) | Multilingual baseline | [mistralai/Mistral-Small-3.1-24B-Instruct-2503][mistral-repo] | [Apache 2.0][apache-lic] |

The Ukrainian families build on the prior art tracked by the
[lang-uk leaderboard](https://github.com/lang-uk) and the
[MamayLM project](https://models.mamay.ai/). Per-tier concrete variants (GGUF / w4a16 / FP8) and
serving knobs live in [docs/inference/config-example.md](docs/inference/config-example.md) and
[samples/models_uk.yaml](samples/models_uk.yaml).

[mamay-col]: https://huggingface.co/collections/INSAIT-Institute/mamaylm-v20-gemma-3
[lapa-repo]: https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct
[gemma-col]: https://huggingface.co/collections/google/gemma-4
[qwen-repo]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
[mistral-repo]: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503
[gemma-lic]: https://ai.google.dev/gemma/terms
[apache-lic]: https://www.apache.org/licenses/LICENSE-2.0

## Data Licenses

Ready-to-use public fixtures and public-screen tasks keep their upstream data terms:

- The committed UA-SQuAD fixture derives from
  [`FIdo-AI/ua-squad`](https://huggingface.co/datasets/FIdo-AI/ua-squad). Its dataset-card
  metadata is MIT-marked, and the fixture applies the upstream derivative-text note that
  SQuAD-derived text inherits [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  See the local [fixture license](samples/goldsets/ua_squad_postedited_v1/LICENSE.md),
  [source metadata](samples/goldsets/ua_squad_postedited_v1/source.json), and
  [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) attribution.
- The Tier-1 public screen does not vendor task records. Its default task sources should be
  checked before publishing or redistributing data:
  [Belebele](https://huggingface.co/datasets/facebook/belebele) and
  [ARC](https://huggingface.co/datasets/allenai/ai2_arc) are CC BY-SA 4.0;
  [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) and
  [MMLU](https://huggingface.co/datasets/cais/mmlu) are MIT;
  [PIQA](https://huggingface.co/datasets/piqa) is marked license-unknown on its dataset card.

Other committed tutorial fixtures are repo-authored unless their local README or provenance file
states otherwise. Preserve attribution and license notices when redistributing derived artifacts.
