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
# Output/result: RAG metrics, serving configs, model-prep output, sweep runs, platform-matrix runs,
# security ASR/defense metrics, prompt candidates, and a debug log under
# $DATA_DIR/llb/logs/quickstart/.
# Note: goldset quickstart skips apt provisioning by default; set QUICKSTART_SKIP_APT=0 to include it.
# Note: QUICKSTART_SETUP_VENV=auto reuses .venv when present; set QUICKSTART_SETUP_VENV=1 to sync.
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
# Default input: committed fixture corpus; reviewer selects QUICKSTART_PROMPT_ID from the summary.
# Output/result: prompt package and prompt-system comparison runs under quickstart leaderboard data.
make quickstart-goldset-prompt
make quickstart-goldset-prompt QUICKSTART_PROMPT_ID=<prompt-id>
```

The granular commands below are the same operations without the wrapper orchestration:

```sh
# Purpose: create or update the local Python environment.
# Default input: pyproject.toml extras from EXTRAS; seeds .env from .env.example if missing.
# Output/result: .venv is ready for CLI, RAG, tracking, board, prep, and test commands.
make venv

# Purpose: isolate all quickstart leaderboard artifacts.
# Default input: none.
# Output/result: run bundles, indexes, serving configs, MLflow, and board data stay under this root.
export DATA_DIR=.data/quickstart-leaderboard

# Purpose: detect the supported CUDA host tier and generate largest-per-tier serve/eval scripts.
# Default input: samples/config-example/manifest.yaml, current nvidia-smi GPU.
# Output/result: $DATA_DIR/llb/serving/gpu-<tier>gb/ with tier.json, serve scripts, run configs.
make detect-gpu-vram
make gen-serving-config

# Purpose: chunk and embed the committed fixture corpus into the default FAISS RAG store.
# Default input: CORPUS=samples/goldsets/ua_squad_postedited_v1/corpus.
# Output/result: chunk records, vector index, and store metadata under $DATA_DIR/llb/rag/.
make build-index

# Purpose: check whether retrieval can find gold source spans before model scoring.
# Default input: GOLDSET=committed fixture, RAG_K=10, index from $DATA_DIR/llb/rag/.
# Output/result: prints n, recall@10, MRR, and PASS or retrieval-bottleneck status.
make validate-retrieval

# Purpose: resolve and prepare candidate model families for this host.
# Default input: samples/models_uk.yaml.
# Output/result: host fit table and pulled/cached runnable candidates. prep-models reuses any
# artifact already in its backend store and refuses a download up front when the cache filesystem
# lacks room for it (no failing an hour into a multi-GiB pull); --dry-run previews the disk plan.
make list-models
make prep-models

# Purpose: run one isolated evaluation cell per runnable candidate model and backend.
# Default input: samples/models_uk.yaml, GOLDSET=committed fixture, SPLIT=final.
# Output/result: run bundles in $DATA_DIR/run-eval/ plus resume markers in
# $DATA_DIR/sweep/qs-committed/cells/; qs-committed is only the user-chosen sweep name.
make sweep SWEEP_ID=qs-committed

# Purpose: compare one logical model base across Ollama, vLLM, and llama.cpp with telemetry.
# Default input: committed fixture, current platform-matrix model defaults, LIMIT=20.
# Output/result: available backend rows under $DATA_DIR/run-eval/; missing vLLM/llama.cpp
# executables are logged as skips unless PLATFORM_MATRIX_STRICT=1.
make platform-matrix

# Purpose: turn the sweep into host-adaptive operator picks + a model-comparison chart.
# Default input: $DATA_DIR/run-eval/ final-split bundles, detected CUDA tier (RECOMMEND_GPU_GB= to
# simulate another tier; RECOMMEND_MIN_CASES= to drop partial runs).
# Output/result: best RAG accuracy, best efficiency (quality/W), best model for THIS host, RAG
# health, and $DATA_DIR/recommend/{summary.md,comparison.png}.
make recommend RECOMMEND_MIN_CASES=50

# Purpose: run security tests as a separate benchmark tier; do not mix ASR with RAG quality.
# Default input: samples/security_cases_uk.json, SECURITY_MODEL=MamayLM 27B GGUF,
# SECURITY_BACKEND=ollama.
# Output/result: ASR, defense rate, refusal-appropriateness, per-family ASR, and security bundle.
make bench-security

# Purpose: prepare prompt-system candidates, review/pin one, then compare final prompt runs.
# Default input: committed fixture corpus; reviewer supplies the pinned prompt id.
# Output/result: prompt package under $DATA_DIR/prompt-system/<run>/ and prompt comparison board.
make prompt-system-prepare PROMPT_SYSTEM_CORPUS=samples/goldsets/ua_squad_postedited_v1/corpus
make prompt-system-review PROMPT_SYSTEM_RUN_DIR=<prompt-run-dir> PROMPT_SYSTEM_ACTION=pin \
  PROMPT_SYSTEM_ID=<prompt-id>
make run-eval PROMPT_SYSTEM_ID=<prompt-id> PROMPT_PACKAGE=<prompt-run-dir>
make prompt-system-compare

# Purpose: inspect canonical run bundles in the local leaderboard UI.
# Default input: $DATA_DIR/run-eval/ plus screen, category, and prompt-system artifacts.
# Output/result: Streamlit serves http://127.0.0.1:8501 until stopped.
make board

# Purpose: inspect the MLflow mirror of canonical evaluation runs.
# Default input: $DATA_DIR/run-eval/ manifests and the local $DATA_DIR/mlflow/ store.
# Output/result: syncs and serves the loc-lm-bench MLflow UI at http://127.0.0.1:5000.
make mlflow
```

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
# Default input: .data/quickstart-pdf-corpus PDFs, local Gemma 4 drafter, two-document smoke subset.
# Output/result: converted markdown, full RAG index, draft goldset, ontology, graph, validation
# metrics, and a debug log under $DATA_DIR/llb/logs/quickstart/.
make quickstart-pdf-corpus

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

The granular commands below are the same operations without the wrapper orchestration:

```sh
# Purpose: name source and generated artifact roots with a consistent hyphenated prefix.
# Default input: source PDFs in .data/quickstart-pdf-corpus.
# Output/result: markdown, RAG, graph, and draft artifacts land in sibling quickstart roots.
export PDF_SOURCE=.data/quickstart-pdf-corpus
export PDF_MD=.data/quickstart-pdf-corpus-md
export PDF_RAG_DATA=.data/quickstart-pdf-corpus-rag
export PDF_DRAFT_MD=.data/quickstart-pdf-corpus-draft-md
export PDF_DRAFT=.data/quickstart-pdf-corpus-draft
export PDF_GRAPH_DATA=.data/quickstart-pdf-corpus-graph

# Purpose: install OCR/layout extras when PDFs include scans.
# Default input: pyproject.toml pdf-quality extra and system OCR packages.
# Output/result: Docling/RapidOCR path is available for image-only PDFs.
make venv EXTRAS=pdf-quality

# Purpose: convert PDFs into markdown files with citation sidecars.
# Default input: PDF_DIR=$PDF_SOURCE, PDF_MIN_CHARS=500, PDF_PARSER=auto.
# Output/result: markdown corpus and quality reports under $PDF_MD.
make pdf-to-markdown PDF_DIR=$PDF_SOURCE PDF_OUT_DIR=$PDF_MD PDF_PARSER=auto

# Purpose: build the full vector index for the converted corpus.
# Default input: CORPUS=$PDF_MD.
# Output/result: FAISS RAG store under $PDF_RAG_DATA/llb/rag/.
env DATA_DIR=$PDF_RAG_DATA make build-index CORPUS=$PDF_MD

# Purpose: create a bounded corpus subset for fast draft review.
# Default input: selected converted markdown/citation files.
# Output/result: draft input corpus under $PDF_DRAFT_MD.
mkdir -p $PDF_DRAFT_MD
cp -R \
  $PDF_MD/pdf-d2e2499d3d06.md \
  $PDF_MD/pdf-d2e2499d3d06.citations.json \
  $PDF_MD/pdf-b117ebb25eb7.md \
  $PDF_MD/pdf-b117ebb25eb7.citations.json \
  $PDF_DRAFT_MD/

# Purpose: draft unverified gold items and ontology from the corpus.
# Default input: local Ollama Gemma 4 endpoint, DRAFT_MAX_ITEMS=8 in this example.
# Output/result: goldset.jsonl, ontology.json, extraction.jsonl, provenance, and verify_sample.csv.
make prepare-goldset-draft DRAFT_CORPUS=$PDF_DRAFT_MD DRAFT_MODEL=gemma4:e4b \
  DRAFT_MAX_ITEMS=8 DRAFT_VERIFY_N=4 DRAFT_NO_THINK=1 DRAFT_OUT_DIR=$PDF_DRAFT \
  DRAFT_TIMEOUT=600

# Purpose: build a knowledge graph from the draft bundle's ontology extraction.
# Default input: BUNDLE=$PDF_DRAFT.
# Output/result: nodes, edges, communities, and graph metadata under $PDF_GRAPH_DATA/llb/graph/.
env DATA_DIR=$PDF_GRAPH_DATA make build-graph BUNDLE=$PDF_DRAFT

# Purpose: validate draft structure and retrieval before human review.
# Default input: draft goldset/corpus and full-corpus RAG index.
# Output/result: structural PASS and recall/MRR gate output.
make validate-goldset GOLDSET=$PDF_DRAFT/goldset.jsonl CORPUS=$PDF_DRAFT/corpus
env DATA_DIR=$PDF_RAG_DATA HF_HUB_OFFLINE=1 make validate-retrieval \
  GOLDSET=$PDF_DRAFT/goldset.jsonl RAG_K=10

# Purpose: human verification gate; only accepted ledgers may be scored.
# Default input: verify_sample.csv created by the draft command.
# Output/result: $PDF_DRAFT/accepted/goldset.jsonl and corpus/ when accepted.
make verify-review VERIFY_WS=$PDF_DRAFT/verify_sample.csv
make verify-accept BUNDLE=$PDF_DRAFT VERIFY_WS=$PDF_DRAFT/verify_sample.csv
```

After `verify-accept` emits `$PDF_DRAFT/accepted/`, run `make quickstart-pdf-corpus-score` or
rerun the granular goldset scoring commands with
`GOLDSET=$PDF_DRAFT/accepted/goldset.jsonl` and `CORPUS=$PDF_DRAFT/accepted/corpus`.
For an existing text or markdown corpus, set `PDF_MD=<existing-corpus-dir>` and start at
`make build-index`; keep the same draft, graph, verification, and post-acceptance steps.

Run `make` with no target to list commands. `.env.example` documents runtime settings.

## Core Capabilities

| Capability | Functional use case | Pipeline commands |
|---|---|---|
| Corpus-grounded gold sets | Convert local PDFs to markdown, then build or ingest Ukrainian eval data with exact source spans, verified splits, and reusable corpus bundles. See [Gold-set guide](docs/guides/goldset-from-scratch.md) and [data prep](docs/guides/data-prep.md). | `make pdf-to-markdown PDF_DIR=<pdf-dir>` -> `make ingest-uk-squad` -> `make validate-goldset` |
| Human verification gates | Cross-check AI-drafted data, review a stratified sample, and emit accepted ledgers before real model scoring. See [verification tooling](docs/guides/verification-tooling.md) and [human evaluation](docs/guides/human-in-the-loop-evaluation.md). | `make verify-sample` -> `make verify-review` -> `make verify-accept` |
| FAISS and GraphRAG retrieval | Build vector and graph stores, validate recall/MRR, and compare retrieval strategies before blaming the model. See [retrieval comparison](docs/guides/graph-vs-faiss-comparison.md). | `make build-index` -> `make build-graph` -> `make validate-retrieval` -> `make compare-retrieval` |
| Local serving and model planning | Resolve which candidate models fit the host, prepare weights, and run through Ollama, vLLM, or llama.cpp. See [vLLM backend guide](docs/guides/vllm-backend.md) and [inference config](docs/inference/config-example.md). | `make list-models` -> `make prep-models` |
| Private model leaderboards | Evaluate candidates on your corpus, isolate sweep cells, tune finalists, and inspect ranked boards with CIs. Then `recommend` distills the sweep into host-adaptive picks (best accuracy, best quality/watt, best model for this GPU tier) plus a comparison chart. See [RAG core](docs/guides/run-rag-core.md) and [MLflow analysis](docs/guides/mlflow-analysis.md). | `make run-eval` -> `make sweep` -> `make pipeline` -> `make recommend` -> `make board` -> `make mlflow` |
| Calibrated judge gates | Use a local DeepEval judge only after human-rated Ukrainian calibration clears the Spearman threshold. See [calibration tooling](docs/guides/calibration-tooling.md) and [judge experiments](docs/guides/judge-experiments.md). | `make calibration-run` -> `make calibration-rate` -> `make calibration-score` -> `make judge-experiment` |
| Prompt-system tuning | Generate reviewable prompt packages, tune on one split, and verify generalization on held-out final data. See [prompt-system guide](docs/guides/prompt-system-rag.md) and [RAG core](docs/guides/run-rag-core.md). | `make prompt-system-prepare PROMPT_SYSTEM_CORPUS=<dir>` -> `make prompt-system-review PROMPT_SYSTEM_RUN_DIR=<dir> PROMPT_SYSTEM_ACTION=pin PROMPT_SYSTEM_ID=<id>` -> `make run-eval PROMPT_SYSTEM_ID=<id> PROMPT_PACKAGE=<dir>` -> `make prompt-system-compare` |
| Security robustness | Score jailbreak, prompt-injection, RAG-injection, exfiltration, and benign-control cases as a separate security tier. See [security learning path](docs/guides/learning-path-security.md). | `make bench-security MODEL=<model> BACKEND=<backend>` |
| Category benchmark suites | Score security, tooling, agentic, summarization, structured output, and text-analysis categories, then publish a guarded composite headline. See [composite headline guide](docs/guides/composite-headline.md) and [category learning path](docs/guides/learning-path-evaluation-categories.md). | `make composite-headline` |
| Agentic harness comparison | Run the same task set through loop, LangGraph, and CrewAI harnesses to separate model quality from orchestration effects. See [CrewAI harness guide](docs/guides/crewai-harness.md) and [category learning path](docs/guides/learning-path-evaluation-categories.md). | `make agentic-harness-compare` |
| Platform matrix telemetry | Compare a logical model base across serving backends with VRAM, throughput, power, and quality-per-watt telemetry. See [platform matrix guide](docs/guides/platform-matrix.md). | `make platform-matrix` |

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
[Quickstart PDF corpus](docs/guides/quickstart-pdf-corpus.md).

## Documentation

Start at the [documentation index](docs/README.md). The main implementation reference is
[current.md](docs/impl/current.md), and contributor guardrails live in [AGENTS.md](AGENTS.md).

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
