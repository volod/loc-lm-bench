# Guides

Operator and learner guides, grouped by topic. Every workflow guide follows the same shape so
you can engage at the depth you need:

1. **At a glance** -- a short flow of the whole workflow, with the human actions marked, so a
   first look gives the important sequence.
2. **Step-by-step commands and quality gates** -- the condensed copy-paste chain for operators
   who already know the flow.
3. **Detailed sections per step** -- the full explanation (options, gotchas, why each gate
   exists) for first-time or careful runs.

## Directory map

```text
docs/guides/
|-- learning-path/   learn the stack: RAG, serving, judging, security, eval categories
|-- quickstart/      end-to-end tracks: committed goldset, PDF corpus, any corpus
|-- data-prep/       create corpora and gold sets (local drafting or external AI services)
|-- human-tooling/   the human gates: data verification, judge calibration, and the why
|-- benchmarking/    run, compare, and analyze scored model runs
'-- development/     contributor environment setup
```

## Choose a scenario

| I want to... | Start here |
| --- | --- |
| Run the fastest model leaderboard on the committed fixture | [Quick Start](quickstart/quick-start.md), with [granular goldset commands](quickstart/quickstart-goldset-commands.md) when needed |
| Benchmark models on my own PDFs | [Quickstart: PDF corpus](quickstart/quickstart-pdf-corpus.md) |
| Benchmark models on a mixed txt/md/pdf directory | [Quickstart: any corpus](quickstart/quickstart-any-corpus.md) |
| Build a gold set end to end (create -> verify -> score) | [Create a gold set](data-prep/goldset-from-scratch.md) |
| Draft eval data with Claude / ChatGPT / NotebookLM (open data only) | [External AI service artifacts](data-prep/external-ai-service-artifacts.md) |
| Review and accept AI-drafted data (the human gate) | [Verification tooling](human-tooling/verification-tooling.md) |
| Decide whether to trust the LLM judge | [Calibration tooling](human-tooling/calibration-tooling.md) |
| Run one model through the RAG core | [Run RAG core](benchmarking/run-rag-core.md) |
| Turn a corpus into a scored RAG recommendation | [Autonomous RAG recommendation](benchmarking/auto-rag.md) |
| Estimate a local model's real-world knowledge cutoff | [Knowledge cutoff](benchmarking/knowledge-cutoff.md) |
| Compare serving backends or hardware tiers | [Platform matrix](benchmarking/platform-matrix.md), [vLLM backend](benchmarking/vllm-backend.md) |
| Compare retrieval strategies (FAISS vs GraphRAG) | [Graph vs FAISS](benchmarking/graph-vs-faiss-comparison.md) |
| Pick a chunker, embedder, or vector backend on measured evidence | [Embedder bake-off](../impl/current/rag-core.md#embedder-conventions-and-bake-off) and [vector-store commands](../impl/current/platform-vector-matrix.md#vector-store-commands) |
| Find out whether retrieval pays for itself at all | [Context ablation](../impl/current/rag-core.md#context-ablation-does-rag-pay-for-itself-rag-vs-long-context-ablation) |
| Clean a corpus of duplicated, stale, or contradictory knowledge | [Corpus hygiene](../impl/current/data-prep.md#corpus-hygiene-conflict-detection-corpus-conflict-detection) |
| Adapt a local model to my corpus (LoRA, distillation, adapters) | [Self-improvement loop](benchmarking/self-improvement-loop.md) |
| Tune and compare prompt systems | [Prompt-system lane](benchmarking/prompt-system-rag.md) |
| Analyze and compare finished runs | [MLflow analysis](benchmarking/mlflow-analysis.md) |
| Publish a guarded composite headline | [Composite headline](benchmarking/composite-headline.md) |
| Learn the concepts behind all of this | [Learning path](learning-path/learning-path.md) |
| Set up a development environment | [Dev setup](development/dev-setup.md) |

## learning-path/ -- learn the stack

- [Learning path](learning-path/learning-path.md) -- the whole stack from basics: a staged
  syllabus, curated links, and a time-boxed plan for a learner with basic knowledge.
- [LLM security learning path](learning-path/learning-path-security.md) -- threat modeling,
  jailbreak and prompt-injection testing, destructive-action controls, bias evaluation, and an
  eight-session practical syllabus.
- [Evaluation categories learning path](learning-path/learning-path-evaluation-categories.md) --
  the capabilities beyond RAG QA: long-document orchestration, structured text-analysis scoring,
  tool use, agentic workflows, summarization, serving robustness, and knowledge-graph retrieval,
  with the essential papers and a time-boxed syllabus.

## quickstart/ -- end-to-end tracks

Start with [Quick Start](quickstart/quick-start.md) for the two wrapper tracks. The remaining guides
provide granular commands and corpus-specific detail.

- [Quick Start](quickstart/quick-start.md) -- the complete committed-goldset and PDF-corpus wrapper
  workflows.
- [Goldset leaderboard: granular commands](quickstart/quickstart-goldset-commands.md) -- the
  committed-goldset leaderboard flow without the wrapper orchestration, one annotated command
  per stage.
- [Quickstart PDF corpus](quickstart/quickstart-pdf-corpus.md) -- the corpus-prep track against
  a local PDF directory: OCR, corpus-specific RAG artifacts, graph artifacts, and the
  human-verification handoff before scoring.
- [Automatic CUDA-host draft selection](../inference/config-example.md#automatic-cuda-host-draft-model-selection)
  -- deterministic GPU-tier buckets, model ranking, context eligibility, unattended behavior,
  and overrides for the PDF and mixed-corpus quickstarts.
- [PDF corpus: granular commands](quickstart/quickstart-pdf-corpus-commands.md) -- the same
  operations without the wrapper orchestration.
- [Quickstart any corpus](quickstart/quickstart-any-corpus.md) -- the same flow over a mixed
  `txt`/`md`/`pdf` directory via `ingest-corpus`, with incremental reuse and resumable drafting.

## data-prep/ -- create corpora and gold sets

- [PDF corpus prep](data-prep/pdf-corpus-prep.md) -- citation-preserving PDF conversion before
  indexing, ontology drafting, or GraphRAG.
- [Data prep](data-prep/data-prep.md) -- the create-stage commands in brief: gold set modes,
  chunking, and judge calibration entry points.
- [Create a gold set (end-to-end)](data-prep/goldset-from-scratch.md) -- the spine:
  create -> validate -> cross-check -> human verification gate -> ledger flip -> calibrate ->
  score, with authoring and review rules.
- [Test artifacts with AI provider services](data-prep/external-ai-service-artifacts.md) --
  open-data drafting of goldsets, security cases, and chains with Claude Projects / NotebookLM /
  ChatGPT Projects; restricted data stays local. Includes the workflow diagram, the command
  chain with gates, and per-step detail.
- [External-service prompt pack](data-prep/external-service-prompts/README.md) -- the
  copy-paste prompts (`00`-`04`) the external drafting workflow uses, with per-service setup.

## human-tooling/ -- the human gates

- [Human-in-the-loop evaluation](human-tooling/human-in-the-loop-evaluation.md) -- the *why*:
  the three irreducibly-human tasks (judge calibration, schema sign-off, data verification),
  with the papers and mental models.
- [Verification tooling](human-tooling/verification-tooling.md) -- operator manual for the
  `verify-sample` / `verify-review` / `verify-accept` gate: stratified sample -> per-item
  review -> accepted-ledger flip, for real-corpus and synthetic bundles.
- [Calibration tooling](human-tooling/calibration-tooling.md) -- operator manual for
  `calibration-run` / `calibration-rate` / `calibration-score`, covering the committed goldset,
  a new goldset, and a text-corpus draft.
- [Judge experiments](human-tooling/judge-experiments.md) -- DeepEval judge smoke experiments
  against local endpoints: wiring, recorded artifacts, and how they relate to the calibration
  gate.

## benchmarking/ -- run, compare, analyze

- [Autonomous RAG recommendation](benchmarking/auto-rag.md) -- resumable corpus ingestion,
  verification, joint model/config search, prompt selection, and final recommendation rendering.
- [Run RAG core](benchmarking/run-rag-core.md) -- retrieve -> generate -> score for one local
  model; the smallest complete scored run.
- [Knowledge cutoff](benchmarking/knowledge-cutoff.md) -- fit a local model's effective public-event
  knowledge horizon with reproducible Optuna and MLflow reports.
- [vLLM backend + telemetry](benchmarking/vllm-backend.md) -- install vLLM, cache weights, and
  run the eval on a real GPU backend with throughput/VRAM/power telemetry.
- [Platform matrix](benchmarking/platform-matrix.md) -- same logical model base across Ollama /
  vLLM / llama.cpp, power metrics, and GPU-class extension configs.
- [Graph vs FAISS comparison](benchmarking/graph-vs-faiss-comparison.md) -- build both retrieval
  stores and score recall/MRR head to head on one gold set.
- [Prompt-system lane](benchmarking/prompt-system-rag.md) -- generate prompt candidates, tune on
  one split, verify generalization on the held-out final split.
- [Self-improvement loop](benchmarking/self-improvement-loop.md) -- export a tuning-split dataset,
  search LoRA hyperparameters, train and register adapters, distill a student, and serve the
  result while the final split stays held out.
- [MLflow analysis](benchmarking/mlflow-analysis.md) -- select the project experiment, compare
  metrics correctly, and inspect canonical case artifacts.
- [Composite headline](benchmarking/composite-headline.md) -- close-out flow for the guarded
  category-suite composite: verify category data, preflight blockers, publish the board.
- [CrewAI harness](benchmarking/crewai-harness.md) -- validate and extend the CrewAI agentic
  harness so harness effects separate from model quality.

## development/ -- contributor setup

- [Dev setup](development/dev-setup.md) -- uv, venv, extras, apt packages, make targets, and
  troubleshooting.

## See also

- [Documentation index](../README.md) -- the docs entry point (design, implementation, guides).
- [Project README](../../README.md) -- project purpose, capabilities, and documentation routes.
- [AGENTS.md](../../AGENTS.md) -- contributor and agent guardrails.
