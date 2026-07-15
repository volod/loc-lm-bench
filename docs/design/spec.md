# Design: Ukrainian Local-Model Selection

## Purpose

loc-lm-bench selects open-weight language models for Ukrainian RAG and text-analysis workloads on
the operator's own corpus and hardware. Its output is a reproducible internal model choice, not a
general public leaderboard.

Public benchmarks are useful candidate filters, but their tasks, language mix, quantization,
retrieval stack, and hardware differ from a local deployment. The project closes that transfer gap
by measuring the exact workload that will run in production.

## Design Intuition

Model quality is only meaningful after three upstream questions have clear answers:

1. Is the evaluation data correct and representative?
2. Can retrieval expose the required evidence?
3. Can the model and context fit on the target host under a reproducible serving configuration?

The benchmark therefore follows this trust chain:

```text
corpus -> source-span labels -> human verification -> retrieval gate
       -> host-fit serving plan -> model scoring -> immutable run bundle -> tiered board
```

A downstream score cannot repair a weak upstream link. In particular, generation models are not
blamed for evidence the retriever never supplied, and an LLM judge cannot certify data that a human
has not reviewed.

## Scope

The primary decision loop covers:

- Ukrainian corpus-grounded RAG;
- structured and narrative text analysis;
- local Ollama, vLLM, and llama.cpp serving;
- host-aware model and context feasibility;
- objective metrics, calibrated judge diagnostics, throughput, VRAM, and power;
- reproducible sweep, recommendation, and board artifacts.

Separate benchmark tiers cover security, tool use, agentic execution, summarization, structured
output, and knowledge cutoff. Their metrics remain separate because a score has meaning only within
the task and data contract that produced it.

The project does not try to be a hosted benchmark service, scheduler, model registry, or generic
agent platform.

## Architecture

```text
Typer CLI / Make workflows
          |
          +-> data prep and human gates
          |      corpus -> draft -> verify -> accepted ledger
          |
          +-> retrieval
          |      chunk -> index -> validate recall/MRR
          |
          +-> execution
          |      resolve backend -> plan memory -> run cases sequentially
          |
          +-> scoring and persistence
          |      objective metrics -> optional calibrated judge -> run bundle
          |
          `-> analysis
                 board -> recommendation -> MLflow mirror
```

Production Python lives under `src/llb/`. `src/llb/main.py` is the CLI entry point and
`src/llb/cli/` owns command registration. Make fragments group operator workflows by function.
Core typed contracts live in domain-specific modules under `src/llb/core/contracts/`; packages do
not provide facade re-exports.

## Data and Ground Truth

Each RAG gold item contains a question, reference answer, source document id, and exact character
spans. Source spans are stable across embedding and chunking changes, while chunk ids are not. A
retrieved chunk counts as evidence only when its document and character range overlap a gold span.

Corpus-derived model output is always a draft. Human verification checks grounding,
answerability, reference correctness, and planted-label integrity before `verified=true` is
written. Accepted ledgers are the only corpus-derived inputs eligible for headline scoring.

Synthetic benchmark data uses planted labels and a separate verification gate. The generating
model never certifies its own output.

## Retrieval Before Generation

Embedding and retrieval quality are evaluated independently with recall at k and mean reciprocal
rank. This isolates the evidence-delivery ceiling from generation quality. If retrieval misses the
gold span, the case is classified as a retrieval miss; when evidence is present and the answer is
wrong, it is a generation miss.

FAISS is the default vector path. GraphRAG and alternative stores share the same source-span metric,
which makes comparisons meaningful without changing gold labels.

## Backend and Hardware Boundary

The evaluator talks to an OpenAI-compatible chat interface. `BackendLauncher` implementations own
backend-specific startup, shutdown, health checks, and telemetry for Ollama, vLLM, and llama.cpp.
Evaluation and scoring code remain backend-neutral.

Before serving, the resolver combines model metadata, quantization, context length, GPU memory,
CPU offload, and backend availability into a host-fit plan. The actual served configuration is
recorded because a model name alone is not a reproducible runtime identity.

One heavyweight model runs at a time. Sequential execution avoids VRAM contention, cross-run cache
effects, and biased telemetry.

## Scoring Policy

Objective task metrics are always available. RAG scoring includes exact/contains/token overlap,
semantic diagnostics, retrieval evidence, groundedness, citation validity, and abstention probes
when configured.

An LLM judge is admitted only after its exact rubric and model clear the configured correlation
gate against human Ukrainian ratings. Below the gate, judge output remains diagnostic and cannot
change the headline. This prevents fluent but unsupported answers from outranking grounded ones.

Quality, throughput, VRAM, and power are retained as separate measurements. Recommendations may
combine them for a named operator goal, such as best accuracy or best quality per watt, but the raw
dimensions remain visible.

## Optimization Without Leakage

Configuration search uses only tuning data. Final data is held out until the selected configuration
is fixed. Sweep cells are isolated and recorded independently, so a failed or infeasible cell does
not contaminate another run.

Prompt-system and fine-tuning workflows obey the same split discipline. Registry and provenance
digests bind tuned artifacts to their source data and configuration.

## Persistence and Reproducibility

The filesystem run bundle is the source of truth. A finalized run records, as applicable:

- resolved model, backend, context, quantization, and adapter identity;
- corpus, gold-set, prompt, and configuration digests;
- per-case scores and retrieval evidence;
- aggregate metrics and reliability;
- hardware and runtime telemetry;
- reports and analysis artifacts.

Artifacts are staged and finalized atomically under `$DATA_DIR/<method>/<run_timestamp>/`. MLflow
mirrors canonical artifacts for comparison and visualization; it is not the primary store.

Boards reject incomplete, unverified, mixed-tier, or non-final records instead of guessing how to
interpret them.

## Reuse and Dependency Policy

The implementation favors maintained Python-native components and small project-owned seams:

- Typer for the CLI;
- Pydantic and typed dictionaries for contracts;
- FAISS plus optional GraphRAG/vector-store backends for retrieval;
- Optuna for bounded tuning;
- MLflow for experiment analysis;
- DeepEval for calibrated judge execution;
- NVML, `nvidia-smi`, and process telemetry for runtime evidence.

Heavy or backend-specific dependencies stay behind optional extras and lazy imports. A base install
can inspect data, plans, and artifacts without importing GPU stacks.

## Success Criteria

The system succeeds when an operator can:

- create or ingest a representative Ukrainian gold set and verify it;
- prove the retriever exposes the labeled evidence;
- identify runnable model/backend configurations for the host;
- execute comparable final-split runs without manual artifact repair;
- explain misses as retrieval, generation, refusal, artifact, or judge disagreement;
- choose a model from recorded quality and resource evidence;
- reproduce the decision from the saved inputs and manifest.

Current implementation detail is indexed in [../impl/current.md](../impl/current.md). Operator
commands and quality gates are indexed in [../guides/README.md](../guides/README.md). Forward work
belongs only in [../impl/plan.md](../impl/plan.md).
