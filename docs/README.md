# loc-lm-bench documentation

Entry point for the docs. Each area lives in its own subdirectory with its own index.

## Start here

- [Project README](../README.md) -- pitch, quick start, and product surface.
- [Design](design/README.md) -- contents map into the design spec (problem, wedge,
  architecture, decisions, prior art).

## Design

- [Design contents](design/README.md) -- topic map: a gist + a jump-in link per topic.
- [Full design spec](design/spec.md) -- the complete source-of-truth document.

## Implementation

- [Current implementation](impl/current.md) -- agent-facing map of current behavior under
  `impl/current/`.
- [Forward plan](impl/plan.md) -- future engineering tasks only.
  Data-creation / verification / calibration are operator workflows, not plan items:
  [create a gold set](guides/goldset-from-scratch.md).

## Guides

- [Learning path](guides/learning-path.md) -- learn the whole stack from basics: a staged
  syllabus, curated links, and a time-boxed plan for a learner with basic knowledge.
- [LLM security learning path](guides/learning-path-security.md) -- extended threat modeling,
  jailbreak and prompt-injection testing, destructive-action controls, bias evaluation, and an
  eight-session practical syllabus.
- [Evaluation categories learning path](guides/learning-path-evaluation-categories.md) -- the
  capabilities beyond RAG QA: long-document/multi-hop orchestration, structured text-analysis
  scoring, tool use / function calling / MCP, agentic workflows, summarization / structured output
  / conversation analysis / reliability, serving robustness, and knowledge-graph retrieval -- with
  the essential papers and a time-boxed syllabus.
- [Human-in-the-loop evaluation](guides/human-in-the-loop-evaluation.md) -- the irreducibly-human
  tasks: validating LLM-as-judge against human ratings (Spearman rho + bootstrap), accountable
  schema/ontology sign-off, and stratified human sample-verification of AI-drafted eval data.
- [Dev setup](guides/dev-setup.md) -- uv, venv, extras, make targets.
- [Data prep](guides/data-prep.md) -- gold set, ingestion, chunking, calibration commands.
- [Local judge experiments](guides/judge-experiments.md) -- DeepEval, Ukrainian prompts,
  local endpoints, smoke artifacts, and calibration.
- [Calibration tooling](guides/calibration-tooling.md) -- operator manual for
  `calibration-run` / `calibration-rate` / `calibration-score`, covering the committed goldset,
  a new goldset, and a text-corpus draft.
- [Verification tooling](guides/verification-tooling.md) -- operator manual for the human
verification gate `verify-sample` / `verify-review` / `verify-accept` gate: stratified sample ->
per-item review -> accepted-ledger flip, for real-corpus and synthetic bundles.
- [Composite headline](guides/composite-headline.md) -- close-out flow for the guarded category suite
  composite: verify category data, stamp category runs, preflight blockers, and publish the board.
- [Platform matrix](guides/platform-matrix.md) -- platform matrix backend matrix, power metrics, and
  GPU-class extension commands.
- [Gold set from scratch](guides/goldset-from-scratch.md) -- published fixture, development
  imports, manual skeleton, and review rules.
- [Test artifacts with AI provider services](guides/external-ai-service-artifacts.md) -- open-data
  drafting of goldsets, security cases, and chains with Claude Projects / NotebookLM / ChatGPT
  Projects, plus the [draft contract](design/external-draft-contract.md) and
  [prompt pack](guides/external-service-prompts/README.md); restricted data stays local.
- [Run RAG core](guides/run-rag-core.md) -- build-index -> run-eval.
- [Quickstart PDF corpus](guides/quickstart-pdf-corpus.md) -- run the
  README quickstart shape against `.data/quickstart-pdf-corpus`, with OCR, corpus-specific RAG
  artifacts, graph artifacts, and the human-verification handoff before scoring.
- [Quickstart any corpus](guides/quickstart-any-corpus.md) -- the same flow over a mixed
  `txt`/`md`/`pdf` directory via `ingest-corpus`, with incremental reuse and resumable drafting.
- [Analyze runs with MLflow](guides/mlflow-analysis.md) - select the project experiment,
  compare metrics, and inspect canonical case artifacts.
- [vLLM backend + telemetry](guides/vllm-backend.md) -- backend telemetry: build-vllm -> run-eval on
  a real model.

## Inference

- [Config examples](inference/config-example.md) -- detect GPU tier, generate serve/run configs
  from [samples/config-example/](../samples/config-example/).

## Project rules

- [AGENTS.md](../AGENTS.md) -- guardrails for contributors and agents.
