# Product Decisions

This page records current decisions that affect implementation shape. Keep rationale here when it
helps future agents avoid re-litigating a settled tradeoff.

## Local Judge

The benchmark uses a local OpenAI-compatible judge by default. The reason is no corpus egress and
reproducibility. The tradeoff is family bias when the judge shares architecture, tokenizer, or
pretraining lineage with candidate models.

Mitigations:

- the judge enters ranking only when calibration rho clears the trust threshold;
- objective correctness remains available and ranks alone when the judge is demoted;
- manifests disclose the judge model and bias note;
- boards reject incompatible judge cohorts.

## Data Egress

Default corpus processing is local. Frontier or Litellm calls are opt-in tools, not the default
path for private material.

Current policy:

- real chat-log corpora use local drafting or verification only;
- real text-analysis corpora may use frontier cross-check when the operator explicitly approves it;
- synthetic bundles may use the configured Litellm path;
- every drafted bundle still needs human verification before headline scoring;
- frontier *scoring* (the `scorer_policy=frontier` lane on `run-eval`) is a separate opt-in: one
  upfront egress consent plus a hard per-run USD and/or call budget enforced by the scorer cost
  ledger under `$DATA_DIR/<method>/<run>/scorer/`. Over-cap aborts are resumable and never silent.
  See [evaluation rigor](rigor-board-judge.md#scorer-policy-seam).

## Closed Graph Ontology

GraphRAG uses the closed node vocabulary in `docs/design/graph-ontology-schema.md`. The closed set
keeps graph queries, node typing, and relation caps stable across model runs. Model-invented types
are normalized to the canonical vocabulary or `MISC`.

## Backend Scope

The serving backends are Ollama, vLLM, and llama.cpp. All three must stay behind the
OpenAI-compatible launcher seam. New backend-specific behavior belongs in launcher, resolver,
planner, telemetry, or preflight modules; it should not leak into scoring logic.

CUDA source builds use `scripts/shared/common.sh:max_jobs()` for parallelism. Ordinary dependencies
use `uv` caches. `$DATA_DIR/wheels/` is only for intentional local-source wheel outputs with ABI
and git revision encoded in the directory name.

## Evaluation Tiers

Tier mixing is out of scope for a single board. Public screens, private RAG results, and each
category suite have separate metric semantics. Use side-by-side sections or explicit handoff
commands rather than one blended leaderboard.

## Agentic Framework Scope

The maintained agentic harness axis is `loop`, `langgraph`, and `crewai`. Additional frameworks
should not be added just to broaden a comparison table. Add a harness only when it changes a
meaningful operational question and can share the same task set, world, scoring, and judge gates.

## Security Guardrails Scope

loc-lm-bench measures model security behavior (`bench-security`, corpus-derived probes); it is a
benchmark, not a production RAG service. Runtime guardrails -- prompt-injection filtering of
retrieved content, output PII/secret filters, and identity-backed authorization -- are out of
scope and belong to the application embedding a recommended model. The benchmark-side governance
layer is limited to plain metadata tags, ACL-scoped retrieval, deletion propagation, stale-store
refusal, and immutable store-directory rollback; see [data prep](data-prep.md) and
[RAG core](rag-core.md). This resolves the corresponding items of the Ukrainian-RAG minimum
production checklist deliberately rather than silently.

## Public Leaderboard Scope

loc-lm-bench is a local/private benchmark. It can consume public Ukrainian benchmark results as
context, but it does not try to become a public hosted leaderboard.
