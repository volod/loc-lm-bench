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

## Corpus-Conflict Confidence

The semantic tier reports a **ranked candidate list, not a set of statistically significant
findings**, and the audit must not be presented as though it did. Threshold calibration measures
the corpus's own comparable cross-document pair similarities, a population that contains the very
duplicates it is meant to detect; enumerated exactly, the null and the observed set are identical,
so empirical FDR is 1.000 at every threshold and a budget of N returns exactly N pairs. Detail and
measurements: [data prep](data-prep.md#known-limitation-there-is-no-independent-null).

Two consequences hold until `conflict-null-model-research` says otherwise. Confidence in a corpus
conflict comes from the **claim tier's adjudication**, not from a cosine or a threshold, so no
autonomous gate should branch on the semantic tier's provisional `duplicate` verdict alone. And no
report, doc, or CLI string may describe a semantic-tier cutoff as a false-positive rate,
significance level, or confidence -- name it a candidate budget or a rank cutoff.

## Context-Ablation Lanes Stay Diagnostic

`closed_book` and `long_context` (`RunConfig.context_strategy`) are measurement lanes, never
default retrieval policies and never leaderboard rows; `rag` remains the ranked lane. This is a
decision, not an omission, and it survives the measured result that `long_context` beat `rag` on
both scored roster models
([RAG core](rag-core.md#context-ablation-evidence)):

- `long_context` is **oracle-grounded**. It reads the item's own gold `doc_id`s, so it knows the
  answer's document for free. That is a legitimate ceiling to measure a retrieval layer against
  and an illegitimate thing to ship -- a real query arrives without a gold label.
- Its gap therefore sizes what chunking still loses, not what an operator would gain. Reading
  "+0.142 objective" as "stuff whole documents instead of retrieving" would be adopting the
  oracle, not the lane.
- `closed_book` scores what the model already knows, which is a contamination and
  parametric-knowledge signal for the corpus, not a system configuration anyone would run.

The one number the ablation is entitled to change is interpretation of a leaderboard row: an
uplift that does not clear zero says the RAG stack is not earning its cost on that corpus, and a
high closed-book match rate says the item set is measuring memory as much as retrieval.

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
