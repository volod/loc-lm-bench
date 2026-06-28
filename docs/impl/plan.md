# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

## Forward Tasks

### pdf-ontology-extraction-calibration

- User-visible outcome: Make long PDF-corpus ontology drafting practical enough for routine
  gold-set and GraphRAG construction on a 16 GB local CUDA host.
- Scope boundary: Reuse `llb ingest-pdf-corpus`, `llb prepare-goldset-draft`, and `llb build-graph`;
  focus on extraction prompt shape, document/window selection controls, and relation/fact yield.
  Do not replace the existing human verification gate or add a separate graph database.
- Data and artifact paths: Use `.data/_doc` as the source PDF directory and write runtime outputs
  under `$DATA_DIR/pdf-corpus/<run>/`, `$DATA_DIR/prepare-goldset/<run>/`, and `$DATA_DIR/llb/graph/`.
  Current mechanics and smoke evidence live in
  [`current/data-prep.md`](current/data-prep.md),
  [`current/robustness-ontology-backends.md`](current/robustness-ontology-backends.md), and
  [`current/graphrag-backend.md`](current/graphrag-backend.md).
- Execution path: Compare local extraction settings such as `--max-tokens`, `--temperature 0`,
  `--no-think`, and optional document/window caps before launching a full multi-hour run. Prefer
  a measurable one-window or one-document probe that reports parse rate, grounded fact count, and
  elapsed time.
- Acceptance gates: Focused tests for any parser or prompt changes, a successful full draft bundle
  with nonzero grounded facts, a graph with useful edge count, a verification worksheet for human
  review, and retrieval comparison against the vector store.
- Documentation target: Update the same current-doc topics with the final command, artifact paths,
  model choice, timing, and score comparison once the workflow is repeatable.

## Adding Future Tasks

Add a task only when there is concrete forward work with enough detail for an engineer or an
agent to execute without guessing. Use a stable descriptive id such as `platform-matrix-power`
or `prompt-system-tuning`; keep the id only while work remains under it.

Each task entry must include:

- User-visible outcome: what new capability or decision the work should create.
- Scope boundary: what is in scope, what is explicitly out of scope, and which existing modules or
  commands should be reused.
- Data and artifact paths: expected corpus, gold set, config, `$DATA_DIR/<method>/<run>/` outputs,
  and any committed `samples/` outputs.
- Execution path: commands, manual run steps, required local services, and any heavy/dependent steps
  that must stay outside quick CI.
- Acceptance gates: tests, lint/type checks, retrieval thresholds, score comparison method, or manual
  evidence required before the item leaves this file.
- Documentation target: the narrow `docs/impl/current/*.md` topic and any guide that should receive
  the resulting behavior and run notes.

When a task surfaces new future work, add that as a new forward task. Put current behavior and
durable decisions in current docs, never in this plan.
