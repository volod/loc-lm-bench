# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

## Forward Tasks

### pdf-ontology-extraction-calibration

- User-visible outcome: Make long PDF-corpus ontology drafting practical enough for routine
  gold-set, needle-in-the-stack, prompt-dictionary, ontology, GraphRAG, and RAG construction on a
  local CUDA host.
- Scope boundary: Reuse the prepared `$DATA_DIR/_doc/_md` PDF corpus, `llb prepare-goldset-draft`,
  `llb build-graph`, and the existing human verification gate. Do not add a separate graph database
  or replace the verification workflow.
- Data and artifact paths: PDF preparation mechanics and parser evidence live in
  [`current/data-prep.md`](current/data-prep.md). Use that corpus root for ontology and graph runs.
  Write draft outputs under `$DATA_DIR/prepare-goldset/<run>/`, graph artifacts under
  `$DATA_DIR/llb/graph/`, and prompt dictionary candidates under the matching run root.
- Execution path: Run a bounded one-window or one-document ontology probe over the prepared PDF
  corpus. Compare local model settings such as `--max-tokens`, `--temperature 0`, `--no-think`, and
  optional document/window caps before launching a full multi-hour draft. Report parse rate,
  page-span citation coverage, grounded fact count, dictionary-term yield, and elapsed time.
- Goldset path: Generate needle-in-the-stack items whose answers validate against PDF citation
  sidecars, then produce a verification worksheet for human review. Keep accepted items in the
  existing reviewed-goldset bundle shape.
- Graph path: Build a graph from the extracted ontology candidates, report useful node/edge counts,
  and compare retrieval against the vector store before using graph context in model scoring.
- Optional parser probe: If manual review finds OCR/layout quality gaps, run explicit
  `PDF_PARSER=marker`, `PDF_PARSER=unstructured`, or `PDF_PARSER=markitdown` probes on the affected
  PDFs and compare citation coverage and text quality before changing the default parser policy.
- Acceptance gates: A full draft bundle with nonzero grounded facts; useful graph edge count; human
  verification worksheet; citation-valid needle items; source-backed prompt dictionary candidates;
  retrieval comparison against the vector store; and `make test` plus `make lint-md`.
- Documentation target: Refresh [`current/data-prep.md`](current/data-prep.md),
  [`current/robustness-ontology-backends.md`](current/robustness-ontology-backends.md), and
  [`current/graphrag-backend.md`](current/graphrag-backend.md) with model choice, timing,
  page-citation coverage, graph counts, and score comparison once the workflow is repeatable.

### lancedb-vector-store-validation

- User-visible outcome: Promote LanceDB from an opt-in adapter lane to a default-validated vector
  store backend for local platform comparisons.
- Scope boundary: Reuse `src/llb/rag/stores/lancedb.py`, `llb build-index --vector-store lancedb`,
  and `llb compare-vector-stores`; do not change the `VectorIndex` protocol or the FAISS default.
- Data and artifact paths: Use the committed sample gold set and corpus for smoke validation, with
  any diagnostic runtime files under `$DATA_DIR/llb/rag/` or a temporary test directory.
- Execution path: Isolate the live LanceDB build/search/save/load round-trip, identify any blocking
  client call or teardown behavior, add a bounded regression test, and then decide whether
  `[rag-lancedb]` belongs in default `make venv`.
- Acceptance gates: Full `make test` exits with zero skips and no timeout when LanceDB live
  coverage is enabled, plus `make lint-md`.
- Documentation target: Refresh [`current/platform-vector-matrix.md`](current/platform-vector-matrix.md)
  and [`../guides/dev-setup.md`](../guides/dev-setup.md) with the final default-extra decision.

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
