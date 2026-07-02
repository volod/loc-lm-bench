# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

## Forward Tasks

### needle-uniqueness-filter

- User-visible outcome: drafted needle items that are RETRIEVABLE in the full-corpus haystack, not
  just span-grounded. On the 4-document quick draft, recall@10 against the 19-document index was
  0.729: the misses are questions phrased broadly enough (for example "Де застосовується
  After-Action Review?") that spans from other doctrine documents outrank the gold span. A needle
  that cannot be retrieved does not measure generation quality.
- Scope boundary: add a post-draft retrieval-uniqueness check that queries the FULL corpus index
  with each drafted question and flags (or, behind an opt-in, drops) items whose gold span is not
  in the top-k. Prefer flagging a `retrieval_rank` field on `needle_items.jsonl` rows plus a
  report metric over silent dropping, so the human review sees both. A second lever is prompt-side:
  extend `prep.ontology.draft` to require document-anchored phrasing (name the manual or context in
  the question). Reuse `llb.rag` retrieval and the existing `validate-retrieval` machinery; out of
  scope: re-ranking or changing the embedder.
- Data and artifact paths: draft bundle under `QUICKSTART_PDF_DRAFT`
  (`.data/quickstart-pdf-corpus-draft/`); full index under
  `.data/quickstart-pdf-corpus-rag/llb/rag/`; new fields land in `needle_items.jsonl` and
  `pdf_ontology_report.json`.
- Execution path: `make quickstart-pdf-corpus-draft` then `make quickstart-pdf-corpus-validate`;
  the 4-document quick-run recipe in `docs/impl/current/data-prep.md` reproduces the baseline in
  about 25 minutes.
- Acceptance gates: unit tests over a fake store for the rank annotation; on the quick-run bundle,
  the flagged subset's recall@10 is 1.0 by construction and the report shows the unique-needle
  fraction; `make ci` green.
- Documentation target: `docs/impl/current/data-prep.md` and `docs/impl/current/rag-core.md`.

### ontology-extract-parallel (optional, performance)

- User-visible outcome: a full-corpus PDF draft finishes in a fraction of the current wall time by
  running extraction windows concurrently instead of strictly sequentially.
- Scope boundary: bound concurrency inside `LLMExtractionAdapter.extract`
  (`src/llb/prep/ontology/extract.py`) with a small worker pool over windows; keep merging and
  grounding deterministic (merge in window order). Server-side parallelism relies on Ollama
  `OLLAMA_NUM_PARALLEL` slots sharing one loaded model; do not add a second model instance. Out of
  scope: parallel drafting of seeds (short calls, little to gain) and any queueing service.
- Data and artifact paths: same draft bundle under `$DATA_DIR/prepare-goldset/<timestamp>/` or
  `QUICKSTART_PDF_DRAFT`; provenance should record the concurrency setting.
- Execution path: `make prepare-goldset-draft ... DRAFT_CONCURRENCY=<n>`; verify against a
  one-document probe first (see `docs/impl/current/robustness-ontology-backends.md` for the probe
  path and the measured per-window baselines to beat).
- Acceptance gates: identical extraction results at concurrency 1 vs N on the fake-endpoint unit
  tests; a measured probe showing aggregate window throughput improves on one GPU without
  raising the per-call timeout; `make ci` green.
- Documentation target: `docs/impl/current/robustness-ontology-backends.md`.

### draft-vllm-endpoint (optional, performance)

- User-visible outcome: quickstart PDF drafting can use a vLLM-served candidate (the fastest
  ranked model on 16 GB hosts serves ~2.5x more tok/s than the best Ollama candidate), instead of
  being restricted to Ollama-served models.
- Scope boundary: reuse the existing vLLM server lifecycle in `src/llb/backends/vllm.py` to start
  the target and point `prepare-goldset-draft --base-url` at it; extend drafter selection
  (`llb.quickstart.model_choice drafter`) to accept vLLM-backed candidates once serving works. The
  blocker to solve first: reasoning-model output control -- the draft endpoint disables hidden
  thinking via Ollama-native `think=false`, and vLLM needs an equivalent (for example
  `chat_template_kwargs`) or JSON output collapses for reasoning models.
- Data and artifact paths: unchanged draft bundle layout; provenance records the endpoint base URL
  and backend.
- Execution path: `make quickstart-pdf-corpus-draft` on a CUDA host with a vLLM-ranked
  recommendation JSON present.
- Acceptance gates: a one-document probe bundle passes calibration gates with a vLLM-served
  reasoning model; drafter auto-selection covered by unit tests for both backends; `make ci`
  green.
- Documentation target: `docs/impl/current/data-prep.md` (drafter selection) and
  `docs/impl/current/robustness-ontology-backends.md` (endpoint).

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
