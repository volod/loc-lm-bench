# Drafting Lanes And Resumable Extraction

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

## Resumable extraction (interrupt-safe drafting)

Because a full-corpus draft is a multi-hour extraction stage, the bundle carries a per-document,
per-window extraction journal (`src/llb/prep/ontology/coverage/journal.py`). Each completed window appends
one line to `extraction_journal.jsonl` (keyed by `(doc_id, window_index)`, deterministic from
`split_document`); a settings sidecar `extraction_journal.meta.json` is written at the start of
extraction and pins the determinism-critical settings (corpus, seed, `max_items`, window size and
overlap, retrieval options) plus the endpoint identity.

`llb prepare-goldset-draft --resume <bundle>` (make: `DRAFT_RESUME=<bundle>`;
`make quickstart-corpus QUICKSTART_CORPUS_RESUME=<bundle>`) re-enters an interrupted bundle: it
reads the meta, reuses journaled windows instead of re-calling the model, re-extracts only the
missing windows, and replays the deterministic seed/draft/emit stages. The result is byte-identical
to an uninterrupted run (same seeds, same kept items). Only parsed JSON objects are journaled;
failed calls remain resumable, while parsed empty objects carry the required `parsed=true` marker.
A row without that marker is malformed and is regenerated. A missing meta aborts the resume with a clear
message. The make target also rejects an explicitly empty command-line `DRAFT_RESUME`, preventing
an unset shell variable from silently starting a fresh default draft.

Resume option restoration lives in `src/llb/cli/prep/draft_resume.py`.
`DraftResumeBuilder` starts from the parsed `DraftRequest`, applies the journal's authoritative
corpus and phase routes, preserves explicit CLI overrides, and returns one resolved request. The
execution path consumes that object directly, so request validation has one entry point.

## Frontier ontology draft lane

The ontology pipeline has explicit extraction and drafting endpoint routes. Immutable route data
and phase telemetry live in `src/llb/prep/ontology/endpoints/config.py`; validation and construction
live in `endpoints/builder.py`; local and Litellm transports live in `endpoint.py`; spend/call
enforcement lives in `src/llb/prep/frontier/telemetry.py`.
`prepare-goldset-draft` keeps both phases local by default. A frontier run requires all of:

- `DRAFT_ENDPOINT=frontier` (or `--endpoint frontier`);
- an interactive confirmation that names the corpus path and Litellm destination;
- at least one guard (`DRAFT_MAX_USD` or `DRAFT_MAX_CALLS`); the CLI supplies a 100-call guard
  when neither is given explicitly.

The default frontier route covers both phases. `DRAFT_FRONTIER_STAGE=extraction` or `drafting`
routes only that phase off-box and requires `DRAFT_LOCAL_MODEL` for the other phase. Example:

```bash
make prepare-goldset-draft \
  DRAFT_CORPUS=<corpus-dir> \
  DRAFT_ENDPOINT=frontier \
  DRAFT_FRONTIER_MODEL=<litellm-model-id> \
  DRAFT_FRONTIER_STAGE=both \
  DRAFT_MAX_USD=<usd-cap> \
  DRAFT_MAX_CALLS=<call-cap>
```

`provenance.json` records the configured extraction/drafting routes plus aggregate and per-phase
call counts, measured cost, latency, and per-call telemetry. A call cap is checked before dispatch.
Provider cost is only knowable after a response, so the spend guard stops immediately after the
call that crosses the cap. A budget stop exits nonzero but leaves `provenance.json` with
`status: aborted`, the reason, telemetry, and the extraction journal so the bundle remains
inspectable and resumable.

`make draft-compare` (`src/llb/prep/ontology/compare/run.py`) runs local extraction once, selects a
bounded deterministic seed set, and drafts those exact seed objects through the local and frontier
routes. It writes self-contained lane bundles, verification worksheets, and
`$DATA_DIR/draft-compare/<timestamp>/comparison.json`. The report includes seed fingerprints,
parse rate, kept yield, calibration gates, and separate rankings for kept yield and verify-sample
accept rate. Accept rate is `pending-human-review` until reviewed worksheets are supplied with
`make draft-compare-report`; that report-only target updates `comparison.json` without calling or
spending on either model again.

The bounded `frontier-ua-draft-lane` probe reuses the committed, repo-authored synthetic
two-document corpus at `samples/text_analysis_bundle_uk/corpus`. Its adjacent `provenance.json`
records the data classification and document hashes. `make frontier-ua-draft-probe` pins this
input, requires an explicit Litellm model, USD cap, call cap, and output root, then presents the
normal corpus-and-destination egress prompt before any provider call.

Human comparison decisions use the same resumable terminal cards as gold-set verification:
`make draft-compare-review DRAFT_COMPARE_OUT_DIR=<comparison-root>` walks the local worksheet and
then the frontier worksheet, saves every decision atomically, and resumes at the first undecided
row. Cross-check/model verdicts remain hidden. `make draft-compare-finalize` refreshes reviewed
accept rates without model calls, records a `finalization` block in `comparison.json`, prints one
`[ok]` or `[fail]` line per acceptance gate, and exits nonzero unless all worksheets, calibration,
parse-rate, ranking, call-cap, and spend-cap checks pass. The lower-level
`make draft-compare-report` remains useful when worksheets live outside their generated paths.

```bash
make draft-compare \
  DRAFT_COMPARE_CORPUS=<corpus-dir> \
  DRAFT_COMPARE_SEEDS=<n> \
  DRAFT_COMPARE_LOCAL_MODEL=<local-model> \
  DRAFT_COMPARE_FRONTIER_MODEL=<litellm-model-id> \
  DRAFT_COMPARE_MAX_USD=<usd-cap>

make draft-compare-report \
  DRAFT_COMPARE_OUT_DIR=<comparison-root> \
  DRAFT_COMPARE_LOCAL_VERIFICATION=<reviewed-local-csv> \
  DRAFT_COMPARE_FRONTIER_VERIFICATION=<reviewed-frontier-csv>

make draft-compare-review DRAFT_COMPARE_OUT_DIR=<comparison-root>
make draft-compare-finalize DRAFT_COMPARE_OUT_DIR=<comparison-root>
```

Deterministic coverage is in
`tests/llb/prep/ontology/test_frontier_lane.py`: refusal occurs before completer construction,
fake-provider spend aborts preserve provenance, mixed phase routes use distinct callables, and the
comparison report fingerprints the shared seeds. No test needs network access or a provider key.

## Sequential local Qwen/Gemma draft comparison

`make local-ua-draft-probe` runs an exact-seed local comparison without data egress. The adaptive
selection policy in `src/llb/prep/ontology/compare/local_models.py` uses the benchmark GPU-tier
detector and selects Qwen/Gemma Ollama pairs for 12, 16, 24, and 32 GiB hosts. Explicit model flags
remain available, but the selected tags must already be installed. The resolved tier, GPU name,
models, context, and override/profile source are recorded under
`comparison.json.execution.resource_selection`.

`src/llb/prep/ontology/compare/local.py` enforces sequential residency: unload all Ollama models,
run the Qwen baseline extraction and drafts, unload Qwen and wait until absent, run Gemma over the
same seed objects, then unload again. `ollama_lifecycle.py` turns unload failure into an error rather
than permitting overlapping model residency. The comparison uses `baseline` and `probe` lane names;
it does not reuse the frontier names or provenance.

The reviewed 16 GiB comparison selected `qwen3:14b` then `gemma4:e4b` with an 8192-token
context. Both parsed 12/12 drafts and passed calibration. Qwen kept 7/12 items (58.3%) in 71.9
seconds of drafting; Gemma kept 5/12 (41.7%) in 31.1 seconds. Gemma matched parse rate, reduced
kept yield by 16.7 percentage points, and was about 2.3 times faster in drafting latency. Human
review accepted every kept item: 7/7 Qwen items and 5/5 Gemma items, both 100%. Finalization passed
all worksheet, calibration, ranking, sequential-execution, and unload checks. This supports Qwen as
the higher-yield baseline and Gemma as the faster probe on this small fixture; reviewed accept rate
does not distinguish them. `ollama ps` was empty after the run, confirming final unload. Runtime
artifacts follow `<comparison-root>/{comparison.json,baseline/,probe/}`.

The 12 GiB profile now uses the same evidence-grade `qwen3:14b` / `gemma4:e4b` pair. Its former
E2B probe was below the project's >=7B live-evidence floor and was not installed on the Blackwell
host, while both larger local artifacts fit sequentially. The override-free 2026-07-28 run on the
12,227 MiB RTX PRO 3000 Blackwell parsed 12/12 drafts in both lanes and passed both calibration
gates. Qwen kept 8/12 (66.7%) with 52.134 seconds of draft-call latency; Gemma kept 6/12 (50.0%)
with 21.326 seconds. The 16.7-point yield advantage and Gemma latency advantage have the same
direction as the reviewed 16 GiB result. This new bundle is not human-reviewed, so it supports host
fit, parsing, calibration, and sequential unload only; it does not add a human accept-rate claim.
Artifact: `$DATA_DIR/draft-compare-local/20260728T095500Z-blackwell12-default/`. The selector
regression is in `tests/llb/prep/ontology/test_local_compare.py`, and the operator table is in
`docs/guides/data-prep/goldset-from-scratch.md`.

`make draft-compare-analyze DRAFT_COMPARE_OUT_DIR=<comparison-root>` reads `comparison.json`, both
lane provenance files, and the live worksheets. It prints model order, shared-seed counts, parsed
and kept ratios, calibration, calls, latency, review progress, human accept rates, and lane deltas.
`COMPARE_ANALYZE_JSON=1` emits normalized JSON and `COMPARE_REQUIRE_GATES=1` exits nonzero when a
calibration gate fails. `make draft-compare-review` and `make draft-compare-finalize` accept either
the frontier lane schema or the local `baseline`/`probe` schema.

The dedicated operator aliases keep this workflow on one variable:

```bash
make local-ua-draft-probe LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>
make local-ua-draft-complete LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root>
make local-ua-draft-analyze LOCAL_DRAFT_COMPARE_OUT_DIR=<comparison-root> \
  COMPARE_ANALYZE_JSON=1
```

`local-ua-draft-complete` runs the resumable human review, finalization, and final table in order.
The individual `local-ua-draft-review` and `local-ua-draft-finalize` aliases remain available for
operators who prefer one explicit gate at a time.
