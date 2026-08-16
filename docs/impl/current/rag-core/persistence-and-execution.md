# Backends, Persistence, And Execution

Part of the [RAG core](../rag-core.md) area of the
[current implementation index](../../current.md).

## Backends

`BackendLauncher` is the core seam:

- `OllamaLauncher` talks to a pre-existing Ollama daemon;
- `VllmLauncher` starts and stops `vllm serve`;
- `LlamaCppLauncher` starts and stops `llama-server`.

All serve through an OpenAI-compatible base URL. When a launcher owns a subprocess, startup logs are
preserved on failure.

## Persistence

`src/llb/tracking/manifest.py` writes canonical run artifacts first:

```text
$DATA_DIR/run-eval/<timestamp>-<run-id>/
  manifest.json
  scores.jsonl
  retrieval.jsonl
```

Parquet is used when `pyarrow` is available; JSONL is the portable fallback. The bundle is staged
in a hidden sibling directory and atomically renamed when canonical files are complete. MLflow
mirroring runs after canonical persistence and is best-effort.

Per-case score rows record `retrieval_hit` and `first_hit_rank`, and `prompt_tokens` whenever the
backend reported one -- the prompt the model actually consumed, which is what lets a lane comparing
two context sizes tell a served context from one silently truncated to the window. It is optional
rather than defaulted to zero, so a backend that reports no usage is distinguishable from a run
whose prompts were empty. `retrieval.jsonl` stores bounded
retrieved chunk text plus source-span coordinates for miss analysis and observability;
`src/llb/executor/cases.py` constructs both the persisted records and the in-process retrieval
pairs used by aggregate metrics and judge records.

### The Persisted Retrieval Record

Shipped (duplicate-occurrences-in-the-retrieval-record, `src/llb/rag/retrieval_records.py`): the
record is built by `retrieved_span` and read back by `record_as_chunk`, one seam for both
directions, because several lanes recompute retrieval metrics from the sidecar instead of from the
live store -- miss classification (`llb.board.miss_analysis`) and multi-span answer coverage
(`llb.eval.answer_quality.coverage`). Those recomputations have to agree with the run that wrote the
bundle, and a chunk that collapsed byte-identical copies ([duplicate chunk
collapse](retrieval-store.md#duplicate-chunk-collapse)) stands for several places at once, so the
record carries them:

- `duplicate_count` -- the TOTAL number of places the chunk's text appears, including its own.
- `duplicate_occurrences` -- the other places, each projected to `doc_id` + offsets + `chunk_id`
  (never the copy's own metadata; the store keeps that).
- Neither key is written for an uncollapsed chunk, so a corpus with no duplicates persists exactly
  the record it always did.

The list is bounded -- a converted-PDF corpus repeats one passage dozens of times and the sidecar
is written per case per hit -- and the bound is content-aware rather than blind: every occurrence
that overlaps one of the ITEM'S OWN gold spans is kept, so the recomputation stays exact, the
remaining slots (`RETRIEVED_OCCURRENCE_LIMIT = 8`) go to the first other occurrences in build
order, and `duplicate_count` always states the true total. A reader can therefore say "3 shown of
58 places" without the record growing with the corpus.

Readers: `retrieval_hit_from_record` and `read_case_coverage` both go through `record_as_chunk`,
so a gold span carried by a duplicate copy counts as retrieved in miss classification and as
covered in the multi-span columns -- previously each would have reported a miss the run did not
have. `MissRecord.retrieved_docs` (in `misses.jsonl`) lists the distinct documents the scored
context carried, first five in rank order, counting every place a collapsed chunk stands for --
which answers "did my context even come from the document I expected?" per miss.

The model PROMPT is deliberately unchanged: `format_context` still renders one `[i] (doc_id)` per
chunk. Listing every place would spend context budget on provenance the answer does not need and
would change the prompt bytes of every scored run, breaking comparability with the recorded
evidence.

Durable evidence (2026-07-23, CUDA host, pinned e5-base, goods corpus at `size=200`, the 95-item
drafted ledger, k=10; artifacts under `$DATA_DIR/duplicate-occurrences/<run>/`): 132 of 950
retrieved rows (13.9%) were collapsed chunks and 53 of 95 items had at least one in their top-10;
the largest recorded `duplicate_count` was 58 with the list capped at 8; the sidecar grew from
426.8 KB to 471.6 KB (+10.5%). Recomputed hit and coverage matched the live metric on all 95 items
-- and so did a recomputation from the occurrence-free record, because no gold span in this ledger
falls inside a repeated block. What changed on this corpus is therefore the guarantee, not the
numbers; the flip it prevents is exercised in CI.

Tests: `tests/llb/rag/test_retrieval_records.py` (the unchanged uncollapsed record, the projected
occurrences, the bound on a 58-copy chunk, the gold-completeness of that bound, and reading a
record back), plus reader-level cases in `tests/llb/board/test_miss_analysis_classification.py`,
`tests/llb/board/test_miss_probe.py` (producer to reader, end to end), and
`tests/llb/eval/test_answer_quality.py`.

## Executor

`src/llb/executor/runner.py` orchestrates one run. It filters unverified items, loads the selected
retrieval backend, executes cases, collects optional telemetry, writes artifacts, mirrors to MLflow,
and prints the row.

`run_eval(..., verified_only=False)` is the one documented exception to the unverified filter. It
exists so a diagnostic lane can score exactly the item set a drafted-grounded retrieval sweep
measured (a drafted multi-hop slice has no accepted counterpart until a reviewer produces one), and
it is deliberately hard to reach by accident: no default path sets it, `run-eval` itself has no flag
for it, and the resulting manifest records `config.item_grounding: drafted` so the bundle is
self-describing. The only caller is `compare-answer-quality --include-drafted`
([GraphRAG](../graphrag-backend/answer-quality-evidence.md#answer-quality-evidence)); a leaderboard
run never uses it.

Isolation and GPU safety live outside the scoring path:

- `src/llb/executor/vram.py`: basic reclaim checks;
- `src/llb/executor/contention.py`: pre-launch vLLM contention guard;
- `src/llb/executor/isolation.py`: process-per-cell sweep and cooldown primitive.

## Durability

`src/llb/executor/durability.py` makes a run survive endpoint flaps, a launcher-owned backend
crash, and host restarts, so a long campaign does not lose hours of model calls to one blip. Three
recovery layers wrap the per-case loop:

- **Per-case retry.** A transient transport failure -- the typed status `timeout` or
  `backend_error` -- retries with capped exponential backoff (`--max-case-retries`,
  `--retry-backoff-s`). A scored answer or any non-transport terminal status (`ok`, `empty`,
  `malformed`, `refusal`, `retrieval_miss`) is a real outcome and is never retried.
- **Journal + resume.** Each completed case appends its terminal state to an append-only
  `cases.progress.jsonl` (keyed by `item_id`) in the staging dir, beside a
  `cases.progress.meta.json` sidecar that pins the config-fingerprint and goldset digests.
  `llb run-eval --resume <run-dir>` (Make: `RESUME=<run-dir>`) reuses the journaled cases instead
  of re-spending their model calls and runs only the remainder; a resume whose config, goldset, or
  split no longer matches the sidecar is refused. Everything downstream of the raw terminal state
  (scoring, retrieval pairs, judge records) is recomputed deterministically, so a resumed run's
  per-case scores are identical to an uninterrupted one -- verified across a real two-process kill
  (`os._exit` mid-run, fresh process resumes) as well as the committed-fixture unit harness.
- **Backend relaunch.** When a case exhausts its per-case retries still in a transport failure and
  the launcher owns a serving process, the backend is relaunched through the existing
  `BackendLauncher.stop()/start()` seam a bounded number of times and the case gets another round.

A case that reaches a terminal state -- including a terminal transport failure after exhausting
retries and relaunches -- is journaled (done-as-is); only a hard kill mid-case leaves a case
un-journaled, so resume re-runs exactly that one. The atomic staged-rename stays the transaction
boundary: the journal and its sidecar are dropped from the staging dir just before finalize, so the
published bundle never carries them. On a graceful interrupt (`KeyboardInterrupt`) or an abrupt
kill the staging dir is preserved for `--resume`; on a genuine error a fresh run's staging is
cleaned up (a resume attempt keeps its staging for another try). Retry, relaunch, and resumed-case
counters are recorded in `manifest.durability`. Sweep cells inherit all of this unchanged because
each cell shells out to `run-eval` (the hidden `.`-prefixed staging dir does not collide with the
sweep's cell-directory diff).

## Sweep RAG-config grid

`llb sweep` runs one isolated cell per runnable model. The `--rag-grid top_k=3,5,8` flag (Make:
`SWEEP_RAG_GRID`, **defaulting to `top_k=3,5,8`**) expands each model into one cell per `top_k`, so
the sweep answers "which `(model, top_k)`" for THIS host, not just "which model". This is the
default because the best depth VARIES by model -- on the 16 GiB committed goldset MamayLM-12B peaks
at `top_k=3` (0.541, well above its 0.501 at `top_k=5`) while Mistral peaks at `top_k=8`, and
gridding flipped the host recommendation from Lapa to MamayLM-12B@top_k=3. Only QUERY-TIME knobs
are gridded -- they change retrieval against the SAME index, so no re-index is needed: `top_k`
(depth) and `fusion_weight` (the hybrid dense/lexical RRF share; a `fusion_weight` axis implies
`retrieval_mode=hybrid`, so build the index with `RETRIEVAL_MODE=hybrid` first). Axes are
`;`-separated and cross-multiplied (`top_k=3,5;fusion_weight=0.4,0.6` -> 4 cells per model).
Every grid knob is a `RunConfig` field and therefore part of the cell fingerprint, so each grid
point gets its own resume key (existing cells resume, not re-run), and `recommend`'s
best-per-model dedup then represents each model by its highest-scoring grid point. Index-time
knobs (`chunk_size`/`chunk_overlap`) are out of scope because they need rebuilt indexes. Set
`SWEEP_RAG_GRID=` (empty) to disable the grid and run one cell per model at the manifest's
single config.

```bash
make sweep SWEEP_ID=grid                              # default grid: 5 models x 3 top_k -> 15 cells
make sweep SWEEP_ID=one SWEEP_RAG_GRID=               # disable: one cell per model
make sweep SWEEP_RAG_GRID="top_k=3,5;fusion_weight=0.4,0.6"   # hybrid fusion grid (hybrid index)
make recommend                                        # ranks each model at its best grid point
```
