# RAG Core

The RAG core evaluates one model over a verified gold split:

```text
retrieve -> generate -> classify -> score -> aggregate -> persist
```

It is intentionally backend-neutral. Backends launch differently, but the evaluator talks to an
OpenAI-compatible chat endpoint and receives normalized response classes.

## Configuration

`src/llb/config.py` defines `RunConfig`, the typed object that flows through retrieval,
generation, scoring, telemetry, and the manifest. YAML configs and CLI overrides share the same
validation path. Unknown keys and invalid ranges fail before work starts.

`src/llb/paths.py` loads `.env`, honors `DATA_DIR`, and resolves relative paths from the project
root instead of the caller's current directory.

## Command Path

```bash
llb prep-models
llb list-models
llb build-index --vector-store faiss
llb validate-retrieval --k 10
llb run-eval --model llama3.2:3b --backend ollama
llb run-eval --config samples/run_config_uk.yaml
llb run-eval --split calibration --worksheet calibration.csv
llb run-eval --score-semantic
llb run-eval --resume .data/run-eval/<timestamp>-<run-id>   # continue an interrupted run
```

Make targets wrap the common path:

```bash
make prep-models
make build-index
make validate-retrieval
make run-eval MODEL=llama3.2:3b BACKEND=ollama LIMIT=20
```

The Makefile defaults `GOLDSET` and `CORPUS` to the committed fixture so smoke runs do not require
network access or data regeneration.
For the local PDF-corpus Gemma 4 quickstart, see
[`docs/guides/quickstart-pdf-corpus.md`](../../guides/quickstart-pdf-corpus.md). That flow builds
`.data/quickstart-pdf-corpus-rag/llb/rag/` from 19 converted PDFs: 13,211 recursive FAISS chunks
and 768-dimensional E5 embeddings (2026-07-02 build). A 4-document quick draft
(`QUICKSTART_PDF_DRAFT_DOCS=<4 ids>`, 70 unverified items, all citation-valid) scored
`recall@10=0.729`, `MRR=0.531` against that full-corpus index -- a true needle-in-haystack check;
the misses are questions phrased broadly enough that spans from OTHER doctrine documents outrank
the gold span. The PDF draft path now annotates `needle_items.jsonl` with `retrieval_rank` from the
full-corpus store; rows with a non-null rank are the retrieval-unique subset at the configured
top-k, and `pdf_ontology_report.json` records the unique-needle fraction. The matching GraphRAG
store lives under
`.data/quickstart-pdf-corpus-graph/llb/graph/` with 290 nodes, 159 edges, and 139 communities
from the same 4-document draft bundle.

## Retrieval Store

`src/llb/rag/store.py` builds `RagStore`:

- chunks the corpus through `llb.rag.chunking`;
- embeds with the pinned multilingual E5 embedder;
- stores chunk records with exact source offsets;
- persists a vector index through the vector-store seam.

The default backend is FAISS. Chroma, Qdrant, and LanceDB use the same `VectorIndex` protocol in
`src/llb/rag/vector_index.py`.

Chunk-to-source linkage (audited 2026-07-04 against the Ukrainian-RAG production checklist): every
chunk record in every strategy and both retrieval modes carries `doc_id`, a unique `chunk_id`, and
exact `char_start`/`char_end` offsets, so any chunk resolves to its verbatim place in the source
document.

Page/section provenance (`src/llb/rag/page_metadata.py`, shipped): after chunking, `RagStore.build`
joins each chunk's char span onto the `pdf-<digest>.citations.json` page-span sidecars that sit
beside the corpus docs, adding `metadata.pages = [first, last]` (source-PDF page numbers) and
`metadata.source_pdf` (the original PDF path) to every chunk whose span intersects a page, in every
strategy and both retrieval modes. The same pass fills `metadata.headers` -- the breadcrumb of
enclosing markdown headings located in the source -- for strategies other than `markdown` (which
already emits it) and for any doc with headings; plain `.md`/`.txt` docs get header breadcrumbs but
no page fields. The join is additive: chunk text, ids, and offsets are byte-identical before and
after. `store_meta.json` records `page_annotation_coverage` (the fraction of indexed chunks that
gained a `pages` field) and `build-index` logs it. In `parent_child` mode both the indexed children
and their parents are annotated, so the fields surface on retrieval hits either way. Retrieved hits
carry these fields, so verify cards, cited answers, miss clustering, and metadata filters can say
"file X, page N, section Y" without re-deriving the join. Page-aligned chunk *boundaries* and the
metadata *filter* seam are forward tasks 10 and 12 in [`plan.md`](../plan.md); governance fields
(`language`, `date`/`version`, ACL) are forward task 17.

Durable evidence (2026-07-04, heavy build on the CUDA host, outside quick CI): a `markdown`/`flat`
store over the quickstart HR PDF corpus (`.data/quickstart-pdf-corpus-hr/_md`, 8 converted docs)
annotated all 2855 indexed chunks with page provenance -- `page_annotation_coverage = 1.0` in
`store_meta.json` -- every chunk carrying `metadata.pages`, `metadata.source_pdf`, and its heading
breadcrumb.

Retrieval modes:

- `flat`: index generation chunks directly;
- `parent_child`: index smaller child chunks and return deduplicated larger parent chunks.

## Retrieval Metrics

`src/llb/rag/retrieval.py` computes recall@k and MRR by source-span overlap. The common gate is
`recall@10 >= 0.8`.

This metric is not a model-ranking axis. It answers whether the retrieval layer is able to surface
the evidence the model needs. If retrieval is poor, answer quality is capped by context quality.

All shipped stores retrieve dense-only (cosine over the pinned E5 embedding). Measured against the
gate, dense-only passes on the committed fixture (`recall@10=0.980`) but falls short on the real
full-corpus PDF index (`recall@10=0.729`, see the quickstart note above), so dense-only has NOT
been proven sufficient for a real Ukrainian corpus. Hybrid retrieval, reranking, and query
processing are forward tasks 12/13/15 in [`plan.md`](../plan.md).

## Generation Graph

`src/llb/eval/graph.py` builds the retrieve-generate flow. LangGraph is imported only when the
graph is built. The graph records one status per case: `ok`, `empty`, `malformed`, `refusal`,
`timeout`, `backend_error`, `retrieval_miss`, or another typed failure from the shared taxonomy.

`src/llb/backends/openai_client.py` normalizes endpoint failures. Backend launchers own process
lifecycle and readiness checks.

## Scoring

`src/llb/scoring/correctness.py` computes objective correctness using normalized token-F1 with
exact and contains helpers. `--score-semantic` records a pinned-embedder cosine signal for
paraphrases and morphology; it is kept separate from the objective unless a ranking policy
explicitly uses it.

`src/llb/scoring/judge.py` runs the local judge only when configured. The judge enters ranking only
when the caller supplies a calibration rho that clears the trust threshold. Otherwise it is
diagnostic and objective correctness ranks alone.

`src/llb/scoring/aggregate.py` produces leaderboard rows. The policy favors quality first, then
throughput, then lower VRAM when telemetry is available.

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
  scores.parquet
  scores.jsonl
```

Parquet is used when `pyarrow` is available; JSONL is the portable fallback. The bundle is staged
in a hidden sibling directory and atomically renamed when canonical files are complete. MLflow
mirroring runs after canonical persistence and is best-effort.

Per-case rows record `retrieval_hit` and `first_hit_rank`, but the retrieved chunk records
themselves are not persisted in the bundle -- `retrieval_pairs` stay in-process
(`src/llb/executor/cases.py`) for aggregate retrieval metrics and judge records. Adding an
additive per-case retrieved-spans record is part of forward task 6 (`miss-analysis-recommendations`
in [`plan.md`](../plan.md)), which needs it for span-overlap miss classification.

## Executor

`src/llb/executor/runner.py` orchestrates one run. It filters unverified items, loads the selected
retrieval backend, executes cases, collects optional telemetry, writes artifacts, mirrors to MLflow,
and prints the row.

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
gridding flipped the host recommendation from Lapa to MamayLM-12B@top_k=3. Only the query-time
`top_k` is gridded -- it changes retrieval depth against the SAME index, so no re-index is needed;
`top_k` is part of the cell fingerprint, so each grid point gets its own resume key (existing cells
resume, not re-run), and `recommend`'s best-per-model dedup then represents each model by its
highest-scoring `top_k`. Index-time knobs (`chunk_size`/`chunk_overlap`) are out of scope because
they need rebuilt indexes. Set `SWEEP_RAG_GRID=` (empty) to disable the grid and run one cell per
model at the manifest's single config.

```bash
make sweep SWEEP_ID=grid                              # default grid: 5 models x 3 top_k -> 15 cells
make sweep SWEEP_ID=one SWEEP_RAG_GRID=               # disable: one cell per model
make recommend                                        # ranks each model at its best top_k
```
