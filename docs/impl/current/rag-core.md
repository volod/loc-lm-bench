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
`.data/quickstart-pdf-corpus-rag/llb/rag/` from 19 converted PDFs: 12,745 recursive FAISS chunks,
768-dimensional E5 embeddings, and a born-digital draft retrieval check of `recall@10=1.000`,
`MRR=0.732` over 7 unverified review items. The matching GraphRAG store lives under
`.data/quickstart-pdf-corpus-graph/llb/graph/` with 11 nodes, 2 edges, and 9 communities.

## Retrieval Store

`src/llb/rag/store.py` builds `RagStore`:

- chunks the corpus through `llb.rag.chunking`;
- embeds with the pinned multilingual E5 embedder;
- stores chunk records with exact source offsets;
- persists a vector index through the vector-store seam.

The default backend is FAISS. Chroma, Qdrant, and LanceDB use the same `VectorIndex` protocol in
`src/llb/rag/vector_index.py`.

Retrieval modes:

- `flat`: index generation chunks directly;
- `parent_child`: index smaller child chunks and return deduplicated larger parent chunks.

## Retrieval Metrics

`src/llb/rag/retrieval.py` computes recall@k and MRR by source-span overlap. The common gate is
`recall@10 >= 0.8`.

This metric is not a model-ranking axis. It answers whether the retrieval layer is able to surface
the evidence the model needs. If retrieval is poor, answer quality is capped by context quality.

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

## Executor

`src/llb/executor/runner.py` orchestrates one run. It filters unverified items, loads the selected
retrieval backend, executes cases, collects optional telemetry, writes artifacts, mirrors to MLflow,
and prints the row.

Isolation and GPU safety live outside the scoring path:

- `src/llb/executor/vram.py`: basic reclaim checks;
- `src/llb/executor/contention.py`: pre-launch vLLM contention guard;
- `src/llb/executor/isolation.py`: process-per-cell sweep and cooldown primitive.

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
