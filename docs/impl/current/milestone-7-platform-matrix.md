# Milestone 7 Platform Matrix Current State

## M7.4 16 GB CUDA Host Backend Matrix

Host:
- GPU: NVIDIA GeForce RTX 4060 Ti, 16380 MiB
- Driver: 595.71.05
- Tier detector: `gpu_tier=16`
- Post-run state: 15281 MiB free, 25.83 W draw

Requested base was "Gemma 4 27B". No exact `gemma4:27b` tag or matching vLLM/llama.cpp artifact
was available locally. The comparable common base for this host is Gemma 4 E4B IT:
- Ollama: `gemma4:e4b`
- vLLM: `google/gemma-4-E4B-it-qat-w4a16-ct`
- llama.cpp: `hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:q4_0-it`

The larger Gemma 4 12B common-base path exists for future reruns:
- vLLM: `google/gemma-4-12B-it-qat-w4a16-ct`
- GGUF backends: `hf.co/google/gemma-4-12B-it-qat-q4_0-gguf`

Run protocol:

    make build-index
    .venv/bin/python -m llb.main run-eval --model gemma4:e4b --backend ollama \
      --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl \
      --split final --limit 20 --telemetry
    .venv/bin/python -m llb.main run-eval \
      --model google/gemma-4-E4B-it-qat-w4a16-ct --backend vllm \
      --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl \
      --split final --limit 20 --telemetry --max-model-len 8192 \
      --gpu-memory-utilization 0.80 --evict
    .venv/bin/python -m llb.main run-eval \
      --model hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:q4_0-it --backend llamacpp \
      --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl \
      --split final --limit 20 --telemetry --max-model-len 8192 --gpu-layers -1

Repeatable command:

    make m7-4-platform-matrix

Override `M7_4_OLLAMA_MODEL`, `M7_4_VLLM_MODEL`, `M7_4_LLAMACPP_MODEL`,
`M7_4_MAX_MODEL_LEN`, `M7_4_GPU_MEMORY_UTILIZATION`, and `M7_4_LIMIT` for another common base.

Results:

| Backend | Model | Objective | Reliability | Tok/s | Peak VRAM MB | Mean W | Tok/W | Quality/W |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ollama | `gemma4:e4b` | 0.4203 | 0.750 | 59.27 | 10930 | 119.28 | 0.4969 | 0.2089 |
| vllm | `google/gemma-4-E4B-it-qat-w4a16-ct` | 0.4091 | 1.000 | 58.24 | 14107 | 106.43 | 0.5472 | 0.2239 |
| llamacpp | `hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:q4_0-it` | 0.4752 | 0.850 | 61.70 | 5517 | 109.24 | 0.5648 | 0.2684 |

All three cells used the same final split (`n=20`), same FAISS RAG index, same retrieval context
(`recall@5=0.900`, `MRR=0.7875`), and telemetry enabled. The judge was demoted because this run did
not pass a `JUDGE_RHO`; objective scores are the ranking signal.

Manifest refs:
- Ollama: `.data/run-eval/20260627T045419.834658Z-048bf1ae269b/manifest.json`
- vLLM: `.data/run-eval/20260627T045636.798744Z-4c0a336fd8fc/manifest.json`
- llama.cpp: `.data/run-eval/20260627T050051.621737Z-21ef2333875a/manifest.json`

## M7.4 Power Metric

`run-eval --telemetry` now samples GPU power during the fixed telemetry prompts when `nvidia-smi`
is reachable. New manifest fields:
- `telemetry.mean_power_w`
- `telemetry.peak_power_w`
- `telemetry.power_samples`
- `telemetry.tokens_per_watt`
- `metrics.mean_power_w`
- `metrics.tokens_per_watt`
- `metrics.quality_per_watt`

`quality_per_watt = objective_score * tokens_per_s / mean_power_w`. This is a quality-weighted
throughput-per-watt metric; raw `tokens_per_watt` remains available for pure serving efficiency.

## llama.cpp Binary Resolution

The llama.cpp launcher now resolves the project-managed binary at
`$DATA_DIR/llb/llamacpp/build/bin/llama-server` before falling back to `PATH`. This lets
`run-eval --backend llamacpp` work after `make build-llamacpp` without requiring a shell `PATH`
edit.

## GPU-Class Matrix Extension

The GPU-class matrix is an operator-run extension path, not a finite plan item. This host validates
the 16 GB row; each additional physical GPU host contributes its own comparable manifest row.

To prepare another GPU class on the target host:

    .venv/bin/python -m llb.main detect-gpu-vram
    .venv/bin/python -m llb.main gen-serving-config

Run a generated row on that host:

    .data/llb/serving/gpu-<tier>gb/serve_<target>.sh
    .data/llb/serving/gpu-<tier>gb/run_eval_<target>.sh

To generate configs for a target tier without running on that tier:

    .venv/bin/python -m llb.main gen-serving-config --gpu-gb 12
    .venv/bin/python -m llb.main gen-serving-config --gpu-gb 24
    .venv/bin/python -m llb.main gen-serving-config --gpu-gb 32

Generated files land under `.data/llb/serving/gpu-<tier>gb/` with `tier.json`, serve scripts, and
run-eval YAML/scripts. The same manifest fields above provide comparable score, throughput, VRAM,
load, and power metrics.

## Multi-Vector-Store Adapters (M7.4)

Chroma, Qdrant, and LanceDB adapters now sit behind the RAG-store seam; FAISS stays the default.

- Seam: `src/llb/rag/vector_index.py` -- `VectorIndex` Protocol (`search`/`save`) +
  `RAG_BACKENDS = (faiss, chroma, qdrant, lancedb)` + `build_vector_index` / `save_vector_index` /
  `load_vector_index` dispatch. `RagStore` keeps the chunk records (ids + source-span offsets) and
  only asks the index to map a query to build-order ids + cosine similarity, so `.retrieve(question,
  k)` and the gold-span metrics are unchanged across backends.
- Adapters `src/llb/rag/stores/`: shared `VectorStoreAdapter` base (build-order id shaping + uniform
  `vectors.npy` persistence; subclasses implement `_index` + `_search_row`) and `ChromaIndex` /
  `QdrantIndex` / `LanceDBIndex`. Each lazy-imports its client and converts the store's distance to a
  FAISS-comparable cosine similarity. Optional extras pin the validated client APIs:
  `[rag-chroma]` = `chromadb==1.5.9`, `[rag-qdrant]` = `qdrant-client==1.18.0`, and
  `[rag-lancedb]` = `lancedb==0.33.0`.
- Live API fixes: Chroma transient indexes use a per-instance collection name so repeated
  save/load round-trips do not collide in one process; Qdrant uses `create_collection` plus
  `query_points(...)` because client 1.18 no longer exposes `search`; LanceDB keeps
  `.metric("cosine")` and maps `_distance` back to cosine similarity.
- The chosen backend is recorded in the store meta (`backend`), so `RagStore.load` re-selects it
  without a config. Build: `build-index --vector-store faiss|chroma|qdrant|lancedb`. Compare:
  `compare-vector-stores --backends faiss,chroma,... --goldset ...` builds the SAME corpus under each
  backend (same chunking + pinned embedder) and reports recall@k / MRR by the source-span metric
  (`rag.compare.build_vector_store_comparison` + the shared `compare_retrieval`). The compare command
  accepts `--corpus-root`; when `--goldset <bundle>/goldset.jsonl` is passed and `<bundle>/corpus`
  exists, it uses that sibling corpus automatically.
- Tests: `tests/test_vector_index.py` -- dispatch, score conversion, base shaping (dependency-free),
  the missing-extra SystemExit per backend, `@slow` FAISS-seam + vector round-trip, and real
  Chroma/Qdrant round-trips when those extras are installed. LanceDB is validated through the CLI
  host path because `lancedb.connect()` can hang under pytest-specific environment variables on this
  host while working normally from the command line. The real-client round-trips stay marked
  `@pytest.mark.slow`, so `make ci` (`pytest -m "not slow"`) remains the GitHub-safe quick suite.

### Real adapter validation (2026-06-27)

Host validation used the committed `samples/goldsets/ua_squad_postedited_v1` bundle and an isolated
`DATA_DIR=.data/m7_4r_validation`. Each backend was built with:

    env DATA_DIR=.data/m7_4r_validation .venv/bin/python -m llb.main build-index \
      --corpus-root samples/goldsets/ua_squad_postedited_v1/corpus \
      --vector-store <backend>

Immediately after each build, `validate-retrieval` reloaded the persisted store and scored the
same 250-item gold set at `k=10`. All four persisted stores produced `recall@10=0.980` and
`MRR=0.847`, matching the FAISS baseline.

The one-shot comparison command:

    env DATA_DIR=.data/m7_4r_validation .venv/bin/python -m llb.main compare-vector-stores \
      --backends faiss,chroma,qdrant,lancedb \
      --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl \
      --k 10 --out .data/m7_4r_validation/llb/rag/vector_store_compare_k10.json

Result:

| Backend | recall@10 | MRR |
| --- | ---: | ---: |
| faiss | 0.980 | 0.847 |
| chroma | 0.980 | 0.847 |
| qdrant | 0.980 | 0.847 |
| lancedb | 0.980 | 0.847 |

`best_recall` is `chroma` only because the report tie-breaks identical recall/MRR by label; there is
no measured retrieval-quality difference on this corpus. No Locust scenario was needed because these
adapters run as in-process local stores, not as live HTTP services. If a future remote vector-store
mode is added, use Locust (`https://github.com/locustio/locust`) to drive concurrent
`retrieve(question, k)` calls against that service path, then compare recall/MRR separately with the
same source-span metric.

Manual procedure for a future text corpus:

1. Prepare a verified bundle with `<bundle>/goldset.jsonl` and `<bundle>/corpus/`; run the normal
   data-verification gate before treating results as headline-capable.
2. Install the needed extras:

        uv pip install --python .venv/bin/python -e ".[rag,rag-chroma,rag-qdrant,rag-lancedb]"

3. For persisted-store reload checks, build and validate one backend at a time:

        export DATA_DIR=.data/vector-store-validation
        .venv/bin/python -m llb.main build-index --corpus-root <bundle>/corpus \
          --vector-store <backend>
        .venv/bin/python -m llb.main validate-retrieval --goldset <bundle>/goldset.jsonl --k 10

   Repeat with `<backend>` = `faiss`, `chroma`, `qdrant`, and `lancedb`. This overwrites the single
   validation store between backends; use a backend-specific `DATA_DIR` if you need to keep each
   persisted store.
4. Run the comparison:

        .venv/bin/python -m llb.main compare-vector-stores \
          --backends faiss,chroma,qdrant,lancedb --goldset <bundle>/goldset.jsonl --k 10 \
          --out <report>.json

   If the gold set and corpus are not siblings, pass `--corpus-root <corpus>`.
