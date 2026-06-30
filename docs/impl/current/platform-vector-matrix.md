# Platform Matrix And Vector Stores

The platform matrix compares a logical model family across serving backends on the same host and
gold split. The vector-store matrix compares local vector backends under the same chunking,
embedding, and source-span retrieval metric.

## Backend Matrix

`make platform-matrix` runs the same logical base across Ollama, vLLM, and llama.cpp when matching
artifacts are available for the host.

```bash
make platform-matrix
```

Useful overrides:

```text
PLATFORM_MATRIX_OLLAMA_MODEL
PLATFORM_MATRIX_VLLM_MODEL
PLATFORM_MATRIX_LLAMACPP_MODEL
PLATFORM_MATRIX_MAX_MODEL_LEN
PLATFORM_MATRIX_GPU_MEMORY_UTILIZATION
PLATFORM_MATRIX_LIMIT
PLATFORM_MATRIX_BACKENDS
PLATFORM_MATRIX_STRICT
```

The matrix uses `run-eval --telemetry`, so each row records objective quality, reliability,
tokens/sec, VRAM, load time, power, tokens per watt, and quality per watt.
By default the Make target runs the requested backend rows that can actually start on the host:
vLLM requires the `vllm` executable, and llama.cpp requires either
`$DATA_DIR/llb/llamacpp/build/bin/llama-server` or `llama-server` on `PATH`. Missing optional
backend binaries are logged as skips; set `PLATFORM_MATRIX_STRICT=1` to make those skips or row
failures fail the target.

The current default common base for a 16 GB CUDA host is Gemma 4 E4B IT:

- Ollama: `gemma4:e4b`;
- vLLM: `google/gemma-4-E4B-it-qat-w4a16-ct`;
- llama.cpp: `hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:q4_0-it`.

If a requested larger base has no matching artifact for one backend, prefer an actually comparable
common base over mixing unrelated checkpoints.

Quickstart validation on the 16 GiB RTX 4060 Ti host used
`.data/quickstart-leaderboard/run-eval/20260630T053945.651376Z-5544ffad36c2/manifest.json`:
Ollama `gemma4:e4b`, 20 final cases, objective `0.420`, reliability `0.750`, `60.04` tok/s,
peak VRAM `13717` MB, `120.03` W mean power, `0.5002` tokens/W, and retrieval
`recall@5=0.900`, `mrr=0.7875`. vLLM and llama.cpp rows were skipped because their serving
executables were not installed.

## Power Metrics

When `nvidia-smi` is reachable, telemetry records:

- `telemetry.mean_power_w`;
- `telemetry.peak_power_w`;
- `telemetry.power_samples`;
- `telemetry.tokens_per_watt`;
- `metrics.mean_power_w`;
- `metrics.tokens_per_watt`;
- `metrics.quality_per_watt`.

`quality_per_watt = objective_score * tokens_per_s / mean_power_w`. Keep raw
`tokens_per_watt` for serving efficiency and `quality_per_watt` for benchmark efficiency.

## GPU-Class Configs

`detect-gpu-vram` and `gen-serving-config` generate host-specific serving scripts and run configs
under `$DATA_DIR/llb/serving/gpu-<tier>gb/`.

```bash
llb detect-gpu-vram
llb gen-serving-config
llb gen-serving-config --gpu-gb 12
llb gen-serving-config --gpu-gb 24
llb gen-serving-config --gpu-gb 32
```

The generated directory contains `tier.json`, serve scripts, and `run-eval` YAML/scripts. Primary
tier targets are MamayLM, Lapa, Gemma 4, and Qwen3.6; extra tier entries such as smaller vLLM
Gemma variants are emitted after those primary targets. This path lets another physical GPU host
contribute comparable manifest rows without hardcoding host paths.
Target ids are family-level keys; for example `gemma-4` generates `serve_gemma_4.sh` while the
tier manifest selects the concrete largest model variant that fits the host.

## llama.cpp Binary Lookup

The llama.cpp launcher first checks the project-managed binary under
`$DATA_DIR/llb/llamacpp/build/bin/llama-server`, then falls back to `PATH`. This lets
`make build-llamacpp` feed `run-eval --backend llamacpp` without requiring a shell profile edit.

## Vector-Store Seam

`src/llb/rag/vector_index.py` defines the `VectorIndex` protocol and backend dispatch:

```text
faiss
chroma
qdrant
lancedb
```

`RagStore` owns chunk records and source offsets. Vector indexes only map query embeddings to
build-order ids plus similarity. That design keeps `.retrieve(question, k)` and source-span
metrics unchanged across backends.

Adapters live under `src/llb/rag/stores/`:

- `base.py`: shared id shaping and persistence helpers;
- `chroma.py`: Chroma adapter;
- `qdrant.py`: Qdrant adapter;
- `lancedb.py`: LanceDB adapter.

Optional extras pin validated client APIs: `[rag-chroma]`, `[rag-qdrant]`, and `[rag-lancedb]`.
The default `make venv` installs the Chroma and Qdrant extras so the full local test suite
exercises their live adapter round-trips without optional-dependency skips. LanceDB remains an
opt-in adapter lane.

## Vector-Store Commands

```bash
llb build-index --corpus-root <bundle>/corpus --vector-store faiss
llb build-index --corpus-root <bundle>/corpus --vector-store chroma
llb build-index --corpus-root <bundle>/corpus --vector-store qdrant
llb build-index --corpus-root <bundle>/corpus --vector-store lancedb
llb validate-retrieval --goldset <bundle>/goldset.jsonl --k 10
llb compare-vector-stores --backends faiss,chroma,qdrant,lancedb \
  --goldset <bundle>/goldset.jsonl --k 10 --out <report>.json
```

When `--goldset <bundle>/goldset.jsonl` is passed and `<bundle>/corpus/` exists,
`compare-vector-stores` uses the sibling corpus automatically. Pass `--corpus-root` when the paths
are separate.

Use one isolated `DATA_DIR` per validation run when you need to keep persisted stores for multiple
backends.
