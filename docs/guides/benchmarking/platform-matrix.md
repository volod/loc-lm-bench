# Platform Matrix

Use this guide to run platform comparisons: same logical model base across backend families,
power-aware telemetry, and GPU-class extension configs.

## At a glance

    1. (optional) install extra backends   make build-vllm; make build-llamacpp
    2. run the matrix                      make platform-matrix     [one row per backend]
    3. read the rows                       objective score, tok/s, peak VRAM, power, quality/watt

The comparability rule is the gate here: one split, one limit, one RAG index, one telemetry
protocol for every backend row. Rows produced with different gold sets, retrieval configs, or
context caps must not be compared. Missing vLLM/llama.cpp binaries are logged as skips unless
`PLATFORM_MATRIX_STRICT=1`.

## Common-Base Backend Run

Default common base for the 16 GB CUDA host:
- Ollama: `gemma4:e4b`
- vLLM: `google/gemma-4-E4B-it-qat-w4a16-ct`
- llama.cpp: `hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:q4_0-it`

Run the full chain:

    make platform-matrix

The target does:
1. rebuild the committed FAISS RAG index;
2. run each requested backend row with telemetry when its serving binary is available;
3. skip missing vLLM or llama.cpp executables with an explicit log line;
4. fail only when no backend row succeeds, unless `PLATFORM_MATRIX_STRICT=1` is set.

Install optional serving backends before requiring a full three-row comparison:

    make build-vllm
    make build-llamacpp

Override the defaults for another common base:

    make platform-matrix \
      PLATFORM_MATRIX_OLLAMA_MODEL=<ollama-tag-or-hf.co-gguf> \
      PLATFORM_MATRIX_VLLM_MODEL=<hf-vllm-checkpoint> \
      PLATFORM_MATRIX_LLAMACPP_MODEL=<hf.co-gguf-source> \
      PLATFORM_MATRIX_MAX_MODEL_LEN=8192 \
      PLATFORM_MATRIX_LIMIT=20

Limit or harden the matrix:

    make platform-matrix PLATFORM_MATRIX_BACKENDS="ollama"
    make platform-matrix PLATFORM_MATRIX_STRICT=1

For a larger Gemma 4 common base, use the 12B artifacts:

    make platform-matrix \
      PLATFORM_MATRIX_VLLM_MODEL=google/gemma-4-12B-it-qat-w4a16-ct \
      PLATFORM_MATRIX_OLLAMA_MODEL=hf.co/google/gemma-4-12B-it-qat-q4_0-gguf \
      PLATFORM_MATRIX_LLAMACPP_MODEL=hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:q4_0

Use one split, one limit, one RAG index, and one telemetry protocol for every backend row. Do not
compare rows produced with different gold sets, splits, limits, retrieval configs, or context caps.

## Power Metrics

`run-eval --telemetry` records power when `nvidia-smi` is reachable:
- `telemetry.mean_power_w`
- `telemetry.peak_power_w`
- `telemetry.tokens_per_watt`
- `metrics.quality_per_watt`

`quality_per_watt = objective_score * tokens_per_s / mean_power_w`.

## Estimating Run Time

Do not size a sweep or matrix run from model parameter count. Decode speed is set by architecture,
not nominal size: a mixture-of-experts model (e.g. `qwen3.6-35b-a3b`, ~3B active of 35B) can be
faster than a dense 12B, attention layout (GQA/MQA vs full MHA) and quantization move tok/s
severalfold, and on a 16 GiB card VRAM-fit vs CPU-offload is usually the biggest factor. Use the
measured `tokens_per_s` from a prior manifest and `load_time + n_cases * out_tokens / tokens_per_s`;
`list-models` shows the `gpu/total` layer split (an `offload` verdict predicts slow decode). See
[Backend Telemetry -> Run-Time Estimation](../../impl/current/backend-telemetry.md#run-time-estimation)
and the [LLM architecture gallery](https://sebastianraschka.com/llm-architecture-gallery/).

## GPU-Class Extension

The GPU-class matrix is an operator-run extension path, not a finite plan item. Each physical host
adds one comparable row by generating tier-specific serving configs and running the generated
scripts on that host.

On the target host:

    .venv/bin/python -m llb.main detect-gpu-vram
    .venv/bin/python -m llb.main gen-serving-config

Generated artifacts land under `.data/llb/serving/gpu-<tier>gb/`:
- `tier.json`
- `serve_*.sh`
- `run_eval_*.yaml`
- `run_eval_*.sh`

Run a generated row:

    .data/llb/serving/gpu-<tier>gb/serve_<target>.sh
    .data/llb/serving/gpu-<tier>gb/run_eval_<target>.sh

Use the same target name, split, limit, retrieval settings, and context policy when comparing rows
across hosts. The manifest from each `run_eval_*.sh` is the comparison record.

To prepare configs for a tier without running on that tier:

    .venv/bin/python -m llb.main gen-serving-config --gpu-gb 12
    .venv/bin/python -m llb.main gen-serving-config --gpu-gb 24
    .venv/bin/python -m llb.main gen-serving-config --gpu-gb 32

Run generated scripts only on the matching physical GPU class. Compare resulting manifests by
objective score, reliability, tok/s, peak VRAM, load time, mean power, and quality-per-watt. Extend
`samples/config-example/manifest.yaml` when adding a new target or tier policy.

## Vector-Store Boundary

Do not treat GraphRAG comparison as the multi-vector-store matrix. The current vector store is
FAISS. A Chroma/Qdrant/LanceDB matrix requires real adapters behind the RAG-store seam, then the
same source-span metric and `run-eval` path can compare them.
