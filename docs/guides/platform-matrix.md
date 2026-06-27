# Platform Matrix

Use this guide to run M7.4 platform comparisons: same logical model base across backend families,
power-aware telemetry, and GPU-class extension configs.

## Common-Base Backend Run

Default common base for the 16 GB CUDA host:
- Ollama: `gemma4:e4b`
- vLLM: `google/gemma-4-E4B-it-qat-w4a16-ct`
- llama.cpp: `hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:q4_0-it`

Run the full chain:

    make m7-4-platform-matrix

The target does:
1. rebuild the committed FAISS RAG index;
2. run Ollama with telemetry;
3. evict resident Ollama models and run vLLM with telemetry;
4. run llama.cpp with telemetry through the project-managed `llama-server` binary.

Override the defaults for another common base:

    make m7-4-platform-matrix \
      M7_4_OLLAMA_MODEL=<ollama-tag-or-hf.co-gguf> \
      M7_4_VLLM_MODEL=<hf-vllm-checkpoint> \
      M7_4_LLAMACPP_MODEL=<hf.co-gguf-source> \
      M7_4_MAX_MODEL_LEN=8192 \
      M7_4_LIMIT=20

For a larger Gemma 4 common base, use the 12B artifacts:

    make m7-4-platform-matrix \
      M7_4_VLLM_MODEL=google/gemma-4-12B-it-qat-w4a16-ct \
      M7_4_OLLAMA_MODEL=hf.co/google/gemma-4-12B-it-qat-q4_0-gguf \
      M7_4_LLAMACPP_MODEL=hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:q4_0

Use one split, one limit, one RAG index, and one telemetry protocol for every backend row. Do not
compare rows produced with different gold sets, splits, limits, retrieval configs, or context caps.

## Power Metrics

`run-eval --telemetry` records power when `nvidia-smi` is reachable:
- `telemetry.mean_power_w`
- `telemetry.peak_power_w`
- `telemetry.tokens_per_watt`
- `metrics.quality_per_watt`

`quality_per_watt = objective_score * tokens_per_s / mean_power_w`.

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
same source-span retrieval metric and `run-eval` path can compare them.
