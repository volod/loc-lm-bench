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

## Multi-Vector-Store Status

Only FAISS is a vector-store implementation in the current RAG store. The existing GraphRAG
comparison is a graph-vs-vector retrieval comparison, not a Chroma/Qdrant/LanceDB vector-store
matrix. A real multi-vector-store run requires new adapters behind the RAG-store seam before it can
be benchmarked honestly.
