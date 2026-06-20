# Real backend: vLLM + telemetry (Milestone 2)

Milestone 1 runs the loop on prebuilt Ollama. Milestone 2 adds a **vLLM** launcher (serves
HF weights behind the same OpenAI-compatible interface) plus a real telemetry hook. This is
the heavy, GPU-host path: vLLM may compile from source (CUDA toolchain), and weights are
multi-GB. Module detail is in [implementation/current.md](../implementation/current.md).

## 1. Install vLLM (once, GPU host)

    make build-vllm                      # MAX_JOBS-capped build + wheel cache under $DATA_DIR/wheels/

`build_vllm.sh` sources `scripts/shared/common.sh`, caps build parallelism via the canonical
`max_jobs()` helper (`min(cores//2, RAM_GiB//14)`, per AGENTS.md), and caches the built
wheels so a rebuild is reused. Override the version with `VLLM_SPEC='vllm==0.6.3' make build-vllm`.

## 2. Cache weights + verify the model id

    make prep-models PREP_BACKEND=vllm   # snapshot-downloads HF weights; a wrong/gated id is reported

`prep-models` is also the **id verification** step: a wrong repo id 404s and a gated repo
needs `HF_TOKEN` in `.env` (it is reported per-model, not fatal).

## 3. Run on the real backend, with telemetry

    make build-index                                  # if not already built
    make run-eval BACKEND=vllm MODEL=google/gemma-4-12B-it-qat-w4a16-ct TELEMETRY=1

or directly:

    llb run-eval --backend vllm --model google/gemma-4-12B-it-qat-w4a16-ct --telemetry

The launcher starts `vllm serve <model>` (controlling `--gpu-memory-utilization` and
`--max-model-len`, recorded for VRAM comparability), waits for readiness, runs the eval, then
kills the server. `--telemetry` adds a steady-state pass and records into the manifest:
**tokens/sec** (fixed prompt set + warmup), **cold-start load time** (separate from
throughput), **peak VRAM** (NVML), **requested vs served context**, and **tokenizer
efficiency** (tokens per UA char). vLLM logs land under
`$DATA_DIR/run-eval/<UTC timestamp>-<run id>/vllm/`.

## Config

vLLM serving knobs live in `RunConfig` (set in YAML or via flags): `backend: vllm`,
`vllm_host`, `vllm_port`, `gpu_memory_utilization`, `max_model_len`, `dtype`, `quantization`,
`measure_telemetry`. Use `make list-models` first to confirm a model fits this host.

## What needs your GPU

The launcher, telemetry, and build helper are built + unit-tested with fakes; the actual
**from-source build + serving a real model** (M2.4) runs only on a CUDA host. After a real
run, feed any fit corrections back into `samples/models_uk.yaml` / `planner.py` defaults.
