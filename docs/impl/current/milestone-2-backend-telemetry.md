# Milestone 2 Current State

## Milestone 2 -- real backend + telemetry (complete)

A real vLLM backend behind the same interface, a steady-state telemetry hook, and the
MAX_JOBS-capped build entrypoint -- validated end to end on a real model (see the
[vLLM guide](../../guides/vllm-backend.md) and `samples/run_config_vllm_uk.yaml`).

### vLLM launcher -- `llb.backends.vllm` (M2.1)
`VllmLauncher` + `build_vllm_command` (pure). Documented under Backends above (incl. the
`launch_env` flashinfer-sampler default and the on-failure log preservation). The thin
`scripts/build_vllm.sh` entrypoint sources `scripts/shared/common.sh`, exports its canonical
`max_jobs()` result (`min(cores//2, RAM_GiB//14)`, AGENTS.md), and delegates to
`llb.build.vllm`. The default binary-only install and all ordinary dependencies use uv's
shared cache. Only a wheel built from `VLLM_SOURCE_DIR=<clean-git-checkout>` is exported
under `$DATA_DIR/wheels/vllm_<abi-key>_git<revision>/`. Weights are cached by `prep-models`.

    make build-vllm # prebuilt wheel via uv shared cache
    VLLM_SOURCE_DIR=../vllm make build-vllm # one ABI-keyed checkout wheel
    make prep-models PREP_BACKEND=vllm # cache HF weights (verifies repo ids)
    llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry # the
    M2.4 run

### Telemetry hook -- `llb.backends.telemetry` (M2.2)
`measure_throughput` runs the steady-state protocol (fixed UA prompt set + fixed
max_new_tokens + N warmup iters) over `launcher.chat`, so tokens/sec is comparable across
models; cold-start `load_time_s` is recorded separately by launchers that own the backend
lifecycle, and remains null for an already-running external daemon such as Ollama.
`VramSampler` polls NVML (injected reader) for peak VRAM. `collect_telemetry` assembles the manifest
record:
steady tokens/sec, tokenizer efficiency (tokens/UA-char), peak VRAM, requested-vs-served
context, load time, gpu-memory-utilization, and detected GPU. Wired into `run-eval`
behind `config.measure_telemetry` (`--telemetry`); recorded under `manifest.telemetry`.

### M2.4 real-model validation (RTX 4060 Ti 16 GB)
`google/gemma-4-E4B-it-qat-w4a16-ct` served via vLLM 0.23.0 and scored under the executor
produced a real ranked row + full telemetry: objective quality 0.801, **63.8 tok/s** steady,
peak VRAM **15.7 GB** (at gpu-memory-utilization 0.80), cold load **112 s**, served context
8192, tokenizer 0.33 tok/UA-char. vLLM resolves `Gemma4ForConditionalGeneration` +
`compressed-tensors` natively; attention falls back to TRITON (Gemma-4 heterogeneous head
dims), the flashinfer sampler is disabled (see `launch_env`), and `max_model_len` is capped
so the KV cache fits (the native 131072 window would over-reserve and fail startup).

Planner-vs-measured fit: the model's **weights load 9.8 GiB**, ~2.3x the old flat ~4.2 GiB
estimate (`params_b x bpw`). w4a16 quantizes only the linear layers while Gemma's 256k-token
embedding stays high-precision, so the flat product under-estimated w4a16 weights. The
embedding-aware estimator that fixes this is now delivered (M4.1 below); the measured floor is the
regression anchor in `samples/models_uk.yaml`.

### Milestone 2 status

- **M2.1** (`VllmLauncher` + `build_vllm_command` + MAX_JOBS build helper / script): DONE
- **M2.2** (telemetry hook (steady tokens/sec, peak VRAM, served ctx, load time, tok/char)): DONE
- - **M2.3** (candidate list in `samples/models_uk.yaml`; vLLM repo ids verified via `prep-models`):
- DONE
- **M2.4** (validated on a real vLLM-served model (gemma-4-E4B-it-w4a16) w/ real telemetry): DONE

The M2.4 run surfaced three non-blocking gaps, all now DELIVERED in Milestone 4 below: the
embedding-aware VRAM estimate (M4.1), a pre-launch VRAM-contention guard (M4.2), and the vLLM
serving knobs as `run-eval` CLI flags (M4.3). The only remaining on-hardware confirmation (a real
contended launch) is tracked forward in [`plan.md`](../plan.md) (M5.6).
