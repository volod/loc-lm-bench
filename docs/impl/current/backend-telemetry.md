# Backend Telemetry

Backend telemetry explains how model-serving backends are launched, measured, and recorded. The
motivation is comparability: model quality, throughput, VRAM, load time, and power should be
captured in the same manifest shape regardless of whether the serving path is Ollama, vLLM, or
llama.cpp.

## vLLM Launcher

`src/llb/backends/vllm.py` implements `VllmLauncher`. It starts `vllm serve <model>`, waits for a
healthy OpenAI-compatible endpoint, exposes chat through the shared client, and stops the process
through the context manager.

Important knobs flow from `RunConfig` and CLI flags:

- `max_model_len`;
- `gpu_memory_utilization`;
- host and port;
- sampler environment from the vLLM preflight verdict.

The launcher preserves startup logs when readiness fails. This is important because vLLM failures
often happen before a JSON API is available.

## Build Rules

`scripts/build_vllm.sh` is the shell entry point. It sources `scripts/shared/common.sh` and uses the
canonical `max_jobs()` helper for source builds. Ordinary installs use `uv` and the shared package
cache. Only wheels intentionally built from a clean local checkout are exported under
`$DATA_DIR/wheels/<package>_<abi-key>_git<revision>/`.

```bash
make build-vllm
VLLM_SOURCE_DIR=../vllm make build-vllm
make prep-models PREP_BACKEND=vllm
```

The repository does not vendor vLLM or CUDA build outputs.

## Telemetry Fields

`src/llb/backends/telemetry.py` contains the backend-neutral measurement protocol.

`measure_throughput` runs fixed Ukrainian prompts with warmup iterations and a fixed output budget.
`VramSampler` polls NVML through an injectable reader. `collect_telemetry` records:

- steady tokens per second;
- tokenizer efficiency in tokens per Ukrainian character;
- peak VRAM;
- requested and served context;
- backend load time when the launcher owns startup;
- GPU memory utilization;
- mean and peak power when available;
- tokens per watt and quality per watt;
- detected GPU metadata;
- backend-specific fields such as vLLM sampler or llama.cpp GPU layer split.

Telemetry is enabled with `--telemetry` or `TELEMETRY=1` through Make.
Report assembly is split into required telemetry fields, optional power metrics, and optional
backend sampler metadata so the manifest shape stays typed while the collection flow stays small.

## Manifest Semantics

Telemetry is stored under `manifest.telemetry`; selected summary values are also mirrored into
`manifest.metrics` for board and MLflow use. A missing field should mean "not measured on this
path", not zero.

When a backend is already running and is reused by `--base-url`, cold load time is intentionally
null. When a launcher owns the process, load time is measured from launch to readiness.

## vLLM Sampler Preflight

`src/llb/backends/preflight.py` probes whether the flashinfer sampler works on the host. The
verdict is cached under `$DATA_DIR/llb/preflight/vllm_sampler.json` and includes the GPU driver so
driver changes can invalidate stale verdicts.

`launch_env` enables flashinfer only when the current verdict says it is safe. An explicit
environment value wins. This keeps the default path robust on consumer CUDA stacks where flashinfer
kernel compilation may fail, while still allowing faster sampling on hosts that support it.
