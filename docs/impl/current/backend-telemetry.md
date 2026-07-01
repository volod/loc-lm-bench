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

## Run-Time Estimation

Eval wall-clock per run is roughly `load_time + n_cases * mean_output_tokens / tokens_per_s`
(decode dominates; prefilling the retrieved RAG context adds a smaller per-case term that grows
with `top_k` and chunk size). `tokens_per_s` is the term that varies most across models, and it is
**measured** by `measure_throughput`, never derived from parameter count. Do not estimate run time
from model size -- architecture decouples size from decode speed:

- **Active vs total parameters (MoE).** A mixture-of-experts model routes each token through a
  small fraction of its weights, so decode cost scales with *active*, not total, parameters. The
  `qwen3.6-35b-a3b` candidate activates ~3B of 35B per token and can decode faster than a dense
  12B (`mamaylm-v2-12b`, `lapa`) despite the larger nominal size.
- **Attention layout.** Grouped-/multi-query attention (GQA/MQA) shrinks the KV cache and the
  memory bandwidth read per decoded token versus full multi-head attention (MHA), so two models of
  equal size can differ several-fold in tok/s. Sliding-window attention (Gemma 3/4) bounds KV
  growth at long context, keeping decode flat where a full-attention model slows down.
- **Quantization / format.** Decode on these hosts is memory-bandwidth-bound, so bits-per-weight
  (`iq3`, `q4_k_m`, `w4a16`, `fp8`, bf16) moves tok/s about as much as parameter count does.
- **VRAM fit vs offload -- usually the dominant factor on a 16 GiB card.** Weights that fit fully
  in VRAM decode at GPU memory bandwidth; a model that spills layers to CPU RAM (Ollama/llama.cpp
  offload) becomes CPU/PCIe-bandwidth-bound and runs far slower. `qwen3.6-35b-a3b:iq3` (~13 GiB)
  fits the card while dense `mistral-small-3.1-24b` and `mamaylm-v2-27b` spill and slow down.
  `list-models` reports the split per model (`gpu/total` layers, `gpu`/`offload` verdict); treat
  `offload` rows as slow until measured.

Measured on the 16 GiB RTX 4060 Ti (committed goldset, final split, Ollama), the nominal size
order is the *reverse* of the speed order -- fit-vs-offload and MoE routing dominate:

| model | arch / format | fits VRAM? | tok/s |
| --- | --- | --- | --- |
| `mamaylm-v2-12b` | dense 12B, Q4_K_M (~7.3 GiB) | yes (~9.4 GiB peak) | ~33 |
| `qwen3.6-35b-a3b` | MoE ~3B active, iq3 (~13 GiB) | yes (~15.9 GiB peak) | ~26 |
| `mistral-small-3.1-24b` | dense 24B, q4 (~15 GiB) | no -- offloads | ~14 |

The dense 24B is the slowest, the 35B MoE is faster, and the 12B that fits fully is fastest.
Note that peak VRAM is truthful for a model that fits (MamayLM ~9.4 GiB) but is capped at card size
for one that offloads (Qwen/Mistral pin ~15.9 GiB), so peak VRAM shows *whether* a model spilled,
not *how much* it needed.

For the model-architecture details behind these factors (MoE routing, attention variants,
sliding-window attention), see the
[LLM architecture gallery](https://sebastianraschka.com/llm-architecture-gallery/). To size a run
on THIS host, read `tokens_per_s` from prior run manifests (or the `recommend` chart's throughput
panel) rather than extrapolating from parameters; `list-models` estimates VRAM fit, not speed.

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
