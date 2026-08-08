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
canonical `max_jobs()` helper for source builds. `make venv` calls this helper automatically on CUDA
hosts (`VENV_INSTALL_VLLM=auto`) after the editable install; set `VENV_INSTALL_VLLM=0` for a lean
environment or `VENV_INSTALL_VLLM=1` to force the vLLM install. Ordinary installs use `uv` and the
shared package cache with binary wheels only. Only wheels intentionally built from a clean local
checkout are exported under `$DATA_DIR/wheels/<package>_<abi-key>_git<revision>/`.

```bash
make venv
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

### Full-Roster Throughput Baseline (2026-08-03, RTX 4060 Ti 16 GiB)

Every logical entry in `samples/configs/models_uk.yaml` measured back to back under one protocol:
`collect_telemetry` with the fixed `telemetry.throughput` Ukrainian prompt set, `max_new_tokens=128`,
one discarded warmup pass, `num_ctx`/`max_model_len` pinned to 4096, and each model unloaded before
the next so every run starts from the same VRAM state. These are SHORT-prompt decode rates; a RAG
lane that prefills retrieved context reads lower for the same model (see the context-ablation rows
in [RAG core](rag-core/context-ablation.md#context-ablation-evidence)).

`min/100` is the derived decode-only run-sizing figure from the estimator above: minutes to answer
100 cases at 256 output tokens each, excluding load time and RAG prefill. `tok/UA-char` is the
tokenizer-efficiency field from the same telemetry record (LOWER is denser output per token) and
carries a content confound -- read the caveat below before ranking on it.

| model | served artifact | backend | tok/s | tok/UA-char | min/100 | peak VRAM (MB) | placement |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `gemma-4-e4b-it-w4a16` | `gemma4:e4b` | ollama | 63.45 | 0.323 | 6.7 | 11657 | GPU-resident |
| `qwen3-30b-a3b` | `qwen3:30b` | ollama | 38.87 | 0.202 | 11.0 | 16096 | offload, ~3.3B active |
| `qwen3.6-35b-a3b-fp8` | `batiai/qwen3.6-35b:iq3` | ollama | 36.90 | 0.353 | 11.6 | 15908 | GPU-resident, ~3B active |
| `lapa-v0.1.2-instruct` | Lapa 12B GGUF Q4_K_M | ollama | 31.08 | 0.201 | 13.7 | 9835 | GPU-resident |
| `mamaylm-v2-12b` | MamayLM 12B GGUF Q4_K_M | ollama | 30.99 | 0.325 | 13.8 | 9837 | GPU-resident |
| `gemma-4-12b-it-w4a16` | `google/gemma-4-12B-it-qat-w4a16-ct` | vllm | 29.48 | 0.317 | 14.5 | 14827 | GPU-resident |
| `gemma-4-26b-a4b` | `gemma4:26b` | ollama | 26.94 | 0.318 | 15.8 | 16002 | offload, ~3.8B active |
| `mistral-small-3.1-24b` | `mistral-small3.1:24b` | ollama | 12.41 | 0.331 | 34.4 | 15878 | offload, dense |
| `mamaylm-v2-27b-fp8` | MamayLM 27B GGUF Q4_K_M | ollama | 7.82 | 0.306 | 54.6 | 15957 | offload 23%/77% |
| `qwen3.6-27b` | `qwen3.6:27b` | ollama | 4.59 | 0.350 | 93.0 | 15233 | offload 44%/56% |

Practical reading of `min/100`: an 82-item final split costs about 5 minutes on `gemma-4-e4b` and
over an hour on `qwen3.6-27b`, so a roster-wide sweep is dominated by its two slowest rows. Add
`load_time_s` once per model (cold Ollama load is seconds; the vLLM row measured 140 s, chiefly
CUDA-graph capture) and a per-case prefill term that grows with `top_k` and chunk size.

What the full roster adds beyond the three-row table above:

- **The smallest entry wins by a wide margin.** `gemma-4-e4b` is 1.63x the next model and 13.8x the
  slowest. Its 63.45 tok/s reproduces the manifest's M2.4 vLLM reference (~64 tok/s) on a different
  backend, which makes the protocol's cross-backend comparability observable rather than assumed.
- **MoE routing beats VRAM residency.** `qwen3-30b-a3b` posts 38.87 tok/s while CPU-offloaded, ahead
  of three dense models that fit entirely in VRAM: with ~3.3B active parameters the offloaded
  experts are mostly not read per token. Fit-vs-offload is the dominant factor only among models of
  comparable active size.
- **Dense offload is where speed collapses.** The two slowest rows are dense 24B/27B artifacts at
  4.59-12.41 tok/s, and `qwen3.6-27b` at 44%/56% CPU/GPU is the roster floor.
- **`tok/UA-char` measures what the model WROTE at least as much as how it tokenizes -- never rank
  on it without reading the generations.** Dividing `tok/s` by it yields a tempting "UA chars/s"
  (`gemma-4-e4b` 196, `qwen3-30b-a3b` 193, `lapa` 155, `mamaylm-v2-12b` 95), which would promote
  `qwen3-30b-a3b` to a near-tie for first. Inspecting the generations shows two different causes
  behind one number: `lapa` emits genuine Ukrainian prose at ~6.1 chars/token against
  `mamaylm-v2-12b`'s ~3.1 on the SAME Gemma-3 base, part tokenizer adaptation and part
  MamayLM's markdown-heavy formatting; but `qwen3-30b-a3b` scores low because it answered a
  Ukrainian prompt in ENGLISH reasoning prose ("Okay, I need to explain what copyright is..."),
  and English tokenizes far denser than Ukrainian. Its apparent character throughput is a
  language artifact, not Ukrainian delivered per second. Quote `tok/s` for run sizing and treat
  `tok/UA-char` as a diagnostic that requires reading the generations -- the same verbosity/content
  confound documented for token-F1 scoring in [RAG core](rag-core/scoring.md#scoring).
- **`think=false` did not stop `qwen3:30b` from emitting visible reasoning.** The launcher sends
  Ollama's native `think: false` on every call, yet this tag returned first-person deliberation in
  the answer body. Treat the manifest's "disable thinking for scoring" note as necessary but not
  sufficient for this tag until re-checked.
- **`gemma-4-12b` has no Ollama path on this host.** Ollama 0.20 rejects the `gemma4` architecture in
  a raw GGUF (`unknown model architecture: 'gemma4'`), so both the curated `gemma4:12b` tag and the
  first-party QAT `q4_0` GGUF fail to load; its row is measured on the manifest-primary vLLM w4a16
  checkpoint instead. The curated `gemma4:e4b` / `:26b` tags are unaffected because Ollama's own
  engine serves them.

The per-model JSON for this baseline is a scratch artifact, not a run bundle: re-measure with the
same protocol rather than citing the numbers after a backend or driver upgrade.

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
