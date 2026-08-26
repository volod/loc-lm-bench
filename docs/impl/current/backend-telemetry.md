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
canonical `llb_max_jobs()` helper for source builds. `make venv` calls this helper automatically on
CUDA hosts (`VENV_INSTALL_VLLM=auto`) after the editable install; set `VENV_INSTALL_VLLM=0` for a lean
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
`VramSampler` polls NVML through an injectable reader. Every sampler in `telemetry_samplers` answers
the same question -- what was true WHILE the generations ran -- so they share one `BackgroundSampler`
contract (injected reader, daemon thread for the length of a `with` block, swallowed read errors)
and differ only in what they keep: the peak (VRAM), every reading (power), or the last reading that
existed (`LastValueSampler`, for a signal that can vanish before the run ends). `collect_telemetry`
records:

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

### Full-Roster Throughput Baseline (RTX 4060 Ti 16 GiB)

Every logical entry in `samples/configs/models_uk.yaml` measured back to back under one protocol:
`collect_telemetry` with the fixed `telemetry.throughput` Ukrainian prompt set, `max_new_tokens=128`,
one discarded warmup pass, `num_ctx`/`max_model_len` pinned to 4096, and each model unloaded before
the next so every run starts from the same VRAM state. These are SHORT-prompt decode rates; a RAG
lane that prefills retrieved context reads lower for the same model (see the context-ablation rows
in [RAG core](rag-core/context-ablation-evidence.md)).

Rows carry the date they were taken, because they are not all one sitting: the roster sweep ran on
2026-08-03 (Ollama 0.20), and a generation that lands later is re-measured under the same protocol
and joins the table beside the generation it replaces
([refreshing one row](#refreshing-one-row-after-a-generation-upgrade)). Comparing rows of different
dates compares a model AND the runtime that served it.

`min/100` is the derived decode-only run-sizing figure from the estimator above: minutes to answer
100 cases at 256 output tokens each, excluding load time and RAG prefill. `tok/UA-char` is the
tokenizer-efficiency field from the same telemetry record (LOWER is denser output per token) and
carries a content confound -- read the caveat below before ranking on it.

| model | served artifact | backend | tok/s | tok/UA-char | min/100 | peak VRAM (MB) | placement | measured |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `gemma-4-e4b-it-w4a16` | `gemma4:e4b` | ollama | 63.45 | 0.323 | 6.7 | 11657 | GPU-resident | 2026-08-03 |
| `qwen3-30b-a3b` (Qwen 3, previous) | `qwen3:30b` | ollama | 38.87 | 0.202 | 11.0 | 16096 | offload, ~3.3B active | 2026-08-03 |
| `qwen3.6-35b-a3b-fp8` (Qwen 3.6, previous) | `batiai/qwen3.6-35b:iq3` | ollama | 36.90 | 0.353 | 11.6 | 15908 | GPU-resident, ~3B active | 2026-08-03 |
| `lapa-v0.1.2-instruct` | Lapa 12B GGUF Q4_K_M | ollama | 31.08 | 0.201 | 13.7 | 9835 | GPU-resident | 2026-08-03 |
| `mamaylm-v2-12b` | MamayLM 12B GGUF Q4_K_M | ollama | 30.99 | 0.325 | 13.8 | 9837 | GPU-resident | 2026-08-03 |
| `gemma-4-12b-it-w4a16` | `google/gemma-4-12B-it-qat-w4a16-ct` | vllm | 29.48 | 0.317 | 14.5 | 14827 | GPU-resident | 2026-08-03 |
| `gemma-4-26b-a4b` | `gemma4:26b` | ollama | 26.94 | 0.318 | 15.8 | 16002 | offload, ~3.8B active | 2026-08-03 |
| `mistral-small-3.1-24b` | `mistral-small3.1:24b` | ollama | 12.41 | 0.331 | 34.4 | 15878 | offload, dense | 2026-08-03 |
| `qwen3.8-27b` (Qwen 3.8, CURRENT) | `qwen3.8:27b` | ollama | 10.38 | 0.351 | 41.1 | 14894 | offload 28%/72% | 2026-08-26 |
| `mamaylm-v2-27b-fp8` | MamayLM 27B GGUF Q4_K_M | ollama | 7.82 | 0.306 | 54.6 | 15957 | offload 23%/77% | 2026-08-03 |
| `qwen3.6-27b` (Qwen 3.6, previous) | `qwen3.6:27b` | ollama | 4.59 | 0.350 | 93.0 | 15233 | offload 44%/56% | 2026-08-03 |

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
  the answer body. Re-checked across the whole roster and resolved in
  [thinking-suppression verdicts](#thinking-suppression-verdicts-per-roster-tag) below: it is this
  ONE tag, no lever fixes it, and the flag is sufficient everywhere else.
- **`gemma-4-12b` had no Ollama path when this baseline was measured.** Ollama 0.20 rejected the
  `gemma4` architecture in a raw GGUF (`unknown model architecture: 'gemma4'`), so both the curated
  `gemma4:12b` tag and the first-party QAT `q4_0` GGUF failed to load and the row above is measured
  on the manifest-primary vLLM w4a16 checkpoint instead. The curated `gemma4:e4b` / `:26b` tags were
  unaffected because Ollama's own engine serves them. **This no longer holds:** on Ollama 0.32.15
  (2026-08-23, same host) `hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0` loads and answers
  normally. The throughput row above is NOT re-measured -- re-run the protocol before quoting a
  `gemma-4-12b` Ollama rate.
- **The current Qwen generation more than doubles the one it replaces, and offload is why.**
  `qwen3.8-27b` (the `qwen3.8:27b` q4_k_m tag, measured 2026-08-26 on the same RTX 4060 Ti under
  Ollama 0.32.15) decodes at **10.38 tok/s** with **28%/72% CPU/GPU** placement and 14894 MB peak
  VRAM, over three measured passes reading 10.38-11.04 tok/s (the fastest of the three shared the
  host with an unrelated request and is not the row). Its record also carries `served_context`
  4096 -- the pinned window was the window served -- and a 5.58 s load, which is a warm-page-cache
  load rather than a first read from disk. Against that: `qwen3.6-27b`'s 4.59 tok/s at **44%/56%**
  on 2026-08-03. Two dense 27B q4 artifacts of nearly equal size, and the faster one is the one
  that keeps more of itself on the card: 41.1 min/100 instead of 93.0 -- an 82-item final split
  falls from about 76 minutes of decode to about 34. The reading is bounded by the dates: the rows
  differ in Ollama major version as well as in generation, so this is "what the Qwen lane costs to
  run today" and NOT an isolated weights-to-weights delta. Re-measuring `qwen3.6:27b` on the
  current Ollama would separate the two, and is deliberately not done here -- the previous
  generation is kept for a QUALITY comparison, and a throughput row carries no quality claim. What
  would overturn it: another Ollama release that changes the offload split (the split, not the
  weights, is what moves this number), or a host with more VRAM, where neither model offloads and
  the ordering may invert.

The 2026-08-03 sweep predates the command below and left only a scratch JSON; rows taken since land
under `$DATA_DIR/measure-throughput/<run timestamp>/`. Neither is a run bundle and neither is what a
reader should cite: after a backend or driver upgrade, re-measure under the same protocol rather
than quoting a number taken against a runtime that no longer exists here.

### Refreshing One Row After A Generation Upgrade

A generation swap invalidates the row it replaces, so the protocol above is a COMMAND rather than a
one-off script -- a row measured months apart is only comparable if nothing about how it was taken
drifted:

```bash
make measure-throughput MODELS=qwen3.8-27b   # one entry; MODELS= omitted measures the whole roster
```

`llb.backends.roster_throughput` owns the protocol constants (128 new tokens, one warmup pass, ctx
4096, the fixed prompt set), the per-entry measurement, the `min/100` derivation, and the markdown
row the table above carries; `llb.cli.models.throughput` is the command. Per model it resolves the
backend the host would actually serve from (`resolve`, so the measured artifact is the one a run
would use), evicts every resident Ollama model, warm-loads the model once (timing that load and
confirming the served window, which Ollama reports only after a request has loaded it), measures,
then runs the cell under the shared isolation contract so the next model starts from a reclaimed,
cooled GPU. Records land under `$DATA_DIR/measure-throughput/<run timestamp>/rows.json`, carrying
the derived reading beside the raw telemetry record it came from.

The pieces: protocol + measurement `src/llb/backends/roster_throughput.py`, command
`src/llb/cli/models/throughput.py` (`make measure-throughput`, `MODELS=`/`CONTEXT=`), Ollama's
placement probe `src/llb/backends/ollama.py`, shared sampler contract
`src/llb/backends/telemetry_samplers.py`; tests in `tests/llb/backends/test_roster_throughput.py`
and `tests/llb/cli/test_cli_measure_throughput.py`.

**Placement is sampled DURING the generations, not after them.** Ollama reports the GPU/CPU byte
split of a resident model on `/api/ps`, and that split -- not the planner's estimate -- is what the
`placement` column states for an Ollama row. The reading has to be taken while the model is serving:
on this host a 17 GB model is evicted the moment something else asks for VRAM, and the first
measurement of `qwen3.8-27b` recorded no split at all because an unrelated request arrived as the
last generation returned. `LastValueSampler` polls the launcher's own probe alongside the peak-VRAM
sampler and keeps the last reading that existed, so the row states what served the run.

For the model-architecture details behind these factors (MoE routing, attention variants,
sliding-window attention), see the
[LLM architecture gallery](https://sebastianraschka.com/llm-architecture-gallery/). To size a run
on THIS host, read `tokens_per_s` from prior run manifests (or the `recommend` chart's throughput
panel) rather than extrapolating from parameters; `list-models` estimates VRAM fit, not speed.

## Thinking Suppression Verdicts Per Roster Tag

2026-08-23, RTX 4060 Ti 16 GiB, Ollama 0.32.15. Every logical entry of
`samples/configs/models_uk.yaml` was sent the same Ukrainian RAG-shaped prompt through the
launcher's own path (`/api/chat`, `think: false`, `num_ctx` 4096, temperature 0, 256-token budget),
and the four tags that matter were then measured on a bounded 20-case `run-eval` cell over the
committed `ua_squad_postedited_v1` final split (`MAX_TOKENS=512`, pinned retrieval: recall@5 =
0.900, MRR 0.787 for every row) so the guard's rates come from real bundles rather than a probe.
A verdict is recorded for EVERY tag, including the ones that never leaked.

| tag | serves via | reasoning template | native flag enough | verdict |
| --- | --- | --- | --- | --- |
| `gemma4:e4b` | ollama | no | n/a | flag alone; measured leak rate 0.00 on the eval cell |
| `gemma4:26b` | ollama | no | n/a | flag alone |
| `hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0` | ollama | no | n/a | flag alone |
| `gemma-4-12b-it-w4a16` | vllm | no | n/a | nothing to suppress; the vLLM launcher sends no thinking flag and does not need one |
| `qwen3.8:27b` | ollama | YES | YES | flag alone; measured leak rate 0.00 on the eval cell |
| `qwen3.6:27b` | ollama | yes | YES | flag alone |
| `batiai/qwen3.6-35b:iq3` | ollama | yes | YES | flag alone |
| `qwen3:30b` | ollama | YES | **NO** | **not scoreable as a non-thinking tag** -- see below |
| `mistral-small3.1:24b` | ollama | no | n/a | flag alone |
| MamayLM v2.0 12B / 27B GGUF | ollama | no | n/a | flag alone |
| `lapa-v0.1.2-instruct` GGUF | ollama | no | n/a | flag alone |

**The current Qwen generation fixes it; the superseded one cannot be fixed.** `qwen3.8:27b` is a
thinking model -- with `think: true` Ollama returns a populated `thinking` field beside the answer
-- and with `think: false` it returns clean Ukrainian prose and an empty `thinking` field. Its
20-case eval cell reads `reasoning_leak_rate` 0.00, `language_mismatch_rate` 0.00, mean completion
15.1 tokens, objective 0.410. `qwen3:30b`, on the same cell, reads `reasoning_leak_rate` **1.00**,
mean completion **465.2** tokens, objective **0.036** -- and reliability 1.000, because every case
is `ok` by status. That contrast is the whole reason the guard exists: nothing in the pre-guard
record distinguishes "answered badly" from "never answered".

**The prompt-level instruction made it worse, and the negative result is the verdict.** Re-running
the same cell with `SUPPRESS_REASONING_PROMPT=1` (the `eval.rag.no_reasoning` system-prompt suffix,
on top of the unchanged `think: false`):

| `qwen3:30b` lane | leak rate | language mismatch | mean leak chars | mean completion tokens | objective |
| --- | ---: | ---: | ---: | ---: | ---: |
| native flag only | 1.00 | 0.00 | 936 | 465.2 | 0.036 |
| native flag + prompt instruction | 1.00 | 0.95 | 1363 | 395.9 | 0.023 |

Naming `<think>` in the instruction did not suppress the block -- it primed the tag into its English
reasoning register. The leak markers flip from a Ukrainian deliberation frame (12 of 20 open
"Давайте проаналізуємо контекст...") to a bare `</think>` plus English openers (15 + 5), the
answers go from 20/20 Ukrainian to 19/20 English, the leaked text grows 46%, and the objective
drops. So: for `qwen3:30b`, neither lever works, and the fix is the ROSTER, not the scorer -- score
`qwen3.8:27b` as the current Qwen generation and treat any `qwen3:30b` row as a measurement of the
serving configuration rather than of the model.

**Read `mean_reasoning_leak_chars` before quoting a throughput number for a leaking tag.**
`qwen3:30b` posts the highest tok/s of the four cells (57.5) precisely because it is emitting 465
tokens of deliberation per case; `qwen3.8:27b` posts 11.2 while delivering the best objective. A
decode rate over text that was never an answer is not a rate an operator can spend.

The guard's fields, detection rules, and the reasoning behind each are in
[scoring](rag-core/scoring.md#response-integrity-guard-thinking-suppression-and-answer-language-guard).
What would overturn these verdicts: an Ollama release that changes a chat template (the `qwen3:30b`
leak is a template artefact, not a weights property), or a new roster generation -- re-run the probe
and the two eval cells rather than carrying these rows forward.

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
