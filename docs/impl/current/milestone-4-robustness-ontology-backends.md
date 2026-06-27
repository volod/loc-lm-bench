# Milestone 4 Current State

## Milestone 4 -- robustness + ontology data prep + third backend (complete)

### Embedding-aware VRAM estimate -- `llb.backends.planner` (M4.1)
The weights estimate is no longer a flat `params_b x bpw`. Partial quants (w4a16 / int4 / fp8)
quantize only the linear layers while the token embedding + norms stay high-precision; with a
256k-token vocab that premium is large. `weights_mib_detailed(params_b, quant_bpw, hi_params,
embed_bpw)` prices the high-precision mass at `embed_bpw` (default 16) and only the remainder at
the quant bpw. The high-precision mass is `hi_precision_params(spec)`: an explicit
`hi_precision_params_b` wins (for quirks the vocab formula misses, e.g. Gemma 3n Per-Layer
Embeddings), else -- ONLY for a partial quant (`PARTIAL_QUANT_FORMATS`) -- the token embedding
(`vocab_size x hidden_size`, plus the untied head) derived from the spec. GGUF k-quants and
bf16/fp32 get no premium (they quantize uniformly). The detailed estimate flows through
`plan_model`, so the `AvailabilityResolver` fit (M3.2) and Optuna's over-VRAM prune (M3.4)
inherit it for free.

Arch fields come from the spec or, when omitted, a cached `config.json`: `enrich_arch(spec)`
reads `vocab_size` / `hidden_size` / `num_hidden_layers` / `tie_word_embeddings` via
`arch_from_config` (handles Gemma's nested `text_config`) using `cached_config_path` (HF cache,
never downloads). It fills only missing fields (curated YAML wins) and skips non-HF sources
(Ollama tags, `hf.co/...:Q4_K_M`). `list-models` / `resolve-models` enrich specs before planning.

Validated against the M2.4 measurement: gemma-4-E4B-it-w4a16 now estimates **9.81 GiB** (the
measured floor is 9.8 GiB) vs the old flat ~4.2 GiB; the Gemma 12B w4a16 gains a ~1.3 GiB
embedding premium (6.3 -> 7.6 GiB) and the 27B fp8 ~1.3 GiB. The E4B floor + the new arch fields
are recorded in `samples/models_uk.yaml` as the regression anchor.

Possible further improvements: the E4B high-precision mass is measurement-anchored
(`hi_precision_params_b`) rather than derived from the Gemma 3n PLE shapes in `config.json`;
sliding-window KV (Gemma 3/4) is still estimated as full attention (conservative at long ctx);
and `enrich_arch` fills gaps rather than letting `config.json` override curated values.

### Pre-launch VRAM-contention guard -- `llb.executor.contention` (M4.2)
Before a VRAM-owning backend (vLLM) starts, `run-eval` runs a guard so a resident process can no
longer trip vLLM's startup free-memory check (the original M2.4 failure: Ollama held ~2.8 GB, so
`gpu-memory-utilization x total` exceeded free VRAM). `plan_guard(total, free, requested_util,
weight_floor)` (pure) caps `gpu-memory-utilization` at `(free - margin) / total` (rounded down,
only ever lowered) -- the non-destructive default AUTO-DERATE -- and returns a `ContentionReport`
{total, free, safe_util, target, residents, derated, fits, action, note}. It ABORTS with an
actionable message when even the derated target cannot hold the M4.1 weight floor + vLLM's ~2 GB
non-weight serving overhead (`DEFAULT_VLLM_OVERHEAD_MB`: CUDA context, peak activations, CUDA-graph
capture) + a minimal KV working set; without that overhead term the guard would derate into a
doomed launch (the live finding: a budget that left 0 for KV blocks tripped vLLM's "No available
memory for the cache blocks"). Free VRAM comes from nvidia-smi (so the derate works without
`[telemetry]`); resident PIDs come from NVML when present (best-effort attribution in the note).

`apply_contention_guard` adds the opt-in escalations: `--evict` unloads Ollama's resident models
(`/api/ps` -> `keep_alive: 0` per model; never kills a process) then re-reads; `--wait` polls free
VRAM until the requested target fits or a timeout. The runner (`_guard_vllm_contention`) calls it
only for vLLM and only on the real launch path (injected launchers in tests skip it), lowers the
launcher's `gpu_memory_utilization` on a derate, and records the `ContentionReport` in the manifest
(`RunManifest.contention`). Readers, the evict, and sleep are injectable; the math + escalations
are unit-tested without a GPU.

Live-validated on the CUDA host (RTX 4060 Ti, vLLM 0.23.0): against a REAL resident VRAM user the
guard derated gpu-memory-utilization 0.80 -> 0.78 (a ~1.9 GB resident, still fitting gemma-4-E4B)
and ABORTED with the actionable note when a ~6 GB resident left only 9153 MB free (< the ~12609 MB
the model needs) -- end-to-end through `run-eval` (exit 1, no vLLM process started), with the real
nvidia-smi free-VRAM read + NVML attributing the resident PID. Possible further improvements: the
guard reads GPU 0 only (single-GPU assumption); the abort's KV headroom is a fixed floor rather than
the arch-derived KV for the served context.

### llama.cpp launcher -- `llb.backends.llamacpp` (M4.5)
The third backend the M3.2 resolver routes to: a model too big for vLLM's no-offload VRAM
resolves to its GGUF, which `llama-server` runs by splitting layers GPU<->CPU. `LlamaCppLauncher`
sits behind the same `BackendLauncher` + OpenAI-compatible `chat_once` seam as Ollama/vLLM, so the
eval/RAG/judge code is unchanged. `build_llamacpp_command` assembles the `llama-server` argv:
`llamacpp_source_args` maps a source to `-m <path.gguf>` (local) or `-hf <repo>[:quant]` (an HF
GGUF repo, incl. the Ollama-style `hf.co/<repo>:<quant>` the resolver's sources carry -- one
string serves on both GGUF backends); `-ngl` is the GPU/CPU offload split and `-c` the served
context. `start()` polls `/health` until 200 (preserving the startup log on failure, mirroring
vLLM), then reads the served `n_ctx` from `/props` (falling back to the requested `ctx_size`).

Telemetry reuses the backend-agnostic `collect_telemetry` (steady tokens/sec + peak VRAM); the
launcher records `n_gpu_layers` + `ctx_size` in its meta, and `TelemetryReport` now carries
`n_gpu_layers` so the served-vs-requested context (`requested_context`/`served_context`) and the
offload split land in the manifest. `llamacpp` is in `GATE_BACKENDS`, so the M3.3 reclaim gate
applies (it owns its VRAM). The runner's `_make_launcher` builds it from `RunConfig.llamacpp_host`
(env `LLAMACPP_HOST`, port parsed from the URL) + `n_gpu_layers`, with the context from
`max_model_len`. The process factory, HTTP probe, and sleep are injectable, so command building,
readiness, chat, telemetry, resolver routing, and the reclaim gate are all unit-tested without
llama.cpp/CUDA.

The `-ngl` offload split is now auto-derived from the planner instead of config-set: `resolve()`
carries the planner's `gpu_layers` on each `BackendCandidate`, and `resolver.llamacpp_offload_split`
returns that split for a resolved llama.cpp model with an OFFLOAD verdict (None when the chosen
backend is not llama.cpp or all layers fit on the GPU). `sweep` reads it and sets
`n_gpu_layers` per cell, so an oversized GGUF spills its non-fitting layers to CPU RAM instead of
the launcher default (-1 == every layer on GPU) OOMing the card. `run-eval` still honors an
explicit `--n_gpu_layers`/config value (single-model path, no resolver pass).

Provisioning the binary: `scripts/build_llamacpp.sh` builds `llama-server` from source with CUDA
(mirrors `build_vllm.sh`: sources `common.sh` for the canonical `max_jobs()` cap, keeps the
checkout clean, writes only under `$DATA_DIR/llb/llamacpp/`; `CUDA_ARCH`/`CUDA_HOST_CXX`/
`CUDA_ROOT` overridable, defaulting to sm_89 + `g++-12` + the newest local CUDA toolkit).

Live validation (RTX 4060 Ti, CUDA 12.6 build, driver 595.71.05): the real launcher served a real
GGUF (`Qwen2.5-0.5B-Instruct` q4_k_m, `-ngl -1`) through the freshly built CUDA `llama-server`
under `isolate_cell` -- `/health` ready in ~2 s, `/props` served context 4096 == requested, a
Ukrainian chat round-trip, steady ~364 tok/s, peak VRAM 1707 MB, and the reclaim gate saw VRAM
return to baseline (residual 0 MB, verdict `reclaimed`). Resolver routing was confirmed with the
live HF probe (a real GGUF-only repo -> `chosen_backend=llamacpp`), and the auto-derived split
produced `-ngl 49 of 62` for an oversized offload candidate.

Possible further improvements: keep extending the `/props` served-context parser if future
llama.cpp builds move `n_ctx` again. Known response shapes are handled with a fallback, and both
all-on-GPU and partial-offload run paths have real-host coverage.

### vLLM serving knobs + flashinfer preflight (M4.3)
`run-eval` now takes `--max-model-len` and `--gpu-memory-utilization` directly (previously only via
`--config`); both flow through `_load_config` -> `RunConfig.with_overrides`, so they are revalidated
by `RunConfig` (range-checked) and no YAML file is needed to tune a single run.

The flashinfer sampling kernel is gated on a preflight instead of a blanket default-off.
`llb.backends.preflight` runs the kernel build ONCE during `build-vllm` (`run_preflight` ->
`probe_sampler`) and records a definitive `SamplerVerdict` ({sampler, flashinfer_version, detail,
checked_at}) under `$DATA_DIR/llb/preflight/vllm_sampler.json`: `flashinfer` when the kernel builds
+ runs on this host, else `native` (the safe sampler). `launch_env` reads `flashinfer_sampler_ok()`
and sets `VLLM_USE_FLASHINFER_SAMPLER=1` only on a `flashinfer` verdict, else `0`; an explicit env
value always wins -- so the sampler is no longer a hardcoded `.env` default (now commented), it is
preflight-driven + overridable. The probe is injectable, so the verdict logic, persistence, and the
launch_env gating are unit-tested without CUDA; the real build-once probe (import flashinfer + a
CUDA sampling call) runs only on the host `build-vllm` targets.

Live-validated on the CUDA host: `run_preflight()` ran the real probe on the RTX 4060 Ti (sm_89)
and recorded the definitive verdict `native` (flashinfer 0.6.12 sampling kernel unavailable here)
to `$DATA_DIR/llb/preflight/vllm_sampler.json`, so `flashinfer_sampler_ok()` returns False and
`launch_env` keeps `VLLM_USE_FLASHINFER_SAMPLER=0` -- the documented sm_89 behavior, now confirmed.

Possible further improvements: auto-PIN a host-compatible flashinfer when the bundled one fails
(today the verdict is build-or-native, no version pinning); record the chosen sampler in the run
manifest for provenance; re-run the preflight on a flashinfer/driver change without a full vLLM
rebuild.

### Ontology-assisted gold-set drafting -- `llb.prep.ontology` (M4.4)
The reserved `GOLDSET_MODE=draft` is now a 7-stage prep pipeline (CLI `prepare-goldset-draft`,
Makefile `GOLDSET_MODE=draft` over `CORPUS`) that drafts UNVERIFIED RAG gold items from a corpus
and links every artifact to exact evidence. It is deliberately NOT a synonym for the M3.5
one-prompt `prepare-goldset`; it is a data-preparation ontology, not a GraphRAG runtime (that is
Milestone 6). One small module per grained stage, each injected-unit-tested:

- **endpoint adapter (`endpoint.py`).** All stages drive one injectable `LLMComplete`.
  `build_complete`
  returns a LOCAL OpenAI-compatible call (`make_client` + `chat_once`, no corpus egress -- the
  default) or, opt-in, the frontier `litellm_complete` (egress -- the Milestone H decision).
  `EndpointConfig` validates kind/model and exposes `egress` + a provenance dict; cost/tokens
  accrue in the shared `ProvenanceLog`.
- **stage 1 inventory (`inventory.py`).** Reads `.md`/`.txt` recursively (corpus-relative ids),
  treats on-disk text as canonical (offsets stay exact), records a sha256 + char count, and
  segments sections (markdown headings, else paragraph blocks) for a coverage axis.
- **stage 2 extract (`extract.py`).** The pluggable `ExtractionAdapter` seam; default
  `LLMExtractionAdapter` does one call/doc for entities + aliases/coreference + events + claims +
  SRO facts. Every quoted span is grounded via `ground_quote` (reusing `frontier.ground_span`)
  against the FULL doc (so a truncated long-doc call still anchors exactly); ungrounded artifacts
  and evidence-less entities are dropped. The Python-native NER/coreference adapter (Stanza / spaCy
  `uk_core_news`) is an opt-in plug-in implementing the protocol, kept out of base deps.
- **Closed entity-type vocabulary (`entity_types.py`).** Entity nodes carry one type from a CLOSED,
  13-type OntoNotes-derived set (`PERSON, NORP, ORG, LOC, LAW, WORK, PRODUCT, EVENT, DATE, DURATION,
  MONEY, QUANTITY, MISC`) -- granular enough for typed facts (e.g. legal codes/treaties -> `LAW`,
  IP objects -> `WORK`, time spans -> `DURATION`). It is ENFORCED, not just suggested: the
  vocabulary (with Ukrainian glosses) is injected into `extraction_prompt`, and every emitted type
  (LLM or spaCy `map_label`) passes through `normalize_entity_type`, so synonyms collapse to their
  canonical type (`GPE`->`LOC`, `WORK_OF_ART`/`PATENT`->`WORK`, `TREATY`->`LAW`, ...) and any
  out-of-vocabulary label becomes `MISC` -- the schema can never silently expand. The signed schema
  is [`docs/design/graph-ontology-schema.md`](../../design/graph-ontology-schema.md). Extend by editing
  the one module (`tests/test_ontology_m56.py` covers the normalizer + closure).
- **stage 3 induce (`induce.py`).** Pure deterministic aggregation of extracted entity types +
  relations into a CONSTRAINED `OntologyCandidate` (capped groups, hapax-dropped) with support
  count, frequency confidence, and example surface forms.
- **stage 4 coverage (`coverage.py`).** Builds fact/entity seeds tagged with strata
  (relation/entity-type x section x difficulty; difficulty from evidence length + relation
  rarity), then a seeded greedy picks coverage-first, fills the budget deterministically.
- **stage 5 draft (`draft.py`).** One UA question/reference/answer-span per seed from a bounded
  context window around the evidence, difficulty- and focus-aware, instructed to avoid give-aways.
- **stage 6 refine (`refine.py`).** Re-grounds via `frontier.build_drafted_items` (now taking a
  `provenance`/`id_prefix`, so unsupported answers are dropped), rejects circular items (answer in
  the question, or question == reference), and dedups per doc by question and by answer span.
- **stage 7 emit (`pipeline.py`).** Assigns splits and writes a self-contained bundle under
  `$DATA_DIR/prepare-goldset/<UTC ts>/`: `goldset.jsonl` (`verified=false`,
  `provenance="ontology-drafted"` -- the new schema value), a verbatim `corpus/` copy (so the
  bundle self-validates), `ontology.json`, `extraction.jsonl`, and a `provenance.json` linking
  endpoint / prompt fingerprints / per-doc hashes / stage counts / cost.

Nothing is verified: a frontier cross-check + a human stratified sample-verify (MH.5) still gate
any scoring. The full flow is proven by a fake-endpoint test (one callable answering both the
extraction and drafting prompts, like a real local model) that runs all stages and validates the
emitted bundle with the M0 validator.

Possible further improvements: ship a concrete Stanza / spaCy `ExtractionAdapter` plug-in (today
only the LLM adapter + seam exist); add the second-frontier cross-check (grounding/non-circularity)
as pipeline code before MH.5; chunk over-long docs for extraction rather than one truncated call
(`EXTRACT_MAX_CHARS`); derive type confidence from a richer signal than raw frequency; and feed the
induced ontology types into the drafting prompt as explicit constraints (today they inform coverage
strata only).

- - **M4.1** (embedding-aware weights (`weights_mib_detailed` + `hi_precision_params`,
- partial-quant-gated) + `config.json` enrichment (`enrich_arch`/`arch_from_config`); fed through
- `plan_model` to resolver + Optuna; YAML arch fields + measured anchor): DONE + LIVE-VALIDATED (E4B
- predicted 9.81 vs measured 9.80 GiB on a live vLLM load, 0.1%)
- - **M4.2** (pre-launch VRAM-contention guard (`plan_guard` derate + abort, `--evict`/`--wait`),
- wired into `run-eval` for vLLM, recorded in the manifest): DONE + LIVE-VALIDATED (real resident
- user: derate 0.80->0.78 + abort end-to-end through `run-eval`, no vLLM started)
- - **M4.5** (llama.cpp launcher (`LlamaCppLauncher` `llama-server` subprocess: `-hf`/`-m` source,
- `-ngl` offload split, `/health`+`/props`), telemetry (`n_gpu_layers` + served ctx), reclaim gate,
- `_make_launcher` wiring; planner-derived `-ngl` (`llamacpp_offload_split` -> `sweep`);
- `scripts/build_llamacpp.sh` (CUDA)): DONE + LIVE-VALIDATED (RTX 4060 Ti: real GGUF served on GPU
- under the isolation gate, VRAM reclaimed; routing + auto-derive confirmed)
- - **M4.3** (run-eval `--max-model-len` / `--gpu-memory-utilization` (revalidated, no YAML) +
- flashinfer sampler preflight (`build-vllm` records a verdict; `launch_env` gates the sampler on
- it)): DONE + LIVE-VALIDATED (host preflight verdict recorded: `native` on sm_89, flashinfer
- 0.6.12)
- - **M4.4** (ontology-assisted draft pipeline (`llb.prep.ontology`: 7 grained stages + endpoint
- adapter, `prepare-goldset-draft` / `GOLDSET_MODE=draft`), exact-evidence-grounded `verified=false`
- `ontology-drafted` bundle with full provenance): DONE (per-stage + fake-endpoint full-flow unit
- tests; frontier cross-check + spaCy/Stanza plug-in are residual)

**Milestone 4 is complete and ALL on-hardware live validation has now passed on the CUDA host**
(RTX 4060 Ti, vLLM 0.23.0, driver 595.71.05): M4.1 the planner's embedding-aware estimate matched a
live vLLM load (predicted 9.81 vs measured 9.80 GiB, gemma-4-E4B w4a16); M4.2 the contention guard
derated (0.80 -> 0.78) and aborted as designed against a real resident VRAM user, end-to-end through
`run-eval` (no vLLM started); M4.3 the host flashinfer preflight verdict was recorded (`native` on
sm_89); M4.5 a real GGUF resolved to and served through the llama.cpp launcher on the GPU under the
isolation gate with the planner-derived `-ngl`. The remaining residuals are NOT live validation:
the small run-path CODE hardening (M4.1 sliding-window KV + config override, M4.2 multi-GPU +
arch-derived KV abort floor, M4.3 flashinfer auto-pin / sampler-in-manifest, M4.5 `/props` shape +
a real partial-offload split) and the M4.4 data-prep hardening (second-frontier cross-check, opt-in
Stanza/spaCy adapter, long-doc chunking, richer ontology confidence) -- carried forward in
[`plan.md`](../plan.md) (M5.6), landing with the M5 verified-data gate + the M6 extraction reuse.
