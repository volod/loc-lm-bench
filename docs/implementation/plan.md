# loc-lm-bench — Implementation Plan (forward work)

## Context

loc-lm-bench is a production-grade, reproducible internal tool that selects the best
open-weight LLM for Ukrainian RAG + text analysis on a single small desktop GPU (RTX
4060 Ti 16 GB). It re-ranks ~6-10 models on *their* data and *their* GPU so the pick is
defensible.

Full spec (source of truth, do not duplicate here): [`docs/design/spec.md`](../design/spec.md).

Milestones 0, 1, and 2 are **complete** and documented in [`current.md`](current.md): the gold
set + data-prep tooling (M0); the eval skeleton (compile-free: prebuilt Ollama, no
vLLM/flash-attn source build) + model prep / feasibility tooling (M1); and one real vLLM
backend + steady-state telemetry, validated end to end on `gemma-4-E4B-it-w4a16` on the RTX
4060 Ti (M2). Milestone 3 (core + depth/acceptance hardening) is delivered and documented in
[`current.md`](current.md). The Tier-1 screen is validated live (both tracks, VRAM gate; task ids
confirmed against lm-eval 0.4.12), and the sweep's live PID-attributed reclaim gate + shared
`isolate_cell` (sweep + screen + Optuna) are done and live-validated on a real vLLM sweep. The
only remaining M3 work is judge calibration (OQ2 + human ratings + live Ragas). 291 tests.

**Quick start:** `make demo-eval` runs the current pipeline end to end and idempotently
(venv -> gold set -> index -> prep-models -> run-eval + telemetry; needs a running Ollama).
The real vLLM path is `llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry` on a
CUDA host. See [`current.md`](current.md) for the per-command breakdown.

This file is the FORWARD plan only: residual M3 acceptance work and Milestone 4 robustness /
assisted-prep work. Delivered detail, including the audit fixes, lives in
[`current.md`](current.md).

## Approach: walking skeleton, then layer

The end-to-end vertical exists and is proven on a real backend (M1 skeleton on prebuilt
Ollama; M2 the vLLM launcher + telemetry validated on `gemma-4-E4B-it-w4a16`: retrieve ->
generate -> score -> ranked row + manifest with real tokens/sec + peak VRAM). **Milestone 3**
(core + depth) has landed (`current.md`); only its gated residuals remain (below). The other
forward layer is:

- **Milestone 4 -- robustness + ontology-assisted data prep.** Non-blocking improvements:
  embedding-aware VRAM estimates, a pre-launch contention guard, vLLM serving ergonomics, and
  an ontology-assisted corpus drafting pipeline.

## Milestone 3 — two-tier + scale + rigor (residual)

M3 core + depth/acceptance hardening are delivered ([`current.md`](current.md)). Only the
gated/external-dep items below remain; sequence numbers are stable workstream ids.

- **M3.8 Judge calibration close-out (carried from M0.5 / M1.5).** The scaffolding is built and
  unit-tested ([`current.md`](current.md)): `make calibration-run` / `run-eval --split calibration
  --worksheet --judge-model` pre-fills a worksheet with model answers + an UNGATED `judge_rating`,
  and `make calibration-score` computes rho + CI at the >= 0.6 gate. The judge is chosen (OQ2 -- a
  local Gemma-4 model, tiered by GPU class, family bias disclosed). Two external steps remain:
  live-validate the default Ragas path + UA prompts (ragas does not import in the current env) and
  collect the human `human_rating` column over the verified calibration split. Until rho clears the
  gate, the judge stays demoted and objective correctness ranks alone. A non-Gemma cross-check judge
  (Qwen3.6 / frontier) to quantify the Gemma-family delta is optional follow-up.

### Deferred until a consumer exists

- **map-reduce / multi-hop LangGraph eval templates (carried from M1.4).** The single-call
  RAG template is built and unit-tested; the map-reduce (long-doc) and multi-hop templates
  follow the same node-closure shape. They land WITH the text-analysis benchmark, whose
  scoring SCHEMA (what counts as recovering a trend / topic / narrative) is an open question
  to settle first. Building them now would be speculative against an undefined consumer.
- **Knowledge-graph / ontology RAG (GraphRAG).** A deliberate expansion beyond the v1 wedge
  (the spec chose "small custom + FAISS"). Feasible with langchain `LLMGraphTransformer` or
  LlamaIndex `PropertyGraphIndex` + a graph store (Neo4j / in-memory) + an extraction LLM;
  the ontology is a constrained node/relationship schema. Heavy deps + an extraction LLM in
  the pipeline, so it needs a scoped milestone + sign-off before building. (The langchain
  chunking strategies and flat + parent-child retrieval are already built in M1.)

## Milestone 4 — robustness + ontology-assisted data prep

M4.1-M4.3 are non-blocking improvements surfaced by the M2.4 real-model run
(gemma-4-E4B-it-w4a16 on the RTX 4060 Ti). M4.4 adds the requested advanced draft mode without
misrepresenting generated material as verified. None blocks M3; each is independently
shippable and unit-testable.

- **M4.1 Embedding-aware VRAM estimate.** `list-models` / the planner under-estimate w4a16
  (and other partial-quant) weights because `params_b x bpw` assumes the whole model is
  quantized; the measured E4B weights were 9.8 GiB vs the predicted ~4.2 GiB (Gemma's
  256k-token embedding stays high-precision). Read `vocab_size` / `hidden_size` / tied-embedding
  from each `config.json` and price the unquantized embedding + norms separately, so the fit
  verdict and Optuna's over-VRAM pruning (M3.4) stay honest for the 12B/27B candidates. Refines
  the `AvailabilityResolver` VRAM fit (M3.2).
- **M4.2 Pre-launch VRAM-contention guard.** The first M2.4 launch failed because Ollama held
  ~2.8 GB resident, so vLLM's startup free-memory check (`gpu-memory-utilization` x total)
  failed. Add a pre-flight that reports the resident users and either waits / evicts (e.g.
  Ollama keep-alive=0) or auto-derates `gpu-memory-utilization` to the actually-free fraction
  before serving. The single-run analogue of the M3.3 cross-cell VRAM-tolerance gate; share
  the NVML reader.
- **M4.3 vLLM serving knobs as CLI flags + a kernel preflight.** Surface `--max-model-len`
  and `--gpu-memory-utilization` on `run-eval` (today only via `--config`), and add a
  `build-vllm` self-check that builds the flashinfer sampling kernel once and pins a
  host-compatible flashinfer (or confirms the native sampler), so `launch_env` can re-enable
  the faster sampler where it compiles (it is defaulted off because flashinfer 0.6.x's
  `sampling.cuh` fails to build against newer CCCL/CUB on consumer sm_89).
- **M4.4 Ontology-assisted corpus gold-set drafting.** Implement the reserved
  `GOLDSET_MODE=draft` as a multi-stage pipeline over a supplied text directory, with either an
  internal OpenAI-compatible endpoint or a configured external LiteLLM provider. The pipeline
  must: (1) inventory and normalize supported documents while preserving offsets; (2) extract
  named entities, aliases/coreference, events, claims, and evidence-backed subject-relation-
  object facts; (3) induce a constrained ontology candidate with confidence and source spans;
  (4) sample coverage across entity types, relations, sections, and difficulty; (5) draft
  Ukrainian question/reference/span triples; (6) exact-ground, deduplicate, and reject
  unsupported or circular items; and (7) emit `verified: false` canonical drafts plus ontology,
  extraction, endpoint, prompt, model, cost, and document-hash provenance under
  `$DATA_DIR/prepare-goldset/<timestamp>/`. Keep extraction adapters modular so a Python-native
  NER/coreference model can be combined with LLM relation/ontology extraction. This is a data-
  preparation ontology, not a GraphRAG runtime or a new retrieval backend. Acceptance:
  injected unit tests cover every stage; a local fake endpoint proves the full flow; no draft
  scores until a human accepts it; and generated ontology/facts link back to exact evidence.
- **M4 acceptance:** the planner's predicted weights land within tolerance of the measured load
  on the gemma-4 w4a16 candidates; a run launches cleanly when another process holds VRAM; the
  vLLM knobs are settable without YAML; and M4.4 produces traceable unverified drafts from a
  nested corpus using both local and external endpoint adapters.

## Critical modules still to build (`src/llb/`)

The M3 modules now exist (`backends/resolver`, `executor/isolation`, `screen/public`,
`optimize/tuner`, `prep/frontier`, `scoring/aggregate` board, `scoring/judge.ragas_scorer`,
`board/`) -- see `current.md`. What remains genuinely unbuilt:

- `backends/` — the **llama.cpp launcher** (the third backend the resolver already routes to;
  `BackendLauncher` + the OpenAI client make it a drop-in alongside Ollama + vLLM).

Built already (documented in `current.md`): `RunConfig`, the `llb` Typer CLI, `rag/` store +
retrieval metrics, the single-call `eval/` graph, objective + semantic `scoring/`, `tracking/`
manifest, the minimal `executor/` runner + VRAM gate, `backends/`
base+client+Ollama+vLLM+hardware+prepare+planner+telemetry, and the M3 layer:
`backends/resolver`, `executor/isolation`, `optimize/tuner`, `screen/public`, `prep/frontier`,
the `scoring/aggregate` board + `scoring/judge.ragas_scorer`, and `board/`.

## Reuse (do not rebuild)

Ragas (RAG metrics + judge), FAISS, sentence-transformers, `openai` client (local backends),
litellm (frontier prep utils), Optuna, MLflow (local), LangGraph, DuckDB, Streamlit,
pynvml + psutil, lm-evaluation-harness-uk (INSAIT, Tier-1 public screen). Reuse public UA
datasets: SQuAD-uk + Belebele-uk (screen/baseline). Candidate seeds incl. MamayLM v2
12B/27B, Lapa, Gemma 3. All lightweight; no servers (no Celery/K8s).

## Verification (forward)

- **M3 (still forward):** the LIVE Ragas judge path + judge calibration rho/CI (M3.8). The
  Tier-1 lm-eval screen and the live PID-attributed sweep gate are validated live; the other
  delivered seams are unit-tested -- see `current.md`.
- **M4:** the embedding-aware estimate predicts measured weights within tolerance; the
  pre-launch guard handles resident VRAM users; and the ontology-assisted draft pipeline emits
  traceable, exact-grounded, unverified candidates from nested corpora.
- **AGENTS.md guardrails:** paths under `.data/llb/`; ASCII logs; confirm/create the MAX_JOBS
  helper before any vLLM source build (the canonical `max_jobs()` helper lands in M2).

## Worktree parallelization

M3 (core + depth) has landed. The small residual work proceeds in mostly independent lanes:
- **judge:** the M3.8 calibration close-out (no GPU, decision/human-gated);
- **data:** M4.4 ontology drafting and its review/provenance workflow;
- the **llama.cpp launcher** (shares the run path -- keep sequential with any run-path change);
- **Milestone 4** polish (M4.1 estimator + M4.2 VRAM guard touch the run path; M4.3 is CLI-only;
  M4.4 is an independent prep subpackage).

## NOT in scope (considered, deferred)

- Security/jailbreak, agentic, MCP/tooling, the 6 agent frameworks, multi-backend comparison,
  multi-vector-store, full GPU-class matrix, quality-per-watt.
- Rejected Codex pushbacks (ruled the other way): defer-Optuna-to-finalists,
  LangGraph-only-where-needed, drop-MLflow, drop-thermal-gate, defer-vLLM.
- Open questions resolved in M2: candidate-model list (OQ3) + vLLM repo ids verified, and the
  MAX_JOBS helper path (OQ6, canonical `max_jobs()` in `scripts/shared/common.sh`).
- Open questions resolved in M3.8: judge locality (OQ2) -- a LOCAL Gemma-4 judge, tiered by GPU
  class (12/16/32 GB), chosen for no corpus egress + reproducibility, with the Gemma-family
  self-preference bias disclosed (`current.md`); the residual is only the live Ragas UA
  validation + human ratings, not the model choice.
- Open questions still to resolve in-milestone: text-analysis scoring schema (before the
  deferred eval templates).
