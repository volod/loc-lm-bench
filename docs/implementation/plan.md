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
4060 Ti (M2). Milestone 3's code layer is also built -- the AvailabilityResolver, the
hard-isolation sweep, two-stage Optuna, the Tier-1 screen adapter, the frontier prep utils, the
Ragas judge scorer, and the N-model average-rank/Pareto/CI board -- leaving only the judge
calibration close-out (OQ2 + human ratings) and the gold-set human verification as gated work.
220 tests.

**Quick start:** `make demo-eval` runs the current pipeline end to end and idempotently
(venv -> gold set -> index -> prep-models -> run-eval + telemetry; needs a running Ollama).
The real vLLM path is `llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry` on a
CUDA host. See [`current.md`](current.md) for the per-command breakdown.

This file is the FORWARD plan only -- the Milestone 3 close-outs (judge calibration, gold-set
verification) + the follow-ups on the shipped M3 modules, and Milestone 4 (post-M2 accuracy +
robustness polish). Delivered detail lives in [`current.md`](current.md).

## Approach: walking skeleton, then layer

The end-to-end vertical exists and is proven on a real backend (M1 skeleton on prebuilt
Ollama; M2 the vLLM launcher + telemetry validated on `gemma-4-E4B-it-w4a16`: retrieve ->
generate -> score -> ranked row + manifest with real tokens/sec + peak VRAM). The **Milestone 3**
layer has since landed -- a Tier-1 public screen, a hard-isolation multi-model sweep, two-stage
Optuna, the gated judge scorer, the frontier prep utils, and a Pareto + average-rank board
(`current.md`), leaving only the M3.8/M3.9 close-outs. The remaining forward layer:

- **Milestone 4 — post-M2 accuracy + robustness polish.** Non-blocking improvements the
  real-model run surfaced: an embedding-aware VRAM estimate, a pre-launch VRAM-contention
  guard, and ergonomics for the vLLM serving knobs.

## Milestone 3 — two-tier + scale + rigor

M3's code layer is **complete** and documented in [`current.md`](current.md) (the nine modules
and their CLIs: `resolve-models`, `sweep`, `tune`, `prepare-goldset`,
`prepare-synthetic-corpus`, `screen-public`, `board`). Only the gated close-outs and the
follow-ups below remain.

- **M3.8 Judge calibration close-out (carried from M0.5 / M1.5).** The gate, the pre-filled
  worksheet, and `scoring/judge.ragas_scorer` are built; the close-out is blocked on the judge
  choice (OQ2) + human ratings. (1) pick the judge (frontier API default, or MamayLM v2 27B
  local); (2) live-validate `ragas_scorer` against that endpoint (the default Ragas wiring is
  unverified -- ragas does not import in the current env); (3) `run-eval --split calibration
  --worksheet <f>`, add the human ratings; (4) `python -m llb.judge.calibration score
  --ratings <f>` and gate at rho >= 0.6 with a CI, else keep the judge demoted. GPU-independent.
- **M3.9 Gold-set human verification (carried from M0).** Flip `verified: true` on reviewed
  gold items (only verified items score Tier-2; the 250 HPLT/ua-squad items are still
  `verified: false`). The M3.5 `prepare-goldset` drafts + `prepare-synthetic-corpus` planted
  labels feed the review. (The screen-dataset wiring -- Belebele-uk -> logprob screen, SQuAD-uk
  -> generation screen -- is already done.)

### Follow-ups on the shipped M3 modules

- **End-to-end chain.** One command chaining screen -> finalist selection -> tuned private eval
  -> board (today each stage runs separately), incl. folding `screen-public` results into a
  finalist gate that feeds the Tier-2 sweep (M3.1).
- **Resolver per-source quant (M3.2).** A spec carries one `quant`, so vLLM-bf16 vs
  Ollama/GGUF-q4 of the same model is not modeled; add per-backend `sources` (incl. the
  commented-out GGUF/Ollama entries for the bf16 UA models in `samples/models_uk.yaml`) for a
  tighter GGUF fit.
- **Optuna depth (M3.4).** Sample backend serving knobs (`max_model_len`,
  `gpu_memory_utilization`); replace the token-estimate over-context prune with a measured fit;
  run each trial through the M3.3 process isolation instead of in-process.
- **Board polish (M3.7).** Add the gated judge column once M3.8 lands; render Tier-1 screen
  boards separately from Tier-2; rank "best per model" by average rank, not objective.
- **Prep grounding (M3.5).** Fuzzy/normalized span grounding (today exact-substring only, so a
  paraphrased quote is dropped); auto-wire the synthetic corpus into `build-index` + a scored run.
- **Live-path confirmation.** Confirm the lm-eval-harness-uk task ids (`belebele_ukr_Cyrl`,
  `squad_uk`) and the Ragas evaluate path against the real harness fork + judge endpoint (both
  are injected/unrun here).

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

## Milestone 4 — post-M2 accuracy + robustness polish

Non-blocking improvements surfaced by the M2.4 real-model run (gemma-4-E4B-it-w4a16 on the
RTX 4060 Ti). None blocks M3; each is independently shippable and unit-testable.

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
- **Acceptance:** the planner's predicted weights land within tolerance of the measured load
  on the gemma-4 w4a16 candidates; a run launches cleanly when another process holds VRAM; the
  vLLM knobs are settable without a YAML file.

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

- **M3 (still forward):** the LIVE lm-eval-harness-uk and Ragas paths (external deps + the
  judge choice, OQ2); the single screen -> finalists -> tuned eval -> board chain run end to
  end (the stages are individually validated -- see `current.md`).
- **M4:** the embedding-aware estimate predicts the measured weights within tolerance; the
  pre-launch guard lets a run start while another process holds VRAM.
- **AGENTS.md guardrails:** paths under `.data/llb/`; ASCII logs; confirm/create the MAX_JOBS
  helper before any vLLM source build (the canonical `max_jobs()` helper lands in M2).

## Worktree parallelization

The M3 lanes have landed (resolver, prep utils + judge scorer, Optuna, board, screen, and the
hard-isolation sweep). The remaining build work is small and mostly independent:
- the **llama.cpp launcher** (Lane A, shares the run path -- keep sequential with any run-path
  change);
- the **M3.8 calibration close-out** and **M3.9 gold-set verification** (no GPU, human-gated);
- **Milestone 4** polish (M4.1 estimator + M4.2 VRAM guard touch the run path; M4.3 is CLI-only).

## NOT in scope (considered, deferred)

- Security/jailbreak, agentic, MCP/tooling, the 6 agent frameworks, multi-backend comparison,
  multi-vector-store, full GPU-class matrix, quality-per-watt.
- Rejected Codex pushbacks (ruled the other way): defer-Optuna-to-finalists,
  LangGraph-only-where-needed, drop-MLflow, drop-thermal-gate, defer-vLLM.
- Open questions resolved in M2: candidate-model list (OQ3) + vLLM repo ids verified, and the
  MAX_JOBS helper path (OQ6, canonical `max_jobs()` in `scripts/shared/common.sh`).
- Open questions still to resolve in-milestone: judge locality + Ragas UA validation
  (OQ2, M3.8), text-analysis scoring schema (before the deferred eval templates).
