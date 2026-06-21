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
4060 Ti (M2). 165 tests.

**Quick start:** `make demo-eval` runs the current pipeline end to end and idempotently
(venv -> gold set -> index -> prep-models -> run-eval + telemetry; needs a running Ollama).
The real vLLM path is `llb run-eval --config samples/run_config_vllm_uk.yaml --telemetry` on a
CUDA host. See [`current.md`](current.md) for the per-command breakdown.

This file is the FORWARD plan only -- Milestone 3 (two-tier screen, scale, rigor, board) and
Milestone 4 (post-M2 accuracy + robustness polish) -- plus the few M0/M1 residuals that are
blocked on an undecided judge or human input, each folded into the milestone that unblocks it.
Completed detail moves to `current.md` as it lands.

## Approach: walking skeleton, then layer

The end-to-end vertical exists and is proven on a real backend (M1 skeleton on prebuilt
Ollama; M2 the vLLM launcher + telemetry validated on `gemma-4-E4B-it-w4a16`: retrieve ->
generate -> score -> ranked row + manifest with real tokens/sec + peak VRAM). Now add layers,
each independently shippable and tested:

- **Milestone 3 — two-tier + scale + rigor.** A Tier-1 public screen narrows candidates;
  a Tier-2 multi-model sweep on survivors with hard process isolation, two-stage Optuna, the
  gated judge, prep utils, and a Pareto + average-rank board.
- **Milestone 4 — post-M2 accuracy + robustness polish.** Non-blocking improvements the
  real-model run surfaced: an embedding-aware VRAM estimate, a pre-launch VRAM-contention
  guard, and ergonomics for the vLLM serving knobs.

## Milestone 3 — two-tier + scale + rigor

- **M3.1 `screen/` (Tier-1 public screen).** Drive lm-eval-harness-uk via `local-completions`
  against the launched OpenAI-compatible endpoint; logprob-capable (vLLM) vs generation-only
  (Ollama) tracks, NEVER cross-ranked; record per-task coverage so the screen is never
  silently partial. New CLI: `screen-public`.
- **M3.2 `backends/AvailabilityResolver`.** HF Hub + Ollama library + GGUF discovery with the
  vLLM>Ollama>llama.cpp priority and VRAM fit. (The feasibility planner already estimates the
  VRAM/RAM fit; the resolver adds discovery + the backend-priority decision.)
- **M3.3 `executor/` hard isolation.** On top of the existing minimal sequential runner +
  basic VRAM gate: one process per (model, config), kill + VRAM-tolerance gate + capped
  thermal cooldown; resumable (skip completed cells); abort loudly on `VramNotReclaimed`;
  record temp/clocks/power.
- **M3.4 `optimize/` two-stage Optuna.** Stage 1 tunes backend + RAG params on the disjoint
  tuning split (embedding PINNED, over-VRAM configs pruned, persistent SQLite); stage 2 scores
  the winning config on the full final split. Only the stage-2 run is the leaderboard entry.
  The RAG search space already exists (built in M1): chunking strategy
  {fixed, sentence, recursive, markdown, semantic} x chunk_size/overlap x top_k x
  retrieval_mode {flat, parent_child} x child_chunk_size.
- **M3.5 `prep/` frontier utils.** `prepare-goldset` (draft-for-review triples) and
  `prepare-synthetic-corpus` (structured planted labels, planter != judge) via litellm. No
  GPU -- fully independent lane.
- **M3.6 `scoring/aggregate` rigor.** Generalize the single-model ranker to N models:
  average-rank headline + weighted-blend view + Pareto + confidence intervals; never mix
  Tier-1 screen and Tier-2 private metrics in one rank; mark CI-overlapping flips
  "statistically unresolved".
- **M3.7 `board/` Streamlit.** Thin page: rank + best-config-per-model + CIs. MLflow UI covers
  deep inspection.
- **M3.8 Judge scorer + calibration close-out (carried from M0.5 / M1.5).** The trust GATE
  and a pre-filled calibration worksheet (`run-eval --split calibration --worksheet`) already
  exist. What remains is blocked on choosing the judge (OQ2) and producing human ratings:
  (1) pick the judge (frontier API default, or MamayLM v2 27B as a local candidate);
  (2) implement `scoring/judge.ragas_scorer` (Ragas faithfulness + answer-relevancy) with
  UA-localized metric prompts; (3) `run-eval --split calibration --worksheet <f>`, then add
  the human ratings; (4) `python -m llb.judge.calibration score --ratings <f>` and gate at
  rho >= 0.6 with a CI -- else keep the judge demoted and let objective + semantic correctness
  rank alone. GPU-independent: can run in parallel with the launcher work.
- **M3.9 Gold-set human verification + screen datasets (carried from M0).** Flip
  `verified: true` on reviewed gold items (only verified items score models in Tier-2; the
  250 HPLT/ua-squad items are currently `verified: false`). Wire Belebele-uk -- which is MCQ,
  not span-labeled -- into the Tier-1 SCREEN alongside SQuAD-uk, NOT into the source-span
  gold set.
- **Acceptance:** screen -> finalists -> tuned private eval -> Pareto/average-rank board;
  reproducible manifests; resume-after-kill works; the judge is calibrated-or-demoted on
  record.

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

- `backends/` — llama.cpp launcher + `AvailabilityResolver` (M3.2). (The base `BackendLauncher`,
  the OpenAI-compatible client, the Ollama + vLLM launchers, the telemetry hook, hardware/RAM
  detection, model prepare, and the feasibility planner already exist.)
- `executor/` — hard isolation (one process per cell, VRAM-tolerance + capped thermal cooldown,
  resume) on top of the existing minimal runner + basic VRAM gate.
- `screen/` — Tier-1 lm-eval-harness-uk adapter (local-completions, per-task logprob coverage).
- `optimize/` — two-stage Optuna (proxy tuning split, persistent SQLite, prune over-VRAM).
- `prep/` — `prepare-goldset` + `prepare-synthetic-corpus` via litellm.
- `scoring/` — Ragas judge scorer (the gate already exists) + average-rank + CIs (the objective
  + semantic ranker already exists).
- `board/` — thin Streamlit (Pareto + best-config + CIs).

Built already (documented in `current.md`): `RunConfig`, the `llb` Typer CLI, `rag/` store +
retrieval metrics, the single-call `eval/` graph, objective + semantic `scoring/`, `tracking/`
manifest, the minimal `executor/` runner + VRAM gate, and `backends/`
base+client+Ollama+vLLM+hardware+prepare+planner+telemetry.

## Reuse (do not rebuild)

Ragas (RAG metrics + judge), FAISS, sentence-transformers, `openai` client (local backends),
litellm (frontier prep utils), Optuna, MLflow (local), LangGraph, DuckDB, Streamlit,
pynvml + psutil, lm-evaluation-harness-uk (INSAIT, Tier-1 public screen). Reuse public UA
datasets: SQuAD-uk + Belebele-uk (screen/baseline). Candidate seeds incl. MamayLM v2
12B/27B, Lapa, Gemma 3. All lightweight; no servers (no Celery/K8s).

## Verification (forward)

- **M3 unit tests:** resolver priority, screen-adapter task-coverage, Optuna over-VRAM
  pruning, average-rank aggregator, judge calibration gate (rho>=0.6 with CI, else demote),
  prep-util provenance.
- **M4:** the embedding-aware estimate predicts the measured weights within tolerance; the
  pre-launch guard lets a run start while another process holds VRAM.
- **Critical E2E:** resume-after-kill mid-sweep; screen -> finalists -> tuned eval -> board.
- **AGENTS.md guardrails:** paths under `.data/llb/`; ASCII logs; confirm/create the MAX_JOBS
  helper before any vLLM source build (the canonical `max_jobs()` helper lands in M2).

## Worktree parallelization

The M2 launcher has landed (it touches the run path); these lanes parallelize:
- Lane A: `backends/` launchers + resolver (touches the run path)
- Lane B: `prep/` utils (litellm, no GPU) -- fully independent; also M3.8 judge scorer (no GPU)
- Lane C: `optimize/` Optuna -- depends on A + `scoring/`
- Lane D: `board/` Streamlit -- depends on the `tracking/` manifest schema (frozen)
- Lane E: `screen/` (lm-eval-harness adapter) -- depends on A (launched endpoint)
Launch B + D in parallel with A; C and E after A (+ scoring for C). `executor/` hard-isolation
shares the run path with A -- keep sequential to avoid merge conflicts.

## NOT in scope (considered, deferred)

- Security/jailbreak, agentic, MCP/tooling, the 6 agent frameworks, multi-backend comparison,
  multi-vector-store, full GPU-class matrix, quality-per-watt.
- Rejected Codex pushbacks (ruled the other way): defer-Optuna-to-finalists,
  LangGraph-only-where-needed, drop-MLflow, drop-thermal-gate, defer-vLLM.
- Open questions resolved in M2: candidate-model list (OQ3) + vLLM repo ids verified, and the
  MAX_JOBS helper path (OQ6, canonical `max_jobs()` in `scripts/shared/common.sh`).
- Open questions still to resolve in-milestone: judge locality + Ragas UA validation
  (OQ2, M3.8), text-analysis scoring schema (before the deferred eval templates).
