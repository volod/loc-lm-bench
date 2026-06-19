# loc-lm-bench v1 — Implementation Plan

## Context

loc-lm-bench is a production-grade, reproducible internal tool that selects the best
open-weight LLM for Ukrainian RAG + text analysis on a single small desktop GPU (RTX
4060 Ti 16 GB). Today the user picks models from public leaderboards that don't transfer
to their corpus, hardware, or task; this tool re-ranks ~6-10 models on *their* data and
*their* GPU so the pick is defensible.

Full spec (source of truth, do not duplicate here): [`docs/design.md`](design.md).
It went through /office-hours, /plan-ceo-review (HOLD SCOPE), and two /plan-eng-review
passes, plus four Codex outside-voice passes. This plan is the build sequence derived
from it.

## Recommended approach: walking skeleton, then layer

Do NOT build all ~12 modules before an eval runs. Build a thin end-to-end vertical
first, then add layers, each independently shippable and tested.

- **Milestone 0 — Assignment (no code).** Find existed and form ~30 Ukrainian RAG items
  (question + SOURCE-SPAN labels + reference answer) and calibrate the judge (Ragas
  config, UA-localized prompts) against your own ratings. If Spearman rho < 0.6, the
  judge is untrusted — learn that before building.
- **Milestone 1 — CUDA-free skeleton.** One Ollama model, fixed config, pinned
  embedding -> FAISS -> one LangGraph RAG graph -> reference-answer-correctness + gated
  judge -> canonical manifest + Parquet (+ MLflow mirror) -> `run-eval` prints one
  ranked row, wrapped by the plain-Python executor + a basic VRAM gate. Seed the RAG
  loop with SQuAD-uk (public, span-labeled UA QA) so the skeleton runs before the
  hand-built gold set exists.
- **Milestone 2 — one real backend + telemetry.** Add the vLLM (or llama.cpp) launcher
  and the real telemetry hook (tokens/sec steady-state, served context, peak VRAM) to
  validate CUDA / HF-loading / tokenizer assumptions before scaling. Resolve the first
  concrete candidate-model list here (it drives launcher priority).
- **Milestone 3 — two-tier + scale + rigor.** Tier-1 public screen (`screen-public`:
  lm-eval-harness-uk driven against the launched endpoint, logprob MCQ tasks on vLLM)
  narrows candidates; Tier-2 multi-model sweep on survivors, backend resolver
  (vLLM>Ollama>llama.cpp), hard process isolation + VRAM-tolerance + capped thermal
  cooldown, two-stage Optuna (disjoint gold partitions, embedding pinned), prep utils,
  Pareto + average-rank board with confidence intervals.

## Critical modules (`src/llb/`)

- `executor/` — plain-Python sequential run executor; VRAM-tolerance gate + capped
  thermal cooldown; resume (skip completed cells). Embed an ASCII diagram of the gate loop.
- `backends/` — `BackendLauncher` (Ollama, vLLM, llama.cpp) + per-backend telemetry hook
  + `AvailabilityResolver`. The adapter normalizes per-backend quirks (stop seqs, context
  truncation, token accounting, errors), not just telemetry.
- `rag/` — chunker, pinned embedding, FAISS index, retrieval metrics (recall@k/MRR by
  SOURCE-SPAN overlap — validates the embedding, not the model).
- `eval/` — ~3 DRY LangGraph templates (single-call, map-reduce, multi-hop).
- `scoring/` — reference answer-correctness (objective, ranks models) + Ragas gated judge
  + `aggregate` (average-rank headline + weighted-blend view, both over Pareto).
- `screen/` — Tier-1 public-screen adapter: launch the endpoint, drive lm-eval-harness-uk
  in local-completions mode, parse results JSON, record per-task logprob coverage so the
  screen is never silently partial.
- `optimize/` — Optuna two-stage (proxy tuning split, persistent SQLite, prune over-VRAM).
- `prep/` — `prepare-goldset` (draft-for-review) + `prepare-synthetic-corpus` (structured
  labels, planter!=judge) via litellm.
- `tracking/` — canonical immutable manifest (JSON/YAML) + Parquet (written first); MLflow
  local mirror.
- `board/` — thin Streamlit (Pareto + best-config + CIs).
- `cli.py` — Typer commands.
- One canonical `RunConfig` Pydantic model flows eval -> scoring -> manifest.

## Reuse (do not rebuild)

Ragas (RAG metrics + judge), FAISS, sentence-transformers, `openai` client (local
backends), litellm (frontier prep utils), Optuna, MLflow (local), LangGraph, DuckDB,
Streamlit, pynvml + psutil, lm-evaluation-harness-uk (INSAIT, Tier-1 public screen).
Reuse public UA datasets: SQuAD-uk + Belebele-uk (RAG seed/baseline). Candidate seeds
incl. MamayLM v2 12B/27B, Lapa, Gemma 3. All lightweight; no servers (no Celery/K8s).

## Verification

- Milestone 1 acceptance: `run-eval` on one Ollama model produces a ranked row with
  reference-correctness + gated-judge scores + a written manifest, CUDA-free.
- Unit tests alongside each module (full test plan is a gstack eng-review artifact,
  consumed by /qa): VRAM/thermal gate (mock NVML), resolver priority, recall@k/MRR on
  fixtures, typed graph failures, Optuna over-VRAM pruning, manifest-before-MLflow,
  prep-util provenance, screen-adapter task-coverage, average-rank aggregator.
- Critical E2E/eval: walking-skeleton end-to-end; resume-after-kill mid-sweep; judge
  calibration gate (rho>=0.6 with CI, else demote to objective-only ranking).
- AGENTS.md guardrails: paths under `.data/llb/`; ASCII logs; confirm/create the MAX_JOBS
  helper before any vLLM source build (Open Question 6).

## Worktree parallelization

After Milestone 1 (sequential foundation), these lanes parallelize:
- Lane A: `backends/` launchers + resolver  (touches run path)
- Lane B: `prep/` utils (litellm, no GPU)  — fully independent
- Lane C: `optimize/` Optuna  — depends on A + `scoring/`
- Lane D: `board/` Streamlit  — depends on `tracking/` manifest schema frozen
- Lane E: `screen/` (lm-eval-harness adapter) — depends on A (launched endpoint)
Launch B + D in parallel with A. C and E after A (+ scoring for C). `executor/` thermal
hardening shares the run path with A — keep sequential to avoid executor merge conflicts.

## NOT in scope (considered, deferred)

- Security/jailbreak, agentic, MCP/tooling, the 6 agent frameworks, multi-backend
  comparison, multi-vector-store, full GPU-class matrix, quality-per-watt.
- Rejected Codex pushbacks (you ruled the other way): defer-Optuna-to-finalists,
  LangGraph-only-where-needed, drop-MLflow, drop-thermal-gate, defer-vLLM.
- Open: first candidate-model list (OQ3), judge locality + Ragas UA validation (OQ2),
  text-analysis scoring schema, MAX_JOBS helper path (OQ6).

- **CROSS-MODEL:** Codex re-raised already-ruled points (Optuna-in-v1, LangGraph-all-flows, MLflow, thermal gate) — user decisions stand. Accepted new findings: ranking on answer-correctness (not retrieval recall), source-span gold labels, grow gold set + CIs. Prior-art re-review: screen wiring kept on the launched endpoint with logprob-capable vs generation-only tracks (never cross-ranked); average-rank refined (never mix Tier-1/Tier-2 metrics in one rank, mark CI-overlapping flips unresolved, Pareto+CIs co-headline).
