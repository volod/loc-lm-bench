# Architecture

The eval harness is backend-agnostic: it talks OpenAI-compatible HTTP, so the same
eval / RAG / judge code runs against any backend. Only thin launchers and a resolver are
backend-specific.

## Two-tier evaluation
- **Tier 1 — public screen** (`screen-public`): INSAIT's lm-evaluation-harness-uk driven
  over the launched endpoint, with logprob-capable (vLLM) vs generation-only (Ollama)
  tracks that are never cross-ranked. Cheaply narrows the candidate field.
- **Tier 2 — private corpus eval**: RAG + text analysis on your gold set, run only on the
  screen's survivors, with two-stage Optuna config search.

## Modules (`src/llb/`)
- `executor/` — plain-Python sequential run executor; one process per (model, config);
  VRAM-tolerance gate + capped thermal cooldown; resumable.
- `backends/` — BackendLauncher (Ollama / vLLM / llama.cpp) + telemetry hook + resolver.
- `rag/` — chunking, pinned embedding, FAISS, retrieval metrics.
- `eval/` — LangGraph eval-flow templates.
- `scoring/` — reference answer-correctness + gated Ragas judge + aggregate.
- `optimize/` — Optuna two-stage.
- `prep/` — gold-set + synthetic-corpus utilities.
- `tracking/` — canonical manifest + Parquet; MLflow mirror.
- `board/` — thin Streamlit leaderboard.
- One canonical `RunConfig` flows eval -> scoring -> manifest.

## Sequential isolation (unbiased measurement)
One backend process per run; between runs, poll until VRAM returns to baseline (tolerance
band), then a capped thermal cooldown. A real leak aborts loudly. Every measurement comes
off a clean, thermally-comparable baseline.

Full detail: [the design spec](../design.md).
