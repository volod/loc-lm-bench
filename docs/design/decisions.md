# Engineering decisions

Lightweight-first, reuse over rebuild, no heavy services (no MLflow server, no vector-DB
server, no Ray / Celery / K8s). vLLM / torch are hardware-matched and installed separately.

- **Config search:** Optuna, two-stage. Tune backend + RAG params on a disjoint proxy split
  (embedding pinned, over-VRAM configs pruned), then score finalists on the full set.
- **Tracking:** MLflow in local file mode (no server) as a UI mirror. Canonical record =
  immutable manifest (JSON/YAML) + Parquet, written first.
- **Eval flows:** LangGraph for all per-case flows (one uniform pattern; clean path to the
  deferred agentic phase).
- **Isolation:** hard process isolation per run + VRAM-tolerance gate + capped thermal
  cooldown, for unbiased speed/VRAM measurement.
- **Prep utils:** `prepare-goldset` (draft-for-review) + `prepare-synthetic-corpus`
  (structured planted labels, planter != judge) via litellm.
- **Output:** the best (backend + RAG) config per model for THIS machine, recorded with the
  run environment, not presented as a durable optimum.

These survived two CEO/eng reviews and four Codex outside-voice passes. The full record,
including the cross-model resolutions, is in [the design spec](../design.md).
