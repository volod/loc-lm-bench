# loc-lm-bench documentation

Entry point for the docs. The short topic docs under `design/` link back to the full
design spec for complete detail.

## Start here
- [Project README](../README.md) — pitch, quick start, status.
- [Design overview](design/overview.md) — the problem, the wedge, what we're building.

## Design (concepts, by topic)
- [Overview](design/overview.md) — problem, status quo, wedge, status.
- [Architecture](design/architecture.md) — modules, two-tier eval, sequential isolation.
- [Evaluation & scoring](design/evaluation.md) — gold set, source-span labels, gated judge, ranking.
- [Engineering decisions](design/decisions.md) — Optuna, MLflow, LangGraph, isolation/thermal, prep utils.
- [Prior art](design/prior-art.md) — lang-uk / INSAIT / MamayLM, and what we reuse.
- [Full design spec](design.md) — the complete source-of-truth document.

## Implementation
- [Current state](implementation/current.md) — what's built today and how to run it.
- [Forward plan](implementation/plan.md) — Milestones 1-3.

## Guides
- [Dev setup](guides/dev-setup.md) — uv, venv, extras, make targets.
- [Data prep](guides/data-prep.md) — gold set, ingestion, chunking, calibration commands.

## Project rules
- [AGENTS.md](../AGENTS.md) — guardrails for contributors and agents.
