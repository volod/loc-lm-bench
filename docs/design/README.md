# Design

The full, approved design lives in one document -- [`spec.md`](spec.md) -- which is the
single source of truth. This page is the contents map: a one-line gist per topic with a
link into the relevant section of the spec, so you can jump in without re-reading the whole
thing. (The earlier per-topic summary files were folded back into the spec to remove
duplication; this index replaces them.)

## What it is

loc-lm-bench is an internal decision machine: it re-ranks ~6-10 open-weight LLMs on *your*
Ukrainian corpus, *your* task, and *your* GPU, so the pick is reproducible and defensible
-- not a transfer of someone else's public-leaderboard ranking. See the design rationale.

## Contents

- **Overview** -- the problem, the status quo ("public leaderboards only"), and the narrowest
  wedge (RAG + text analysis on the user's corpus). Read
  [Problem Statement](spec.md#problem-statement) and
  [Target User & Narrowest Wedge](spec.md#target-user--narrowest-wedge).
- **Premises** -- the load-bearing assumptions: per-model backend resolution, the GATED
  LLM judge, the pinned/validated embedding, gold set as the real work, reuse over rebuild.
  [Premises](spec.md#premises).
- **Architecture & approach** -- Approach A (thin decision loop) built with Approach B's two
  seams (BackendLauncher + reproducibility manifest), and the v1 build shape.
  [Recommended Approach](spec.md#recommended-approach).
- **Evaluation & scoring** -- the two scoring layers (retrieval validation vs generation
  ranking), source-span gold labels, the gated judge, and the Pareto + average-rank output.
  [Eng-review resolutions](spec.md#eng-review-resolutions-2026-06-19) and the
  [v1 build shape](spec.md#recommended-approach).
- **Engineering decisions** -- the lightweight-first tool stack, the sequential-execution
  correctness contract, two-stage Optuna, MLflow-as-mirror, LangGraph for all eval flows,
  and the prep utilities.
  [v1 Engineering Decisions](spec.md#v1-engineering-decisions-ceo-review-2026-06-19).
- **Prior art** -- how lang-uk / INSAIT / MamayLM are consumed (two-tier eval, average-rank,
  dataset reuse) without duplicating them.
  [Prior-Art Integration](spec.md#prior-art-integration-lang-uk--insait--mamaylm-2026-06-19).
- **Open questions & success criteria** -- what's still undecided and how we know v1 worked.
  [Open Questions](spec.md#open-questions), [Success Criteria](spec.md#success-criteria).
- **Appendix** -- the considered tool stack, VRAM/context planning formulas, the candidate
  model matrix, and the full (deferred) benchmark taxonomy.
  [Reference Material](spec.md#appendix-reference-material-consolidated-from-initial-draft-specs).

## Status

Approved (via /office-hours, /plan-ceo-review HOLD SCOPE, two /plan-eng-review passes, and
four Codex outside-voice passes). Build progress is tracked in
[../impl/current.md](../impl/current.md) and
[../impl/plan.md](../impl/plan.md).
