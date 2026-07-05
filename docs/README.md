# loc-lm-bench documentation

Entry point for the docs. Each area lives in its own subdirectory with its own index; this page
routes you from a high-level topic to the specific document.

## Start here

- [Project README](../README.md) -- pitch, quick start, and product surface.
- [Choose a scenario](guides/README.md#choose-a-scenario) -- "I want to..." routing table into
  the guides (benchmark my PDFs, build a gold set, verify drafted data, compare backends, ...).
- [Learning path](guides/learning-path/learning-path.md) -- learn the whole stack from basics.

## Topic map

| Topic | Index | What lives there |
| --- | --- | --- |
| Design | [design/README.md](design/README.md) | Problem, wedge, architecture, decisions, prior art; [spec.md](design/spec.md) is the source of truth |
| Implementation | [impl/current.md](impl/current.md) | Agent-facing map of delivered behavior under `impl/current/` |
| Forward plan | [impl/plan.md](impl/plan.md) | Future engineering tasks only |
| Guides | [guides/README.md](guides/README.md) | Operator workflows and learning material, grouped by topic |
| Inference | [inference/config-example.md](inference/config-example.md) | GPU tier detection and generated serve/run configs |

## Guides by group

The [guides index](guides/README.md) carries the full annotated listing and the scenario router;
the groups are:

- [learning-path/](guides/README.md#learning-path----learn-the-stack) -- staged syllabi for the
  stack, LLM security, and the evaluation categories beyond RAG QA.
- [quickstart/](guides/README.md#quickstart----end-to-end-tracks) -- the end-to-end tracks
  (committed goldset, PDF corpus, any corpus) with wrapper and granular commands.
- [data-prep/](guides/README.md#data-prep----create-corpora-and-gold-sets) -- corpus and
  gold-set creation: local drafting, the end-to-end gold-set spine, and open-data drafting with
  external AI services.
- [human-tooling/](guides/README.md#human-tooling----the-human-gates) -- the human gates:
  data verification, judge calibration, judge experiments, and the why behind them.
- [benchmarking/](guides/README.md#benchmarking----run-compare-analyze) -- scored runs:
  RAG core, vLLM telemetry, platform matrix, retrieval comparison, prompt systems, MLflow
  analysis, composite headline, agentic harnesses.
- [development/](guides/README.md#development----contributor-setup) -- contributor environment
  setup.

## Project rules

- [AGENTS.md](../AGENTS.md) -- guardrails for contributors and agents.
