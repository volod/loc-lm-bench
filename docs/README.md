# loc-lm-bench documentation

Entry point for the docs. Each area lives in its own subdirectory with its own index.

## Start here
- [Project README](../README.md) -- pitch, quick start, status.
- [Design](design/README.md) -- contents map into the design spec (problem, wedge, architecture, decisions, prior art).

## Design
- [Design contents](design/README.md) -- topic map: a gist + a jump-in link per topic.
- [Full design spec](design/spec.md) -- the complete source-of-truth document.

## Implementation
- [Current state](implementation/current.md) -- delivered Milestones 0-3 and their commands.
- [Forward plan](implementation/plan.md) -- the ordered roadmap (M4 -> M5 -> M6 + a human-only lane).

## Guides
- [Learning path](guides/learning-path.md) -- learn the whole stack from basics: a staged
  syllabus, curated links, and a time-boxed plan for a learner with basic knowledge.
- [LLM security learning path](guides/learning-path-security.md) -- extended threat modeling,
  jailbreak and prompt-injection testing, destructive-action controls, bias evaluation, and an
  eight-session practical syllabus.
- [Dev setup](guides/dev-setup.md) -- uv, venv, extras, make targets.
- [Data prep](guides/data-prep.md) -- gold set, ingestion, chunking, calibration commands.
- [Local judge experiments](guides/judge-experiments.md) -- DeepEval, Ukrainian prompts,
  local endpoints, smoke artifacts, and calibration.
- [Gold set from scratch](guides/goldset-from-scratch.md) -- published fixture, development
  imports, manual skeleton, and review rules.
- [Run the eval skeleton](guides/run-skeleton.md) -- Milestone 1: build-index -> run-eval.
- [Analyze runs with MLflow](guides/mlflow-analysis.md) - select the project experiment,
  compare metrics, and inspect canonical case artifacts.
- [vLLM backend + telemetry](guides/vllm-backend.md) -- Milestone 2: build-vllm -> run-eval on a real model.

## Inference
- [Config examples](inference/config-example.md) -- detect GPU tier, generate serve/run configs from [samples/config-example/](../samples/config-example/).

## Project rules
- [AGENTS.md](../AGENTS.md) -- guardrails for contributors and agents.
