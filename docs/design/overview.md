# Overview

loc-lm-bench selects the best open-weight LLM for Ukrainian RAG and text analysis on one
small desktop GPU. It re-ranks ~6-10 candidate models on YOUR corpus and YOUR hardware, so
the choice is reproducible and defensible.

## The problem
Public leaderboards measure general capability on someone else's tasks, quantization, and
unlimited VRAM. They do not test your corpus, your RAG stack, your 16 GB desktop, or your
task shape. The ranking may not transfer, and you cannot defend "why this model."

## Status quo
Picking models from public leaderboards only: no measurement on your own data, no
reproducible record, no defensible justification.

## The wedge (v1)
RAG + text analysis, scored on your own corpus, across ~6-10 models, on your GPU. Score =
quality (objective reference-answer correctness + a gated judge) plus tokens/sec and
does-it-fit-in-VRAM. Everything else (security, agentic, MCP, multi-backend comparison) is
deferred and written down, not built.

## Status
Milestone 0 (data prep) is built and tested; Milestones 1-3 are planned. See
[implementation/current.md](../implementation/current.md).

Full detail: [the design spec](../design.md).
