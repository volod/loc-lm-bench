# Evaluation Rigor

Evaluation rigor covers host-aware model selection, isolated execution, tuning discipline, public
screening, board ranking, and local judge integration. The common theme is preventing convenience
shortcuts from leaking into model rankings.

This page is the AREA INDEX: each stage of that discipline lives in its own page under
[`rigor-board-judge/`](rigor-board-judge/).

| Page | What it answers |
| --- | --- |
| [Backend resolution, sweeps, and search](rigor-board-judge/tuning-and-search.md) | Which backend serves a model on this host, why sweeps run isolated, two-stage and multi-objective tuning, and the joint model + config search |
| [Screen, board, and recommendation](rigor-board-judge/board-and-recommendation.md) | The public screen, how the board ranks, and what the recommendation summary is allowed to claim |
| [Miss analysis and context probes](rigor-board-judge/diagnostics.md) | Why a run missed, where in the context an answer has to sit to be found, and whether a model abstains when the context cannot support an answer |
| [Ukrainian robustness and security adaptation](rigor-board-judge/robustness-benchmarks.md) | Query-robustness benchmarking under Ukrainian perturbations, and the security-lane adaptation |
| [Local judge and scorer policy](rigor-board-judge/judging.md) | The local judge, the scorer-policy seam, frontier judge agreement and cost, and the frontier prep utilities |
