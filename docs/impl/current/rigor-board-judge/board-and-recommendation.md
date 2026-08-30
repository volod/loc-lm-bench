# Screen, Board, And Recommendation

Part of the [Evaluation rigor](../rigor-board-judge.md) area of the [current implementation index](../../current.md).

## Public Screen

`src/llb/screen/public.py` adapts `lm-eval-harness-uk` to a running local endpoint. It keeps
logprob and generation tracks separate because their metrics are not comparable.

```bash
llb screen-public --model <model> --backend vllm --isolated
llb pipeline --top-n 2 --trials 20
```

`screen-public` writes coverage-aware reports under `$DATA_DIR/screen/`, each recording the example
cap (`--limit`) it was taken at so a smoke report is never mistaken for a full one. `--evict` and
`--wait` are the same two pre-launch VRAM-guard opt-ins `run-eval` has: a vLLM screen shares the
card with whatever ran before it, and the guard derates, waits, or unloads Ollama's residents rather
than dying inside the engine. `pipeline` reads those reports, selects per-track finalists, tunes
them on the private RAG split discipline, and prints the final board.

## Board Ranking

`src/llb/scoring/aggregate.py` generalizes ranking beyond one row.

Ranking guards:

- RAG base quality is an explicit 75% token-recall / 25% token-precision composite, with the
  unchanged token-F1 objective, found-rate, and mean completion length displayed beside it;
- average rank across shared quality signals rather than a silent undeclared blend;
- bootstrap confidence intervals from per-case series;
- unresolved marks when adjacent CIs overlap;
- Pareto marks over quality, throughput, and VRAM;
- hard rejection of mixed tiers or incompatible judge cohorts;
- duplicate model-config rejection before ranking.

The RAG policy lives in `src/llb/scoring/verbosity.py`. New run bundles persist both the aggregate
and aligned per-case `ranking_score`, so board CIs, Pareto quality, best-per-model selection,
tuning quality, and quality-per-watt all read the same score. The run table prints the formula
above the columns. Category tiers and legacy RAG bundles have no `ranking_score` and therefore
continue to use their own objective. The measured policy decision and rank changes are recorded
in [RAG core](../rag-core/scoring.md#headline-decomposition-and-declared-ranking-policy).

`src/llb/board/` loads run bundles and renders Streamlit views. Loading is split by concern:
`runs`, `categories`, `harnesses`, `prompt_systems`, and `io`. The board uses final private runs
for RAG leaderboards and separate sections for public screens, category tiers, harness comparisons,
and prompt-system comparisons.

```bash
make board
```

## Recommendation Summary

`llb recommend` (`make recommend`) distills a sweep into the few operator-facing picks a leaderboard
implies but does not state, plus a comparison chart. It reuses the board loaders
(`load_run_records` -> `best_per_model`) and the `aggregate` ranking (`rank_board`, `pareto_front`),
adding the host-efficiency + retrieval fields the `ModelResult` omits (`quality_per_watt`,
`mean_power_w`, `recall@k`, `MRR`); recommendation construction lives in
`src/llb/board/recommend/build.py` and the
matplotlib chart in `src/llb/board/charts.py` (guarded `[viz]` extra).

Picks:

- Recommended for this host: the highest-accuracy model that is feasible, Pareto-optimal, AND fits
  the GPU tier's VRAM budget with headroom (`peak_vram_mb <= 0.92 * total`). This is the
  HOST-ADAPTIVE pick -- on the same bundles a 16 GiB host recommends Lapa while a (simulated) 24 GiB
  host recommends the larger MamayLM-27B, because the budget admits it. The pick also names its
  `best RAG top_k`, which is meaningful once the sweep gridded `top_k` (see the RAG-config grid in
  [`rag-core.md`](../rag-core/persistence-and-execution.md#sweep-rag-config-grid)): best-per-model
  dedup represents each model by its highest-scoring retrieval depth, so the recommendation answers
  `(model, top_k)`, not just model.
- Best RAG accuracy: rank-1 by objective/blended quality.
- Best efficiency: max `quality_per_watt` (the platform-matrix benchmark-efficiency axis).
- Fastest: max tokens/sec.

The host pick is quality optimization SUBJECT TO host constraints that relax in order
(performance -> VRAM -> Pareto). `--min-tokens-per-s` (`RECOMMEND_MIN_TOK_S=`, 0 = off) adds a
good-enough-performance floor on top of the VRAM fit: the pick must clear the floor, and when it
does the summary names any higher-accuracy models that were traded away for speed, so the operator
sees exactly what the floor cost. All report prose is sourced from `board.recommend.*` prompt
templates (`prompts/templates/board/recommend/`) rather than inline literals, so the wording is
reviewable in files; `format_summary_md` only computes values and assembles the line list.

Only the dominant `(split, n_cases)` cohort is ranked. Comparing models is apples-to-apples only
within a shared split AND case count, so `select_cohort` keeps the cohort with the most models
(ties -> the larger `n_cases`, the more robust comparison) and lists the rest under an
`Excluded (off-cohort, not ranked): MODEL n=N` note rather than ranking a 20-case platform-matrix
row beside an 82-case sweep. `--min-cases` still pre-filters smoke bundles BEFORE the best-per-model
dedup so a 3-case manual run never shadows a full sweep; the cohort split is the backstop when
several real case counts coexist (the quickstart's `--min-cases 1` default would otherwise rank a
2-case bundle beside the 82-case cohort). `--gpu-gb` simulates another CUDA tier's VRAM budget for
the fit check. Outputs land at `$DATA_DIR/recommend/{summary.md,comparison.png}`, and
`quickstart-goldset` runs it as the final eval step.

Below the per-model picks the summary appends a `## RAG configuration detail (model x config)`
section: `load_config_cells` keeps every final-split `(model, RAG-config)` cell (best re-run per
cell, NOT collapsed to best-per-model), so a model swept at several retrieval depths shows all of
them. The table groups by model, marks each model's best config, and — when nothing was gridded —
appends a note pointing at `SWEEP_RAG_GRID`. This is the detailed proof that the winning
configuration is demonstrated per model, not assumed, complementing the best-per-model headline.

```bash
make recommend RECOMMEND_MIN_CASES=50          # detected host tier
make recommend RECOMMEND_GPU_GB=24             # would a 24 GiB box pick a bigger model?
make recommend-agent-profile                   # compose the whole agent configuration
```

`--agent-profile` (`make recommend-agent-profile`) additionally composes the host pick together with
the prompt system, adapter, context policy and order, retrieval knobs, and loop policy into ONE
`agent_profile.json` plus a rationale, where every value names the run that measured it and every
gap is visible as a gap. The command and its module layout live in
[the composed agent operating profile](../extended-workflows/agent-operating-profile.md).

Validated on the 16 GiB RTX 4060 Ti committed-goldset sweep (5 families, 82 final cases): MamayLM-27B
led accuracy (objective 0.546), Lapa was the recommended host pick (0.505, fits with headroom),
Qwen3.6 led efficiency (0.216 quality/W), and the Ukrainian-specialized models out-scored the
multilingual Mistral Small 3.1 (0.399) and Qwen baselines on Ukrainian RAG.
