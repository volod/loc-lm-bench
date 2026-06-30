# Evaluation Rigor

Evaluation rigor covers host-aware model selection, isolated execution, tuning discipline, public
screening, board ranking, and local judge integration. The common theme is preventing convenience
shortcuts from leaking into model rankings.

## Backend Resolution

`src/llb/backends/resolver.py` chooses a runnable backend for a logical model. A model can declare
one source or a per-backend `sources:` map. The resolver combines:

- availability probes for Hugging Face repos, Ollama tags, and GGUF sources;
- host fit planning from GPU VRAM plus system RAM;
- backend priority: vLLM, then Ollama, then llama.cpp;
- artifact-specific metadata such as quantization and architecture fields.

```bash
llb resolve-models
llb resolve-models --offline
llb resolve-models --context 8192
```

The design favors actual serveability over nominal parameter size. vLLM must fit its serving
context in GPU memory; Ollama and llama.cpp may offload layers to CPU RAM.

## Isolated Sweeps

`src/llb/executor/isolation.py` defines the reusable process-per-cell primitive used by sweeps,
public screens, and isolated Optuna trials. The primitive:

- snapshots baseline GPU state;
- runs one backend-owning cell in its own process;
- checks VRAM reclaim after the cell;
- distinguishes new leaked PIDs from tolerated baseline shifts when PID attribution is available;
- applies a capped thermal cooldown.

```bash
llb sweep --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl --sweep-id run1
llb sweep --sweep-id run1
```

The `run1` value is a user-chosen sweep name; it writes under `$DATA_DIR/sweep/run1/`.
Cells publish stable markers under `$DATA_DIR/sweep/<id>/cells/`. Marker keys ignore the display
run name and keep reproducibility-relevant config fields.
After backend resolution, the sweep command also checks local serving prerequisites before creating
cells: vLLM cells require the `vllm` executable, and llama.cpp cells require a project-managed or
PATH-visible `llama-server`. Missing binaries are reported as skips instead of failed benchmark
cells, while real cell execution errors are still recorded and counted as failures.

## Two-Stage Tuning

`src/llb/optimize/tuner.py` uses Optuna for RAG parameter search. Stage 1 searches only on the
`tuning` split. Stage 2 evaluates the winning config on the `final` split, and only that final run
is a leaderboard candidate.

The search space includes chunking strategy, chunk size, overlap fraction, `top_k`, retrieval mode,
child chunk size, and vLLM serving knobs where relevant. The embedder is pinned and is not a search
dimension.

```bash
llb tune --model llama3.2:3b --backend ollama --trials 30 --study uk1 \
  --goldset samples/goldsets/ua_squad_postedited_v1/goldset.jsonl
```

Over-context configs are pruned before model calls. Measured OOMs can also prune trials. Persistent
SQLite studies live under `$DATA_DIR/optuna/`.

## Public Screen

`src/llb/screen/public.py` adapts `lm-eval-harness-uk` to a running local endpoint. It keeps
logprob and generation tracks separate because their metrics are not comparable.

```bash
llb screen-public --model <model> --backend vllm --isolated
llb pipeline --top-n 2 --trials 20
```

`screen-public` writes coverage-aware reports under `$DATA_DIR/screen/`. `pipeline` reads those
reports, selects per-track finalists, tunes them on the private RAG split discipline, and prints
the final board.

## Board Ranking

`src/llb/scoring/aggregate.py` generalizes ranking beyond one row.

Ranking guards:

- average rank across shared quality signals rather than a silent arbitrary blend;
- bootstrap confidence intervals from per-case series;
- unresolved marks when adjacent CIs overlap;
- Pareto marks over quality, throughput, and VRAM;
- hard rejection of mixed tiers or incompatible judge cohorts;
- duplicate model-config rejection before ranking.

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
`mean_power_w`, `recall@k`, `MRR`); the logic lives in `src/llb/board/recommend.py` and the
matplotlib chart in `src/llb/board/charts.py` (guarded `[viz]` extra).

Picks:

- Recommended for this host: the highest-accuracy model that is feasible, Pareto-optimal, AND fits
  the GPU tier's VRAM budget with headroom (`peak_vram_mb <= 0.92 * total`). This is the
  HOST-ADAPTIVE pick -- on the same bundles a 16 GiB host recommends Lapa while a (simulated)
  24 GiB host recommends the larger MamayLM-27B, because the budget admits it. The pick also names
  its `best RAG top_k`, which is meaningful once the sweep gridded `top_k` (see the RAG-config grid
  in [`rag-core.md`](rag-core.md#sweep-rag-config-grid)): best-per-model dedup represents each model
  by its highest-scoring retrieval depth, so the recommendation answers `(model, top_k)`, not just
  model.
- Best RAG accuracy: rank-1 by objective/blended quality.
- Best efficiency: max `quality_per_watt` (the platform-matrix benchmark-efficiency axis).
- Fastest: max tokens/sec.

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

```bash
make recommend RECOMMEND_MIN_CASES=50          # detected host tier
make recommend RECOMMEND_GPU_GB=24             # would a 24 GiB box pick a bigger model?
```

Validated on the 16 GiB RTX 4060 Ti committed-goldset sweep (5 families, 82 final cases): MamayLM-27B
led accuracy (objective 0.546), Lapa was the recommended host pick (0.505, fits with headroom),
Qwen3.6 led efficiency (0.216 quality/W), and the Ukrainian-specialized models out-scored the
multilingual Mistral Small 3.1 (0.399) and Qwen baselines on Ukrainian RAG.

## Local Judge

`src/llb/scoring/judge.py` uses a local OpenAI-compatible endpoint for DeepEval G-Eval metrics.
The judge is gated by calibration rho. If it is not trusted, objective correctness ranks alone and
judge output remains diagnostic.
The DeepEval scorer separates empty-answer short-circuiting, injected-evaluator scoring, default
DeepEval scoring, and diagnostics merging so zero-valued candidate failures remain distinct from
local judge failures.

```bash
llb judge-experiment --judge-model <model> --judge-base-url <url>
llb judge-smoke --judge-model <model> --judge-base-url <url>
```

`judge-experiment` records prompts, fixed Ukrainian sanity cases, served-model metadata, and scores
under `$DATA_DIR/judge-experiment/<timestamp>/`. `judge-smoke` runs one strict-JSON grounded case
before a long judged run and exits non-zero with a reason if the local judge cannot produce a usable
score.

The local-judge choice is deliberate: corpus data should not leave the host by default. The tradeoff
is model-family bias when the judge shares lineage with candidate models. That bias is disclosed in
manifests and controlled by the calibration gate and judge-cohort guard.

## Frontier Prep Utilities

`src/llb/prep/frontier.py` contains GPU-free Litellm-backed utilities that emit unverified review
material:

- `prepare_goldset`: drafts question, answer, and exact source span triples from real documents;
- `prepare_synthetic_corpus`: generates synthetic documents with planted labels.

Both are injectable for tests and write provenance. A planter model must differ from the judge model
to avoid circular evaluation.
