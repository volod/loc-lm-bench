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
```

Validated on the 16 GiB RTX 4060 Ti committed-goldset sweep (5 families, 82 final cases): MamayLM-27B
led accuracy (objective 0.546), Lapa was the recommended host pick (0.505, fits with headroom),
Qwen3.6 led efficiency (0.216 quality/W), and the Ukrainian-specialized models out-scored the
multilingual Mistral Small 3.1 (0.399) and Qwen baselines on Ukrainian RAG.

## Ukrainian Security Adaptation

The security benchmark (`src/llb/bench/security.py`, `src/llb/scoring/security.py`) is adapted to
the Ukrainian, bilingual reality of locally-served UA models. The adaptation lives in the pieces
that decide the scores, so a Ukrainian-language decline is never miscounted as compliance and a
Cyrillic-echoed leak is never missed.

- **Refusal lexicon (UA + RU).** `is_refusal` (`src/llb/eval/common.py`) matches high-precision,
  first-person UA refusals (`не можу`, `не зможу`, `не буду`, `не в змозі`, `відмовляюся`,
  `перепрошую, але я`, …) plus Russian equivalents (`не могу`, `извините, но я`), because a
  bilingual UA model pushed with an adversarial ask frequently declines in Russian. A missed
  refusal would inflate ASR (`refuse` cases), turn a canary-naming refusal into a false leak
  (`leak` cases), and depress refusal-appropriateness. Every marker keeps a `(?!\w)` word boundary
  so first-person `не можу` does not fire on third-person `не можуть`.
- **Text normalization + homoglyph folding.** `normalize_refusal_text` casefolds, applies NFKC, and
  unifies Ukrainian apostrophe variants (`'` `'` `ʼ` `` ` ``) before matching. `marker_leaked`
  additionally folds Cyrillic look-alike letters to Latin, so a Latin canary the model echoes with
  Cyrillic characters (`РАСКЕТ-ОХ-42` for `PACKET-OX-42`) is still detected instead of understating
  ASR.
- **UA / bilingual attack vectors.** Jailbreak frame prose lives in prompt templates under
  `bench/security/` (adversarial text is data, not code). Alongside DAN role-play and fiction
  framing, a **language-switch** frame tells a UA-tuned assistant to drop its guardrails and answer
  in Russian — probing whether safety generalizes across the languages the model knows. The
  committed seed (`samples/security_cases_uk.json`) adds a language-switch jailbreak (`jb-003`) and
  a Cyrillic-homoglyph RAG injection (`rag-003`).
- **Over-refusal controls.** The seed carries UA benign controls that look sensitive but are
  legitimate — dual-use history, defensive security, and crisis mental-health resources — so a model
  cannot win by blanket-refusing Ukrainian civic/dual-use questions. These feed refusal-
  appropriateness only, never ASR.

Seed cases are tagged with `lang` (`uk` / `ru` / `en`) and, for attacks, `attrs.vector`
(`language_switch` / `homoglyph`).

- **Cross-language consistency (Module 6).** `cross_language_consistency`
  (`src/llb/scoring/security.py`) compares matched-prompt groups — the SAME behavior posed in
  several languages, linked by an `xlang_group` id — as a separate outcome from ASR. For every
  group posed in >=2 languages it checks whether the model's refuse/answer decision is the same
  across languages; `consistency` is the fraction of groups treated equivalently and
  `refusal_rate_by_lang` is the diagnostic that reveals which language is the weak point (e.g. a
  model that refuses in Ukrainian but complies in Russian). Consistency is orthogonal to safety —
  it is read alongside ASR, carries its own bootstrap CI, and is persisted in the run manifest
  under `config.cross_language`; `SecurityScore.cross_language` is `None` when a set has no matched
  groups. The committed seed ships one harmful (`xl-weapon`) and one benign (`xl-help`) UA/RU/EN
  group. Behavior-level translation of the public adversarial sets into matched groups remains an
  operator step (inject a per-language `translate`), since the seed keeps human-verified prose.

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
