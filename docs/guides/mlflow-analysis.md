# Analyze loc-lm-bench runs with MLflow

MLflow is a review and comparison mirror. The canonical source of truth remains each
immutable run directory under `$DATA_DIR/run-eval/<timestamp>-<run-id>/`.

## Start the UI

Run at least one evaluation, then start the local UI:

    make demo-eval
    make mlflow

Open `http://127.0.0.1:5000`. Keep the terminal running while using the UI and press
Ctrl-C when finished. Override the bind address or port when needed:

    make mlflow MLFLOW_HOST=0.0.0.0 MLFLOW_PORT=5050

At startup, the command reconciles canonical run directories into the shared
`$DATA_DIR/mlflow/mlflow.db` store. A healthy message looks like:

    [mlflow] canonical sync: 2 created, 5 updated, 0 current, 0 failed

Later starts should report the records as `current` instead of duplicating them.

## Select the correct experiment

In the left experiment list, select **loc-lm-bench**. Do not select **Default**: MLflow
creates that system experiment automatically and this project intentionally leaves it empty.

Each run is named:

    <model> | <backend> | <canonical-run-id>

The final component matches `run_id` in the canonical `manifest.json`. The
`llb.canonical_run_id` tag provides the same join key for filters and automation.

## Recommended run-table columns

Use the Runs table column selector to show these fields:

| Field | Interpretation |
|---|---|
| `metrics.quality.objective_score` | Mean reference-answer token F1; primary objective score |
| `metrics.quality.reliability` | Fraction of cases with status `ok`; must remain near 1 |
| `metrics.quality.tokens_per_s` | Ranking throughput, using steady telemetry when available |
| `metrics.retrieval.recall_at_k` | Fraction of gold source spans retrieved in top-k |
| `metrics.retrieval.mrr` | Mean reciprocal rank of the first matching source span |
| `metrics.telemetry.steady_tokens_per_s` | Fixed-prompt throughput after warmup |
| `metrics.telemetry.peak_vram_mb` | Peak total GPU memory observed during telemetry |
| `metrics.telemetry.n_failed` | Failed fixed-prompt telemetry requests |
| `metrics.telemetry.tokens_per_char` | Generated tokenizer efficiency for Ukrainian text |
| `metrics.hardware.gpu_total_mb` | Total detected GPU memory for reproducibility |
| `metrics.cases.n` | Number of scored cases; compare scores only at matching scale |
| `metrics.judge.trusted` | `1` only when judge calibration passed its threshold |

## Parameters and tags

Parameters capture the effective benchmark configuration. Check these before comparing runs:

- `model`, `backend`, `embedding_model`
- `strategy`, `chunk_size`, `chunk_overlap`, `top_k`, `retrieval_mode`
- `max_tokens`, `temperature`, `max_model_len`, `quantization`, `dtype`
- `goldset_path`, `score_semantic`, `measure_telemetry`

Useful tags include:

- `llb.canonical_run_id`: joins MLflow to `$DATA_DIR/run-eval/.../manifest.json`
- `llb.model` and `llb.backend`: convenient UI filters
- `llb.created_at`: canonical UTC creation time
- `llb.gpus` and `llb.gpu_drivers`: hardware identity
- `llb.mirror_schema`: mirror record version used for automatic reconciliation

Example UI filters:

    tags.llb.backend = 'ollama'
    tags.llb.model = 'llama3.2:3b'
    metrics.retrieval.recall_at_k >= 0.8
    metrics.quality.reliability = 1

## Compare runs correctly

1. Filter to the same gold set and compatible retrieval/configuration parameters.
2. Reject or investigate runs with reliability below 1 or telemetry failures above 0.
3. Confirm retrieval recall meets the gate before attributing quality differences to the LLM.
4. Compare `quality.objective_score` first, then throughput and peak VRAM as tie-breakers.
5. Open selected runs in MLflow's Compare view to chart metrics side by side.
6. Use canonical artifacts for case-level diagnosis.

Do not make private-corpus model-selection claims from `make demo-eval`: it scores up to 20
cases from the committed post-edited public development fixture. Benchmark decisions require
your verified private final split, matching configurations, and enough cases for uncertainty
estimates.

## Inspect case-level artifacts

Open a run and select **Artifacts -> canonical**:

- `manifest.json`: effective config, aggregate metrics, retrieval, judge state, telemetry,
  environment, and canonical run id.
- `scores.parquet` or `scores.jsonl`: one row per case, including status, objective score,
  retrieval hit/rank, latency, token count, and answer preview.
- `vllm/`: backend logs when the run launched vLLM.

MLflow does not replace the case table with aggregate charts. Download `scores.parquet` for
DuckDB, pandas, or notebook analysis when individual errors need inspection.

## Troubleshooting

- **Experiment appears empty:** select `loc-lm-bench`, not `Default`.
- **Canonical runs are missing:** restart `make mlflow` and inspect the sync counts. Any
  nonzero `failed` count names the manifest that could not be mirrored.
- **`/mlflow/users/current` returns 404:** expected for this local UI because optional basic
  authentication is not enabled; experiment and run APIs continue to return 200.
- **No database exists:** run `make demo-eval` or another `run-eval` command first.
- **UI is on another port:** use the exact URL printed by `make mlflow`.
