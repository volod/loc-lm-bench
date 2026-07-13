# Local Model Knowledge-Cutoff Benchmark

## Delivered behavior

`llb bench-knowledge-cutoff` and `make bench-knowledge-cutoff` run a non-interactive effective
knowledge-cutoff benchmark against local Ollama, vLLM, llama.cpp, or an already running local
OpenAI-compatible endpoint. The implementation lives in
`src/llb/bench/knowledge_cutoff/`, with Typer wiring in
`src/llb/cli/bench/knowledge_cutoff.py` and Make orchestration in `make/eval.mk`.

The loader accepts an operator JSONL file or the `events`/`train` configuration of
`apoorvumang/knowledge-cutoff-benchmark`. Moving Hugging Face revisions are resolved to an exact
commit before loading; a supplied 40-character commit is used directly. Local files are recorded
by SHA-256. Dataset imports are lazy behind the `cutoff` extra.

The benchmark uses project-native methodology:

- stable per-event answer permutation removes source answer-position bias;
- prompts disclose neither the current date nor that this is a recency test;
- low/medium-predictability real events alone feed the monthly curve;
- deterministic letter parsing produces correct/incorrect/abstain evidence;
- a seeded Optuna study fits a monotone logistic curve with a fixed four-choice chance floor and
  learned ceiling, cutoff midpoint, and scale;
- living-person and fake-event rows stay outside the fit and expose over-prediction/confabulation;
- raw threshold landmarks remain in the report beside the primary Optuna estimate.

Canonical output is
`$DATA_DIR/knowledge-cutoff/<run_timestamp>/{manifest.json,scores.jsonl,report.json,report.md}`.
Reports join the manifest and scores in the same atomic staging transaction before the shared
MLflow mirror runs, and `tracking/mlflow.py` includes `report.*` among canonical mirrored artifacts
for any run that has them.

## Validation

The focused fake-completion suite in
`tests/llb/bench/knowledge_cutoff/test_knowledge_cutoff.py` covers validation, local and injected
Hugging Face loading, exact revision provenance, prompt balancing/date blindness, parser variants,
curve/control aggregation, seeded fitting, smoke sampling, CLI registration, and a persisted
no-network/no-GPU run.

The live Hugging Face check on 2026-07-13 loaded 330 events spanning 2024-01 through 2026-06 and
resolved the dataset to commit `70ac8333a6fdd742f73f85a02a303aafba84617e`.

## Attribution boundary

The idea and dataset source were inspired by Apoorv Saxena's
[`knowledge-cutoff`](https://github.com/apoorvumang/knowledge-cutoff) project. Its Hugging Face
dataset card marks the data CC BY 4.0. No upstream application source was copied; the local
backend, Optuna, persistence, reporting, and CLI implementation follow this repository's
architecture. See the [operator guide](../../guides/benchmarking/knowledge-cutoff.md) for the full
workflow and attribution notice.
