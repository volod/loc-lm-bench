# Local Ukrainian judge experiments

The judge calibration gate judge path uses DeepEval 4 G-Eval metrics with explicit Ukrainian
criteria for faithfulness and answer relevancy. Ragas is not part of the dependency graph:
its installed 0.4.3 release did not import against the project's current LangChain stack, and
compatibility shims or dependency downgrades would make the benchmark fragile.

The judge remains a diagnostic until its Spearman correlation with human ratings reaches the
configured threshold. A successful smoke experiment validates model connectivity and structured
output; it does not satisfy that calibration gate.

## Setup

Install the project extras and configure a local OpenAI-compatible endpoint:

    make venv
    export DEEPEVAL_JUDGE_BASE_URL=http://127.0.0.1:8000/v1
    export DEEPEVAL_JUDGE_API_KEY=local
    export DEEPEVAL_TELEMETRY_OPT_OUT=YES

`DEEPEVAL_JUDGE_API_KEY` is a placeholder unless the local server enforces authentication. Keep
`DEEPEVAL_TELEMETRY_OPT_OUT=YES` for a local-only experiment.

For vLLM, use the model id returned by `GET /v1/models`. For Ollama's OpenAI-compatible API, use
`http://127.0.0.1:11434/v1` and the locally installed tag.

## Recorded smoke experiment

Run three fixed Ukrainian cases: supported/relevant, unsupported/relevant, and
supported/irrelevant.

    make judge-experiment \
      JUDGE_MODEL=google/gemma-4-12B-it-qat-w4a16-ct \
      JUDGE_BASE_URL=http://127.0.0.1:8000/v1

Ollama example:

    make judge-experiment \
      JUDGE_MODEL=gemma4:latest \
      JUDGE_BASE_URL=http://127.0.0.1:11434/v1

The command writes
`$DATA_DIR/judge-experiment/<UTC timestamp>/result.json`. The artifact records the served model,
endpoint, exact Ukrainian evaluation steps, inputs, and both scores. It excludes API keys. Check
that the supported/relevant case scores high on both axes, the unsupported answer scores lower on
faithfulness, and the irrelevant answer scores lower on answer relevancy.

The local model must return valid JSON for DeepEval's structured result. If parsing fails, inspect
the model server logs and use a model with reliable instruction following; do not relax the
evaluation prompt or silently replace malformed scores.

## Calibration experiment

With candidate and judge endpoints reachable, generate a pre-filled worksheet over the committed
human-reviewed calibration split. The defaults target a local Ollama judge (`gemma3:27b` on
:11434) with the embedder pinned to CPU, so this is just:

    make calibration-run

To use a vLLM judge instead, override the knobs:

    make calibration-run \
      MODEL=llama3.2:3b BACKEND=ollama \
      JUDGE_MODEL=google/gemma-4-12B-it-qat-w4a16-ct \
      JUDGE_BASE_URL=http://127.0.0.1:8000/v1

The worksheet receives model answers and ungated judge ratings. A human reviewer then fills
`human_rating` independently with the interactive rater (the judge column is hidden by default, so
ratings are not anchored to the judge):

    make calibration-rate

and computes the decision (`RATINGS` defaults to the worksheet, `calibration/<CAL_NAME>.csv`):

    make calibration-score

The score command reports Spearman rho, a bootstrap confidence interval, and whether the judge
clears `rho >= 0.6`. Until a completed human worksheet passes, normal evaluations keep the judge
demoted and rank by objective correctness. The full rater reference and the new-goldset /
text-corpus-draft cases are in the [calibration-tooling manual](calibration-tooling.md).

On a 16 GB GPU, a 12B judge normally cannot share VRAM with a vLLM candidate. Use an Ollama GGUF
with CPU offload, a smaller test judge, or another machine on the same local network. Record the
actual endpoint and served model for every comparison; never substitute a cloud judge under the
same experiment name.
