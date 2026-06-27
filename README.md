# loc-lm-bench -- Production Local LLM Benchmark

loc-lm-bench is a production-ready benchmark for selecting open-weight LLMs on local Ukrainian RAG
and text-analysis workloads. It evaluates candidate models on your corpus and hardware, records the
full run bundle, and ranks results with reproducible telemetry, confidence intervals, and reviewable
data gates.

## Quick Start

Requires [`uv`](https://docs.astral.sh/uv/) and a running local backend such as Ollama.
CUDA and `HF_TOKEN` are only needed for GPU serving or gated model weights.

```sh
make venv
make demo-eval

make build-index CORPUS=<dir>
make validate-retrieval

make sweep SWEEP_ID=run1
make pipeline

make board
make mlflow
```

Run `make` with no target to list commands. `.env.example` documents runtime settings.

## Core Capabilities

- **Corpus-grounded gold sets.** Build or ingest Ukrainian eval data with exact source spans,
  verified splits, and reusable corpus bundles.
  Entry points: [Gold-set guide](docs/guides/goldset-from-scratch.md),
  [data prep](docs/guides/data-prep.md), `make validate-goldset`, `make ingest-uk-squad`.
- **Human verification gates.** Cross-check AI-drafted data, review a stratified sample, and emit
  accepted ledgers before real model scoring.
  Entry points: [verification tooling](docs/guides/verification-tooling.md),
  [human evaluation](docs/guides/human-in-the-loop-evaluation.md),
  `make verify-sample`, `make verify-review`, `make verify-accept`.
- **FAISS and GraphRAG retrieval.** Build vector and graph stores, validate recall/MRR, and compare
  retrieval strategies before attributing failures to the model.
  Entry points: [retrieval comparison](docs/guides/graph-vs-faiss-comparison.md),
  `make build-index`, `make build-graph`, `make validate-retrieval`, `make compare-retrieval`.
- **Local serving and model planning.** Resolve which candidate models fit the host, prepare
  weights, and run through Ollama, vLLM, or llama.cpp.
  Entry points: [vLLM backend guide](docs/guides/vllm-backend.md),
  [inference config](docs/inference/config-example.md), `make list-models`, `make prep-models`.
- **Private model leaderboards.** Evaluate candidate models on your corpus, isolate sweep cells,
  tune finalists, and inspect ranked boards with confidence intervals.
  Entry points: [run skeleton](docs/guides/run-skeleton.md),
  [MLflow analysis](docs/guides/mlflow-analysis.md), `make run-eval`, `make sweep`,
  `make pipeline`, `make board`, `make mlflow`.
- **Calibrated judge gates.** Use a local DeepEval judge only after human-rated Ukrainian
  calibration clears the Spearman threshold.
  Entry points: [calibration tooling](docs/guides/calibration-tooling.md),
  [judge experiments](docs/guides/judge-experiments.md), `make calibration-run`,
  `make calibration-rate`, `make calibration-score`, `make judge-experiment`.
- **Prompt-system tuning.** Generate reviewable prompt packages, tune on one split, and verify
  generalization on a held-out final split.
  Entry point: [prompt-system guide](docs/guides/prompt-system-rag.md).
- **Category benchmark suites.** Score security, tooling, agentic, summarization, structured
  output, and text-analysis categories, then publish a guarded composite headline.
  Entry points: [composite headline guide](docs/guides/composite-headline.md),
  [category learning path](docs/guides/learning-path-evaluation-categories.md),
  `make composite-headline`.
- **Agentic harness comparison.** Run the same task set through the loop, LangGraph, and CrewAI
  harnesses to separate model quality from orchestration effects.
  Entry point: [CrewAI harness guide](docs/guides/crewai-harness.md).
- **Platform matrix telemetry.** Compare a logical model base across serving backends with VRAM,
  throughput, power, and quality-per-watt telemetry.
  Entry point: [platform matrix guide](docs/guides/platform-matrix.md), `make platform-matrix`.

## Documentation

Start at the [documentation index](docs/README.md). The main implementation reference is
[current.md](docs/impl/current.md), and contributor guardrails live in [AGENTS.md](AGENTS.md).

## Data Licenses

Ready-to-use public fixtures and public-screen tasks keep their upstream data terms:

- The committed UA-SQuAD fixture derives from
  [`FIdo-AI/ua-squad`](https://huggingface.co/datasets/FIdo-AI/ua-squad). Its dataset-card
  metadata is MIT-marked, and the fixture applies the upstream derivative-text note that
  SQuAD-derived text inherits [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  See the local [fixture license](samples/goldsets/ua_squad_postedited_v1/LICENSE.md),
  [source metadata](samples/goldsets/ua_squad_postedited_v1/source.json), and
  [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) attribution.
- The Tier-1 public screen does not vendor task records. Its default task sources should be
  checked before publishing or redistributing data:
  [Belebele](https://huggingface.co/datasets/facebook/belebele) and
  [ARC](https://huggingface.co/datasets/allenai/ai2_arc) are CC BY-SA 4.0;
  [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) and
  [MMLU](https://huggingface.co/datasets/cais/mmlu) are MIT;
  [PIQA](https://huggingface.co/datasets/piqa) is marked license-unknown on its dataset card.

Other committed tutorial fixtures are repo-authored unless their local README or provenance file
states otherwise. Preserve attribution and license notices when redistributing derived artifacts.
