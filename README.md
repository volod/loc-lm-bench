# loc-lm-bench -- Production Local LLM Benchmark

loc-lm-bench is a production-ready benchmark for selecting open-weight LLMs on local Ukrainian RAG
and text-analysis workloads. It evaluates candidate models on your corpus and hardware, records the
full run bundle, and ranks results with reproducible telemetry, confidence intervals, and reviewable
data gates.

The product includes corpus-grounded gold-set tooling, FAISS and GraphRAG retrieval, Ollama, vLLM,
and llama.cpp serving, hardware-aware model planning, calibrated judge gates, category benchmark
suites, MLflow analysis, and a Streamlit leaderboard.

## Quick Start

Requires [`uv`](https://docs.astral.sh/uv/) and a running local backend such as Ollama. CUDA and
`HF_TOKEN` are only needed for GPU serving or gated model weights.

```sh
make venv
make demo-eval

make build-index CORPUS=<dir>
make validate-retrieval

llb sweep --sweep-id run1
llb pipeline

llb board
make mlflow
```

Run `make` with no target to list commands. `.env.example` documents runtime settings.

## Core Capabilities

| Capability | Entry point |
|---|---|
| Corpus-grounded gold sets | [Gold-set guide](docs/guides/goldset-from-scratch.md) |
| Local serving backends | [vLLM backend guide](docs/guides/vllm-backend.md) |
| Hardware-aware planning | [Inference config](docs/inference/config-example.md) |
| Calibrated judge gates | [Calibration tooling](docs/guides/calibration-tooling.md) |
| Ranked boards and MLflow | [MLflow analysis](docs/guides/mlflow-analysis.md) |
| FAISS and GraphRAG retrieval | [Retrieval comparison](docs/guides/graph-vs-faiss-comparison.md) |
| Human review gates | [Human-in-the-loop evaluation](docs/guides/human-in-the-loop-evaluation.md) |

## Documentation

Start at the [documentation index](docs/README.md). The main implementation reference is
[current.md](docs/impl/current.md), and contributor guardrails live in [AGENTS.md](AGENTS.md).

loc-lm-bench consumes public Ukrainian evaluation projects as reference material and reranks models
on a private corpus. Re-check dataset licenses and preserve attribution before redistributing any
derived artifacts.
