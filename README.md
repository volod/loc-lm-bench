# loc-lm-bench -- Production Local LLM Benchmark

loc-lm-bench is a benchmark for selecting open-weight LLMs on local Ukrainian RAG and text-analysis
workloads. It evaluates candidate models on your corpus and hardware, records the full run bundle,
and ranks results with reproducible telemetry, confidence intervals, and reviewable data gates.

## Core Capabilities

| Capability | Functional use case | Pipeline commands |
|---|---|---|
| Corpus-grounded gold sets | Convert local PDFs to markdown, then build or ingest Ukrainian eval data with exact source spans, verified splits, and reusable corpus bundles. See [Gold-set guide](docs/guides/data-prep/goldset-from-scratch.md) and [data prep](docs/guides/data-prep/data-prep.md). | `make pdf-to-markdown PDF_DIR=<pdf-dir>` -> `make ingest-uk-squad` -> `make validate-goldset` |
| Human verification gates | Cross-check AI-drafted data, review a stratified sample, and emit accepted ledgers before real model scoring. See [verification tooling](docs/guides/human-tooling/verification-tooling.md) and [human evaluation](docs/guides/human-tooling/human-in-the-loop-evaluation.md). | `make verify-sample` -> `make verify-review` -> `make verify-accept` |
| Adaptive local draft comparison | Detect the CUDA VRAM tier, run a fitting Qwen baseline and Gemma probe sequentially over exact shared seeds, unload each model between lanes, and inspect machine/human quality deltas. See [Gold-set guide](docs/guides/data-prep/goldset-from-scratch.md#finish-the-bounded-ukrainian-local-comparison). | `make local-ua-draft-probe` -> `make local-ua-draft-complete` -> `make local-ua-draft-analyze` |
| FAISS and GraphRAG retrieval | Build vector and graph stores, validate recall/MRR, and compare retrieval strategies before blaming the model. See [retrieval comparison](docs/guides/benchmarking/graph-vs-faiss-comparison.md). | `make build-index` -> `make build-graph` -> `make validate-retrieval` -> `make compare-retrieval` |
| Ukrainian query-side processing | Improve Ukrainian queries before retrieval without touching the corpus: casefold/apostrophe/transliteration normalization, corpus-vocabulary typo tolerance, alias/glossary expansion, and an opt-in logged LLM rewrite -- with an A/B report proving each step's recall/MRR delta. See [RAG core](docs/impl/current/rag-core.md) query-side processing. | `make build-query-glossary BUNDLE=<draft>` -> `make validate-retrieval QUERY_PREP=normalize,typos,glossary QUERY_PREP_AB=1` -> `make run-eval QUERY_PREP=normalize,typos,glossary` |
| Groundedness and citation metrics | Score answer-side RAG quality beyond reference overlap: deterministic groundedness fraction, `[i]` citation validity + hallucinated-citation rate, and insufficient-context abstention probes (gold evidence removed -> the model should decline). Additive columns that never change the headline. See [RAG core](docs/impl/current/rag-core.md) groundedness and citation metrics. | `make run-eval CITED_ANSWERS=1 SCORE_GROUNDEDNESS=1 INSUFFICIENT_CONTEXT_PROBES=20` |
| Local serving and model planning | Resolve which candidate models fit the host, prepare weights, and run through Ollama, vLLM, or llama.cpp. See [vLLM backend guide](docs/guides/benchmarking/vllm-backend.md) and [inference config](docs/inference/config-example.md). | `make list-models` -> `make prep-models` |
| Private model leaderboards | Evaluate candidates on your corpus, isolate sweep cells, tune finalists, and inspect ranked boards with CIs. Then `recommend` distills the sweep into host-adaptive picks (best accuracy, best quality/watt, best model for this GPU tier) plus a comparison chart. See [RAG core](docs/guides/benchmarking/run-rag-core.md) and [MLflow analysis](docs/guides/benchmarking/mlflow-analysis.md). | `make run-eval` -> `make sweep` -> `make pipeline` -> `make recommend` -> `make board` -> `make mlflow` |
| Calibrated judge gates | Use a local DeepEval judge only after human-rated Ukrainian calibration clears the Spearman threshold. See [calibration tooling](docs/guides/human-tooling/calibration-tooling.md) and [judge experiments](docs/guides/human-tooling/judge-experiments.md). | `make calibration-run` -> `make calibration-rate` -> `make calibration-score` -> `make judge-experiment` |
| Prompt-system tuning | Generate reviewable prompt packages, tune on one split, and verify generalization on held-out final data. See [prompt-system guide](docs/guides/benchmarking/prompt-system-rag.md) and [RAG core](docs/guides/benchmarking/run-rag-core.md). | `make prompt-system-prepare PROMPT_SYSTEM_CORPUS=<dir>` -> `make prompt-system-review PROMPT_SYSTEM_RUN_DIR=<dir> PROMPT_SYSTEM_ACTION=pin PROMPT_SYSTEM_ID=<id>` -> `make run-eval PROMPT_SYSTEM_ID=<id> PROMPT_PACKAGE=<dir>` -> `make prompt-system-compare` |
| Security robustness | Score jailbreak, prompt-injection, RAG-injection, exfiltration, and benign-control cases as a separate security tier. See [security learning path](docs/guides/learning-path/learning-path-security.md). | `make bench-security MODEL=<model> BACKEND=<backend>` |
| Category benchmark suites | Score security, tooling, agentic, summarization, structured output, and text-analysis categories, then publish a guarded composite headline. See [composite headline guide](docs/guides/benchmarking/composite-headline.md) and [category learning path](docs/guides/learning-path/learning-path-evaluation-categories.md). | `make composite-headline` |
| Real-world knowledge cutoff | Estimate the effective month where a local model's recall of unpredictable public events decays toward chance, using a revision-pinned Hugging Face event set, position-balanced MCQs, seeded Optuna fitting, controls, and JSON/Markdown/MLflow reports. See the [knowledge-cutoff guide](docs/guides/benchmarking/knowledge-cutoff.md). | `make bench-knowledge-cutoff MODEL=<model> BACKEND=<backend>` |
| Agentic harness comparison | Run the same task set through loop, LangGraph, and CrewAI harnesses to separate model quality from orchestration effects. See [CrewAI harness guide](docs/guides/benchmarking/crewai-harness.md) and [category learning path](docs/guides/learning-path/learning-path-evaluation-categories.md). | `make agentic-harness-compare` |
| Platform matrix telemetry | Compare a logical model base across serving backends with VRAM, throughput, power, and quality-per-watt telemetry. See [platform matrix guide](docs/guides/benchmarking/platform-matrix.md). | `make platform-matrix` |

## Documentation

Start at the [documentation index](docs/README.md). 
Begin with the [Quick Start](docs/guides/quickstart/quick-start.md), or use
[PDF Corpus Prep](docs/guides/data-prep/pdf-corpus-prep.md) when you only need citation-preserving
PDF conversion. 

For task-oriented workflows -- benchmark my PDFs, build a gold set, verify drafted data, compare
backends -- use the [guides index](docs/guides/README.md) and its 
["Choose a scenario"](docs/guides/README.md#choose-a-scenario) table. 

The main implementation reference is [current.md](docs/impl/current.md),
and contributor guardrails live in [AGENTS.md](AGENTS.md).

## Model Families and Licenses

The default candidate sweep compares five open-weight families -- two Ukrainian-specialized and
three multilingual baselines. Each links to its upstream weights; comply with the listed license
when serving or redistributing.

| Family | Focus | Default weights | License |
| --- | --- | --- | --- |
| MamayLM v2 (INSAIT) | Ukrainian-specialized | [MamayLM v2.0 (Gemma 3) collection][mamay-col] | [Gemma Terms][gemma-lic] |
| Lapa v0.1.2 (lang-uk) | Ukrainian-specialized | [lapa-llm/lapa-v0.1.2-instruct][lapa-repo] | [Gemma Terms][gemma-lic] |
| Gemma 4 (Google) | Multilingual baseline | [google/gemma-4 collection][gemma-col] | [Gemma Terms][gemma-lic] |
| Qwen 3.6 (Alibaba) | Multilingual baseline | [Qwen/Qwen3.6-35B-A3B][qwen-repo] | [Apache 2.0][apache-lic] |
| Mistral Small 3.1 (Mistral AI) | Multilingual baseline | [mistralai/Mistral-Small-3.1-24B-Instruct-2503][mistral-repo] | [Apache 2.0][apache-lic] |

The Ukrainian families build on the prior art tracked by the
[lang-uk leaderboard](https://github.com/lang-uk) and the
[MamayLM project](https://models.mamay.ai/). Per-tier concrete variants (GGUF / w4a16 / FP8) and
serving knobs live in [docs/inference/config-example.md](docs/inference/config-example.md) and
[samples/configs/models_uk.yaml](samples/configs/models_uk.yaml).

[mamay-col]: https://huggingface.co/collections/INSAIT-Institute/mamaylm-v20-gemma-3
[lapa-repo]: https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct
[gemma-col]: https://huggingface.co/collections/google/gemma-4
[qwen-repo]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
[mistral-repo]: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503
[gemma-lic]: https://ai.google.dev/gemma/terms
[apache-lic]: https://www.apache.org/licenses/LICENSE-2.0

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
- The knowledge-cutoff benchmark does not vendor its default event set. It loads
  [`apoorvumang/knowledge-cutoff-benchmark`](https://huggingface.co/datasets/apoorvumang/knowledge-cutoff-benchmark),
  whose dataset card marks the data CC BY 4.0, and records the resolved revision. The method and
  dataset choice are inspired by Apoorv Saxena's
  [`knowledge-cutoff`](https://github.com/apoorvumang/knowledge-cutoff) project; no upstream
  application source is copied. Preserve that attribution for downloaded or redistributed data.

Other committed tutorial fixtures are repo-authored unless their local README or provenance file
states otherwise. Preserve attribution and license notices when redistributing derived artifacts.
