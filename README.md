# loc-lm-bench -- Production Local LLM Benchmark

loc-lm-bench is a benchmark for selecting open-weight LLMs on local Ukrainian RAG and text-analysis
workloads. It evaluates candidate models on your corpus and hardware, records the full run bundle,
and ranks results with reproducible telemetry, confidence intervals, and reviewable data gates.

## Core Capabilities

| Capability | Functional use case | Pipeline commands |
|---|---|---|
| Autonomous corpus-to-recommendation | Point one command at a corpus and get a scored RAG recommendation: ingest -> ontology draft -> verification gate -> retrieval -> joint model/config search -> prompt system -> final eval -> recommendation, journaled so an interrupted run resumes at the last completed stage. Runs fully autonomously (`SCORER_POLICY=auto`) or stops at the human gate. The rows below are those stages run individually, plus the lanes beyond them. See [Auto-RAG guide](docs/guides/benchmarking/auto-rag.md) and [Auto-RAG](docs/impl/current/auto-rag.md). | `make auto-rag CORPUS=<dir> SCORER_POLICY=auto` (or `SCORER_POLICY=human`) |
| Corpus-grounded gold sets | Convert local PDFs -- or a mixed txt/md/pdf directory -- into one corpus, then draft or ingest Ukrainian eval data with exact source spans, verified splits, and reusable corpus bundles (`make ingest-uk-squad` starts from the committed UA-SQuAD fixture instead). See [Gold-set guide](docs/guides/data-prep/goldset-from-scratch.md) and [data prep](docs/guides/data-prep/data-prep.md). | `make pdf-to-markdown PDF_DIR=<pdf-dir>` / `make ingest-corpus CORPUS_ROOT=<mixed-dir>` -> `make prepare-goldset-draft` -> `make validate-goldset` |
| Corpus hygiene and conflict control | Find what a corpus repeats or contradicts before it quietly distorts recall: an intra-document repeated-block census with per-question yield, exact/normalized duplicate-chunk collapse plus the near-duplicate residue that survives it (`make measure-duplicate-residue`), and a four-tier (hash / lexical / semantic / claim) duplicate-stale-contradiction audit. Detection never edits a corpus byte; suppressions are human-reviewed and reversible. See [conflict detection](docs/impl/current/data-prep.md#corpus-hygiene-conflict-detection-corpus-conflict-detection) and [collapse tiers](docs/impl/current/rag-core.md#near-duplicate-residue-and-the-collapse-tiers). | `make strip-corpus-repeats CORPUS=<dir>` -> `make audit-corpus-conflicts CORPUS=<dir> EFFORT=semantic` -> `make review-workbench REVIEW_PATH=<review-jsonl>` -> `make resolve-corpus-conflicts FINDINGS=<findings-jsonl> REVIEWED=<review-jsonl> APPLY=1` |
| Human verification gates | Cross-check AI-drafted data with a second frontier model, review a stratified sample (optionally multi-annotator, with a kappa agreement report and an adjudication pass), and emit accepted ledgers before real model scoring. One terminal workbench opens every ledger kind -- verification, calibration, conflicts, translations. See [verification tooling](docs/guides/human-tooling/verification-tooling.md), [review workbench](docs/impl/current/review-workbench.md), and [human evaluation](docs/guides/human-tooling/human-in-the-loop-evaluation.md). | `make cross-check-goldset BUNDLE=<draft> CROSS_CHECK_MODEL=<second-frontier>` -> `make verify-sample` -> `make verify-review` -> `make verify-adjudicate` -> `make verify-accept` |
| Adaptive local draft comparison | Detect the CUDA VRAM tier, run a fitting Qwen baseline and Gemma probe sequentially over exact shared seeds, unload each model between lanes, and inspect machine/human quality deltas. See [Gold-set guide](docs/guides/data-prep/goldset-from-scratch.md#finish-the-bounded-ukrainian-local-comparison). | `make local-ua-draft-probe` -> `make local-ua-draft-complete` -> `make local-ua-draft-analyze` |
| External and closed-service evaluation | Bring in data and systems that do not run locally: curate and import open-data drafts from Claude / ChatGPT / NotebookLM with grounding and retrievability checks, and human-score an answered JSONL log from a closed RAG service using the same objective signals plus a source-span audit. Restricted corpora stay local. See [external AI service artifacts](docs/guides/data-prep/external-ai-service-artifacts.md) and [external answer scoring](docs/impl/current/rag-core.md#external-answer-log-scoring). | `make curate-drafts CURATE_KIND=<kind>` -> `make import-external-draft ARTIFACT=<file> CORPUS=<dir>` -> `make score-external-rag EXTERNAL_RAG_ANSWERS=<jsonl>` |
| FAISS and GraphRAG retrieval | Build vector and graph stores, validate recall/MRR, and compare retrieval strategies before blaming the model: dense, hybrid dense+BM25+RRF, cross-encoder reranking, graph-vector fusion, and an incremental refresh that re-embeds only what changed and reports drift. See [retrieval comparison](docs/guides/benchmarking/graph-vs-faiss-comparison.md) and [store lifecycle](docs/impl/current/rag-core.md#store-lifecycle-dynamic-corpus-refresh). | `make build-index` -> `make build-graph` -> `make validate-retrieval` -> `make compare-retrieval` -> `make refresh-index` |
| Retrieval bake-offs with measured uncertainty | Decide chunker, embedder, vector backend, and graph share on evidence rather than defaults: every lane reports recall@k / MRR beside its own MEASUREMENT FLOOR (how far numeric noise alone moves the metric) and a paired bootstrap interval with the item-level win/loss/tie ledger, so a comparison ends in an explicit adopt-or-retain verdict instead of a third decimal. See [embedder bake-off](docs/impl/current/rag-core.md#embedder-conventions-and-bake-off), [vector-store commands](docs/impl/current/platform-vector-matrix.md#vector-store-commands), and [fusion evidence](docs/impl/current/graphrag-backend.md#graph-vector-fusion-evidence). | `make compare-retrieval CHUNK_STRATEGIES=<a,b> NOISE_FLOOR=1` -> `make compare-embeddings NOISE_FLOOR=1` -> `make compare-vector-stores NOISE_FLOOR=1` -> `make compare-graph-fusion NOISE_FLOOR=1` |
| Does retrieval pay for itself? | Measure what RAG actually buys instead of assuming it: score one item set closed-book vs RAG vs whole-document long context with paired uplift intervals, per-question-type slices, and a contamination flag for items the model already answers with no evidence -- then compare the ANSWERS two retrieval lanes produce on the identical items. See [context ablation](docs/impl/current/rag-core.md#context-ablation-does-rag-pay-for-itself-rag-vs-long-context-ablation) and [answer-quality evidence](docs/impl/current/graphrag-backend.md#answer-quality-evidence). | `make compare-context-strategies MODEL=<model> GOLDSET=<gs>` -> `make compare-answer-quality MODEL=<model>` |
| Ukrainian query-side processing | Improve Ukrainian queries before retrieval without touching the corpus: casefold/apostrophe/transliteration normalization, corpus-vocabulary typo tolerance, alias/glossary expansion, and an opt-in logged LLM rewrite -- with an A/B report proving each step's recall/MRR delta. See [RAG core](docs/impl/current/rag-core.md) query-side processing. | `make build-query-glossary BUNDLE=<draft>` -> `make validate-retrieval QUERY_PREP=normalize,typos,glossary QUERY_PREP_AB=1` -> `make run-eval QUERY_PREP=normalize,typos,glossary` |
| Failure analysis and robustness probes | Find WHERE a run loses points instead of guessing: classify and cluster one bundle's misses (retrieval vs generation vs scoring) with an optional re-probe at other `top_k`, benchmark noisy Ukrainian queries (keyboard layout, apostrophe variants, transliteration) against each mitigation lane, and probe lost-in-the-middle by planting the gold chunk at head/middle/tail for a per-model context-order recommendation. See [miss analysis](docs/impl/current/rigor-board-judge.md#miss-analysis-analyze-misses), [query robustness](docs/impl/current/rigor-board-judge.md#ukrainian-query-robustness-benchmark), and [context position](docs/impl/current/rigor-board-judge.md#context-position-probe-probe-context-position). | `make analyze-misses RUN_DIR=<bundle>` -> `make bench-query-robustness MODEL=<model>` -> `make probe-context-position MODEL=<model>` |
| Groundedness and citation metrics | Score answer-side RAG quality beyond reference overlap: deterministic groundedness fraction, `[i]` citation validity + hallucinated-citation rate, and insufficient-context abstention probes (gold evidence removed -> the model should decline). Additive columns that never change the headline. See [RAG core](docs/impl/current/rag-core.md) groundedness and citation metrics. | `make run-eval CITED_ANSWERS=1 SCORE_GROUNDEDNESS=1 INSUFFICIENT_CONTEXT_PROBES=20` |
| Local serving and model planning | Resolve which candidate models fit the host (GPU + RAM, KV-cache-aware), generate the serve + run-eval config for the detected VRAM tier, prepare weights, and run through Ollama, vLLM, or llama.cpp. See [vLLM backend guide](docs/guides/benchmarking/vllm-backend.md) and [inference config](docs/inference/config-example.md). | `make detect-gpu-vram` -> `make list-models` -> `make gen-serving-config` -> `make prep-models` |
| Private model leaderboards | Evaluate candidates on your corpus, isolate sweep cells, run a successive-halving joint model+RAG-config search, tune finalists, and inspect ranked boards with CIs. Then `recommend` distills the sweep into host-adaptive picks (best accuracy, best quality/watt, best model for this GPU tier) plus a comparison chart. See [RAG core](docs/guides/benchmarking/run-rag-core.md), [joint search](docs/impl/current/rigor-board-judge.md#joint-model--config-search), and [MLflow analysis](docs/guides/benchmarking/mlflow-analysis.md). | `make run-eval` -> `make sweep` -> `make joint-search` -> `make pipeline` -> `make recommend` -> `make board` -> `make mlflow` |
| Local fine-tuning and adapter lifecycle | Adapt a model to your corpus while the final split stays held out: export SFT/DPO records from tuning-split misses, run a budgeted LoRA/QLoRA hyperparameter search, train an adapter, iterate the self-improvement loop or distill a local teacher into a smaller student, then register, serve, and garbage-collect adapters through a registry that records eval evidence, staleness, and a contamination guard. See [self-improvement loop](docs/guides/benchmarking/self-improvement-loop.md) and [adapter registry](docs/impl/current/extended-workflows.md#adapter-registry-and-lifecycle). | `make export-finetune-set RUN_DIR=<tuning-run>` -> `make finetune-hparams` -> `make finetune-adapter` -> `make self-improve` / `make distill` -> `make register-adapter` -> `make serve-adapter` |
| Calibrated judge gates | Use a local DeepEval judge only after human-rated Ukrainian calibration clears the Spearman threshold; optionally measure a frontier judge's agreement against both the human and local references, plus its cost per item, behind explicit egress consent and a hard spend cap. See [calibration tooling](docs/guides/human-tooling/calibration-tooling.md) and [judge experiments](docs/guides/human-tooling/judge-experiments.md). | `make calibration-run` -> `make calibration-rate` -> `make calibration-score` -> `make judge-experiment` (optional: `make frontier-judge-agreement FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=<cap>`) |
| Prompt-system tuning | Generate reviewable prompt packages, tune on one split, and verify generalization on held-out final data. See [prompt-system guide](docs/guides/benchmarking/prompt-system-rag.md) and [RAG core](docs/guides/benchmarking/run-rag-core.md). | `make prompt-system-prepare PROMPT_SYSTEM_CORPUS=<dir>` -> `make prompt-system-review PROMPT_SYSTEM_RUN_DIR=<dir> PROMPT_SYSTEM_ACTION=pin PROMPT_SYSTEM_ID=<id>` -> `make run-eval PROMPT_SYSTEM_ID=<id> PROMPT_PACKAGE=<dir>` -> `make prompt-system-compare` |
| Security robustness | Score jailbreak, prompt-injection, RAG-injection, exfiltration, and benign-control cases as a separate security tier, and derive corpus-specific content-safety cases from your own draft bundle behind a human review gate. See [security learning path](docs/guides/learning-path/learning-path-security.md) and [category suite](docs/impl/current/category-benchmark-suite.md#security). | `make bench-security MODEL=<model> BACKEND=<backend>` -> `make derive-security-cases BUNDLE=<draft> SECURITY_DERIVE_OUT=<cases.json>` -> `make bench-security-derived SECURITY_DERIVE_CASES=<cases.json>` |
| Multi-turn chains and context policies | Score chain-of-questions sets where each step depends on the previous answers, and rank the context policy that carries them -- fresh retrieval, full transcript, running summary, or a staged librarian -> analyst -> answerer role sequence -- for one fixed model, with bootstrap CIs on final-answer correctness. See [context-policy comparison](docs/impl/current/extended-workflows.md#context-policy-comparison). | `make chain-goldset-pipeline CHAIN_CORPUS=<dir>` -> `make chain-goldset-finalize` -> `make bench-chain-context CHAIN_CONTEXT_MODEL=<model>` |
| Category benchmark suites | Score security, tooling, agentic, summarization, structured output, and text-analysis categories, then publish a guarded composite headline. See [composite headline guide](docs/guides/benchmarking/composite-headline.md) and [category learning path](docs/guides/learning-path/learning-path-evaluation-categories.md). | `make composite-headline` |
| Real-world knowledge cutoff | Estimate the effective month where a local model's recall of unpredictable public events decays toward chance, using a revision-pinned Hugging Face event set, position-balanced MCQs, seeded Optuna fitting, controls, and JSON/Markdown/MLflow reports. Run it on the English event set, or on a frozen human-reviewed Ukrainian translation of the same items to separate knowledge from language. See the [knowledge-cutoff guide](docs/guides/benchmarking/knowledge-cutoff.md) and the [bilingual workflow](docs/impl/current/knowledge-cutoff.md#ukrainian-bilingual-calibration-workflow). | `make bench-knowledge-cutoff MODEL=<model> BACKEND=<backend>` -> `make knowledge-cutoff-bilingual` |
| Agentic harness comparison | Run the same task set through loop, LangGraph, and CrewAI harnesses to separate model quality from orchestration effects. See [CrewAI harness guide](docs/guides/benchmarking/crewai-harness.md) and [category learning path](docs/guides/learning-path/learning-path-evaluation-categories.md). | `make agentic-harness-compare` |
| Platform matrix telemetry | Compare a logical model base across serving backends with VRAM, throughput, power, and quality-per-watt telemetry. See [platform matrix guide](docs/guides/benchmarking/platform-matrix.md). | `make platform-matrix` |

`make help` lists every target with its options; the command areas are mapped in
[current implementation](docs/impl/current.md).

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
