# loc-lm-bench -- Production Local LLM Benchmark

loc-lm-bench is a benchmark for selecting open-weight LLMs on local Ukrainian RAG and text-analysis
workloads. It evaluates candidate models on your corpus and hardware, records the full run bundle,
and ranks results with reproducible telemetry, confidence intervals, and reviewable data gates.

## Core Capabilities

One command runs the whole path. Everything below it is a stage you can also run on its own.

| Entry point | What it does | Command |
|---|---|---|
| Autonomous corpus-to-recommendation | Point one command at a corpus and get a scored RAG recommendation: ingest -> ontology draft -> verification gate -> retrieval -> joint model/config search -> prompt system -> final eval -> recommendation. Journaled, so an interrupted run resumes at the last completed stage; runs fully autonomously (`SCORER_POLICY=auto`) or stops at the human gate. See [Auto-RAG guide](docs/guides/benchmarking/auto-rag.md) and [Auto-RAG](docs/impl/current/auto-rag.md). | `make auto-rag CORPUS=<dir> SCORER_POLICY=auto` |

The eight areas below are those stages run individually, plus the lanes beyond them. `make help`
lists every target with its options; the command areas are mapped in
[current implementation](docs/impl/current.md).

### 1. Corpus and gold data

Turn source documents into verified records. Only reviewed items ever score a model.

| Capability | What it gives you | Pipeline commands |
|---|---|---|
| Corpus ingestion | Local PDFs -- or a mixed txt/md/pdf directory -- become one corpus with citation-preserving conversion. See [PDF corpus prep](docs/guides/data-prep/pdf-corpus-prep.md). | `make pdf-to-markdown PDF_DIR=<pdf-dir>` / `make ingest-corpus CORPUS_ROOT=<mixed-dir>` |
| Corpus-grounded gold sets | Ukrainian eval data with exact source spans, verified splits, and reusable corpus bundles; `make ingest-uk-squad` starts from the committed UA-SQuAD fixture instead. See [Gold-set guide](docs/guides/data-prep/goldset-from-scratch.md) and [data prep](docs/guides/data-prep/data-prep.md). | `make prepare-goldset-draft` -> `make validate-goldset` (widen a multi-hop slice with `make widen-multihop-draft`) |
| Human verification gates | Cross-check AI-drafted data with a second frontier model, review a stratified sample (optionally multi-annotator, with kappa agreement and adjudication), and emit accepted ledgers before any real scoring. One terminal workbench opens every ledger kind -- verification, calibration, conflicts, translations. See [verification tooling](docs/guides/human-tooling/verification-tooling.md), [review workbench](docs/impl/current/review-workbench.md), and [human evaluation](docs/guides/human-tooling/human-in-the-loop-evaluation.md). | `make cross-check-goldset BUNDLE=<draft> CROSS_CHECK_MODEL=<second-frontier>` -> `make verify-sample` -> `make verify-review` -> `make verify-adjudicate` -> `make verify-accept` |
| External and closed-service evaluation | Curate and import open-data drafts from Claude / ChatGPT / NotebookLM with grounding and retrievability checks, and human-score an answered JSONL log from a closed RAG service with the same objective signals plus a source-span audit. Restricted corpora stay local. See [external AI service artifacts](docs/guides/data-prep/external-ai-service-artifacts.md) and [external answer scoring](docs/impl/current/rag-core/run-path.md#external-answer-log-scoring). | `make curate-drafts CURATE_KIND=<kind>` -> `make import-external-draft ARTIFACT=<file> CORPUS=<dir>` -> `make score-external-rag EXTERNAL_RAG_ANSWERS=<jsonl>` |
| Adaptive local draft comparison | Detect the CUDA VRAM tier, run a fitting Qwen baseline and Gemma probe sequentially over exact shared seeds, unload each model between lanes, and inspect machine/human quality deltas. See [Gold-set guide](docs/guides/data-prep/goldset-from-scratch.md#finish-the-bounded-ukrainian-local-comparison). | `make local-ua-draft-probe` -> `make local-ua-draft-complete` -> `make local-ua-draft-analyze` |

### 2. Corpus hygiene and conflict control

Find what a corpus repeats or contradicts before it quietly distorts recall. Detection never edits a
corpus byte; suppressions are human-reviewed and reversible.

| Capability | What it gives you | Pipeline commands |
|---|---|---|
| Repetition census and collapse residue | An intra-document repeated-block census with per-question yield, plus the near-duplicate residue that survives exact/normalized duplicate-chunk collapse. See [collapse tiers](docs/impl/current/rag-core/retrieval-store.md#near-duplicate-residue-and-the-collapse-tiers). | `make strip-corpus-repeats CORPUS=<dir>` / `make audit-repeat-yield` / `make measure-duplicate-residue` |
| Duplicate / stale / contradiction audit | A four-tier (hash / lexical / semantic / claim) audit with corpus-calibrated cutoffs and a measured claim-tier precision block behind its own calibration gate. See [conflict detection](docs/impl/current/data-prep/conflict-detection.md#corpus-hygiene-conflict-detection-corpus-conflict-detection). | `make audit-corpus-conflicts CORPUS=<dir> EFFORT=semantic` |
| Review budget before you spend it | How many DECISIONS a finding count actually is: the distinct-unit census, decision groups ranked by what is at stake, `to decide` vs `to review`, the review count projected under each resolution policy, and the decision range between both grouping rules. See [decision groups](docs/impl/current/data-prep/conflict-decision-groups.md#how-many-decisions-the-row-count-is). | `make audit-corpus-conflicts PROJECT_POLICY=conservative,prefer-newer` -> `make compare-conflict-granularity GRANULARITY_RUNS="<run> <run>"` |
| Reversible resolution | Review each group in the workbench, then plan or apply a reversible overlay -- with rollback. Stage attribution re-reads from a finished bundle alone, no model or store. See [conflict resolution](docs/impl/current/data-prep/conflict-resolution.md). | `make review-workbench REVIEW_PATH=<review-jsonl>` -> `make resolve-corpus-conflicts FINDINGS=<findings-jsonl> REVIEWED=<review-jsonl> APPLY=1` -> `make recompute-conflict-stage STAGE_RUNS="<run> <run>"` |

### 3. Record linkage and identity

Decide which records denote the same thing with a probability and a labelled operating point rather
than one hand-set constant. Linkage answers "are these the same thing", never "do these contradict
each other", and a proposed merge never rewrites a corpus, a gold set, or a stored graph -- adopting
one is a separate decision taken on retrieval or review evidence.

| Capability | What it gives you | Pipeline commands |
|---|---|---|
| Probabilistic linkage seam | A record table, a comparison specification, and blocking rules in; pairwise match probabilities and identity clusters out, with the trained model written into the run bundle so the answer replays without re-fitting. Reviewer labels turn the threshold into a read-off point on a precision/recall curve. Splink 4 on DuckDB, no GPU and no network; the `linkage` extra is in the default `make venv`. See [entity resolution](docs/impl/current/entity-resolution.md) and the [committed Ukrainian sample](samples/linkage/README.md). | `make link-records RECORDS=<jsonl> LINK_SPEC=<json> LINK_LABELS=<labels-jsonl>` -> `make replay-linkage RECORDS=<jsonl> LINK_BUNDLE=<prior-run-dir>` |
| Gold-item duplicate lane (shadow) | A drafting dedup drop arrives with a match probability, the field-level agreements behind it, and the prior item it lost to, instead of one question cosine. The shipped constant still decides every drop and the fitted model is scored beside it, so the disagreement list is the artifact. See [the gold-item lane](docs/impl/current/entity-resolution.md#the-gold-item-lane). | `make prepare-goldset-draft DRAFT_CORPUS=<dir> DRAFT_DEDUP_AGAINST=<prior-bundle> DRAFT_DEDUP_LINKAGE_SHADOW=1` |
| Graph node and document-edition lanes | Propose a graph node-cluster overlay and PRICE it on the graph lane before adopting it, and cluster re-ingested document editions out of the conflict audit's lexical tier -- one probability in place of its two hand-set cutoffs, with each decline written with its reason. See [the graph node lane](docs/impl/current/entity-resolution.md#the-graph-node-lane), [linkage evidence](docs/impl/current/graphrag-backend/entity-resolution-evidence.md), and [the document-edition lane](docs/impl/current/entity-resolution.md#the-document-edition-lane). | `make resolve-graph-entities GOLDSET=<gs>` / `make audit-corpus-conflicts CORPUS=<dir> EFFORT=lexical LINKAGE=1` |

### 4. Retrieval

Build the stores, then decide chunker, embedder, backend, and graph share on evidence rather than
defaults.

| Capability | What it gives you | Pipeline commands |
|---|---|---|
| FAISS and GraphRAG stores | Vector and graph stores, recall/MRR validation, and an incremental refresh that re-embeds only what changed and reports drift. See [retrieval comparison](docs/guides/benchmarking/graph-vs-faiss-comparison.md) and [store lifecycle](docs/impl/current/rag-core/retrieval-store.md#store-lifecycle-dynamic-corpus-refresh). | `make build-index` -> `make build-graph` -> `make validate-retrieval` -> `make refresh-index` |
| Which chunker, under one size cap | Seven strategies re-read under the SAME hard `size` cap, so no lane can win by emitting bigger chunks. `table` is the strategy for converted-PDF markdown: a table that fits stays one chunk, a longer one packs whole rows, and every block carries its heading breadcrumb plus the header row's source offsets, so a middle row block has not lost its column names. See [chunking strategies](docs/impl/current/rag-core/chunking.md#seven-strategy-re-read-under-the-size-cap) and [table-aware chunking](docs/impl/current/rag-core/chunking.md#table-aware-chunking). | `make compare-retrieval CHUNK_STRATEGIES=table,recursive,sentence NOISE_FLOOR=1` -> `make build-index CHUNK_STRATEGY=table` |
| Bake-offs with measured uncertainty | Every lane reports recall@k / MRR beside its own MEASUREMENT FLOOR -- how far numeric noise alone moves the metric -- and a paired interval with the item-level win/loss/tie ledger, so a comparison ends in an explicit adopt-or-retain verdict instead of a third decimal. Evidence [intactness](docs/impl/current/rag-core/retrieval-metrics.md#evidence-intactness-span_char_coveragek--span_intactk) (`cover@k` / `intact@k`) rides the same bootstrap draw, so a chunk that carried half a gold span stops reading as a whole hit -- without ranking the board. When intactness says the evidence arrived in PIECES, two levers are measured against it on one command (`CHUNK_SIZES=` rebuilds per `size` cap, `STITCH=1` merges contiguous retrieved chunks without retrieving anything new), each priced in the `chars@k` served-context column -- see [two levers against fragmented evidence](docs/impl/current/rag-core/fragmented-evidence.md). An encoder that ships its forward pass as repository code is declined by the roster screen until you opt in (`EMBED_ALLOW_REMOTE_CODE=1`). See [embedder bake-off](docs/impl/current/rag-core/embedders.md#embedder-conventions-and-bake-off), [vector-store commands](docs/impl/current/platform-vector-matrix.md#vector-store-commands), and [fusion evidence](docs/impl/current/graphrag-backend.md). | `make compare-retrieval CHUNK_STRATEGIES=<a,b> NOISE_FLOOR=1` -> `make compare-embeddings NOISE_FLOOR=1` -> `make compare-vector-stores NOISE_FLOOR=1` -> `make compare-graph-fusion NOISE_FLOOR=1` |
| Does a rank gain reach the ANSWER? | An encoder that only ranks the same evidence earlier is discarded by a recall@k bar. The adoption sweep scores both encoders end to end over a `top_k` x reranker grid and says where the rank gain reaches the answer -- then whether two models agree, whether a declared model property predicts it, and the cheapest screen that decides it. See [first-hit-rank adoption bar](docs/impl/current/rag-core/first-hit-rank-adoption.md). | `make compare-embedder-adoption EMBED_BASELINE=<a> EMBED_CANDIDATE=<b>` -> `make compare-adoption-models` / `make compare-adoption-roster` / `make compare-adoption-screen` |
| Which cross-encoder, and is it worth the VRAM? | One shared candidate pool retrieved once per item, re-sorted by every reranker, with the reranker-OFF row in the same table: rank quality, mean first-hit rank, wall-clock per query, cold load, and resident/peak VRAM. Declare what the generator holds and a candidate that cannot fit beside it is skipped with the footprint that decided it. See [reranker bake-off](docs/impl/current/rag-core/reranker-bakeoff.md). | `make compare-rerankers CONFIG=<run-config.yaml> RERANK_GENERATOR_VRAM_MB=<mb> NOISE_FLOOR=1` |
| Question-type routing | Tune sidecar-free graph-fusion routing thresholds on one split with the question-type labels hidden from every routing decision, freeze one policy, then score it on held-out final data. See [graph-vector fusion](docs/impl/current/rag-core/graph-vector-fusion.md) and [what the calibration measured](docs/impl/current/graphrag-backend/sidecar-free-routing-calibration.md). | `make calibrate-fusion-routing GOLDSET=<gs>` |
| Ukrainian query-side processing | Improve Ukrainian queries before retrieval without touching the corpus: casefold/apostrophe/transliteration normalization, corpus-vocabulary typo tolerance, alias/glossary expansion, and an opt-in logged LLM rewrite -- with an A/B report proving each step's recall/MRR delta. What the typo step's conservative constants COST is measured rather than assumed: one sweep prices each constant on retrieval plus a per-edit audit that separates "restored the user's word" from "rewrote the question into something the corpus contains", and returns pin / adopt / expose per constant. Russian and code-switched queries against the same Ukrainian corpus are measured on the same bench. See [query-side processing](docs/impl/current/rag-core/rerank-and-query.md#query-side-processing-uk-query-processing), [the restoration sweep](docs/impl/current/rag-core/rerank-and-query.md#restoration-constraint-sweep-restoration-constraint-threshold-sweep), and [the cross-lingual lane](docs/impl/current/rigor-board-judge/robustness-benchmarks.md#cross-lingual-query-lane). | `make build-query-glossary BUNDLE=<draft>` -> `make validate-retrieval QUERY_PREP=normalize,typos,glossary QUERY_PREP_AB=1` -> `make sweep-restoration-constraints GOLDSET=<gs>` -> `make run-eval QUERY_PREP=normalize,typos,glossary` |
| Why a multi-hop item misses | When `all-spans@k` will not move for any ranking knob, ranking is not the explanation. The probe ranks every labeled span twice -- by the item's question and by the span's own text as the retrievability control -- and classifies each item by its worst hop into `covered` / `budget` / `query` / `unreachable`, which point at opposite fixes. Pair it with a decomposition lane and the report counts conversions and regressions against the ORIGINAL diagnosis cohorts. See [per-hop evidence](docs/impl/current/graphrag-backend/retrieval-budget-evidence.md#the-per-hop-probe-lane). | `make probe-multihop-hops CONFIG=<run-config.yaml> GOLDSET=<gs> HOP_PROBE_BUDGETS=10,25,50` (add `QUERY_PREP=decompose QUERY_PREP_MODEL=<local-model>` for the paired lane) |
| Does retrieval pay for itself? | Score one item set closed-book vs RAG vs whole-document long context with paired uplift intervals, per-question-type slices, and a contamination flag for items answered with no evidence -- then compare the ANSWERS two retrieval lanes produce on identical items, scored under a second model reading byte-identical context so that "the extra evidence does not reach the answer" separates from "this one tune does not use it". See [context ablation](docs/impl/current/rag-core/context-ablation.md#context-ablation-does-rag-pay-for-itself-rag-vs-long-context-ablation) and [answer-quality evidence](docs/impl/current/graphrag-backend/answer-quality-evidence.md#answer-quality-evidence). | `make compare-context-strategies MODEL=<model> GOLDSET=<gs>` -> `make compare-answer-quality MODEL=<model> FUSION_COMPARISON=<sweep>/comparison.json` |

### 5. Answer quality and failure analysis

Score the answer side beyond reference overlap, and find WHERE a run loses points instead of
guessing.

| Capability | What it gives you | Pipeline commands |
|---|---|---|
| Groundedness and citation metrics | Deterministic groundedness fraction, `[i]` citation validity + hallucinated-citation rate, and insufficient-context abstention probes (gold evidence removed -> the model should decline). Additive columns that never change the headline. See [groundedness and citation metrics](docs/impl/current/rag-core/scoring.md#groundedness-and-citation-metrics-groundedness-citation-metrics). | `make run-eval CITED_ANSWERS=1 SCORE_GROUNDEDNESS=1 INSUFFICIENT_CONTEXT_PROBES=20` |
| Typed answer envelope | Ask the model for a DECLARED answer -- text, an explicit abstention flag, and per-claim citations -- validated at the generation boundary, so a completion in the wrong shape ends in a typed status (`malformed` vs `schema_invalid`) after one bounded repair instead of being scored as a wrong answer, and every answer-side metric reads a field instead of a regex over prose. See [the typed answer envelope](docs/impl/current/rag-core/scoring.md#typed-rag-answer-envelope-typed-rag-answer-envelope). | `make run-eval ANSWER_FORMAT=envelope MAX_TOKENS=768 MODEL=<model>` -> `make analyze-answer-envelope RUN_DIRS="<bundle> <bundle>"` |
| Headline decomposition | Read one fixed item set under token F1, recall, found-rate, and the declared format policy, so a verbosity difference cannot pass for a quality difference. See [scoring](docs/impl/current/rag-core/scoring.md#headline-decomposition-and-declared-ranking-policy). | `make analyze-verbosity RUN_DIRS="<bundle> <bundle>"` |
| Failure analysis and robustness probes | Classify and cluster one bundle's misses (retrieval vs generation vs scoring), benchmark noisy Ukrainian queries -- keyboard layout, apostrophe variants, transliteration -- against each mitigation lane, and probe lost-in-the-middle by planting the gold chunk at head/middle/tail. See [miss analysis](docs/impl/current/rigor-board-judge/diagnostics.md#miss-analysis-analyze-misses), [query robustness](docs/impl/current/rigor-board-judge/robustness-benchmarks.md#ukrainian-query-robustness-benchmark), and [context position](docs/impl/current/rigor-board-judge/diagnostics.md#context-position-probe-probe-context-position). | `make analyze-misses RUN_DIR=<bundle>` -> `make bench-query-robustness MODEL=<model>` -> `make probe-context-position MODEL=<model>` |

### 6. Model selection, serving, and adaptation

Which model to run on this host -- and how to make it better on your corpus without leaking the
final split.

| Capability | What it gives you | Pipeline commands |
|---|---|---|
| Local serving and model planning | Resolve which candidate models fit the host (GPU + RAM, KV-cache-aware), generate the serve + run-eval config for the detected VRAM tier, prepare weights, and run through Ollama, vLLM, or llama.cpp. See [model families](docs/reference/model-families.md), [vLLM backend guide](docs/guides/benchmarking/vllm-backend.md), and [inference config](docs/inference/config-example.md). | `make detect-gpu-vram` -> `make list-models` -> `make gen-serving-config` -> `make prep-models` |
| Private model leaderboards | Evaluate candidates on your corpus, isolate sweep cells, run a successive-halving joint model+RAG-config search, tune finalists, and read ranked boards with CIs. `recommend` distills the sweep into host-adaptive picks: best accuracy, best quality/watt, best model for this GPU tier. See [RAG core](docs/guides/benchmarking/run-rag-core.md), [joint search](docs/impl/current/rigor-board-judge/tuning-and-search.md#joint-model--config-search), and [MLflow analysis](docs/guides/benchmarking/mlflow-analysis.md). | `make run-eval` -> `make sweep` -> `make joint-search` -> `make pipeline` -> `make recommend` -> `make board` -> `make mlflow` |
| Prompt-system tuning | Generate reviewable prompt packages, tune on one split, and verify generalization on held-out final data. See [prompt-system guide](docs/guides/benchmarking/prompt-system-rag.md). | `make prompt-system-prepare PROMPT_SYSTEM_CORPUS=<dir>` -> `make prompt-system-review PROMPT_SYSTEM_RUN_DIR=<dir> PROMPT_SYSTEM_ACTION=pin PROMPT_SYSTEM_ID=<id>` -> `make run-eval PROMPT_SYSTEM_ID=<id> PROMPT_PACKAGE=<dir>` -> `make prompt-system-compare` |
| Calibrated judge gates | Use a local DeepEval judge only after human-rated Ukrainian calibration clears the Spearman threshold; optionally measure a frontier judge's agreement against both the human and local references, plus its cost per item, behind explicit egress consent and a hard spend cap. See [calibration tooling](docs/guides/human-tooling/calibration-tooling.md) and [judge experiments](docs/guides/human-tooling/judge-experiments.md). | `make calibration-run` -> `make calibration-rate` -> `make calibration-score` -> `make judge-experiment` (optional: `make frontier-judge-agreement FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=<cap>`) |
| Fine-tuning and adapter lifecycle | Export SFT/DPO records from tuning-split misses, search LoRA/QLoRA hyperparameters on a budget, train, iterate the self-improvement loop or distill a teacher into a smaller student, then register, serve, and garbage-collect adapters through a registry that records eval evidence, staleness, and a contamination guard -- while the final split stays held out. See [self-improvement loop](docs/guides/benchmarking/self-improvement-loop.md) and [adapter registry](docs/impl/current/extended-workflows/adapter-registry.md#adapter-registry-and-lifecycle). | `make export-finetune-set RUN_DIR=<tuning-run>` -> `make finetune-hparams` -> `make finetune-adapter` -> `make self-improve` / `make distill` -> `make register-adapter` -> `make serve-adapter` -> `make list-adapters` / `make gc-adapters` |
| Platform matrix telemetry | Compare a logical model base across serving backends with VRAM, throughput, power, and quality-per-watt telemetry. See [platform matrix guide](docs/guides/benchmarking/platform-matrix.md). | `make platform-matrix` |

### 7. Beyond RAG QA: category benchmarks

The same evidence discipline applied to workloads that are not single-turn retrieval QA.

| Capability | What it gives you | Pipeline commands |
|---|---|---|
| Security robustness | Score jailbreak, prompt-injection, RAG-injection, exfiltration, and benign-control cases as a separate security tier, and derive corpus-specific content-safety cases from your own draft bundle behind a human review gate. See [security learning path](docs/guides/learning-path/learning-path-security.md) and [category suite](docs/impl/current/category-benchmark-suite.md#security). | `make bench-security MODEL=<model> BACKEND=<backend>` -> `make derive-security-cases BUNDLE=<draft> SECURITY_DERIVE_OUT=<cases.json>` -> `make bench-security-derived SECURITY_DERIVE_CASES=<cases.json>` |
| Agent loop policy | Separate the controller from the model: sweep step budget, malformed-call handling, and repeated-identical-call handling on a fixed task world, and ship one recommended cell per model only when the paired completion delta supports it. Localized repeat feedback is tested for transfer across seed, family, task family, and channel. See [loop-policy recommendation](docs/impl/current/extended-workflows/loop-policy-recommendation.md) and [repeat-feedback transfer](docs/impl/current/extended-workflows/repeat-feedback-transfer.md). | `make bench-agentic-loop MODEL=<model> BACKEND=<backend>` -> `make bench-agentic-loop-repeat-feedback` |
| Agent context policies | Rank how an agent spends its context window for one fixed model and task set -- `full`, `observation_cap`, `keep_last_n`, `compact` -- with aggregate-safe trimming, summarizer cost charged to compact, and the cap / head-share / `keep_last_n` constants swept and pinned or exposed. See [agent context policies](docs/impl/current/extended-workflows/agent-context-policies.md#aggregate-safe-trimming) and [compact versus cap](docs/impl/current/extended-workflows/compact-versus-cap.md). | `make bench-agentic-context MODEL=<model> BACKEND=<backend>` -> `make bench-agentic-context-sweep` -> `make bench-agentic-context-compact-long` |
| Agentic harness comparison | Run the same task set through loop, LangGraph, and CrewAI harnesses to separate model quality from orchestration effects. See [CrewAI harness guide](docs/guides/benchmarking/crewai-harness.md) and [agentic harness](docs/impl/current/extended-workflows/agentic-harness.md). | `make agentic-harness-compare` |
| Multi-turn chains | Score chain-of-questions sets where each step depends on the previous answers, and rank the context policy that carries them -- fresh retrieval, full transcript, running summary, or a staged librarian -> analyst -> answerer role sequence -- for one fixed model, with bootstrap CIs on final-answer correctness. See [context-policy comparison](docs/impl/current/extended-workflows/context-policy-comparison.md#context-policy-comparison). | `make chain-goldset-pipeline CHAIN_CORPUS=<dir>` -> `make chain-goldset-finalize` -> `make bench-chain-context CHAIN_CONTEXT_MODEL=<model>` |
| Category suites and composite headline | Score security, tooling, agentic, summarization, structured output, and text-analysis categories, then publish a guarded composite headline. See [composite headline guide](docs/guides/benchmarking/composite-headline.md) and [category learning path](docs/guides/learning-path/learning-path-evaluation-categories.md). | `make composite-headline` |
| Real-world knowledge cutoff | Estimate the effective month where a model's recall of unpredictable public events decays toward chance: a revision-pinned event set, position-balanced MCQs, seeded Optuna fitting, and controls. Run it in English, or on a frozen human-reviewed Ukrainian translation of the same items to separate knowledge from language. See the [knowledge-cutoff guide](docs/guides/benchmarking/knowledge-cutoff.md) and the [bilingual workflow](docs/impl/current/knowledge-cutoff.md#ukrainian-bilingual-calibration-workflow). | `make bench-knowledge-cutoff MODEL=<model> BACKEND=<backend>` -> `make knowledge-cutoff-bilingual` |

### 8. Evidence discipline

Rules for reading a result, shared across every comparison lane above rather than restated per lane.

| Capability | What it gives you | Pipeline commands |
|---|---|---|
| Paired verdicts and the measurement floor | A comparison ends in adopt-or-retain: paired bootstrap intervals over shared index sets, a predeclared MDE with the item count that powers it, family-wise adjustment when a verdict selects a grid row, a minimum-evidence gate that refuses an unreadable reading, and what a withdrawn reading needs before it may be read again. See [paired verdicts](docs/impl/current/rag-core/paired-verdicts.md). | `NOISE_FLOOR=1` on any comparison lane -> `make audit-paired-readings` |
| Published-number provenance | Every published number resolves back to the run artifact and field it came from, against committed run aggregates and their content pins; a value whose evidence no longer states it is named in one refusal. See [published values](docs/impl/current/extended-workflows/published-values.md#committed-aggregates-content-pins-and-the-growth-budget). | `make bench-agentic-published-provenance` |
| What a constant change invalidates | Before changing a shipped context-policy constant, get the list of published numbers that change invalidates -- by prompt-sequence replay, with no GPU, and auditing a compound change as ONE change. Runs in CI on the act that creates the problem. See [policy-constant audit](docs/impl/current/extended-workflows/policy-constant-audit.md#what-a-policy-constant-change-invalidates). | `make bench-agentic-policy-change-audit POLICY_FIELD=<field> POLICY_BASELINE=<a> POLICY_CANDIDATE=<b>` |

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
| Gemma 4 (Google) | Multilingual baseline | [google/gemma-4 collection][gemma-col] | [Apache 2.0][apache-lic] |
| Qwen 3.6 (Alibaba) | Multilingual baseline | [Qwen/Qwen3.6-35B-A3B][qwen-repo] | [Apache 2.0][apache-lic] |
| Mistral Small 3.1 (Mistral AI) | Multilingual baseline | [mistralai/Mistral-Small-3.1-24B-Instruct-2503][mistral-repo] | [Apache 2.0][apache-lic] |

What each family is in the sweep to answer, which artifact serves on which VRAM tier, the serving
traps, gated-model handling, and how to add a family:
**[model families, tiers, and licenses](docs/reference/model-families.md)**.

[mamay-col]: https://huggingface.co/collections/INSAIT-Institute/mamaylm-v20-gemma-3
[lapa-repo]: https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct
[gemma-col]: https://huggingface.co/collections/google/gemma-4
[qwen-repo]: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
[mistral-repo]: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503
[gemma-lic]: https://ai.google.dev/gemma/terms
[apache-lic]: https://www.apache.org/licenses/LICENSE-2.0

## Data Licenses

The repository's code is [MIT](LICENSE); its data is not. The committed UA-SQuAD fixture inherits
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) from SQuAD-derived text, the
public-screen and knowledge-cutoff task sets are fetched rather than vendored and keep their own
upstream terms, and the remaining committed fixtures are repo-authored unless a local notice says
otherwise. Preserve attribution and license notices when redistributing derived artifacts.

Per-fixture terms, the task-source table, and the redistribution checklist:
**[data licenses and attribution](docs/reference/data-licenses.md)**.
