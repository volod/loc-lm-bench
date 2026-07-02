# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

## Forward Tasks

The tasks below form the corpus-to-recommendation spine, ordered by development sequence:
first make any-corpus goldset production automatic and high-yield (1-2), open the external
Ukrainian draft lane beside it (3), raise reviewer throughput for the larger drafts (4), make
long eval campaigns durable (5), derive corpus-specific security probes (6), explain wrong
answers and fold them into recommendations (7), then add chain-of-questions data and the
context-policy comparison that consumes it (8-9). Tasks 3 and 4 can proceed in parallel with 2;
task 6 needs only the ontology artifacts from 1; task 9 depends on 8.

### 1. corpus-any-autopipeline

- User-visible outcome: one command turns any mixed `txt`/`md`/`pdf` directory into a validated
  RAG index plus an unverified needle-in-haystack draft bundle, and an interrupted multi-hour
  draft resumes instead of restarting. The grouped quickstart currently covers PDF directories
  only and ontology inventory reads only `.md`/`.txt` (see [data prep](current/data-prep.md)),
  so mixed corpora need manual staging and a killed draft loses every finished extraction
  window.
- Scope boundary: in scope -- a unified `llb ingest-corpus` that routes PDFs through the
  existing `llb ingest-pdf-corpus` converter and passes `.md`/`.txt` through with the same
  manifest shape (`source_sha256`, incremental reuse, skip diagnostics); a
  `make quickstart-corpus` wrapper generalizing the PDF quickstart stages
  (convert -> index -> draft -> graph -> validate) to the mixed corpus; a per-document,
  per-window extraction journal inside the draft bundle plus
  `llb prepare-goldset-draft --resume <bundle>` that skips journaled windows and re-enters the
  seed/draft stages deterministically. Out of scope -- new document formats (docx, html), new
  chunking strategies, changes to the verification gate. Reuse `src/llb/prep/pdf_corpus.py`,
  `src/llb/prep/ontology/inventory.py`, `src/llb/prep/ontology/pipeline.py`, and
  `scripts/quickstart.sh` stage layout.
- Data and artifact paths: staged corpus under `<src>/_md` (PDF default today) or
  `--out-dir`; draft bundles under `$DATA_DIR/prepare-goldset/<timestamp>/` with a new
  `extraction_journal.jsonl`; quickstart logs under `$DATA_DIR/llb/logs/quickstart/`; a small
  mixed-format fixture under `samples/corpus/` for tests.
- Execution path: `llb ingest-corpus --root <dir> --out-dir <dir>`;
  `make quickstart-corpus CORPUS_SRC=<dir>`;
  `llb prepare-goldset-draft --resume <bundle>` after an interruption. Full-corpus drafting
  stays a manual heavy step outside quick CI; unit tests use the fixture corpus with a fake
  endpoint.
- Acceptance gates: `make ci` green; rerun over an unchanged mixed corpus reports `reused: true`
  for every document; a draft killed mid-extraction and resumed produces the same bundle
  contents as an uninterrupted run (same seeds, same kept items) on the fixture corpus;
  `make validate-goldset` passes against the produced bundle.
- Documentation target: [data prep](current/data-prep.md); generalize
  [`docs/guides/quickstart-pdf-corpus.md`](../guides/quickstart-pdf-corpus.md) or add
  `docs/guides/quickstart-any-corpus.md`.

### 2. draft-yield-quality-max

- User-visible outcome: draft bundles maximize meaningful, knowledge-based questions from the
  corpus instead of stopping at a flat item cap: coverage-target drafting across entity,
  relation, section, and semantic-type strata with an exhaustion report; multi-hop questions
  drafted from knowledge-graph paths; near-duplicate suppression against earlier bundles; and
  per-item question-type plus difficulty labels reviewers and analyzers can filter on.
- Scope boundary: in scope -- extend `src/llb/prep/ontology/coverage.py` with per-stratum
  coverage targets and a "seeds remaining vs drafted" report; a graph-path seed source that
  walks 2-hop subject-relation-object chains from the GraphRAG store
  (see [GraphRAG](current/graphrag-backend.md)) and drafts questions grounded in multi-span
  evidence across sections or documents; an embedding-cosine near-duplicate filter (pinned E5)
  against one or more prior bundles; a closed question-type taxonomy (factoid, definition,
  procedural, numeric, comparative, multi-hop) recorded in item provenance without breaking the
  `GoldItem` schema. Out of scope -- changing the human verification gate, new extraction
  backends, judge changes. Reuse `src/llb/prep/ontology/{draft,refine,needles}.py` and
  `src/llb/graph/`.
- Data and artifact paths: `pdf_ontology_report.json` gains a coverage matrix and dedup counts;
  `needle_items.jsonl` rows gain `question_type` and `difficulty`; graph input from
  `$DATA_DIR/llb/graph/` or the bundle's own extraction.
- Execution path:
  `make prepare-goldset-draft DRAFT_COVERAGE_TARGET=<n-per-stratum> DRAFT_MULTI_HOP=1
  DRAFT_DEDUP_AGAINST=<bundle[,bundle]>`; heavy full-corpus drafts stay manual; unit tests
  cover seed exhaustion, path walking, and dedup with fixtures.
- Acceptance gates: on the local quickstart PDF corpus, a coverage-target draft keeps more
  citation-valid needles than the current 180-cap default at an equal-or-better accept rate on
  a `make verify-sample VERIFY_N=40` review; multi-hop items carry >= 2 grounded spans and pass
  span-exact validation; injected paraphrase duplicates are removed in unit tests;
  retrieval-unique needle fraction is reported per question type.
- Documentation target: [data prep](current/data-prep.md) and
  [robust backends and ontology drafting](current/robustness-ontology-backends.md).

### 3. frontier-ua-draft-lane

- User-visible outcome: for the best Ukrainian question quality and completeness, an operator
  can opt a draft run into a best-of-breed external API (litellm-routed) for extraction,
  drafting, or both, with an explicit consent gate, a hard budget cap, per-call cost telemetry
  in provenance, and a side-by-side local-vs-frontier yield and quality report over the same
  seeds. Frontier cross-check exists (`make cross-check-goldset CROSS_CHECK_MODEL=`); the draft
  lane does not.
- Scope boundary: in scope -- a frontier endpoint option for the ontology pipeline reusing the
  litellm conventions in `src/llb/prep/frontier.py` behind the same endpoint seam as Ollama and
  vLLM drafting (`src/llb/prep/ontology/endpoint.py`); `--max-usd` and `--max-calls` guards
  that abort cleanly and record the reason; an interactive egress consent prompt naming the
  corpus and destination (policy stays as recorded in
  [product decisions](current/scope-boundaries.md)); a `llb draft-compare` command that drafts
  the same bounded seed subset locally and via frontier and reports kept-yield, gate results,
  and verify-sample accept rate. Out of scope -- making frontier the default, egress for
  scoring or judging, retries of the egress policy discussion.
- Data and artifact paths: `provenance.json` gains `endpoint.cost_usd`, call counts, and
  latency; comparison reports under `$DATA_DIR/draft-compare/<timestamp>/`.
- Execution path:
  `make prepare-goldset-draft DRAFT_ENDPOINT=frontier DRAFT_FRONTIER_MODEL=<litellm-id>
  DRAFT_MAX_USD=<n>`; `llb draft-compare --corpus-root <dir> --seeds <n> --frontier-model <id>
  --local-model <model>`; unit tests use an injected fake litellm completer.
- Acceptance gates: no network call happens without the flag plus consent (unit-tested via the
  injected completer); the budget guard aborts mid-draft and the bundle remains inspectable
  with the abort recorded; a 2-document probe against a real frontier model passes bundle gates
  with parse rate at least matching the local drafter; the comparison report ranks both lanes
  on kept-yield and accept rate.
- Documentation target: [data prep](current/data-prep.md) frontier lane notes;
  [`docs/guides/goldset-from-scratch.md`](../guides/goldset-from-scratch.md).

### 4. verify-cli-throughput

- User-visible outcome: a human reviewer clears materially more items per hour in the terminal
  review session: a confidence-ordered queue (cross-check verdict and `retrieval_rank` decide
  order), on-card evidence with PDF page citations, accept-with-edit that re-grounds an edited
  answer span immediately, additive sample enlargement that never re-shows decided rows,
  session stats with an items-per-hour ETA, and coded rejection reasons exported for draft
  feedback.
- Scope boundary: in scope -- extend `src/llb/goldset/verify.py` and
  `src/llb/goldset/verify_session.py`; keep the worksheet CSV shape backward compatible (new
  optional columns only); a `verify-sample` merge mode that draws additional stratified rows
  while carrying prior decisions forward; a rejection-reason summary artifact the drafting
  pipeline can read to tighten prompts. Out of scope -- a web UI, multi-annotator merging,
  changes to acceptance arithmetic.
- Data and artifact paths: worksheets under the draft bundle as today; a
  `rejection_reasons.json` summary beside the accepted ledger.
- Execution path: `make verify-sample BUNDLE=<draft> VERIFY_N=<n> VERIFY_MERGE=1`;
  `make verify-review VERIFY_WS=<ws> VERIFY_ORDER=confidence`; unit tests for merge
  idempotence, ordering, and re-grounding of edited spans.
- Acceptance gates: `make ci` green including session golden-path tests; merging a larger
  sample adds only new rows and preserves every decided row byte-for-byte; an edited answer
  that no longer matches its span is blocked until re-grounded; manual evidence -- one recorded
  40-item review pass on the quickstart draft with the measured items-per-hour noted in the
  current docs.
- Documentation target: [data prep](current/data-prep.md) verification gate;
  [`docs/guides/human-in-the-loop-evaluation.md`](../guides/human-in-the-loop-evaluation.md)
  and [`docs/guides/verification-tooling.md`](../guides/verification-tooling.md).

### 5. durable-eval-runner

- User-visible outcome: `llb run-eval` (and by reuse the category benches) survives endpoint
  flaps, backend crashes, and host restarts: transient per-case failures retry with capped
  backoff, completed cases journal as they finish, `--resume <run-dir>` continues a partial run
  instead of re-spending model calls, and a crashed launcher-owned backend relaunches a bounded
  number of times. The client and runner seams currently have no retry or per-case checkpoint
  (`src/llb/backends/openai_client.py`, `src/llb/executor/runner.py`); only sweeps resume, at
  whole-cell granularity.
- Scope boundary: in scope -- a retry policy keyed to the typed failure taxonomy (retry
  `timeout` and `backend_error`; never retry `refusal`, `malformed`, or scored answers); an
  append-only `cases.progress.jsonl` journal inside the staged hidden run directory; resume
  keyed to the config fingerprint plus goldset digest, refusing a mismatched resume; backend
  relaunch through the existing `BackendLauncher` seam with attempts recorded; retry, resume,
  and relaunch counters in the manifest. The atomic staged-rename finalize stays the
  transaction boundary. Out of scope -- distributed execution, a database, changing scoring.
- Data and artifact paths: `$DATA_DIR/run-eval/<timestamp>-<run-id>/` layout unchanged; the
  journal lives only in the staging directory and is dropped from the finalized bundle.
- Execution path: `llb run-eval --resume <run-dir>`, `make run-eval RESUME=<run-dir>`, knobs
  `--max-case-retries` and `--retry-backoff-s`; unit tests drive a fake endpoint that fails
  transiently n times then succeeds, and a kill-then-resume harness over the committed fixture.
- Acceptance gates: `make ci` green; a run interrupted mid-way and resumed yields
  byte-identical `scores.jsonl` ordering-independent content versus an uninterrupted run on the
  committed fixture with a deterministic fake endpoint; retries never double-score a case
  (idempotent case keys); manifest records the durability counters; sweep cells inherit the
  behavior without marker-key changes.
- Documentation target: [RAG core](current/rag-core.md) executor and persistence sections.

### 6. security-corpus-probes

- User-visible outcome: the security tier gains corpus-specific cases derived from the target
  corpus itself: prohibited-topic denial-guard probes built from the corpus ontology's
  sensitive topics and entities, benign near-boundary controls that catch over-refusal on
  legitimate corpus questions, and matched-pair bias probes (entity or group swapped, behavior
  fixed) scored for decision consistency -- all flowing through the existing detectors,
  cross-language grouping, and the human verification gate before any headline use.
- Scope boundary: in scope -- `llb derive-security-cases --bundle <draft-bundle>` reading
  `ontology.json` and `extraction.jsonl`, generating cases locally through the same drafter
  endpoint seam; emitted cases reuse the committed case schema (`lang`, `attrs.vector`,
  `xlang_group`) so `bench-security`, `cross_language_consistency`, and refusal-appropriateness
  work unchanged (see [category suite](current/category-benchmark-suite.md) and the Ukrainian
  security adaptation in [evaluation rigor](current/rigor-board-judge.md)); a bias-pair
  consistency metric alongside ASR reusing the matched-group machinery. Out of scope -- new
  detector kinds, safety classifiers, ranking unverified case sets.
- Data and artifact paths: derived cases under `$DATA_DIR/security-derive/<timestamp>/cases.json`
  with per-case grounding spans back to the corpus; a small derived-and-verified sample
  committed under `samples/` for regression.
- Execution path: `llb derive-security-cases --bundle <draft> --out <cases.json>`;
  `make bench-security SECURITY_CASES=<cases.json> MODEL=<m> BACKEND=<b>`; verification through
  the existing `verify-sample`/`verify-review`/`verify-accept` path.
- Acceptance gates: every generated probe cites a corpus topic or entity with an exact span
  (unit-tested); benign controls feed refusal-appropriateness only, never ASR; a full
  `bench-security` run over derived quickstart-corpus cases reports per-family ASR plus
  bias-pair consistency with bootstrap CIs; unverified derived sets are rejected from
  composite/headline paths.
- Documentation target: [category suite](current/category-benchmark-suite.md) security section;
  [`docs/guides/learning-path-security.md`](../guides/learning-path-security.md).

### 7. miss-analysis-recommendations

- User-visible outcome: after any run or sweep, one command explains the wrong answers: each
  miss classified as retrieval miss (gold span absent from context), generation miss (evidence
  present, answer wrong), refusal, format/scoring artifact, or judge disagreement; misses
  clustered by document, topic, and question type; and ranked, evidence-backed recommendations
  (raise or lower `top_k`, change chunking, add prompt-system dictionary terms, try the named
  alternative model) that `llb recommend` folds into its summary.
- Scope boundary: in scope -- `src/llb/board/miss_analysis.py` plus `llb analyze-misses`,
  consuming per-case `scores.jsonl`, retrieved spans, typed statuses, and judge diagnostics
  from finalized run bundles; a bounded probe mode that re-runs only the miss subset at
  alternative retrieval depths to confirm or reject the retrieval hypothesis; a misses section
  in the `recommend` summary sourced from prompt templates like the existing report prose. Out
  of scope -- automatic re-tuning (the Optuna tuner owns search), mutating run bundles.
- Data and artifact paths: `$DATA_DIR/miss-analysis/<timestamp>/{report.md,misses.jsonl}`;
  `$DATA_DIR/recommend/summary.md` gains the misses section when an analysis exists.
- Execution path: `llb analyze-misses --run-dir <run>` and
  `make analyze-misses RUN_DIR=<run>`; probe mode
  `llb analyze-misses --run-dir <run> --probe-top-k 3,8`; unit tests over a synthetic scored
  bundle covering every miss class.
- Acceptance gates: the classifier separates retrieval misses from generation misses using span
  overlap on the synthetic bundle with zero cross-class leakage; on the committed-fixture
  sweep, every recommendation line names its numeric evidence; the probe mode is resumable via
  the task-5 runner; `make ci` green.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) recommendation
  section; [`docs/guides/mlflow-analysis.md`](../guides/mlflow-analysis.md).

### 8. chain-goldset-generation

- User-visible outcome: the draft pipeline also emits chain-of-questions test sets: ordered 2-4
  step sequences in which each step supplies more specific context for the topic (topic
  overview -> narrowing detail -> exact fact), every step carrying its own reference answer and
  exact source spans, validated and human-verified with the same discipline as flat items.
- Scope boundary: in scope -- a `ChainItem` schema in `src/llb/goldset/chains.py` whose steps
  embed `GoldItem`-compatible question/answer/span fields plus a chain id, step order, and a
  dependency note describing what the previous step establishes; chain drafting seeded from
  knowledge-graph paths and heading hierarchies (reuses the task-2 graph-path walker);
  span-exact validation via an extended `validate-goldset`; verify-session support rendering a
  chain as one card with per-step checks. Out of scope -- the scoring runner (task 9),
  multi-annotator flows.
- Data and artifact paths: draft bundles gain `chains.jsonl`; a committed fixture
  `samples/goldsets/<name>/chains.jsonl` with verified chains for smoke tests.
- Execution path: `make prepare-goldset-draft DRAFT_CHAINS=1`;
  `llb validate-goldset --chains <chains.jsonl> --corpus <dir>`; chains flow through
  `verify-sample`/`verify-review`/`verify-accept` unchanged at the command level.
- Acceptance gates: every step of every chain passes span-exact validation; steps within a
  chain reference distinct spans and the final step's answer is not answerable from step-1
  context alone (checked by the retrieval-uniqueness filter applied per step); at least 10
  chains reviewed and accepted into the committed fixture; `make ci` green.
- Documentation target: [data prep](current/data-prep.md).

### 9. context-policy-bench

- User-visible outcome: for one model and a verified chain set, a ranked comparison of
  context-management policies -- fresh retrieval per step, accumulated full history, summarized
  history, and staged role/system-prompt sequences (for example librarian -> analyst ->
  answerer built from prompt-system packages) -- with per-step and final-answer correctness,
  ending in a written recommendation on which harness and context policy improves scores and
  how to sequence system prompts for better answers.
- Scope boundary: in scope -- `src/llb/bench/chain_context.py` reusing the multi-hop
  retrieve/controller/answer substrate (`src/llb/eval/multi_hop.py`), prompt-system packages
  for role prompts (see [extended workflows](current/extended-workflows.md)), and the category
  persistence and board machinery; the policy is the row label and the model is fixed,
  mirroring the harness-comparison discipline; a recommendation block sourced from prompt
  templates naming the winning policy and its per-step evidence. Out of scope -- new agentic
  frameworks (settled in [product decisions](current/scope-boundaries.md)), cross-model
  blending in one board.
- Data and artifact paths: run bundles under `$DATA_DIR/chain-context/<timestamp>/`; board
  section rows beside the harness comparison; recommend summary gains a context-policy line
  when bundles exist.
- Execution path:
  `llb bench-chain-context --chains <chains.jsonl> --model <m> --backend <b>
  --policies fresh,history,summary,roles`; a make target with the standard MODEL/BACKEND
  variables; unit tests over a fake endpoint asserting the exact context assembled per policy
  per step.
- Acceptance gates: context-assembly unit tests pass for all four policies; a run over the
  committed chain fixture produces per-step and final scores with bootstrap CIs and ranks
  policies for the fixed model; provenance records policy, prompt-system ids, and chain set
  digest; verified-data stamping matches the category suite rules; `make ci` green.
- Documentation target: [extended workflows](current/extended-workflows.md);
  [`docs/guides/prompt-system-rag.md`](../guides/prompt-system-rag.md).

## Adding Future Tasks

Add a task only when there is concrete forward work with enough detail for an engineer or an
agent to execute without guessing. Use a stable descriptive id such as `platform-matrix-power`
or `prompt-system-tuning`; keep the id only while work remains under it.

Each task entry must include:

- User-visible outcome: what new capability or decision the work should create.
- Scope boundary: what is in scope, what is explicitly out of scope, and which existing modules or
  commands should be reused.
- Data and artifact paths: expected corpus, gold set, config, `$DATA_DIR/<method>/<run>/` outputs,
  and any committed `samples/` outputs.
- Execution path: commands, manual run steps, required local services, and any heavy/dependent steps
  that must stay outside quick CI.
- Acceptance gates: tests, lint/type checks, retrieval thresholds, score comparison method, or manual
  evidence required before the item leaves this file.
- Documentation target: the narrow `docs/impl/current/*.md` topic and any guide that should receive
  the resulting behavior and run notes.

When a task surfaces new future work, add that as a new forward task. Put current behavior and
durable decisions in current docs, never in this plan.
