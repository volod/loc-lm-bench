# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

## Forward Tasks

The forward work is split into two sections by **who must act to complete it**:

- **[Agent Implementation Tasks](#agent-implementation-tasks)** land to `make ci` green with
  committed fixtures, injected fakes, and deterministic harnesses. A few carry a heavy real-model
  run for durable evidence; those runs are deterministic and execute on the CUDA host without any
  human judgment, so they stay in this section.
- **[Human-Assisted Tasks](#human-assisted-tasks)** cannot reach their stated acceptance without a
  human in the loop: the deliverable *is* human judgment (verification-gate reviews, drafting
  oversight, measured reviewer throughput) or requires a human authorization (egress consent +
  API spend). An agent can still build the supporting code and unit tests; the marked **human
  step** is what gates completion.

Task numbers are stable ids and never change; every task carries an explicit `Dependencies` line,
and the recommended build order within each section follows those lines.

For remaining tasks that depend on retrieval behavior, use the current RAG baseline documented in
[RAG core](current/rag-core.md) and the mixed-corpus ingestion baseline documented in
[data prep](current/data-prep.md). For tasks that depend on scoring or judging, the calibrated
local-judge baseline and tuning/sweep behavior live in
[evaluation rigor](current/rigor-board-judge.md); the prompt-system package flow and other
extended workflows live in [extended workflows](current/extended-workflows.md).

Every task below carries an explicit `Agent status` line with one of four markers:

- **CLEAR** -- agent-buildable to `make ci` green with fixtures/fakes; no run evidence, no human
  gate.
- **RUN NEEDED** -- agent-buildable, but acceptance requires a heavy deterministic run; every dev
  box is a proper CUDA host, so the agent executes these runs itself on the current machine.
- **BLOCKED BY HUMAN** -- the acceptance gate consumes an artifact only a human step can produce.
- **HUMAN-GATED** -- the deliverable itself is human judgment or authorization; supporting code and
  unit tests are agent-buildable.

## Agent Implementation Tasks

Add new agent-buildable work here per [Adding Future Tasks](#adding-future-tasks).

### multi-objective-rag-tuner

Extend `src/llb/optimize/` from single-objective quality maximization to Optuna multi-objective
search over quality and wall-clock latency (plus optional frontier scoring cost when the
frontier scorer lane is active; see [scorer policy seam](current/rigor-board-judge.md#scorer-policy-seam)),
using `NSGAIISampler` and `MedianPruner`-style early pruning on case subsets. Promote the
embedding model from a pinned constant to a categorical knob drawn from the embedding bake-off
shortlist, and add an explicit context-budget knob that couples `top_k`, `chunk_size`, and
`max_model_len` into a bounded token budget.

- Agent status: RUN NEEDED
- Dependencies: none hard (quality-vs-latency is self-contained); optional cost objective reuses
  the frontier cost ledger in [scorer policy seam](current/rigor-board-judge.md#scorer-policy-seam).
  Reuse `src/llb/optimize/tuner.py`, `tuning_space.py`, `tuner_runtime.py`, and the bake-off
  shortlist from `src/llb/rag/embedding_bakeoff.py`.
- User-visible outcome: `llb tune` gains a `--objectives quality,latency[,cost]` mode that emits a
  Pareto front plus named per-goal picks (best quality, best quality-per-second, cheapest within
  an accuracy floor) instead of a single winner.
- Scope boundary: in scope -- multi-objective study setup, pruning hooks, embedder knob with
  store-rebuild awareness, context-budget knob, Pareto report. Out of scope -- model selection
  inside the loop (see `joint-model-config-search`) and any change to the two-split
  tuning/final discipline.
- Data and artifact paths: studies under `$DATA_DIR/optuna/`; Pareto report JSON + Markdown under
  `$DATA_DIR/tune/<run>/`; the committed UA fixture
  `samples/goldsets/ua_squad_postedited_v1/` remains the CI-facing gold set.
- Execution path: `llb tune --objectives quality,latency --trials 40 --study <id>` on the CUDA
  host with a local backend; CI covers the study plumbing with a fake eval hook and zero-cost
  trials.
- Acceptance gates: `make ci` green; a deterministic heavy run on the fixture shows a Pareto
  front with at least two non-dominated points and a per-goal pick table; embedder-knob trials
  prove the store is rebuilt (not reused) when the embedder changes.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) tuning section;
  [RAG core](current/rag-core.md) for the context-budget knob semantics.

### frontier-ledger-case-checkpoint

(Optional.) Persist per-case frontier judge scores keyed by case index in the scorer ledger so a
budget-abort resume skips already-scored cases instead of re-spending on them. Today the ledger
resumes spend totals and the run journal resumes generation, but a mid-batch frontier abort
re-judges from the start of the unscored batch.

- Agent status: CLEAR
- Dependencies: none. Reuse `src/llb/scoring/policy/ledger.py` and the frontier scorer in
  `src/llb/scoring/policy/frontier.py` (see
  [scorer policy seam](current/rigor-board-judge.md#scorer-policy-seam)).
- User-visible outcome: resuming after a frontier budget abort continues judging from the first
  unscored case with no duplicate provider spend.
- Scope boundary: in scope -- case-index checkpoint in the ledger, skip-already-scored in
  `frontier_scorer`. Out of scope -- changing budget semantics or headline ranking.
- Data and artifact paths: extend `$DATA_DIR/<method>/<run>/scorer/ledger.jsonl` entries; no new
  roots.
- Execution path: unit tests with fake completers that abort mid-batch then resume.
- Acceptance gates: `make ci` green; resume test proves the second run issues N - K new calls
  after K cases were already scored.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md#scorer-policy-seam).

### joint-model-config-search

Fold model selection into the optimization loop with a successive-halving schedule: a cheap
screen pass over the `samples/configs/models_uk.yaml` candidates on a small tuning subset, then a
per-finalist multi-objective RAG tune in isolated sweep cells, then one comparable final-split
scoreboard across finalists so the recommendation covers model + RAG config + serving knobs
together instead of tuning RAG for one pre-chosen model.

- Agent status: RUN NEEDED
- Dependencies: `multi-objective-rag-tuner`. Reuse `src/llb/executor/isolation.py` sweep cells,
  the resolver in `src/llb/backends/resolver.py`, and the existing `pipeline` command's
  screen-to-finalist handoff as the schedule skeleton.
- User-visible outcome: one command produces "best model for this corpus and host, with its tuned
  RAG configuration" rather than requiring the operator to pick the model before tuning.
- Scope boundary: in scope -- the halving schedule, per-finalist tune orchestration, combined
  scoreboard, host-fit filtering via the planner. Out of scope -- new backends, new candidate
  families, and public-screen changes.
- Data and artifact paths: `$DATA_DIR/joint-search/<run>/` with per-cell bundles, halving ledger,
  and the final scoreboard JSON + Markdown; candidate manifest stays
  `samples/configs/models_uk.yaml`.
- Execution path: `llb joint-search --candidates samples/configs/models_uk.yaml --trials <n>`
  sequentially on the CUDA host (one heavyweight model at a time); CI drives the schedule with
  fake eval results.
- Acceptance gates: `make ci` green; a heavy deterministic run over at least three candidates
  shows the halving ledger eliminating candidates on the tuning split only, and the final
  scoreboard built exclusively from final-split runs; no tuning/final leakage in manifests.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) alongside sweep and
  pipeline behavior.

### knowledge-tree-prompt

Generate compact knowledge-tree system-prompt candidates from the induced ontology and graph
communities: a token-budgeted tree of the corpus domain vocabulary, key entity clusters, and
community summaries, rendered as a system-prompt block. Produce candidates inside the
prompt-system package flow, expose tree depth and token budget as tunable knobs, and A/B each
candidate against a no-tree baseline with the existing prompt-system comparison path.

- Agent status: RUN NEEDED
- Dependencies: none hard; pairs with `multi-objective-rag-tuner` if tree knobs join a study.
  Reuse `src/llb/prep/ontology/` induction output, `src/llb/graph/community.py` and
  `src/llb/graph/summary.py`, and the package flow in `src/llb/prompt_system/` documented in
  [extended workflows](current/extended-workflows.md).
- User-visible outcome: `prompt-system-prepare` emits knowledge-tree candidates next to existing
  prompt candidates, and `prompt-system-compare` reports whether a tree measurably helps this
  corpus before it is pinned.
- Scope boundary: in scope -- tree rendering from existing ontology/graph artifacts, knob
  plumbing, A/B wiring. Out of scope -- new extraction stages, ontology schema changes (the
  closed vocabulary in `docs/design/graph-ontology-schema.md` stays fixed), and judge changes.
- Data and artifact paths: candidates and comparison reports under the prompt-system run
  directory `$DATA_DIR/prompt-system/<run>/`; requires an ontology bundle or graph store from an
  existing draft/build run as input.
- Execution path: `make prompt-system-prepare PROMPT_SYSTEM_CORPUS=<dir>` with tree generation
  enabled, then `make run-eval PROMPT_SYSTEM_ID=<id>` and `make prompt-system-compare`; CI covers
  rendering and budgeting with a committed miniature ontology fixture.
- Acceptance gates: `make ci` green; rendered trees respect the token budget for every depth
  setting; a heavy A/B run on the UA fixture records the tree-vs-baseline delta with CIs (the
  delta itself is evidence, not a pass threshold).
- Documentation target: [extended workflows](current/extended-workflows.md) prompt-system
  section; [GraphRAG](current/graphrag-backend.md) for the community-summary reuse note.

### auto-rag-orchestrator

Build the end-to-end autonomous pipeline `llb auto-rag` (plus a `make auto-rag` target): corpus in,
scored optimal RAG configuration out. Stages: ingest -> ontology goldset draft -> verification
gate -> retrieval validation -> joint model + config tune -> knowledge-tree prompt candidate ->
final-split eval -> recommendation bundle. Every gate consumes a policy: `auto` resolves through
the `ScorerPolicy` seam (local judge or budget-capped frontier), `human` pauses the run and hands
the pending records to the review workbench, then resumes. The run journal must be resumable after
interruption at any stage, following the ontology pipeline journal pattern.

- Agent status: RUN NEEDED
- Dependencies: `multi-objective-rag-tuner`, `joint-model-config-search`,
  `knowledge-tree-prompt`; gate policies reuse
  [scorer policy seam](current/rigor-board-judge.md#scorer-policy-seam). Human-assisted gates
  additionally depend on `review-core-textual-workbench` (cross-section block for the `human`
  policy path only -- the `auto` path must land without it).
- User-visible outcome: one command takes a Ukrainian corpus directory and produces
  `rag_recommendation.yaml` (model, backend, serving knobs, chunking, retrieval mode, fusion,
  rerank, query prep, context budget, prompt-system id) plus a Markdown report with the score
  evidence -- fully autonomous by default, human-gated per stage on request.
- Scope boundary: in scope -- stage orchestration, gate policy plumbing, journal + resume,
  recommendation rendering. Out of scope -- new stage implementations (every stage reuses an
  existing command path) and any hosted-service behavior.
- Data and artifact paths: `$DATA_DIR/auto-rag/<run>/` containing the journal, per-stage bundle
  links, scorer ledger, `rag_recommendation.yaml`, and `report.md`; input is any corpus directory
  accepted by `ingest-corpus`.
- Execution path: `make auto-rag CORPUS=<dir> SCORER_POLICY=auto` for the autonomous lane;
  `SCORER_POLICY=human` for gated runs; CI drives the full stage graph with fakes and a
  miniature corpus fixture.
- Acceptance gates: `make ci` green; journal tests prove resume-after-kill at every stage
  boundary; a heavy deterministic run on the UA fixture corpus completes autonomously end-to-end
  and emits a recommendation bundle whose final-split scores match a manually chained run of the
  same stages.
- Documentation target: new topic file `current/auto-rag.md` plus an index row in
  [current.md](current.md); operator guide under `docs/guides/benchmarking/`.

### dynamic-corpus-refresh

Support dynamic corpora: diff the content-hash corpus manifest against the indexed state, apply
incremental chunk/embed/index updates for changed documents only across all store kinds (FAISS
and alternative vector stores, the lexical BM25 index, and the graph store), and emit a drift
report that re-runs retrieval validation on the gold set and recommends a re-tune when the
recall/MRR delta crosses a configured threshold.

- Agent status: CLEAR
- Dependencies: none. Reuse the corpus manifest from ingest, the stale-store checks in
  `src/llb/rag/store_validation.py`, and the immutable store-directory rollback behavior
  documented in [RAG core](current/rag-core.md).
- User-visible outcome: `llb refresh-index` updates stores in minutes proportional to the changed
  documents instead of a full rebuild, and tells the operator when the corpus has drifted enough
  that the tuned configuration should be re-searched.
- Scope boundary: in scope -- manifest diff, per-document incremental update paths, drift report,
  re-tune recommendation flag. Out of scope -- automatic re-tuning (the operator or the
  orchestrator decides), gold-set regeneration, and file watching/daemon behavior.
- Data and artifact paths: refreshed stores under the existing `$DATA_DIR/llb/rag/` layout with a
  new immutable store generation per refresh; drift reports under `$DATA_DIR/refresh/<run>/`.
- Execution path: `llb refresh-index --config <run-config>` after corpus edits; CI covers
  add/modify/delete document cases against small fixture stores for every store kind.
- Acceptance gates: `make ci` green; incremental refresh produces retrieval results identical to
  a from-scratch rebuild on the same corpus state (equivalence test per store kind); deletion
  propagation removes retired chunks from dense, lexical, and graph paths.
- Documentation target: [RAG core](current/rag-core.md) store lifecycle section;
  [data prep](current/data-prep.md) for the manifest-diff contract.

### review-core-textual-workbench

Build a shared review core `src/llb/review/` and a unified Textual TUI workbench on top of it,
then migrate the six existing terminal review flows (goldset verify, judge calibration rating,
external-RAG scoring, draft-compare review, knowledge-cutoff UA review, prompt-system review)
onto thin adapters over that core. The workbench gives every flow the same record model, verdict
ledger, keyboard navigation, and a consistent color scheme that visually separates data panes
(record content, evidence, metadata) from action elements (verdict keys, navigation, progress),
with dataset/record/strata progress indicators.

- Agent status: CLEAR
- Dependencies: none; `auto-rag-orchestrator` human gates consume it. Reuse the session/ledger
  logic in `src/llb/goldset/verify_session/`, `src/llb/judge/rate/session.py`,
  `src/llb/scoring/external_rag_session/`, `src/llb/cli/prep/draft_compare.py`, the
  knowledge-cutoff review chain, and `src/llb/prompt_system/review.py` as the behavior sources.
- User-visible outcome: one `llb review <ledger-or-run-dir>` entry point opens the right adapter
  automatically; reviewers learn one set of keys and one color language across every human gate,
  and each flow keeps its exact verdict semantics and ledger format.
- Scope boundary: in scope -- the review core (record model, verdict ledger API, navigation,
  theming), the Textual app, six adapters, and CLI wiring that keeps the existing per-flow
  commands working. Out of scope -- a web frontend, changes to any ledger file format, and new
  verdict semantics.
- Data and artifact paths: no new artifact roots; adapters read and write the existing per-flow
  ledger paths. Add `review = ["textual>=0.60"]` as a new optional extra in `pyproject.toml`.
- Execution path: `llb review <path>` or the existing flow commands; snapshot and interaction
  tests run headless via Textual's pilot harness in `make ci`.
- Acceptance gates: `make ci` green; per-adapter round-trip tests prove ledgers written through
  the workbench are byte-compatible with the legacy sessions; pilot-harness tests cover
  navigation, verdict entry, resume, and the data-vs-action color roles; ASCII-only output in
  logs and ledgers.
- Documentation target: [verification tooling guide](../guides/human-tooling/verification-tooling.md)
  and a new workbench section in a `current/` topic (extend
  [data prep](current/data-prep.md) verification section or add `current/review-workbench.md` if
  the material outgrows it).

## Human-Assisted Tasks

Add new human-gated work here per [Adding Future Tasks](#adding-future-tasks) when acceptance
requires human judgment or authorization.

### frontier-judge-authorization

Authorize and calibrate the frontier scorer lane against real providers. An agent builds any
missing report tooling; the human step is supplying provider keys, granting the egress consent,
setting a real spend cap, and reviewing the resulting agreement evidence.

- Agent status: HUMAN-GATED
- Dependencies: [scorer policy seam](current/rigor-board-judge.md#scorer-policy-seam). Human
  step that gates completion: the operator provides Anthropic / OpenAI / Google keys in `.env`,
  records the consent, approves the per-run budget cap, and signs off on the agreement report.
- User-visible outcome: a decision record stating whether each frontier judge is trusted for
  autonomous gates on Ukrainian data, plus calibrated default budget caps derived from measured
  cost-per-item.
- Scope boundary: in scope -- running the frontier lane on the committed UA fixture, computing
  Spearman rho for frontier-vs-human and frontier-vs-local-judge agreement, and a cost-per-item
  table per provider. Out of scope -- sending any private corpus to a provider and changing the
  headline-ranking policy.
- Data and artifact paths: agreement report and cost table under
  `$DATA_DIR/frontier-judge/<run>/`; input ratings reuse the existing human calibration ledger;
  fixture is `samples/goldsets/ua_squad_postedited_v1/`.
- Execution path: run `llb run-eval --scorer-policy frontier ...` over the calibration worksheet
  with each provider, then the agreement/cost report command; requires live provider access and
  spend, so the run stays outside CI entirely.
- Acceptance gates: report exists with rho per provider and cost-per-item with the cap math; the
  human accepts or rejects each provider for autonomous use; default caps land in the sample
  configs with the decision recorded.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) judge section and
  [product decisions](current/scope-boundaries.md) for the trust decision per provider.

### autonomous-vs-assisted-acceptance

Acceptance-test the full upgrade with a human operator: run `auto-rag` on a real Ukrainian corpus
twice -- once fully autonomous, once with human-assisted gates in the review workbench -- and have
the human judge both the reviewer experience and the recommendation quality.

- Agent status: HUMAN-GATED
- Dependencies: `auto-rag-orchestrator` and `review-core-textual-workbench` (Agent Implementation
  section -- cross-section block). Human step that gates completion: the operator performs both
  runs, reviews gated records in the workbench, measures their own throughput against the legacy
  per-flow sessions, and accepts or rejects the recommendation bundles.
- User-visible outcome: recorded evidence that the autonomous lane produces an acceptable
  recommendation without human action, and that the assisted lane's unified workbench is at least
  as fast and less error-prone than the legacy TUIs.
- Scope boundary: in scope -- the two runs, reviewer-throughput measurement (records per minute,
  correction rate), a comparison of the two recommendation bundles, and the acceptance decision.
  Out of scope -- fixing findings (each finding becomes a new forward task).
- Data and artifact paths: both run bundles under `$DATA_DIR/auto-rag/<run>/`; throughput notes
  and the acceptance record under `$DATA_DIR/auto-rag/<run>/acceptance/`.
- Execution path: `make auto-rag CORPUS=<dir> SCORER_POLICY=auto`, then
  `make auto-rag CORPUS=<dir> SCORER_POLICY=human` with workbench review at each gate; both on
  the CUDA host with a real corpus the operator owns.
- Acceptance gates: the human signs the acceptance record; both bundles are complete and
  reproducible from their manifests; throughput numbers and any usability findings are captured
  as new forward tasks before this item leaves the file.
- Documentation target: `current/auto-rag.md` acceptance evidence and
  [human evaluation guide](../guides/human-tooling/human-in-the-loop-evaluation.md).

## Adding Future Tasks

Add a task only when there is concrete forward work with enough detail for an engineer or an
agent to execute without guessing. Use a stable descriptive id such as `platform-matrix-power`
or `prompt-system-tuning`; keep the id only while work remains under it. Place it under
**Agent Implementation Tasks** if it can land to `make ci` green with fixtures/fakes (heavy
deterministic runs on the CUDA host are fine), or under **Human-Assisted Tasks** if a human
review/judgment or authorization gates completion; either way give it a `Dependencies` line and
mark any cross-section block explicitly.

Each task entry must include:

- Dependencies: prerequisite tasks (by number/id), any cross-section block, and -- for
  human-assisted tasks -- the specific human step that gates completion.
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
