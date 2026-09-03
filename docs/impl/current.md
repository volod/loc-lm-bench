# loc-lm-bench Current Implementation

This index is for agents and maintainers who need the current implementation shape: what exists,
where it lives, how the major flows run, and why the important design choices were made.

For the product design, read [`docs/design/spec.md`](../design/spec.md). For future work, read
[`docs/impl/plan.md`](plan.md).

The three files are joined by the
[capability registry](../design/spec.md#capability-registry): each capability names what evaluates
it and links to the area page below that documents it, and every open plan task declares the
capability it serves. `make lint-spec-plan` fails when a task serves a capability the spec does not
register, when a shipped capability has no implementation docs, or when the plan's groups drift out
of the registry's order. A capability found while implementing is added to the spec through
[Extending this specification](../design/spec.md#extending-this-specification) before it becomes
tasks here.

## How this documentation is organized

Three levels, so a search stops at the smallest page that answers the question:

1. **This index** -- the areas, and which one owns a question.
2. **An area page** (`current/<area>.md`) -- one screen of orientation plus the tree of pages under
   it. Larger areas own a directory of pages; smaller ones are a single page.
3. **A topic page** (`current/<area>/<topic>.md`) -- one subject: what was built, where it lives
   (modules / commands / tests), how to run it, and the measured result that settled it.

Evidence stays with the decision it supports rather than in an appendix, so a topic page ends in
the runs behind it.

Two neighbours sit outside this tree: [plan.md](plan.md) holds work that remains, and
[future-research.md](future-research.md) holds questions investigated to a negative answer -- what
closed each one and what would make it worth reopening.

## Areas

| Area | Owns | Shape |
| --- | --- | --- |
| [Overview](current/overview.md) | System shape, setup, repo layout, artifact roots | page |
| [Artifact contracts](current/artifact-contracts.md) | Versioned record identity, compatibility dispatch, dataset bindings, generated schemas and catalog, and the data-prep families on them | 2 pages |
| [Data prep](current/data-prep.md) | Gold data, ingestion, drafting, verification, corpus hygiene | 10 pages |
| [RAG core](current/rag-core.md) | The retrieve -> generate -> score pipeline and every measured decision in it | 14 pages |
| [Entity resolution](current/entity-resolution.md) | Probabilistic record linkage: the shared identity seam and the gold-item, graph-node, and document-edition lanes on it | page |
| [GraphRAG](current/graphrag-backend.md) | Knowledge-graph retrieval and graph-vs-vector evidence | 5 pages |
| [Extended workflows](current/extended-workflows.md) | Agentic harnesses, agent context policies, prompt systems, fine-tuning | 14 pages |
| [Evaluation rigor](current/rigor-board-judge.md) | Model resolution, sweeps, tuning, joint search, board, judge, miss analysis | 5 pages |
| [Auto-RAG](current/auto-rag.md) | Autonomous corpus-to-RAG orchestration, resume, verification, recommendation | page |
| [Robotics RAG](current/robotics-rag.md) | Pinned episode evidence and device contracts, proposal gating, emulator safety, and paired operation evaluation | 4 pages |
| [Review workbench](current/review-workbench.md) | Unified terminal review UI, adapters, keys, ledger compatibility | page |
| [Backend telemetry](current/backend-telemetry.md) | vLLM launcher, telemetry fields, backend build rules | page |
| [Robust backends](current/robustness-ontology-backends.md) | VRAM planning, contention guard, llama.cpp, ontology drafting, the axiom layer over the ledger | page |
| [Platform matrix](current/platform-vector-matrix.md) | Backend matrix, power telemetry, vector-store adapters | page |
| [Category suite](current/category-benchmark-suite.md) | Security, tooling, agentic, summarization, structured, text analysis | page |
| [Knowledge cutoff](current/knowledge-cutoff.md) | Effective real-world knowledge cutoff for local models | page |
| [Prompt templates](current/prompt-templates.md) | Prompt template registry and review workflow | page |
| [Host validation](current/host-validation.md) | Host acceptance checklist, runtime version floors, and the quality/complexity/shell gates | 7 pages |
| [Model roster](current/model-roster.md) | Model family/generation register, the generated family tables, the upstream currency probe, and generation upgrades | page |
| [Product decisions](current/scope-boundaries.md) | Settled scope and decision motivation | page |

## Frequent lookups

Questions that land deeper than an area page. Everything else is one hop from the area index above.

### Retrieval and scoring

| Need | Read |
| --- | --- |
| Headline token precision/recall/found-rate decomposition and declared format weight | [Generation graph and scoring](current/rag-core/scoring.md#headline-decomposition-and-declared-ranking-policy) |
| Whether an answer carried BOTH facts of a multi-hop item, not just how much it overlaps the reference | [Answer-side gold-span coverage](current/rag-core/scoring.md#answer-side-gold-span-coverage-answer-side-span-coverage-metric) |
| Whether a model leaked its reasoning into the answer or replied in the wrong language, and which roster tags need suppression | [Response-integrity guard](current/rag-core/scoring.md#response-integrity-guard-thinking-suppression-and-answer-language-guard) and [the per-tag verdicts](current/backend-telemetry.md#thinking-suppression-verdicts-per-roster-tag) |
| Whether RAG pays for itself: closed-book vs rag vs long-context lanes | [Context ablation](current/rag-core/context-ablation.md) |
| How much repeated text an index still holds, and which collapse tier to build with | [Retrieval store](current/rag-core/retrieval-store.md#near-duplicate-residue-and-the-collapse-tiers) |
| Cold/warm encoder throughput on CUDA hosts (load vs compile vs steady encode) | [Embedders](current/rag-core/embedders.md#blackwell-encoder-throughput-decomposition) |
| Cheap CUDA embedder (e5-small) when quality is flat on a 12 GiB host | [Embedders](current/rag-core/embedders.md#blackwell-sub-base-encoder-roster-e5-small) |
| Which embedder to adopt, and whether a rank-quality gain is worth it | [First-hit-rank adoption bar](current/rag-core/first-hit-rank-adoption.md) |
| Which cross-encoder reranker to run, and what it costs in latency and VRAM | [Reranker bake-off](current/rag-core/reranker-bakeoff.md) |

### Reading a verdict

| Need | Read |
| --- | --- |
| Whether a paired verdict may be read at all, and the item count an unreadable one needs | [Minimum-evidence gate](current/rag-core/paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading) |
| Family-wise error control when a verdict selects a grid row, cell, or candidate | [Selection-adjusted grid verdicts](current/rag-core/paired-verdicts.md#selection-adjusted-grid-verdicts) |
| Predeclared MDE, paired-power item counts, and realized sensitivity in comparison lanes | [Paired-power contract](current/rag-core/paired-verdicts.md#paired-power-contract-for-comparison-lanes) |
| What a withdrawn reading needs before it may be read again | [The re-decision](current/rag-core/paired-verdicts.md#the-re-decision-what-a-withdrawn-reading-needs) |

### Agent context policy

The compact-versus-cap chain in reading order: the policies, the winner on a long transcript, where
the crossover sits, what has been published from it, and what a constant change invalidates.

| Need | Read |
| --- | --- |
| Aggregate-safe agent observation trim + compact finish recovery (count-slice) | [Agent context policies](current/extended-workflows/agent-context-policies.md#aggregate-safe-trimming) |
| Active compact vs observation-cap on long and memory-dependent transcripts, including summarizer cost | [Compact versus cap](current/extended-workflows/compact-versus-cap.md) |
| Where compact stops repaying its summary call: cap-fitting cost crossover over depth and prompt guard | [Crossover geometry](current/extended-workflows/crossover-geometry.md#cap-fitting-boundary-surface) |
| Compact routing on one axis: trigger (`compact_share * guard`) and the fold step it selects | [Crossover geometry](current/extended-workflows/crossover-geometry.md#the-routing-rule-lives-on-the-trigger-axis) |
| The compact crossover as a fold-step boundary ("fold no later than step k") rather than a char guard | [Crossover geometry](current/extended-workflows/crossover-geometry.md#the-crossover-is-a-fold-step-not-a-char-guard) |
| The compact summarize call's input bound (`summary_input_cap`), unavoidable elision cost, and entry-aware prototype | [Summary-input bounds and elision](current/extended-workflows/summary-input-elision.md) |
| Agent context-policy constant sweep (cap / head-share / keep_last_n pin-or-expose) | [Constant sweeps](current/extended-workflows/context-policy-constants.md#agent-context-policy-constants) |
| keep_last_n on longer transcripts (medium-search keep grid) | [Constant sweeps](current/extended-workflows/context-policy-constants.md#keep_last_n-on-longer-transcripts) |

### Published numbers and their provenance

| Need | Read |
| --- | --- |
| Model-free audit of which published compact evidence a summarize-bound change can move, and the crossovers restated under the shipped bound | [Published values](current/extended-workflows/published-values.md) |
| Resolving a published number back to its run artifact and field (provenance pointers, committed run aggregates and their content pins, one refusal naming every value that no longer resolves) | [Committed aggregates and pins](current/extended-workflows/published-values.md#committed-aggregates-content-pins-and-the-growth-budget) |
| Declaring what a published number is derived from (`derived_from`) and the registered arithmetic over it (`operation`), and marking a derived value not-judged against the moved measurement at the root of what it rests on | [Derivation and arithmetic](current/extended-workflows/published-values.md#what-a-published-value-declares-it-is-derived-from) |
| Which published agentic numbers any context-policy constant change invalidates (prompt-sequence replay, no GPU) | [Policy-constant audit](current/extended-workflows/policy-constant-audit.md#what-a-policy-constant-change-invalidates) |
| The committed geometry where a compound policy change and a per-field reading disagree (the compound guarantee's own test) | [Policy-constant audit](current/extended-workflows/policy-constant-audit.md#the-compound-guarantee-has-a-geometry-that-tests-it) |
| Every pair of auditable policy constants, which one couples enough to separate the two readings, and what blocks the other fourteen | [Policy-constant audit](current/extended-workflows/policy-constant-audit.md#one-pair-separates-and-the-other-fourteen-are-answered) |
| The CI gate that pins the shipped context-policy constants and fails a drift with its re-run scope | [Policy-constant audit](current/extended-workflows/policy-constant-audit.md#the-audit-runs-in-ci-on-the-act-that-creates-the-problem) |
