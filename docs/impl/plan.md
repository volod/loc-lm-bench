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

### conflict-null-model-research

**Research task** -- the answer is not known in advance, and a negative result is a valid outcome
that must be recorded rather than worked around.

Find a defensible independent null for corpus-conflict detection, so the semantic tier can report
a real false-positive rate instead of a rank cutoff. The current calibration measures the
similarity distribution of the corpus's own comparable cross-document pairs, which contains
whatever genuine duplicates the corpus has; with the pair space enumerated exactly the null and
the observed population are the same set, empirical FDR is identically 1.000 at every threshold,
and a budget of `N` returns exactly `N` pairs by construction (measured; see
[data prep](current/data-prep.md#known-limitation-there-is-no-independent-null)). Every downstream
question an operator asks -- "is this pair worth reading?", "did tightening the threshold remove
noise or evidence?", "is this corpus dirtier than that one?" -- currently has no statistical
answer.

Candidate approaches to evaluate, cheapest first; none is known to work:

- **Cross-corpus null.** Score chunks of the target corpus against chunks of an unrelated Ukrainian
  corpus. Pairs across corpus boundaries are unrelated by construction. Risk: a domain/register
  shift makes the null too easy, understating the threshold.
- **Within-document permutation.** Destroy the semantic relationship while preserving the corpus's
  marginal geometry -- shuffle tokens or sentences within a chunk before embedding. Risk: sentence
  encoders are partly bag-of-words, so a shuffled chunk may stay close to its original and the null
  lands too high.
- **Held-out-document null.** Bootstrap over document pairs, using the fact that most DOCUMENT
  pairs share no content, to estimate a per-document-pair rather than per-chunk-pair null. Risk:
  document pairs are few, so the tail is unresolvable on a small corpus -- the same saturation
  problem already measured for chunk-pair sampling.
- **Labelled calibration set.** Use the committed `samples/corpora/conflicts_uk_v1/` planted
  relations as ground truth to fit a threshold with a real measured precision/recall curve, then
  test whether that transfers to the quickstart corpora. Risk: seven planted pairs is a very small
  fit set, and the fixture uses a hashed-BoW fake embedder in CI.

- Agent status: RUN NEEDED
- Dependencies: the calibrated threshold and the enumerated distribution are current behavior
  ([data prep](current/data-prep.md#corpus-calibrated-cosine-threshold---max-candidate-pairs)).
  Reuse `estimate_null_distribution`, `VectorSet.cross_group_similarities`, and the planted-relation
  fixture. The comparable set excludes structurally repeated metadata blocks; use the measured
  post-filter population in [data prep](current/data-prep.md#what-the-semantic-tier-excludes-and-why).
- User-visible outcome: either a null the audit can quote a real false-positive rate against, or a
  recorded finding that cosine over sentence-encoder chunk vectors cannot support one -- which
  would justify moving threshold selection to the claim tier's measured precision instead.
- Scope boundary: in scope -- constructing and comparing candidate nulls, measuring each against
  the planted fixture and both quickstart corpora, and a written verdict per approach. Out of
  scope -- changing the relation vocabulary or the tier order, and shipping any new default before
  a null demonstrably beats the rank cutoff.
- Data and artifact paths: comparison under `$DATA_DIR/corpus-conflicts/null-research/<run>/`;
  no new committed fixtures unless an approach earns one.
- Execution path: a research harness invoked per null model over both quickstart stores plus the
  fixture; CI covers each null constructor deterministically over committed vectors, with the
  heavy corpus comparison run on the CUDA host.
- Acceptance gates: each candidate null is measured on the planted fixture, where the true relation
  labels are known, and reports precision/recall at its resolved threshold; an approach is adopted
  only if it beats the current rank cutoff on the fixture AND its resolved threshold recovers the
  claim-bearing HR swept baseline without flooding goods. If none does, the negative result is
  [product decisions](current/scope-boundaries.md) and the rank-cutoff framing stays.
- Documentation target: the corpus-hygiene known-limitation section of
  [data prep](current/data-prep.md), and [product decisions](current/scope-boundaries.md) for the
  adopt-or-reject verdict.

### graph-vector-fusion-retrieval

Fuse the GraphRAG lane into retrieval instead of keeping graph an either/or backend: a fused
retriever queries the vector store (dense or hybrid) and the graph store through the shared
`.retrieve(question, k)` seam and merges the candidate lists with the existing weighted
reciprocal-rank fusion generalized to n lists. A `graph_weight` `RunConfig` knob sets the graph
share (`0.0` == vector-only), duplicate chunks returned by both lanes are deduplicated by span,
and every fused chunk keeps its exact source offsets so recall@k / MRR score the fused ranking on
unchanged rules.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse `GraphStore.retrieve` ([GraphRAG](current/graphrag-backend.md)),
  `rrf_fuse` in `src/llb/rag/lexical.py`, and the any-backend wrapper shape of
  `src/llb/rag/rerank.py`.
- User-visible outcome: multi-hop and entity-linking questions draw on graph evidence without
  giving up dense recall on factoid questions -- one retrieval configuration instead of a
  per-question backend choice.
- Scope boundary: in scope -- the fusion wrapper, the `graph_weight` knob (manifest + sweep/tuner
  fingerprint), fused `compare-retrieval` rows, and span-level dedup. Out of scope -- graph
  construction, community detection, and the closed ontology.
- Data and artifact paths: no new artifact roots; a fused run needs an existing vector store plus
  graph store (for example the quickstart pair in [RAG core](current/rag-core.md)).
- Execution path: `llb run-eval --retrieval-backend fused --graph-weight 0.3 ...` plus a fused
  row lane in `make compare-retrieval`; CI proves fusion order, weight extremes, and dedup over
  fake vector/graph stores.
- Acceptance gates: `make ci` green; `graph_weight=0.0` reproduces the vector-only ranking
  exactly; a heavy run over the quickstart accepted goldset reports dense vs graph vs fused
  recall@10 / MRR with the multi-hop and comparative question-type slices broken out.
- Documentation target: [GraphRAG](current/graphrag-backend.md) and the hybrid-retrieval section
  of [RAG core](current/rag-core.md).

### query-prep-hyde-decompose

Add two model-backed steps to the `src/llb/rag/query_prep/` pipeline: `hyde` embeds a short
local-model hypothetical answer in place of the raw question on the dense side (the lexical side
keeps the raw query), and `decompose` splits a multi-part Ukrainian question into sub-queries,
retrieves per sub-query, and fuses the per-sub-query candidate lists with weighted RRF. Both run
through the same backend endpoint seam as the existing `rewrite` step, record their generated
text per case, and never touch stored corpus text. Extend the `validate-retrieval
--query-prep-ab` report to accept an endpoint so model-backed steps get the same per-step
recall/MRR delta attribution as the pure steps.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the query pipeline, A/B report, and step-dependency resolution in
  [RAG core](current/rag-core.md#query-side-processing-uk-query-processing) plus `rrf_fuse`.
- User-visible outcome: an evidence-backed answer to whether hypothetical-answer embedding or
  sub-question fusion recovers the broadly-phrased-question recall misses that dense-only
  retrieval leaves on real Ukrainian corpora.
- Scope boundary: in scope -- the two steps, endpoint wiring in `runner_setup`, the A/B
  extension, and per-case provenance fields in `scores.jsonl`. Out of scope -- multi-turn agentic
  retrieval, corpus-side changes, and any default-on change (both steps stay opt-in).
- Data and artifact paths: no new roots; A/B reports beside the existing `validate-retrieval`
  output; generated queries recorded in the run bundle per case.
- Execution path: `make validate-retrieval QUERY_PREP=normalize,hyde QUERY_PREP_AB=1` with new
  endpoint knobs (`QUERY_PREP_MODEL=<m> QUERY_PREP_BACKEND=<b>`); `make run-eval QUERY_PREP=...`;
  CI drives both steps over a fake endpoint and fake store.
- Acceptance gates: `make ci` green; an empty lane stays an exact no-op; output is deterministic
  under a fixed fake completion; a heavy A/B over the quickstart accepted goldset against the
  full-corpus store attributes each step's recall@10 / MRR delta and records the verdict.
- Documentation target: [RAG core](current/rag-core.md) query-side processing.

### rag-vs-long-context-ablation

Build `llb compare-context-strategies` (`make compare-context-strategies`): score one model on
the final split under three context lanes -- `closed_book` (no retrieved context; the model
answers from its weights), `rag` (the run configuration as-is), and `long_context` (the item's
full source document laid into the prompt, budget-checked through `fits_context`; an item whose
document exceeds the model's usable context is counted as a skip, never silently truncated). Each
lane persists an ordinary run bundle; the comparison report renders per-lane objective plus two
derived numbers -- retrieval uplift (`rag - closed_book`) and long-context delta
(`long_context - rag`) -- and flags per item when the closed-book answer already matches the
reference (a contamination / parametric-knowledge signal).

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the `run-eval` seams, `fits_context` / `context_budget`
  ([RAG core](current/rag-core.md#context-budget)), and the report shape of `compare-retrieval`.
- User-visible outcome: the operator learns whether RAG pays for itself per model on their corpus
  -- how much a Ukrainian-tuned model (MamayLM, Lapa) already answers closed-book, and whether
  whole-document stuffing beats chunked retrieval within that model's usable context.
- Scope boundary: in scope -- lane orchestration, prompt assembly for the two new lanes, the
  report, and the contamination flag. Out of scope -- any ranking-policy change (the lanes are
  diagnostics; the `rag` lane stays the leaderboard row) and context-window extension tricks.
- Data and artifact paths: lane bundles under the standard `$DATA_DIR/run-eval/`; comparison
  report under `$DATA_DIR/context-ablation/<run>/{report.md,comparison.json}`.
- Execution path: `make compare-context-strategies MODEL=<m> BACKEND=<b> GOLDSET=<gs>`; CI drives
  all three lanes over a fake endpoint and the committed fixtures.
- Acceptance gates: `make ci` green; the `rag` lane's per-case scores are identical to a plain
  `run-eval` of the same configuration; a heavy run over the committed UA fixture on at least two
  roster models records the three-lane table and the contamination rate.
- Documentation target: a new [RAG core](current/rag-core.md) subsection; a
  [product decisions](current/scope-boundaries.md) note if a lane is rejected as a default.

### table-aware-chunking

Add a `table` strategy to `src/llb/rag/chunking/`: chunk boundaries never split a markdown table
row, a table that fits `size` stays one chunk carrying its nearest heading breadcrumb, and an
oversized table splits between row blocks with the header row's offsets recorded as additive
`metadata.table_header_span` -- chunk text stays a verbatim corpus slice with exact offsets.
Non-table text routes through the `recursive` splitter. Extend `compare-retrieval` with a
per-question-type breakdown (joined from `item_provenance.jsonl` when the sidecar exists) so the
numeric and comparative slices -- where tables carry the answers in converted Ukrainian PDF
corpora -- are scored beside the aggregate.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the chunking dispatch seam (`chunk_spans`), the markdown table output
  of the PDF conversion lane ([data prep](current/data-prep.md)), and the question-type taxonomy
  in the draft sidecars.
- User-visible outcome: numeric and comparative questions whose evidence lives in tables stop
  losing recall to mid-table chunk cuts, and the per-type breakdown shows exactly which question
  slice a chunking change helps or hurts.
- Scope boundary: in scope -- the strategy, tuner registration behind `--extended-chunkers`, and
  the per-type `compare-retrieval` breakdown. Out of scope -- cell-level table QA, corpus text
  rewriting, and HTML tables.
- Data and artifact paths: per-strategy stores under the existing comparison layout
  `$DATA_DIR/llb/rag/<strategy>/`; no new roots.
- Execution path: `make build-index CHUNK_STRATEGY=table`; `make compare-retrieval
  CHUNK_STRATEGIES=table,recursive,sentence GOLDSET=<gs>`; CI covers offset round-trips and
  row-boundary alignment on a committed markdown-table fixture.
- Acceptance gates: `make ci` green; every chunk stays offset-exact under `validate-goldset`;
  a heavy comparison over the quickstart accepted goldset reports aggregate plus numeric-slice
  recall@10 / MRR against `recursive` and `sentence`.
- Documentation target: [RAG core](current/rag-core.md) chunking strategies and the
  [data prep](current/data-prep.md) chunking list.

### ua-embedder-domain-finetune

Fine-tune the pinned multilingual E5 embedder on the operator's corpus: export contrastive
(question, gold-chunk) pairs from tuning-split gold items only (positives are chunks overlapping
the item's gold spans; hard negatives come from the BM25 lexical index), train with a
sentence-transformers contrastive objective behind lazy imports, and emit a tuned-embedder
directory whose manifest records the base model, dataset digest, item ids, and split counts. A
split guard refuses pairs from calibration or final ids (the `assert_tuning_only` discipline from
the LoRA hparam search). `compare-embeddings` accepts the tuned directory as a candidate so
uplift is measured by the standard source-span metric on the held-out final split, and the
store/query embedder fingerprint guard keeps a tuned-embedder store from being queried by any
other encoder.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the embedder conventions and bake-off in
  [RAG core](current/rag-core.md#embedder-conventions-and-bake-off), the lexical index for hard
  negatives, and the split-guard pattern in `src/llb/finetune/hparam_search/`.
- User-visible outcome: a corpus-adapted Ukrainian retriever the operator can adopt with measured
  final-split evidence, closing the recall gap on domain terms the general E5 encoder misses.
- Scope boundary: in scope -- pair export, the trainer, the manifest, bake-off integration, and
  the split guard. Out of scope -- cross-encoder (reranker) fine-tuning, generation-model
  fine-tuning (owned by the existing finetune lane), and hosted training.
- Data and artifact paths: pair datasets and tuned models under
  `$DATA_DIR/finetune-embedder/<model-slug>/<timestamp>/`; evaluation through the existing
  `$DATA_DIR/compare-embeddings/` layout.
- Execution path: `make finetune-embedder GOLDSET=<gs> CORPUS=<dir>` then
  `make compare-embeddings` with the tuned directory added as a candidate; CI uses a fake trainer
  plus the hashed-BoW embedder pattern from the curation tests, no GPU.
- Acceptance gates: `make ci` green; the guard refuses a pair set naming calibration/final ids;
  a heavy CUDA run trains on the quickstart tuning split and reports tuned-vs-base recall@10 /
  MRR on the held-out final split with an explicit adopt-or-keep-base verdict.
- Documentation target: [RAG core](current/rag-core.md) embedder section and
  [extended workflows](current/extended-workflows.md) for the trainer lane.

### ua-model-roster-refresh

Refresh the Ukrainian candidate roster: survey current UA-capable open-weight instruct releases
(public UA benchmarks such as the lang-uk leaderboard and INSAIT releases are the candidate
filter, per the spec), add each viable model to `samples/configs/models_uk.yaml` with per-backend
`sources:` (Ollama tag, vLLM repo with multi-quant records, GGUF) plus license notes, extend the
GPU-tier serving manifests where a new family earns a tier target, and place the additions on the
scoreboard with a joint-search run over the committed fixture.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the resolver multi-quant conventions in
  [platform matrix](current/platform-vector-matrix.md) and the joint-search schedule in
  [evaluation rigor](current/rigor-board-judge.md).
- User-visible outcome: recommendations compare against the current generation of UA-capable
  models instead of a frozen roster; every new entry resolves, fit-plans, and sweeps exactly like
  the existing ones.
- Scope boundary: in scope -- registry entries, prep fixtures, resolver unit fixtures per new
  entry, serving-tier additions, and one joint-search evidence run. Out of scope -- fine-tuning
  the new models and hosted/API-only models (local serving only).
- Data and artifact paths: `samples/configs/models_uk.yaml` and the serving-tier manifests;
  joint-search evidence under `$DATA_DIR/joint-search/<run>/`.
- Execution path: `make list-models`, `llb resolve-models`, then
  `make joint-search JOINT_SEARCH_TRIALS=<n>` on the CUDA host; CI covers resolver and planner
  fixtures for every added entry without downloads.
- Acceptance gates: `make ci` green; every added entry resolves offline in fixtures and either
  resolves to a runnable backend on the 16 GiB host or records an explicit larger-tier
  requirement; the joint-search scoreboard covers each addition or names its host-fit skip
  reason.
- Documentation target: [platform matrix](current/platform-vector-matrix.md) roster and tier
  notes; [evaluation rigor](current/rigor-board-judge.md) for the scoreboard evidence.

### ua-query-robustness-bench

Benchmark end-to-end robustness to realistic Ukrainian input noise: for each verified gold item
generate seeded deterministic query variants in three classes -- Latin-typed transliteration
(inverting the injective romanization map), apostrophe-variant plus mixed-script homoglyph
substitution (reusing the security suite's folding tables), and keyboard-adjacent Cyrillic typos
at a configured rate -- and score each class end to end. Variant rows follow the probe pattern:
they land in `robustness.jsonl` beside the run bundle and never enter `scores.jsonl` or the
correctness aggregates. The report gives per-class objective and recall deltas against the clean
run (separating retrieval degradation from generation degradation) and re-runs each class with
the `query_prep` mitigation lane enabled, so the mitigation's recovery is measured rather than
assumed.

- Agent status: RUN NEEDED
- Dependencies: none. Reuse the pure helpers in `src/llb/rag/query_prep/`, the normalization and
  homoglyph tables in `src/llb/eval/common.py` and `src/llb/scoring/security.py`, and the probe
  persistence pattern of the insufficient-context probe
  ([evaluation rigor](current/rigor-board-judge.md)).
- User-visible outcome: a per-model robustness profile for the messy queries real Ukrainian users
  type (transliteration, mixed script, typos), plus measured evidence for when the query-prep
  mitigation steps should be enabled by default.
- Scope boundary: in scope -- deterministic variant generation, probe execution, the delta
  report, and the mitigation on/off comparison. Out of scope -- model-generated surzhyk or
  Russian-language paraphrase variants (they need a drafting plus human-verification lane; add
  a follow-up task if the deterministic classes prove insufficient) and any headline-ranking
  change.
- Data and artifact paths: `$DATA_DIR/query-robustness/<run>/{report.md,robustness.jsonl}`;
  clean baselines stay ordinary run bundles.
- Execution path: `make bench-query-robustness MODEL=<m> BACKEND=<b> GOLDSET=<gs>`; CI drives
  variant determinism and the report over a fake endpoint and fake store.
- Acceptance gates: `make ci` green; variants are deterministic per seed and never leak into
  correctness aggregates; a heavy run over the committed UA fixture on at least two roster
  models records the per-class deltas and the mitigation recovery table.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) beside the other
  probes, with a pointer from [RAG core](current/rag-core.md) query-side processing.

## Human-Assisted Tasks

Add new human-gated work here per [Adding Future Tasks](#adding-future-tasks) when acceptance
requires human judgment or authorization.

### corpus-conflict-resolution-review

Review the unresolved semantic conflict candidates through the workbench, then feed the accepted
ledger back into the resolver and repeat the retrieval plus verified answer-quality comparison.
The resolver behavior and the reason semantic candidates have no automatic suppression authority
are current behavior in
[data prep](current/data-prep.md#corpus-conflict-resolution-corpus-conflict-resolution).

- Agent status: HUMAN-GATED
- Dependencies: the resolution lane is current behavior. Human step that gates completion: an
  authorized corpus reviewer chooses `keep_both`, `drop_a`, or `drop_b` for every escalated row
  and signs off on the resulting suppression directives before application.
- User-visible outcome: an accepted or rejected suppression policy backed by reviewed conflict
  labels and a repeatable effect report, instead of adopting semantic similarity candidates as
  deletions.
- Scope boundary: in scope -- workbench review, accepted-ledger application, the same before/after
  metrics, and an adopt-or-revert decision. Out of scope -- changing detector thresholds,
  rewriting source text, or adding the resolver to auto-rag before the reviewed run supports it.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/resolution_review.jsonl`,
  `plan.json`, `conflict_overlay.json`, and `effect.md`; no new artifact root.
- Execution path: `make review-workbench REVIEW_PATH=<resolution-review-jsonl>`, then
  `make resolve-corpus-conflicts FINDINGS=<findings-jsonl> REVIEWED=<resolution-review-jsonl>
  APPLY=1 STORE=<store-dir> GOLDSET=<goldset-jsonl>` and repeat the fixed verified objective run.
- Acceptance gates: every review row has a decision; the regenerated plan has no unresolved
  records; rollback still restores the exact baseline; the human accepts only if retrieval and
  verified objective metrics do not regress.
- Documentation target: the resolution evidence subsection of [data prep](current/data-prep.md)
  and the conflict adapter notes in [review workbench](current/review-workbench.md).

### frontier-judge-authorization

Authorize the frontier scorer lane against real providers. The report tooling is current behavior
([frontier judge agreement and cost report](current/rigor-board-judge.md#frontier-judge-agreement-and-cost-report));
what remains is entirely the human authorization and the judgment it produces.

- Agent status: HUMAN-GATED human_decision: panding
- Command: once a real Anthropic key is in `.env` (~$0.40, 86 items):

  ```bash
  make frontier-judge-agreement \
    FRONTIER_JUDGE_MODELS=anthropic/claude-sonnet-4-5 \
    FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=1.00
  ```

- Dependencies: the agreement lane is current behavior; it runs on the 86-row calibration
  worksheet `calibration/ua_squad_postedited_v1.csv` (every row carries both a human and a local
  judge rating). Human step that gates completion: the operator puts a real Anthropic / OpenAI /
  Google key in `.env` (all three are currently blank placeholders, so no live run is possible),
  approves the per-run spend cap, and signs off on the resulting report.
- User-visible outcome: a decision record stating whether each frontier judge is trusted for
  autonomous gates on Ukrainian data, plus default budget caps derived from measured
  cost-per-item rather than from a guess.
- Scope boundary: in scope -- running the existing lane on the committed UA fixture, reviewing
  the rho and cost tables, recording an accept/reject per provider, and landing the resolved caps
  in the sample configs. Out of scope -- sending any private corpus to a provider, changing the
  headline-ranking policy, and any further report tooling.
- Data and artifact paths: `$DATA_DIR/frontier-judge/<run>/`; fixture is
  `samples/goldsets/ua_squad_postedited_v1/`.
- Execution path: `make frontier-judge-agreement FRONTIER_JUDGE_MODELS=<id>[,<id>...]
  FRONTIER_EGRESS_CONSENT=1 FRONTIER_MAX_USD=<cap>`; needs live provider access and spend, so it
  stays outside CI entirely.
- Acceptance gates: the report carries a non-`n/a` rho per provider against both references and a
  priced cost-per-item with cap math; the human replaces `human_decision: pending` with an accept
  or reject per provider; the accepted caps land in the sample configs with the decision recorded.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) judge section and
  [product decisions](current/scope-boundaries.md) for the trust decision per provider.

### frontier-judge-retrieved-context-agreement

Optional. Re-measure frontier-vs-human judge agreement with *retrieved* contexts instead of the
gold-span windows the authorization lane uses. The current lane deliberately holds retrieval
constant by grounding each item on a window of its gold source document, which isolates judge
behavior but also hands the judge cleaner evidence than a scored run ever gives it. A judge that
ranks well on oracle context may rank differently when the context contains distractors or misses
the answer entirely -- exactly the cases where an autonomous gate matters most. Add a context
source switch to `load_agreement_items` that pulls each item's top-k retrieved chunks from an
existing store, then report both grounding modes side by side so the gap is visible.

- Agent status: HUMAN-GATED
- Dependencies: blocked by `frontier-judge-authorization` (needs the same provider keys and
  spend). Reuse the agreement lane in `src/llb/scoring/frontier_agreement/` and the store-loading
  seam used by the context-position probe.
- User-visible outcome: evidence for whether frontier-judge trust measured on oracle context
  transfers to the noisy contexts a real scored run produces.
- Scope boundary: in scope -- the retrieved-context source, the two-mode comparison, and the
  delta. Out of scope -- changing the default grounding of the authorization lane before the
  comparison says it should.
- Data and artifact paths: no new roots; a second grounding mode inside the existing
  `$DATA_DIR/frontier-judge/<run>/` bundle.
- Execution path: `make frontier-judge-agreement` with a grounding-mode knob plus a built store;
  CI covers mode selection over a fake store and fake completers.
- Acceptance gates: `make ci` green; the gold-span mode reproduces the current numbers exactly; a
  live run reports rho under both modes with the delta called out.
- Documentation target: the frontier-judge agreement subsection of
  [evaluation rigor](current/rigor-board-judge.md).

### autonomous-vs-assisted-acceptance

Acceptance-test the full upgrade with a human operator: run `auto-rag` on a real Ukrainian corpus
twice -- once fully autonomous, once with human-assisted gates in the review workbench -- and have
the human judge both the reviewer experience and the recommendation quality.

- Agent status: HUMAN-GATED
- Dependencies: the autonomous lane is current behavior ([Auto-RAG](current/auto-rag.md));
  assisted review uses the [review workbench](current/review-workbench.md). Human step that gates
  completion: the operator
  performs both
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
