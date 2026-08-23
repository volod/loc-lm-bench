# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every task line in this file must describe work that remains. Current behavior,
operator workflows, durable evidence, and design decisions live in [`current.md`](current.md) and
the topic files under [`current/`](current/). The product spec lives in
[`docs/design/spec.md`](../design/spec.md).

Every task serves a capability registered in
[the spec](../design/spec.md#capability-registry), and the capability groups below appear in the
registry's row order -- the implementation line. That is the whole scheduling rule: **take the first
task of the earliest group that still has one.** Which entry is best written, most recently added,
or most sharply specified does not enter into it.

Work that no registered capability covers is not blocked -- it is a signal the spec has a gap. Close
the gap first: [Extending this specification](../design/spec.md#extending-this-specification) is the
six-step lifecycle for turning a discovery into a registered capability with a declared evaluation,
and only then into tasks here. `make lint-spec-plan` enforces the join in both directions.

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

Both sections carry the same capability groups in the same order, so the two run in parallel: an
agent works its section's line while a human-gated item waits on its reviewer.

Task ids are stable and never change. Every task carries a `Serves` line naming its capability and a
`Dependencies` line. Within a group the same logic applies one level down:

1. a task another task names as a prerequisite comes first;
2. then work that changes what later tasks measure -- a chunker, an encoder roster, or a detector
   change invalidates every reading taken against the old one;
3. then required before `(optional)`;
4. then cheap before expensive: a `CLEAR` task costs a fixture test, a `RUN NEEDED` task costs a GPU
   run, so a run of CLEAR work lands as one pass rather than interleaved.

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
- **RESEARCH** -- in addition to Agent status. It marks the task implementation as unknown in
advance, and a negative result is a valid outcome that must be recorded rather than worked around.

**`(optional)` is binding within a group.** It marks work already judged not to be the next thing
worth doing for that capability, so it sorts behind every non-optional task in its own group. It
does NOT reorder the line: an optional task of an earlier capability still precedes a required task
of a later one, because the trust chain is what the order encodes.

## Agent Implementation Tasks

Agent-buildable work, grouped by the capability it serves and ordered by the
implementation line in the [capability registry](../design/spec.md#capability-registry).
Take the first task of the earliest group that still has one; see
[Adding Future Tasks](#adding-future-tasks) before adding one.

### Answer scoring -- `answer-scoring`

#### thinking-suppression-and-answer-language-guard

`qwen3:30b` answers a Ukrainian benchmark prompt with first-person English deliberation ("Okay, I
need to explain...") even though the launcher sends Ollama's native `think: false` on every call,
so the thinking suppression the manifest relies on is not sufficient for that tag. Two scoring
risks follow: reasoning text inflates the generated-token count that throughput and cost are
derived from, and an English answer to a Ukrainian prompt is scored as content rather than caught
as an off-language response. Add a per-response guard that detects a leaked-reasoning prefix and a
dominant-script/language mismatch against the prompt, record both as named per-case flags in the
run bundle beside the existing reliability fields, and decide per model whether suppression needs a
prompt-level instruction on top of the API flag. Evidence for the observation is in the full-roster
throughput baseline in [backend telemetry](current/backend-telemetry.md).

- Serves: `answer-scoring` -- [Answer scoring](../design/spec.md#scoring-policy)
- Agent status: RUN NEEDED
- Dependencies: the throughput protocol in
  [backend telemetry](current/backend-telemetry.md#telemetry-fields) and the correctness/reliability
  fields in [RAG core](current/rag-core/scoring.md#scoring).
- User-visible outcome: a run bundle shows how many answers leaked reasoning or answered in the
  wrong language, per model, instead of silently scoring them as ordinary content.
- Scope boundary: in scope -- the detection flags, their manifest fields, and a per-model
  suppression verdict. Out of scope -- rewriting the judge or changing the objective's definition.
- Execution path: re-run the roster throughput protocol capturing generations, then a bounded
  `run-eval` cell per affected tag.
- Acceptance gates: `make ci` green with injected fake generations covering leaked-reasoning,
  off-language, and clean answers; the flags appear in the persisted manifest; every roster tag
  carries an explicit suppression verdict, including the tags where no leak was observed.
- Documentation target: the roster baseline in
  [backend telemetry](current/backend-telemetry.md) and the scoring fields in
  [RAG core](current/rag-core/scoring.md#scoring).

#### retrieved-document-long-context-lane

The measured long-context lane is oracle-grounded -- it reads the item's own gold `doc_id`s, so it
sizes a CEILING and cannot be adopted
([product decisions](current/scope-boundaries.md#context-ablation-lanes-stay-diagnostic)). Add the
shippable sibling: a `retrieved_document` context strategy that takes the top-ranked RETRIEVED
chunk's document (no gold label), lays that whole document into the prompt under the same budget
check and `context_overflow` skip rule, and reports beside the existing lanes. The measured
oracle-versus-rag gap (+0.142 / +0.080 objective on two roster models) then splits into the part
an operator can actually capture by widening the unit of retrieval from a chunk to its document,
and the part that was pure oracle advantage.

- Serves: `answer-scoring` -- [Answer scoring](../design/spec.md#scoring-policy)
- Agent status: RUN NEEDED
- Dependencies: none. Reuse the context-source seam, `fits_context_chars`, and the comparison in
  [RAG core](current/rag-core/context-ablation.md#context-ablation-does-rag-pay-for-itself-rag-vs-long-context-ablation).
- User-visible outcome: the operator learns whether "retrieve the chunk, send the document" is a
  real configuration worth shipping, or whether the long-context gain was the gold label all along.
- Scope boundary: in scope -- the strategy, its document-selection rule (top-1 versus top-k
  distinct documents), the budget/skip path, and a four-lane comparison. Out of scope -- changing
  the chunker, the ranking policy, and context-window extension tricks.
- Data and artifact paths: `$DATA_DIR/context-ablation/<run>/`.
- Execution path: `make compare-context-strategies CONTEXT_LANES=closed_book,rag,retrieved_document,long_context`;
  CI covers document selection and the skip rule over fake lane stores.
- Acceptance gates: `make ci` green; the three existing lanes reproduce their current rows exactly;
  a heavy run on both scored roster models reports the four-lane table with paired intervals and an
  explicit adopt-or-reject verdict for the new lane.
- Documentation target: the context-ablation section of [RAG core](current/rag-core.md) and the
  diagnostic-lane boundary in [product decisions](current/scope-boundaries.md).

#### context-ablation-question-type-slices (optional)

The context ablation slices by question type, but the committed UA fixture ships no
`needle_items.jsonl` sidecar, so every heavy run so far reported ONE pooled number per lane ([RAG
core](current/rag-core/context-ablation.md#context-ablation-evidence)). Pooling hides the question
the lane is most useful for: retrieval almost certainly pays for itself unevenly -- a factoid whose
answer is one span versus a comparative or numeric question whose evidence is scattered. Run the
ablation on a gold set that HAS the sidecar (the quickstart-PDF accepted goldset, or a drafted
multi-hop bundle) so the uplift and the long-context delta are reported per slice, and record which
slices retrieval fails to pay for.

- Serves: `answer-scoring` -- [Answer scoring](../design/spec.md#scoring-policy)
- Agent status: RUN NEEDED
- Dependencies: none. The slicing is already wired; this needs a labeled item set and the run.
  Question-type labels come from the needle sidecar
  ([data prep](current/data-prep.md)).
- User-visible outcome: the operator learns WHICH questions retrieval pays for on their corpus,
  instead of one pooled average over a mixed set.
- Scope boundary: in scope -- the run, the per-slice reading, and a verdict per slice. Out of
  scope -- new metrics, new lanes, and any ranking-policy change.
- Data and artifact paths: `$DATA_DIR/context-ablation/<run>/`.
- Execution path: `make compare-context-strategies GOLDSET=<sidecar-bearing goldset> CORPUS=<dir>`
  on the CUDA host; no new CI coverage.
- Acceptance gates: `make ci` green; the report carries a non-empty slice table with paired
  intervals per slice and states which slices the uplift interval fails to clear zero on.
- Documentation target: the context-ablation evidence subsection of
  [RAG core](current/rag-core.md).

#### closed-book-decoding-stability (optional)

A closed-book score is a noisier measurement than a grounded one: two identical invocations of the
same lane on the same 82 items differed on 11 answers and moved the lane mean 0.160 -> 0.153, while
the `rag` and `long_context` lanes were byte-identical ([RAG
core](current/rag-core/context-ablation.md#context-ablation-evidence)). An ungrounded prompt leaves
a much flatter next-token distribution, so kernel-level nondeterminism flips tokens. The drift
stayed well inside the uplift interval and changed no verdict, but a contamination rate quoted to
one decimal place is currently over-stated precision. Measure it: repeat the closed-book lane N
times at a fixed seed, report the between-repeat spread of the lane mean and of the contamination
rate, and either quote the ablation's closed-book numbers with that spread or make the lane
reproducible (pinned sampler / seeded backend options) if the backend allows it.

- Serves: `answer-scoring` -- [Answer scoring](../design/spec.md#scoring-policy)
- Agent status: RUN NEEDED
- Dependencies: none. Reuse `compare-context-strategies` with a repeated `closed_book` lane and the
  existing paired-bootstrap reporting.
- User-visible outcome: the operator knows how much of a closed-book delta is measurement noise
  before reading it as parametric knowledge.
- Scope boundary: in scope -- repeat runs, the spread statistic, and whichever of the two remedies
  the measurement supports. Out of scope -- changing the objective metric and swapping backends.
- Data and artifact paths: `$DATA_DIR/context-ablation/<run>/`.
- Execution path: N repeats of the closed-book lane on the committed UA fixture on the CUDA host;
  CI covers the spread statistic over committed fixture rows.
- Acceptance gates: `make ci` green; the report states the between-repeat spread of the lane mean
  and the contamination rate, and the ablation docs quote closed-book numbers accordingly.
- Documentation target: the context-ablation evidence subsection of
  [RAG core](current/rag-core.md).

### Graph retrieval and ontology -- `graph-retrieval`

#### ontology-axiom-layer

The induced ontology is a type INVENTORY, not a set of constraints: `induce_ontology` emits entity
and relation types with counts, confidence, and examples under the `MAX_ENTITY_TYPES` /
`MAX_RELATION_TYPES` caps (`src/llb/prep/ontology/extraction/induce.py`, `models.py`), and nothing
in that artifact can be VIOLATED. The graph build accepts whatever the extractor emits -- `add_fact`
creates a lightweight `MISC` fact-only node for an unrecognized endpoint rather than refusing the
fact, and the approved schema states that rule outright ("no grounded fact is dropped", [graph
ontology schema](../design/graph-ontology-schema.md)). So a corpus ledger can assert that one patent
has two different durations, that one work has two exclusive owners, or that one entity is both
`PERSON` and `ORG`, and no stage notices -- after which the drafting pipeline turns those facts into
gold questions and the graph lane retrieves them as evidence. Build the ledger-side half of the
validation architecture: an AXIOM layer over the closed vocabulary and the induced relations --
functional and inverse-functional properties (at most one object per subject, and vice versa),
`domain`/`range` type constraints per relation, disjoint entity-type pairs, symmetry/asymmetry/
irreflexivity, and cardinality bounds -- plus a checker that reads an extraction ledger and reports
every violation with the axiom it breaks and BOTH offending facts' evidence spans.

Serialize the axioms as RDFS/OWL Turtle (`owl:FunctionalProperty`, `owl:InverseFunctionalProperty`,
`owl:disjointWith`, `rdfs:domain`, `rdfs:range`) so the constraint set is standard, diffable, and
reviewable by someone who does not read this codebase -- but keep the SHIPPED checker pure Python
over the existing typed models, so no runtime dependency is added, matching the optional-extras
discipline the rest of the repo keeps. A standard reasoner (`rdflib` + `owlrl` behind an
`[ontology]` extra, marked `heavy_env`) rides along in CI as a CROSS-CHECK only: it must agree with
the in-repo checker on the committed axiom fixture, and a disagreement is a bug in the in-repo
checker. That keeps OWL semantics as the reference without letting a reasoner into the answer path.

- Serves: `graph-retrieval` -- [Graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology)
- Agent status: RUN NEEDED
- Dependencies: none in code. Reuse the closed vocabulary and `normalize_entity_type` in
  `src/llb/prep/ontology/extraction/entity_types.py`, the typed `SROFact` / `Entity` / `OntologyCandidate`
  models in `src/llb/prep/ontology/models.py`, the caps and confidence blend in
  `src/llb/prep/ontology/constants.py`, the fact ingestion seam in `src/llb/graph/build.py`, and the
  violation-report renderer pattern from `src/llb/conflicts/report/render.py`.
- User-visible outcome: a corpus ledger whose logical inconsistencies are visible and named before
  they become gold questions or retrieved evidence, and a constraint set an operator can read,
  version, and hand to a domain expert.
- Scope boundary: in scope -- the axiom schema and its Turtle serialization, the pure checker, the
  reasoner cross-check, the violation report, the base-rate measurement over the committed corpora,
  and an opt-in `--refuse-violations` build flag. Out of scope -- inferring axioms from corpus
  frequency and shipping them unreviewed (a frequency-induced axiom only restates what the extractor
  emitted; acceptance is `ontology-axiom-signoff`), full OWL DL reasoning, changing the 13-type
  vocabulary or the relation caps, deleting any fact by default, and touching retrieval.
- Data and artifact paths: the candidate axiom set committed at `samples/ontology/axioms_uk_v1.ttl`
  plus its typed JSON form beside it; violation reports under `$DATA_DIR/ontology-validation/<run>/`.
- Execution path: `make validate-ontology-axioms EXTRACTION=<bundle>/extraction.jsonl
  AXIOMS=samples/ontology/axioms_uk_v1.ttl` over the drafted bundles of both quickstart corpora on
  the CUDA host (extraction ledgers already exist; no new inference unless a bundle is missing); CI
  covers each axiom class over a committed fixture carrying one planted violation per class plus the
  reasoner cross-check.
- Acceptance gates: `make ci` green; the pure checker and the `owlrl` reasoner return the identical
  violation set on the committed fixture; the report states the base rate per axiom class on both
  quickstart corpora, and a corpus with ZERO violations is recorded as a measured finding (that
  axiom class buys nothing on that corpus) rather than as a silent pass; the graph build is
  byte-identical unless `--refuse-violations` is passed; every reported violation cites both facts'
  exact spans, so a reviewer can adjudicate without re-reading the corpus.
- Documentation target: the ontology-assisted drafting section of
  [robustness and ontology](current/robustness-ontology-backends.md#ontology-assisted-drafting) and
  a constraints section in [graph ontology schema](../design/graph-ontology-schema.md).

#### ontology-validated-answer-gate

Compose the two halves into the shipped two-step gate -- Pydantic at the door, the ontology at the
ledger. Step one already ships: the completion either parses into the typed `AnswerEnvelope` or ends
in a typed status ([RAG core](current/rag-core/scoring.md#typed-rag-answer-envelope-typed-rag-answer-envelope)).
Step two is new: the envelope's asserted triples are checked
against the accepted axiom set AND against the corpus ledger the retrieved context came from, so an
answer that violates a functional property, a `domain`/`range` constraint, or a disjointness pair --
or that contradicts a ledger fact whose evidence is IN the retrieved chunks -- ends as
`ontology_violation` or takes one bounded repair instead of being scored as a fluent answer. This is
the step no existing signal covers: groundedness asks whether the answer's tokens appear in a chunk,
which a semantically impossible answer assembled from real chunk tokens passes cleanly.

The measurement has to be read honestly, because the obvious failure mode of any validator is
refusing correct work: report the gate's CATCH rate (violations caught per 100 answers) and its
FALSE-REJECTION rate (answers the gate rejects that the reference scores correct) as separate
numbers, report abstention rate and answered-item count beside the objective, and read the objective
delta on the items the UNGATED lane also answered -- otherwise a gate that improves the mean by
declining the hard items looks like a win.

- Serves: `graph-retrieval` -- [Graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology)
- Agent status: RUN NEEDED
- Dependencies: `ontology-axiom-layer` (the axioms to validate against); enabling an axiom at answer
  time also needs `ontology-axiom-signoff`, so the unsigned-axiom path must be refused rather than
  defaulted. The typed object, its boundary, and its bounded repair are shipped -- validate the
  envelope's `claims[].triple` and extend the same boundary rather than adding a second one.
  Reuse the paired verdict machinery in `src/llb/rag/embedding_bakeoff/uncertainty.py` and
  `separates()` in `src/llb/rag/fusion_evidence/stats.py`, the lane-comparison shape of
  `compare-answer-quality`, and the ledger lookup in `src/llb/graph/retrieval.py`.
- User-visible outcome: an operator learns whether semantic validation of RAG answers is worth its
  cost on their corpus -- how many logically impossible answers it stops, how many correct answers
  it wrongly refuses, and what the repair round trip costs in tokens and wall clock.
- Scope boundary: in scope -- the ledger-side check, the `ontology_violation` status, the bounded
  repair, the catch / false-rejection / cost columns, a per-axiom-class adopt-or-reject verdict, and
  a committed violation fixture. Out of scope -- rewriting the answer on the model's behalf beyond
  the one repair, judge-based validation, changing the headline objective, enabling any axiom class
  by default before its measured numbers support it, and inventing a ledger fact the corpus does not
  carry.
- Data and artifact paths: `$DATA_DIR/answer-validation/<run>/` for the lane comparison; the
  fixture at `samples/benchmarks/ontology_violations_uk.json` (the layout its sibling case files
  already use), carrying one planted violating answer per
  axiom class PLUS correct answers a naive checker would reject (a legitimately multi-valued
  relation, a paraphrased entity that normalizes to the same node, an entity typed `MISC` by
  fallback), so the false-rejection number is measured on adversarial cases rather than asserted.
- Execution path: `make compare-answer-validation VALIDATION_LANES=off,pydantic,pydantic+ontology
  MODEL=<model> GOLDSET=<accepted> AXIOMS=<signed-ttl>` over roster-family strata until the declared
  family-coverage and paired-precision targets are reached; CI drives all three lanes, both
  statuses, and the repair path over the fake completer and a fake ledger -- no GPU.
- Acceptance gates: `make ci` green; the `off` lane reproduces the recorded run bundles exactly; the
  fixture's planted violations are caught at 100% per axiom class and the adversarial correct
  answers produce a NAMED false-rejection rate, not a claim of zero; the heavy run reports the
  objective delta against `off` on the commonly-answered items with a paired interval and the
  standard adopt-or-retain verdict, plus abstention rate, answered count, repair rate, and added
  tokens/latency per answer; an axiom class ships enabled only when its catch rate clears its
  false-rejection rate under that verdict, and every class that does not is recorded as measured-and-
  not-adopted; an unsigned axiom file is refused with a named error rather than silently enabled.
- Documentation target: a two-step answer-validation section in
  [RAG core](current/rag-core/scoring.md#scoring) beside the groundedness metrics, and the adopt-or-reject
  record per axiom class in [product decisions](current/scope-boundaries.md).

#### graph-lane-score-ties (optional)

The graph lane's own recall is decided by tie order for two thirds of the questions it is scored on:
its link-relevance scores saturate into long exact-tie blocks, the rank-k cut falls inside one for
68 of 95 items (33 of 35 on the multi-hop slice), and that is what sets the fusion sweep's whole
measurement floor at `+/-0.021` recall@10 overall and `+/-0.043` on the focus slice
([GraphRAG](current/graphrag-backend/fusion-sweep-evidence.md#the-sweep-re-read-against-its-measurement-floor)).
The ranking is reproducible -- `_rank_dedup` breaks ties on `(doc_id, char_start, char_end)` -- but
a document id is not a relevance signal, so every graph-only row is quoted to three decimals it has
not earned. Find out whether the tie blocks are reducible: measure how much of each tie block is one
relevance value versus rounding, and if a finer signal exists (edge weight, hop distance, mention
count, community rank as a continuous term rather than a bucket) score the lane with it and
re-measure the floor.

- Serves: `graph-retrieval` -- [Graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology)
- Agent status: RUN NEEDED
- Dependencies: none. Reuse the scoring in `src/llb/graph/linking.py` / `src/llb/graph/retrieval.py`
  and the floor lane (`compare-graph-fusion --noise-floor`) to read the result.
- User-visible outcome: either graph-only rows whose recall reflects relevance instead of document
  order, or a recorded finding that the lane's evidence is inherently tied and its rows must be
  read at the floor's precision.
- Scope boundary: in scope -- the tie-block census, one finer relevance signal if the census
  supports one, and the before/after floor. Out of scope -- changing the graph schema, the fusion
  mechanics, and the RRF rank basis (fused rows already report a zero band).
- Data and artifact paths: `$DATA_DIR/graph-vector-fusion-multihop/<run>/`.
- Execution path: `make compare-graph-fusion NOISE_FLOOR=1 SPLIT=` before and after, on the CUDA
  host; CI covers the scoring change over the committed graph fixtures.
- Acceptance gates: `make ci` green; the report states the fragile count and floor per graph row
  before and after, and whether any recorded graph-row verdict changes.
- Documentation target: the retrieval-strategies section of
  [GraphRAG](current/graphrag-backend.md) and the floor table in [RAG core](current/rag-core.md).

### Host fit and serving -- `host-fit-serving`

#### gemma4-gguf-runner-gap (optional)

The host Ollama cannot serve a `gemma4` GGUF at all: 0.20 answers both the curated `gemma4:12b`
tag and the first-party QAT `q4_0` GGUF with `unknown model architecture: 'gemma4'`, so the
`gemma-4-12b-it-w4a16` entry has no Ollama path and only its vLLM checkpoint is measurable here
(see the full-roster throughput baseline in
[backend telemetry](current/backend-telemetry.md)).
Any per-model result that resolves through Ollama therefore silently skips one manifest entry.
Pin the minimum Ollama version that knows `gemma4` in the host setup path, make the resolver report
an architecture-unsupported source as a NAMED skip instead of a generic backend error, and re-measure
the entry on Ollama so the roster is served by one backend end to end.

- Serves: `host-fit-serving` -- [Host fit and serving](../design/spec.md#backend-and-hardware-boundary)
- Agent status: RUN NEEDED
- Dependencies: the resolver source-selection behavior in
  [platform matrix](current/platform-vector-matrix.md#multi-quant-vllm-resolution).
- User-visible outcome: an unsupported-architecture source fails with a source-specific message
  naming the runtime and the required version, and the roster has no backend-shaped hole.
- Scope boundary: in scope -- the version floor, the named skip, and the re-measured entry. Out of
  scope -- vendoring or building a llama.cpp runner.
- Execution path: raise the host Ollama, re-run the roster throughput protocol, refresh the
  baseline table.
- Acceptance gates: `make ci` green; a source whose architecture the runtime rejects produces the
  named error in a test with an injected fake; the refreshed table carries a measured Ollama row for
  every manifest entry or records the entry as backend-unsupported with its reason.
- Documentation target: the roster baseline in
  [backend telemetry](current/backend-telemetry.md) and the host setup notes in
  [host validation](current/host-validation.md).

### Optimization search -- `optimization-search`

#### ua-embedder-domain-finetune

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

- Serves: `optimization-search` -- [Optimization search](../design/spec.md#optimization-without-leakage)
- Agent status: RUN NEEDED
- Dependencies: none. Reuse the embedder conventions and bake-off in
  [RAG core](current/rag-core/embedders.md#embedder-conventions-and-bake-off), the lexical index for
  hard negatives, and the split-guard pattern in `src/llb/finetune/hparam_search/`.
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
  MRR on the held-out final split, where the adopt-or-keep-base verdict is the bake-off's own
  paired one -- the tuned row must clear zero against the base encoder, not merely outrank it
  ([RAG core](current/rag-core/paired-verdicts.md#paired-uncertainty-and-the-adopt-or-retain-verdict)).
- Documentation target: [RAG core](current/rag-core.md) embedder section and
  [extended workflows](current/extended-workflows.md) for the trainer lane.

### Run bundles and the board -- `run-bundle-board`

#### ua-model-roster-long-run (optional)

Confirm the refreshed-roster ranking at research scale: predeclare a minimum detectable objective
gain and ranking-stability criterion, derive the tuning-screen size from paired power, and run
multi-objective trials until the stability rule or a declared resource budget stops the search.
Score the full held-out final split and add the public Ukrainian screen tracks before making a
default-model adoption decision. Report bootstrap uncertainty and quality/latency Pareto tradeoffs
so a small-sample rank reversal cannot silently change the recommended model.

- Serves: `run-bundle-board` -- [Run bundles and the board](../design/spec.md#persistence-and-reproducibility)
- Agent status: RUN NEEDED
- Dependencies: use the roster/runtime behavior in
  [platform matrix](current/platform-vector-matrix.md#ukrainian-model-roster-refresh) and the
  bounded baseline in [evaluation rigor](current/rigor-board-judge/tuning-and-search.md#joint-model--config-search).
- User-visible outcome: a stable refreshed-roster recommendation with uncertainty, public-task
  coverage, and an explicit adopt-or-retain verdict.
- Scope boundary: in scope -- larger private joint search, public-screen lanes, uncertainty, and
  the adoption verdict. Out of scope -- model fine-tuning and hosted/API-only candidates.
- Data and artifact paths: `$DATA_DIR/joint-search/<run>/`, `$DATA_DIR/screen/`, and the matching
  current-doc evidence section.
- Execution path: run `make joint-search` on a CUDA host with the refreshed candidates and full
  final split, then run the public screen for both finalists.
- Acceptance gates: `make ci` green; the search artifact records the effect, power, stability, and
  stopping assumptions plus the derived screen size and consumed trial budget; no final-split
  leakage into tuning; confidence-aware ranking; explicit quality-versus-latency recommendation.
- Documentation target: [evaluation rigor](current/rigor-board-judge.md) host evidence.

### Agentic and context-policy workloads -- `agentic-workloads`

#### agent-operating-profile-recommendation

Every ingredient of an agent configuration is measured somewhere in this repo and nowhere together:
`llb recommend` renders host-adaptive model picks plus separate miss-analysis, self-improvement,
fine-tune-campaign, and context-policy sections (`src/llb/board/recommend/sections.py`), the
context-order recommendation comes out of the position probe, the retrieval knobs out of the
comparison lanes, the prompt-system id out of `prompt-system-compare`, and the adapter out of the
registry. An operator wanting to stand up an agent must read five sections and hand-assemble, and
nothing checks that the pieces were measured on the same corpus, store, or model. Compose them:
`llb recommend --agent-profile` emits ONE `agent_profile.json` plus a markdown rationale naming
model and backend, prompt-system id, adapter (or none), context policy, context order, `top_k` /
reranker / context budget, and the loop policy. Each field carries its value, the artifact path the
value came from, that lane's own verdict and uncertainty, and its freshness -- and a field whose
lane never ran is emitted as `unmeasured`, never as a default dressed up as a recommendation, which
is the whole failure mode a composed profile invites.

- Serves: `agentic-workloads` -- [Agentic and context-policy workloads](../design/spec.md#agentic-and-context-policy-workloads)
- Agent status: RUN NEEDED
- Dependencies: the
  [agent loop-policy
  recommendation](current/extended-workflows/loop-policy-recommendation.md#agent-loop-policy-recommendation)
  supplies the loop-policy field; the context-policy field comes from the `agentic-context` bundles
  ([extended
  workflows](current/extended-workflows/agent-context-policies.md#agent-context-management-policies)),
  and for memory-dependent work its guard-dependent routing rule comes from the cap-fitting boundary
  surface ([extended
  workflows](current/extended-workflows/crossover-geometry.md#cap-fitting-boundary-surface)); the
  rest are current behavior. Reuse `src/llb/board/recommend/` (sections, build, render), the adapter
  registry's `staleness()` and its retrieval-fingerprint axis ([extended
  workflows](current/extended-workflows/adapter-registry.md#staleness)), and the shared borderline
  vocabulary in `src/llb/rag/fusion_evidence/stability.py` so a field resting on a knife-edge row is
  marked the same way every lane marks it.
- User-visible outcome: one artifact an operator (or a runtime) can act on, where every recommended
  value is traceable to the run that measured it and every gap is visible as a gap.
- Scope boundary: in scope -- the composition, the per-field evidence/verdict/freshness record, the
  `unmeasured` state, the consistency guard (fields measured against a different corpus, store
  fingerprint, or model are refused rather than mixed), the JSON schema, and the markdown rationale.
  Out of scope -- running any lane on the operator's behalf, inventing a value for an unmeasured
  field, ranking policy changes, and shipping a runtime that consumes the profile.
- Data and artifact paths: `$DATA_DIR/agent-profile/<run>/{agent_profile.json,profile.md}`, composed
  from the existing per-lane roots; no new evidence root.
- Execution path: `make recommend-agent-profile` after the component lanes have run on the CUDA
  host; CI covers composition, the `unmeasured` state, the consistency guard, and the staleness
  demotion over fixture bundles -- no GPU.
- Acceptance gates: `make ci` green; a profile built with no bundles at all is entirely `unmeasured`
  and still a valid artifact; every populated field's evidence path resolves and its verdict matches
  the lane artifact it cites; a stale adapter or a store whose retrieval fingerprint changed demotes
  every field that depends on it, with the changed knob named; the recommended values replay as
  `run-eval` / `bench-agentic` flags that reproduce the recommended configuration.
- Documentation target: a recommendation-composition section in
  [extended workflows](current/extended-workflows.md) and the recommendation entry in
  [overview](current/overview.md).

#### agent-context-policy-entry-aware-summary-fold-adoption (optional)

Promote the entry-aware summary-input prototype into an explicit context-policy choice and decide
whether it should replace the shipped whole-transcript head-and-tail trim. Run it across the typed
memory, aggregate-search, repeated-fold, and crossover workloads on two host-fit families; compare
completion, total model-input cost, summary prompt bytes, and fold count against `head_tail`; and
route the new strategy through policy-change audit and published-value provenance before any default
change. Require no head/tail or aggregate regression and preserve the middle-critical recovery.

- Serves: `agentic-workloads` -- [Agentic and context-policy workloads](../design/spec.md#agentic-and-context-policy-workloads)
- Agent status: RUN NEEDED
- Dependencies: reuse the gated recovery and fixed-byte comparison in
  [summary-input bounds and elision](current/extended-workflows/summary-input-elision.md#middle-critical-transfer-and-entry-aware-prototype),
  plus the [policy-constant audit](current/extended-workflows/policy-constant-audit.md).
- User-visible outcome: an operator gets a supported summary-fold strategy for sessions where
  critical evidence can occupy any entry, rather than an evidence-only prototype.
- Scope boundary: in scope -- public policy configuration, cross-workload regression, audit and
  provenance integration, and a default decision. Out of scope -- increasing the resolved window
  or changing compaction trigger/hysteresis.
- Documentation target:
  [summary-input bounds and elision](current/extended-workflows/summary-input-elision.md#middle-critical-transfer-and-entry-aware-prototype).

#### agent-context-policy-hysteresis-second-fold (optional)

Every cap-fitting cell measured so far folds EXACTLY once per episode, which is why the guard drops
out of the cost once the trigger is fixed: after the first summary, trigger hysteresis raises the
next trigger to the full guard, and no tested transcript grows back that far
([extended workflows](current/extended-workflows/crossover-geometry.md#the-routing-rule-lives-on-the-trigger-axis)).
The trigger-only rule is therefore established only in the one-fold regime, and the regime where
compact is most interesting -- long agent sessions that fold repeatedly -- is unmeasured.

Do NOT look for that geometry inside the cap-fitting band: a cap-fitting cell cannot fold twice, for
a structural reason rather than a lack of searching
([extended workflows](current/extended-workflows/imperfect-play-margin.md#why-a-cap-fitting-cell-folds-exactly-once)),
so "shrink the guard toward the cap peak" cannot work and pushing depth alone does not either. The
repeatedly folding cells live BELOW the cap peak, where the `observation_cap` arm overflows and
there is no compact-minus-cap delta to compare -- which is the real obstacle this task has to solve
first. Restate the equal-trigger claim on a comparison that survives without a cap arm (compact
against compact at equal triggers, on total model-input tokens), run one equal-trigger family over
the committed two-fold geometry, and state whether the deltas separate. The later folds also carry
the running summary into the summarize input, and the shipped `window` bound sizes that input from
the budget rather than the trigger, so record the per-fold summarize input beside the deltas -- a
growing prior summary is the other way the guard could re-enter.

- Serves: `agentic-workloads` -- [Agentic and context-policy workloads](../design/spec.md#agentic-and-context-policy-workloads)
- Agent status: RUN NEEDED
- Dependencies: the committed repeatedly folding geometry and its per-fold summarize inputs are
  current behavior (`samples/benchmarks/agentic_compact_two_fold_geometry_design.json`); reuse the
  collapse design's family contract and equivalence band, but not its cap-fitting cell gate, which
  by construction refuses every cell in this regime.
- User-visible outcome: either the trigger-only routing rule extended to repeated compaction, or an
  explicit "one fold only" boundary on the rule an operator would otherwise over-apply.
- Scope boundary: in scope -- the cap-arm-free restatement of the equal-trigger comparison, one
  family over the committed two-fold geometry, and the validity statement. Out of scope -- new task
  shapes, changing shipped compaction hysteresis, and relaxing the cap-fitting gate on the studies
  that legitimately use it.
- Documentation target:
  [extended workflows](current/extended-workflows/crossover-geometry.md#the-routing-rule-lives-on-the-trigger-axis).

#### agent-context-policy-repeated-fold-completion-replication (optional)

The current completion reading covers two deterministic memory cases on one qualified model through
three measured folds
([extended workflows](current/extended-workflows/imperfect-play-margin.md#completion-through-repeated-folds)).
Strengthen the routing claim with a predeclared larger case set and a second model family: require
both families to pass the one-fold eligibility gate, preserve identical cases and seed across fold
cells and marker arms within each family, and report paired completion uncertainty at each measured
fold count. This separates a robust fold-count rule from a ceiling result on two easy codes.

- Serves: `agentic-workloads` -- [Agentic and context-policy workloads](../design/spec.md#agentic-and-context-policy-workloads)
- Agent status: RUN NEEDED
- Dependencies: reuse the compact-only runner, eligibility gate, measured-fold grouping, and marker
  ablation documented in the linked current page; pick the second family by the local-model host-fit
  rules rather than weakening the task for a smaller model.
- User-visible outcome: an operator learns whether the three-fold completion result transfers beyond
  one model and two cases before treating it as a general session-routing bound.
- Scope boundary: in scope -- a larger predeclared task set, one additional qualified family, paired
  uncertainty, and a cross-family reading. Out of scope -- folds deeper than three, a new compaction
  algorithm, and changing the shipped marker-preservation default.
- Data and artifact paths: the existing `$DATA_DIR/agentic-compact-vs-cap/<run>/` layout, with family
  and task-set digests in every aggregate.
- Execution path: extend `make bench-agentic-context-compact-repeated-fold` with a replication design
  on the CUDA host; CI covers the multi-family aggregation and refusal paths with fakes.
- Acceptance gates: `make ci` green; each family passes its control before repeated cells run; every
  fold group reaches the predeclared paired-evidence floor; the report either extends the three-fold
  rule across families or names the first family/fold where it fails.
- Documentation target:
  [extended workflows](current/extended-workflows/imperfect-play-margin.md#completion-through-repeated-folds).

### Corpus conflict and governance -- `corpus-conflict-audit`

#### corpus-ingestion-reports-the-governance-coverage-the-audit-blames-it-for (optional)

The audit tells an operator a zero policy delta is "fixable at INGESTION (record `effective_date`
or `version`)", but ingestion itself never mentions it: `make ingest-corpus` writes a manifest whose
governance fields are empty and reports success, so the gap is only ever discovered after a store
build and an audit run
([decision groups](current/data-prep/conflict-decision-groups.md#the-precondition-behind-a-zero-delta)).
Report the same document-side count at ingestion time -- documents carrying `effective_date` /
`version`, per field -- in the ingest summary and in the corpus manifest, phrased as what it costs
(no supersession can ever be derived on this corpus) rather than as a warning to scroll past. It
must stay a REPORT: a corpus without governance dates is a legitimate corpus and ingestion must not
fail, refuse, or invent a date for it.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: CLEAR
- Dependencies: none. `document_coverage` in `src/llb/conflicts/governance/coverage.py` is the
  count and takes governance dicts rather than corpus objects, so the ingest path can reuse it as
  is, and `document_pair_orderability` beside it is the same input again -- report both, since a
  corpus dated end to end with one shared edition is orderable by neither;
  `manifest_governance_by_doc` / `item_governance` in `src/llb/prep/corpus/governance.py` are
  where ingestion already holds the same fields.
- User-visible outcome: an operator learns their corpus cannot carry a dated supersession while
  they are still ingesting it, not two commands and one GPU run later.
- Scope boundary: in scope -- the count in the ingest summary and manifest, and the one-line
  consequence. Out of scope -- failing or refusing an undated ingest, inferring dates from document
  text or file mtime, and any change to the audit-side counts.
- Data and artifact paths: the existing corpus manifest under the ingested corpus root.
- Execution path: ingest-side counting with fixture tests; no GPU.
- Acceptance gates: `make ci` green; an undated corpus ingests successfully and reports zero
  coverage with the consequence named; a dated corpus reports its per-field counts; the manifest
  round-trips the counts and the audit's own coverage agrees with them on the same corpus.
- Documentation target: the corpus-ingestion section of
  [data prep](current/data-prep.md) plus a pointer from
  [decision groups](current/data-prep/conflict-decision-groups.md#the-precondition-behind-a-zero-delta).

#### conflict-tree-reuse-gate-is-not-the-function-that-claims-to-be-it (optional)

`tree_is_reusable` (`src/llb/conflicts/semantic_tree/refresh.py`) documents itself as the rule that
stops a tree built under one encoder from being queried under another, and nothing in `src/` calls
it: the only caller is `tests/llb/conflicts/test_tree.py`. The gate that actually runs is
`prepare_projected_index`'s `source_fingerprint`, which hashes the encoder, the dimension, the
centering flag, the corpus fingerprint, the store manifest, and the whole chunk table into one
value, so it is strictly stronger AND covers only the PROJECTED path -- the full-space path builds a
fresh tree every run and reuses nothing. Two live functions describing one gate is how a later
change gets made in the wrong place. Decide which it is: delete the unused one and say the
fingerprint is the gate, or make it the cheap pre-check the full-space path is missing and give that
path a persisted tree to reuse.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: CLEAR
- Dependencies: none. `tree_is_reusable` and `TREE_VERSION` (`semantic_tree/node.py`) are one side,
  `_source_fingerprint` / `prepare_projected_index`
  (`src/llb/conflicts/semantic_tree/projected_index.py`) the other, and `_active_tree`
  (`semantic_run.py`) is the full-space path that persists nothing.
- User-visible outcome: one documented rule for when a persisted tree may be reused, in the place
  the reuse actually happens.
- Scope boundary: in scope -- the verdict, the deletion or the wiring, and the test that follows it.
  Out of scope -- changing the tree format, changing what the fingerprint covers, and adding
  persistence to the full-space path unless the verdict picks that.
- Execution path: source change with fixture tests; no GPU.
- Acceptance gates: `make ci` green; every reuse decision in `src/` goes through exactly one
  function; a tree built under a different encoder is still refused on both paths.
- Documentation target: [conflict
  detection](current/data-prep/conflict-detection.md).

#### conflict-group-ids-that-survive-a-re-run (optional)

A group id is `G<n>` from `findings.jsonl` file order, and that order is the score-ranked one -- so
a group id is stable inside one run and NOT across two runs of the same corpus, because the claim
adjudicator's scores are not bit-reproducible. Measured: two claim-tier runs of the committed
fixture returned the same 17 rows, the same relations, and the same document pairs, yet the group
holding the dated supersession was `G4` in one and `G3` in the other
([decision groups](current/data-prep/conflict-decision-groups.md#what-the-policy-choice-costs-measured)).
Nothing joins across runs today, so nothing is broken -- but an operator comparing this week's
audit to last week's, or a dashboard keying a decision on `G3`, silently compares two different
decisions. Give a group an identity derived from its ROWS (a digest over its member `finding_id`s,
which are content hashes) carried beside the positional id, so a cross-run comparison joins on
something the ordering cannot move.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: CLEAR
- Dependencies: none. `finding_id` (`src/llb/conflicts/tiers/hashing.py`) is already the content-addressed
  row identity and `group_summaries` (`grouping/artifact.py`) is where a group's member list is built;
  the positional `group_id` must stay exactly as it is, since `plan.json` and the review ledger
  join on it within a run.
- User-visible outcome: an operator can tell whether the decision they triaged last week is the
  decision the audit is showing them today.
- Scope boundary: in scope -- the row-derived group key, its appearance in `groups.json` and
  `plan.json`, and a test that re-ordering the same rows preserves it. Out of scope -- replacing
  the positional id, changing the grouping rule, and building a cross-run diff command.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts only.
- Execution path: sidecar and plan change with fixture tests; no GPU.
- Acceptance gates: `make ci` green; a fixture whose rows are re-ordered so the positional ids move
  keeps every row-derived key; two audits of the same corpus whose rows differ only in order agree
  on the keys group for group.
- Documentation target:
  [decision groups](current/data-prep/conflict-decision-groups.md#the-groupsjson-sidecar).

#### conflict-groups-sidecar-carries-the-ranking-inputs (optional)

The report ranks decision groups by stake, but `groups.json` does not carry what the ranking is
computed from: its summaries hold `rows`, `relations`, and `top_score`, so a consumer wanting the
same order must re-derive the to-decide count from the relation map and re-implement `stake_key`
([decision groups](current/data-prep/conflict-decision-groups.md#the-groupsjson-sidecar)). That is
the same drift the shared `finding_id` was introduced to prevent, one level up. Add `decide_rows`
and the rendered `rank` to each group summary, and the same `rank` to the plan's `decisions`, so a
dashboard, a runtime, or a second report reads the audit's own ordering rather than an
approximation of it.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: CLEAR
- Dependencies: none. `stake_key` in `src/llb/conflicts/report/findings.py` is the ranking;
  `group_summaries` in `grouping/artifact.py` builds the sidecar from `findings.jsonl` rows, so the
  rank must be computable from rows alone to keep the sidecar derivable without the report.
  `decide_count` (`src/llb/conflicts/constants.py`) is the count and the plan's `decisions` already
  carry it, so the sidecar must reuse it rather than add a third implementation.
- User-visible outcome: every consumer of the audit shows the operator the same first decision.
- Scope boundary: in scope -- the two fields, the shared ranking helper, and its test against the
  rendered table. Out of scope -- changing the ranking itself, group identity, and `findings.jsonl`.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts only.
- Execution path: sidecar and rendering change with fixture tests; no GPU.
- Acceptance gates: `make ci` green; the sidecar's rank order equals the report's decision table
  order on a fixture whose stake ranking differs from its file order; group ids stay in file order.
- Documentation target: [conflict
  detection](current/data-prep/conflict-decision-groups.md#the-groupsjson-sidecar).

#### conflict-decision-chain-length-in-the-stake-ranking (optional)

The decision table ranks a group on `to decide` then rows, and both treat a group as flat -- but the
audit now measures how many distinct pieces of shared evidence each group's chain runs through
(`quoted_group_split` in
[decision groups](current/data-prep/conflict-decision-groups.md#how-many-decisions-the-row-count-is)),
and on the measured bundles the spread inside one row count is large: a 6-row fan resting on ONE
chunk is one decision, while goods G1's 51 rows run through 23. Two groups with the same row count
therefore cost an operator very different amounts, and the ranking cannot see the difference. Fold
the chain length into `stake_key` (or add it as a decision-table column and state why it is not
ranked on), and measure how often the order actually changes on the committed bundles -- a signal
that never reorders anything is not worth a column.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: CLEAR
- Dependencies: none. `stake_key` in `src/llb/conflicts/report/findings.py` is the ranking and
  `shared_unit_indices` in `src/llb/conflicts/grouping/granularity.py` is the chain length; the
  ranking must stay computable from `findings.jsonl` rows alone so `groups.json` can carry it.
- User-visible outcome: the first decision the report offers is the one that actually costs the most,
  not the one with the most rows.
- Scope boundary: in scope -- the chain-length term, its effect measured on the committed bundles,
  and the keep-or-drop verdict. Out of scope -- changing group identity, changing either grouping
  rule, and ranking on a projected count.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts only.
- Execution path: rendering change with fixture tests; recompute the order over committed bundles,
  no GPU.
- Acceptance gates: `make ci` green; group ids never move; the report states on how many of the
  measured bundles the order changed, including the bundles where it did not.
- Documentation target:
  [decision groups](current/data-prep/conflict-decision-groups.md#how-many-decisions-the-row-count-is).

#### conflict-stage-attribution-counts-the-pairs-each-knob-buys (optional)

The attribution names ONE stage and one pair, and a run that loses pairs at three stages says
nothing about how much of the corpus each knob would recover: the purpose-built chunking-gap run
loses two of its three document pairs to the missing document and one to candidate selection, and
the reading names only the first
([decision groups](current/data-prep/conflict-decision-groups.md#which-stage-lost-the-orderable-pair)).
An operator sizing a fix wants the split -- turning this knob reaches N of the M orderable pairs
the run lost, and the rest are elsewhere. Count the lost orderable pairs per stage (the census the
one-pair scan already walks the classes for) and print it beside the named pair, keeping the single
named pair as the headline so the reading does not turn back into a list of knobs. Cost is the
constraint: a full census is the quadratic sweep the one-pair rule exists to avoid, so bound it by
the per-document classes (a document lost at a stage bounds its own pairs) or cap the count and say
it is a floor.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: CLEAR
- Dependencies: none. `document_stage` and `REPORT_STAGE_ORDER` in
  `src/llb/conflicts/governance/stage_rule.py` are the classes and their order;
  `orderable_document_pairs` in `governance/coverage.py` is the denominator and is already counted
  without enumerating pairs.
- User-visible outcome: an operator learns whether the named knob recovers most of what the run
  lost or one pair of it.
- Scope boundary: in scope -- the per-stage count, its cost bound, and one rendered line. Out of
  scope -- naming a second pair, adding a stage, and re-running detection.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts only.
- Execution path: audit-side reading with fixture tests; recompute over the bundles on disk, no GPU.
- Acceptance gates: `make ci` green; the per-stage counts sum to the lost orderable pairs on a
  fixture that loses pairs at three stages; the cost stays inside the stated bound on a corpus
  large enough for the difference to show.
- Documentation target:
  [decision groups](current/data-prep/conflict-decision-groups.md#which-stage-lost-the-orderable-pair).

#### conflict-budget-replay-counts-the-orderable-pairs-it-costs (optional)

The budget re-read reports how many DOCUMENT pairs a smaller candidate budget would have returned
(5 of 8 on the fixture at budget 2), but the coverage reading the whole page is about counts
ORDERABLE pairs -- so an operator cannot tell whether a cheaper budget costs supersession evidence
or only pairs the corpus could never order anyway
([bundle record](current/data-prep/conflict-bundle-record.md#what-a-smaller-candidate-budget-would-have-returned)).
The named-pair reading does not close the gap either: on every measured bundle the named pair was
identical at every budget, because the corpus-first lost pair is lost at all of them. Report the
orderable count beside the total -- the record already carries each document's ordering fields, so
`compare_editions` over the recorded documents is the whole computation -- and state what a budget
costs in the units the reading is quoted in.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: CLEAR
- Dependencies: none. `returned_pairs_at_budget` in `src/llb/conflicts/bundle/stage_replay.py` is
  the set and `documents_of` beside it carries the ordering fields; `compare_editions`
  (`governance.py`) is the orderability test the coverage already uses.
- User-visible outcome: an operator lowering the candidate budget learns whether it costs evidence
  or only noise.
- Scope boundary: in scope -- the orderable count at a budget, beside the run's own, and its line in
  the report. Out of scope -- re-adjudicating rows, a per-stage census, and changing the named pair.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts only.
- Execution path: replay-side counting with fixture tests; no GPU.
- Acceptance gates: `make ci` green; at the run's own budget the orderable count equals the
  bundle's recorded `orderable_pairs`; a fixture whose budget drops only unorderable pairs reports a
  cost of zero orderable pairs while the total falls.
- Documentation target:
  [bundle record](current/data-prep/conflict-bundle-record.md#what-a-smaller-candidate-budget-would-have-returned).

#### conflict-bundle-records-the-store-it-read-but-not-where-it-was (optional)

A bundle can now be PLACED against a store -- `recompute-conflict-stage --store <dir>` says whether
the store on disk is the one that run read
([bundle record](current/data-prep/conflict-bundle-record.md#the-store-the-bundle-does-not-copy)) --
but the operator has to supply the path, and a sweep can only be pointed at ONE store while its
bundles were taken over many. The bundle records the store's identity and not its LOCATION, so
"which of my stores is this bundle about" is unanswerable even when every candidate store is on the
host. Record the store directory the run read (relative to `$DATA_DIR` where it sits under it, since
an absolute path is host-specific and the repo forbids hardcoding one), and let the re-read resolve
it -- falling back to the explicit flag, and saying plainly when the recorded location is gone
rather than reporting a mismatch it did not test.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: CLEAR
- Dependencies: none. `StoreView.index_dir` is the resolved path the audit already holds,
  `store_identity.py` is where the identity is recorded and compared, and `resolve_data_dir` /
  `resolve_store_dir` (`src/llb/core/`) are how a recorded location must be re-resolved.
- User-visible outcome: an archive sweep tells an operator which of their stores each bundle's
  readings are about, without being told the answer first.
- Scope boundary: in scope -- the recorded location, its resolution on re-read, the explicit-flag
  precedence, and the "recorded store is gone" reading. Out of scope -- recording anything else
  about the store, searching the host for a matching store, and changing the identity digest.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/summary.json`.
- Execution path: artifact change with fixture tests; no GPU.
- Acceptance gates: `make ci` green; a bundle whose recorded store still exists is placed against it
  with no flag; a bundle whose store has been deleted says so rather than reporting a mismatch; an
  explicit `--store` still wins; no absolute host path reaches the artifact.
- Documentation target:
  [bundle record](current/data-prep/conflict-bundle-record.md#the-store-the-bundle-does-not-copy).

#### conflict-summary-group-granularity-repeats-its-own-prose-in-every-bundle (optional)

With the store manifest gone from the `tree` block, the second-largest key in a 250-document
`summary.json` is `group_granularity` at **2,537 of 13,409 bytes** (19%), and about a third of it is
not data: `unit` (181 bytes) plus one `description` per grouping rule are BUILD constants -- the same
sentences in every bundle every run ever wrote, explaining a rule whose prose already lives in
[decision groups](current/data-prep/conflict-decision-groups.md#how-many-decisions-the-row-count-is).
The rest is `rules.*.sizes` and `quoted_group_split`, which are linear in GROUPS and therefore in
ROWS -- the one growth term in `summary.json` that the record's own size bound (linear in DOCUMENTS
or bounded by a run parameter) does not cover, so a claim-tier run on a conflict-dense corpus grows
it without limit. Price both halves against the bundles on disk: drop or version the prose (a rule
name plus a schema version is what a consumer joins on), and decide whether `sizes` should stay a
list or become the `size_counts` histogram it is already recorded beside.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: CLEAR
- Dependencies: none. `finding_granularity` and `quoted_group_split`
  (`src/llb/conflicts/grouping/granularity.py`) build the block, `bundle/fold.py` is the gate every other
  fold in the bundle is held to, and `schema_version` inside the block is the migration seam.
- User-visible outcome: a bundle's size tracks its corpus and its run parameters, instead of the
  number of rows the adjudicator happened to return.
- Scope boundary: in scope -- the measured saving per half, a keep-or-change verdict, and the
  schema bump if the form changes. Out of scope -- changing either grouping rule, the decision
  range, and what the report renders from the block.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/summary.json`.
- Execution path: artifact change with fixture tests; no GPU.
- Acceptance gates: `make ci` green; every bundle reading replays identically through the old and
  new forms; a corpus whose row count grows tenfold grows the block less than tenfold, or the
  reading states why a list is kept.
- Documentation target:
  [bundle record](current/data-prep/conflict-bundle-record.md#the-size-the-record-actually-costs).

#### conflict-claim-tier-cross-encoder-prefilter (optional)

Spend the cross-encoder's ordering on adjudication COST instead of on a rate. Scoring a candidate
pair with a cross-encoder orders the claim tier's own verdicts where cosine does not -- monotone
score bins whose top bin is entirely conflicts on the planted fixture and the high-recall corpus
([closure](current/data-prep/conflict-null-closure.md)) -- yet the claim tier still adjudicates the
ranked list in cosine order, so it pays a model call for rows a 568M cross-encoder can already place
at the bottom. Re-rank the candidate rows with the pinned cross-encoder before adjudication and
measure what that buys: adjudicated rows needed to reach the same set of found conflicts, and the
conflicts lost if any. The scorer must stay injectable, and a corpus where the ordering is flat
(goods, four conflicts in fifty rows) must degrade to today's behavior rather than drop rows.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: reuse `controls/cross_encoder.py` (pair scoring batched by left passage,
  calibration binning) and the `RerankScorer` seam in `src/llb/rag/rerank.py`; the claim tier and its
  artifacts are current behavior in [conflict
  detection](current/data-prep/conflict-detection.md#effort-tiers).
- User-visible outcome: `audit-corpus-conflicts --effort claim` reaches the same conflicts for fewer
  adjudication calls on corpora where the ordering is informative, with the saving recorded in the
  run artifacts.
- Scope boundary: in scope -- an optional pre-filter stage between candidate generation and
  adjudication, its per-corpus flat-ordering fallback, and the cost/recall evidence. Out of scope --
  quoting any cross-encoder score as a probability, rate, or confidence; changing the relation
  vocabulary; and dropping a row the claim tier would have called a conflict without recording it.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts only.
- Execution path: add the pre-filter behind an injected scorer with a deterministic fixture test,
  then one CUDA-host claim run per quickstart corpus paired against the same run without it.
- Acceptance gates: `make ci` green with the injected scorer; no conflict found by the unfiltered
  run is missing from the filtered run on either quickstart corpus; and the recorded saving is stated
  per corpus, including the corpus where it is zero.
- Documentation target: [conflict detection](current/data-prep/conflict-detection.md).

#### conflict-adjudicator-probe-difficulty (optional)

The frozen calibration probe is passed **24/24** by MamayLM-Gemma-3-12B, which means the gate is
currently proving only that an adjudicator is not badly broken -- a probe nobody fails cannot
separate a good adjudicator from an adequate one, and the audit will happily quote a precision
figure from either
([conflict detection](current/data-prep/conflict-detection.md#the-frozen-calibration-probe)). Add a
harder frozen tier: pairs whose actionable/complementary split is genuinely arguable (a restated
fact under a different heading, two numeric claims about different quantities in the same
sentence shape, a partial supersession where only one clause changed), score two host-fit model
families against both tiers, and either raise `MIN_ADJUDICATOR_ACCURACY_LCB` on the evidence or
record that the gate's job is a floor rather than a ranking.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. Reuse `src/llb/conflicts/claim/calibration.py`, its heading-addressed probe
  format, and the planted fixture; a new hard tier may need new fixture sections, which must stay
  offset-exact and pass the existing corpus-unchanged assertion.
- User-visible outcome: the calibration gate distinguishes adjudicators an operator would actually
  choose between, instead of only rejecting a broken one.
- Scope boundary: in scope -- the harder probe tier, its frozen labels and rationale, a two-family
  comparison, and a gate-threshold decision. Out of scope -- changing the adjudication prompt, the
  relation vocabulary, and scoring agreement on anything but the actionable binary before the
  comparison supports it.
- Data and artifact paths: `samples/corpora/conflicts_uk_v1/adjudicator_probe.json` and the
  existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts.
- Execution path: one calibration-only run per model family on the CUDA host; CI covers the new
  tier's passage resolution and label balance.
- Acceptance gates: `make ci` green; both families score both tiers; the report states whether the
  hard tier separates them and either raises the gate or records why it stays a floor.
- Documentation target: [conflict detection](current/data-prep/conflict-detection.md#the-frozen-calibration-probe).

#### conflict-precision-bound-at-document-clustering (optional)

The clustered bound treats a CHUNK as the independent unit, and on a concentrated corpus that is
still too generous. Measured: every one of the goods corpus's 8 actionable rows at budget 100 points
at the same right document, so at the document level the corpus supplies one observation, not eight
-- yet the chunk-level bound resamples 42 left and 51 right chunks
([conflict detection](current/data-prep/conflict-detection.md#measured-both-quickstart-corpora)).
The bound refused a floor there anyway, which is why this is a sharpening rather than a correction,
but nothing establishes which unit the audit should quote when the two disagree. Compute the bound
at both clusterings on the same rows, report them side by side on both quickstart corpora, and state
the rule: quote the document-level bound always, quote the chunk-level bound when the corpus spreads
its conflicts, or quote both.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. `two_way_proportion_bound` already takes arbitrary cluster keys, so the
  document-level bound is the same estimator over `doc_id` keys; the rows and their verdicts come
  from the existing budget-100 runs and need no re-adjudication.
- User-visible outcome: the precision floor an operator reads is clustered on the unit their corpus
  actually repeats, instead of on whichever unit the chunker happened to produce.
- Scope boundary: in scope -- the second clustering, the side-by-side report, and a stated rule for
  which bound the audit quotes. Out of scope -- a third clustering level, changing the estimator,
  and changing the calibration gate.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts only.
- Execution path: recompute both bounds over the committed budget-100 per-row ledgers on the CUDA
  host (no model calls needed); CI covers both clusterings over fixture rows.
- Acceptance gates: `make ci` green; both corpora report both bounds at every measured budget; the
  report names which bound the audit quotes and why, and the chunk-level bound never reads lower
  than the document-level one on the same rows.
- Documentation target: [conflict detection](current/data-prep/conflict-detection.md#measured-claim-tier-precision).

#### conflict-candidate-record-cap-on-a-natively-dense-corpus (optional)

The candidate record's cap is priced, but on a density no operator would run at: no corpus on this
host saturates the list at any real threshold, so the depth/cost curve was measured by dropping the
cosine to 0.25 on the 250-document quickstart corpus, which forms 3,127 candidate rows against the
38 the same corpus returns at the 0.6 it was actually audited at
([bundle record](current/data-prep/conflict-bundle-record.md#how-deep-the-prefix-reaches-and-what-the-depth-costs)).
Two things that decide the cap are therefore substitutes rather than measurements. The RANKS-PER-PAIR
collapse ratio (1.11 at cap 200, 1.22 over the whole list) is what turns a cap in document pairs into
an answerable depth in ranks, and on a corpus whose density comes from genuine near-duplicates rather
than from a loosened threshold it could be far higher -- the same cap would then buy much less depth.
And the depth a re-read is asked at rests on a structural bound (the question is downward, so the
run's own budget is the ceiling) plus two operating budgets, never on an observed re-read. Re-price
it on a corpus that saturates the list at an operating cosine, and record the budget each
`recompute-conflict-stage` run is actually asked at so the depth side stops being an argument.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. `--max-candidate-record-pairs` and the recorded `cap` are current behavior;
  `CandidateRecord.of` in `src/llb/conflicts/bundle/candidate_record.py` is where the collapse
  happens and `covered_to_rank` is the ratio's numerator. The corpus is the missing input: it needs
  many near-duplicate documents, not a lowered threshold.
- User-visible outcome: an operator on a duplicate-heavy corpus learns whether the shipped cap still
  answers the budgets they ask, instead of inheriting a number priced on a synthetic density.
- Scope boundary: in scope -- one natively dense corpus, its ranks-per-pair ratio at several caps,
  the re-read depths observed, and a keep-or-change verdict on the constant. Out of scope --
  recording chunk-level pairs, removing the cap, changing the refusal past it, and re-deciding the
  cap before a corpus that saturates at an operating threshold exists.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` and
  `$DATA_DIR/corpus-conflict-stage/<run>/` artifacts only.
- Execution path: one semantic-tier run per cap on the CUDA host; CI covers the ratio and the
  refusal over fixtures.
- Acceptance gates: `make ci` green; the reading states the ranks-per-pair ratio at each measured cap
  beside the quickstart corpus's, and either keeps `DEFAULT_CANDIDATE_RECORD_PAIRS` with its reason
  or moves it; a stage re-read records the budget it was asked at.
- Documentation target:
  [bundle record](current/data-prep/conflict-bundle-record.md#how-deep-the-prefix-reaches-and-what-the-depth-costs).

#### conflict-claim-yield-across-store-generations (optional)

The claim tier's candidate list at a fixed budget is a RANK cutoff into the store's own similarity
ordering, so it is a property of the store as much as of the corpus -- and the two measured goods
budget-100 runs disagree sharply about how much the corpus contains: 8 actionable rows on the
1,139-chunk store at resolved cosine 0.3648, 1 actionable row on the 954-chunk store at 0.3604
([conflict detection](current/data-prep/conflict-decision-groups.md#measured-on-the-goods-corpus)). The
two runs differ in chunk count, duplicate collapse, and resolved threshold at once, so nothing
establishes which factor moved the yield, and an operator cannot tell whether a low actionable count
means a clean corpus or an unlucky store. Vary one factor at a time (duplicate collapse on/off,
chunk size, budget) on the same corpus and record which one the yield tracks.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. Reuse `make build-index` per store variant and `make audit-corpus-conflicts
  EFFORT=claim MAX_CANDIDATE_PAIRS=100`; the calibration probe and precision block are current
  behavior.
- User-visible outcome: an operator reading a low actionable count learns whether it is evidence
  about their corpus or about the store they happened to build.
- Scope boundary: in scope -- the one-factor-at-a-time store sweep, the per-variant actionable
  yield and overlap of the returned rows, and a stated reading. Out of scope -- changing the
  chunker, the threshold calibration, and the candidate budget defaults before the sweep supports it.
- Data and artifact paths: one `$DATA_DIR/corpus-conflicts/<run>/` per variant.
- Execution path: one claim run per store variant on the CUDA host; no new CI coverage beyond the
  existing fixtures.
- Acceptance gates: `make ci` green; every variant reports its actionable yield and its row overlap
  with the baseline variant; the reading names the factor the yield tracks, or records that the
  variants do not separate.
- Documentation target: [conflict
  detection](current/data-prep/conflict-detection.md#measured-claim-tier-precision).

#### conflict-policy-delta-on-an-operator-corpus-with-dated-revisions (optional)

The policy-choice delta and its share of actionable rows are measured, but on exactly one corpus
where the share is non-zero -- the committed 7-document fixture, which was planted to contain one
dated supersession
([decision groups](current/data-prep/conflict-decision-groups.md#what-the-policy-choice-costs-measured)).
Every other corpus on this host carries no governance dates at all, so its zero share is structural
and carries no information about magnitude. The open question is unchanged and now sharper: on a
corpus of genuine dated revisions, what share of the actionable rows does the policy choice move?
Run the shipped `--project-policy conservative,prefer-newer` reading against the 8-document HR
corpus (operator data, absent from this host) or another corpus with governance dates on both sides
of a revision pair, and record the share beside the fixture's.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. The projection, the delta, and the `moved_rows` / `moved_share` reading are
  current behavior; the corpus needs `effective_date` or `version` on both sides of a revision
  pair, which is what promotes `contradicts` to `superseded_by`.
- User-visible outcome: an operator learns whether the resolution-policy choice is a real decision
  on corpora like theirs, or a knob that is free in practice.
- Scope boundary: in scope -- one claim-tier run per dated corpus and the share beside the
  fixture's. Out of scope -- adding a policy, changing how `superseded_by` is derived, and making
  any policy the default.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` layout.
- Execution path: one claim-tier run per corpus on the CUDA host with both policies projected; no
  new CI coverage beyond the existing fixtures.
- Acceptance gates: `make ci` green; every corpus reports its delta and the share of actionable
  rows it moves; each projected column still equals the `plan.json` `review_rows` the same policy
  writes on the same rows.
- Documentation target:
  [decision groups](current/data-prep/conflict-decision-groups.md#what-the-policy-choice-costs-measured).

#### conflict-policy-share-across-repeat-audits-of-one-corpus (optional)

The policy-choice share is quoted from ONE audit and is not reproducible across audits of the same
corpus: three claim-tier runs of the committed fixture at identical settings returned the same 17
rows, the same 9 actionable rows, and the same 0.4286 claim-tier precision, yet the third called one
row `subsumed_by` where the first two called it `superseded_by` -- so the delta read 1 of 9 (11.1%)
instead of 2 of 9 (22.2%)
([decision groups](current/data-prep/conflict-decision-groups.md#what-the-policy-choice-costs-measured)).
The share is a count of the one relation the policies part on, so it inherits the adjudicator's
sampling variance undivided, and the endpoint runs at `temperature 0.2` with no seed
(`EndpointConfig` in `src/llb/prep/ontology/endpoints/config.py`). Audit one corpus N times at the
shipped settings, report the spread of the relation mix and of `moved_share`, and decide between the
two fixes the spread implies: pin the adjudication call (temperature 0 plus a seed where the backend
honors one) so a repeat audit is comparable, or quote the share with a run-to-run band. A negative
result -- the spread is small enough that a point estimate is honest -- is a valid outcome.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. `make audit-corpus-conflicts EFFORT=claim PROJECT_POLICY=conservative,prefer-newer`
  and its artifacts are current behavior; the three bundles above are the first three samples and
  need no re-running. Do not change the relation vocabulary or the prompt -- that would measure a
  different adjudicator instead of this one's variance.
- User-visible outcome: an operator comparing this week's policy share to last week's learns which
  part of the difference is their corpus and which part is the model being asked twice.
- Scope boundary: in scope -- the repeat runs, the spread of the relation mix and the share, and the
  pin-or-band decision. Out of scope -- changing the adjudication prompt, the calibration gate, and
  quoting a band before the spread is measured.
- Data and artifact paths: one `$DATA_DIR/corpus-conflicts/<run>/` per repeat.
- Execution path: N claim-tier runs of one corpus on the CUDA host; CI covers whatever pinning the
  decision adopts, over the injected adjudicator.
- Acceptance gates: `make ci` green; every repeat reports its relation mix and share; the reading
  states the spread and either pins the call or records the band the share must be quoted with.
- Documentation target:
  [decision groups](current/data-prep/conflict-decision-groups.md#what-the-policy-choice-costs-measured).

#### conflict-decision-group-partition-refinement (optional)

The audit now quotes a RANGE because neither measured rule is a decision count: the transitive
closure is a partition but too coarse, and the shared-unit rule is finer but a COVER -- 65 to 80
percent of rows join two of its groups on every corpus measured, so its count cannot be funded one
review each
([decision groups](current/data-prep/conflict-decision-groups.md#how-many-decisions-the-row-count-is)).
A single number needs a third rule that is BOTH a partition and finer than the closure. Measure one:
cut the closure's chain at its weakest joins (a shared unit carrying only low-score pairs, or only
`complementary` relations, is a chunk two decisions merely pass through) and check whether the
resulting partition lands inside the measured range rather than collapsing back to one of its ends.
A negative result -- no cut rule lands inside without splitting a genuine decision -- is a valid
outcome and closes the question.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED, RESEARCH
- Dependencies: none. Reuse `granularity.py`'s two rules as the bracket the third must sit inside and
  `compare-conflict-granularity` to read the result over committed runs.
- User-visible outcome: either one decision count an operator can fund directly, or a recorded reason
  the audit will keep quoting a range.
- Scope boundary: in scope -- the cut rule, its partition proof, and the comparison against both
  measured ends. Out of scope -- changing detection, changing `findings.jsonl` or the group ids
  `plan.json` joins on, and adopting a third rule as the quoted one before it lands inside the range.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` and
  `$DATA_DIR/corpus-conflict-granularity/<run>/` layouts.
- Execution path: recompute over committed run artifacts on the CUDA host; the measured bundles are
  all narrower than the 8-document HR corpus (which is operator data and absent from this host), so
  the run needs at least one bundle from a corpus that supplies a genuinely long chain. CI covers the
  cut rule and the partition invariant over fixture rows.
- Acceptance gates: `make ci` green; the third rule is asserted to be a partition on every fixture;
  every measured bundle reports its count under all three rules; the reading states whether the
  partition lands strictly inside the range, or records that it does not and why.
- Documentation target:
  [decision groups](current/data-prep/conflict-decision-groups.md#how-many-decisions-the-row-count-is).

### Documentation integrity -- `documentation-integrity`

#### temp-artifact-citations-on-the-12gb-host

The delivered docs must not cite evidence by a `$DATA_DIR/<method>/<run-id>/` path: that directory
is host-local and temporary, so it is gone after a cleanup, absent on a fresh checkout, and absent
on every other machine -- and a bare run label like `20260815T-bare-id-squad-cos060` is no better,
because it identifies nothing a reader can use ([AGENTS.md](../../AGENTS.md), "Citing a measured
result"). The pages whose runs live on THIS host now carry the description, the date, the host, and
the numbers instead. What remains is 59 path citations and 6 bare run labels, across 20 pages whose
runs were measured on the OTHER GPU host (the
12 GiB box -- the `blackwell12`, `context-ablation`, `encoder-throughput`, `verbosity-sensitivity`,
`query-robustness`, and `agentic-compact-window-elision` families among them). They were left alone
deliberately: converting a citation means reading the run's own numbers out of the bundle and
inlining them, which cannot be done from a host that does not hold it. Run this task ON that host.

- Serves: `documentation-integrity` -- [Documentation integrity](../design/spec.md#specification-and-plan-integrity)
- Agent status: BLOCKED BY HUMAN (needs a session on the 12 GiB GPU host; no human judgment beyond
  running it there)
- Dependencies: none in code. The citation rule and the target shape are in
  [AGENTS.md](../../AGENTS.md); the already-converted pages under
  [GraphRAG](current/graphrag-backend/answer-quality-evidence.md#answer-quality-evidence) are the
  worked example.
- User-visible outcome: every measured result in the delivered docs stays checkable from the page
  alone, on any machine, after every run directory is deleted.
- Scope boundary: in scope -- replacing each remaining path citation with a run-label citation plus
  the load-bearing numbers read out of that run's own bundle, and adding the reproduction recipe and
  boundaries the page is missing. Out of scope -- re-running any measurement, changing any recorded
  number, converting a citation whose bundle is missing on that host too (record it as unrecoverable
  in the page instead), and touching the `$DATA_DIR` path TEMPLATES that document where a command
  writes.
- Data and artifact paths: read-only over that host's own `$DATA_DIR`; this task writes only docs.
- Execution path: enumerate the remaining citations with a scan for BOTH forms -- a backticked
  `$DATA_DIR/` or `.data/` span whose path carries a `<8-digit>T` run segment, and a backticked bare
  `<8-digit>T...` run label outside a table row -- convert each page, then re-scan to confirm none
  is left; `make lint-md` covers wrapping and links.
- Acceptance gates: `make ci` green; `make lint-md` green; the scan above returns zero of BOTH
  forms across `docs/`; every page it touched states, for each result, what was measured on what,
  the date, the host, the numbers the verdict rests on, and the reading those numbers support.
  Once the scan is clean on both hosts,
  add it to `make lint-doc-links` so a re-introduced path citation fails the build instead of being
  found by the next reader.
- Documentation target: the 19 pages the scan names, and
  [overview](current/overview.md#documentation-and-specification-gates) for the new check.

#### conflict-bundle-record-page-is-past-the-split-threshold (optional)

[bundle record](current/data-prep/conflict-bundle-record.md) is ~740 lines and its headings describe
two subjects: what SHAPE the record is written in (the id table, the label it dropped, the affix
fold, the count default, the store identity, and the size table that prices them) and what the
record ANSWERS (the verdict table, the exclusion floor, the candidate budget, the depth/cost curve).
That is well past the ~500-line split rule in [AGENTS.md](../../AGENTS.md), and the shape subject is
where every successive size change lands, so the page grows in one place and has now been pushed
past the rule three times. The test side is already split along exactly this seam
(`test_bundle_record.py` against `test_bundle_id_table.py` and `test_store_identity.py`), so the
heading seam is established rather than proposed --
move the shape subsections to a new topic page, add its row to the area page, and repoint the
inbound links.
Anchors keep their fragments, so only paths change and `make lint-doc-links` proves the move.

- Serves: `documentation-integrity` -- [Documentation integrity](../design/spec.md#specification-and-plan-integrity)
- Agent status: CLEAR
- Dependencies: none. `make lint-md` (which runs `lint-doc-links`) is the whole gate; the inbound
  links are in `plan.md`, `conflict-detection.md`, and `conflict-decision-groups.md`.
- User-visible outcome: a reader looking for what a bundle can answer stops scrolling past three
  successive byte-accounting decisions to reach it.
- Scope boundary: in scope -- the split, the area-page row, and the link repointing. Out of scope
  -- rewriting the moved text and changing any measured result.
- Acceptance gates: `make lint-md` green with zero broken links; neither page is past ~500 lines;
  no anchor text changes.
- Documentation target: [data prep](current/data-prep.md) and the two pages the split produces.

#### conflict-decision-groups-page-is-past-the-split-threshold (optional)

[decision groups](current/data-prep/conflict-decision-groups.md) is ~800 lines and its headings
describe two subjects: how many decisions a row count is (census, grouping rules, ranking, the
`groups.json` sidecar) and what a policy choice costs (the projection, the governance-coverage
precondition, the stage attribution and its bundle re-read). That is past the ~500-line split rule
in [AGENTS.md](../../AGENTS.md), and the second subject is where every recent addition lands, so the
page grows in one place. Split along the heading seam -- move the projection/coverage/stage
subsections to a new topic page, add its row to the area page, and repoint the inbound links.
Anchors keep their fragments, so only paths change and `make lint-doc-links` proves the move.

- Serves: `documentation-integrity` -- [Documentation integrity](../design/spec.md#specification-and-plan-integrity)
- Agent status: CLEAR
- Dependencies: none. `make lint-md` (which runs `lint-doc-links`) is the whole gate; the inbound
  links are in `plan.md`, `conflict-detection.md`, and `conflict-resolution.md`.
- User-visible outcome: a reader looking for what a policy choice costs stops scrolling past the
  counting rules to reach it.
- Scope boundary: in scope -- the split, the area-page row, and the link repointing. Out of scope
  -- rewriting the moved text and changing any measured result.
- Acceptance gates: `make lint-md` green with zero broken links; neither page is past ~500 lines;
  no anchor text changes.
- Documentation target: [data prep](current/data-prep.md) and the two pages the split produces.

#### entity-resolution-page-is-a-single-page-area-with-four-subjects (optional)

[entity resolution](current/entity-resolution.md) is ~650 lines and its headings describe four
subjects: the shared linkage seam (its vocabulary, artifacts, commands, and committed sample) and
one page's worth of lane apiece for gold items, graph nodes, and document editions. That is past the
~500-line split rule in [AGENTS.md](../../AGENTS.md), and the page is a single-page AREA rather than
a container, so every new lane lands on it. Turn it into an area directory: keep the orientation,
the confidence contract, and the boundary on `entity-resolution.md` with a table of its pages, and
move the seam and the three lanes to `current/entity-resolution/<topic>.md`. Anchors keep their
fragments, so only paths change and `make lint-doc-links` proves the move.

- Serves: `documentation-integrity` -- [Documentation integrity](../design/spec.md#specification-and-plan-integrity)
- Agent status: CLEAR
- Dependencies: none. `make lint-md` (which runs `lint-doc-links`) is the whole gate; the inbound
  links are in `plan.md`, `current.md`, `robustness-ontology-backends.md`, two `graphrag-backend/`
  pages, `data-prep/ingestion-corpora.md`, `data-prep/conflict-detection.md`, and the
  `samples/corpora/editions_uk_v1/README.md`.
- User-visible outcome: a reader who wants one lane stops scrolling past the other three and the
  seam to reach it, and the next lane to move onto the seam gets a page instead of a section.
- Scope boundary: in scope -- the split, the area-page table, and the link repointing. Out of scope
  -- rewriting the moved text, changing any measured result, and merging the per-lane test
  paragraphs.
- Acceptance gates: `make lint-md` green with zero broken links; no page past ~500 lines; no anchor
  text changes.
- Documentation target: [entity resolution](current/entity-resolution.md) and the pages the split
  produces.

## Human-Assisted Tasks

Work whose acceptance needs human judgment or authorization, in the same capability
order. An agent can build the supporting code and unit tests; the marked human step is
what gates completion.

### Gold data and ground truth -- `gold-data`

#### multihop-ledger-human-acceptance

Accept (or reject) the drafted multi-hop retrieval slice through the verification gate, then re-run
BOTH draft-grounded lanes on the accepted ledger -- the fusion sweep and the end-to-end
answer-quality comparison -- so the graph-weight verdict rests on human-reviewed questions instead
of drafted ones. The drafted set, its worksheet, the matched vector/graph stores, and the measured
draft-grounded sweep plus answer-quality comparison are current behavior in
[GraphRAG](current/graphrag-backend.md); every
drafted multi-hop item is span-exact and Ukrainian-gated by construction, but only a reviewer can
say whether a shared-bridge question genuinely needs both facts.

- Serves: `gold-data` -- [Gold data and ground truth](../design/spec.md#data-and-ground-truth)
- Agent status: HUMAN-GATED
- Dependencies: the [widened handoff](current/graphrag-backend/fusion-sweep-evidence.md#widened-multi-hop-review-handoff)
  supplies the worksheet, while the [paired-power
  contract](current/rag-core/paired-verdicts.md#paired-power-contract-for-comparison-lanes) derives
  the accepted-ledger requirement from a predeclared minimum detectable retrieval gain, expected
  discordance, confidence, and power. Human step that gates completion: a reviewer decides
  `accept`/`reject` for every worksheet row -- specifically whether the question is answerable ONLY
  with both cited spans -- and signs off on the resulting accepted ledger.
- User-visible outcome: a graph-weight recommendation for multi-hop retrieval backed by a
  human-accepted ledger, or a recorded finding that shared-bridge drafting does not produce
  genuine multi-hop questions and the slice must come from another source.
- Scope boundary: in scope -- worksheet review, `verify-accept`, re-running the sweep over BOTH
  span-identity policies on the accepted ledger, and the adopt-or-reject verdict per knob --
  including whether `graph_fusion_span_identity` flips from `exact` to `overlap` as the shipped
  default, which is currently gated only by the drafted ledger
  ([GraphRAG](current/graphrag-backend/span-and-depth-evidence.md#span-identity-evidence)). Out of
  scope -- graph schema changes and fusion mechanics (the candidate-depth and span-identity verdicts
  are current behavior in
  [GraphRAG](current/graphrag-backend/span-and-depth-evidence.md#candidate-depth-evidence)).
- Data and artifact paths: the widened drafted bundle and worksheet named in
  [the handoff](current/graphrag-backend/fusion-sweep-evidence.md#widened-multi-hop-review-handoff),
  plus a new `$DATA_DIR/graph-vector-fusion-multihop/<run>/` sweep over `accepted/goldset.jsonl` and
  its `answer-quality/` comparison.
- Execution path: start at `make verify-review VERIFY_WS=<widened-multi-hop-worksheet>`, then
  `make verify-accept VERIFY_WS=<widened-multi-hop-worksheet>
  BUNDLE=<widened-multi-hop-bundle>`, then
  `make compare-graph-fusion GOLDSET=<accepted>/goldset.jsonl
  GRAPH_FUSION_CANDIDATES=k,50 GRAPH_FUSION_SPAN_IDENTITY=exact,overlap`,
  then `make compare-answer-quality GOLDSET=<accepted>/goldset.jsonl FUSION_COMPARISON=<that
  sweep>/comparison.json` -- WITHOUT `INCLUDE_DRAFTED`, since an accepted ledger no longer needs
  the drafted-grounding escape.
- Acceptance gates: every worksheet row has a decision; the accepted ledger satisfies its
  predeclared paired-power requirement; retention is reported by relation pair, document mode, and
  source document with uncertainty; the re-run sweep reports the same rows with paired intervals and
  the human records the adopt-or-reject verdict per graph strategy and per span-identity policy; the
  answer-quality comparison re-runs on the accepted ledger with `grounding: verified`. If power
  remains insufficient or rejection failures are statistically concentrated in a stratum, run the
  relation-stratified widening workflow from [data
  prep](current/data-prep/ingestion-corpora.md#widening-a-multi-hop-review-slice) against the latest
  reviewed ledger.
- Documentation target: the graph-vector fusion evidence section of
  [GraphRAG](current/graphrag-backend.md).

### Entity resolution -- `entity-resolution`

#### entity-merge-labelled-set

Produce the reviewer-labelled merge set that turns a linkage probability into an operating point. An
unsupervised fit ranks pairs and clusters them, but nothing in it says where to cut: the
confidence contract publishes a match probability only with the labelled set it was scored against,
so until a human has read a sample of proposed merges there is no defensible default threshold for
either the graph overlay or the gold-item suppression policy. Sample the proposed merges across the
probability range -- not only the confident ones, because the threshold lives where the model is
unsure -- and record same-thing / different-thing / cannot-tell per pair through the review
workbench, so the decisions land in a ledger like every other human gate.

- Serves: `entity-resolution` --
  [Entity resolution](../design/spec.md#entity-resolution-and-record-linkage)
- Agent status: HUMAN-GATED
- Dependencies: whichever supplies the candidate pairs to label -- the gold-item shadow lane's
  disagreement list ([the gold-item lane](current/entity-resolution.md#the-gold-item-lane)) or the
  graph node lane's scored pairs
  ([the graph node lane](current/entity-resolution.md#the-graph-node-lane)). The seam's label-based
  fit and labelled accuracy curve already exist -- see
  [entity resolution](current/entity-resolution.md). Human step that gates
  completion: a Ukrainian-reading reviewer decides each sampled pair. Reuse the adapter pattern in
  [review workbench](current/review-workbench.md) rather than building a second review surface.
- User-visible outcome: both linkage domains gain a threshold with a precision and recall attached
  to it, and the operator can see what a stricter or looser cut would cost.
- Scope boundary: in scope -- the workbench adapter for merge decisions, a stratified sample across
  probability bands, the label ledger, the labelled accuracy curve computed from it, and the
  threshold recommendation that follows. Out of scope -- labelling every candidate pair, labelling
  CONFLICT relations (a different question with a different adjudicator), and adopting a threshold
  that the paired retrieval or drafting evidence does not support.
- Data and artifact paths: `$DATA_DIR/entity-merge-labels/<run>/` -- the sampled pairs, the decision
  ledger, and the accuracy curve; the ledger is the frozen label set later fits are scored against.
- Execution path: generate the sample from a linkage run's `pairs.jsonl`, review it in the
  workbench, then re-score the model against the ledger with the seam's label-based fit and accuracy
  analysis.
- Acceptance gates: every sampled pair carries a decision or an explicit cannot-tell; the sample
  covers each probability band, including the uncertain middle; the curve reports precision and
  recall at several thresholds with the reviewer cost the sample took; the recommended threshold
  is stated with what it would merge and what it would leave apart.
- Documentation target: the labelling section of `docs/impl/current/entity-resolution.md`, with the
  measured reviewer cost going to [review workbench](current/review-workbench.md).

#### gold-item-drop-policy-adoption

Decide whether the gold-item drop policy moves from the shipped cosine constant to the fitted
linkage model, and make the decision on the labelled curve rather than on the shadow report alone.
The shadow lane already publishes, per drafting run, the cut that reproduces today's decisions and
every item where a probability cut and the constant disagree -- see
[the gold-item lane](current/entity-resolution.md#the-gold-item-lane). What it cannot supply is
which side of each disagreement is right: a paraphrase the constant kept and the model would drop
is either a duplicate the constant missed or a distinct question the model would destroy, and only
a reviewer decision separates those. Score the model against the merge ledger, read the threshold
off the labelled accuracy curve, and either flip `NEAR_DUP_COSINE_THRESHOLD` and the multi-hop
answer constant for the model at that threshold, or record why the labelled curve does not support
the move.

- Serves: `entity-resolution` --
  [Entity resolution](../design/spec.md#entity-resolution-and-record-linkage)
- Agent status: BLOCKED BY HUMAN
- Dependencies: the reviewer merge ledger from `entity-merge-labelled-set` under Human-Assisted
  Tasks is the gate -- without it there is no curve to read a threshold off. The lane, its record
  specification, and the shadow report already exist; the seam's label-fitted m estimation and
  labelled accuracy curve already exist.
- User-visible outcome: a drafting run's suppression decision carries a threshold with a precision
  and recall attached to it, and a reviewer who wants a looser or stricter policy can see what it
  would cost before changing it.
- Scope boundary: in scope -- scoring the fitted model against the merge ledger, the threshold
  recommendation, flipping the default (or recording the negative result), and re-running the
  shadow report at the adopted cut so the change is visible in one diff. Out of scope --
  re-deduplicating already-accepted ledgers, changing what a gold item contains, changing the
  drafting prompts, and adopting a cut the labelled curve does not support.
- Data and artifact paths: the drafting bundle's `linkage/` gains the accuracy curve a labelled fit
  writes; the adopted threshold lands beside the constants it replaces.
- Execution path: the drafting entrypoint with the shadow lane enabled and the merge ledger passed
  as the label table, over the same fixtures the lane is tested on.
- Acceptance gates: `make ci` green; the model is scored against the ledger with precision and
  recall at the adopted cut and at the shipped constant's operating point, so the two policies are
  compared on the same labelled pairs; a cut that does not beat the constant on both is recorded as
  a negative result and the constant stays.
- Documentation target: the gold-item section of `docs/impl/current/entity-resolution.md`.

### Retrieval evidence -- `retrieval-evidence`

#### embedder-decision-on-a-resolvable-item-set

The embedder choice is undecidable on the item sets the repo currently has, and the paired lane says
so precisely: on the accepted converted-PDF ledger 36 of 40 questions are TIED between the leader
and the incumbent, so the 95% paired interval spans `[-0.050, +0.150]` and only a consistent
~4-question gap could ever clear zero; on the committed fixture the baseline already retrieves
0.980, leaving 5 questions of headroom for any candidate to win ([RAG
core](current/rag-core/embedders.md#the-recommendation-re-read-with-paired-uncertainty)). The
sub-base roster addition `intfloat/multilingual-e5-small` is ~3x faster on warm CUDA with flat
quality on n=82 and still RETAIN ([RAG
core](current/rag-core/embedders.md#blackwell-sub-base-encoder-roster-e5-small)) -- include it when
the enriched ledger re-runs so a cheap CUDA swap can clear an adoption bar if the discordance is
there. Both existing sets are at their ceiling, which is a property of the QUESTIONS, not of the
encoders. Build an item set that can decide it: predeclare a minimum detectable recall gain and the
split size it needs, then assemble a ledger enriched with questions the incumbent currently MISSES
(mine the per-item vectors in `report.json` for baseline zeros, plus domain-term and
morphology-heavy questions the general E5 encoder is expected to fail), accept it through the
verification gate, and re-run the bake-off on it. Record whether any candidate then separates -- a
recorded "still undecidable at n=N" is a valid outcome and is what would justify closing the
question. The size the ENRICHMENT has to buy is already priced: the withdrawn `e5-large` adopt
differs on 5 of 250 items, and at that rate the reporting level needs 300 items, which no committed
goldset reaches -- so plain "more questions" is not the route, raising the discordance rate is
(double the rate, halve the floor) ([the
re-decision](current/rag-core/paired-verdicts.md#the-re-decision-what-a-withdrawn-reading-needs)).

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: BLOCKED BY HUMAN
- Dependencies: the [shared paired-power contract](current/rag-core/paired-verdicts.md#paired-power-contract-for-comparison-lanes)
  supplies the item count the predeclared gain needs; the paired bake-off lane, the verdict, and
  `report.json` are current behavior ([RAG
  core](current/rag-core/paired-verdicts.md#paired-uncertainty-and-the-adopt-or-retain-verdict)).
  Human step that gates completion: a reviewer accepts the enriched question set through `make
  verify-review` / `make verify-accept`, since an unaccepted ledger cannot settle a default.
- User-visible outcome: either a measured embedder swap an operator can adopt, or a recorded
  statement of how many questions a decision would need -- instead of a permanently open ranking.
- Scope boundary: in scope -- the power target, the enriched ledger, its acceptance, and one
  re-run per corpus with the verdict restated. Out of scope -- new candidates, fine-tuning (that
  is `ua-embedder-domain-finetune`), and widening the verdict bar beyond recall@k.
- Data and artifact paths: the existing `$DATA_DIR/compare-embeddings/<run>/` layout.
- Execution path: `make compare-embeddings GOLDSET=<accepted-enriched-goldset> NOISE_FLOOR=1` on
  the CUDA host; no new CI coverage.
- Acceptance gates: the predeclared minimum detectable gain and split size are written down BEFORE
  retrieval; every candidate row reports its paired interval on the accepted ledger; the verdict is
  recorded as adopt, retain, or explicitly undecidable at the reached sample size.
- Documentation target: the embedder bake-off evidence in [RAG core](current/rag-core.md) and the
  recommendation line in [platform matrix](current/platform-vector-matrix.md#embedding-bake-off).

#### goods-fusion-weight-accepted-ledger

Settle the goods-corpus fusion-weight verdict on an item set someone accepted. The recorded verdict
("the BM25 side costs recall at w=0.5, pin `FUSION_WEIGHT=0.7`") was measured on a verified 44-item
quickstart-PDF accepted goldset that is no longer on disk, and the lexical-row re-read could not
reproduce it: on the SAME corpus at the SAME chunking, the 95-item drafted goldset inverts it --
fusion ADDS recall at w=0.5 (+0.021, +0.053 with lemmas, against a +/-0.000 floor) and w=0.7 is the
worst of the three weights for the best row ([RAG
core](current/rag-core/hybrid-retrieval.md#lexical-row-re-read-of-the-fusion-weight-verdict)). The
pin is already withdrawn; what remains is deciding whether the recorded verdict was an artifact of
its item set or of the drafting, which only an accepted ledger over that corpus can answer.

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: HUMAN-GATED
- Dependencies: none in code -- `make compare-retrieval HYBRID=1 NOISE_FLOOR=1` and the
  verification gate are both current behavior. Human step that gates completion: a reviewer
  accepts (or rejects) enough of the drafted goods questions through
  `make verify-review` / `make verify-accept` to produce an accepted goods ledger.
- User-visible outcome: an operator on a converted-PDF Ukrainian corpus gets a fusion-weight
  recommendation backed by human-reviewed questions, instead of two drafted-versus-vanished item
  sets that disagree.
- Scope boundary: in scope -- worksheet review, the accepted ledger, one re-run per recorded
  weight with the `lexical` row, and a keep-or-change verdict on the recorded table. Out of scope
  -- widening the weight grid, new fusion mechanics, and changing the shipped defaults before the
  accepted run supports it.
- Data and artifact paths: the drafted bundle under
  `$DATA_DIR/graph-vector-fusion-multihop/goods-draft/` plus a new
  `$DATA_DIR/lexical-row-reread/goods-accepted-w<weight>/` per weight.
- Execution path: `make verify-review` then `make verify-accept` over the goods draft, then
  `make compare-retrieval CONFIG=<cfg> GOLDSET=<accepted>/goldset.jsonl SPLIT= HYBRID=1
  NOISE_FLOOR=1` at `FUSION_WEIGHT=0.5,0.6,0.7`.
- Acceptance gates: every worksheet row has a decision; the three weights are scored on the
  identical accepted item set with the `lexical` row and the corpus's floor; the recorded table is
  restated as reproduced, corrected, or retired.
- Documentation target: the hybrid-retrieval evidence section of
  [RAG core](current/rag-core/hybrid-retrieval.md#hybrid-retrieval-dense--bm25--rrf).

#### fusion-routing-calibration-power (optional)

Increase the sidecar-free routing calibration's statistical power before reconsidering its
production defaults. The first held-out measurement cannot separate its positive retrieval deltas
from zero; see the compact result and frozen-policy diagnostics in
[GraphRAG](current/graphrag-backend/sidecar-free-routing-calibration.md#sidecar-free-heuristic-calibration).
Assemble a larger, independent multi-span tuning/final ledger, declare its minimum detectable gain
and split sizes before retrieval, then repeat the frozen-policy workflow without widening the
threshold grid.

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: BLOCKED BY HUMAN
- Dependencies: the [shared paired-power contract](current/rag-core/paired-verdicts.md#paired-power-contract-for-comparison-lanes)
  supplies the predeclared split sizes, and `multihop-ledger-human-acceptance` must provide a
  non-empty accepted multi-span ledger. Human
  step that gates completion: accept enough additional genuinely multi-span questions to meet the
  predeclared split sizes.
- User-visible outcome: operators can distinguish a useful sidecar-free route from a sparse-win
  artifact before changing the fallback defaults.
- Scope boundary: in scope -- a prospective power target, disjoint tuning/final splits, and one
  repeat of the existing deterministic calibration. Out of scope -- widening the threshold grid,
  a learned router, and selecting on final.
- Data and artifact paths: a new `$DATA_DIR/graph-vector-fusion-multihop/<run>/` calibration over
  the accepted ledger.
- Execution path: run `make calibrate-fusion-routing` with the predeclared splits, then run the
  masked `make compare-graph-fusion` reproduction for the frozen policy on each split.
- Acceptance gates: route precision/recall and paired retrieval intervals meet the predeclared
  power target; a threshold changes only if the tuning gain clears zero without single-span
  regression and the untouched final split confirms the same direction.
- Documentation target: the sidecar-free calibration subsections of
  [RAG core](current/rag-core.md) and [GraphRAG](current/graphrag-backend.md).

#### embedding-clustered-chunk-merging (optional)

The measured near-duplicate residue is real but not text-reachable: on the goods corpus 20.7% of the
exact-collapsed chunks have a neighbour at cosine >= 0.99, and the `normalized` collapse tier merges
26 of those 13105 pairs ([RAG
core](current/rag-core/retrieval-store.md#near-duplicate-residue-and-the-collapse-tiers)). Only an
embedding-side merge can reach the rest, which the collapse lane deliberately does not do because a
false merge silently deletes a distinct passage from the index. Decide it with a measured
false-merge rate instead of by assumption: cluster the survivors by cosine at several thresholds,
sample the merges for human reading (the residue report's pair sampler already renders them), and
score retrieval per threshold against the corpus's own measurement floor. Adopt only if a threshold
lowers the fragile count without a recall regression AND its sampled false-merge rate is acceptable
on a corpus whose facts differ by one number.

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: HUMAN-GATED
- Dependencies: none. Reuse `measure_duplicate_residue` in `src/llb/rag/duplicates/residue.py` for
  the clustering and the sampler, and `collapse_duplicate_chunks` for the merge itself. Human step
  that gates completion: a reviewer reads the sampled merges at the candidate threshold and calls
  the false-merge rate acceptable or not.
- User-visible outcome: either an embedding-side merge tier with a measured false-merge rate, or a
  recorded finding that the residue must be left in the index.
- Scope boundary: in scope -- the clustering, the sampled review, the per-threshold retrieval run,
  and the adopt-or-reject verdict. Out of scope -- learned merge policies and corpus rewriting.
- Data and artifact paths: `$DATA_DIR/retrieval-noise-floor/<run>/`.
- Execution path: `make measure-duplicate-residue` per threshold, then `make compare-retrieval
  CHUNK_STRATEGIES=sentence,recursive NOISE_FLOOR=1` per candidate on the CUDA host.
- Acceptance gates: `make ci` green; the report carries the per-threshold recall against the floor,
  the fragile count, and the human's false-merge reading.
- Documentation target:
  [RAG core](current/rag-core/retrieval-store.md#near-duplicate-residue-and-the-collapse-tiers).

#### procedural-size-lever-resolution (optional)

The `size` lever against fragmented procedural evidence came ONE item short of a reading: on the
14-item `procedural` slice of the goods ledger, `size400` doubles whole-span delivery (0.357 ->
0.714) on 7 discordant items split 6/1, an exact randomization p of 0.0625 against the 0.025 a
separation needs, where a clean 7/0 would have separated ([two levers against fragmented
evidence](current/rag-core/fragmented-evidence.md)). The slice already clears the minimum-evidence
gate, so this is not an underpowered comparison -- it is a decidable one that came out short, and a
handful of accepted procedural items is what settles it. Enrich the procedural slice through the
verification gate, re-read the same `CHUNK_SIZES` lever on it, and -- only if it then separates --
take the end-to-end answer reading the spec requires of a retrieval configuration change, so the
served-context price is weighed against an answer the model actually produced rather than against
intactness alone.

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: HUMAN-GATED
- Dependencies: none in code. Reuse the `--sizes` lanes and per-slice paired readings in
  `src/llb/rag/comparison/run.py`. Human step that gates completion: a reviewer accepts the added
  procedural items through the verification gate, since a drafted item cannot license a default
  change.
- User-visible outcome: the operator learns whether a wider `size` cap is worth roughly double the
  served context on procedural questions, or that it is not, on an item set that can say so.
- Scope boundary: in scope -- the added items, the re-read of the same three caps, and the
  answer-side reading conditioned on the model that took it. Out of scope -- changing the default
  `size` without the end-to-end reading, and re-tuning the chunker.
- Data and artifact paths: the existing `$DATA_DIR/table-aware-chunking/<run>/` comparison layout.
- Execution path: `make compare-retrieval CONFIG=<goods.yaml> CHUNK_SIZES=200,400,800 STITCH=1` on
  the CUDA host, then `make compare-answer-quality` on the same items if the slice separates.
- Acceptance gates: `make ci` green; the procedural slice reports its paired `span_intact@k` delta
  with the discordant count beside it; a separation is followed by the answer-side reading naming
  its model, and a non-separation is recorded as the negative result it is.
- Documentation target: [two levers against fragmented
  evidence](current/rag-core/fragmented-evidence.md).

#### cross-lingual-query-fixture-review (optional)

Review the Russian and UA/RU questions in the committed cross-language overlay for linguistic
naturalness and semantic equivalence to each paired Ukrainian question, then either accept the
whole overlay or keep it diagnostic-only with corrected draft rows. The diagnostic lane and its
current evidence are described in [evaluation
rigor](current/rigor-board-judge/robustness-benchmarks.md#cross-lingual-query-lane).

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: HUMAN-GATED
- Dependencies: none in code. Human step that gates completion: a Ukrainian/Russian reviewer reads
  every UA/RU/RU triplet and confirms that both variants preserve the question's factual intent,
  named entities, numbers, and answer target.
- User-visible outcome: the language overlay either becomes accepted evidence or remains clearly
  drafted with a complete list of rows requiring correction.
- Scope boundary: in scope -- linguistic/semantic review, corrections, and one uniform review-state
  transition. Out of scope -- changing source spans or answers, translating the corpus, adding new
  languages, and changing query-prep defaults.
- Data and artifact paths: `samples/goldsets/ua_squad_postedited_v1_ru/goldset.jsonl`; a review may
  change all rows together to `provenance: human-verified` / `verified: true` only after every row
  passes.
- Execution path: compare each overlay pair with its id-matched Ukrainian row, correct rejected
  questions, run `make validate-goldset` against the original corpus, then re-run
  `make bench-query-robustness QUERY_ROBUSTNESS_CLASSES=language_ru,language_mixed`.
- Acceptance gates: every triplet has a reviewer decision; all accepted rows preserve the exact
  non-question payload; the fixture has one uniform review state; validation and the paired CUDA
  rerun are green; the current evidence page records the reviewed result.
- Documentation target: the cross-lingual section of
  [evaluation rigor](current/rigor-board-judge/robustness-benchmarks.md#cross-lingual-query-lane).

### Answer scoring -- `answer-scoring`

#### calibrate-headline-format-weight

Calibrate the fact/format tradeoff against adversarial context-copy answers and human pairwise
utility labels, then retain or revise the declared weight without changing the decomposition
contract ([current scoring](current/rag-core/scoring.md#headline-decomposition-and-declared-ranking-policy)).
Sweep predeclared weights over matched terse, fluent-but-wrong, verbose-supported, and
context-copy cases; measure agreement with the accepted labels and stability across model
families; require a held-out confirmation before changing the default.

- Serves: `answer-scoring` -- [Answer scoring](../design/spec.md#scoring-policy)
- Agent status: BLOCKED BY HUMAN
- Dependencies: a reviewer-accepted pairwise preference ledger covering the four answer shapes.
- User-visible outcome: the explicit format share is grounded in operator utility and resists a
  model that raises recall by repeating retrieved context.
- Scope boundary: in scope -- fixture and worksheet generation, weight sweep, family-stratified
  agreement, uncertainty, and an adopt-or-retain verdict. Out of scope -- prompt rewriting,
  judge-model substitution, and changing the precision/recall/found-rate columns.
- Data and artifact paths: accepted labels under `$DATA_DIR/review/headline-format/`; sweep
  artifacts under `$DATA_DIR/verbosity-sensitivity/<run>/weight-calibration/`.
- Execution path: agent prepares and validates the blinded worksheet; human reviewers accept the
  pairwise labels; agent runs the deterministic sweep and CUDA-host family confirmation.
- Acceptance gates: `make ci` green; all answer-shape strata are represented; held-out agreement
  and confidence intervals are reported per family; any default-weight change names every roster
  rank it changes.
- Documentation target: [RAG core](current/rag-core/scoring.md#headline-decomposition-and-declared-ranking-policy)
  and the ranking policy in [evaluation rigor](current/rigor-board-judge.md).

### Judge calibration -- `judge-calibration`

#### frontier-judge-authorization

Authorize the frontier scorer lane against real providers. The report tooling is current behavior
([frontier judge agreement and cost report](current/rigor-board-judge/judging.md#frontier-judge-agreement-and-cost-report));
what remains is entirely the human authorization and the judgment it produces.

- Serves: `judge-calibration` -- [Judge calibration](../design/spec.md#judge-admission)
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

#### frontier-judge-retrieved-context-agreement

Optional. Re-measure frontier-vs-human judge agreement with *retrieved* contexts instead of the
gold-span windows the authorization lane uses. The current lane deliberately holds retrieval
constant by grounding each item on a window of its gold source document, which isolates judge
behavior but also hands the judge cleaner evidence than a scored run ever gives it. A judge that
ranks well on oracle context may rank differently when the context contains distractors or misses
the answer entirely -- exactly the cases where an autonomous gate matters most. Add a context
source switch to `load_agreement_items` that pulls each item's top-k retrieved chunks from an
existing store, then report both grounding modes side by side so the gap is visible.

- Serves: `judge-calibration` -- [Judge calibration](../design/spec.md#judge-admission)
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

### Graph retrieval and ontology -- `graph-retrieval`

#### ontology-axiom-signoff

Accept or reject each candidate axiom from `ontology-axiom-layer`, one at a time, before any of them
can gate an answer. An axiom is BUSINESS LOGIC, not a measurement: whether "has owner" admits one
value or many, whether `PERSON` and `ORG` are genuinely disjoint in this domain, and whether a
relation's range is closed are claims about the world that no corpus statistic can settle. Inducing
them from corpus frequency would only restate what the extractor happened to emit -- the same
circularity the conflict tier already hit, where the null and the observed population turned out to
be the same set ([data
prep](current/data-prep/conflict-detection.md#known-limitation-there-is-no-independent-null)) -- and
the cost of a wrong axiom is asymmetric and silent: at the ledger it deletes a true fact from the
report's attention, and at the answer gate it converts correct answers into `ontology_violation`.
The corpus cannot review itself here, which is why this is the one piece of the validation
architecture that sits in this section. The existing signed type-vocabulary review is the precedent
for the form ([graph ontology schema](../design/graph-ontology-schema.md)).

- Serves: `graph-retrieval` -- [Graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology)
- Agent status: HUMAN-GATED
- Dependencies: `ontology-axiom-layer` supplies the candidate axioms, their Turtle rendering, and
  the per-axiom evidence (supporting facts, contradicting facts, and the measured base rate on both
  quickstart corpora). Reuse the review-workbench ledger pattern
  ([review workbench](current/review-workbench.md)) so the decisions are recorded the same way every
  other review ledger is. Human step that gates completion: a domain reviewer decides `accept` or
  `reject` for EVERY candidate axiom and signs the resulting axiom file.
- User-visible outcome: a signed, dated constraint set an operator can point the answer gate at,
  where every enabled axiom is a decision someone made rather than a statistic the corpus produced.
- Scope boundary: in scope -- the per-axiom review worksheet (each axiom rendered as Turtle plus a
  Ukrainian-language gloss and its supporting/contradicting facts with exact spans), the review
  pass, the signed axiom file, and a recorded reason per rejection. Out of scope -- authoring new
  axiom CLASSES (that is `ontology-axiom-layer`), changing the 13-type vocabulary or the relation
  caps, and enabling any axiom the reviewer did not accept.
- Data and artifact paths: the worksheet under `$DATA_DIR/ontology-validation/<run>/axiom_review.jsonl`;
  the signed set committed at `samples/ontology/axioms_uk_v1.ttl` with the sign-off line in its
  header, mirroring the dated sign-off convention of
  [graph ontology schema](../design/graph-ontology-schema.md).
- Execution path: `make validate-ontology-axioms` to regenerate the candidates and their evidence,
  then `make review-workbench REVIEW_PATH=<axiom-review-jsonl>`; no GPU is required for the review
  itself.
- Acceptance gates: every candidate axiom has a decision and every rejection has a stated reason;
  the committed axiom file contains only accepted axioms and carries the reviewer name and date;
  re-running the checker over both quickstart corpora with the signed set reports its violation base
  rate per accepted axiom; `ontology-validated-answer-gate` refuses to enable an axiom absent from
  the signed file.
- Documentation target: the ontology-assisted drafting section of
  [robustness and ontology](current/robustness-ontology-backends.md#ontology-assisted-drafting), the
  constraints section of [graph ontology schema](../design/graph-ontology-schema.md), and the
  trust boundary in [product decisions](current/scope-boundaries.md).

### Autonomous orchestration -- `autonomous-orchestration`

#### autonomous-vs-assisted-acceptance

Acceptance-test the full upgrade with a human operator: run `auto-rag` on a real Ukrainian corpus
twice -- once fully autonomous, once with human-assisted gates in the review workbench -- and have
the human judge both the reviewer experience and the recommendation quality.

- Serves: `autonomous-orchestration` -- [Autonomous orchestration](../design/spec.md#autonomous-orchestration)
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

### Corpus conflict and governance -- `corpus-conflict-audit`

#### conflict-adjudicator-label-slice

Produce frozen human labels for real candidate rows so the audit's measured claim-tier precision can
be trusted off the planted fixture. The shipped calibration gate scores the adjudicator only against
the seven-document planted probe, whose relations are synthetic by construction and which the
current host model passes 24/24 ([conflict
detection](current/data-prep/conflict-detection.md#the-frozen-calibration-probe)); nothing measures
whether the model agrees with a human on HR or goods rows.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: HUMAN-GATED
- Dependencies: the audit's precision block and its calibration gate are current behavior
  ([conflict detection](current/data-prep/conflict-detection.md#measured-claim-tier-precision)) and
  consume the resulting bound; candidate ranking and claim adjudication are current behavior too.
  Human step that gates completion: an authorized reviewer assigns one relation from the claim
  vocabulary to every row of the frozen slice without seeing the model's verdict.
- User-visible outcome: a committed frozen slice plus a measured human-versus-adjudicator agreement
  bound -- what lets a precision number transfer to corpora the planted fixture does not represent.
- Scope boundary: in scope -- slice selection stratified by corpus and rank band, blind review,
  agreement measurement, and the resulting calibration bound. Out of scope -- relabelling the planted
  fixture, changing the relation vocabulary, and using the reviewed slice to fit any threshold.
- Data and artifact paths: `$DATA_DIR/corpus-conflicts/<run>/` for the slice and agreement report;
  the frozen labels are committed under `samples/`.
- Execution path: generate the stratified slice from a claim-tier run, review it with
  `make review-workbench REVIEW_PATH=<slice-jsonl>`, then re-measure agreement against it.
- Acceptance gates: every slice row carries a human relation; agreement is reported with a clustered
  interval; and the precision block stays suppressed on any corpus whose agreement bound is unmet.
- Documentation target: [conflict detection](current/data-prep/conflict-detection.md) and [review
  workbench](current/review-workbench.md).

#### corpus-conflict-resolution-review

Review the unresolved semantic conflict candidates through the workbench, then feed the accepted
ledger back into the resolver and repeat the retrieval plus verified answer-quality comparison.
The resolver behavior and the reason semantic candidates have no automatic suppression authority
are current behavior in
[data prep](current/data-prep/conflict-resolution.md#corpus-conflict-resolution-corpus-conflict-resolution).

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
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

### Operator review tooling -- `operator-review-tooling`

#### conflict-group-review-throughput (optional)

Whole-group review is now possible but unmeasured: a reviewer can settle a decision group with one
`keep_both` row, and on the goods semantic bundle six such rows settle all 100 escalations
([conflict resolution](current/data-prep/conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger))
-- but nothing establishes that a human reading ONE group record decides as accurately as one
reading its rows, which is the assumption the whole collapse rests on. Measure it: have a reviewer
settle one corpus's escalations row by row and another's group by group, record wall-clock time per
decision and the disagreement rate between the two passes on the same rows, and state whether group
review is safe for `keep_both` at the group sizes this repo actually produces (largest 51 rows).

- Serves: `operator-review-tooling` -- [Operator review tooling](../design/spec.md#operator-review-tooling)
- Agent status: HUMAN-GATED -- the deliverable is the reviewer's measured throughput and agreement;
  the ledger, the timing capture, and the disagreement report are agent-buildable.
- Dependencies: reuse the grouped review ledger and the group-wide keep in
  [conflict resolution](current/data-prep/conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger)
  and the review TUI adapter in `src/llb/review/adapters/conflicts.py`.
- User-visible outcome: an operator learns whether reviewing by decision costs accuracy before
  adopting it as the default review mode.
- Scope boundary: in scope -- the paired review passes, per-decision timing, the disagreement
  report, and a recommendation on group size limits. Out of scope -- extending group decisions to
  destructive actions before the measurement supports it, and any change to the grouping rule.
- Data and artifact paths: the existing `$DATA_DIR/corpus-conflicts/<run>/` artifacts plus a
  timing ledger beside them.
- Execution path: two review passes over the committed goods bundle; CI covers the timing capture
  and the disagreement report over fixture ledgers.
- Acceptance gates: `make ci` green; both passes cover the identical rows; the report states time
  per decision and the disagreement rate, and either recommends group review or names the group
  size above which it stops being safe.
- Documentation target: [conflict
  resolution](current/data-prep/conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger)
  and the [verification gate](current/data-prep/verification-gate.md).

#### conflict-review-ledger-cost-model (optional)

The review ledger now ranks its blocks by `to review`, on the assumption that N open rows cost a
reviewer N times one row ([conflict
resolution](current/data-prep/conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger)).
That is almost certainly wrong in the direction that matters: the goods semantic bundle's 51-row
group is 51 rows against ONE shared chunk, so a reviewer who reads that chunk once decides the rest
by comparison, while 51 single-row groups are 51 unrelated reads. Ranking on raw open rows therefore
over-states the concentrated group exactly where grouping was introduced to stop over-stating it.
Measure it: time a reviewer (or a scripted proxy over the review TUI) on a concentrated block versus
the same number of scattered rows, fit a per-group cost of the form `a + b * open_rows`, and rank on
the fitted cost -- or record that the linear rank is within the measurement's own noise.

- Serves: `operator-review-tooling` -- [Operator review tooling](../design/spec.md#operator-review-tooling)
- Agent status: HUMAN-GATED
- Dependencies: reuse the ledger, its group blocks, and the review TUI's group titles; the two
  goods bundles under `$DATA_DIR/corpus-conflicts/` already supply one concentrated and one
  scattered shape.
- User-visible outcome: the first decision the ledger shows a reviewer is the one that actually
  costs the most of their time, not the one with the most rows.
- Scope boundary: in scope -- the timing protocol, the fitted cost, and the ranking key it implies.
  Out of scope -- changing what a review record is, group identity, and the audit-side ranking,
  which has no policy to rank on.
- Human step: a reviewer works both shapes under measurement; the fit cannot be produced from
  artifacts alone.
- Acceptance gates: `make ci` green; the report states the fitted per-group and per-row costs with
  their uncertainty, and either changes `_stake_order` or records that the linear rank survives.
- Documentation target: [conflict
  resolution](current/data-prep/conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger).

## Adding Future Tasks

**First decide whether the work is a task at all.** Three kinds of finding look like forward work
and are not:

- A **chore**: a source file over the soft line limit, or a run artifact already inside its size
  bound that could still be smaller. Do it inline while editing the file, or leave it. Written up
  with acceptance gates, a chore competes with the product on equal footing and each round produces
  the next round.
- An **audit of our own output**: an artifact's field layout, an artifact's byte count. The measure
  of an audit is a decision it changes, not a byte it saves.
- A **new domain capability** the spec does not describe. This one IS worth doing, and it is not a
  plan task yet: run the six steps of
  [Extending this specification](../design/spec.md#extending-this-specification) first -- state the
  problem in operator terms, amend the spec, declare the evaluation, register the capability, and
  only then write tasks under it. That order is what makes a discovery durable rather than a private
  detour.

**Then write the task.** It needs a stable descriptive id such as `platform-matrix-power` or
`prompt-system-tuning`, kept only while work remains under it, and enough detail for an engineer or
an agent to execute without guessing. File it under the `###` group of the capability it serves, in
**Agent Implementation Tasks** if it can land to `make ci` green with fixtures/fakes (heavy
deterministic runs on the CUDA host are fine), or under **Human-Assisted Tasks** if a human
review/judgment or authorization gates completion; mark any cross-section block explicitly.

Each task entry must include:

- Serves: the capability id from the
  [capability registry](../design/spec.md#capability-registry) this work advances, linked to the
  spec section that owns it. Checked by `make lint-spec-plan`.
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

When a task surfaces new future work, route it by the three kinds above: a chore gets done or
dropped, a self-audit gets dropped, and a genuine capability goes through the spec lifecycle before
it becomes tasks. Extension is welcome and finishing a task while adding nothing is equally normal --
what is not normal is a task quietly appearing under a capability nobody amended the spec for. Put
current behavior and durable decisions in current docs, never in this plan.

A RESEARCH task whose answer comes back negative leaves this file too, but it is not simply deleted:
move it to [future research](future-research.md) with what closed it and the conditions that would
make it worth reopening. Its measurements belong in the current docs like every other finished piece
of work; what future-research.md adds is the reasoning a later reader needs before spending the same
effort again.
