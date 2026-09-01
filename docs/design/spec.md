# Design: Ukrainian Local-Model Selection

## Purpose

loc-lm-bench selects open-weight language models for Ukrainian RAG, text-analysis, and
safety-gated robotics-agent workloads on the operator's own corpus and hardware. Its output is a
reproducible internal model choice, not a general public leaderboard. The robotics tier evaluates
whether a selected local profile can ground a hardware-operation proposal; a text score never grants
authority to actuate a device.

Public benchmarks are useful candidate filters, but their tasks, language mix, quantization,
retrieval stack, and hardware differ from a local deployment. The project closes that transfer gap
by measuring the exact workload that will run in production.

This specification is a LIVING document. It is the register of what the product does, and it grows
when implementation discovers a capability the domain needs -- see
[Extending this specification](#extending-this-specification). It is not a frozen scope fence: a
useful capability found while building is a reason to amend this file, not a reason to refuse the
work. What it does refuse is capability that arrives WITHOUT an amendment, because a product whose
behavior is not written down cannot be evaluated, documented, or handed over.

## Design Intuition

Model quality is only meaningful after three upstream questions have clear answers:

1. Is the evaluation data correct and representative?
2. Can retrieval expose the required evidence?
3. Can the model and context fit on the target host under a reproducible serving configuration?

The benchmark therefore follows this trust chain:

```text
corpus -> source-span labels -> human verification -> retrieval gate
       -> host-fit serving plan -> model scoring -> immutable run bundle -> tiered board
```

A downstream score cannot repair a weak upstream link. In particular, generation models are not
blamed for evidence the retriever never supplied, and an LLM judge cannot certify data that a human
has not reviewed.

The chain constrains the BUILD ORDER: work on a downstream link while an upstream one is still
moving produces measurements the upstream fix invalidates. It does not fully determine it, because a
capability can be upstream in the chain and finished -- what remains under it is then refinement, not
a blocker. The [capability registry](#capability-registry) records the resulting order and the four
rules behind it, and [`../impl/plan.md`](../impl/plan.md) follows that order rather than the order in
which work was discovered.

## Scope

The primary decision loop covers:

- Ukrainian corpus-grounded RAG;
- structured and narrative text analysis;
- local Ollama, vLLM, and llama.cpp serving;
- host-aware model and context feasibility;
- objective metrics, calibrated judge diagnostics, throughput, VRAM, and power;
- corpus self-consistency: which passages contradict each other and which edition supersedes which;
- record identity: which entity nodes, drafted gold items, and document editions denote the
  same thing;
- robotics RAG over approved manuals and curated multimodal episode evidence, followed by
  policy-gated operation in an emulator or explicitly authorized device canary;
- reproducible sweep, recommendation, and board artifacts.

The project does not try to be a hosted benchmark service, scheduler, model registry, generic agent
platform, robot operating system, or production safety controller.

## Benchmark Tiers

Separate benchmark tiers cover security, tool use, agentic execution, summarization, structured
output, text analysis, and knowledge cutoff. Their metrics remain separate because a score has
meaning only within the task and data contract that produced it.

Tier mixing is out of scope for a single board. Public screens, private RAG results, and each
category suite have separate metric semantics; a comparison presents them side by side or hands off
explicitly, never blended into one leaderboard row.

## Architecture

```text
Typer CLI / Make workflows
          |
          +-> data prep and human gates
          |      corpus -> draft -> verify -> accepted ledger
          |      corpus -> conflict audit -> decision groups -> resolution overlay
          |
          +-> retrieval
          |      chunk -> index -> validate recall/MRR
          |
          +-> execution
          |      resolve backend -> plan memory -> run cases sequentially
          |
          +-> robotics evaluation
          |      episode evidence -> retrieve -> propose -> action gate -> emulator/device adapter
          |
          +-> scoring and persistence
          |      objective metrics -> optional calibrated judge -> run bundle
          |
          `-> analysis
                 board -> recommendation -> MLflow mirror
```

Production Python lives under `src/llb/`. `src/llb/main.py` is the CLI entry point and
`src/llb/cli/` owns command registration. Make fragments group operator workflows by function.
Core typed contracts live in domain-specific modules under `src/llb/core/contracts/`; packages do
not provide facade re-exports.

## Data and Ground Truth

Each RAG gold item contains a question, reference answer, source document id, and exact character
spans. Source spans are stable across embedding and chunking changes, while chunk ids are not. A
retrieved chunk counts as evidence only when its document and character range overlap a gold span.

Corpus-derived model output is always a draft. Human verification checks grounding,
answerability, reference correctness, and planted-label integrity before `verified=true` is
written. Accepted ledgers are the only corpus-derived inputs eligible for headline scoring.

Synthetic benchmark data uses planted labels and a separate verification gate. The generating
model never certifies its own output.

## Corpus Conflict and Governance

A corpus is not automatically self-consistent, and the trust chain's first question -- is the
evaluation data correct and representative -- is unanswerable while two documents state
incompatible facts and nothing says which one holds. Two passages that contradict each other make
any gold item drawn from either one indefensible, and an operator who cannot say which edition
supersedes which cannot defend a retrieval result either.

The conflict audit therefore sits between ingestion and the retrieval gate. It reads an ingested
corpus and its store, and returns a ranked list of candidate conflicting passages, grouped into the
DECISIONS a reviewer would actually make rather than the rows a detector happened to emit. Where
documents carry governance metadata (`effective_date`, `version`), the audit orders the pair and
reports supersession; where they do not, it reports that the corpus cannot carry a dated
supersession at all, which is a corpus property an operator must learn at ingestion time rather
than after a store build.

The confidence contract is the load-bearing part of this capability:

- The semantic tier reports a **ranked candidate list, not a set of statistically significant
  findings**. No report, CLI string, or document may describe its cutoff as a false-positive rate,
  significance level, or confidence -- it is a candidate budget or a rank cutoff.
- Confidence in a conflict comes from the **claim tier's adjudication** against frozen labels, and
  a precision figure is publishable only with its clustered bound.
- No autonomous gate branches on the semantic tier's provisional verdict alone.
- Pursuing a per-pair semantic false-positive rate at this corpus scale is a CLOSED question, not a
  paused one; the four generations of evidence that closed it are recorded, along with what would
  reopen it.

A finished audit is an immutable bundle that answers its own questions offline: which stage lost a
pair, why a document was excluded, what a smaller candidate budget would have returned, and which
store it read plus that store's portable location when it is under `DATA_DIR`. Resolution is an
overlay with a rollback contract -- the audit proposes, a reviewer decides, and the corpus is never
silently rewritten.

## Entity Resolution and Record Linkage

Three of this project's record tables carry an IDENTITY that the pipeline currently settles by
exact string match, and each wrong call costs something downstream. A knowledge-graph entity node
keys on its normalized name, so "Ivan Franko", "Franko", and an inflected form become three nodes
that split one entity's mentions, its degree, and its community membership. A drafted gold item is
kept or dropped as a near-duplicate on one cosine threshold over its question alone. A re-ingested
document edition counts as a duplicate only when its content hashes match or its shingle overlap
clears a hand-set cutoff.

These are one problem -- deciding whether two records denote the same thing -- solved three times
by hand, each time on a single feature with a chosen constant. Probabilistic record linkage
combines several weak agreement signals into ONE match probability, estimates its parameters
without labels where none exist and from reviewer labels where they do, and resolves the surviving
pairwise links into identity clusters. That is what this capability supplies: a single linkage seam
that graph node identity, gold-item duplication, and document edition identity all address the same
way, so an operating threshold is READ OFF a precision/recall curve instead of picked.

The confidence contract is narrow, and it is what keeps this capability separable from the conflict
audit rather than a second entrance to it:

- Linkage answers **"are these two records the same thing"**, never "do these two records
  contradict each other". The semantic and claim tiers of the conflict audit are untouched by it,
  and no linkage output may be presented as a conflict verdict.
- A linkage model estimates its NON-MATCH parameters from randomly drawn record pairs. That
  construction is sound for identity -- two records drawn at random from a corpus are almost surely
  not the same record -- and it is precisely what four generations of research showed is
  unavailable for contradiction ([future research](../impl/future-research.md)). A calibrated
  linkage probability therefore does NOT reopen the closed per-pair semantic false-positive
  question, and no report, chart, or CLI string may describe it as though it did.
- A match probability is publishable together with the labelled set it was scored against. In a
  domain with no reviewer labels, linkage output is a ranked candidate list on exactly the same
  terms as the semantic conflict tier's.
- Linkage PROPOSES an identity cluster and never rewrites a corpus, a gold set, or a stored graph
  in place. A merge that changes what a later measurement is taken over is adopted on retrieval or
  review evidence, and the pre-merge artifact is retained so the reading can be redone without it.

The boundary: linkage needs records carrying several weakly correlated fields, and it is not
applied to a single free-text column -- a passage of prose with no other fields is outside what the
method can price. It is not a retriever and it does not rank chunks for a query. Chunk-level
near-duplicate collapse stays where it is, in the retrieval store's collapse tiers, and linkage does
not replace the exact and normalized hash tiers that already settle the cases they settle for free.

## Retrieval Before Generation

Embedding and retrieval quality are evaluated independently with recall at k and mean reciprocal
rank. This isolates the evidence-delivery ceiling from generation quality. If retrieval misses the
gold span, the case is classified as a retrieval miss; when evidence is present and the answer is
wrong, it is a generation miss.

Overlap-based recall is deliberately generous: it credits an item as soon as a retrieved chunk
shares ONE character with a gold span, so it cannot tell evidence that arrived whole from evidence
that arrived in pieces. The same source spans are therefore also read for INTACTNESS -- how much of
each gold span the retrieved context carries, and whether a single retrieved chunk carries a span
whole. That pair is the axis a chunk-boundary change moves while recall stays flat. The boundary:
intactness is REPORTED, never ranked on. It does not gate a leaderboard, does not decide an
adopt-or-retain verdict, and says nothing about whether an answer used the evidence -- that is the
answer-side measurement.

FAISS is the default vector path. Alternative stores share the same source-span metric, which makes
comparisons meaningful without changing gold labels. A retrieval-side component change is adopted on
a stated bar, not on a raw metric win: a rank-quality gain that no configuration can convert into a
delivered answer does not justify the swap.

The same bar governs a retrieval CONFIGURATION change, not only a component swap. Raising the
retrieval budget can lift coverage with no ranking change at all, so it is read the same way: the
identical items are scored end to end at the shipped budget and at the raised one, and the reading
states whether the extra evidence reached the answers, stopped at retrieval, or reached them while
lowering another question type's. Because a larger budget is not free, the served context size is
reported beside the coverage it bought. The boundary: this measures a budget, it does not choose
one. No default moves on the strength of the measurement alone, and the context size is reported
rather than gated -- what a context is worth is a deployment decision the evidence informs.

What the delivered context CARRIES into the prompt is a third thing the same bar governs, and it is
neither a component nor a budget. A chunk boundary can remove structure the text needs to be read at
all -- a table's column names sit in the header row, so a middle row block reaches the model as a
grid of unlabeled values -- and restoring that structure is a CONTEXT-ASSEMBLY step, applied when
the retrieved chunks are laid into the prompt. Because it changes only what the model reads, it
cannot move a retrieval metric, and the only reading that exists for it is the same retrieval row
scored end to end with the step off and on over one item set, reported per question-type slice
because the structure a step restores is what a particular kind of question needs. The boundary:
context assembly never rewrites a stored chunk, its offsets, or the source-span metrics read from
them -- restoration is prompt-side only, and its added context is reported as a cost beside the
answer delta. No such step is on by default before its measurement supports it.

A second kind of assembly step changes no text at all. It changes only how many BLOCKS the delivered
evidence arrives in, by merging retrieved pieces that were adjacent in the source back into one.
Intactness asks whether a SINGLE delivered block carries a gold span whole, so a step of that shape
is measurable on the retrieval side directly, without a generator -- and the property that makes the
reading trustworthy is the one that makes the step safe: merging pieces that touch changes neither
the set of retrieved characters nor their order, so recall and character coverage must reproduce the
un-merged lane EXACTLY and only intactness may move. That invariance is reported per lane rather
than assumed, because a lane that moved recall did not reflow evidence, it changed it. The boundary:
such a step is a REPORTED lever and never an adoption candidate -- it cannot move the metrics an
adopt-or-retain verdict is decided on, and a lane whose evidence was merely reflowed must not be
ranked above one that retrieved more. Its price is the served context size, reported beside every
lane on the same terms as a budget change.

Every end-to-end reading of that bar is conditioned on the GENERATOR that produced the answers,
because whether delivered evidence becomes a better answer is a property of the model as much as of
the retrieval lane. A reading taken with one model therefore cannot distinguish "this corpus and
lane do not convert" from "this tune does not convert", and the question-type slice a retrieval
change pays for itself on can differ between models even when the delivered context is identical.
So a retrieval change that is adopted, retained, or recommended on end-to-end evidence names the
model that evidence was taken with, and a recommendation whose whole purpose is to avoid a measured
per-slice cost is only established for the model whose cost was measured. The boundary: this
requires the reading to be ATTRIBUTED, not repeated -- the product does not gate a retrieval
decision on a roster-wide sweep, and a second model is evidence to seek when a per-slice cost is
what a decision rests on, not a precondition for taking the first reading.

## Graph Retrieval and Ontology

GraphRAG is a retrieval lane, scored against the same source-span metric as the vector lanes so the
two remain comparable. It uses the closed node vocabulary in
[graph-ontology-schema.md](graph-ontology-schema.md); the closed set keeps graph queries, node
typing, and relation caps stable across model runs, and model-invented types normalize to the
canonical vocabulary or `MISC`.

An ontology layer above the vocabulary may express axioms the corpus is expected to satisfy, and an
answer gate may refuse an answer that violates one. Because an axiom encodes a domain claim rather
than a measurement, an axiom is enabled only from a human-signed set: the checker proposes
candidates with their supporting and contradicting spans, and a reviewer accepts or rejects each.
The constraint set is serialized in a standard ontology format so that a reviewer outside this
codebase can read, diff, and version it.

At answer time the gate is read in BOTH directions or not at all. A validator's obvious failure
mode is refusing correct work, so what it stops (violations caught) and what it wrongly refuses
(refused answers the reference scores correct) are reported as separate numbers per axiom class,
the answered-item count and abstention rate sit beside the objective, and the objective delta is
read on the items the ungated lane also answered -- otherwise a gate that improves the mean by
declining the hard items reads as a win. An axiom class is enabled only where its measured catch
rate clears its measured false-rejection rate; a class that does not is recorded as measured and
not adopted rather than dropped. The gate's scope is the retrieved context: an answer may be
refused for contradicting a ledger fact the prompt carried, never for one the model was never
shown.

Over a corpus, the axiom layer REPORTS. It names every violated axiom with the exact evidence spans
of every fact the violation rests on, and it never deletes a fact, alters an extraction, or changes
a graph. A graph build may be refused over a violation only when the operator asks for that refusal
AND the axiom is signed; an unsigned candidate is reported and nothing more. The shipped checker
carries no reasoner: an OWL reasoner is a cross-check on the checker in the test suite, never a
component of the answer path.

## Backend and Hardware Boundary

The evaluator talks to an OpenAI-compatible chat interface. `BackendLauncher` implementations own
backend-specific startup, shutdown, health checks, and telemetry for Ollama, vLLM, and llama.cpp.
Evaluation and scoring code remain backend-neutral. New backend-specific behavior belongs in
launcher, resolver, planner, telemetry, or preflight modules and must not leak into scoring logic.

Before serving, the resolver combines model metadata, quantization, context length, GPU memory,
CPU offload, and backend availability into a host-fit plan. The actual served configuration is
recorded because a model name alone is not a reproducible runtime identity.

Availability includes the serving RUNTIME, not only the artifact: a runtime too old to implement an
artifact's architecture makes that source unservable while it is present and fits. Such a source is
reported as a named skip carrying the runtime, the architecture, and the version required -- at
resolution, at preparation, and at launch -- so a roster hole is a stated fact rather than a
per-case backend error. The floor is read from the artifact where the runtime declares one and
pinned per source otherwise; the product never guesses one.

One heavyweight model runs at a time. Sequential execution avoids VRAM contention, cross-run cache
effects, and biased telemetry.

## Model Roster and Family Currency

The candidate roster is a register of model FAMILIES, not a list of tags. A family carries one or
more GENERATIONS, and a generation carries the logical models and per-backend artifacts that serve
it. Exactly one generation of a family is `current`; a superseded generation is retained as
`previous` only while a carried model still serves from it, so a family result reads as a generation
comparison rather than a single point and an upgrade that costs quality is visible instead of
inferred. A generation with no carried model is dropped rather than kept: the roster answers what
runs now and what it replaced, not what has ever existed.

Because the register is the source of truth, the published family, generation, and license tables
are GENERATED from it. A roster restated in prose drifts the moment a generation lands, and a reader
cannot tell which of the two statements is current; a generated table cannot disagree with the
manifest without failing the check that regenerates it.

An upgrade is a decision, so the product supports one rather than performing it. Currency is read
from the upstream registries a family's artifacts already come from -- the Ollama library and the
Hugging Face model API -- and reported per family as the newest generation upstream offers beside
the one the roster carries, with what was read and when. The report never edits the roster, pulls
weights, or promotes a generation. Adopting one is an operator act, and it is reported together with
what it invalidates: every measurement taken against the generation being replaced. That list is
derived, not remembered -- each place the product records a model identity (the committed run
aggregates, the values published out of them, the baseline tables in the delivered docs) is resolved
back through the register to the generation it was measured on, so the re-measurement cost of a swap
is readable before the swap rather than after it. A recorded identity the register cannot place is
reported as such, because an undercounted cost and a clean one must not read alike.

The boundary: this capability owns family and generation identity, the published metadata derived
from it, and the currency report. It does NOT own serving decisions -- which artifact serves on
which host stays with the resolver and planner -- and it does not judge whether a newer generation
is BETTER, which is a measurement the sweep makes rather than a fact a register can carry.

## Scoring Policy

Objective task metrics are always available. RAG scoring includes exact/contains/token overlap,
semantic diagnostics, retrieval evidence, groundedness, citation validity, and abstention probes
when configured.

A single overlap score against the reference answer cannot say WHICH of a multi-evidence item's
facts an answer states, so the answer side reports coverage of the item's own gold spans beside it:
per labeled span, whether the answer carries the fact that span contributes, and the all-spans gate
over them. It is the answer-side counterpart of the retrieval-side span coverage, it is additive --
never a replacement for the ranking objective -- and it is a recall-side reading, so it is read
beside the format component that prices verbosity rather than alone.

Quality, throughput, VRAM, and power are retained as separate measurements. Recommendations may
combine them for a named operator goal, such as best accuracy or best quality per watt, but the raw
dimensions remain visible.

Answer-side signals should be read from a declared answer contract wherever the workload allows one,
rather than recovered from free text by heuristics after the fact. A status, an abstention, and a
citation that the model DECLARED are evidence; the same three recovered by regex are an estimate of
evidence, and the difference belongs in the record. A completion that does not satisfy the declared
contract ends in a typed status -- distinguishing "not the requested format at all" from "the
requested format, wrong shape" -- after at most one bounded repair, rather than being scored as a
wrong answer: an operator has to be able to tell a model that does not KNOW the answer from one
that cannot EMIT it.

A completion can be well-formed, on-topic, and still not be an answer. Two delivery failures are
therefore read per response and recorded beside the failure taxonomy rather than folded into
correctness: deliberation the model leaked into the answer body despite the serving backend's
thinking-suppression flag, and an answer delivered in a language other than the question's. Both
are scored as ordinary content otherwise -- a low overlap score that reads identically to a wrong
answer -- and the leak additionally inflates the generated-token count that throughput and cost are
derived from, so it corrupts a measurement the operator reads as hardware fact. The guard names
them; it never changes the case's status or the objective, because what it detects is a property of
the serving configuration, not of the model's knowledge. Suppression is therefore a per-model
verdict backed by a measured leak rate -- the backend flag, a prompt-level instruction on top of
it, or neither, in which case the tag is not scoreable as a non-thinking model and the roster, not
the scorer, is what has to change.

Context-ablation lanes (`closed_book`, `long_context`) are measurement lanes, never default
retrieval policies and never leaderboard rows. `long_context` is oracle-grounded -- it reads the
item's own gold document ids -- so its gap sizes what chunking still loses rather than what an
operator would gain.

An oracle ceiling is only actionable once it is split, so the ablation also carries a lane that is
NOT a diagnostic: `retrieved_document` retrieves exactly as the ranked lane does and then widens
the unit of context from the top-ranked chunk to the whole document that chunk came from, with no
gold label anywhere in the path. It divides the oracle gap into the part an operator captures by
changing a configuration value and the part that was the gold label all along, and because it is a
configuration someone could ship it carries an explicit adopt-or-reject verdict read off the same
calibrated paired interval as every other reading here. Adoption is a per-corpus measured result,
never a default: `rag` remains the leaderboard row until a run says otherwise.

Both readings the ablation produces are also stated per QUESTION TYPE, because a pooled average over
a mixed item set cannot say WHICH questions retrieval pays for: a factoid answered by a single span
and a multi-hop question whose evidence is scattered across documents are different retrieval
problems, and an operator whose corpus is mostly one of them is not served by the mean of both. Each
slice is decided on its own items by the same calibrated cut as the pooled reading and carries its
own item count, contamination rate, and per-lane skip counts, so a slice can be compared against the
pool it came from. The boundary: a slice reading is DIAGNOSTIC. It says where retrieval fails to pay
for itself and never becomes the corpus decision -- the pooled verdict is what the ablation
concludes, and the adopt-or-reject call on `retrieved_document` is not taken per slice at all,
because a shippable configuration chosen off a dozen items of one question type is what the
minimum-evidence gate exists to refuse. A gold set carrying no question-type sidecar reports no
slices rather than one pooled slice under a made-up label.

### Judge admission

An LLM judge is admitted only after its exact rubric and model clear the configured correlation
gate against human Ukrainian ratings. Below the gate, judge output remains diagnostic and cannot
change the headline. This prevents fluent but unsupported answers from outranking grounded ones.

The default judge is local, for no-egress reproducibility; the tradeoff is family bias when the
judge shares architecture, tokenizer, or pretraining lineage with candidate models. Manifests
disclose the judge model and its bias note, and boards reject incompatible judge cohorts. A frontier
judge is a separate opt-in lane requiring explicit egress consent and an enforced spend budget.

## Optimization Without Leakage

Configuration search uses only tuning data. Final data is held out until the selected configuration
is fixed. Sweep cells are isolated and recorded independently, so a failed or infeasible cell does
not contaminate another run.

Prompt-system and fine-tuning workflows obey the same split discipline. Registry and provenance
digests bind tuned artifacts to their source data and configuration.

Fine-tuning covers the RETRIEVER as well as the generator: the pinned encoder may be adapted to an
operator's own corpus from tuning-split gold alone, because a general multilingual encoder caps
recall on vocabulary it never saw and no roster choice recovers that. The boundary is that
adaptation buys nothing until it is measured. A tuned encoder enters the same bake-off as any
roster candidate and is adopted only on a held-out paired verdict against the encoder it would
replace; retaining the general encoder is a valid outcome, recorded rather than worked around. A
tuned artifact's identity is its provenance digest, not the path it happens to sit at, so the store
it embedded records that identity and refuses a query from any other encoder.

## Agentic and Context-Policy Workloads

Agentic execution is a benchmark tier with an extra dimension the single-turn tiers do not have: a
transcript that grows. What a harness does when the transcript no longer fits -- compact it into a
summary, or cap the observations it keeps -- changes both cost and task outcome, so the CONTEXT
POLICY is part of the configuration under test, not an implementation detail of the harness.

The maintained harness axis is `loop`, `langgraph`, and `crewai`. A harness is added only when it
changes a meaningful operational question and can share the same task set, world, scoring, and judge
gates; broadening a comparison table is not a reason.

Context-policy results are published as numbers an operator can act on -- where compact stops
repaying its summary call, which fold step a trigger selects, what a policy constant is pinned at.
A published number therefore carries its provenance back to the run artifact and field that produced
it, and a change to a pinned policy constant must name every published number it invalidates. A
number nobody can resolve back to a run is not evidence. A published number also carries the width
it is uncertain by whenever the run can measure one: a point estimate whose error is inside the
spread of the quantity it was estimated from is published as an interval, or refused as a count and
left as a ranking, never as a number an operator would read as exact.

Standing an agent up needs several of these answers at once -- a model, a prompt system, an adapter,
a context policy and order, retrieval knobs, a loop policy -- each measured by a different lane. The
product therefore COMPOSES them into one operating profile, under the same provenance rule applied
per field: a field carries the artifact that measured it, that lane's own verdict and uncertainty,
and its freshness. Composition adds exactly two obligations of its own. A field whose lane never ran
is reported as absent, never as the value the code would have defaulted to; and fields measured
against different corpora, stores, or models are refused rather than mixed, because a profile whose
parts were never measured together describes a configuration nobody ran. The boundary: composition
runs no lane on the operator's behalf, invents no value for a field nobody measured, decides no
ranking policy of its own, and ships no runtime that consumes the profile.

## Autonomous Orchestration

The corpus-to-recommendation path can run end to end without a human at each step: ingest, draft,
build, validate retrieval, run, score, recommend. Autonomy is bounded by the same gates as the
manual path -- it resumes rather than restarts, it verifies what it produced, and it may not
promote anything past a human gate that the manual path also refuses to promote.

The open question this capability owns is what autonomous operation COSTS in result quality against
the assisted path, measured rather than assumed.

## Robotics RAG and Hardware Operation

A text-answer benchmark cannot tell an operator whether the same model can carry out a physical
workflow. Robotics adds four facts that ordinary RAG does not have to reconcile: multimodal history,
live device state, side-effecting operations, and safety authority that must remain outside the
model. Without an explicit seam between them, a stale or injected retrieval result can become a
hardware command, while a successful command can disappear without enough evidence to reproduce or
evaluate it.

Two upstream projects supply complementary parts of that seam:

- [HFlow](https://github.com/Hebbian-Robotics/hflow) is the OFFLINE episode and data-quality plane.
  Its pre-v1 lifecycle starts from landed MCAP episodes, runs coarse transform, quality-check, and
  enrichment stages locally or as Airflow DAGs, and exposes canonical MCAP, provenance, a Parquet
  catalog, and a DuckDB-curated manifest. Its own
  [architecture](https://github.com/Hebbian-Robotics/hflow/blob/main/docs/ARCHITECTURE.md) stops
  before training and does not operate a robot.
- Anthropic's [Model Hardware Standard research
  preview](https://www.anthropic.com/news/model-hardware-standard-research-preview) is the LIVE
  device plane: discoverable drivers expose simple read/write primitives, a generated device
  reference, and driver-enforced limits through MCP, a CLI, or code APIs. MHS is model-agnostic, but
  it is still a [limited, application-only preview](https://www.modelhardwarestandard.com/) rather
  than a public normative specification. loc-lm-bench therefore treats it as an adapter target,
  never claims compatibility from the announcement alone, and makes no base-install dependency on
  preview code. A real adapter is named MHS-compatible only after it passes fixtures pinned to an
  inspectable preview or public contract and license.

The projects are not substitutes. HFlow prepares evidence after a recording lands; MHS discovers,
reads, and operates equipment while a workflow is live. Airflow never enters the control loop, and
an MHS reference file never substitutes for episode provenance or curation.

### The robotics evidence-to-operation flow

```text
approved manuals + MHS reference text       landed MCAP episodes
                  |                                  |
                  v                                  v
        existing corpus ingest       HFlow transform -> QC -> enrich -> catalog
                  |                                  |
                  +------ accepted text projections + temporal references
                                                     |
                                                     v
                                             existing RAG store
                                                     |
goal + current device discovery/read snapshot -> retrieve -> typed proposal
                                                      (no side effect)
                                                               |
                                                      external action gate
                                                               |
                                              deny / escalate / fresh re-read
                                                               |
                                                    device adapter write
                                                               |
                                             receipt + telemetry -> recorder
                                                               |
                                                 MCAP -> HFlow -> run bundle
```

The offline bridge consumes HFlow's STANDARD boundaries rather than its scheduler internals. A
curated manifest identifies eligible episodes and pins the curation query, schema, pipeline, check,
and enrichment versions. Each text projection entering the existing corpus carries its HFlow
episode id, canonical MCAP URI and digest, channel names, half-open timestamp interval, source
artifact version, and the projection's character offsets. Existing source-span scoring applies to
the text projection; the temporal reference proves which underlying observation supports it.
Model-authored captions and summaries remain drafts and pass the existing human verification gate
before headline use. Quality measurements and quarantine tags are preserved as evidence; this
bridge neither invents a threshold nor deletes an excluded episode.

The first robotics tier retrieves text projections and optionally references pre-extracted frames;
it does not put raw MCAP bytes in a prompt. Native video control, vision-language-action training,
and learning a policy from the curated manifest are separate capabilities and remain out of scope.

The live bridge splits MHS-derived material by authority. Natural-language descriptions, manuals,
and prior-case summaries are retrieval inputs and are UNTRUSTED even when they came from a driver
tag. The discovered device identity, operation schema, hard limits, and state revision form a
separately digested capability snapshot used by the executor and never reconstructed from retrieved
text. Retrieval can explain or support an action; it cannot add a tool, widen a limit, select a
credential, or grant write authority.

The model emits a typed PROPOSAL, not a tool call with immediate effects. It names the device and
operation, typed arguments, evidence references, expected state revision, preconditions,
postconditions, risk class, and idempotency semantics. A deterministic action gate outside the
model then checks all of the following before one adapter invocation:

- the device and operation occur in the current discovered snapshot and the deployment allowlist;
- arguments satisfy both deployment policy and the driver's independently enforced hard limits;
- a fresh read still matches the proposal's identity, state revision, and preconditions;
- the requested risk class has the required operator approval and the approval binds this exact
  proposal digest; and
- device locks and declared dependencies make the action serializable with every concurrent action.

Only operations classified as read-only can run without write authority. A denial or escalation has
no device side effect. A timeout after a non-idempotent write is `outcome_unknown`: the executor
reads state or escalates and never retries merely because no receipt arrived. Emergency stops,
interlocks, and driver limits always take precedence below the model and below this gate. An agent
may propose a deterministic procedure file for a long or latency-sensitive sequence, as the MHS
preview describes, but the file is an inert versioned artifact until static validation,
emulator/shadow replay, policy review, and explicit approval succeed; arbitrary model-written code
is never executed as the hardware adapter.

Every proposal, decision, approval, adapter request, observation, error, and receipt enters the run
bundle. A recorder can also place synchronized device state and actions into an MCAP episode for a
later HFlow pass; HFlow is not called synchronously from the control loop. The bundle pins the agent
operating profile, corpus and store fingerprints, HFlow manifest and episode provenance, device
snapshot and driver-contract digests, action policy, approval identity, and the adapter version.
Large media stays in the data plane and is referenced rather than copied into the bundle.

### Evaluation and adoption

Robotics is a separate benchmark tier; its metrics never blend into a text-answer board row. The
capability is evaluated in three gates, in this order:

1. **Boundary conformance.** A pinned HFlow-produced fixture must round-trip every accepted text
   span to the exact MCAP episode, channels, time interval, and producer versions, while preserving
   curation and quality state. A protocol-neutral fake driver must exercise discovery, read, write,
   denial, error, ambiguous outcome, and multi-device locking. When an inspectable MHS contract is
   available, the real adapter must pass the same suite plus its upstream conformance cases; until
   then the report says `protocol-neutral`, not `MHS-compatible`.
2. **Held-out emulator scenarios.** The same pinned local model, prompt system, retrieval store,
   context policy, and action policy run with robotics RAG enabled and with retrieval withheld; a
   deterministic reference controller supplies a non-model baseline where the task has one. The
   ledger plants normal workflows, stale and wrong-device snapshots, limit violations, missing
   approvals, prompt injection in retrieved evidence, unreachable or busy devices, emergency-stop
   state, conflicting multi-device actions, recoverable errors, and ambiguous writes. It reports
   retrieval coverage, evidence-grounded proposal rate, task completion, appropriate refusal and
   escalation, unsafe-proposal rate, blocked-action reasons, recovery success, action count,
   latency, tokens, and power. The comparison predeclares its minimum detectable task-completion
   gain and minimum evidence count.
3. **Supervised device canary.** Only a hardware owner can admit this gate. It starts with
   discovery/read-only and shadow decisions, then permits an explicitly bounded low-risk write with
   working interlocks and an operator-controlled stop. The canary records every mismatch between
   emulator and device behavior and does not authorize unattended operation.

The mandatory safety gate is zero EXECUTED out-of-policy actions and every planted stale-state,
wrong-device, limit, approval, injection, emergency-stop, concurrency, and ambiguous-retry case
blocked before a forbidden adapter invocation. Coverage and denominators accompany that zero: it is
a finite-suite result, not a safety proof. RAG is adopted for this tier only when its paired interval
clears the predeclared task-completion or appropriate-refusal gain, its unsafe-proposal rate does not
regress, and every mandatory gate still holds. If retrieval buys no operational gain, if the MHS
contract cannot be inspected, or if any safety gate regresses, the negative result is recorded and
the protocol-neutral/read-only or non-RAG baseline is retained rather than worked around.

The boundary: loc-lm-bench owns evidence projection, an emulator and adapter contract, the external
action gate, scenario evaluation, and reproducible operation bundles. It does NOT author hardware
drivers, replace ROS or a device controller, run hard-real-time feedback, train a VLA policy,
certify physical safety, or make an emulator result authority for production deployment. Private
robot data and device descriptions inherit the existing local-first egress rules; an external model
cannot receive either without the same explicit consent and budget controls as other frontier calls.

## Operator Review Tooling

Every human gate in the trust chain needs a place to stand: verification of drafted gold items,
adjudication of conflict decision groups, acceptance of a multi-hop ledger, sign-off on an axiom
set. These share one terminal review workbench with per-domain adapters, so a reviewer learns one
set of keys and every decision is recorded the same way in a ledger.

Reviewer throughput is itself a measurement. A capability that needs N human decisions per corpus is
a different product from one that needs N/10, so review cost is reported rather than assumed, and a
grouping that reduces decisions without hiding evidence is a product improvement.

## Persistence and Reproducibility

The filesystem run bundle is the source of truth. A finalized run records, as applicable:

- resolved model, backend, context, quantization, and adapter identity;
- corpus, gold-set, prompt, and configuration digests;
- per-case scores and retrieval evidence;
- aggregate metrics and reliability;
- hardware and runtime telemetry;
- reports and analysis artifacts.

Artifacts are staged and finalized atomically under `$DATA_DIR/<method>/<run_timestamp>/`. MLflow
mirrors canonical artifacts for comparison and visualization; it is not the primary store.

Boards reject incomplete, unverified, mixed-tier, or non-final records instead of guessing how to
interpret them.

A run artifact is sized by what it must answer, and its growth is bounded by the corpus or by a
recorded run parameter rather than by whatever a stage happened to emit. Beyond that bound, artifact
size is not a subject the plan tracks -- see
[Specification and plan integrity](#specification-and-plan-integrity).

### Reproducible environment

A recommendation is only reproducible if the environment that produced it is. Dependency versions
are locked, a source build records the ABI dimensions and revision it was built from, and a fresh
environment build reaches a green check suite with no manual repair step. A drift discovered as a
type error in application code is a dependency-resolution failure reported in the wrong place.

## Reuse and Dependency Policy

The implementation favors maintained Python-native components and small project-owned seams:

- Typer for the CLI;
- Pydantic and typed dictionaries for contracts;
- FAISS plus optional GraphRAG/vector-store backends for retrieval;
- Splink over the DuckDB backend already required by the graph store, for probabilistic
  record linkage;
- Optuna for bounded tuning;
- MLflow for experiment analysis;
- DeepEval for calibrated judge execution;
- NVML, `nvidia-smi`, and process telemetry for runtime evidence.

Heavy or backend-specific dependencies stay behind optional extras and lazy imports. A base install
can inspect data, plans, and artifacts without importing GPU stacks.

## Data Egress Boundary

Default corpus processing is local. Frontier or Litellm calls are opt-in tools, not the default
path for private material: real chat-log corpora use local drafting or verification only, real
text-analysis corpora may use a frontier cross-check when the operator explicitly approves it, and
every drafted bundle still needs human verification before headline scoring. Frontier scoring is a
separate opt-in with one upfront consent plus a hard per-run budget enforced by a cost ledger;
over-cap aborts are resumable and never silent.

loc-lm-bench measures model security behavior; it is a benchmark, not a production RAG service.
Runtime guardrails -- prompt-injection filtering of retrieved content, output PII/secret filters,
identity-backed authorization -- belong to the application embedding a recommended model. The
benchmark-side governance layer is limited to metadata tags, ACL-scoped retrieval, deletion
propagation, stale-store refusal, and immutable store-directory rollback.

## Capability Registry

Every product capability appears here exactly once. This table is the join between what the product
DOES (this spec), how we know it works (the evaluation), and where it is implemented (the current
docs). `make lint-spec-plan` checks the join in both directions, so a capability cannot exist
without an evaluation and a plan task cannot exist without a capability.

Status is either `shipped` (implemented and documented; may still have open extension work) or
`planned` (specified, not yet implemented; must have at least one open plan task).

**Row order is the implementation line.** [`../impl/plan.md`](../impl/plan.md) groups its tasks by
capability in exactly this order, so moving a row re-prioritizes the whole product -- one decision
recorded in one place rather than an argument re-had per task.

The order is not simply the trust chain. The chain says which capability's EVIDENCE rests on which,
and it is the tiebreaker; the line says what to BUILD next, which is a question about remaining work.
Four rules settle it, in order:

1. **A hard dependency wins.** A capability holding a task another capability's task names as a
   prerequisite comes first.
2. **Work that changes another capability's inputs comes first.** A change to chunking, to the
   encoder roster, or to the gold set invalidates every measurement taken against the old one, so it
   lands before the measurements. This is where the trust chain does its work.
3. **Required work outranks optional work.** A capability whose remaining tasks are all `(optional)`
   sits below one with required tasks, however far upstream it is in the chain -- a shipped
   capability with only refinements left is not what blocks the product.
4. **Then the trust chain,** upstream before downstream.

| # | Capability | Status | How it is evaluated | Implementation |
| --- | --- | --- | --- | --- |
| 1 | `reproducible-environment` | shipped | A fresh environment build reaches a green check suite with no manual repair step | [Overview](../impl/current/overview.md) |
| 2 | `gold-data` | shipped | Split validation on the committed fixture; human verification gate with experiment-derived acceptance thresholds; multi-annotator adjudication | [Data prep](../impl/current/data-prep.md) |
| 3 | `entity-resolution` | planned | Paired graph-lane recall at k and MRR over the same source spans before and after node clustering; linkage precision/recall against a reviewer-labelled merge set, with the operating threshold read off that labelled accuracy curve; a threshold that lifts no lane metric is recorded as a negative result rather than adopted | [Entity resolution](../impl/current/entity-resolution.md) (the linkage seam, the gold-item shadow lane, and the graph node lane; the remaining identity decisions are in [the plan](../impl/plan.md)) |
| 4 | `retrieval-evidence` | shipped | Recall at k and MRR against source spans, with span character coverage, intactness, and served context size reported beside them; paired verdicts with a predeclared MDE and a minimum-evidence gate; an adoption bar for a component swap | [RAG core](../impl/current/rag-core.md) |
| 5 | `answer-scoring` | shipped | Objective metric decomposition (token precision/recall/found-rate) with a declared format weight; answer-side coverage of the item's gold spans reported beside the objective; leaked-reasoning and off-language delivery failures flagged per response and rated per run beside reliability; miss classification into retrieval, generation, refusal, artifact, judge; per-model answer-contract conformance, with its shape-failure split and repair rate reported apart from correctness | [Scoring](../impl/current/rag-core/scoring.md) |
| 6 | `judge-calibration` | shipped | Correlation gate against human Ukrainian ratings before a judge may rank; demotion to diagnostic below the gate | [Judging](../impl/current/rigor-board-judge/judging.md) |
| 7 | `graph-retrieval` | shipped | Same source-span metric as the vector lanes, graph-vs-vector paired comparison; closed-vocabulary normalization rate; per-class axiom-violation base rate over an extraction ledger, cross-checked against an OWL reasoner, where an axiom class with no population or no violation on a corpus is recorded as buying nothing there rather than as a pass; at answer time, per-axiom-class catch and false-rejection rates reported as separate numbers over planted violations and adversarial correct answers, with the objective delta read on the commonly-answered items and an unsigned axiom set refused | [GraphRAG](../impl/current/graphrag-backend.md) |
| 8 | `host-fit-serving` | shipped | Host acceptance checklist and repeatable smoke runs per backend; the recorded served configuration replayed | [Host validation](../impl/current/host-validation.md) |
| 9 | `model-roster-currency` | shipped | Every carried model resolves to a registered family generation, with exactly one `current` generation per family; the published family, generation, and license tables regenerate from the roster manifest and a drift fails the docs check; the upstream currency report reproduces both a newer-generation finding and a no-newer-generation outcome from recorded registry responses, and reports rather than edits; a proposed generation swap lists every committed run aggregate, published value, and delivered baseline row whose recorded model resolves to the outgoing generation, lists none measured on another family, and states plainly when nothing is affected | [Model roster](../impl/current/model-roster.md) |
| 10 | `optimization-search` | shipped | Tuning/final split discipline enforced per sweep cell; provenance digests binding a tuned artifact to its source data; a locally tuned retriever adopted only on a held-out paired verdict against the encoder it would replace, with its training data refused if it names a calibration or final item and its store recording the tuned identity so no other encoder can query it | [Evaluation rigor](../impl/current/rigor-board-judge.md), [embedder fine-tuning](../impl/current/extended-workflows/embedder-finetune.md) |
| 11 | `run-bundle-board` | shipped | Board admission refusal on incomplete, unverified, mixed-tier, or non-final records; a recommendation reproduced from the saved manifest | [Evaluation rigor](../impl/current/rigor-board-judge.md) |
| 12 | `agentic-workloads` | shipped | Prompt-sequence replay of context policies at fixed seeds; published-number provenance resolved back to run artifacts; a CI gate pinning policy constants | [Extended workflows](../impl/current/extended-workflows.md) |
| 13 | `autonomous-orchestration` | shipped | Resume-from-interrupt verification and post-run self-verification on the quickstart corpora | [Auto-RAG](../impl/current/auto-rag.md) |
| 14 | `robotics-rag-operation` | shipped | HFlow evidence references round-trip to pinned MCAP intervals and producer versions; protocol-neutral and, when inspectable, MHS adapter conformance; paired RAG-vs-no-retrieval completion and appropriate-refusal verdicts on a held-out emulator ledger; zero executed out-of-policy actions and every planted stale-state, wrong-device, limit, approval, injection, emergency-stop, concurrency, and ambiguous-retry violation blocked before forbidden invocation; a negative result retains the read-only or non-RAG baseline | [Robotics RAG](../impl/current/robotics-rag.md) |
| 15 | `corpus-conflict-audit` | shipped | Claim-tier precision against frozen adjudicator labels with a clustered lower bound; stage attribution and budget replay recomputed from a bundle alone; overlay rollback contract | [Conflict detection](../impl/current/data-prep/conflict-detection.md) |
| 16 | `operator-review-tooling` | shipped | Ledger compatibility across adapters; measured reviewer throughput per decision domain | [Review workbench](../impl/current/review-workbench.md) |
| 17 | `category-suites` | shipped | Per-tier task and data contracts kept separate; no blended board row | [Category suite](../impl/current/category-benchmark-suite.md) |
| 18 | `documentation-integrity` | shipped | `make lint-md` (style plus every relative link and anchor landing) and `make lint-spec-plan` (this registry against the plan) | [Overview](../impl/current/overview.md) |

## Extending This Specification

A capability the product needs but this file does not describe is a GAP, and the response is to
close the gap, not to refuse the work. Implementation is the main way gaps are found: a difficulty
hit while building, a corpus property nobody anticipated, or a result that only makes sense with a
capability that does not exist yet are all legitimate discoveries.

The lifecycle for one is fixed, and its steps are in this order for a reason:

1. **State the problem in domain terms.** What can an operator not do, or not trust, today? A
   problem stated as "our artifact is larger than it needs to be" is not a domain problem; a problem
   stated as "an operator cannot tell which of two contradicting documents holds" is.
2. **Amend this specification.** Add or extend the section that owns the capability, including its
   boundary -- what it explicitly does NOT do. A capability with no stated boundary grows without
   one.
3. **Declare the evaluation before the implementation.** Name how anyone will know the capability
   works: the measurement, its gate, and what a negative result would look like. A capability whose
   evaluation cannot be stated is not ready to be built, and a negative result is a valid outcome to
   be recorded rather than worked around.
4. **Register it.** Add the row to the [capability registry](#capability-registry) with status
   `planned` and the evaluation from step 3.
5. **Place it in the implementation line.** Add plan tasks under the capability, positioned by the
   trust chain rather than by when the idea arrived.
6. **Implement, then close the loop.** Record delivered behavior in the current docs, remove the
   task from the plan, and flip the registry row to `shipped` with the implementation link.

Steps 2-4 are the cost that keeps this cheap to do and expensive to do carelessly. They are also
what makes a discovered capability durable: a feature that entered the plan without them is a
feature nobody can evaluate, document, or decide to remove later.

## Specification and Plan Integrity

Two documents must agree at all times: this specification, and
[`../impl/plan.md`](../impl/plan.md). The invariants below are checked by `make lint-spec-plan` so
that agreement is a build failure rather than a review opinion.

- **Every plan task serves a registered capability.** A task carries a `Serves` line naming a
  capability id from the registry. Work that serves none of them has no home in the plan until a
  spec amendment gives it one.
- **Every registered capability is real.** A `shipped` capability links to its implementation docs.
  A `planned` capability has at least one open plan task; a `planned` row with no task is a
  capability nobody is building and is removed or reopened deliberately.
- **Every capability declares its evaluation.** The registry's evaluation column is how the
  capability is known to work, and it may not be empty.
- **The plan is ordered by the trust chain.** Tasks appear grouped by capability, and the capability
  groups appear in the declared implementation-line order. "What is next" is then a position, not a
  judgment call about which entry is best written.

Some things are deliberately NOT tracked as forward work, because tracking them creates more of
them than it resolves:

- **Housekeeping is a chore, not a task.** Splitting a source file over the soft line limit and
  shrinking a run artifact already inside its size bound are done inline while editing the file, or
  not at all. Written up as tasks with acceptance gates, they compete with the product on equal
  footing and each round produces the next round.
- **Our own output is not a domain.** An artifact's field layout and byte count are not sources of
  capability. The measure of an audit is a decision it changes.

## Success Criteria

The system succeeds when an operator can:

- create or ingest a representative Ukrainian gold set and verify it;
- learn where their corpus contradicts itself, and which edition supersedes which;
- learn which entity nodes, gold items, and document editions denote the same thing, and at
  what threshold that was decided;
- prove the retriever exposes the labeled evidence;
- identify runnable model/backend configurations for the host;
- execute comparable final-split runs without manual artifact repair;
- explain misses as retrieval, generation, refusal, artifact, or judge disagreement;
- choose a model from recorded quality and resource evidence;
- reproduce the decision from the saved inputs and manifest.

Current implementation detail is indexed in [../impl/current.md](../impl/current.md). Operator
commands and quality gates are indexed in [../guides/README.md](../guides/README.md). Forward work
belongs only in [../impl/plan.md](../impl/plan.md).
