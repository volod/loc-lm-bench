# loc-lm-bench -- Implementation Plan (forward work)

Forward-only: every line in this file describes work that remains. Current behavior, operator
workflows, durable evidence, and design decisions live in [`current.md`](current.md) and the topic
files under [`current/`](current/). A task that leaves this file keeps its original contract and
acceptance evidence in its [task record](records/README.md); a research question that closed
negative keeps its reopening conditions in [future research](future-research.md). The product spec
lives in [`docs/design/spec.md`](../design/spec.md).

Every task serves a capability registered in
[the spec](../design/spec.md#capability-registry), and the capability groups below appear in the
registry's row order -- the implementation line. That is the whole scheduling rule: **take the first
task of the earliest group that still has one, skipping a task whose `Requires` names a task still
in this file.** Which entry is best written, most recently added, or most sharply specified does not
enter into it. `make plan-status` applies the rule and names each lane's priority task; read that
task's `Dependencies` before starting it.

Lanes, statuses, task fields, `Requires` versus `Dependencies`, ordering inside a group, and the
completion transition are defined once, in the
[planning workflow](../guides/development/planning-workflow.md). Read it before adding a task.

Work that no registered capability covers is not blocked -- it is a signal the spec has a gap. Close
the gap first: [Extending this specification](../design/spec.md#extending-this-specification) is the
six-step lifecycle for turning a discovery into a registered capability with a declared evaluation,
and only then into tasks here. `make lint-spec-plan` enforces the join in both directions, the
task shape, and the forward-only wording of this file.

For remaining tasks that depend on retrieval behavior, use the current RAG baseline documented in
[RAG core](current/rag-core.md) and the mixed-corpus ingestion baseline documented in
[data prep](current/data-prep.md). For tasks that depend on scoring or judging, the calibrated
local-judge baseline and tuning/sweep behavior live in
[evaluation rigor](current/rigor-board-judge.md); the prompt-system package flow and other
extended workflows live in [extended workflows](current/extended-workflows.md).

## Pre-release rule: no versioning, no backward compatibility

Nothing this project writes has been released, so nothing on disk is a compatibility obligation.
Until this file holds no task, a task must NOT introduce a second schema version, a
migration edge, a compatibility shim, a re-export alias, a deprecation window, or a "read the old
form too" branch. When a durable form has to change, change it in place: update the model,
regenerate the exports, update the fixtures, and rebuild whatever no longer parses. A run
artifact an earlier build of our own code wrote is reproducible input, not a supported form.

The standing exceptions are already declared and are not a licence to add more: the
compatibility-probe family that keeps the migration mechanism exercised in CI, and the families
whose older versions describe bundles this project actually wrote -- the data-prep, store-meta,
and run-manifest families a registered migration already carries forward. See
[compatibility behavior](current/artifact-contracts/foundation-and-evolution.md#compatibility-behavior).

Backward compatibility and schema migration become real requirements at release, once this file is
empty and the code is published; the mechanism is kept working for that day. A task written before
that point does not get to spend it.

## Agent Implementation Tasks

Work an agent can take to acceptance on its own: `make ci` green with committed fixtures, injected
fakes, and deterministic harnesses (`CLEAR`), or that plus a heavy deterministic run the agent
executes on the CUDA host (`RUN NEEDED`). Grouped by capability in registry order.

### Scanned book recovery -- `scanned-book-recovery`

Complete the recovery implementation, synthetic gates, and agent-run qualification first.
Extract the library after this required line of agent work. The pilot and target run are
provisional until the [human release gates](#human-assisted-tasks) supply the reference and
acceptances. The optional closing audit follows the required gates.

#### book-retrieval-reading

Prepare the paired retrieval comparison before operator review so an accepted book item can
exercise both corpora through one replayable command.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- replay over a planted corpus and synthetic gold.
- Dependencies: [RAG core](current/rag-core.md) and
  [paired verdicts](current/rag-core/paired-verdicts.md).
- User-visible outcome: a report of recall at k and MRR against source spans and a paired comparison
  against the existing PDF lane, joined on the originating PDF page.
- Scope boundary: retain the baseline OCR language and the same index and retrieval settings
  on both corpora. Report separately the cases whose baseline page text contains the answer span.
  Keep retrieval settings fixed and leave answer scoring to its own evaluation.
- Data and artifact paths: fixture corpora and items under `tests/`; corpus and index runs under
  `$DATA_DIR`.
- Execution path: reuse `make pdf-to-markdown`, `make build-index`, and the retrieval evaluation
  targets over synthetic inputs.
- Acceptance gates: `make ci` green; refuse stale or unverified gold; pin the span and page
  readings, the item count and minimum detectable effect, the baseline-text split, and the
  inconclusive and negative verdict paths. Real book evidence follows reference and gold review.
- Documentation target: `docs/impl/current/book-recovery/corpus-projection.md`.

#### book-reference-graph-reading

A technical handbook's hardest questions join a sentence to the table, equation, or figure it cites.
The vector lane retrieves the sentence without the table: a numeric grid embeds poorly and is its
own projected document. The book already prints the edge between them, and the projection's
reference sidecar records it. Build a graph from those printed references and prepare the paired reading
against the vector lane over the same projection
([method survey](../design/graph-construction-methods.md#the-printed-reference-graph)).

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- deterministic graph, provenance, and paired comparison fixtures.
- Requires: `book-retrieval-reading`
- Dependencies: the printed references each projected document records
  ([corpus projection](current/book-recovery/corpus-projection.md#printed-references));
  the graph store, its strategies, and the fusion seam
  ([modules and CLI](current/graphrag-backend/modules-and-cli.md));
  the minimum-evidence gate on a paired reading
  ([paired verdicts](current/rag-core/paired-verdicts.md)). The derived-source contract is
  [graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology).
- User-visible outcome: `llb build-graph` over a projected book builds a graph where
  every node mention and edge names a span the book prints. A replayable comparison measures
  vector, graph, and fused retrieval on the same verified items.
- Scope boundary: in scope -- section, figure, table, equation, page, and index-term nodes and the
  `contains`, `cites`, and `indexes` relations as a closed set beside the entity vocabulary, never
  normalized into it; index terms linked to the question by the existing lexical linker; no edge
  built from a citation the sidecar records as unresolved; `local_khop` over the graph, since a
  cited block is one `cites` hop away, alone and fused with the vector lane through the existing
  fusion seam; `ppr` rows beside it when `graph-seeded-propagation-ranking` has left the plan by
  the time of the run, with their absence stated otherwise; the reading on the verified book items
  under the same retrieval configuration as `book-retrieval-reading`, split by whether an item's
  spans include a table, equation, or figure block. Out of scope -- any edge the book does not
  print, blending the structural graph and an extracted graph into one store, retuning retrieval
  for the book, and answer scoring.
- Data and artifact paths: a book-reference builder beside `src/llb/graph/build.py`; the graph
  store, the index, and the comparison under `$DATA_DIR`.
- Execution path: the synthetic book inside `make ci`, where a sentence citing a table
  reaches the table block in one `cites` hop. Exercise the same inputs through
  `make build-graph`, `make build-index`, and `make compare-graph-fusion` with `NOISE_FLOOR=1`.
- Acceptance gates: `make ci` green; every mention and edge resolves to its printed evidence
  offsets; two builds are byte-identical; an unresolved citation supplies no edge. Pin the paired
  report, minimum detectable effect, intervals, and technical-evidence slices, including a null
  result. Run the real-book comparison only with the release gold set.
- Documentation target: `docs/impl/current/book-recovery/corpus-projection.md` beside the book's
  vector reading, and the structural vocabulary in
  [graph ontology schema](../design/graph-ontology-schema.md) outside the signed entity vocabulary.

#### book-full-run-on-the-target-scan

Qualify the whole pipeline on the 1,067-page target scan with a provisional recovery profile.
Interrupt it, resume it, and stage the vault and projection with a report of what needs review.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: RUN NEEDED -- a multi-day sequential GPU run on the CUDA host.
- Dependencies: the target scan under `DATA_DIR`, its operator-declared redistribution
  class, and a host with enough VRAM for the provisional configuration.
  See the [publication audit](current/book-recovery/full-run-evidence.md#publication-and-resume-audit).
- User-visible outcome: a mechanically validated vault and ingested corpus projection
  for all 1,067 pages, with limitations and a named provisional profile.
- Scope boundary: qualify the whole target run, including chapter-sized batches,
  deliberate interruption and resume, vault publication and ingestion. Use fixture and diagnostic
  checks to pick the provisional profile. Reference accuracy, profile adoption, and operator
  review belong to the release gates.
- Data and artifact paths: `$DATA_DIR/book-ocr/<run_timestamp>/`; the operator's vault path; the
  projected corpus directory.
- Execution path: `make book-ocr` over the full page range under the GPU isolation contract of
  [heavy runs and evidence](../guides/development/heavy-runs-and-evidence.md).
- Acceptance gates: `make ci` green; `book-ocr validate` green with all 1,067 pages
  accounted for; the resume repeats no completed stage; a resumed render matches a clean render
  from the same cache; ingested chunks have scan provenance; report wall time against the pilot
  estimate and queue sizes per class and chapter. Mark the report as provisional.
- Documentation target: `docs/impl/current/book-recovery/full-run-evidence.md` and the usage guide
  `docs/guides/data-prep/book-ocr.md`.

#### book-recovery-library-extraction

A RAG project needs the scan-to-corpus workflow as an installable recovery tool. Extract the
qualified recovery implementation and commands before requesting operator acceptance.

- Serves: `scanned-book-recovery` -- [Reusable library and release gates](../design/spec.md#reusable-library-and-release-gates)
- Agent status: CLEAR -- build and exercise the wheel in an isolated environment.
- Requires: `book-full-run-on-the-target-scan`, `book-reference-graph-reading`
- Dependencies: the [run and resume contract](current/book-recovery/pipeline-and-resume.md),
  [review and uncertainty](current/book-recovery/uncertainty-and-review.md), and the
  [projection boundary](current/book-recovery/corpus-projection.md).
- User-visible outcome: a separately installable Python library and CLI for recovery, pilot
  and release review, validation, vault export, and corpus projection. Existing
  `llb book-ocr` commands delegate to the library. A RAG consumer needs only exported contracts
  and the public interface.
- Scope boundary: move recovery modules, contracts, and command assets behind the library
  interface. Keep OCR backends optional, caller paths explicit, and model downloads out of
  installation. Keep RAG ingestion, graph and vector indexing, and benchmark evaluation in their
  consuming project. Build locally before release and keep copyrighted scans outside the wheel.
- Data and artifact paths: a standalone package source tree and its wheel, synthetic integration
  fixtures, and release-candidate book outputs under `DATA_DIR`.
- Execution path: use a clean `uv` environment with the recovery wheel, exercise its API and CLI
  from outside the benchmark checkout, then repeat the fixture through `llb book-ocr`.
- Acceptance gates: `make ci` green; the wheel installs and runs without `llb` or benchmark
  configuration; an independent fixture consumer recovers, resumes, validates and imports
  the projected corpus; both command paths preserve artifact ids, page provenance, uncertainty,
  licence enforcement, and idempotence. An import from the benchmark into the library or an
  unexplained output difference fails the gate.
- Documentation target: `docs/impl/current/book-recovery/pipeline-and-resume.md` and the usage guide
  `docs/guides/data-prep/book-ocr.md`.

#### scanned-book-recovery-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. The tasks in both lanes each prove their own stage, and the synthetic book rehearses
the chain once; what none of them can see is the whole against its spec -- a property two
tasks each assumed the other owned, a stage that resolves a value from the page without returning
it from `effective_params` (the class of bug the run ledger already hit once), a seam whose
fixture agreed with itself but not with what the full run, the Obsidian acceptance, and the
retrieval reading found.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: CLEAR -- fixtures and injected fakes; a finding that needs a GPU run becomes a task
  in this group.
- Requires: `book-full-run-on-the-target-scan`, `book-retrieval-reading`, `book-reference-graph-reading`,
  `book-recovery-library-extraction`, `book-pilot-reference-transcription`,
  `book-review-queue-resolution`, `book-goldset-verification`, `book-vault-obsidian-acceptance`
- Dependencies: none beyond `Requires`; the limitations the full run, the Obsidian checklist, and
  the retrieval reading recorded are this audit's input.
- User-visible outcome: every load-bearing property of the spec section and every clause of the
  registry evaluation names the test that pins it; each limitation the real runs recorded is a
  regression fixture or a stated non-goal; the seams below are integration-tested.
- Scope boundary: in scope -- the property-to-test map; a logic review of the seams: cache-key
  completeness of every stage, the coordinate frame from variant to ledger to crop, overlay
  anchoring across ledger rebuilds, id and anchor stability under a parameter or engine change,
  document revisions across re-projections, the projected reference sidecar against the vault's
  links and the reference graph against that sidecar, the standalone library boundary, page-class
  routing and the engine lease, and
  the redistribution class on every export path; integration tests for each unpinned seam, extending
  the synthetic book; synthetic regression fixtures for the real-run limitations (never page
  content); shared helpers and module splits inside `src/llb/prep/book/`. Out of scope -- re-running
  the pilot or the retrieval reading, committing any page image or recovered text, and new recovery
  capability (the spec lifecycle).
- Data and artifact paths: `src/llb/prep/book/`, `tests/llb/prep/book/`, and the audit's task
  record.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: the [book recovery](current/book-recovery.md) area page and the topic
  pages whose behavior changed.

### Versioned data and artifact contracts -- `artifact-contracts`

#### artifact-contract-training-and-integration-migration

The remaining durable surface covers model preparation, optimization, fine-tuning, screening,
inference handoffs, and robotics integration. Several of these are exported directly to another
runtime, while robotics already supplies strict Pydantic records that should join the shared
registry instead of remaining a separate versioning island.

- Serves: `artifact-contracts` -- [Versioned data and artifact contracts](../design/spec.md#versioned-data-and-artifact-contracts)
- Agent status: CLEAR
- Dependencies: [run, study, and board
  contracts](current/artifact-contracts/run-and-evaluation-contracts.md) supply the run manifest,
  score-row, and study records optimization and training consume and emit back to; none open.
- User-visible outcome: model, training, and robotics handoffs expose the same inspectable identity,
  validation, and compatibility behavior as corpus and run bundles.
- Scope boundary: in scope -- durable producers and consumers under `src/llb/backends/`,
  `src/llb/optimize/`, `src/llb/finetune/`, `src/llb/inference/`, `src/llb/screen/`,
  `src/llb/judge/`, `src/llb/robotics/`, and `src/llb/standalone/`, plus their CLI entry points;
  model and adapter manifests, training datasets and journals, search ledgers, compatibility
  reports, recommendations, standalone answer exports, evidence and action records, and Parquet
  projection schemas. Out of scope -- model weights, PEFT and tokenizer-native files,
  Optuna/SQLite internals, MCAP payloads, hardware-driver protocols, and ML framework checkpoints.
- Data and artifact paths: the current model, tuning, fine-tune, screen, inference, and robotics
  `$DATA_DIR/<method>/<run>/` roots; exported adapter and recommendation metadata; robotics fixtures;
  and compatibility fixtures under `samples/artifact_contracts/integrations/`.
- Execution path: CPU-only contract and fake-adapter tests inside `make ci`; no training, model
  download, GPU load, external API, device connection, or HFlow installation is required.
- Acceptance gates: `make ci` green; current manifests and robotics exchanges round-trip through
  the shared registry; cross-file digests and schema ids bind datasets to opaque
  members; unsupported versions refuse before training, model load, external export, or device
  proposal evaluation; the existing robotics safety invariants remain unchanged.
- Documentation target:
  `docs/impl/current/artifact-contracts/training-and-integration-contracts.md`, the
  artifact-contracts area index, and links from model, optimization, fine-tuning, and robotics topic
  pages.

#### artifact-contract-boundary-enforcement

Close the migration with one operator surface and one repository gate. A new serializer must not be
able to bypass the registry, and an external application must not need producer-specific Python to
discover, validate, or materialize a supported contract.

- Serves: `artifact-contracts` -- [Versioned data and artifact contracts](../design/spec.md#versioned-data-and-artifact-contracts)
- Agent status: CLEAR
- Requires: `artifact-contract-training-and-integration-migration`
- Dependencies: every in-scope producer must have a contract before temporary migration exemptions
  can be removed.
- User-visible outcome: `llb artifacts inspect`, `llb artifacts validate`, and
  `llb artifacts migrate --out <target>` work consistently across datasets and run bundles, and CI
  prevents new unregistered durable data from entering the repository.
- Scope boundary: in scope -- CLI commands in `src/llb/cli/`, a machine-readable contract catalog,
  consumer and producer inventory enforcement, explicit boundary exemptions, generated-schema drift
  enforcement, and copy-only migration provenance. Out of scope -- in-place mutation, automatic
  migration on write, remote schema registries, data discovery outside paths the operator supplies,
  and indefinite support for every historical version.
- Data and artifact paths: the registered `schemas/artifacts/` catalog, all compatibility fixtures,
  operator-selected artifact paths, and a new target under `$DATA_DIR/artifact-migrations/<run>/` for
  explicit materializations; the source path remains unchanged.
- Execution path: add `make validate-artifacts PATH=<artifact-path>` with a `##` help description;
  run the no-unregistered-producer check and external-consumer fixtures inside `make ci`.
- Acceptance gates: `make ci` and `make lint-md` green; every in-scope producer and consumer is in
  the registry and temporary migration exemptions are empty; every permanent exemption names the
  boundary owner and reason; inspect and validate require no optional GPU stack; explicit migration
  records source digest and the complete version path; an injected raw JSON writer makes the
  repository gate fail.
- Documentation target: `docs/impl/current/artifact-contracts/operator-workflow-and-coverage.md`,
  the artifact-contracts area index, CLI reference, and the data egress boundary where exported
  metadata is described.

#### artifact-contract-retrieval-diagnostic-sidecars (optional)

The retrieval comparison and routing-calibration sidecars are registered; the remaining
`src/llb/cli/rag/` analysis bundles -- the embedding and reranker bake-offs, the multi-hop probe,
duplicate residue, retrieval validation, and fusion evidence -- still write unvalidated report
dictionaries. Bring them onto the same registered families so an archived diagnostic is readable
by a build that has moved on.

- Serves: `artifact-contracts` -- [Versioned data and artifact contracts](../design/spec.md#versioned-data-and-artifact-contracts)
- Agent status: CLEAR
- Dependencies: the paired-statistics rows these bundles share are already registered -- see
  [retrieval and graph contracts](current/artifact-contracts/retrieval-and-graph-contracts.md);
  none open.
- User-visible outcome: an archived bake-off, probe, or residue report states which contract it is
  and refuses rather than reading as an empty section when a field has moved.
- Scope boundary: in scope -- the report writers under `src/llb/cli/rag/` and the models behind
  them under `src/llb/rag/`; Pydantic contracts for each bundle, reusing the shared paired
  statistics rows. Out of scope -- the rendered Markdown reports beside them, and any change to
  what a bundle measures.
- Data and artifact paths: the existing bake-off, probe, residue, and evidence roots under
  `$DATA_DIR/<method>/<run>/`, plus fixtures under `samples/artifact_contracts/retrieval_graph/`.
- Execution path: CPU-only fixture writes and reads inside `make ci`; no encoder, reranker, or
  vector backend is loaded.
- Acceptance gates: `make ci` green; each registered bundle reads back identically with and
  without its identity; a report a producer can no longer state refuses before it is written.
- Documentation target: the retrieval and graph contracts page and the RAG-core topic pages that
  own each bundle.

#### artifact-contracts-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. The migration lands family by family -- training, inference, screening, robotics, then
the operator CLI and the repository gate -- while its promise is a property of the whole tree: no
durable record escapes the registry, and an external application validates any of them without
`llb`. A producer moved onto the registry whose consumer still reads raw JSON, or the reverse,
passes both tasks' gates.

- Serves: `artifact-contracts` -- [Versioned data and artifact contracts](../design/spec.md#versioned-data-and-artifact-contracts)
- Agent status: CLEAR -- fixtures and injected fakes; a finding that needs a heavy run becomes a
  task in this group.
- Requires: `artifact-contract-training-and-integration-migration`,
  `artifact-contract-boundary-enforcement`
- Dependencies: none beyond `Requires`.
- User-visible outcome: every registered family names its producer, its consumers, and the test
  that round-trips it, and each cross-family handoff is integration-tested from writer to reader.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: the writer and every reader of a family resolving through
  the same registry entry, cross-file digests verified at read as well as at write, schema drift
  caught without a version declaration, `inspect`, `validate`, and `migrate` reading one catalog,
  and no second schema version introduced under the pre-release rule; integration tests for each
  unpinned seam, including one chain from a training dataset through its journal and adapter
  manifest to a recommendation that the external-consumer fixture validates; shared helpers across
  the contract modules. Out of scope -- families beyond the migrated surface, and any compatibility
  shim.
- Data and artifact paths: `src/llb/core/contracts/`, `src/llb/artifacts/`, `schemas/artifacts/`,
  the fixtures under `samples/artifact_contracts/`, and the audit's task record.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: the [artifact contracts](current/artifact-contracts.md) area page and the
  topic pages whose behavior changed.

### Gold data and ground truth -- `gold-data`

#### gold-data-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. The accepted multi-hop ledger becomes the input of two lanes at once -- the fusion sweep
and the answer-quality comparison -- through the verification gate. Whether its verified grounding,
its paired-power requirement, and its corpus-version binding survive every hop is a property of
that chain, not of the review that produced the ledger.

- Serves: `gold-data` -- [Gold data and ground truth](../design/spec.md#data-and-ground-truth)
- Agent status: CLEAR -- fixtures and injected fakes; a finding that needs a heavy run becomes a
  task in this group.
- Requires: `multihop-ledger-human-acceptance`
- Dependencies: none beyond `Requires`.
- User-visible outcome: an accepted ledger is proven to reach every lane that reads it as verified,
  and a drafted item cannot ride along into a verified reading.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: `verify-review` to `verify-accept` to the accepted ledger
  to `compare-graph-fusion` and `compare-answer-quality`, the `grounding` flag and the
  `INCLUDE_DRAFTED` escape, the paired-power requirement checked where a ledger is consumed and not
  only where it is accepted, and the corpus-version binding on the accepted bundle; integration
  tests for each unpinned seam over committed worksheet fixtures and injected fakes. Out of scope --
  re-reviewing items and re-running the sweep on the real ledger.
- Data and artifact paths: `src/llb/goldset/`, the verification and comparison tests, and the
  audit's task record.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: [verification gate](current/data-prep/verification-gate.md) and the
  [GraphRAG](current/graphrag-backend.md) pages whose behavior changed.

### Entity resolution -- `entity-resolution`

#### relation-predicate-linkage-lane (optional)

The induced relation vocabulary runs to nearly one surface per fact, so most relation-scoped axioms
have no population on a given corpus
([ontology axiom layer](current/robustness-ontology-backends.md#ontology-axiom-layer)). Some
surfaces name one predicate, some name a predicate and its inverse, and some name a narrower
predicate. KGGen and AutoOntic cluster relations for this reason
([method survey](../design/graph-construction-methods.md#the-methods)). The linkage seam can price
the same decision on the fields a predicate record carries, and the DIRECTION of the argument
overlap is what separates an inverse from a synonym.

- Serves: `entity-resolution` --
  [Entity resolution](../design/spec.md#entity-resolution-and-record-linkage)
- Agent status: RUN NEEDED -- the fit and its reading run over the three draft extraction ledgers
  the axiom base rates read, on the CUDA host.
- Research: yes
- Dependencies: the linkage seam and the overlay pattern of the graph node lane
  ([entity resolution](current/entity-resolution.md#the-graph-node-lane)); the axiom checker and its
  ledgers ([ontology axiom layer](current/robustness-ontology-backends.md#ontology-axiom-layer)).
- User-visible outcome: a predicate overlay proposes which relation surfaces are one predicate and
  which pairs are inverses. A reading shows what the overlay does to each relation-scoped axiom's
  checked population. The proposals stay a ranked candidate list until a reviewer labels them.
- Scope boundary: in scope -- one record per distinct relation surface; comparisons on surface
  distance, embedding cosine, the argument-type signature, forward argument-pair overlap, and
  reversed overlap; a pair agreeing only when reversed proposed as `inverse_of` and never merged;
  the overlay read by the axiom checker only when the operator names it; graph retrieval left
  unchanged, since it reads no relation label. Out of scope -- rewriting an extraction ledger, the
  axiom file, or a stored graph; an LLM synonym-versus-subtype judge (a narrower predicate is a
  non-match here); labelling; and adopting any cut without reviewer labels.
- Data and artifact paths: predicate records beside the existing lanes under `src/llb/linkage/` and
  `src/llb/prep/ontology/linkage/`; the reading under `$DATA_DIR/relation-predicate-linkage/<run>/`.
- Execution path: a fixture ledger with a planted synonym pair, inverse pair, and distinct pair
  inside `make ci`; on the CUDA host, the fit over the three ledgers, then
  `make validate-ontology-axioms` over each with and without the overlay.
- Acceptance gates: `make ci` green; the fixture's synonym pair clusters, its inverse pair is
  proposed as an inverse, and its distinct pair stays apart; a graph retrieval run with the overlay
  named matches one without it exactly; the reading states the checked-unit count per ledger and per
  relation-scoped axiom with and without the overlay; the proposals are reported as a ranked list,
  never as a precision; an overlay that populates no axiom it did not already populate is recorded
  as buying nothing there, and the negative goes to [future research](future-research.md).
- Documentation target: a predicate-lane section of `docs/impl/current/entity-resolution.md`.

#### entity-resolution-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. One reviewer-labelled merge ledger is meant to set two operating points -- the graph
overlay's and the gold-item suppression policy's -- through one label-fitted model. The labelling
task and the adoption task each prove their own half; whether a cannot-tell label is treated the
same way in both curves, and whether a drafting run applies exactly the cut its curve recommended,
shows only across them.

- Serves: `entity-resolution` --
  [Entity resolution](../design/spec.md#entity-resolution-and-record-linkage)
- Agent status: CLEAR -- fixtures and injected fakes; a finding that needs a heavy run becomes a
  task in this group.
- Requires: `entity-merge-labelled-set`, `gold-item-drop-policy-adoption`
- Dependencies: none beyond `Requires`.
- User-visible outcome: the path from a workbench merge decision to a merge or a suppression in
  either lane is integration-tested, and neither lane can run on the other's threshold.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: workbench decisions to the label ledger to the
  label-fitted model to the accuracy curve to the adopted threshold, cannot-tell handling in
  precision and recall, the shadow report at the adopted cut matching the drafting decision, and
  each lane reading its own operating point; integration tests for each unpinned seam over the
  lanes' committed fixtures. Out of scope -- new labels and a third linkage domain.
- Data and artifact paths: `src/llb/linkage/`, the lane tests, and the audit's task record.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: [entity resolution](current/entity-resolution.md).

### Retrieval evidence -- `retrieval-evidence`

#### reranker-end-to-end-crosscheck-on-an-evidence-model

The off-by-default reranker verdict asks to be overturned with END-TO-END evidence, and the only
end-to-end run behind it cannot supply that: it scored `llama3.2:3b` -- below the project's >=7B
live-evidence floor -- on n=14, where the two arms' intervals overlap so widely that the reading is
"no measured gain" rather than a measured loss, and its bundles are retained on no host
([rerank](current/rag-core/rerank-and-query.md#reranking-and-context-order-rerank-context-order)).
The retrieval half still stands on its own at that small k. So the open question is narrow -- does
that retrieval lift reach the ANSWER on a model the project accepts for evidence? Re-run the same
`rerank_candidates=0,30` cross-check on a >=7B local model, at an n predeclared to separate the two
recorded objectives, and restate the off-by-default verdict as reproduced, corrected, or retired. A
negative result -- the lift still does not reach the answer -- is the valuable outcome, because it
is what the shipped default rests on.

- Serves: `retrieval-evidence` -- [RAG core](../design/spec.md#capability-registry)
- Agent status: RUN NEEDED
- Dependencies: none in code. `make sweep` already takes the grid, and the paired lane already
  supplies the interval and the minimum-evidence gate
  ([paired verdicts](current/rag-core/paired-verdicts.md#the-minimum-evidence-gate-on-a-paired-reading)).
- User-visible outcome: an operator deciding whether to pay ~0.96 s/query for reranking gets an
  answer measured on a model they would actually serve, instead of one bounded by a 3B smoke model.
- Scope boundary: in scope -- the predeclared n, the two-arm run on a >=7B model, and the restated
  verdict. Out of scope -- changing the shipped `rerank_candidates` default before the run supports
  it, widening the pool grid, and swapping the reranker.
- Data and artifact paths: one `$DATA_DIR/run-eval/<run>/` per arm under the sweep's own root.
- Execution path: `make sweep SWEEP_RAG_GRID="rerank_candidates=0,30"
  RERANKER=BAAI/bge-reranker-v2-m3` on a CUDA host with a >=7B model resident; CI is unchanged.
- Acceptance gates: `make ci` green; the model is >=7B; the predeclared n is stated BEFORE the run
  and met; both arms report objective with paired intervals and the win/loss/tie ledger; the reading
  states whether the retrieval lift reaches the answer, and whether the minimum-evidence gate is
  cleared.
- Documentation target:
  [rerank](current/rag-core/rerank-and-query.md#reranking-and-context-order-rerank-context-order).

#### retrieval-evidence-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. Its open verdicts -- reranking end to end, the embedder, the goods fusion weight --
each settle on their own lane, while what they share sits between them: the paired-power contract,
the minimum-evidence gate, the measurement floor, and the path from a recorded verdict to a shipped
default. A fix made inside one comparison lane can leave the others computing the same interval
another way.

- Serves: `retrieval-evidence` -- [Retrieval evidence](../design/spec.md#retrieval-before-generation)
- Agent status: CLEAR -- fixtures and injected fakes; a finding that needs a heavy run becomes a
  task in this group.
- Requires: `reranker-end-to-end-crosscheck-on-an-evidence-model`,
  `embedder-decision-on-a-resolvable-item-set`, `goods-fusion-weight-accepted-ledger`
- Dependencies: none beyond `Requires`.
- User-visible outcome: every shipped retrieval default traces to the recorded verdict behind it,
  and the comparison lanes compute one paired interval for the same arms.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: each retrieval default against its recorded verdict, the
  predeclared item count and effect enforced before retrieval rather than only stated, `make
  sweep`, `compare-retrieval`, and `compare-embeddings` sharing one paired-statistics path, and the
  measurement floor applied the same way in every lane; integration tests for each unpinned seam,
  including one fixture item set read through every lane to the same interval for the same arms and
  a default that disagrees with its verdict failing a test. Out of scope -- re-running any verdict,
  this group's optional readings, and widening a grid.
- Data and artifact paths: `src/llb/rag/`, `src/llb/cli/rag/`, their tests, and the audit's task
  record.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: [paired verdicts](current/rag-core/paired-verdicts.md) and the RAG-core
  pages whose behavior changed.

### Answer scoring -- `answer-scoring`

#### answer-scoring-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. The format-weight calibration moves one declared constant, and that constant feeds the
headline decomposition, the board ranking, and every roster rank published from it. Whether a
weight change reaches the board only through the declared ranking policy, and whether the
context-copy shape stays penalized after it, shows only across scoring and the board.

- Serves: `answer-scoring` -- [Answer scoring](../design/spec.md#scoring-policy)
- Agent status: CLEAR -- fixtures and injected fakes; a finding that needs a heavy run becomes a
  task in this group.
- Requires: `calibrate-headline-format-weight`
- Dependencies: none beyond `Requires`.
- User-visible outcome: a weight change is proven to move ranks only through the declared policy,
  with every rank it moves named.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: the declared weight to the decomposition to the board
  rank, the held-out confirmation required before a default change, the four answer shapes scored
  after the move, and the precision, recall, and found-rate columns left unchanged by it;
  integration tests for each unpinned seam over fixture answers of the four shapes and a fixture
  board. Out of scope -- new labels and judge substitution.
- Data and artifact paths: `src/llb/scoring/`, `src/llb/board/`, their tests, and the audit's task
  record.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: [scoring](current/rag-core/scoring.md).

### Judge calibration -- `judge-calibration`

#### judge-calibration-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. The frontier lane crosses the boundary this project treats as irreversible -- data
leaving the host -- and its authorization is split across a consent flag, a spend cap, a provider
key, and a human decision field. Each is tested where it is read; whether every path to a provider
passes all of them, and whether a judge whose decision is still pending stays out of every
autonomous gate, is a property of the lane as a whole.

- Serves: `judge-calibration` -- [Judge calibration](../design/spec.md#judge-admission)
- Agent status: CLEAR -- fake providers and committed worksheets; a finding that needs a live run
  becomes a task in this group.
- Requires: `frontier-judge-authorization`
- Dependencies: none beyond `Requires`.
- User-visible outcome: no provider request is reachable without consent and a cap, and a frontier
  judge ranks nothing until its recorded decision admits it.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: consent, cap, and key checked before the first request on
  every provider path, the cap math against the measured cost per item, a pending decision or a
  below-gate correlation demoting the judge to diagnostic wherever a gate reads it, and the egress
  refusal of non-redistributable material; integration tests for each unpinned seam with fake
  providers. Out of scope -- live provider runs and new providers.
- Data and artifact paths: `src/llb/scoring/frontier_agreement/`, the judge tests, and the audit's
  task record.
- Execution path: CPU-only inside `make ci`; no network.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: [judging](current/rigor-board-judge/judging.md).

### Graph retrieval and ontology -- `graph-retrieval`

#### graph-seeded-propagation-ranking

Both graph strategies rank a NODE by a coarse level -- a hop distance or a community -- so every
span of a node ties and the rank cut falls inside the tie. The multi-hop reading puts the both-hops
ceiling at the budget rather than at any ranking knob the fusion grid sweeps
([retrieval budget evidence](current/graphrag-backend/retrieval-budget-evidence.md#is-the-both-hops-ceiling-a-budget-or-a-query-problem)),
and hard node merging lost recall at every cut that merged anything
([entity resolution evidence](current/graphrag-backend/entity-resolution-evidence.md)). Personalized
PageRank from the question's linked seeds, the HippoRAG method, ranks every node by how much of the
walk reaches it. A bridge entity that two seeds both connect to then outranks a node one seed
reaches alone, and a soft similarity edge between two near-identical nodes lets relevance flow
without asserting they are one. Build it as a third strategy and read it against the two it would
join.

- Serves: `graph-retrieval` -- [Graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology)
- Agent status: RUN NEEDED -- the paired reading runs over the built multi-hop stores on the CUDA
  host.
- Research: yes
- Dependencies: the seed linker and the span serializer
  ([retrieval strategies](current/graphrag-backend/modules-and-cli.md#retrieval-strategies)); the
  node mention vectors the resolution lane already embeds
  ([graph node lane](current/entity-resolution.md#the-graph-node-lane)); the fusion sweep and its
  measurement floor
  ([fusion sweep evidence](current/graphrag-backend/fusion-sweep-evidence.md#the-graph-weight-sweep-lane)).
- User-visible outcome: `--retrieval-strategy ppr` ranks graph spans by the propagated mass that
  reaches them from the question's seeds, optionally across declared similarity edges, through the
  same span-scored seam as the other two strategies. The comparison, fusion, and answer-quality
  lanes can select it. A recorded verdict says whether it carries more multi-hop evidence at a fixed
  budget than `local_khop` and `global_community`.
- Scope boundary: in scope -- PPR over the stored graph by sparse power iteration on the `rag`
  extra's SciPy, with no graph-analytics dependency; seeds weighted by the existing linker's
  scores; the damping factor, iteration cap, and tolerance as named constants; similarity edges
  between nodes whose mention vectors clear a declared cosine floor, with an edges-off arm; spans
  scored by the mass of the nodes they ground, so the span-affinity band still orders exact ties; a
  deterministic ranking for a fixed graph; the strategy in the config fingerprint and in the
  comparison, fusion, and sweep grids. Out of scope -- the `hipporag` package, which pins its own
  Python and LLM stack; LLM-side triple filtering; changing either shipped strategy or the default
  backend; merging any node; and a similarity edge reaching the axiom checker or the linkage seam.
- Data and artifact paths: `src/llb/graph/` (a propagation module beside `retrieval.py`) and
  `src/llb/graph/constants.py`; the readings under `$DATA_DIR/graph-vector-fusion-multihop/<run>/`.
- Execution path: fixture graphs inside `make ci`; on the CUDA host, `make compare-graph-fusion`
  with `ppr` in the compared strategies and `NOISE_FLOOR=1`, over the widened multi-hop draft and
  the goods ledger the earlier fusion readings used, at the shipped budget and at k=50.
- Acceptance gates: `make ci` green; one graph and one seed set give one ranking twice; on a fixture
  where a shared bridge node carries both hops, both hop spans rank above a node one seed reaches
  alone; the reading states per corpus and per question-type slice the paired deltas in
  `all-spans@k`, span coverage, recall@k, and MRR against both shipped strategies and the vector
  row, with intervals and the measurement floor; a drafted ledger's reading is labelled drafted;
  the similarity-edge arm is read against the edges-off arm as the soft-identity answer to the
  hard-merge negative; a row that separates is read end to end through `compare-answer-quality` on
  the same items, naming its generator, before any recommendation, and changes no default here; a
  strategy that separates from neither shipped strategy is recorded as buying nothing, and the
  negative goes to [future research](future-research.md) with what would reopen it.
- Documentation target: the strategy in `docs/impl/current/graphrag-backend/modules-and-cli.md`,
  and the reading on a new evidence page in the [GraphRAG](current/graphrag-backend.md) area index.

#### graph-closed-ie-extraction-adapter (optional)

Every graph and every drafted gold set rests on generative extraction: one model call per window,
with any quote the model paraphrases dropped at grounding. A compact span classifier given the
closed vocabulary as its labels (GLiNER) returns exact spans by construction and runs far faster,
which matters most on a 1,067-page book. Whether its entities make a graph the lane can use, on
Ukrainian corpora where its coverage is not documented, is a measurement.

- Serves: `graph-retrieval` -- [Graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology)
- Agent status: RUN NEEDED -- the extraction and the paired reading run on the CUDA host.
- Research: yes
- Dependencies: the `ExtractionAdapter` seam, with the spaCy adapter as its precedent
  ([spaCy adapter](current/robustness-ontology-backends.md#spacy-adapter-and-long-documents)); the
  closed vocabulary ([graph ontology schema](../design/graph-ontology-schema.md)).
- User-visible outcome: `--extractor gliner` writes a `DocExtraction` whose entities are exact spans
  typed in the closed vocabulary. Its facts come either from the generative extractor or, in the
  model-light variant, from a `co_occurs` relation evidenced by the sentence two entities share. An
  operator learns what each variant costs and what it does to graph retrieval.
- Scope boundary: in scope -- the adapter behind its own optional extra, with a pinned Apache-2.0
  multilingual checkpoint recorded per run; the 13 types passed as its labels and still passed
  through `normalize_entity_type`; the sentence co-occurrence facts as a separately labelled
  variant; entity agreement with the generative extraction, the normalization rate, and extraction
  wall time per corpus. Out of scope -- replacing the default extractor, fine-tuning a checkpoint,
  the non-commercial REBEL weights, and drafting gold items from the co-occurrence variant, because
  a co-occurrence is not a stated fact.
- Data and artifact paths: `src/llb/prep/ontology/endpoints/` beside the spaCy adapter; the bundles
  under `$DATA_DIR/prepare-goldset/<run>/` and the comparisons under
  `$DATA_DIR/graph-vector-fusion-multihop/<run>/`.
- Execution path: a fake classifier inside `make ci`; on the CUDA host, extraction over the corpus
  behind the widened multi-hop draft and over the projected book slice, then `make build-graph` per
  variant and `make compare-graph-fusion` over the same items with `NOISE_FLOOR=1`.
- Acceptance gates: `make ci` green; every emitted mention grounds at its offsets; the reading
  states per corpus the wall time, the entity agreement with the generative extraction, and each
  variant's paired graph-lane deltas against the generative graph on the same items with
  intervals; a checkpoint whose Ukrainian entities are unusable is recorded as such by
  measurement; a variant that costs graph recall beyond the floor is not adopted, and the negative
  goes to [future research](future-research.md).
- Documentation target: the adapter section of
  [robustness and ontology](current/robustness-ontology-backends.md#spacy-adapter-and-long-documents),
  and the reading in the [GraphRAG](current/graphrag-backend.md) area.

#### graph-community-detection-candidates (optional)

`global_community` is the graph strategy that reaches the most multi-hop evidence at a larger
budget, and its communities come from deterministic label propagation, which fragments a sparse
graph into many small communities. Leiden, the community step of Microsoft GraphRAG, guarantees
connected communities and exposes a resolution. Whether it gives the narrative layer communities
worth retrieving is a paired reading.

- Serves: `graph-retrieval` -- [Graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology)
- Agent status: RUN NEEDED -- graph rebuilds and the paired reading on the CUDA host.
- Dependencies: the community step
  ([modules and CLI](current/graphrag-backend/modules-and-cli.md)); the tuning/final split
  discipline ([evaluation rigor](current/rigor-board-judge.md)).
- User-visible outcome: a recorded choice between label propagation and Leiden for the narrative
  layer, with the community count and size distribution each produces and what each does to
  `global_community` retrieval.
- Scope boundary: in scope -- Leiden from `graspologic-native` behind its own optional extra,
  seeded, with the seed and resolution recorded in the graph metadata; the resolution chosen on the
  tuning split and read on final; communities detected the same way after an entity overlay. Out of
  scope -- community summaries as retrieval context (they stay diagnostic), a hierarchy of
  community levels, and changing the default before a final-split separation.
- Data and artifact paths: `src/llb/graph/community.py` and `src/llb/graph/constants.py`; the
  readings under `$DATA_DIR/graph-vector-fusion-multihop/<run>/`.
- Execution path: fixture graphs inside `make ci`; on the CUDA host, `make build-graph` per detector
  over the stores the fusion readings used, then `make compare-graph-fusion` over the
  `global_community` rows at the shipped budget and at k=50 with `NOISE_FLOOR=1`.
- Acceptance gates: `make ci` green; one seed partitions one graph identically twice; the reading
  states per corpus the paired deltas with intervals and the floor, beside the partition
  statistics; label propagation stays the default unless Leiden separates on the final split, and a
  negative is recorded with the partition statistics that explain it.
- Documentation target: the community step in
  `docs/impl/current/graphrag-backend/modules-and-cli.md`, and the reading in the
  [GraphRAG](current/graphrag-backend.md) area.

#### graph-retrieval-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. An axiom reaches an answer through three readers -- the extraction-ledger checker, the
OWL cross-check, and the answer gate -- and the sign-off adds the rule all three must honour: only
a signed axiom may act. Whether every reader refuses an unsigned or rejected axiom, and whether the
checker and the reasoner agree on the signed set, shows only across them. The graph lane also gains
a third strategy, and whether every strategy still emits only grounded spans through one serializer
shows only across the tasks that add them.

- Serves: `graph-retrieval` -- [Graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology)
- Agent status: CLEAR -- fixtures and injected fakes; a finding that needs a heavy run becomes a
  task in this group.
- Requires: `ontology-axiom-signoff`, `graph-seeded-propagation-ranking`
- Dependencies: none beyond `Requires`.
- User-visible outcome: an axiom absent from the signed file is refused by every reader, and the
  checker and the reasoner agree on every signed axiom's fixtures. Every strategy emits only spans
  that resolve to their document text, and a similarity edge changes no axiom reading.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: the per-axiom signature (`dcterms:creator`,
  `dcterms:date`) to the checker to the violation base rate to the answer gate, the refusal of an
  unsigned axiom at each entry point, checker-versus-reasoner agreement on the signed set, catch
  and false-rejection rates computed by one path, each strategy through the one serializer, and the
  structural vocabulary kept out of entity normalization; integration tests for
  each unpinned seam over the committed axiom file and fixture ledgers. Out of scope -- new axiom
  classes and vocabulary changes.
- Data and artifact paths: `src/llb/prep/ontology/`, `samples/ontology/`, their tests, and the
  audit's task record.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: [robustness and ontology](current/robustness-ontology-backends.md).

### Autonomous orchestration -- `autonomous-orchestration`

#### autonomous-orchestration-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. The acceptance run drives `auto-rag` twice on a real corpus -- autonomous and
human-gated -- and turns its usability findings into tasks; it leaves no CI test behind for the
failures it saw. Two modes sharing one stage graph, and a resume while a workbench decision is
still pending, are seams only the two-mode run exposes.

- Serves: `autonomous-orchestration` -- [Autonomous orchestration](../design/spec.md#autonomous-orchestration)
- Agent status: CLEAR -- the quickstart corpora and injected fakes; a finding that needs a heavy run
  becomes a task in this group.
- Requires: `autonomous-vs-assisted-acceptance`
- Dependencies: none beyond `Requires`; the acceptance record's findings are this audit's input.
- User-visible outcome: each failure the acceptance run surfaced is a fixture, and both modes resume
  from an interrupt with bundles reproducible from their manifests.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: the autonomous and human-gated modes over one stage
  graph, a resume with a gated record pending, post-run self-verification in both modes, and
  recommendation bundles re-derived from their manifests; integration tests for each unpinned seam
  over the quickstart corpora. Out of scope -- re-running the acceptance and new orchestration
  stages.
- Data and artifact paths: `src/llb/auto_rag/`, its tests, and the audit's task record.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: [Auto-RAG](current/auto-rag.md).

### Robotics RAG and hardware operation -- `robotics-rag-operation`

#### robotics-rag-operation-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. The MHS adapter and the supervised canary open the first paths to a real device, and
the safety claim -- every write traverses the project gate, and an ambiguous result never retries
-- must hold on every path, not only on those the conformance suite and the canary phases happen to
exercise.

- Serves: `robotics-rag-operation` -- [Robotics RAG and hardware operation](../design/spec.md#robotics-rag-and-hardware-operation)
- Agent status: CLEAR -- recorded, redacted responses and the emulator; no device is contacted.
- Requires: `robotics-mhs-preview-conformance`, `robotics-supervised-device-canary`
- Dependencies: none beyond `Requires`.
- User-visible outcome: no adapter write is reachable except through the project gate, and a
  recorded canary session replays in CI with every planted violation blocked.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: the emulator and the MHS adapter under one conformance
  suite, a phase advance requiring a fresh preflight and a new approval digest, the mismatch log
  covering every field the emulator models, and no credential in any bundle; integration tests for
  each unpinned seam replaying a redacted recorded session. Out of scope -- any live device call and
  safety certification.
- Data and artifact paths: `src/llb/robotics/`, `samples/robotics/`, their tests, and the audit's
  task record.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: [robotics RAG](current/robotics-rag.md).

### Corpus conflict and governance -- `corpus-conflict-audit`

#### conflict-adjudicator-think-control (optional)

`build_adjudicator` sets a model, a backend, a temperature, and a seed, and nothing else, so a
reasoning model on the Ollama path spends its token budget on reasoning and emits no verdict:
gemma4:12b returned 30 unparsable completions out of 40 probe pairs and the calibration gate
correctly refused it ([measured claim-tier
precision](current/data-prep/conflict-claim-precision.md#measured-five-model-families-against-both-tiers)).
`EndpointConfig` already carries `think` and `num_ctx`; the adjudicator never passes them, so one
of the five registered families cannot serve as an adjudicator at all. Plumb a think control
through the audit and the calibration command, and re-score that family against both probe tiers to
say whether the refusal was plumbing or quality.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: none. `EndpointConfig.think` and the Ollama native path already exist; the gap is
  that `src/llb/conflicts/claim/adjudicator.py` does not expose them.
- User-visible outcome: an operator can point the claim tier at a reasoning model without every
  verdict coming back unparsable.
- Scope boundary: in scope -- the think (and, if the same run shows it is needed, `num_ctx`)
  control on both commands, and one calibration re-score of the refused family. Out of scope --
  changing the prompt, the parser's strictness, and any default that would turn reasoning on for
  the families that already work without it.
- Data and artifact paths: `$DATA_DIR/corpus-conflict-calibration/<run>/` only.
- Execution path: one calibration-only run for the re-scored family on the CUDA host; CI covers the
  flag reaching the endpoint config with an injected fake.
- Acceptance gates: `make ci` green; the re-scored family returns parsable verdicts on both tiers,
  and the page records its agreement beside the other four.
- Documentation target: [measured claim-tier precision](current/data-prep/conflict-claim-precision.md#measured-five-model-families-against-both-tiers).

#### conflict-probe-hard-tier-promotion (optional)

The hard probe tier separates adjudicators the base tier cannot, but the gate still reads only the
base tier: at 16 pairs the same 0.60 Wilson bound means `>= 14/16`, so one relabelled pair moves a
family across it, and the ordering rests on one host, one seed, and one model per family
([measured claim-tier
precision](current/data-prep/conflict-claim-precision.md#measured-five-model-families-against-both-tiers)).
Reproduce the ordering on a second seed and widen the hard tier until its pass mark does not turn
on a single pair, then either add `hard` to `TIER_ACCURACY_GATES` or record that the separation is
a ranking an operator reads rather than a bar the audit enforces.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: RUN NEEDED
- Dependencies: the tiered probe and the per-tier gate map are current behavior ([measured
  claim-tier precision](current/data-prep/conflict-claim-precision.md#the-frozen-calibration-probe));
  promoting a tier is one entry in `TIER_ACCURACY_GATES`.
- User-visible outcome: an adjudicator whose specificity collapses on arguable pairs either cannot
  publish a precision figure, or is documented as an operator's own call -- not left undecided.
- Scope boundary: in scope -- more hard pairs in the same corpus, a second-seed re-score of the
  families already measured, and the promotion decision. Out of scope -- a third tier, a second
  hard corpus, and changing what the base tier contains.
- Data and artifact paths: `samples/corpora/conflicts_uk_v1/probe_hard/`,
  `samples/corpora/conflicts_uk_v1/adjudicator_probe.json`, and
  `$DATA_DIR/corpus-conflict-calibration/<run>/`.
- Execution path: one calibration-only run per family per seed on the CUDA host; CI covers the
  widened tier's passage resolution and label balance.
- Acceptance gates: `make ci` green; the widened tier's pass mark does not turn on one pair, the
  measured ordering holds across seeds, and the page states whether `hard` gates.
- Documentation target: [measured claim-tier precision](current/data-prep/conflict-claim-precision.md#the-frozen-calibration-probe).

#### conflict-precision-bound-at-document-clustering (optional)

The clustered bound treats a CHUNK as the independent unit, and on a concentrated corpus that is
still too generous. Measured: every one of the goods corpus's 8 actionable rows at budget 100 points
at the same right document, so at the document level the corpus supplies one observation, not eight
-- yet the chunk-level bound resamples 42 left and 51 right chunks
([conflict detection](current/data-prep/conflict-claim-precision.md#measured-both-quickstart-corpora)).
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
- Documentation target: [conflict detection](current/data-prep/conflict-claim-precision.md#measured-claim-tier-precision).

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
  detection](current/data-prep/conflict-claim-precision.md#measured-claim-tier-precision).

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
- Agent status: RUN NEEDED
- Research: yes
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

#### corpus-conflict-audit-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. The chain runs from detection through adjudication, calibration, the plan, the review
ledger, and resolution to application and rollback, and its two human tasks change what flows
through it: frozen labels that can suppress the precision block, and reviewed decisions that remove
chunks from a store. Each stage pins itself; whether an applied reviewed ledger still rolls back to
the exact baseline, and whether a suppressed precision figure stays suppressed in every report that
quotes it, is a property of the chain.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: CLEAR -- the committed fixture corpus and an injected adjudicator; a finding that
  needs a heavy run becomes a task in this group.
- Requires: `conflict-adjudicator-label-slice`, `corpus-conflict-resolution-review`
- Dependencies: none beyond `Requires`.
- User-visible outcome: the committed fixture corpus runs the whole chain in CI, from detection to an
  applied and rolled-back resolution, with every report agreeing on the precision it may quote.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: detection to adjudication to calibration to `plan.json`
  to the review ledger to resolution to apply and rollback, the agreement bound suppressing
  precision in every report that reads it, the group ids `plan.json` joins on staying stable across
  re-reads, and stage attribution recomputed from a bundle alone; integration tests for each
  unpinned seam. Out of scope -- this group's optional research readings and any detector or
  threshold change.
- Data and artifact paths: `src/llb/conflicts/`, `samples/corpora/conflicts_uk_v1/`, their tests,
  and the audit's task record.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: [conflict detection](current/data-prep/conflict-detection.md) and the
  topic pages whose behavior changed.

### Operator review tooling -- `operator-review-tooling`

#### operator-review-tooling-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. The workbench is one review surface for many decision domains -- verification,
conflicts, merges, axioms, the book review queue, dual-control scenarios -- and each adapter is
tested on its own. The capability's evaluation is ledger compatibility ACROSS adapters, which no
single adapter's task checks.

- Serves: `operator-review-tooling` -- [Operator review tooling](../design/spec.md#operator-review-tooling)
- Agent status: CLEAR -- committed fixture ledgers; no reviewer is needed for the audit itself.
- Dependencies: none; it covers the adapters registered when it starts, and names the ones still
  to come.
- User-visible outcome: every registered adapter passes one shared test -- open, decide, quit
  mid-record, reopen, resume -- and writes a ledger the compatibility check accepts.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: the adapter registry and the ledger shape, resume after
  an interruption, and a decision never lost when a session ends mid-record; one integration test
  parametrized over every registered adapter. Out of scope -- this group's optional throughput
  measurements and new decision domains.
- Data and artifact paths: `src/llb/review/`, its tests, and the audit's task record.
- Execution path: CPU-only inside `make ci`.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: [review workbench](current/review-workbench.md).

### Simulated-user dual-control workloads -- `simulated-user-dialogue`

#### dual-control-suite-seam

Nothing in this repo drives a conversation with a second actor. `bench-agentic` runs one model
against a `ToolWorld` and reads planted assertions off it ([category
suite](current/category-benchmark-suite.md)); there is no user simulator, no written policy the
agent must obey, and no reward that decomposes into world end state versus what the agent had to
say. Build that seam once, over tau2-bench's orchestrator rather than a second one of our own:
it already carries the domain/policy/tool/task contracts, the end-state reward, and a task set the
external-suite lane can screen against. This task is the seam and its artifact -- no Ukrainian
domain yet, no headline number.

- Serves: `simulated-user-dialogue` -- [Simulated-user dual-control workloads](../design/spec.md#simulated-user-dual-control-workloads)
- Agent status: CLEAR
- Dependencies: none. The tier reuses the existing backend launcher and isolation contract
  ([host validation](current/host-validation.md)) and the category-run bundle writer
  (`persist_category_run`); neither has open work that blocks it.
- User-visible outcome: `llb bench dual-control --domain <id> --model <id>` runs a dual-control
  case set against a local endpoint for BOTH the agent and the user simulator, and writes a run
  bundle whose reward decomposition, per-trial outcomes, pinned suite tag, domain digest, and
  user-simulator identity can be re-read without rerunning the models.
- Scope boundary: in scope -- a `tau2` optional extra pinned to a git tag; a lazy-imported adapter
  under `src/llb/bench/dual_control/` that registers our agent/user construction against the suite
  registry and passes `api_base` through `llm_args` so `openai/<model>` resolves to a local vLLM or
  Ollama endpoint; a `TAU2_DATA_DIR` binding to `$DATA_DIR`; a registered artifact contract for the
  run bundle carrying `suite_tag`, `domain_id`, `domain_digest`, `agent_model`, `user_model`,
  `user_endpoint`, `num_trials`, and the per-case component rewards; and the throughput meter every
  other category runner threads. Out of scope -- voice, audio, and full-duplex paths; the gym/RL
  interface; the suite's own knowledge-retrieval pipeline (disabled, our retrieval stack is
  unaffected); leaderboard submission; and any Ukrainian domain, which is
  `ua-dual-control-domain`.
- Data and artifact paths: `$DATA_DIR/dual-control/<run_timestamp>/`; a committed miniature domain
  fixture under `samples/benchmarks/dual_control/` (policy, world, two cases, one reference
  trajectory) so CI never depends on the suite's bundled data; the pinned tag recorded in
  `pyproject.toml` and echoed into every bundle.
- Execution path: `make bench-dual-control` with a `##` help description. CI runs the fixture domain
  with an injected fake completion for both actors -- no network, no GPU, no suite data download.
  The real-endpoint path is a separate heavy invocation on the CUDA host.
- Acceptance gates: `make ci` green with the suite absent from the base install (the import is lazy
  and its absence produces a named refusal, not a traceback); a fixture run's reward recomputes from
  the saved bundle alone and equals the value the run reported; a bundle whose `user_model` differs
  from another's makes a comparison REFUSE rather than average the two; removing `suite_tag` from
  the bundle makes a test fail.
- Documentation target: a new `docs/impl/current/extended-workflows/dual-control-tier.md`, its row
  in the [extended workflows](current/extended-workflows.md) area index, and the reuse note in
  [product decisions](current/scope-boundaries.md).

#### dual-control-screen-discrimination

Once the seam exists, the first question is whether the tier buys anything. The external-suite lane
runs the suite's English domains unmodified as a candidate screen; the honest test is whether it
SEPARATES roster models that `bench-agentic` already ranks as tied. If it does not, the tier is a
screen and nothing more, and that is the result to record rather than a lane to keep feeding.

- Serves: `simulated-user-dialogue` -- [Simulated-user dual-control workloads](../design/spec.md#simulated-user-dual-control-workloads)
- Agent status: CLEAR -- the run is deterministic at fixed seeds and executes on the CUDA host with
  no human judgment.
- Requires: `dual-control-suite-seam`
- Dependencies: none.
- User-visible outcome: a recorded verdict on whether dual-control screening adds ranking
  information over the single-actor agentic tier, and a per-model reliability number an operator can
  read as "succeeds k times out of n" rather than as a mean.
- Scope boundary: in scope -- the English suite lane over the roster finalists at fixed seeds and a
  pinned user-simulator model; success-across-trials reported with its uncertainty width under the
  [published-number rule](current/extended-workflows/published-values.md); a paired reading against
  the `bench-agentic` ranking on the commonly-run models, under the existing minimum-evidence gate
  ([paired verdicts](current/rag-core/paired-verdicts.md)). Out of scope -- any localized domain;
  changing the agentic tier's ranking; a board row (the external lane never enters one); and tuning
  the agent prompt to lift the suite score, which would make the screen ours rather than the
  suite's.
- Data and artifact paths: `$DATA_DIR/dual-control/<run_timestamp>/` per model; the committed run
  aggregate the reading cites, under the same content pin every other published aggregate carries.
- Execution path: heavy, sequential, one backend process per model under the GPU isolation contract
  ([heavy runs and evidence](../guides/development/heavy-runs-and-evidence.md)); the analysis
  recomputes from the saved bundles with no GPU.
- Acceptance gates: `make ci` green; the reading states the model count, the predeclared effect it
  could detect, and either the separation it found or a plain statement that the lane buys nothing
  over the agentic tier on this roster; a null outcome is written up in
  [future research](future-research.md) with what would reopen it, and the lane stays a screen.
- Documentation target: `docs/impl/current/extended-workflows/dual-control-tier.md` and, if the
  outcome is null, [future research](future-research.md).

#### dual-control-tool-protocol-conformance (optional)

The suite's orchestrator speaks native OpenAI `tools=`; `bench-tooling` already carries the finding
that local models differ in how well they do ([category
suite](current/category-benchmark-suite.md)). A roster model that cannot emit a well-formed tool
call through its serving backend will score zero on every dual-control case for a reason that has
nothing to do with conversation, and reporting that as a conversational failure would be wrong.

- Serves: `simulated-user-dialogue` -- [Simulated-user dual-control workloads](../design/spec.md#simulated-user-dual-control-workloads)
- Agent status: CLEAR
- Requires: `dual-control-suite-seam`
- Dependencies: none.
- User-visible outcome: a per-model, per-backend statement of whether native tool calling is
  available at all for the dual-control tier, so a zero is attributable to the protocol rather than
  read as conversational incapacity.
- Scope boundary: in scope -- a conformance probe over the roster on each serving backend, its
  malformed-call rate, and a refusal that names the model/backend pair when the protocol is
  unavailable. Out of scope -- writing a text-protocol shim into the suite's orchestrator, teaching
  a model to call tools, and any change to `bench-tooling`'s own protocol axis.
- Data and artifact paths: the conformance record inside the dual-control run bundle; no new root.
- Execution path: a short probe against a live endpoint on the CUDA host; the fixture path in CI
  uses an injected fake that emits one well-formed and one malformed call.
- Acceptance gates: `make ci` green; a model/backend pair with no native tool support produces a
  named refusal rather than a zero-scored run.
- Documentation target: `docs/impl/current/extended-workflows/dual-control-tier.md`.

#### simulated-user-dialogue-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. The seam, the English screen, and the Ukrainian domain each bring their own gates -- a
lazy import, a reward recomputed from the bundle, a refused mixed comparison, a replay check -- and
the registry evaluation states them as one contract. Whether the Ukrainian domain's bundles carry
the identity fields the screen's comparisons refuse on, and whether no board can carry both lanes,
shows only once all three exist.

- Serves: `simulated-user-dialogue` -- [Simulated-user dual-control workloads](../design/spec.md#simulated-user-dual-control-workloads)
- Agent status: CLEAR -- committed fixture domains with fakes for both actors; a finding that needs
  a real-endpoint run becomes a task in this group.
- Requires: `dual-control-suite-seam`, `dual-control-screen-discrimination`,
  `ua-dual-control-domain`
- Dependencies: none beyond `Requires`.
- User-visible outcome: a fixture-domain run per lane passes the whole contract in CI, and every
  refusal the evaluation names is integration-tested.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: the absent-suite refusal, the reward recomputed from the
  bundle, the user-model mismatch refusal, the replay check on the Ukrainian domain, the external
  lane refused on a localized board, and the tool-protocol refusal where it exists; integration
  tests for each unpinned seam. Out of scope -- real-endpoint runs and new domains.
- Data and artifact paths: `src/llb/bench/dual_control/`, `samples/benchmarks/dual_control/`, their
  tests, and the audit's task record.
- Execution path: CPU-only inside `make ci`; no network.
- Acceptance gates: `make ci` green; the map covers every property and evaluation clause; each new
  integration test is shown once in the record to fail with its seam broken on purpose; every
  finding is fixed, routed to a task in this group, or sent through the spec lifecycle, and listed
  in the record.
- Documentation target: `docs/impl/current/extended-workflows/dual-control-tier.md`.

### Documentation integrity -- `documentation-integrity`

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
- Data and artifact paths: `docs/impl/current/data-prep/conflict-bundle-record.md`, the new
  topic page beside it, and the [data prep](current/data-prep.md) area table; no run artifact.
- Execution path: move the subsections along the heading seam, add the area-page row, repoint
  the inbound links, and run `make lint-md`; no code change and no run.
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
- Data and artifact paths: `docs/impl/current/data-prep/conflict-decision-groups.md`, the new
  topic page beside it, and the [data prep](current/data-prep.md) area table; no run artifact.
- Execution path: move the subsections along the heading seam, add the area-page row, repoint
  the inbound links, and run `make lint-md`; no code change and no run.
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
- Data and artifact paths: `docs/impl/current/entity-resolution.md`, the
  `current/entity-resolution/` pages it becomes, and the inbound pages named under Dependencies;
  no run artifact.
- Execution path: move the subsections along the heading seam, add the area-page row, repoint
  the inbound links, and run `make lint-md`; no code change and no run.
- Acceptance gates: `make lint-md` green with zero broken links; no page past ~500 lines; no anchor
  text changes.
- Documentation target: [entity resolution](current/entity-resolution.md) and the pages the split
  produces.

#### summary-input-elision-page-is-past-the-split-threshold (optional)

[summary-input bounds and elision](current/extended-workflows/summary-input-elision.md) is ~520
lines and its headings describe two subjects: what the summarize-input BOUND is (the step-aligned
cap, and what eliding under it costs on a typed-memory shape) and what the entry-aware TRIM is (the
middle-critical transfer, the adoption ladder with its arm-order and guard-fit subsections, and the
shipped default). That is past the ~500-line split rule in [AGENTS.md](../../AGENTS.md), and the
trim subject is the one that grows: it arrived as a prototype, became a policy field, and has now
taken a default move, each time landing on the same page. The seam is already a heading boundary
and the code is split along it too (`llb.bench.memory.window_elision` against
`llb.bench.summary_trim`), so move the trim sections to their own topic page, add its row to the
area page, and repoint the inbound links. Anchors keep their fragments, so only paths change and
`make lint-doc-links` proves the move.

- Serves: `documentation-integrity` -- [Documentation integrity](../design/spec.md#specification-and-plan-integrity)
- Agent status: CLEAR
- Dependencies: none. `make lint-md` (which runs `lint-doc-links`) is the whole gate; the inbound
  links are in `extended-workflows.md` and `samples/benchmarks/agentic_context_policy_pins.json`,
  whose `published_in` anchors the pin gate checks by path.
- User-visible outcome: a reader who wants the shipped trim default stops scrolling past the whole
  bound story and two study ladders to reach it.
- Scope boundary: in scope -- the split, the area-page row, the `published_in` repointing, and the
  link repointing. Out of scope -- rewriting the moved text and changing any measured result.
- Data and artifact paths: `docs/impl/current/extended-workflows/summary-input-elision.md`, the
  new topic page beside it, the area table, and
  `samples/benchmarks/agentic_context_policy_pins.json`; no run artifact.
- Execution path: move the subsections along the heading seam, add the area-page row, repoint
  the inbound links, and run `make lint-md`; no code change and no run.
- Acceptance gates: `make lint-md` green with zero broken links; neither page is past ~500 lines;
  no anchor text changes; the pin gate's doc-anchor test still passes.
- Documentation target: [extended workflows](current/extended-workflows.md) and the two pages the
  split produces.

#### documentation-integrity-implementation-audit (optional)

The [closing audit](../guides/development/planning-workflow.md#closing-implementation-audit) of this
capability. The rules for the plan, the spec, the records, and the docs tree are stated in prose in
[AGENTS.md](../../AGENTS.md) and the planning workflow, and enforced in part by `make
lint-spec-plan`, `make lint-doc-links`, and the record-name check. The gap between what the prose
promises and what a gate refuses is invisible from either side -- including the closing-audit rule
this task belongs to, which no gate checks yet.

- Serves: `documentation-integrity` -- [Documentation integrity](../design/spec.md#specification-and-plan-integrity)
- Agent status: CLEAR
- Dependencies: none.
- User-visible outcome: every rule the prose states is either refused by a gate over a synthetic
  fixture or named as contributor judgment, and `make plan-status` agrees with the gate on the same
  tree.
- Scope boundary: in scope -- the property-to-test map over the spec section and the registry
  evaluation; a logic review of the seams: each parser-checkable rule in AGENTS.md and the planning
  workflow against the checks in `src/llb/quality/`, the closing-audit rule (one per capability, last
  in its group, requiring every open required task of its capability) either gated or recorded as
  judgment, and `plan-status` and the gate reading one plan model; integration tests over a
  synthetic repository tree with one planted violation per rule. Out of scope -- this group's page
  splits and any rule the spec does not state.
- Data and artifact paths: `src/llb/quality/`, `tests/llb/quality/`, and the audit's task record.
- Execution path: CPU-only inside `make ci` and `make lint-md`.
- Acceptance gates: `make ci` and `make lint-md` green; the map covers every property and evaluation
  clause; each new integration test is shown once in the record to fail with its seam broken on
  purpose; every finding is fixed, routed to a task in this group, or sent through the spec
  lifecycle, and listed in the record.
- Documentation target: [documentation and specification
  gates](current/overview.md#documentation-and-specification-gates).

## Human-Assisted Tasks

Work whose acceptance needs human judgment or authorization (`BLOCKED BY HUMAN`, `HUMAN-GATED`), in
the same capability order. An agent can build the supporting code and unit tests; the human step
named in `Dependencies` is what gates completion.

### Scanned book recovery -- `scanned-book-recovery`

These are release gates for the extracted library. First verify the pilot to select the release
profile, then exercise the review queue on the release candidate. The gold set and Obsidian gates
use the resulting pinned corpus and vault. Release requires all four gates.

#### book-pilot-reference-transcription

The release pipeline comparison needs an independent human reference.
Every block on every pilot page -- text, formula, table, figure box, caption, reading order, printed
page number -- has to be checked against the scan by a person, because a reference drafted by the
candidates and accepted unread would hand the pilot to whichever candidate drafted it.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: HUMAN-GATED -- an agent pre-drafts every pilot page in the reference format as a
  merge of at least two shortlisted engines, with every disagreement flagged and no engine named,
  keeps the unedited pre-draft, and prepares a side-by-side worksheet per page; a person must
  correct and accept each page against the scan.
- Requires: `book-recovery-library-extraction`
- Dependencies: the reference format is
  [quality metrics](current/book-recovery/quality-metrics.md). The pilot list is
  [page classes](current/book-recovery/tool-selection.md#page-classes). Human step that gates
  completion: a reviewer reads every pilot page against its
  scan and corrects text,
  LaTeX, table structure, figure boxes, caption links, reading order, and the printed page number,
  then records an accept per page.
- User-visible outcome: a verified reference transcription of the frozen pilot,
  a scored and selected release profile, and a mechanically validated vault and corpus from
  the extracted library using that profile.
- Scope boundary: check every block against the source scan. Preserve the blind
  pre-draft and review history. Count an existing page acceptance only if its scan and exact
  accepted transcription still match. Use the verified reference for the frozen comparison and
  optional-correction decision. Republish if the selected profile differs from the provisional
  one. Keep recovered page content under `DATA_DIR`.
- Data and artifact paths: reference and run artifacts under `$DATA_DIR/book-ocr/`;
  checksum, reviewer record, library build, and selected profile beside
  `samples/book_ocr/<book-slug>/pilot_pages.json`.
- Execution path: review every page in the extracted library's worksheet, validate the
  digest with the score command, replay frozen candidates, and stage the release corpus and vault
  through the packaged commands.
- Human step: correct and accept every pilot page; a page left unaccepted keeps the comparison from
  starting.
- Acceptance gates: `make ci` green; every pilot page carries an acceptance bound to
  the source and reference bytes; the scorer verifies the recorded digest. Report per-class
  margins and intervals against the existing PDF lane, dangerous substitutions, and agreement
  with the unedited draft. Name candidates that win no class and keep the baseline where the
  interval supplies no evidence of improvement. Check specialist formula fidelity and table
  similarity against the selected pipeline, and report figure overlap and caption association.
  Count fixed errors and formerly correct tokens made wrong separately. Enable correction only
  if overall error falls and dangerous substitutions do not rise. Record negative decisions with
  reopening conditions in [future research](future-research.md). Before downstream review, pin
  the release vault, corpus and report with full page coverage, source provenance, and evidence
  of reproducible publication after resume for the selected profile.
- Documentation target: `docs/impl/current/book-recovery/tool-selection.md`.

#### book-review-queue-resolution

Exercise human queue decisions on a seeded sample of the release vault. The review must show
that a correction follows its source region and that remaining uncertainty stays visible.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: HUMAN-GATED -- every decision is a person reading the scan.
- Requires: `book-pilot-reference-transcription`
- Dependencies: Human step that gates completion: review the seeded entries in the
  packaged worksheet against their crops and source pages. Sample chapter and page classes
  and include unresolved technical content and missing regions where present.
- User-visible outcome: an operator can resume decisions and see the reviewed
  text and remaining markers in the republished vault and corpus.
- Scope boundary: resolve the release sample through the workbench and republish
  from the overlay. Remaining uncertainty stays visible, with chapter-by-chapter cleanup at
  the operator's discretion.
- Data and artifact paths: the overlay and review queue inside the run directory.
- Execution path: use the packaged review, render, project, and validate commands
  on the selected profile. Pin the resulting artifacts for gold and Obsidian acceptance.
- Human step: decide each queue entry: accept raw, accept the candidate, enter the text, or mark it
  illegible.
- Acceptance gates: every sampled item carries a human verdict; the refreshed outputs
  validate, preserve provenance, and report queue size before and after. Remaining markers and
  scan links resolve in both exports. Record the new output digests if decisions change them.
- Documentation target: `docs/impl/current/book-recovery/full-run-evidence.md`.

#### book-goldset-verification

The recovered book becomes benchmark evidence only through the same gate as any other corpus: drafted
questions over the projection, each verified by a person against its source span and, through the
page provenance, against the scan itself.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: HUMAN-GATED -- an agent drafts the items over the ingested projection; a person must
  verify each one before any number from the book is published.
- Requires: `book-review-queue-resolution`
- Dependencies: the drafting lanes and the verification gate are reused unchanged
  ([verification gate](current/data-prep/verification-gate.md)). Human step that gates completion:
  a reviewer verifies grounding, answerability, and reference correctness of each drafted item,
  opening the scanned page through the item's page provenance whenever the recovered text is in
  doubt.
- User-visible outcome: an accepted gold set tied to the release corpus and question
  language, plus the paired retrieval and printed-reference graph readings on those items.
- Scope boundary: draft with the existing gold-set workflow over the pinned projection,
  verify its items, and compare retrieval against the baseline and the structural graph.
  Keep the graph vocabulary to printed references. Use the same retrieval settings, continue to
  separate the baseline-text coverage split, and split graph results by technical evidence.
  Translation and answer scoring are outside this gate.
- Data and artifact paths: the draft bundle and accepted ledger under `$DATA_DIR`; no item text from
  the book in git.
- Execution path: `make prepare-goldset-draft`, `make review-workbench`, and
  `make validate-goldset`; then run the paired vector and graph evaluations on the verified set.
- Human step: verify every drafted item; an item whose span overlaps an unresolved marker is
  rejected rather than repaired from memory.
- Acceptance gates: every item carries a verdict and the accepted ledger validates
  against the release corpus digest; no accepted span overlaps an unresolved marker. Report the
  item count, minimum detectable effect, paired deltas and intervals per declared slice, and
  interpretation. Record negative retrieval or graph results and what would reopen them in
  [future research](future-research.md).
- Documentation target: `docs/impl/current/book-recovery/corpus-projection.md`.

#### book-vault-obsidian-acceptance

The validator proves links resolve on disk; only the Obsidian app shows whether a PDF page link
opens the right page, whether LaTeX and HTML tables render, and whether the book reads naturally.
Those are judgments about the deliverable itself, and they belong to the person who will use it.

- Serves: `scanned-book-recovery` -- [Scanned book recovery](../design/spec.md#scanned-book-recovery)
- Agent status: HUMAN-GATED -- an agent publishes the vault and draws the sample checklist; a person
  must open it in the recorded Obsidian version and judge it.
- Requires: `book-review-queue-resolution`
- Dependencies: Human step that gates completion: the operator opens the published vault in Obsidian
  (version recorded) and works a sampled checklist -- PDF page links and their fallback, embeds,
  display and inline math, Markdown and HTML tables, previous/next and contents navigation, resolved
  cross-references, uncertainty markers -- plus a sampled audit of passages, formulas, tables, and
  figures against their scanned pages.
- User-visible outcome: a recorded verdict that the vault works as a reading and auditing tool in
  the operator's Obsidian, with each failed check turned into a fix under its owning task or a new
  task in this group.
- Scope boundary: in scope -- the sampled checklist, the passage audit, and the recorded verdicts.
  Out of scope -- resolving the review queue, and restyling the vault beyond what a failed check
  requires.
- Data and artifact paths: the published vault; the checklist and verdicts committed under
  `samples/book_ocr/<book-slug>/` without page content.
- Execution path: the agent draws a seeded sample across chapters and page classes and writes the
  checklist; the operator works it in Obsidian.
- Human step: work the checklist and the passage audit, and record a verdict per item.
- Acceptance gates: every item carries a verdict bound to the release vault digest;
  confirm the PDF page-link form or its fallback in the recorded Obsidian version; check sampled
  paragraph continuity, table captions and cell values, formulas and figures against the scan;
  route failures to fixes and repeat every check affected by a fix before release.
- Documentation target: `docs/impl/current/book-recovery/full-run-evidence.md` and the usage guide
  `docs/guides/data-prep/book-ocr.md`.

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
- Requires: `entity-merge-labelled-set`
- Dependencies: its reviewer merge ledger is the gate -- without it there is no curve to read a
  threshold off. The lane, its record
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
- Requires: `multihop-ledger-human-acceptance`
- Dependencies: the [shared paired-power contract](current/rag-core/paired-verdicts.md#paired-power-contract-for-comparison-lanes)
  supplies the predeclared split sizes, and the required task must provide a non-empty accepted
  multi-span ledger. Human
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
- Agent status: HUMAN-GATED
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

#### frontier-judge-retrieved-context-agreement (optional)

Re-measure frontier-vs-human judge agreement with *retrieved* contexts instead of the
gold-span windows the authorization lane uses. The current lane deliberately holds retrieval
constant by grounding each item on a window of its gold source document, which isolates judge
behavior but also hands the judge cleaner evidence than a scored run ever gives it. A judge that
ranks well on oracle context may rank differently when the context contains distractors or misses
the answer entirely -- exactly the cases where an autonomous gate matters most. Add a context
source switch to `load_agreement_items` that pulls each item's top-k retrieved chunks from an
existing store, then report both grounding modes side by side so the gap is visible.

- Serves: `judge-calibration` -- [Judge calibration](../design/spec.md#judge-admission)
- Agent status: HUMAN-GATED
- Requires: `frontier-judge-authorization`
- Dependencies: the same provider keys and spend approval. Reuse the agreement lane in
  `src/llb/scoring/frontier_agreement/` and the store-loading
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

Accept or reject each candidate axiom in `samples/ontology/axioms_uk_v1.ttl`, one at a time, before
any of them can gate an answer. An axiom is BUSINESS LOGIC, not a measurement: whether "has owner"
admits one value or many, whether `PERSON` and `ORG` are genuinely disjoint in this domain, and
whether a relation's range is closed are claims about the world that no corpus statistic can settle.
Inducing them from corpus frequency would only restate what the extractor happened to emit -- the
same circularity the conflict tier already hit, where the null and the observed population turned
out to be the same set ([data
prep](current/data-prep/conflict-detection.md#known-limitation-there-is-no-independent-null)) -- and
the cost of a wrong axiom is asymmetric and silent: at the ledger it deletes a true fact from the
report's attention, and at the answer gate it converts correct answers into `ontology_violation`.
The corpus cannot review itself here, which is why this is the one piece of the validation
architecture that sits in this section. The existing signed type-vocabulary review is the precedent
for the form ([graph ontology schema](../design/graph-ontology-schema.md)).

- Serves: `graph-retrieval` -- [Graph retrieval and ontology](../design/spec.md#graph-retrieval-and-ontology)
- Agent status: HUMAN-GATED
- Dependencies: the candidate axioms, their Turtle rendering, and the per-axiom evidence rows
  (supporting facts, contradicting facts, and the measured base rate per corpus) already ship as
  `axiom_evidence.jsonl` beside each validation run ([robustness and
  ontology](current/robustness-ontology-backends.md#ontology-axiom-layer)). Reuse the
  review-workbench ledger pattern
  ([review workbench](current/review-workbench.md)) so the decisions are recorded the same way every
  other review ledger is. Human step that gates completion: a domain reviewer decides `accept` or
  `reject` for EVERY candidate axiom and signs the resulting axiom file.
- User-visible outcome: a signed, dated constraint set an operator can point the answer gate at,
  where every enabled axiom is a decision someone made rather than a statistic the corpus produced.
- Scope boundary: in scope -- the per-axiom review worksheet (each axiom rendered as Turtle plus a
  Ukrainian-language gloss and its supporting/contradicting facts with exact spans), the review
  pass, the signed axiom file, and a recorded reason per rejection. Out of scope -- authoring new
  axiom CLASSES or changing the checker, changing the 13-type vocabulary or the relation caps, and
  enabling any axiom the reviewer did not accept.
- Data and artifact paths: the worksheet under `$DATA_DIR/ontology-validation/<run>/axiom_review.jsonl`,
  built from the `axiom_evidence.jsonl` the validation run already writes beside it; the signed set
  committed at `samples/ontology/axioms_uk_v1.ttl`, where a signature is `dcterms:creator` +
  `dcterms:date` on that axiom's `owl:Axiom` annotation block, mirroring the dated sign-off
  convention of [graph ontology schema](../design/graph-ontology-schema.md).
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

### Robotics RAG and hardware operation -- `robotics-rag-operation`

#### robotics-mhs-preview-conformance

Replace the protocol-neutral fake with a real adapter only when an authorized preview participant
can provide an inspectable, versioned MHS contract and a non-production test endpoint. Map discovery,
reference generation, read, write, enforced limits, errors, and receipts onto the project records,
then run the same conformance suite as the emulator. A missing schema, incompatible license,
unstable identity/state semantics, or transport that bypasses the external gate is a valid negative
result and must leave the project labelled protocol-neutral.

- Serves: `robotics-rag-operation` -- [Robotics RAG and hardware operation](../design/spec.md#robotics-rag-and-hardware-operation)
- Agent status: HUMAN-GATED
- Research: yes
- Dependencies: the [robotics boundary contracts](current/robotics-rag/boundary-contracts.md) and
  the [action gate and emulator](current/robotics-rag/action-gate-and-emulator.md). Human step
  that gates completion: an authorized MHS preview participant accepts the applicable terms,
  identifies the exact contract/package version, and grants time-bounded access to a simulator or
  read-only test device without placing credentials in the repository or run artifacts.
- User-visible outcome: either a version-scoped MHS adapter with reproducible conformance evidence,
  or a precise incompatibility/insufficient-contract decision that prevents an unsupported
  compatibility claim.
- Scope boundary: in scope -- one optional adapter, contract and license pinning, discovery/read
  against a non-production endpoint, fake write conformance with no physical effect, and deviations
  from the project contract. Out of scope -- applying for access without the operator, accepting
  terms on their behalf, live hardware writes, bypassing the project gate, storing credentials, and
  declaring compatibility across untested MHS versions.
- Data and artifact paths: sanitized contract metadata and conformance output under
  `$DATA_DIR/robotics-mhs-conformance/<run>/`; committed tests use redacted recorded responses or an
  injected fake under `samples/robotics/mhs/`, never preview package bytes whose terms forbid it.
- Execution path: add `make robotics-mhs-conformance` behind a `robotics-mhs` optional dependency
  group and explicit endpoint configuration; keep preview access outside quick CI and replay the
  adapter against contract-safe fixtures in `make ci`.
- Acceptance gates: the human records the exact contract and allowed artifact handling; discovery,
  read, limit refusal, error, and receipt cases map without loss; every write attempt still traverses
  the project gate; changed MHS identity or schema makes the adapter fixture stale; the report says
  `MHS-compatible` only for the tested version; a negative result removes this task to future
  research with its reopening condition instead of weakening the adapter contract.
- Documentation target: `docs/impl/current/robotics-rag/mhs-adapter.md` and the dependency/security
  boundary in `docs/impl/current/robotics-rag.md`.

#### robotics-supervised-device-canary

Run the admitted MHS adapter on one isolated workcell under its hardware owner's supervision. Start
with discovery/read-only calls and shadow proposals, reconcile every observed identity, unit, limit,
state transition, and receipt against the emulator, and only then authorize one predeclared low-risk
write whose physical envelope, rollback, interlocks, and stop procedure the owner signs. Any drift
returns the capability to shadow mode; the canary never turns on unattended operation.

- Serves: `robotics-rag-operation` -- [Robotics RAG and hardware operation](../design/spec.md#robotics-rag-and-hardware-operation)
- Agent status: HUMAN-GATED
- Requires: `robotics-mhs-preview-conformance`
- Dependencies: the [held-out emulator benchmark](current/robotics-rag/benchmark.md). Human step
  that gates completion: the hardware owner supplies
  the isolated device, approves the risk and rollback worksheet, verifies interlocks and an
  operator-controlled emergency stop, observes the entire run, grants the proposal-bound write
  approval, and signs the canary verdict.
- User-visible outcome: a bounded real-device reading that names where emulator behavior transfers
  and where it does not, without silently promoting a benchmark result into deployment authority.
- Scope boundary: in scope -- read-only and shadow phases, one low-risk bounded write, pre/post-state
  capture, stop and rollback drills, emulator/device mismatch logging, and a human verdict. Out of
  scope -- safety certification, people or uncontained hazards in the work envelope, high-risk
  tools, generated procedure files, model-driven emergency actions, parallel live writes,
  unattended running, and generalizing from one workcell to another.
- Data and artifact paths: run bundle under `$DATA_DIR/robotics-canary/<run>/`; raw telemetry and MCAP
  remain in the authorized HFlow data root and are referenced by digest; the signed risk worksheet,
  approvals, redacted device snapshot, and verdict live in the run bundle with no secrets.
- Execution path: add `make bench-robotics-canary MODE=read,shadow,bounded-write`; require a fresh
  preflight before each phase, and require a new explicit approval digest before the write phase.
  The operator, not the model, advances phases.
- Acceptance gates: the owner signs the worksheet and final verdict; the physical stop prevents
  motion independently of the model and adapter; zero out-of-policy invocations occur; every action
  has matching pre-state, decision, request, receipt, and post-state evidence; an ambiguous result
  triggers read/escalation with no retry; every emulator/device mismatch is reported; any mandatory
  gate failure retains shadow/read-only mode and records the negative result.
- Documentation target: `docs/impl/current/robotics-rag/device-canary.md` and a canary procedure under
  `docs/guides/human-tooling/`.

### Corpus conflict and governance -- `corpus-conflict-audit`

#### conflict-adjudicator-label-slice

Produce frozen human labels for real candidate rows so the audit's measured claim-tier precision can
be trusted off the planted fixture. The shipped calibration gate scores the adjudicator only against
the seven-document planted probe, whose relations are synthetic by construction and which the
current host model passes 24/24 ([conflict
detection](current/data-prep/conflict-claim-precision.md#the-frozen-calibration-probe)); nothing measures
whether the model agrees with a human on HR or goods rows.

- Serves: `corpus-conflict-audit` -- [Corpus conflict and governance](../design/spec.md#corpus-conflict-and-governance)
- Agent status: HUMAN-GATED
- Dependencies: the audit's precision block and its calibration gate are current behavior
  ([measured claim-tier
  precision](current/data-prep/conflict-claim-precision.md#measured-claim-tier-precision)) and
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
- Data and artifact paths: the two goods bundles under `$DATA_DIR/corpus-conflicts/<run>/` as the
  measured shapes; the timing capture and fitted-cost report under the measuring command's own
  `$DATA_DIR/<method>/<run>/` root.
- Execution path: the timing capture, the fit, and the ranking key are agent-buildable and covered
  by fixture tests in `make ci`; the reviewer then works a concentrated and a scattered block in
  the review TUI under that capture, and the result either changes `_stake_order` in
  `src/llb/conflicts/resolution/review.py` or records that the linear rank survives.
- Acceptance gates: `make ci` green; the report states the fitted per-group and per-row costs with
  their uncertainty, and either changes `_stake_order` or records that the linear rank survives.
- Documentation target: [conflict
  resolution](current/data-prep/conflict-resolution.md#decision-groups-in-the-plan-and-the-review-ledger).

### Simulated-user dual-control workloads -- `simulated-user-dialogue`

#### ua-dual-control-domain

The external-suite lane screens candidates in English; only a Ukrainian domain can inform the local
choice this project exists to make. Build one on material this repo already owns -- a written policy
in Ukrainian, a small tool-backed world, and cases whose user-side scenario is written the way a
Ukrainian speaker would actually state a half-formed request. This is human-gated for one reason:
whether a case's Ukrainian scenario states the same task, under the same policy, with the same
missing information, is a judgment no fixture can make, and a mistranslated scenario silently
becomes a different benchmark.

- Serves: `simulated-user-dialogue` -- [Simulated-user dual-control workloads](../design/spec.md#simulated-user-dual-control-workloads)
- Agent status: HUMAN-GATED -- an agent can build the domain, the tools, the world, the reference
  trajectories, and the replay check; a Ukrainian reviewer must accept the policy and each case's
  scenario before any number from the lane is published.
- Requires: `dual-control-suite-seam`
- Dependencies: the agent-lane seam is the only cross-lane prerequisite; the domain registers into
  it.
- User-visible outcome: a Ukrainian dual-control domain a local model can be scored on, whose
  numbers may inform a model choice because the language, the policy, and the world are the ones
  the operator would deploy.
- Scope boundary: in scope -- one domain: a Ukrainian policy document, its tool set and world
  schema, a seeded world, the case set with user-side scenarios and reference trajectories, and the
  reviewer pass that accepts them. Out of scope -- translating the suite's bundled English domains
  (an airline reservation policy is not the operator's workload); a second domain before the first
  one has a reading; voice; and any blending of this lane's row with the English screen's.
- Data and artifact paths: the domain under `samples/benchmarks/dual_control/uk/` (policy, world
  seed, tools schema, cases); the reviewer ledger through the existing
  [review workbench](current/review-workbench.md) adapter seam; runs under
  `$DATA_DIR/dual-control/<run_timestamp>/`.
- Execution path: domain construction and the replay check are CPU-only inside `make ci`; the
  reviewer pass runs through the review workbench; the scored run is heavy and sequential on the
  CUDA host.
- Human step: a Ukrainian reviewer accepts (or rejects, with a reason) the policy text and each
  case's scenario as stating the intended task; the lane publishes nothing until that ledger is
  complete.
- Acceptance gates: `make ci` green; replaying every case's reference trajectory on a fresh world
  reproduces that case's target end state, and a deliberately altered translation that moves the
  target makes the check FAIL rather than silently rebase it; the reviewer ledger records an
  accept/reject per case with its reason; the lane's row is refused on any board that also carries
  the English screen.
- Documentation target: `docs/impl/current/extended-workflows/dual-control-tier.md` and the
  [review workbench](current/review-workbench.md) adapter list.
