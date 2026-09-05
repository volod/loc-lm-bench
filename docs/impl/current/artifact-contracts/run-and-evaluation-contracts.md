# Run, Study, And Board Artifact Contracts

Part of the [artifact contracts](../artifact-contracts.md) area of the
[current implementation index](../../current.md). The mechanism these families use is recorded in
[foundation and evolution](foundation-and-evolution.md); the store, graph, and prompt-system
families a run bundle is measured against are in
[retrieval and graph contracts](retrieval-and-graph-contracts.md).

## Delivered boundary

A run bundle is the primary downstream API of this project: a board ranks out of one, a study
cites one, and an external consumer validates one. It is a SET of files that only mean something
together, and the manifest is now what makes the set self-describing rather than a directory a
reader guesses at from filenames.

| Family | Member | Current version |
| --- | --- | --- |
| `llb.run-manifest` | `manifest.json` | 2.0.0 |
| `llb.case-score` | `scores.jsonl` of a RAG evaluation | 1.0.0 |
| `llb.retrieval-case` | `retrieval.jsonl` | 1.0.0 |
| `llb.agentic-case` | `scores.jsonl` of the agentic category | 1.0.0 |
| `llb.security-case` | `scores.jsonl` of the security category | 1.0.0 |
| `llb.structured-case` | `scores.jsonl` of the structured category | 1.0.0 |
| `llb.summarization-case` | `scores.jsonl` of the summarization category | 1.0.0 |
| `llb.text-analysis-case` | `scores.jsonl` of the text-analysis category | 1.0.0 |
| `llb.tooling-case` | `scores.jsonl` of the tooling category | 1.0.0 |
| `llb.study-design` | a benchmark study's `*-design.json` | 1.0.0 |
| `llb.study-analysis` | a benchmark study's `*-analysis.json` and its other readings | 1.0.0 |
| `llb.run-progress` | `cases.progress.jsonl` | 1.0.0 |
| `llb.run-progress-meta` | `cases.progress.meta.json` | 1.0.0 |
| `llb.judge-budget-abort` | `scorer/abort.json` | 1.0.0 |
| `llb.miss-analysis` | `analysis.json` | 1.0.0 |
| `llb.miss-case` | `misses.jsonl` | 1.0.0 |
| `llb.auto-rag-manifest` | auto-RAG `manifest.json` | 1.0.0 |
| `llb.auto-rag-stage-result` | `stages/<stage>/result.json` | 1.0.0 |
| `llb.auto-rag-journal-event` | auto-RAG `journal.jsonl` | 1.0.0 |
| `llb.rag-recommendation` | `rag_recommendation.yaml` | 1.0.0 |

The models live under `src/llb/core/contracts/run_bundle/` (`manifest.py`, `rows.py`,
`journals.py`, `studies.py`, `board.py`, `auto_rag.py`); the declarations, the readers and writers,
the dataset descriptor, the publication gate, and the survey live under
`src/llb/artifacts/run_bundle/`. `RunManifest` in `src/llb/tracking/manifest.py` IS the contract
model now -- it adds no field and changes no meaning, only the two defaults (`created_at`, `env`)
no producer should have to state by hand.

Deliberately unregistered and out of scope: model response text before it is parsed into a score
column, backend and server logs staged beside a bundle, rendered Markdown reports, and MLflow's
own database schema. The mirror stays a best-effort projection of the canonical record and is
never read back as one.

## Deriving a row contract instead of restating one

Every per-case row already had a `TypedDict` the producers build directly, and those declarations
are where the columns and their readings are documented -- `CaseScoreRow` alone carries fifty of
them. Restating that list a second time as a Pydantic model would put the two copies one release
apart the first time a metric is added, and the copy the registry validated would be the stale one.

So `row_contract` in `src/llb/core/contracts/run_bundle/rows.py` DERIVES the contract from the
`TypedDict`: a required key becomes a required field, a `NotRequired` key becomes an optional one
(a column a run did not measure is absent, never zero), and the identity fields are added on top.
The derivation is deterministic, so the generated JSON Schema is stable, and a column added to the
row reaches the schema and the registry in the same change that adds it.

## What a bundle says about its own members

Version 2 of the manifest adds the two declarations version 1 could not carry.

- **`score_rows`** says what the rows of `scores.jsonl` answer to. A run evaluation and the six
  category benchmarks name a registered contract, and every row is validated against it BEFORE
  publication. A benchmark study names itself instead, with the exact column set it published --
  a weaker claim than a contract and still a checkable one, because a reader can ask whether the
  rows it opened are the rows the run said it wrote.
- **`artifacts`** declares every additional file, with its media type, its byte count, and the
  digest it was published at. Each entry names either a record contract or a `human_report`
  reason. There is no third form.

`persist_run` therefore no longer accepts a `name -> text` mapping. It takes already-declared
`RunArtifact` values, and it requires exactly one of `score_contract` or `score_owner`: a bundle
that cannot say what its rows are is a bundle a later reader has to guess about.

## Naming what a study owns

Twenty-odd benchmark studies publish a cell, seed, or crossover table rather than per-case rows,
with the columns that study measured. Those columns are not a family a cross-cutting reader could
know ahead of time, so the bundle binds `scores.jsonl` as an OPAQUE member owned by the method,
for the same reason a store binds its FAISS index that way: the dataset manifest can only bind a
structured member to a record contract, and this table has none. What it does have is the column
set the run published, and the survey checks the rows against exactly that -- a row carrying a
column the run never declared is a different table under the same name.

A pre-contract bundle is bound the same way and says so: its owner reads `unstated`, because an
older run never recorded what its rows answered to.

## Study records that must not move

A prospective design is the point of these studies -- the sample, the families, the effect the run
must reach, and the adoption gates are fixed BEFORE the run, and a design edited afterwards is not
a design. Both the design and the analysis are also cited by DIGEST by
[published-value evidence](../extended-workflows/published-values.md), so re-encoding an archived
record to carry an identity would break the citation that points at it.

So identity is STAMPED rather than stored, exactly as `llb.conflict-stage-inputs` maps its integer
version in [data-prep contracts](data-prep-contracts.md#two-encodings-that-stay-as-they-are). The
producer builds `llb.study-design` or `llb.study-analysis` from the local form and validates it
before publication; a reader rebuilds the same record from the file plus the study id the bundle
declared. What the contract adds over an unread JSON blob is the part a cross-cutting reader needs:
which study a record belongs to, which of the two records it is, and that the body is an object (or
a table of them) at all. The body's own keys stay the study's.

`src/llb/bench/artifacts.py` makes that declaration once for every category and study, rather than
at each of the twenty-odd call sites, and it decides on the CONTENT rather than the filename: a
`.md` file is a rendered report and is declared exempt with its reason, a `.json` file is a design
when it predeclares itself -- naming its study and stating the integer version its own validator
checks -- and a reading the study took otherwise. Anything else refuses, which is what keeps a
third shape from quietly appearing.

An analysis states no study id of its own, so `RunArtifactDeclaration` carries one. Without it the
bundle would hold a reading nobody could attribute, and attribution is the whole value of an
archived one.

## The one migration that preserves a reading

| Family | Old form | What the migration states |
| --- | --- | --- |
| `llb.run-manifest` | `1.0.0`: no `score_rows`, no `artifacts` | Neither is invented: the score-row contract stays absent and the artifact list stays empty |

`score_rows` absent means "this bundle does not state what its rows answer to", never "its rows
have none", and an empty `artifacts` list means the same about its additional files. Everything a
version 1 manifest DID carry -- the run identity, the config, the environment, the metrics, and the
judge, telemetry, contention, and durability records -- is the same field at the same meaning in
version 2 and is carried through untouched.

## Refusing before the board ranks

`admitted_manifest` in `src/llb/board/io.py` is what every board loader now reads a run head
through -- the RAG board, the category board, the harness and context-policy comparisons, the
prompt-system board, and the miss probe. It refuses two things and drops them with their reason:

- a manifest naming a family this build does not know, or a version it cannot read (a future major
  validates as JSON and hides every field a newer writer added);
- rows whose stamped identity contradicts what the manifest declared they were.

Only the first stamped row is read for the second check. A score file is written in one pass by one
producer, so a single row separates a bundle whose members agree from one whose members were mixed,
and the board pays one line rather than one validation per case.

A pre-contract bundle is not such a case: it is read at the version the family declares its history
to be, migrated forward, and admitted like any other.

## Refusing before the rename

Publication is a rename, and a rename is not reversible from the outside: the moment a staging
directory becomes `$DATA_DIR/<method>/<run>/`, a board may read it and a study may cite it. So
`validate_staged_bundle` reads every member back from the STAGED bytes first -- the manifest
through its contract, the score rows through what the run declared, the retrieval sidecar through
its row family, and each additional artifact through its contract or its exemption -- and reports
every refusal rather than the first. A refusal there costs a run that was never published, which is
the cheap end of the trade against a board reading nobody can trust.

## Resume records are durable too

None of the resume records is published inside a bundle: the journal and its meta are dropped from
staging before the rename, and the budget abort is written beside a scorer's resumable state. They
are durable all the same, because a resume READS them and a resumed bundle must score to the same
rows an uninterrupted one would.

`JournaledCaseState` is deliberately narrower than the in-memory `RagState`: the question, the gold
spans, and the assembled context are inputs a resume rebuilds, while everything it names is an
output that cannot be recomputed without calling the model again. `_JOURNALED_STATE_KEYS` is now
DERIVED from that contract, so the set the journal trims to and the set the contract validates
cannot drift -- a column added to a score row without being added there is a column a resumed case
would silently lose. Each journal line is validated as the LINE about to be written, because a
numpy score coerced on the way out is what a resume will actually read back.

## Checking one bundle

```bash
make check-run-bundle RUN_BUNDLE=<run-dir>
```

`check-run-bundle` resolves the manifest first, then reads every member through what THAT manifest
declares, reporting all refusals rather than stopping at the first. It exits 1 when any member
refuses and 2 when the head itself cannot be read. The underlying command is
`llb check-run-bundle <run-dir>`.

## Fixtures and the gates over them

`samples/artifact_contracts/run_bundles/` holds the same run written twice -- `current/` as this
build publishes it and `legacy/` as this project wrote it before the registry existed -- plus a
manifest from an unsupported future major and a bundle whose rows stamp a family its manifest did
not declare.

`tests/llb/artifacts/test_run_bundle_contracts.py` asserts that the pre-contract manifest reads
back as the record a current writer produces (differing only in the two absences it could not
record), that the migrated bundle yields the SAME board reading as the current one, that a future
major and mixed-version rows each refuse before board admission, that an invalid score row and an
undeclared artifact each refuse before publication, that a study record round-trips to the bytes it
was written as, and that an artifact whose bytes changed after publication refuses on its digest.
`tests/llb/cli/test_cli_check_run_bundle.py` covers the operator command and its two exit codes.

`samples/artifact_contracts/external_validate.py` repeats the current, bound-row, pre-contract, and
future-refusal validations against the generated JSON Schemas without importing `llb`, and runs
inside `make check-artifact-contracts`. The study record beside them is deliberately not validated
there: its file is the study's own local form, and the mapping onto `llb.study-design` is this
project's, exactly as the conflict bundle's integer version is.

## Validation result

On 2026-09-04 on the CUDA host (RTX PRO 3000 Blackwell Laptop, 12 GiB, driver 610.57.04), `make ci`
passed 4881 tests with 2 skipped and 50 deselected, including the 17 run-bundle contract tests and
the 4 `check-run-bundle` CLI tests, and `make check-artifact-contracts` validated the regenerated
schemas, the catalog, the ODCS projection, and the external process. The registry now generates 65
schema files across 51 families.

The migration was also read against the runs this host already holds rather than against fixtures
alone: 109 of the 110 archived `manifest.json` files under the host's run roots resolved through
`llb.run-manifest`, every one of them at version 1.0.0 migrated forward to 2.0.0, and
`check-run-bundle` read all three members of a real 66-case bundle back. The single refusal is a
`joint-search` manifest -- a search ledger, not a run bundle, sharing the filename -- which belongs
to the model and training surface the remaining `artifact-contracts` plan task covers.

The reading: no supported historical bundle on this host needs a fixture to be readable, and the
one file the contract refuses is one it is right to refuse, because it is not the record it names.
What would overturn it: a producer under the migrated surface that writes a durable record without
a registered contract. `make check-artifact-contracts` cannot see that -- the producer-side
discipline is carried by review, not by a gate, until
`artifact-contract-boundary-enforcement` lands. This capability is deterministic and service-free,
so no model was loaded and no GPU memory was allocated for the result.
