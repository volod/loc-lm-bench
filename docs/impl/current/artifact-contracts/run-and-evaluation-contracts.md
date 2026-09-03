# Run, Board, And Orchestration Contracts

Part of the [Artifact contracts](../artifact-contracts.md) area of the
[current implementation index](../../current.md).

## Delivered boundary

A run bundle is the project's primary downstream API, and every machine-readable file one is made
of is now a registered contract family: the run manifest, the per-case score row, the benchmark
cell, the retrieved-span record, the withheld-context probe row, the budget-abort record, the
resume journal row and its identity meta, and the design and analysis sidecars a study writes. The
board and orchestration records that read those bundles joined the same registry: the miss analysis
and its miss rows, the composed agent operating profile, and the auto-RAG manifest, journal event,
stage result, stage links, and recommendation. Each names a `schema_id`, one current
`schema_version`, and a strict Pydantic model, and each producer builds that model instead of
handing a `dict[str, object]` to `json.dumps`.

Out of scope and deliberately unregistered: model response text before parsing, backend logs,
rendered Markdown, and MLflow's own database schema. MLflow remains an injected best-effort mirror
that starts only after the canonical bundle is on disk.

Every family here is at its INITIAL version, `1.0.0`, with no migration: nothing has been released,
so a second version would describe a form nobody ever wrote. Where an older writer simply recorded
less, the single model says so with an optional field. The recipe for adding a version when one is
finally needed is in
[foundation and evolution](foundation-and-evolution.md#adding-a-version-when-one-is-finally-needed).

The models live in `src/llb/core/contracts/runs.py` (the manifest), `run_bundle.py` (the rows and
sidecars), and `orchestration.py` (the board and auto-RAG records). The declarations live in
`src/llb/artifacts/runs/families.py`, the directory description in `datasets.py`, the row seam in
`rows.py`, the additional-member vocabulary in `members.py`, and the readers every consumer shares
in `bundle.py`.

`RunManifest` moved out of `llb.tracking.manifest` into `llb.core.contracts.runs`, where the other
run contracts live: a family module that had to import the writer to reach its model would close a
cycle through `llb.artifacts`. `llb.tracking.manifest` now owns publication alone.

## Two shapes, and why the difference is deliberate

A record whose columns are the SAME for every producer is modelled column by column: the evaluation
case score, the retrieved-span record, the resume journal row, the abort record, the probe row.

A record whose body is the LANE'S OWN is an envelope -- identity around a body under one named
field, the same split the retrieval comparison sidecar already draws between its envelope and its
report:

| Family | Body field | Why the body is not modelled |
| --- | --- | --- |
| `llb.benchmark-cell` | `cell` | A memory-fold lane's cell is a ladder level and its band, a tool lane's is an episode and its calls, a seed row is a seed and its verdict. Naming those columns would make the contract a union of every benchmark this project will ever add |
| `llb.study-design` | `design` | A design grows a factor per added condition |
| `llb.study-analysis` | `analysis` | An analysis grows a section per added measurement, so freezing it would make every added reading a contract change |
| `llb.auto-rag-stage-result` | `result` | Each stage owns what it measured |
| `llb.auto-rag-stage-links` | `stages` | The map of stage to result, which was a bare map on disk |

The body field is also `legacy_document_field` -- the field a pre-contract file's whole content
became -- so one declaration serves the writer, the reader, and an outside consumer working from
the published catalog. A benchmark row was written flat before the envelope existed, and reads back
flat now.

## Identity lives on disk, not in a row

`llb.artifacts.runs.rows` puts identity on the FILE and takes it off as the record enters the
process, exactly as `llb.artifacts.records` already does for chunk rows and for the same reason:
every consumer of a score row keys on `objective_score`, `status`, and `item_id`, and adding two
keys to each in-memory row would change what a few hundred assertions, aggregations, and DataFrame
projections see for no reading anyone takes.

`encode_record` writes `{schema_id, schema_version, ...record}` -- or `{identity, <body field>:
record}` for an envelope family -- and VALIDATES before returning, so a producer that grew a column
its contract does not declare fails at the write rather than leaving a bundle nobody can read back.
`decode_record` validates through the registry and returns the logical record without its identity
and with the body unwrapped.

A stated `null` is dropped, at every level a contract models and inside an envelope's body. `null`
here means "this producer recorded no such thing", which is what an absent key already means to
every consumer -- `row.get("first_hit_rank")` answers `None` either way -- so a row reads the same
whether an old writer omitted the column or a current one wrote it out.

## One reader per member

Every lane that ranks a board, compares two runs item by item, exports a fine-tuning set, or
analyses misses used to open `scores.jsonl` with its own `json.loads` loop. `llb.artifacts.runs.
bundle` is now the single reader: `read_run_manifest`, `read_score_rows`, `read_retrieval_rows`,
`read_case_rows`, `read_case_series`, `read_study_design`, `read_study_analysis`. That is what
makes a refusal mean something -- a bundle from a build this one cannot read is named at the door
instead of being aggregated with the half of it this reader understands.

`read_score_rows` answers the same flat rows for an evaluation bundle and a benchmark bundle, so a
caller that only wants `objective_score` never has to know which kind it opened. `llb/board/io.py`
is gone: its readers moved to the bundle module and `mean_or_none` moved to its one caller.

## What an older bundle is missing, and who supplies it

| Family | What a bundle written earlier omits | How it reads |
| --- | --- | --- |
| `llb.case-score` | `token_precision`, `token_recall`, `ranking_score` -- 244 of the 340 evaluation bundles on this host predate the pair-based ranking score | Optional in the model; a reader that needs one reads the absence, never a default |
| `llb.case-score` | every lane column -- query preparation, the declared envelope, the ontology gate, the citation metrics | Optional; an absent column says the lane did not run |
| `llb.case-retrieval` | `duplicate_count` / `duplicate_occurrences` on an uncollapsed chunk | Optional; their absence says the chunk collapsed nothing |
| every family | the identity itself | `legacy_version` names the version to assume, which for all of these is the only version |

## Publication is behind a read-back

`persist_run` writes the manifest, the score rows, the retrieval rows, and every declared member
into the staging directory, then DESCRIBES the staged bundle, reads every member back through its
binding, and publishes `dataset_manifest.json` -- all before `staging.replace(out_dir)`. A member
that cannot be read at the current contract refuses there, while the only thing that exists is a
staging directory the caller is about to delete. A bundle that reaches `$DATA_DIR` is one this
build could read again.

The published description is also what makes a bundle self-describing: it records which score
contract the rows are bound to, which no inspection of a legacy bundle's columns can tell reliably.
A bundle written before descriptions existed states its kind in its manifest instead -- a benchmark
category run records the `category` it is a cell of and an evaluation run has none -- which
`run_bundle_kind` reads.

## No arbitrary bytes enter a bundle

`persist_run(artifacts=...)` used to take `Mapping[str, str]`: any name, any content. It now takes
`RunMember`s, and each says which registered contract validates it or that it is the one declared
exemption:

```python
artifacts=[
    study_design("second-fold-design.json", design),
    study_analysis("second-fold-analysis.json", analysis),
    table_report("second-fold.md", "Compact trigger rule through a second fold", table),
]
```

`human_report` / `table_report` are the exemption: Markdown written for a person has no machine
consumer to protect, and `table_report` also owns the fenced-table rendering every benchmark lane
was repeating inline. `member_problems` refuses an unregistered contract, a non-Markdown human
report, a path that is not a plain file name, and a duplicate name -- all of them before anything
is written.

## Refusing before the expensive step

- A bundle whose manifest this build cannot read refuses inside `read_run_manifest`, which every
  board loader, recommendation builder, fine-tuning export, and paired comparison passes through.
- `load_run_records` and `load_category_run_records` RE-RAISE an `ArtifactContractError` rather
  than logging and skipping. A board that quietly dropped a bundle a newer build wrote would rank a
  roster missing exactly those runs and read as complete.
- `refuse_unreadable_run_bundle` is the whole-bundle gate: every member read at the current
  contract AND checked against the description published with it. `load_scored_bundle` calls it
  before a miss analysis reads anything, so an analysis never mixes rows from a rewritten file.

## Inspecting a bundle without a model

```bash
llb check-run <run-dir>
llb check-run <run-dir> --kind benchmark
llb check-run <run-dir> --upgrade
```

`check-run` surveys every member -- reporting all refusals rather than stopping at the first -- and
imports no backend, no store, and no encoder. A bundle this build published is checked against
exactly the members it declared; one written earlier is described at the kind its own manifest
states, which `--kind` overrides. `--upgrade` rewrites the members an older writer produced at the
current contract; while every family here is at its initial version it therefore rewrites nothing,
and the flag is the same one `check-bundle` and `check-store` carry rather than a second
implementation of it.

The command shares its whole implementation with those two: `llb.artifacts.runs.datasets` describes
the directory, `llb.artifacts.dataset_reading` reads and surveys it, and `llb.cli.artifact_survey`
renders the report, so a run bundle's output reads exactly like a store's.

## A row member can be a bare body too

Registering the benchmark cell found a real gap in the foundation. `llb.artifacts.io` normalized a
pre-contract file only for `json` and `yaml` members, on the reasoning that "a row member is one
object per line by construction". That held while every row family's old form was already a record.
A benchmark cell's old form is the bare body, so every parsed record -- row members included --
now passes through `ContractRegistry.normalize`, which wraps only when the family declares a body
field and the value carries no identity.

## Fixtures and the gates over them

`samples/artifact_contracts/run_bundles/` holds an evaluation bundle and a benchmark bundle at the
current contracts, the pre-contract form of each under `legacy/`, and a manifest from an
unsupported future major. The pre-contract twins differ in exactly what a real older bundle differs
in: no identity anywhere, no published description, bare cells where the benchmark bundle now
writes an envelope, and bare bodies where the study sidecars now write one.

`src/llb/artifacts/runs/fixture.py` builds a complete score row, retrieval row, manifest, metrics,
telemetry report, and agent profile, so a test states only what it is about and a fixture cannot
drift away from the contract when a column joins it.

`tests/llb/artifacts/test_run_bundle_contracts.py` asserts that the current and pre-contract
bundles load to identical logical records, that a benchmark cell reads flat in both forms, that a
loaded record carries no identity, that every family is at one initial version with no migration
and no refusal, that a case row missing a required column never reaches disk, that a staged member
that cannot be read back refuses before the rename and leaves nothing behind, that a published
bundle names its own score contract, that an additional member without a registered contract or the
human-report exemption is refused, that a future major refuses at the reader and at the board, that
the whole-bundle gate reports every unreadable member at once, that a bundle published without a
description still reads, and that the resume journal and its meta read back in both forms.
`tests/llb/cli/test_cli_check_run.py` covers the operator command: what each kind reports, that a
pre-contract benchmark bundle needs no `--kind` flag, that `--upgrade` leaves a pre-contract bundle
byte-for-byte alone, the two exit codes a refusal uses, and the digest refusal on a tampered member.
`samples/artifact_contracts/external_validate.py` repeats the current-form and pre-contract
validations against the generated JSON Schemas without importing `llb`, including the bundle's own
dataset manifest, and runs inside `make check-artifact-contracts`.

## Validation result

On 2026-09-03 on the RTX 4060 Ti 16 GB CUDA host, `make ci` passed 4894 tests with 50 deselected in
241 s, including the 16 run-bundle contract tests and the 7 `check-run` CLI tests, and
`make check-artifact-contracts` validated the regenerated schemas, the catalog, the ODCS
projection, and the external process. The registry generates 59 schema files across 49 families,
18 of them added here and every one of the 18 at a single initial version.

The migration was read against the artifacts this host already held, which is the reading that
matters most. All 1017 run bundles under `$DATA_DIR` -- 340 evaluation bundles and 677 benchmark
category bundles across 17 method roots -- read at the current contracts through the whole-bundle gate,
with no rewrite; the remaining 8 `*/*/manifest.json` files under those roots are the auto-RAG and
joint-search study manifests, which are different families and are named as such rather than read
as run manifests. The single auto-RAG run on this host round-tripped its manifest, its 8 stage
results, its stage links, and its recommendation YAML; all 4 composed agent profiles round-tripped.

Reading the real bundles is what shaped three parts of the contract. `token_precision`,
`token_recall`, and `ranking_score` are optional because 244 evaluation bundles predate them. The
profile anchor's `retrieval_fingerprint` is an open KNOB MAP, not a string, because that is what
every profile on this host records and what the drift report names a moved knob out of. And the
benchmark-cell normalization gap above surfaced only when 677 real benchmark bundles were read
through the dataset reader rather than through a fixture.

This capability is deterministic and service-free, so no model was loaded and no GPU memory was
allocated for the result. What would overturn it: a run, board, or orchestration producer added
without a registered contract, which `make check-artifact-contracts` cannot see -- the producer-side
discipline is carried by review until the repository gate lands. Two families also have no real
artifact on this host to read against: `llb.miss-analysis` / `llb.miss-record` (no miss analysis has
been run here) and `llb.context-probe` (no bundle carries `probes.jsonl`), so both stand on fixtures
alone.
