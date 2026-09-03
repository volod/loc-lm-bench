# Retrieval, Graph, And Prompt-System Contracts

Part of the [Artifact contracts](../artifact-contracts.md) area of the
[current implementation index](../../current.md).

## Delivered boundary

Every project-owned machine-readable artifact of the retrieval surface is a registered contract
family: the indexed chunk row and the vector-store metadata, the graph node and edge rows, the
graph store metadata and its diagnostic community summaries, the five members of a prompt-system
package, the query-preparation glossary, and the machine-readable sidecar every retrieval
comparison writes. Each names a `schema_id`, one current `schema_version`, and a strict Pydantic
model per version, and each producer builds that model instead of handing a `dict[str, object]` to
`json.dumps`.

Out of scope and deliberately unmodelled: the FAISS index, the BM25 postings file, the adapter
backends' vector matrix, the conflict-detection semantic-tree sidecar, and a graph DuckDB
database. Those formats belong to their owners, so a store binds them OPAQUELY -- by digest, and by
the owner and format version that a digest alone cannot supply -- rather than pretending to model
their bytes. Also unregistered: Markdown reports, which are written for people.

Every one of them is at its INITIAL version, `1.0.0`, with no migration: nothing here has been
released, so a second version would describe a form nobody ever wrote. Where an older writer simply
recorded less, the single model says so with an optional field rather than a second version. The
recipe for adding a version when one is finally needed is in
[foundation and evolution](foundation-and-evolution.md#adding-a-version-when-one-is-finally-needed).

The models live under `src/llb/core/contracts/retrieval/` (`store.py`, `graph.py`,
`prompt_system.py`, `comparison.py`, `query.py`); the declarations live in
`src/llb/artifacts/retrieval/families.py`, the directory descriptions in `datasets.py`, and the
encode/decode seam in `llb/artifacts/records.py`. The two foundation families that grew a key here
-- `llb.dataset-manifest` and `llb.artifact-catalog` -- each stayed at `1.0.0`: the manifest had no
released form to preserve, and the catalog is regenerated from the registry on every check.

## Identity lives on disk, not in a chunk

A chunk row and a graph node row are handed to retrieval as plain mappings, and every downstream
reader -- metadata filters, fusion, the span metrics, the comparison lanes -- keys on the fields
the chunker wrote. So `llb.artifacts.records` puts identity on the FILE and takes it off
again as the record enters the process: `encode` writes `{schema_id, schema_version, ...record}`,
and `decode` validates through the registry, migrating an older row on the way, and returns the
logical record without its identity.

`decode` also drops a stated `null`. In these families `null` means "this store recorded no such
thing", which is what an absent key already means to every reader here, so a record reads the same
whether an old producer omitted the key or a current one wrote it out as null.

Exactly one place per file knows this. `read_store_chunks` and `read_store_meta`
(`llb.rag.vector_store.persistence`) are where the incremental refresh, the conflict tiers, and the
duplicate-residue report open a store; `read_node_rows`, `read_edge_rows`, `read_graph_meta`, and
`read_community_summaries` (`llb.graph.store_io`) are where a graph is opened. `load_extractions`
and `load_ontology` (`llb.graph.ingest`) read the graph's build inputs the same way -- they used to
validate a `DocExtraction` with extras IGNORED, which is exactly how a newer bundle's added fields
would have gone missing without a word.

## Reading a file that is not a record

Some pre-contract files are not objects at all: an anthology was a bare array of passages, a
community-summary file was the bare `{community_id: summary}` map, a comparison sidecar was the
measurement body alone. None of them has anywhere to put an identity, and `legacy_version` alone
cannot help -- there is no record to stamp.

One old form is not a whole file but a field: a prompt-system candidate with no knowledge tree has
always written the EMPTY object, and every tree-enabled grid renders such controls. That is a
declared alternative in the candidate contract rather than a reader's fixup, so the generated JSON
Schema accepts it too and an outside reader needs no Python of ours to validate an archived
package.

`ContractDefinition.legacy_document_field` closes the whole-file gap. It names the field of the current
record that such a file's whole content became, and `ContractRegistry.normalize` wraps it before
dispatch. One declaration serves three readers: the domain reader, the dataset reader, and an
outside reader working from the published catalog, which is why `legacy_document_field` joined the
catalog entry beside `legacy_read_version`.

## What an older file is missing, and who supplies it

No family here migrates, so the question a migration would have answered is answered by the model
instead: which fields could an older writer not have recorded, and what does a reader do about it.

| Family | What an older file omits | How it reads |
| --- | --- | --- |
| `llb.rag-store-meta` | `collapse_duplicates` and `duplicate_tier` -- stores built before the knobs existed are still on disk | Optional in the model; the refresh reads collapse as on and the residue report reads the `exact` tier, exactly as they always have |
| `llb.rag-store-meta` | `backend`, and the corpus, governance, lexical, and duplicate blocks | Optional; `load_store` reads the backend as FAISS, and the rest are genuinely absent readings, not defaults |
| `llb.rag-chunk` | `overlap` on a `parent_child` child, `parent_id` on a flat chunk | Optional; neither is a default, each is a property of the chunk |
| `llb.prompt-system-candidates` | a knowledge tree, written as the EMPTY object by every no-tree control | A declared alternative in the contract, so the generated schema accepts it too |
| every family | the identity itself | `legacy_version` names the version to assume, which for all of these is the only version |

The one thing a single version cannot absorb is a field that changes MEANING or a required field an
older writer could not have produced. Neither has happened here; when one does, the recipe in
[foundation and evolution](foundation-and-evolution.md#adding-a-version-when-one-is-finally-needed)
says what to declare.

## Binding what this project does not own

`OpaqueBinding` is a new field on the dataset member: an `owner`, a `format_version`, and a
description. It is a required part of an opaque member, so a member with no record contract and no
named owner is refused when the manifest is built rather than described as a shrug.
`llb.artifacts.retrieval.datasets` declares one per opaque member of a store or graph --
`index.faiss` to `faiss@IndexFlatIP/1`, `lexical_index.json` to the tokenizer version its postings
ARE (`bm25-uk-v2`), each adapter backend's `vectors.npy` to `numpy@npy/1`, the semantic-tree
sidecar to `llb.conflicts.semantic_tree` at its own tree version, and `graph.duckdb` to `duckdb`.

That binding is also what makes an index checkable at all. A vector index has no identity of its
own, so the digest recorded beside the chunk rows is the only thing that says the index next to
them is the one built from them. `save_store` therefore publishes `dataset_manifest.json` as its
last act; a graph publishes its own from `save_graph_inputs`, because the build inputs a refresh
chains from are written after the store and a manifest that predates half a directory describes
nothing useful. `RagStore.load` and `GraphStore.load` then call `refuse_tampered_dataset` before
reading anything. A directory published without a manifest is not a refusal -- every store written before
this existed is such a directory -- but one whose manifest disagrees with its bytes is.

The manifest describes the members present when the directory was published. A sidecar another
tool writes later (the conflict semantic tree is the real case) is simply not among them; the live
description that `check-store` builds does include it.

## Refusing before the expensive step

- A store or graph whose metadata this build cannot read refuses inside `read_store_meta` /
  `read_graph_meta`, which every load path passes through -- so a store from a future major is
  named at the door instead of retrieving with the half of it this reader can see.
- A store or graph whose members no longer match its published manifest refuses in
  `refuse_tampered_dataset`, before a query runs.
- A prompt-system package refuses in `refuse_unreadable_prompt_system`, called from
  `resolve_prompt_package` before one candidate is taken out of it: the manifest a benchmark
  records as provenance is read as part of the same check.

## Inspecting a store without a model

```bash
make check-store STORE=<index-dir>
make check-store STORE=<graph-dir> STORE_KIND=graph
make check-store STORE=<prompt-system-run-dir> STORE_KIND=prompt-system
make check-store STORE=<index-dir> STORE_UPGRADE=1
```

`check-store` surveys every member -- reporting all refusals rather than stopping at the first --
and reports opaque members by owner and format version rather than by record count. It imports no
FAISS, no DuckDB, and no encoder, so a store can be inspected on a machine that could not query it.
`--upgrade` rewrites the members an older writer produced at the current contract; while every
retrieval family is at its initial version it therefore rewrites nothing, and the flag is there as
the same one `check-bundle` uses rather than as a second implementation of it. The underlying
command is `llb check-store <dir> --kind store|graph|prompt-system [--upgrade]`.

The command shares its whole implementation with `check-bundle`
([data-prep contracts](data-prep-contracts.md)): `llb.artifacts.datasets` describes a directory,
`llb.artifacts.dataset_reading` reads, surveys, and upgrades it, and `llb.cli.artifact_survey`
renders the report, so a store's output reads exactly like a bundle's.

## The comparison sidecar owns its envelope, not its body

Every retrieval comparison writes a Markdown report for a person and a JSON sidecar for everything
else. `llb.retrieval-comparison` wraps that sidecar in a versioned envelope -- identity, which
`kind` of reading it is (`comparison`, `calibration`, `probe`, `validation`, `run-config`), and the
command that produced it -- and carries the measurement under `report`.

The body is deliberately not modelled. Its shape is the lane's own: rows are named by swept
parameters, slices by the corpus's question types, and a new lever adds a section, so freezing it
would make every added measurement a contract change. What the envelope fixes is what a reader
needs before opening anything. `llb.rag.comparison.sidecar` is the single writer and reader;
`lane_labels_from_comparison` reads a recorded sweep verdict through it, and an archived
pre-envelope sidecar reads back with `produced_by` stated as `unrecorded` rather than invented.

## Fixtures and the gates over them

`samples/artifact_contracts/retrieval_graph/` holds a store, a graph, and a prompt-system package
at the current contracts, the pre-contract form of each under `legacy/`, and store and graph
metadata from an unsupported future major under `unsupported-future/`. A pre-contract fixture
differs from its current twin in exactly what a real older file differs in: no identity anywhere,
bare arrays where the package now writes documents, an empty knowledge-tree object on the no-tree
control, and no duplicate-collapse knobs on the store meta. The store fixture's
`index.faiss` and `lexical_index.json` are placeholder text: their bytes are never parsed, which is
the point of an opaque binding, and committing them is what lets the manifest's digest check and
the owner declaration be exercised without FAISS.

`tests/llb/artifacts/test_retrieval_graph_contracts.py` asserts that the current and pre-contract
store, graph, and package load to identical logical records, that a store which recorded no
duplicate knobs reads as having recorded none while agreeing on everything it did record, that a
loaded chunk carries no identity, that every family is at one initial version with no migration and
no refusal, that opaque members name their owner and format version, that a changed index refuses
on its digest while a store with no manifest does not, that a future major refuses at both metadata
readers and at the prompt-system gate, that a pre-contract store needs no upgrade and `--upgrade`
leaves it byte-for-byte alone, and that a sidecar and a glossary round-trip through their envelopes
in both forms. `samples/artifact_contracts/external_validate.py` repeats the current-form and pre-contract
validations against the generated JSON Schemas without importing `llb`, including the store's own
dataset manifest, and runs inside `make check-artifact-contracts`.
`tests/llb/cli/test_cli_check_store.py` covers the operator command: what each kind reports, what
`--upgrade` rewrites and what it leaves byte-for-byte alone, the two exit codes a refusal uses, and
that inspecting a store and a graph leaves `faiss`, `duckdb`, and `sentence_transformers` unimported.

## Validation result

On 2026-09-03 on the RTX 4060 Ti 16 GB CUDA host, `make ci` passed 4863 tests with 50 deselected
in 243 s, including the 19 retrieval, graph, and prompt-system contract tests and the 7
`check-store` CLI tests, and `make check-artifact-contracts` validated the regenerated schemas, the
catalog, the ODCS projection, and the external process. The registry generates 41 schema files
across 31 families, 13 of them added here and every one of the 13 at a single initial version.

The migration was also read against the artifacts this host already held, which is the reading that
matters most: every pre-contract vector store under `$DATA_DIR` (`check-store` reports 3 to 7
members each, depending on whether the store carries a lexical index and a conflict semantic-tree
sidecar), a 423-node/242-edge graph store, and two 36-candidate prompt-system packages. All of them
read at the current contract with no rewrite, including the four stores that recorded no
duplicate-collapse knobs -- which is why those two fields are optional rather than a second version.
The prompt-system packages are why the candidate contract declares the empty knowledge-tree object
as an alternative and carries `baseline_prompt_system_id`: 18 of those 36 candidates are no-tree
controls that wrote the empty object, and the other 18 name the control they are read against --
neither was in the first model, and both were found by reading a real package rather than a fixture.

This capability is deterministic and service-free, so no model was loaded and no GPU memory was
allocated for the result. What would overturn it: a retrieval, graph, or prompt-system producer
added without a registered contract, which `make check-artifact-contracts` cannot see -- the
producer-side discipline is carried by review, not by a gate. The measurement body of a comparison
sidecar is also outside the contract by design, so a change to a lane's report shape is a change no
version says anything about.
