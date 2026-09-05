# Retrieval, Graph, And Prompt-System Artifact Contracts

Part of the [artifact contracts](../artifact-contracts.md) area of the
[current implementation index](../../current.md). The mechanism these families use is recorded in
[foundation and evolution](foundation-and-evolution.md); the corpus, gold, and drafting families
they consume are in [data-prep contracts](data-prep-contracts.md).

## Delivered boundary

A built vector store, a built graph, and a prepared prompt-system package are each a SET of files
that only mean something together, and every project-owned member of the three is now a registered
contract family. So are the two comparison sidecars an adoption decision is later re-read from.

| Family | Member | Current version |
| --- | --- | --- |
| `llb.rag-chunk` | `chunks.jsonl`, `parents.jsonl` | 1.0.0 |
| `llb.rag-store-meta` | `store_meta.json` | 2.0.0 |
| `llb.graph-node` | `nodes.jsonl` | 1.0.0 |
| `llb.graph-edge` | `edges.jsonl` | 1.0.0 |
| `llb.graph-meta` | `graph_meta.json` | 1.0.0 |
| `llb.graph-community-summaries` | `community_summaries.json` | 1.0.0 |
| `llb.prompt-system-manifest` | `manifest.json` | 1.0.0 |
| `llb.prompt-system-anthology` | `anthology.json` | 1.0.0 |
| `llb.prompt-system-doc-metadata` | `doc_metadata.json` | 1.0.0 |
| `llb.prompt-system-mapping` | `graph_rag_mapping.json` | 1.0.0 |
| `llb.prompt-system-candidates` | `candidates.json` | 1.0.0 |
| `llb.retrieval-comparison` | `compare-retrieval --out` | 1.0.0 |
| `llb.fusion-routing-calibration` | `calibration.json` | 1.0.0 |

The models live under `src/llb/core/contracts/retrieval_graph/` (`stores.py`, `graph.py`,
`prompt_system.py`, `statistics.py`, `comparison.py`, `calibration.py`); the declarations, the
readers and writers, the dataset descriptors, and the load-time gate live under
`src/llb/artifacts/retrieval_graph/`.

Deliberately unregistered and out of scope: the FAISS index, the per-backend `vectors.npy` matrix,
the BM25 posting list, a materialized DuckDB database, and the embedding model itself. This project
stores those files but does not define their bytes, so it binds them instead (below) rather than
modelling a layout their own library owns.

## Naming what this project does not own

`llb.dataset-manifest` is at 1.1.0, with 1.0.0 read-and-migrate. The one thing version 1 could not
say is whose format an opaque member is written in, so version 1.1 adds `opaque_binding` -- an
`owner`, that owner's own `format` and `format_version`, and a description -- and requires it on
every opaque member while forbidding it on a structured one. The migration carries every structured
member forward untouched; a version 1 manifest that DID bind an opaque member has no owner to
carry, so the restated record fails validation and the read refuses with that reason rather than
inventing one.

A store records the same binding for itself. `store_meta.json` carries an `index_members` list --
member id, path, owner, the owner's format and version, a SHA-256 digest, and a byte count -- built
at publication from the files that were just written. `save_store` fills it from
`VECTOR_INDEX_OWNERS` (FAISS for the default backend, numpy for each platform-matrix adapter's
`vectors.npy`) plus the lexical index at its tokenizer version, so the member names the library
that wrote it and the version that library was.

## Refusing before the query

`refuse_unreadable_store` runs inside `load_store`, before a vector backend is imported. It asks
one question in two halves:

- the metadata resolves through the registry -- a record from a future major validates as JSON,
  looks like the family it names, and hides every field a newer writer added;
- every declared index member is present and still hashes to the digest recorded when the
  generation was published -- a member truncated, swapped, or rebuilt since then no longer matches
  the rows beside it.

A directory member (an adapter backend persists one) is hashed over its whole file tree, so the
same check covers both shapes. A store with no metadata at all is not a refusal: the caller that
requires one fails with its own missing-store error.

## Two encodings that stay as they are

Two member families keep the compact form they have always been written in, for the same reason
the conflict bundle does in [data-prep contracts](data-prep-contracts.md#two-encodings-that-stay-as-they-are):
their bytes are load-bearing.

- **Chunk and parent rows.** A store holds hundreds of thousands of them, and stamping an identity
  on every line would multiply the file for no reader. The producer validates each row against
  `llb.rag-chunk` and then writes the row itself; the store's dataset manifest binds the member at
  its version, which is how an external reader supplies one.
- **Graph node and edge rows.** The same, at the smaller scale of tens of thousands of records.

Everything else is a single document and carries its own identity. Three of those documents were
bare JSON arrays or a bare mapping -- shapes that cannot carry an identity at all -- so each is now
published as a document whose one field holds that list or mapping, and each reader accepts both
forms. A published document always names its identity and a bare one never can, so `schema_id` is
what tells them apart, rather than a field name a salient term could collide with.

## The one migration that preserves a reading

| Family | Old form | What the migration states |
| --- | --- | --- |
| `llb.rag-store-meta` | `1.0.0`: `collapse_duplicates`, `duplicate_tier`, and `index_members` absent | Both knobs stated at the constants `store_refresh` and `duplicate-residue` were already defaulting, and no index members declared |

`index_members` is empty after the migration because an older generation never recorded which
opaque files it was built with. A reader treats that as "this generation does not state its index
members", never as "it has none": the digest gate simply has nothing to check.

## Generations as datasets

`src/llb/artifacts/retrieval_graph/datasets.py` describes a vector-store generation, a graph
generation, and a prompt-system package as `llb.dataset-manifest` records. Members are DISCOVERED,
so a flat store has no parents member and says so by omission, while a member that is present and
unreadable is a refusal. A store's opaque members come from its own metadata rather than from the
directory -- which file the index lives in depends on the backend the store was built with, and
the store is the thing that knows -- and each is bound at the digest the generation PUBLISHED, so
the survey is a tamper check rather than a restatement of what the bytes hash today.

```bash
make check-generation GENERATION=<store-dir>
make check-generation GENERATION=<graph-dir> GENERATION_KIND=graph
make check-generation GENERATION=<package-dir> GENERATION_KIND=prompt-system
```

`check-generation` resolves the live generation under a base directory (a refresh publishes
immutable `generations/<ts>/` children), surveys every member -- reporting all refusals rather than
stopping at the first -- and exits 1 when any refuses, 2 when the directory holds no registered
member. The underlying command is `llb check-generation <dir> --kind store|graph|prompt-system`.

## Fixtures and the gates over them

`samples/artifact_contracts/retrieval_graph/` holds the same three generations written twice:
`current/` as this build publishes them and `legacy/` as this project wrote them before the
registry existed, plus a store metadata record from an unsupported future major and the two
comparison sidecars.

`tests/llb/artifacts/test_retrieval_graph_contracts.py` asserts that the pre-contract store
metadata reads back as the record a current writer produces (differing only in the two absences it
could not record), that chunk rows read identically from both generations, that a future major and
a changed or missing index member each refuse before query execution, that the store dataset binds
every member and names each opaque owner, that the graph and prompt-system pairs read to identical
records, and that a sidecar reads the same with and without its identity.
`tests/llb/cli/test_cli_check_generation.py` covers the operator command: what a current store
reports, what version a pre-contract one would be read at, and the two exit codes a refusal uses.
`samples/artifact_contracts/external_validate.py` repeats the current, bound-row, and pre-contract
validations against the generated JSON Schemas without importing `llb`, and runs inside
`make check-artifact-contracts`.

## Validation result

On 2026-09-04 on the CUDA host, `make ci` passed 4860 tests with 2 skipped and 50 deselected,
including the 19 retrieval/graph contract tests and the 5 `check-generation` CLI tests, and
`make check-artifact-contracts` validated the regenerated schemas, the 1.1.0 catalog, the ODCS
projection, and the external process. The registry now generates 44 schema files across 31
families. A real FAISS store built on this host recorded its index member as `faiss 1.14.3` and
its posting list at tokenizer version `bm25-uk-v2`, and `check-generation` read all five members
back. This capability is deterministic and service-free, so no model was loaded and no GPU memory
was allocated for it. What would overturn it: a retrieval, graph, or prompt-system producer added
without a registered contract, which `make check-artifact-contracts` cannot see -- the
producer-side discipline is carried by review, not by a gate.
