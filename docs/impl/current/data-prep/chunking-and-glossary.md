# Chunking And Query Glossary Handoff

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

## Chunking

The `src/llb/rag/chunking/` package keeps every chunk offset-exact. Strategies:

- `fixed`: dependency-free fixed windows;
- `sentence`: dependency-free sentence-aware chunks;
- `recursive`: LangChain recursive splitter when available, pure fallback otherwise;
- `markdown`: heading-aware chunks with breadcrumb metadata;
- `semantic`: pinned-embedder breakpoints while preserving source offsets;
- `page`: PDF page/citation-aware boundaries that never cross a page-sidecar span;
- `heading`: heading-hierarchy packing with heading lines kept in the chunk text;
- `late`: sentence spans embedded by whole-document token pooling (late chunking);
- `table`: markdown-table-aware boundaries -- a chunk never cuts a table ROW, and a split table
  records its header row's offsets in `metadata.table_header_span`.

The `page`/`heading`/`late`/`table` details, comparison command, and durable evidence live in the
[RAG core](../rag-core.md) chunking-strategies section.

```bash
make build-rag-store
python -m llb.rag.chunking --corpus-root <dir> --out-dir .data/llb/rag \
  --strategy markdown --size 800 --overlap 120 --embed
```

Production RAG indexes are built through `llb build-index` or `make build-index`.

## Query Glossary (uk-query-processing)

`llb build-query-glossary --bundle <draft>` (or `make build-query-glossary BUNDLE=<draft>`) turns a
draft bundle's `prompt_dictionary_candidates.jsonl` into a `query_glossary.json` for the query-side
`glossary` step (`src/llb/rag/query_prep/glossary.py` `build_glossary_from_candidates`). Each candidate
`term` becomes a canonical entry carrying its recorded aliases plus a romanized Latin variant
(`--no-transliterations` disables the seeding); entries are sorted by canonical term for a
deterministic artifact. Hand-add more surzhyk / transliteration aliases by editing the emitted JSON
-- the `glossary` retrieval step appends every surface form of any entry the query triggers, never
mutating the stored corpus. A committed fixture lives at `samples/query-prep/` (dictionary
candidates + the generated `query_glossary.json`). The lane's retrieval behavior, A/B report, and
durable deltas live in the [RAG core](../rag-core.md) query-side-processing section.
