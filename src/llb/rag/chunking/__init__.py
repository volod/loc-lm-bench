"""Build a RAG store from documents using different chunking strategies.

Every strategy returns chunks anchored to `doc_id` + character offsets, so retrieval can be
scored against source-span gold labels by overlap (consistent with `llb.goldset.schema`).
That offset invariant is the constraint on which splitters we can reuse.

Strategies:
  - fixed      pure-Python fixed character window with overlap (zero deps)
  - sentence   pure-Python: pack whole sentences up to ~size (never cut mid-sentence)
  - recursive  langchain `RecursiveCharacterTextSplitter` (add_start_index -> exact offsets)
  - markdown   structure-aware: headers parsed from the SOURCE (offset-exact) + recursive
               sub-split of long sections; header breadcrumbs go into chunk `metadata`
  - semantic   native: embed sentences with the PINNED embedder, break at distance spikes
               (offset-exact; langchain's SemanticChunker does not preserve source offsets)
  - page       PDF page/citation-aware: chunk boundaries never cross a `*.citations.json`
               page-sidecar span (see `llb.rag.page_metadata`); pages longer than `size`
               are sub-split WITHIN the page; docs without a sidecar fall back to recursive
  - heading    heading-hierarchy (layout-aware): a whole heading subtree that fits `size`
               becomes ONE chunk (heading lines INCLUDED in the text, unlike `markdown`);
               oversized subtrees recurse into child headings; every chunk carries the full
               breadcrumb in `metadata.headers`
  - late       late chunking: spans are IDENTICAL to `sentence` (so any retrieval delta
               isolates the embedding effect), but vectors are pooled from whole-document
               token embeddings (`llb.rag.late_encoding`) instead of per-chunk encoding

Submodules (import from the specific one you need -- there is no re-export surface):
  - `spans`      primitive fixed/sentence span helpers and shared validation
  - `recursive`  the pinned langchain recursive splitter lane
  - `structure`  markdown / heading / page structure-aware strategies + page sidecar lookup
  - `semantic`   native semantic chunking
  - `dispatch`   the `STRATEGIES` registry and the `chunk_spans` dispatcher
  - `corpus`     `iter_docs` / `chunk_text` / `chunk_corpus` / `summarize` over a corpus tree
  - `build`      FAISS index building and the `python -m llb.rag.chunking` CLI

`recursive` (and the `markdown` sub-split) use `langchain-text-splitters`, pinned in the base
dependencies so chunk boundaries are reproducible across environments; a missing or
version-mismatched install fails loudly rather than silently rechunking. `semantic` needs the
pinned embedder from the `[rag]` extra, lazily imported.
"""
