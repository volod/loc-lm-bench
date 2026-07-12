"""PDF extraction backends for the local Ukrainian document corpus.

The per-parser extractor modules (`pymupdf`, `docling`, `marker`, `unstructured`, `markitdown`)
build on the shared data model in `model` and are dispatched by `dispatch.extract_pdf_markdown`.
The orchestration (parser selection, rendering, manifest/citation I/O, reuse) lives in the parent
`llb.prep.pdf_corpus`, which re-exports the public names here.
"""
