# PDF Corpus Prep

Use citation-preserving conversion before indexing, ontology drafting, or GraphRAG runs:

```sh
make pdf-to-markdown
make pdf-to-markdown PDF_DIR=<pdf-dir> PDF_OUT_DIR=<out-dir> PDF_MIN_CHARS=500 PDF_PARSER=auto
```

`PDF_DIR` defaults to `$DATA_DIR/quickstart-pdf-corpus`, which is
`.data/quickstart-pdf-corpus` unless `.env` overrides `DATA_DIR`. When `PDF_OUT_DIR` is omitted,
markdown files, page citation sidecars,
`pdf_corpus_manifest.json`, and `pdf_corpus_quality.json` are written to `<pdf-dir>/_md`; for
example, `.data/quickstart-pdf-corpus/_md`. `PDF_PARSER=auto` uses PyMuPDF4LLM for born-digital
PDFs and Docling OCR for image-only scans when the `pdf-quality` extra and OCR apt packages are
installed.
A scanned book that needs page-level recovery rather than one flat document per PDF goes through
the resumable [book-ocr run](book-ocr.md) instead.
For a full `.data/quickstart-pdf-corpus` example with Gemma 4 and corpus-specific artifact paths,
see
[Quickstart PDF corpus](../quickstart/quickstart-pdf-corpus.md).
