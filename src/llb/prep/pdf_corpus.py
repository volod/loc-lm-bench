"""PDF corpus ingestion for local Ukrainian document collections.

The rest of the RAG/goldset pipeline consumes `.md` and `.txt` files with stable character
offsets. This module turns a local PDF directory into that canonical text corpus using
per-parser extractors. The extracted `.md` files become the source of truth for later span
validation; original PDFs are recorded only as provenance in the manifest.

The implementation lives in the `llb.prep.pdf` package -- the shared data model in `pdf.model`, one
module per parser (`pdf.pymupdf`, `pdf.docling`, `pdf.marker`, `pdf.unstructured`, `pdf.markitdown`),
the dispatcher in `pdf.dispatch`, and the orchestration in `pdf.furniture` (page-furniture
stripping), `pdf.render` (discovery + rendering), `pdf.quality` (quality scoring + parser
selection), `pdf.manifest` (manifest/citation I/O), `pdf.reuse` (unchanged-source reuse), and
`pdf.ingest` (the corpus driver). This module re-exports the package's public names so
`llb.prep.pdf_corpus.<name>` keeps working.
"""

from llb.prep.pdf.dispatch import extract_pdf_markdown  # noqa: F401  (re-exported)
from llb.prep.pdf.furniture import strip_page_furniture  # noqa: F401  (re-exported)
from llb.prep.pdf.ingest import (  # noqa: F401  (re-exported)
    _ingest_one_pdf,
    ingest_pdf_corpus,
)
from llb.prep.pdf.manifest import (  # noqa: F401  (re-exported: ontology.artifacts, tests)
    _citation_path,
    _write_citations,
)
from llb.prep.pdf.model import (  # noqa: F401  (re-exported from pdf.model)
    DEFAULT_MARKDOWN_DIRNAME,
    DOCLING_TOOL,
    MARKER_TOOL,
    MARKITDOWN_TOOL,
    PARSER_AUTO,
    PDF_CITATION_SUFFIX,
    PDF_CORPUS_MANIFEST,
    PDF_CORPUS_QUALITY,
    PDF_PARSERS,
    PDF_SUFFIX,
    PYMUPDF4LLM_TOOL,
    UNSTRUCTURED_TOOL,
    PdfCorpusItem,
    PdfCorpusResult,
    PdfDiagnostics,
    PdfExtraction,
    PdfExtractionQuality,
    PdfPageChunk,
    PdfPageCitation,
    PdfParserAttempt,
    PdfTextExtractor,
    RenderedPdfDoc,
    clean_pdf_text,
    inspect_pdf,
)
from llb.prep.pdf.render import (  # noqa: F401  (re-exported)
    default_markdown_out_dir,
    doc_id_for_pdf,
    iter_pdf_files,
)
from llb.prep.pdf.reuse import _sha256_file  # noqa: F401  (re-exported: prep.corpus_ingest)

__all__ = [
    # data model + constants (re-exported from pdf.model)
    "DEFAULT_MARKDOWN_DIRNAME",
    "DOCLING_TOOL",
    "MARKER_TOOL",
    "MARKITDOWN_TOOL",
    "PARSER_AUTO",
    "PDF_CITATION_SUFFIX",
    "PDF_CORPUS_MANIFEST",
    "PDF_CORPUS_QUALITY",
    "PDF_PARSERS",
    "PDF_SUFFIX",
    "PYMUPDF4LLM_TOOL",
    "UNSTRUCTURED_TOOL",
    "PdfCorpusItem",
    "PdfCorpusResult",
    "PdfDiagnostics",
    "PdfExtraction",
    "PdfExtractionQuality",
    "PdfPageChunk",
    "PdfPageCitation",
    "PdfParserAttempt",
    "PdfTextExtractor",
    "RenderedPdfDoc",
    "clean_pdf_text",
    "inspect_pdf",
    # extraction dispatch (re-exported from pdf.dispatch)
    "extract_pdf_markdown",
    # public orchestration
    "default_markdown_out_dir",
    "doc_id_for_pdf",
    "ingest_pdf_corpus",
    "iter_pdf_files",
    "strip_page_furniture",
]
