"""Write the single PDF page an engine that reads documents asks for.

The file is a working input, deleted before the stage commits: the run already holds the source.
"""

from pathlib import Path
from typing import Any


def write_page_pdf(document: Any, pdf_page: int, dest: Path) -> None:
    """One-page PDF of `pdf_page` (1-based) extracted from an open PyMuPDF document."""
    import fitz

    out = fitz.open()
    try:
        out.insert_pdf(document, from_page=pdf_page - 1, to_page=pdf_page - 1)
        out.save(dest, garbage=3, deflate=True)
    finally:
        out.close()
