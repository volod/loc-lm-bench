"""Load a rendered vault and the book-ocr run that produced it."""

import json
from pathlib import Path

from llb.prep.book.config import read_config
from llb.prep.book.ledger.manifest import read_manifest
from llb.prep.book.render.constants import PDF_PAGE_FRAGMENT, RENDER_MANIFEST_FILENAME
from llb.prep.book.structure.io import read_structure, structure_path
from llb.prep.book.validate.text import content_docs, extract_links, load_docs, toc_doc
from llb.prep.book.validate.types import BookContext


def load_context(book_dir: Path, run_dir: Path | None = None) -> BookContext:
    """The vault, its notes, and the run. Refuse a directory that is not a vault."""
    book = book_dir.resolve()
    manifest = _render_manifest(book)
    run = locate_run(book, run_dir, manifest)
    config = read_config(run)
    structure = read_structure(structure_path(run))
    if structure.source_sha256 != config.source.sha256:
        raise ValueError("document structure is for a different source PDF")
    fragment = manifest.get("pdf_page_fragment") or PDF_PAGE_FRAGMENT
    docs = load_docs(book)
    noted = [doc for doc in docs if not doc.rel.startswith("metadata/")]
    return BookContext(
        book_dir=book,
        run_dir=run,
        config=config,
        structure=structure,
        manifest=read_manifest(run),
        fragment_name=str(fragment),
        docs=docs,
        toc=toc_doc(docs),
        content=content_docs(docs),
        links=[link for doc in noted for link in extract_links(doc, book)],
    )


def locate_run(book_dir: Path, explicit: Path | None, manifest: dict[str, object]) -> Path:
    """The run recorded in the vault, the parent that holds it, or `--run`."""
    if explicit is not None:
        return _require_run(explicit)
    recorded = manifest.get("run_dir")
    if isinstance(recorded, str) and recorded:
        candidate = Path(recorded)
        if (candidate / "config.json").is_file():
            return _require_run(candidate)
    sha = manifest.get("source_sha256")
    for parent in book_dir.parents:
        if not (parent / "config.json").is_file():
            continue
        try:
            config = read_config(parent)
        except ValueError:
            continue
        if not isinstance(sha, str) or config.source.sha256 == sha:
            return parent
    raise ValueError(f"{book_dir} does not name a book-ocr run; pass --run")


def _render_manifest(book_dir: Path) -> dict[str, object]:
    path = book_dir / "metadata" / RENDER_MANIFEST_FILENAME
    if not path.is_file():
        raise ValueError(
            f"{book_dir} is not a reading vault (no metadata/{RENDER_MANIFEST_FILENAME})"
        )
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not readable JSON") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a JSON object")
    return loaded


def _require_run(path: Path) -> Path:
    read_config(path)
    return path.resolve()
