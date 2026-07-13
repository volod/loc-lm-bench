"""Read-only corpus, cross-check, retrieval-rank, and page-citation context."""

import json
from pathlib import Path

from llb.goldset.verify_base import CONTEXT_CHARS, CROSS_CHECK_SUFFIX, RETRIEVAL_RANK_SOURCES
from llb.rag.page_metadata import intersect_pages, load_page_citations


def load_cross_check(bundle: Path) -> dict[str, dict[str, object]]:
    """Index cross-check verdicts in a bundle by item id."""
    verdicts: dict[str, dict[str, object]] = {}
    for path in sorted(Path(bundle).glob(f"*{CROSS_CHECK_SUFFIX}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for verdict in payload.get("verdicts", []):
            item_id = verdict.get("item_id")
            if item_id:
                verdicts[str(item_id)] = verdict
    return verdicts


def corpus_window(text: str, char_start: int, char_end: int, ctx: int = CONTEXT_CHARS) -> str:
    """Render a cited span, delimited by >>> and <<<, in surrounding corpus text."""
    lo = max(0, char_start - ctx)
    hi = min(len(text), char_end + ctx)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return (
        f"{prefix}{text[lo:char_start]}>>>{text[char_start:char_end]}<<<{text[char_end:hi]}{suffix}"
    )


def corpus_text(corpus_root: Path, doc_id: str, cache: dict[str, str | None]) -> str | None:
    if doc_id not in cache:
        path = corpus_root / doc_id
        cache[doc_id] = path.read_text(encoding="utf-8") if path.is_file() else None
    return cache[doc_id]


def load_retrieval_ranks(bundle: Path) -> dict[str, int]:
    """Index positive per-item retrieval ranks from known bundle sidecars."""
    ranks: dict[str, int] = {}
    for name in RETRIEVAL_RANK_SOURCES:
        path = Path(bundle) / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line.strip()) if line.strip() else {}
            except json.JSONDecodeError:
                continue
            item_id = row.get("id")
            rank = row.get("retrieval_rank")
            if item_id and isinstance(rank, int) and rank > 0:
                ranks[str(item_id)] = rank
    return ranks


def page_citation_for_span(
    corpus_root: Path,
    doc_id: str,
    char_start: int,
    char_end: int,
    cache: dict[str, tuple[str | None, list[dict[str, object]]] | None],
) -> str:
    """Render a source PDF page citation when a sidecar covers the span."""
    if doc_id not in cache:
        cache[doc_id] = load_page_citations(Path(corpus_root), doc_id)
    cite = cache[doc_id]
    if cite is None:
        return ""
    source, spans = cite
    pages = intersect_pages(char_start, char_end, spans)
    if not pages:
        return ""
    label = f"p.{pages[0]}" if pages[0] == pages[-1] else f"p.{pages[0]}-{pages[-1]}"
    return f"{Path(source).name if source else ''} {label}".strip()
