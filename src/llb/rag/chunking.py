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

`recursive` (and the `markdown` sub-split) use `langchain-text-splitters`, pinned in the base
dependencies so chunk boundaries are reproducible across environments; a missing or
version-mismatched install fails loudly rather than silently rechunking. `semantic` needs the
pinned embedder from the `[rag]` extra, lazily imported.

CLI (also `make build-rag-store`):
    python -m llb.rag.chunking --corpus-root samples/corpus --out-dir .data/llb/rag \\
        --strategy all --size 800 --overlap 120
Add `--embed` (needs `[rag]`) to also build a FAISS index per strategy.
"""

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llb.core.contracts import ChunkRecord, ChunkSummary, JsonObject
from llb.prep.corpus_governance import manifest_governance_by_doc

PURE_STRATEGIES = ("fixed", "sentence")
STRATEGIES = ("fixed", "sentence", "recursive", "markdown", "semantic", "page", "heading", "late")

_TERM = re.compile(r"[.!?…]+")
_CLOSERS = '”»")]’'
_LOG = logging.getLogger(__name__)


def validate_chunking(size: int, overlap: int) -> None:
    """Validate invariants shared by every chunking implementation."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) char spans for sentences, covering the whole text."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _TERM.finditer(text):
        end = m.end()
        while end < len(text) and text[end] in _CLOSERS:
            end += 1
        if end >= len(text) or text[end].isspace():
            if text[start:end].strip():
                spans.append((start, end))
            start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans


def fixed_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    validate_chunking(size, overlap)
    step = size - overlap
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        spans.append((i, min(n, i + size)))
        if i + size >= n:
            break
        i += step
    return spans


def _pack(spans: list[tuple[int, int]], size: int) -> list[tuple[int, int]]:
    """Greedily merge adjacent spans while the merged span fits within `size`."""
    out: list[tuple[int, int]] = []
    cur_start = cur_end = None
    for start, end in spans:
        if cur_start is None:
            cur_start, cur_end = start, end
        elif end - cur_start <= size:
            cur_end = end
        else:
            assert cur_end is not None
            out.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_start is not None:
        assert cur_end is not None
        out.append((cur_start, cur_end))
    return out


def sentence_chunk_spans(text: str, size: int) -> list[tuple[int, int]]:
    return _pack(sentence_spans(text), size)


# Chunk boundaries are part of the index contract, so the recursive splitter is a pinned base
# dependency (see `dependencies` in pyproject.toml). Keep this in lockstep with that pin -- a
# missing or version-drifted install fails loudly here instead of silently rechunking.
_REQUIRED_TEXT_SPLITTERS = "1.1.2"
_recursive_splitter_cls: Any = None


def _require_recursive_splitter() -> Any:
    """Return the pinned `RecursiveCharacterTextSplitter`, failing early on a bad install."""
    global _recursive_splitter_cls
    if _recursive_splitter_cls is not None:
        return _recursive_splitter_cls
    try:
        from importlib.metadata import version

        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:  # a required base dependency is missing
        raise RuntimeError(
            "recursive/markdown chunking requires `langchain-text-splitters"
            f"=={_REQUIRED_TEXT_SPLITTERS}` (a base dependency); reinstall with "
            "`uv pip install -e .`."
        ) from exc
    found = version("langchain-text-splitters")
    if found != _REQUIRED_TEXT_SPLITTERS:
        raise RuntimeError(
            f"langchain-text-splitters {found} is installed, but chunk boundaries are pinned to "
            f"{_REQUIRED_TEXT_SPLITTERS}. Reinstall the pinned version so indexes stay reproducible."
        )
    _recursive_splitter_cls = RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter


def recursive_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Offset-exact spans from the pinned langchain `RecursiveCharacterTextSplitter`.

    Every span is verified to reproduce its exact source slice, so a splitter that ever emits a
    non-slice chunk raises here rather than letting misaligned offsets reach the index.
    """
    splitter = _require_recursive_splitter()(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True
    )
    spans: list[tuple[int, int]] = []
    for doc in splitter.create_documents([text]):
        content = doc.page_content
        start = doc.metadata["start_index"]
        end = start + len(content)
        if text[start:end] != content:
            raise ValueError(
                "recursive splitter produced a chunk that is not an exact source slice; "
                "refusing to index misaligned offsets."
            )
        spans.append((start, end))
    return spans


_MD_HEADER = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.M)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def markdown_spans(text: str, size: int, overlap: int) -> list[tuple[int, int, JsonObject]]:
    """Structure-aware split on markdown headers; header breadcrumbs land in metadata.

    Headers are parsed from the SOURCE so every span is an exact source substring. (langchain's
    MarkdownHeaderTextSplitter rejoins section content and loses offsets, which would break the
    source-span metric.) Sections longer than `size` are sub-split with `recursive_spans`
    (the pinned langchain RecursiveCharacterTextSplitter).
    """
    headers = [
        (m.start(), m.end(), len(m.group(1)), m.group(2).strip()) for m in _MD_HEADER.finditer(text)
    ]

    def emit(
        out: list[tuple[int, int, JsonObject]],
        body_start: int,
        body_end: int,
        meta: JsonObject,
    ) -> None:
        bs, be = _trim(text, body_start, body_end)
        if be <= bs:
            return
        if be - bs <= size:
            out.append((bs, be, meta))
        else:
            for rs, re_end in recursive_spans(text[bs:be], size, overlap):
                out.append((bs + rs, bs + re_end, meta))

    out: list[tuple[int, int, JsonObject]] = []
    if not headers:
        emit(out, 0, len(text), {"headers": {}})
        return out

    if headers[0][0] > 0:  # preamble before the first header
        emit(out, 0, headers[0][0], {"headers": {}})

    stack: dict[int, str] = {}
    for i, (h_start, h_end, level, title) in enumerate(headers):
        stack = {lvl: t for lvl, t in stack.items() if lvl < level}
        stack[level] = title
        body_end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        meta = {"headers": {f"h{lvl}": stack[lvl] for lvl in sorted(stack)}}
        emit(out, h_end, body_end, meta)
    return out


def page_aligned_spans(
    text: str, size: int, overlap: int, page_spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Spans whose boundaries never cross a page-sidecar span (PDF page/citation-aware).

    The document is partitioned into regions: each sidecar page span plus the gaps around
    them (front matter, inter-page markers). A region that fits `size` becomes one span; a
    longer region is sub-split WITHIN itself via `recursive_spans`, so no chunk ever
    straddles a page boundary and every citation resolves to exactly one page range.
    """
    validate_chunking(size, overlap)
    n = len(text)
    regions: list[tuple[int, int]] = []
    cursor = 0
    for raw_start, raw_end in sorted(page_spans):
        start, end = max(cursor, raw_start, 0), min(raw_end, n)
        if end <= start:
            continue
        if start > cursor:
            regions.append((cursor, start))  # gap before this page (never merged into it)
        regions.append((start, end))
        cursor = end
    if cursor < n:
        regions.append((cursor, n))

    spans: list[tuple[int, int]] = []
    for region_start, region_end in regions:
        rs, re_ = _trim(text, region_start, region_end)
        if re_ <= rs:
            continue
        if re_ - rs <= size:
            spans.append((rs, re_))
        else:
            spans.extend((rs + s, rs + e) for s, e in recursive_spans(text[rs:re_], size, overlap))
    return spans


def heading_spans(text: str, size: int, overlap: int) -> list[tuple[int, int, JsonObject]]:
    """Heading-hierarchy (layout-aware) split: whole subtrees pack into one chunk when they fit.

    Unlike `markdown_spans` (one chunk per leaf section BODY, header lines stripped), this
    strategy keeps heading lines INSIDE the chunk text -- the layout the embedder sees matches
    the layout a reader sees -- and a heading whose entire subtree (itself + all nested
    subsections) fits within `size` becomes a single chunk. Oversized subtrees emit their own
    section (heading line + immediate body, recursively sub-split) and then recurse into each
    child heading. Every chunk carries the full breadcrumb of enclosing headings in
    `metadata.headers`.
    """
    headers = [
        (m.start(), m.end(), len(m.group(1)), m.group(2).strip()) for m in _MD_HEADER.finditer(text)
    ]
    out: list[tuple[int, int, JsonObject]] = []

    def emit(body_start: int, body_end: int, meta: JsonObject) -> None:
        bs, be = _trim(text, body_start, body_end)
        if be <= bs:
            return
        if be - bs <= size:
            out.append((bs, be, meta))
        else:
            for rs, re_ in recursive_spans(text[bs:be], size, overlap):
                out.append((bs + rs, bs + re_, meta))

    if not headers:
        emit(0, len(text), {"headers": {}})
        return out
    if headers[0][0] > 0:  # preamble before the first heading
        emit(0, headers[0][0], {"headers": {}})

    # Full breadcrumb per heading (same stack rule as markdown_spans / heading_breadcrumb).
    crumbs: list[dict[str, str]] = []
    stack: dict[int, str] = {}
    for _, _, level, title in headers:
        stack = {lvl: t for lvl, t in stack.items() if lvl < level}
        stack[level] = title
        crumbs.append({f"h{lvl}": stack[lvl] for lvl in sorted(stack)})

    def subtree_end(i: int) -> int:
        level = headers[i][2]
        for j in range(i + 1, len(headers)):
            if headers[j][2] <= level:
                return headers[j][0]
        return len(text)

    def emit_subtree(i: int) -> None:
        h_start, h_end, _, _ = headers[i]
        end = subtree_end(i)
        meta: JsonObject = {"headers": crumbs[i]}
        if end - h_start <= size:  # the whole subtree, heading line included, is one chunk
            emit(h_start, end, meta)
            return
        j = i + 1
        first_child = headers[j][0] if j < len(headers) and headers[j][0] < end else end
        body_s, body_e = _trim(text, h_end, first_child)
        if body_e > body_s:  # own section text (skip heading-only chunks; the breadcrumb
            emit(h_start, first_child, meta)  # already carries the title to child chunks)
        while j < len(headers) and headers[j][0] < end:
            emit_subtree(j)
            child_end = subtree_end(j)
            j += 1
            while j < len(headers) and headers[j][0] < child_end:
                j += 1  # skip the grandchildren emit_subtree(j) already covered

    i = 0
    while i < len(headers):
        emit_subtree(i)
        top_end = subtree_end(i)
        i += 1
        while i < len(headers) and headers[i][0] < top_end:
            i += 1
    return out


def doc_page_spans(corpus_root: Path, doc_id: str) -> list[tuple[int, int]] | None:
    """Page char spans for `doc_id` from its citation sidecar, or None when it has none."""
    # Function-level import: page_metadata imports this module for `_MD_HEADER`.
    from llb.rag.page_metadata import load_page_citations

    cite = load_page_citations(Path(corpus_root), doc_id)
    if cite is None:
        return None
    _, spans = cite
    out = [
        (span["char_start"], span["char_end"])
        for span in spans
        if isinstance(span.get("char_start"), int) and isinstance(span.get("char_end"), int)
    ]
    return out or None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(pct / 100.0 * (len(ordered) - 1)))
    return ordered[idx]


def semantic_spans(
    text: str, size: int, embedder: Any, threshold_pct: float = 90.0
) -> list[tuple[int, int]]:
    """Native semantic chunking: break where consecutive-sentence embedding distance spikes.

    Uses the PINNED embedder (injected) and our sentence offsets, so chunks are exact source
    substrings -- unlike langchain's SemanticChunker, whose joined text breaks the span metric.
    Groups longer than `size` are packed down so the KV/retrieval budget is respected.
    """
    sents = sentence_spans(text)
    if len(sents) <= 1:
        return sents
    vectors = embedder.encode_passages([text[s:e] for s, e in sents])
    dists = [
        1.0 - sum(float(a) * float(b) for a, b in zip(vectors[i], vectors[i + 1]))
        for i in range(len(vectors) - 1)
    ]
    threshold = _percentile(dists, threshold_pct)
    groups: list[tuple[int, int]] = []
    group_start = 0
    for i, dist in enumerate(dists):
        if dist > threshold:
            groups.append((group_start, i))
            group_start = i + 1
    groups.append((group_start, len(sents) - 1))

    spans: list[tuple[int, int]] = []
    for g0, g1 in groups:
        start, end = sents[g0][0], sents[g1][1]
        if end - start <= size:
            spans.append((start, end))
        else:
            spans.extend(_pack(sents[g0 : g1 + 1], size))
    return spans


def chunk_spans(
    text: str,
    strategy: str,
    size: int,
    overlap: int,
    embedder: Any = None,
    page_spans: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int, JsonObject]]:
    """Unified (start, end, metadata) spans for a strategy.

    `page_spans` feeds the `page` strategy (sidecar page char spans from `doc_page_spans`);
    without them -- a plain `.md`/`.txt` doc, or `parent_child` children re-chunking a parent
    slice whose page coordinates are unknown -- `page` falls back to `recursive`.
    """
    validate_chunking(size, overlap)
    if strategy == "fixed":
        return [(s, e, {}) for s, e in fixed_spans(text, size, overlap)]
    if strategy in ("sentence", "late"):  # late = sentence spans + late-pooled vectors
        return [(s, e, {}) for s, e in sentence_chunk_spans(text, size)]
    if strategy == "recursive":
        return [(s, e, {}) for s, e in recursive_spans(text, size, overlap)]
    if strategy == "markdown":
        return markdown_spans(text, size, overlap)
    if strategy == "heading":
        return heading_spans(text, size, overlap)
    if strategy == "page":
        if page_spans:
            return [(s, e, {}) for s, e in page_aligned_spans(text, size, overlap, page_spans)]
        return [(s, e, {}) for s, e in recursive_spans(text, size, overlap)]
    if strategy == "semantic":
        if embedder is None:
            raise SystemExit('ERROR: the "semantic" strategy needs an embedder (the [rag] extra).')
        return [(s, e, {}) for s, e in semantic_spans(text, size, embedder)]
    raise ValueError(f"unknown strategy: {strategy}")


def chunk_text(
    text: str,
    doc_id: str,
    strategy: str,
    size: int,
    overlap: int,
    embedder: Any = None,
    page_spans: list[tuple[int, int]] | None = None,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    spans = chunk_spans(text, strategy, size, overlap, embedder, page_spans=page_spans)
    for k, (start, end, meta) in enumerate(spans):
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}#{strategy}#{k:04d}",
                "char_start": start,
                "char_end": end,
                "text": text[start:end],
                "strategy": strategy,
                "size": size,
                "overlap": overlap,
                "metadata": meta,
            }
        )
    return chunks


def iter_docs(corpus_root: Path) -> Iterator[tuple[str, str]]:
    root = Path(corpus_root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".txt", ".md"):
            yield str(path.relative_to(root)), path.read_text(encoding="utf-8")


def chunk_corpus(
    corpus_root: Path,
    strategy: str,
    size: int,
    overlap: int,
    embedder: Any = None,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    governance_by_doc = manifest_governance_by_doc(corpus_root)
    for doc_id, text in iter_docs(corpus_root):
        page_spans = doc_page_spans(corpus_root, doc_id) if strategy == "page" else None
        doc_chunks = chunk_text(
            text, doc_id, strategy, size, overlap, embedder, page_spans=page_spans
        )
        governance = governance_by_doc.get(doc_id)
        if governance:
            for chunk in doc_chunks:
                chunk["metadata"] = {**(chunk.get("metadata") or {}), **governance}
        chunks.extend(doc_chunks)
    return chunks


def summarize(chunks: list[ChunkRecord]) -> ChunkSummary:
    sizes = [c["char_end"] - c["char_start"] for c in chunks]
    n = len(sizes)
    return {
        "n": n,
        "avg": sum(sizes) // n if n else 0,
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
    }


def build_faiss(
    chunks: list[ChunkRecord],
    model_name: str,
    index_dir: Path,
    strategy: str,
    corpus_root: Path | None = None,
) -> None:
    """Embed chunk texts and write a FAISS index. Needs the `[rag]` extra.

    The `late` strategy pools whole-document token embeddings instead of encoding each
    chunk text, so it needs `corpus_root` to re-read the source documents.
    """
    try:
        import faiss
        import numpy as np
        import sentence_transformers  # noqa: F401

        from llb.rag.embedding import Embedder
    except ImportError:
        _LOG.warning(
            "[build-rag-store] --embed needs the [rag] extra "
            "(sentence-transformers, faiss). Skipping '%s'.",
            strategy,
        )
        return
    index_dir.mkdir(parents=True, exist_ok=True)
    if strategy == "late":
        if corpus_root is None:
            raise ValueError("the 'late' strategy needs corpus_root to embed whole documents")
        from llb.rag.late_encoding import encode_store_vectors

        vectors = encode_store_vectors(chunks, corpus_root, Embedder(model_name))
    else:
        vectors = np.asarray(Embedder(model_name).encode_passages([c["text"] for c in chunks]))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_dir / f"{strategy}.faiss"))
    _LOG.info(
        "[build-rag-store] embedded %d chunks -> %s.faiss (dim %d)",
        len(chunks),
        strategy,
        vectors.shape[1],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk a corpus into a RAG store.")
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--strategy", default="all", choices=("all", *STRATEGIES))
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument(
        "--embed", action="store_true", help="also build a FAISS index ([rag] extra)"
    )
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    args = parser.parse_args(argv)

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    embedder = None
    if "semantic" in strategies:
        try:
            import sentence_transformers  # noqa: F401

            from llb.rag.embedding import Embedder

            embedder = Embedder(args.model)
        except ImportError:
            _LOG.warning(
                "[build-rag-store] 'semantic' needs the [rag] extra "
                "(sentence-transformers); skipping it."
            )
            strategies = [s for s in strategies if s != "semantic"]
    chunks_dir = args.out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info(
        "[build-rag-store] corpus=%s size=%d overlap=%d",
        args.corpus_root,
        args.size,
        args.overlap,
    )
    _LOG.info("  %-10s %7s %6s %6s %6s", "strategy", "chunks", "avg", "min", "max")
    for strategy in strategies:
        chunks = chunk_corpus(args.corpus_root, strategy, args.size, args.overlap, embedder)
        out_path = chunks_dir / f"{strategy}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        s = summarize(chunks)
        _LOG.info(
            "  %-10s %7d %6d %6d %6d",
            strategy,
            s["n"],
            s["avg"],
            s["min"],
            s["max"],
        )
        if args.embed:
            build_faiss(
                chunks, args.model, args.out_dir / "index", strategy, corpus_root=args.corpus_root
            )

    _LOG.info("[build-rag-store] chunks written -> %s", chunks_dir)
    return 0


if __name__ == "__main__":
    from llb.core.runtime import run

    sys.exit(run(main))
