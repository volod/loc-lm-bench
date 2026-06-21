"""Build a RAG store from documents using different chunking strategies.

Every strategy returns chunks anchored to `doc_id` + character offsets, so retrieval can be
scored against source-span gold labels by overlap (consistent with `llb.goldset.schema`).
That offset invariant is the constraint on which splitters we can reuse.

Strategies:
  - fixed      pure-Python fixed character window with overlap (zero deps)
  - sentence   pure-Python: pack whole sentences up to ~size (never cut mid-sentence)
  - recursive  langchain `RecursiveCharacterTextSplitter` (add_start_index -> exact offsets);
               falls back to a pure-Python paragraph->sentence->char split if `[rag]` is absent
  - markdown   structure-aware: headers parsed from the SOURCE (offset-exact) + recursive
               sub-split of long sections; header breadcrumbs go into chunk `metadata`
  - semantic   native: embed sentences with the PINNED embedder, break at distance spikes
               (offset-exact; langchain's SemanticChunker does not preserve source offsets)

`fixed` / `sentence` / `markdown` work without extra deps (markdown sub-splits via the pure
recursive fallback when `[rag]` is absent); `recursive` prefers langchain-text-splitters and
`semantic` needs the pinned embedder -- both from the `[rag]` extra, lazily imported.

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

from llb.contracts import ChunkRecord, ChunkSummary, JsonObject

PURE_STRATEGIES = ("fixed", "sentence")
STRATEGIES = ("fixed", "sentence", "recursive", "markdown", "semantic")

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


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) spans for paragraphs (runs separated by a blank line)."""
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"[^\n].*?(?=\n[ \t]*\n|\Z)", text, re.S):
        start, end = m.start(), m.end()
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
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


def _recursive_fallback(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Pure-Python paragraph -> sentence -> char split (used when `[rag]` is absent)."""
    out: list[tuple[int, int]] = []
    for para_start, para_end in paragraph_spans(text):
        if para_end - para_start <= size:
            out.append((para_start, para_end))
            continue
        sub = text[para_start:para_end]
        for rel_start, rel_end in _pack(sentence_spans(sub), size):
            if rel_end - rel_start <= size:
                out.append((para_start + rel_start, para_start + rel_end))
            else:
                seg = sub[rel_start:rel_end]
                for fs, fe in fixed_spans(seg, size, overlap):
                    out.append((para_start + rel_start + fs, para_start + rel_start + fe))
    return out


def _recursive_langchain(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True
    )
    spans: list[tuple[int, int]] = []
    for doc in splitter.create_documents([text]):
        start = doc.metadata["start_index"]
        spans.append((start, start + len(doc.page_content)))
    return spans


def recursive_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """langchain RecursiveCharacterTextSplitter when available, else the pure fallback."""
    try:
        return _recursive_langchain(text, size, overlap)
    except ImportError:
        return _recursive_fallback(text, size, overlap)


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
    (langchain RecursiveCharacterTextSplitter when present, pure fallback otherwise).
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
    text: str, strategy: str, size: int, overlap: int, embedder: Any = None
) -> list[tuple[int, int, JsonObject]]:
    """Unified (start, end, metadata) spans for a strategy."""
    validate_chunking(size, overlap)
    if strategy == "fixed":
        return [(s, e, {}) for s, e in fixed_spans(text, size, overlap)]
    if strategy == "sentence":
        return [(s, e, {}) for s, e in sentence_chunk_spans(text, size)]
    if strategy == "recursive":
        return [(s, e, {}) for s, e in recursive_spans(text, size, overlap)]
    if strategy == "markdown":
        return markdown_spans(text, size, overlap)
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
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for k, (start, end, meta) in enumerate(chunk_spans(text, strategy, size, overlap, embedder)):
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
    for doc_id, text in iter_docs(corpus_root):
        chunks.extend(chunk_text(text, doc_id, strategy, size, overlap, embedder))
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


def build_faiss(chunks: list[ChunkRecord], model_name: str, index_dir: Path, strategy: str) -> None:
    """Embed chunk texts and write a FAISS index. Needs the `[rag]` extra."""
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
            build_faiss(chunks, args.model, args.out_dir / "index", strategy)

    _LOG.info("[build-rag-store] chunks written -> %s", chunks_dir)
    return 0


if __name__ == "__main__":
    from llb.runtime import run

    sys.exit(run(main))
