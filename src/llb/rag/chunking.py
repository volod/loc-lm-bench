"""Build a RAG store from documents using different chunking strategies.

Strategies (pure-Python, no extra deps):
  - fixed:     fixed character window with overlap
  - sentence:  pack whole sentences up to ~size (never cut mid-sentence)
  - recursive: paragraph -> sentence -> hard char window, greedily packed

Every chunk records `doc_id` + char offsets, so retrieval can be scored against
source-span gold labels by overlap (consistent with `llb.goldset.schema`).

CLI (also `make build-rag-store`):
    python -m llb.rag.chunking --corpus-root samples/corpus --out-dir .data/llb/rag \\
        --strategy all --size 800 --overlap 120
Add `--embed` (needs the `[rag]` extra) to also build a FAISS index per strategy.
"""

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

STRATEGIES = ("fixed", "sentence", "recursive")

_TERM = re.compile(r"[.!?…]+")
_CLOSERS = "”»\")]’"


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
    if size <= 0:
        raise ValueError("size must be > 0")
    overlap = max(0, min(overlap, size - 1))
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
            out.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    if cur_start is not None:
        out.append((cur_start, cur_end))
    return out


def sentence_chunk_spans(text: str, size: int) -> list[tuple[int, int]]:
    return _pack(sentence_spans(text), size)


def recursive_spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
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
                for fix_start, fix_end in fixed_spans(seg, size, overlap):
                    out.append((para_start + rel_start + fix_start, para_start + rel_start + fix_end))
    return out


def chunk_offsets(text: str, strategy: str, size: int, overlap: int) -> list[tuple[int, int]]:
    if strategy == "fixed":
        return fixed_spans(text, size, overlap)
    if strategy == "sentence":
        return sentence_chunk_spans(text, size)
    if strategy == "recursive":
        return recursive_spans(text, size, overlap)
    raise ValueError(f"unknown strategy: {strategy}")


def chunk_text(text: str, doc_id: str, strategy: str, size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for k, (start, end) in enumerate(chunk_offsets(text, strategy, size, overlap)):
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
            }
        )
    return chunks


def iter_docs(corpus_root: Path) -> Iterator[tuple[str, str]]:
    root = Path(corpus_root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".txt", ".md"):
            yield str(path.relative_to(root)), path.read_text(encoding="utf-8")


def chunk_corpus(corpus_root: Path, strategy: str, size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for doc_id, text in iter_docs(corpus_root):
        chunks.extend(chunk_text(text, doc_id, strategy, size, overlap))
    return chunks


def summarize(chunks: list[dict]) -> dict:
    sizes = [c["char_end"] - c["char_start"] for c in chunks]
    n = len(sizes)
    return {
        "n": n,
        "avg": sum(sizes) // n if n else 0,
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
    }


def build_faiss(chunks: list[dict], model_name: str, index_dir: Path, strategy: str) -> None:
    """Embed chunk texts and write a FAISS index. Needs the `[rag]` extra."""
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            f"[build-rag-store] --embed needs the [rag] extra "
            f"(sentence-transformers, faiss). Skipping '{strategy}'.",
            file=sys.stderr,
        )
        return
    index_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    vectors = np.asarray(
        model.encode([c["text"] for c in chunks], normalize_embeddings=True),
        dtype="float32",
    )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_dir / f"{strategy}.faiss"))
    print(f"[build-rag-store] embedded {len(chunks)} chunks -> {strategy}.faiss (dim {vectors.shape[1]})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk a corpus into a RAG store.")
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--strategy", default="all", choices=("all", *STRATEGIES))
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--embed", action="store_true", help="also build a FAISS index ([rag] extra)")
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    args = parser.parse_args(argv)

    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    chunks_dir = args.out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build-rag-store] corpus={args.corpus_root} size={args.size} overlap={args.overlap}")
    print(f"  {'strategy':<10} {'chunks':>7} {'avg':>6} {'min':>6} {'max':>6}")
    for strategy in strategies:
        chunks = chunk_corpus(args.corpus_root, strategy, args.size, args.overlap)
        out_path = chunks_dir / f"{strategy}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        s = summarize(chunks)
        print(f"  {strategy:<10} {s['n']:>7} {s['avg']:>6} {s['min']:>6} {s['max']:>6}")
        if args.embed:
            build_faiss(chunks, args.model, args.out_dir / "index", strategy)

    print(f"[build-rag-store] chunks written -> {chunks_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
