"""FAISS index building and the `python -m llb.rag.chunking.build` CLI (also `make build-rag-store`).

CLI:
    python -m llb.rag.chunking.build --corpus-root samples/corpus --out-dir .data/llb/rag \\
        --strategy all --size 800 --overlap 120
Add `--embed` (needs `[rag]`) to also build a FAISS index per strategy.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from llb.core.contracts import ChunkRecord
from llb.rag.chunking.corpus import chunk_corpus, summarize
from llb.rag.chunking.dispatch import STRATEGIES

_LOG = logging.getLogger(__name__)


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
