"""Choosing what a question must NOT retrieve: hard lexical negatives, and the honest fill.

Random negatives teach an encoder almost nothing -- a general multilingual encoder already separates
a question from an unrelated paragraph. What it gets wrong on a domain corpus is the passage that
shares the question's vocabulary and carries none of its evidence, which is exactly what BM25 over
the same chunks ranks first. So the pool is lexical, and the fill below exists only for the rows a
small corpus leaves short.

Two rejections are load-bearing and neither is optional:

  - a chunk that overlaps the item's own gold spans is NOT a negative, however far down the corpus
    it sits -- the lexical pool drops those, and the fill re-checks because it draws from anywhere;
  - a chunk repeating the positive's TEXT elsewhere hits no labelled span, so the overlap predicate
    would call it a negative; training against it would teach the encoder to separate a passage
    from itself.

Pure Python over chunk records: no encoder, no store.
"""

import random

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.rag.retrieval import chunk_hits_any
from llb.rag.vector_store.lexical_index import LexicalIndex

# Hard negatives per training row, and the BM25 depth searched per question before gold-hitting
# rows are dropped. The depth is generous so a question whose evidence sits in several chunks still
# leaves a full set of survivors.
DEFAULT_NEGATIVES = 4
NEGATIVE_SEARCH_MULTIPLIER = 6


def hard_negatives(
    question: str,
    spans: list[SourceSpanRecord],
    chunks: list[ChunkRecord],
    lexical: LexicalIndex,
    negatives: int,
) -> list[ChunkRecord]:
    """BM25 rows for the question that carry none of its evidence, best lexical match first."""
    depth = max(negatives * NEGATIVE_SEARCH_MULTIPLIER, negatives + 1)
    return [
        chunks[ordinal]
        for ordinal, _score in lexical.search(question, depth)
        if not chunk_hits_any(chunks[ordinal], spans)
    ]


def row_negatives(
    spans: list[SourceSpanRecord],
    positive: ChunkRecord,
    chunks: list[ChunkRecord],
    pool: list[ChunkRecord],
    negatives: int,
    rng: random.Random,
) -> tuple[list[ChunkRecord], int]:
    """Up to `negatives` negatives for one row, plus how many of them were LEXICALLY hard.

    The lexical pool comes first and the deterministic corpus draw fills what it could not, so a
    row is as hard as the corpus allows and no harder. Both rejections in the module docstring
    apply on both paths.
    """
    positive_text = str(positive["text"])
    taken = {positive_text}
    hard: list[ChunkRecord] = []
    for chunk in pool:
        if len(hard) >= negatives:
            break
        text = str(chunk["text"])
        if text in taken:
            continue
        taken.add(text)
        hard.append(chunk)
    chosen = list(hard)
    for chunk in rng.sample(chunks, k=len(chunks)):
        if len(chosen) >= negatives:
            break
        text = str(chunk["text"])
        if text in taken or chunk_hits_any(chunk, spans):
            continue
        taken.add(text)
        chosen.append(chunk)
    return chosen, len(hard)
