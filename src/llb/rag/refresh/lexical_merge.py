"""Incremental BM25 refresh: rebuild postings reusing token counts of unchanged chunks.

`LexicalIndex` keys postings by build-order ordinals, so any add/modify/delete renumbers the
whole index. The expensive part of a rebuild is tokenization (with Ukrainian lemmatization it is
a pymorphy3 parse per token); the postings themselves already hold each chunk's exact term
frequencies. This module inverts the old postings back to per-ordinal term counts and merges
them with freshly tokenized texts for changed documents only, producing an index identical to
`LexicalIndex.build` over the merged chunk texts.
"""

from collections import Counter

from llb.rag.lexical import Lemmatizer, LexicalIndex, tokenize

# One merged-store entry in the new build order: an old ordinal to reuse (int) or the text of a
# freshly chunked unit to tokenize (str).
MergeEntry = int | str


def invert_postings(index: LexicalIndex) -> list[dict[str, int]]:
    """Per-ordinal `term -> tf` maps recovered exactly from the postings lists."""
    per_ordinal: list[dict[str, int]] = [{} for _ in range(index.n_docs)]
    for term, entries in index.postings.items():
        for ordinal, tf in entries:
            per_ordinal[ordinal][term] = tf
    return per_ordinal


def merge_lexical_index(
    old: LexicalIndex, entries: list[MergeEntry], lemmatizer: Lemmatizer | None = None
) -> LexicalIndex:
    """Build the refreshed index over `entries` (new build order), tokenizing only new texts.

    Reused entries carry their exact old term counts, so the result is identical to
    `LexicalIndex.build` over the merged texts; deleted chunks simply have no entry. The
    lemmatization mode is inherited from `old` (`lemmatizer` injects a fake for tests).
    """
    old_terms = invert_postings(old)
    needs_tokenizing = any(isinstance(entry, str) for entry in entries)
    query_lemmatizer = lemmatizer
    if query_lemmatizer is None and old.lemmatize and needs_tokenizing:
        query_lemmatizer = old._query_lemmatizer()  # noqa: SLF001 -- reuse/lazy-load the real one
    postings: dict[str, list[tuple[int, int]]] = {}
    doc_lengths: list[int] = []
    for ordinal, entry in enumerate(entries):
        if isinstance(entry, int):
            terms = old_terms[entry]
            doc_lengths.append(old.doc_lengths[entry])
        else:
            tokens = tokenize(entry, query_lemmatizer if old.lemmatize else None)
            terms = dict(Counter(tokens))
            doc_lengths.append(len(tokens))
        for term, tf in sorted(terms.items()):
            postings.setdefault(term, []).append((ordinal, tf))
    return LexicalIndex(postings, doc_lengths, old.lemmatize, query_lemmatizer)
