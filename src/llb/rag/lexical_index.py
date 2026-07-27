"""Persistent BM25 index over normalized chunk tokens."""

import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from llb.rag.lexical import (
    BM25_B,
    BM25_K1,
    LEXICAL_INDEX_VERSION,
    Lemmatizer,
    load_uk_lemmatizer,
    tokenize,
)


class LexicalIndex:
    """Deterministic pure-Python BM25 over chunk texts (build-order ids, like the vector index)."""

    def __init__(
        self,
        postings: dict[str, list[tuple[int, int]]],
        doc_lengths: list[int],
        lemmatize: bool,
        lemmatizer: Lemmatizer | None = None,
    ):
        self.postings = postings  # term -> [(chunk_ordinal, term_frequency)] sorted by ordinal
        self.doc_lengths = doc_lengths
        self.lemmatize = lemmatize
        self._lemmatizer = lemmatizer
        self.n_docs = len(doc_lengths)
        self.avg_doc_len = (sum(doc_lengths) / self.n_docs) if self.n_docs else 0.0

    @classmethod
    def build(
        cls, texts: Iterable[str], lemmatize: bool = False, lemmatizer: Lemmatizer | None = None
    ) -> "LexicalIndex":
        """Index `texts` in order. With `lemmatize`, tokens collapse to lemmas at index time
        (query tokens are lemmatized identically in `search`); the texts themselves are never
        modified. `lemmatizer` injects a fake for tests; default is the pymorphy3 Ukrainian one.
        """
        if lemmatize and lemmatizer is None:
            lemmatizer = load_uk_lemmatizer()
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_lengths: list[int] = []
        for ordinal, text in enumerate(texts):
            tokens = tokenize(text, lemmatizer if lemmatize else None)
            doc_lengths.append(len(tokens))
            for term, tf in sorted(Counter(tokens).items()):
                postings.setdefault(term, []).append((ordinal, tf))
        return cls(postings, doc_lengths, lemmatize, lemmatizer)

    def _query_lemmatizer(self) -> Lemmatizer | None:
        if not self.lemmatize:
            return None
        if self._lemmatizer is None:  # loaded index: resolve the real lemmatizer lazily
            self._lemmatizer = load_uk_lemmatizer()
        return self._lemmatizer

    def search(
        self, query: str, k: int, allowed: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Top-k `(chunk_ordinal, bm25_score)` for `query`, best first, ties broken by ordinal.

        `allowed` restricts candidates to those ordinals (the metadata-filter seam applies
        BEFORE fusion); only chunks matching at least one query term are returned.
        """
        if k < 1 or not self.n_docs:
            return []
        scores: dict[int, float] = {}
        for term in tokenize(query, self._query_lemmatizer()):
            entries = self.postings.get(term)
            if not entries:
                continue
            idf = math.log(1.0 + (self.n_docs - len(entries) + 0.5) / (len(entries) + 0.5))
            for ordinal, tf in entries:
                if allowed is not None and ordinal not in allowed:
                    continue
                norm = 1.0 - BM25_B + BM25_B * (self.doc_lengths[ordinal] / self.avg_doc_len)
                scores[ordinal] = scores.get(ordinal, 0.0) + idf * (
                    tf * (BM25_K1 + 1.0) / (tf + BM25_K1 * norm)
                )
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:k]

    def save(self, path: Path | str) -> None:
        payload = {
            "version": LEXICAL_INDEX_VERSION,
            "lemmatize": self.lemmatize,
            "doc_lengths": self.doc_lengths,
            "postings": {term: entries for term, entries in sorted(self.postings.items())},
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "LexicalIndex":
        """Load a persisted index, refusing one whose postings predate the current tokenizer.

        Postings ARE tokenizer output, and a query is tokenized by the code doing the loading, so
        an index written by a different tokenizer version silently mismatches every term the
        tokenizer changed. Refusing with a rebuild message is the same discipline the store applies
        to a changed corpus fingerprint.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = str(payload.get("version", "unknown"))
        if version != LEXICAL_INDEX_VERSION:
            raise SystemExit(
                f"[rag] the lexical index at {path} was written by tokenizer version {version}, "
                f"but this build tokenizes as {LEXICAL_INDEX_VERSION}; its postings would not "
                "match the queries. Rebuild the store with "
                "`llb build-index --retrieval-mode hybrid`."
            )
        postings = {
            term: [(int(ordinal), int(tf)) for ordinal, tf in entries]
            for term, entries in payload["postings"].items()
        }
        return cls(postings, [int(n) for n in payload["doc_lengths"]], bool(payload["lemmatize"]))
