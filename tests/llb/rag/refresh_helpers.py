"""Shared fakes and corpus builders for the dynamic-corpus-refresh tests.

The embedder is a deterministic hashed bag-of-words encoder (the curation-test pattern) that
records every passage batch, so tests can assert the refresh embedded ONLY the changed
documents' chunks. Corpora are tiny on-disk `.md` trees chunked with the pure `fixed` strategy,
so every store kind builds in the lightweight CI install (no GPU, no sentence-transformers).
"""

import json
import re
import zlib
from pathlib import Path

import numpy as np

from llb.rag.store import RagStore

DIM = 64

CHUNK_SIZE = 60
CHUNK_OVERLAP = 10

# Version 1 of the corpus; v2 modifies b.md, deletes c.md, and adds d.md.
V1_DOCS = {
    "a.md": (
        "Тарас Шевченко народився у селі Моринці. Він написав збірку Кобзар. "
        "Поет також малював і жив у Петербурзі."
    ),
    "b.md": (
        "Іван Франко народився у Нагуєвичах. Франко написав поему Мойсей. "
        "Він багато років працював у Львові."
    ),
    "c.md": (
        "Леся Українка народилася у Новограді-Волинському. Вона написала "
        "драму-феєрію Лісова пісня. УНІКАЛЬНИЙ-ТЕРМІН-ЛЕСІ."
    ),
}
V2_DOCS = {
    "a.md": V1_DOCS["a.md"],
    "b.md": (
        "Іван Франко народився у Нагуєвичах. Франко написав поему Мойсей. "
        "Останні роки він провів у Криворівні."
    ),
    "d.md": (
        "Михайло Коцюбинський народився у Вінниці. Він написав повість "
        "Тіні забутих предків про Карпати."
    ),
}


class CountingEmbedder:
    """Deterministic hashed bag-of-words encoder recording every passage batch."""

    model_name = "fake-hashed-bow"

    def __init__(self) -> None:
        self.passage_calls: list[list[str]] = []

    def _vec(self, text: str):
        vec = np.zeros(DIM, dtype="float32")
        for token in text.casefold().split():
            vec[zlib.crc32(token.encode("utf-8")) % DIM] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def _matrix(self, texts):
        if not texts:
            return np.zeros((0, DIM), dtype="float32")
        return np.stack([self._vec(t) for t in texts])

    def encode_passages(self, texts):
        texts = list(texts)
        self.passage_calls.append(texts)
        return self._matrix(texts)

    def encode_queries(self, texts):
        return self._matrix(list(texts))

    @property
    def embedded_texts(self) -> list[str]:
        return [text for batch in self.passage_calls for text in batch]


_TOKEN = re.compile(r"\S+")


class TokenLevelEmbedder(CountingEmbedder):
    """CountingEmbedder plus the token-level hooks the `late` strategy pools over.

    Tokens are whitespace runs with exact char offsets; each token embeds to the same hashed
    bag-of-words vector `_vec` gives it, so pooling is deterministic on the build and refresh
    paths alike. Encoded window texts are recorded so tests can assert late re-encoding touched
    only the changed documents.
    """

    model_name = "fake-token-bow"

    def __init__(self) -> None:
        super().__init__()
        self.token_windows: list[str] = []

    def max_seq_tokens(self) -> int:
        return 64

    def passage_token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in _TOKEN.finditer(text)]

    def encode_passage_tokens(self, text: str):
        self.token_windows.append(text)
        spans = self.passage_token_offsets(text)
        return spans, [self._vec(text[s:e]).tolist() for s, e in spans]


def write_citation_sidecar(corpus: Path, doc_id: str, page: int) -> Path:
    """Write a `*.citations.json` sidecar mapping the whole doc text to one source-PDF page."""
    stem = Path(doc_id).stem
    text = (corpus / doc_id).read_text(encoding="utf-8")
    payload = {
        "source": f"{stem}.pdf",
        "pages": [{"page": page, "char_start": 0, "char_end": len(text)}],
    }
    path = corpus / f"{stem}.citations.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def write_corpus(root: Path, docs: dict[str, str]) -> Path:
    """(Re)write `root` to contain exactly `docs`."""
    root.mkdir(parents=True, exist_ok=True)
    for stale in root.glob("*.md"):
        stale.unlink()
    for name, text in docs.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def build_store(
    corpus: Path,
    embedder: CountingEmbedder,
    *,
    strategy: str = "fixed",
    mode: str = "flat",
    backend: str = "faiss",
    lemmatizer=None,
) -> RagStore:
    """Build a small store over `corpus` with a pure chunker and the fake embedder.

    The default `fixed` strategy and the `sentence`/`late` strategies are all dependency-free
    (`late` additionally needs the token hooks of `TokenLevelEmbedder`).
    """
    return RagStore.build(
        corpus,
        strategy,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        mode=mode,
        child_size=30,
        vector_store=backend,
        embedder=embedder,
        lexical_lemmas=lemmatizer is not None,
        lemmatizer=lemmatizer,
    )


def retrieval_ids(store, questions: list[str], k: int = 5) -> list[list[str]]:
    """Ranked chunk ids per question -- the retrieval-equivalence probe."""
    return [[hit["chunk_id"] for hit in store.retrieve(q, k)] for q in questions]


QUESTIONS = [
    "Де народився Тарас Шевченко?",
    "Яку поему написав Франко?",
    "Хто написав Тіні забутих предків?",
    "Де провів останні роки Франко?",
]
