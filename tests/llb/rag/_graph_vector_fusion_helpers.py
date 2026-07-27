"""Graph-vector fusion ordering, endpoints, depth, span identity, and runner wiring tests."""

from llb.core.contracts.rag import ChunkRecord


class FakeRetriever:
    def __init__(self, hits: list[ChunkRecord]) -> None:
        self.hits = hits
        self.calls = 0
        self.depths: list[int] = []

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        self.calls += 1
        self.depths.append(k)
        return self.hits[:k]


def _chunk(name: str, start: int, end: int, *, lane: str) -> ChunkRecord:
    return {
        "doc_id": "doc.md",
        "char_start": start,
        "char_end": end,
        "text": name,
        "chunk_id": f"{lane}-{name}",
        "metadata": {"lane": lane},
    }


def _numbered(lane: str, count: int) -> list[ChunkRecord]:
    return [_chunk(f"{lane}{i}", i * 10, i * 10 + 10, lane=lane) for i in range(count)]


def _mention(name: str, start: int, end: int) -> ChunkRecord:
    """A graph evidence span: a few dozen characters cut around an entity, not a chunk."""
    return _chunk(name, start, end, lane="graph")
