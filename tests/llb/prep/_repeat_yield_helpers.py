"""Per-question yield audit for `--repeat-blocks drop` (`llb.prep.pdf.repeat_yield`).

Pure: fake stores exposing the `.retrieve` seam and hand-built gold items, plus one end-to-end
CLI test over the committed intra-document-repeats fixture with a straddling item. No FAISS, no
embedder, no GPU on the pure lane.
"""

from pathlib import Path


from typer.testing import CliRunner


from llb.core.contracts.rag import ChunkRecord


from llb.goldset.schema import GoldItem, SourceSpan


RUNNER = CliRunner()


FIXTURE = Path("samples/corpora/intra_document_repeats_uk_v1/corpus")


REPEATED_DOC = "nastanova-oblik.md"


PROCEDURE = "Порядок збереження документа"


class _StubStore:
    """Returns a fixed chunk list per question (the id it should hit, or nothing)."""

    def __init__(self, hits: dict[str, ChunkRecord]) -> None:
        self._hits = hits

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        hit = self._hits.get(question)
        return [hit] if hit is not None else []


def _item(item_id: str, start: int, end: int) -> GoldItem:
    return GoldItem(
        id=item_id,
        question=item_id,  # the question doubles as the stub-store key
        reference_answer="a",
        source_doc_id="d.md",
        source_spans=[
            SourceSpan(doc_id="d.md", char_start=start, char_end=end, text="x" * (end - start))
        ],
        provenance="human-authored",
        verified=True,
        split="final",
    )


def _chunk(start: int, end: int) -> ChunkRecord:
    return {"doc_id": "d.md", "char_start": start, "char_end": end, "text": "x" * (end - start)}
