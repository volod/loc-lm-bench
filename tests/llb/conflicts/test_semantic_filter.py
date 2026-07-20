"""Structural semantic-tier filtering over synthetic records and the committed fixture."""

from llb.conflicts.corpus import load_corpus_docs
from llb.conflicts.semantic_filter import select_content_chunks
from llb.core.contracts.rag import ChunkRecord

from conflict_helpers import DOC_ARCHIVE, FIXTURE_CORPUS, fake_store_view


def _chunk(doc_id: str, text: str, heading: str) -> ChunkRecord:
    return {
        "doc_id": doc_id,
        "char_start": 0,
        "char_end": len(text),
        "text": text,
        "metadata": {"headers": {"h1": heading}},
    }


def test_fixture_repeated_metadata_is_excluded_and_one_off_prose_survives():
    """The corpus fixture proves the rule through the real heading chunker."""
    store = fake_store_view()
    docs = load_corpus_docs(FIXTURE_CORPUS)
    body_offsets = {doc.doc_id: doc.body_offset for doc in docs}
    selection = select_content_chunks(store.chunks, body_offsets)
    registry = {
        ordinal for ordinal, chunk in enumerate(store.chunks) if "Реєстр видання" in chunk["text"]
    }
    ordinary = next(
        ordinal
        for ordinal, chunk in enumerate(store.chunks)
        if chunk["doc_id"] == DOC_ARCHIVE and "Справи постійного зберігання" in chunk["text"]
    )

    assert len(registry) == 2
    assert registry.isdisjoint(selection.ordinals)
    assert ordinary in selection.ordinals
    assert selection.metadata_blocks == 2


def test_repeated_claim_prose_under_a_shared_heading_is_preserved():
    """Shared section names and shared prose are evidence candidates, not metadata by themselves."""
    claim = " ".join(
        [
            "The written request must be registered on arrival and reviewed by the responsible",
            "officer within the period established by policy before a response is issued",
        ]
    )
    chunks = [_chunk("a.md", claim, "General rules"), _chunk("b.md", claim, "General rules")]

    assert select_content_chunks(chunks, {}).ordinals == {0, 1}


def test_metadata_heading_must_appear_only_once_per_document():
    """A repeated numbered section inside one document is not corpus-wide boilerplate."""
    record = (
        "Edition 2024 issue 18 page 42 entry 1001 archive bulletin 7 dated 01 09 2024 "
        "series 3 registration code 445566 print run 250 copies"
    )
    chunks = [
        _chunk("a.md", record, "Issue register"),
        _chunk("a.md", record.replace("1001", "1002"), "Issue register"),
        _chunk("b.md", record.replace("1001", "2001"), "Issue register"),
    ]

    assert select_content_chunks(chunks, {}, min_tokens=0).ordinals == {0, 1, 2}
