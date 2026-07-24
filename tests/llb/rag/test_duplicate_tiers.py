"""Near-duplicate collapse tiers (`llb.rag.duplicate_tiers`) and the residue they measure.

Pure unit tests over the committed `samples/corpora/near_duplicate_chunks_uk_v1/` fixture (three
Ukrainian regulations whose shared furniture repeats with the differences a PDF conversion
produces) plus hand-built records and hand-built vectors: no FAISS, no GPU, no embedder.
"""

from pathlib import Path

import pytest

from llb.core.contracts.rag import ChunkRecord
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.duplicate_residue import format_residue_report, measure_duplicate_residue
from llb.rag.duplicate_tiers import (
    TIER_EXACT,
    TIER_MASKED,
    TIER_NORMALIZED,
    duplicate_key,
)
from llb.rag.duplicates import (
    collapse_duplicate_chunks,
    duplicate_occurrences,
    duplicate_stats,
    expand_duplicate_chunks,
    format_duplicate_stats,
)

FIXTURE = Path("samples/corpora/near_duplicate_chunks_uk_v1/corpus")

# The fixture's planted tiers under `heading@400`; see its README -- these numbers ARE the fixture.
FIXTURE_STRATEGY, FIXTURE_SIZE, FIXTURE_OVERLAP = "heading", 400, 30
FIXTURE_CHUNKS = 15
# tier -> (groups, collapsed, indexed after collapse)
FIXTURE_TIERS = {
    TIER_EXACT: (2, 2, 13),
    TIER_NORMALIZED: (2, 3, 12),
    TIER_MASKED: (4, 7, 8),
}


def fixture_chunks() -> list[ChunkRecord]:
    return chunk_corpus(FIXTURE, FIXTURE_STRATEGY, FIXTURE_SIZE, FIXTURE_OVERLAP)


def _chunk(doc: str, start: int, end: int, text: str, chunk_id: str) -> ChunkRecord:
    return {
        "doc_id": doc,
        "chunk_id": chunk_id,
        "char_start": start,
        "char_end": end,
        "text": text,
        "metadata": {},
    }


def test_normalizer_ignores_case_and_whitespace_but_not_content():
    assert duplicate_key("Цей  ДОКУМЕНТ\nчинний.", TIER_NORMALIZED) == duplicate_key(
        "цей документ чинний", TIER_NORMALIZED
    )
    assert duplicate_key("ставка 15", TIER_NORMALIZED) != duplicate_key("ставка 7", TIER_NORMALIZED)


def test_masking_merges_digit_runs_and_only_digit_runs():
    assert duplicate_key("Сторінка 1 з 12", TIER_MASKED) == duplicate_key(
        "Сторінка 4 з 12", TIER_MASKED
    )
    assert duplicate_key("ставка 15", TIER_MASKED) != duplicate_key("тариф 15", TIER_MASKED)


def test_a_token_free_chunk_falls_back_to_its_verbatim_text():
    # normalizing to the empty token stream must never merge unrelated separator chunks
    assert duplicate_key("---", TIER_NORMALIZED) != duplicate_key("===", TIER_NORMALIZED)
    assert duplicate_key("---", TIER_MASKED) != duplicate_key("===", TIER_MASKED)


def test_unknown_tier_is_refused():
    with pytest.raises(ValueError, match="unknown duplicate tier"):
        duplicate_key("text", "fuzzy")


def test_fixture_plants_the_documented_tier_ladder():
    chunks = fixture_chunks()
    assert len(chunks) == FIXTURE_CHUNKS
    for tier, (groups, collapsed, indexed) in FIXTURE_TIERS.items():
        stats = duplicate_stats(chunks, tier)
        assert (stats["groups"], stats["collapsed"], stats["unique"]) == (
            groups,
            collapsed,
            indexed,
        )
        assert stats["tier"] == tier


def test_the_normalized_tier_does_not_reach_an_apostrophe_variant():
    """The reused `hash`-tier normalizer tokenizes before unifying apostrophes (fixture README)."""
    chunks = [c for c in fixture_chunks() if "Застереження" in str(c["text"])]
    assert len(chunks) == 3
    keys = {duplicate_key(str(c["text"]), TIER_NORMALIZED) for c in chunks}
    assert len(keys) == 2  # the U+2019 copy keeps its own key


def test_collapse_at_a_coarser_tier_stays_offset_exact_and_keeps_each_copys_text():
    chunks = fixture_chunks()
    collapse = collapse_duplicate_chunks(chunks, TIER_MASKED)
    assert len(collapse.chunks) == FIXTURE_TIERS[TIER_MASKED][2]
    for survivor in collapse.chunks:
        source = (FIXTURE / str(survivor["doc_id"])).read_text(encoding="utf-8")
        assert source[survivor["char_start"] : survivor["char_end"]] == survivor["text"]
        for copy in duplicate_occurrences(survivor):
            other = (FIXTURE / str(copy["doc_id"])).read_text(encoding="utf-8")
            text = copy.get("text", survivor["text"])
            assert other[copy["char_start"] : copy["char_end"]] == text


def test_expansion_is_exact_and_refuses_a_row_for_a_differing_copy():
    chunks = fixture_chunks()
    survivors = collapse_duplicate_chunks(chunks, TIER_MASKED).chunks
    expanded, rows = expand_duplicate_chunks(survivors)
    by_doc: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        by_doc.setdefault(str(chunk["doc_id"]), []).append(chunk)
    restored: dict[str, list[ChunkRecord]] = {}
    for chunk in expanded:
        restored.setdefault(str(chunk["doc_id"]), []).append(chunk)
    assert restored == by_doc  # every copy back with ITS OWN text, in build order
    for chunk, row in zip(expanded, rows):
        # a row is offered only where the stored vector encodes this very text
        assert row is None or chunk["text"] == survivors[row]["text"]
    assert any(row is None for row in rows)  # the masked tier merged texts that differ


def test_exact_tier_records_stay_free_of_a_text_key():
    survivors = collapse_duplicate_chunks(fixture_chunks(), TIER_EXACT).chunks
    copies = [copy for survivor in survivors for copy in duplicate_occurrences(survivor)]
    assert copies and all("text" not in copy for copy in copies)


def test_the_build_summary_names_the_tier_it_measured():
    line = format_duplicate_stats(duplicate_stats(fixture_chunks(), TIER_MASKED))
    assert "digit-masked-equivalent to another" in line
    assert "8 indexed (7 collapsed)" in line


def _vectors(rows: list[list[float]]):
    import numpy as np

    return np.asarray(rows, dtype="float32")


def test_residue_reports_the_tier_ladder_and_the_cosine_bands():
    chunks = [
        _chunk("a.md", 0, 9, "Сторінка 1 з 12", "a#0"),
        _chunk("b.md", 0, 9, "Сторінка 4 з 12", "b#0"),
        _chunk("c.md", 0, 9, "Ставка становить 15 відсотків", "c#0"),
        _chunk("d.md", 0, 9, "Зовсім інший технічний розділ", "d#0"),
    ]
    report = measure_duplicate_residue(
        chunks,
        _vectors([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        store_tier=TIER_EXACT,
        thresholds=(0.99,),
    )
    assert report["n_indexed"] == 4
    assert report["tiers"][TIER_EXACT]["collapsed"] == 0
    assert report["tiers"][TIER_NORMALIZED]["collapsed"] == 0
    assert report["tiers"][TIER_MASKED]["collapsed"] == 1  # the two page footers
    band = report["bands"][0]
    assert (band["pairs"], band["chunks"]) == (2, 4)  # both pairs of identical vectors
    assert band["masked_pairs"] == 1  # only the footer pair is text-reachable
    assert band["normalized_pairs"] == 0


def test_residue_samples_what_each_kind_of_merge_would_do():
    chunks = [
        _chunk("a.md", 0, 9, "Ставка становить 15 відсотків", "a#0"),
        _chunk("b.md", 0, 9, "Ставка становить 7 відсотків", "b#0"),
        _chunk("c.md", 0, 9, "Технічна довідка про подачу насоса", "c#0"),
        _chunk("d.md", 0, 9, "Технічна довідка щодо подачі насоса", "d#0"),
    ]
    report = measure_duplicate_residue(
        chunks,
        _vectors([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        store_tier=TIER_EXACT,
        thresholds=(0.99,),
    )
    digit = report["digit_merge_examples"]
    assert len(digit) == 1 and digit[0]["masked_equal"] and not digit[0]["normalized_equal"]
    assert "15" in digit[0]["a"] and "7" in digit[0]["b"]
    near = report["near_duplicate_examples"]
    assert len(near) == 1 and near[0]["cosine"] == pytest.approx(1.0)
    assert not near[0]["masked_equal"]  # no text tier reaches this pair
    rendered = format_residue_report(report)
    assert "text tiers" in rendered and "cosine bands" in rendered
