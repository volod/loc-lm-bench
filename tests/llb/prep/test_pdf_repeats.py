"""Intra-document repeated blocks: the census, both handling modes, and what follows the rewrite.

Pure unit tests over the committed `samples/corpora/intra_document_repeats_uk_v1/` fixture (one
converted-PDF-shaped manual repeating its own boilerplate, plus a second document sharing one
block) and hand-built documents: no PDF parser, no embedder, no GPU.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.goldset.schema import GoldItem, SourceSpan, load_goldset
from llb.main import app
from llb.prep.pdf.ingest import ingest_pdf_corpus
from llb.prep.pdf.model import PDF_CITATION_SUFFIX, PdfExtraction, PdfPageChunk
from llb.prep.pdf.repeat_corpus import REPEAT_REPORT_NAME, strip_corpus_repeats
from llb.prep.pdf.repeats import (
    REPEAT_ANCHOR,
    REPEAT_DROP,
    REPEAT_KEEP,
    heading_breadcrumb,
    remap_span,
    rewrite_repeated_blocks,
)
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.duplicates import duplicate_stats

RUNNER = CliRunner()

FIXTURE = Path("samples/corpora/intra_document_repeats_uk_v1/corpus")
REPEATED_DOC = "nastanova-oblik.md"
# The fixture's planted repetition; see its README -- these numbers ARE the fixture.
FIXTURE_BLOCKS, FIXTURE_GROUPS, FIXTURE_LARGEST = 18, 2, 3
FIXTURE_DROPPED_BLOCKS, FIXTURE_ANCHORED_BLOCKS = 4, 6
FIXTURE_CHARS, FIXTURE_DROP_CHARS, FIXTURE_ANCHOR_CHARS = 1957, 1440, 2137
# Chunk-level census of the same fixture at `sentence@200/30`.
FIXTURE_INTRA_GROUPS, FIXTURE_CROSS_GROUPS = 2, 1

PROCEDURE = "Порядок збереження документа"
SUPPORT = "**Служба підтримки:**"


def fixture_text() -> str:
    return (FIXTURE / REPEATED_DOC).read_text(encoding="utf-8")


def test_census_reports_the_planted_intra_document_repetition() -> None:
    census = rewrite_repeated_blocks(fixture_text(), mode=REPEAT_KEEP).census

    assert census["blocks"] == FIXTURE_BLOCKS
    assert (census["groups"], census["largest_group"]) == (FIXTURE_GROUPS, FIXTURE_LARGEST)
    assert census["handled_groups"] == FIXTURE_GROUPS
    assert census["handled_blocks"] == 0  # `keep` measures, it never rewrites


def test_keep_mode_leaves_the_document_byte_identical() -> None:
    text = fixture_text()

    rewrite = rewrite_repeated_blocks(text, mode=REPEAT_KEEP)

    assert rewrite.text == text
    assert rewrite.edits == []


def test_drop_keeps_the_first_copy_and_removes_the_rest() -> None:
    text = fixture_text()

    rewrite = rewrite_repeated_blocks(text, mode=REPEAT_DROP)

    assert rewrite.census["handled_blocks"] == FIXTURE_DROPPED_BLOCKS
    assert rewrite.text.count(PROCEDURE) == 1
    assert rewrite.text.count(SUPPORT) == 1
    assert len(rewrite.text) < len(text)
    # nothing is rewritten, only removed: every surviving line is still the corpus's own text
    assert all(line in text for line in rewrite.text.splitlines() if line.strip())


def test_drop_leaves_repeated_table_headers_and_headings_alone() -> None:
    """Structure is not furniture: a repeated table header makes the tables under it readable."""
    rewrite = rewrite_repeated_blocks(fixture_text(), mode=REPEAT_DROP)

    assert rewrite.text.count("|**Поле**|**Опис**|") == 2


def test_anchor_keeps_every_copy_and_makes_it_distinct() -> None:
    rewrite = rewrite_repeated_blocks(fixture_text(), mode=REPEAT_ANCHOR)

    assert rewrite.census["handled_blocks"] == FIXTURE_ANCHORED_BLOCKS
    assert rewrite.text.count(PROCEDURE) == FIXTURE_LARGEST
    anchored = [line for line in rewrite.text.splitlines() if line.startswith("> ")]
    assert len(anchored) == FIXTURE_ANCHORED_BLOCKS
    assert len(set(anchored)) == FIXTURE_LARGEST  # one anchor per section, not one per document
    assert "Розділ 2. Переміщення майна" in anchored[2]


def test_heading_breadcrumb_skips_the_rendered_document_title() -> None:
    text = "# Source PDF: a.pdf\n\n## Розділ 1\n\n### Крок\n\nтекст\n"

    assert heading_breadcrumb(text, text.index("текст")) == "Розділ 1 > Крок"
    assert heading_breadcrumb("текст", 0) == ""


def test_remap_span_follows_a_dropped_copy_onto_the_survivor() -> None:
    text = fixture_text()
    rewrite = rewrite_repeated_blocks(text, mode=REPEAT_DROP)
    third = text.index(PROCEDURE, text.index(PROCEDURE, text.index(PROCEDURE) + 1) + 1)

    moved = remap_span(rewrite.edits, third, third + len(PROCEDURE))

    assert moved is not None
    assert rewrite.text[moved[0] : moved[1]] == PROCEDURE
    assert moved[0] == rewrite.text.index(PROCEDURE)  # the one surviving copy


def test_remap_span_refuses_a_span_straddling_a_rewrite() -> None:
    text = fixture_text()
    rewrite = rewrite_repeated_blocks(text, mode=REPEAT_DROP)
    second = text.index(PROCEDURE, text.index(PROCEDURE) + 1)

    assert remap_span(rewrite.edits, second - 40, second + len(PROCEDURE)) is None


def test_every_untouched_span_keeps_its_text_under_both_modes() -> None:
    """The offset map is exact: an unrelated span still reads as itself after the rewrite."""
    text = fixture_text()
    unique = "Списання майна виконується комісією"
    start = text.index(unique)

    for mode in (REPEAT_DROP, REPEAT_ANCHOR):
        rewrite = rewrite_repeated_blocks(text, mode=mode)
        moved = remap_span(rewrite.edits, start, start + len(unique))
        assert moved is not None
        assert rewrite.text[moved[0] : moved[1]] == unique


def test_corpus_census_counts_only_repeats_inside_one_document() -> None:
    report = strip_corpus_repeats(FIXTURE, mode=REPEAT_KEEP)

    per_doc = {document["doc_id"]: document for document in report["documents"]}
    assert per_doc[REPEATED_DOC]["census"]["groups"] == FIXTURE_GROUPS
    # the support block is shared with the second document, but appears there ONCE
    assert per_doc["dovidka-oblik.md"]["census"]["groups"] == 0
    assert report["chars_before"] == report["chars_after"] == FIXTURE_CHARS


def test_chunk_census_splits_intra_from_cross_document_groups() -> None:
    stats = duplicate_stats(chunk_corpus(FIXTURE, "sentence", 200, 30))

    assert stats["intra_document_groups"] == FIXTURE_INTRA_GROUPS
    assert stats["cross_document_groups"] == FIXTURE_CROSS_GROUPS
    assert stats["groups"] == FIXTURE_INTRA_GROUPS + FIXTURE_CROSS_GROUPS


@pytest.mark.parametrize(
    ("mode", "chars"),
    [(REPEAT_DROP, FIXTURE_DROP_CHARS), (REPEAT_ANCHOR, FIXTURE_ANCHOR_CHARS)],
)
def test_strip_corpus_repeats_rewrites_into_a_new_root(
    tmp_path: Path, mode: str, chars: int
) -> None:
    out = tmp_path / mode

    report = strip_corpus_repeats(FIXTURE, out, mode=mode)

    assert report["chars_before"] == FIXTURE_CHARS
    assert report["chars_after"] == chars
    assert sum(len((out / doc).read_text(encoding="utf-8")) for doc in _corpus_docs(out)) == chars
    assert fixture_text() == (FIXTURE / REPEATED_DOC).read_text(encoding="utf-8")  # never in place
    assert json.loads((out / REPEAT_REPORT_NAME).read_text(encoding="utf-8"))["mode"] == mode


def test_strip_corpus_repeats_remaps_a_goldset_onto_the_stripped_corpus(tmp_path: Path) -> None:
    text = fixture_text()
    second = text.index(PROCEDURE, text.index(PROCEDURE) + 1)
    unique = text.index("Списання майна виконується комісією")
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(
        "\n".join(
            item.model_dump_json()
            for item in (
                _item("repeat", text, second, len(PROCEDURE)),
                _item("unique", text, unique, len("Списання майна виконується комісією")),
            )
        ),
        encoding="utf-8",
    )
    out = tmp_path / "drop"

    report = strip_corpus_repeats(
        FIXTURE, out, mode=REPEAT_DROP, goldset=goldset, goldset_out=out / "goldset.jsonl"
    )

    # the "repeat" item's evidence sat on a dropped copy -> re-homed onto the survivor; "unique"
    # is untouched.
    assert report["goldset"] == {
        "items": 2,
        "remapped": 2,
        "dropped": [],
        "rehomed": ["repeat"],
    }
    stripped = (out / REPEATED_DOC).read_text(encoding="utf-8")
    for item in load_goldset(out / "goldset.jsonl"):
        span = item.source_spans[0]
        assert stripped[span.char_start : span.char_end] == span.text


def test_conversion_applies_the_selected_repeat_mode_and_keeps_citations_exact(
    tmp_path: Path,
) -> None:
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "manual.pdf").write_bytes(b"%PDF")
    body = "Порядок збереження документа: натисніть кнопку Зберегти.\n\nУнікальний текст сторінки"

    def extractor(_path: Path, _tool: str) -> PdfExtraction:
        return PdfExtraction(
            text=body,
            parser="pymupdf4llm",
            pages=[PdfPageChunk(page=page, text=f"{body} {page}.") for page in (1, 2, 3)],
        )

    result = ingest_pdf_corpus(
        pdf_root, tmp_path / "md", extractor=extractor, min_chars=10, repeat_blocks=REPEAT_DROP
    )

    item = result.items[0]
    assert item.repeat_blocks == REPEAT_DROP
    assert item.doc_id is not None
    doc_text = (tmp_path / "md" / item.doc_id).read_text(encoding="utf-8")
    assert doc_text.count("Порядок збереження документа") == 1
    citations = json.loads(
        (tmp_path / "md" / f"{Path(item.doc_id).stem}{PDF_CITATION_SUFFIX}").read_text(
            encoding="utf-8"
        )
    )
    for page in citations["pages"]:
        assert doc_text[page["text_start"] : page["text_end"]].strip()


def test_conversion_reuse_is_keyed_on_the_repeat_mode(tmp_path: Path) -> None:
    """Switching the mode must reconvert: the previous output is a different rendering."""
    pdf_root = tmp_path / "pdf"
    pdf_root.mkdir()
    (pdf_root / "manual.pdf").write_bytes(b"%PDF")
    calls: list[Path] = []

    def extractor(path: Path, _tool: str) -> str:
        calls.append(path)
        return "Український текст документа. " * 30

    out = tmp_path / "md"
    ingest_pdf_corpus(pdf_root, out, extractor=extractor, min_chars=100)
    ingest_pdf_corpus(pdf_root, out, extractor=extractor, min_chars=100)
    assert len(calls) == 1  # unchanged source and mode: reused

    ingest_pdf_corpus(pdf_root, out, extractor=extractor, min_chars=100, repeat_blocks=REPEAT_DROP)
    assert len(calls) == 2


def test_cli_censuses_without_touching_the_corpus(tmp_path: Path) -> None:
    report = tmp_path / "census.json"

    result = RUNNER.invoke(
        app,
        ["strip-corpus-repeats", "--corpus", str(FIXTURE), "--report", str(report)],
    )

    assert result.exit_code == 0, result.output
    assert "repeated block groups: 2" in result.output
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == REPEAT_KEEP and payload["out_root"] is None


def test_cli_refuses_a_rewriting_mode_without_an_output_root() -> None:
    result = RUNNER.invoke(
        app, ["strip-corpus-repeats", "--corpus", str(FIXTURE), "--mode", REPEAT_DROP]
    )

    assert result.exit_code == 2
    assert "--out" in result.output


def _corpus_docs(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in sorted(root.rglob("*.md"))]


def _item(item_id: str, text: str, start: int, length: int) -> GoldItem:
    return GoldItem(
        id=item_id,
        question="Як зберегти документ?",
        reference_answer="Натисніть кнопку Зберегти.",
        source_doc_id=REPEATED_DOC,
        source_spans=[
            SourceSpan(
                doc_id=REPEATED_DOC,
                char_start=start,
                char_end=start + length,
                text=text[start : start + length],
            )
        ],
        provenance="human-authored",
        verified=True,
        split="final",
    )
