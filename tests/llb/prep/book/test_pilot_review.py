"""A draft, an acceptance, and the score input stay distinct across restarts."""

import json
from pathlib import Path

import pytest

from llb.prep.book.metrics.geometry import Box
from llb.prep.book.metrics.reference import digest_pages, load_reference, verify_reference_digest
from llb.prep.book.metrics.submission import HypBlock, PageSubmission
from llb.prep.book.pilot_review.merge import merge_readings
from llb.prep.book.pilot_review.prepare import write_json
from llb.prep.book.pilot_review.session import Review, SaveRequest, StaleRevision
from llb.prep.book.source import sha256_file


def _block(block_id: str, text: str, box: Box, order: int) -> HypBlock:
    return HypBlock(block_id, "body", box, order, text, None, None, None, None)


def test_merge_flags_disagreements_and_keeps_alternate_only_blocks() -> None:
    layout = PageSubmission(
        "private-layout",
        1,
        (_block("same-id", "A 5", Box(10, 10, 30, 50), 0),),
        False,
    )
    alternate = PageSubmission(
        "private-line",
        1,
        (
            _block("same-id", "A S", Box(12, 12, 28, 17), 0),
            _block("extra", "Missing heading", Box(2, 2, 25, 8), 1),
        ),
        False,
    )
    blocks, notes = merge_readings(layout, alternate)
    assert [b["text"] for b in blocks] == ["Missing heading", "A 5"]
    assert [n["block_id"] for n in notes] == [b["block_id"] for b in blocks]
    assert notes[0]["flags"] == ["No region owns this line."]
    assert notes[1]["flags"] == ["Text differs.", "Block boundaries differ."]
    assert notes[1]["alternate"][0]["text"] == "A S"
    assert "private-" not in json.dumps((blocks, notes))


@pytest.fixture
def review_root(tmp_path: Path) -> Path:
    fixture = Path(__file__).parent / "fixtures" / "metrics" / "reference" / "pages" / "p0001.json"
    draft = json.loads(fixture.read_bytes())
    scan = tmp_path / "scans" / "p0001.png"
    scan.parent.mkdir()
    scan.write_bytes(b"synthetic scan sentinel")
    pages = tmp_path / "predraft" / "pages"
    write_json(pages / "p0001.json", draft)
    write_json(
        tmp_path / "review.json",
        {
            "schema_id": "llb.book-ocr-pilot-review",
            "schema_version": "1.0.0",
            "source_sha256": "a" * 64,
            "pilot_sha256": "b" * 64,
            "predraft_sha256": digest_pages(pages),
            "pages": [
                {
                    "id": "p0001",
                    "number": draft["pdf_page"],
                    "page_class": draft["page_class"],
                    "orientation": 0,
                    "scan_sha256": sha256_file(scan),
                    "notes": [],
                }
            ],
        },
    )
    return tmp_path


def test_resume_accept_and_export_revision_bound_to_scan(review_root: Path) -> None:
    review = Review(review_root)
    original = review.earlier_page("p0001", 0)
    edited = {**original, "printed_page": "58"}
    assert review.save("p0001", SaveRequest(page=edited, reviewer="Ada", accept=False, base=0)) == 1
    reopened = Review(review_root)
    assert reopened.worksheet("p0001")["page"] == edited
    with pytest.raises(StaleRevision):
        reopened.save("p0001", SaveRequest(page=original, reviewer="Ada", accept=False, base=0))
    with pytest.raises(ValueError, match="awaiting acceptance"):
        reopened.finalize()

    reopened.save("p0001", SaveRequest(page=edited, reviewer="Ada", accept=True, base=1))
    output = reopened.finalize()
    assert reopened.finalize() == output
    assert reopened.earlier_page("p0001", 0) == original
    assert load_reference(output)[0].printed_page == "58"
    record = output / "acceptance.json"
    assert json.loads(record.read_bytes())["pages"][0]["reviewer"] == "Ada"
    verify_reference_digest(output, record, reopened.manifest.source_sha256)
    write_json(output / "pages" / "p0001.json", original)
    with pytest.raises(ValueError, match="differs from the reference checksum"):
        verify_reference_digest(output, record, reopened.manifest.source_sha256)


def test_acceptance_requires_parseable_formulas_and_the_same_scan(review_root: Path) -> None:
    review = Review(review_root)
    edited = review.earlier_page("p0001", 0)
    formula = next(b for b in edited["blocks"] if b["kind"] == "formula")
    formula["latex"] = r"\frac{1}{"
    request = SaveRequest(page=edited, reviewer="Ada", accept=True, base=0)
    with pytest.raises(ValueError, match="latex does not parse"):
        review.save("p0001", request)
    review.scan("p0001").write_bytes(b"wrong scan")
    with pytest.raises(ValueError, match="scan checksum changed"):
        review.save("p0001", request)
    assert review.worksheet("p0001")["version"] == 0
