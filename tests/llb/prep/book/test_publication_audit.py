"""Exercise the production commands on a longer book and follow a hit to source pixels."""

import json
import shutil
from pathlib import Path

import fitz
import numpy as np
import pytest

from llb.core.contracts.data_prep.book_references import BookReferences
from llb.prep.book.preprocess import images
from llb.prep.book.project import write
from llb.prep.book.render import swap
from llb.prep.book.review.constants import ENTER_TEXT
from llb.prep.book.review.decide import apply_review_decision
from llb.prep.book.review.io import queue_path, read_queue, write_queue
from llb.prep.book.structure.io import read_structure, structure_path
from llb.prep.book.validate.constants import MARKER_MISMATCH, QUEUE_STALE
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.page_metadata import annotate_page_metadata
from llb.rag.vector_store.lexical_index import LexicalIndex
from tests.llb.prep.book._stages import SimulatedKill
from tests.llb.prep.book.synthetic_book import (
    DISPUTE,
    FAILED_PAGE,
    write_scan,
)
from tests.llb.prep.book.synthetic_publish import _goldset, _snapshot, _vault_text

from tests.llb.prep.book.publication_fixture import (
    _REPEATED_FOLIO_PAGE,
    _END,
    _LONG,
    _book,
    _command,
    _measured_render,
    _project,
    _render,
    _run,
    _settle,
)


def test_publication_and_resume_on_a_long_book(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    short = _run(write_scan(tmp_path / "library/control.pdf"), tmp_path / "short", monkeypatch)
    control = _measured_render(short, tmp_path / "control", monkeypatch)
    scan = write_scan(tmp_path / "library/scan.pdf", page_count=_LONG, color_page=_LONG)
    clean = _run(scan, tmp_path / "clean-run", monkeypatch)
    resumed = _run(scan, tmp_path / "resumed-run", monkeypatch, kill=True)
    clean_vault = tmp_path / "clean-vault"
    vault = tmp_path / "vault"
    long = _measured_render(clean, clean_vault, monkeypatch)
    assert long.count > 4 * control.count
    assert 0 < control.peak <= 2 and 0 < long.peak <= 2
    _retry_publication(resumed, vault, monkeypatch)
    assert _portable(_snapshot(vault)) == _portable(_snapshot(clean_vault))
    _fallback_pixels(scan, resumed, vault)

    corpus = tmp_path / "corpus"
    clean_corpus = tmp_path / "clean-corpus"
    _retry_projection(resumed, corpus, monkeypatch)
    _project(clean, clean_corpus)
    assert _portable(_snapshot(corpus)) == _portable(_snapshot(clean_corpus))
    _ambiguous(resumed, vault, corpus)
    staged = tmp_path / "staged"
    _command("ingest-corpus", "--root", corpus, "--out-dir", staged, "--min-chars", "1")
    gold = {"staged": staged}
    _goldset(gold)
    _source_hit(staged, vault, scan.name)
    _review_revision(resumed, corpus, str(gold["marked"]))


def _retry_publication(run: Path, vault: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _render(run, vault)
    expected = _snapshot(vault)
    prior = vault / "previous-render-only.txt"
    prior.touch()
    partial = vault.with_name(f".{vault.name}.partial")
    previous = vault.with_name(f".{vault.name}.trash")
    rmtree = shutil.rmtree

    def interrupt(path: Path, *args: object, **kwargs: object) -> None:
        if path in {partial, previous} and (path / prior.name).is_file():
            assert vault.is_dir() and not prior.exists()
            raise SimulatedKill()
        rmtree(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(swap.shutil, "rmtree", interrupt)
        with pytest.raises(SimulatedKill):
            _render(run, vault)
    assert any((path / prior.name).is_file() for path in (partial, previous))
    _render(run, vault)
    assert not partial.exists() and not previous.exists()
    assert _snapshot(vault) == expected


def _portable(tree: dict[str, bytes]) -> dict[str, bytes]:
    """Compare across run directories while retaining every semantic field."""
    for name in tree:
        if name == "metadata/render.json" or name.endswith(".md.metadata.json"):
            payload = json.loads(tree[name])
            for field in ("run_dir", "acquisition_run_id", "capture_time"):
                payload.pop(field, None)
            tree[name] = json.dumps(payload, sort_keys=True).encode()
    return tree


def _retry_projection(run: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_text = write._write

    def interrupt(path: Path, content: str) -> None:
        if path.name.endswith(".references.json") and any(corpus.glob("*.md")):
            raise SimulatedKill()
        write_text(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(write, "_write", interrupt)
        with pytest.raises(SimulatedKill):
            _project(run, corpus)
    (committed,) = corpus.glob("*.md")
    assert committed.with_suffix(".citations.json").is_file()
    assert committed.with_name(f"{committed.name}.references.json").is_file()
    assert committed.with_name(f"{committed.name}.metadata.json").is_file()
    assert "0 revision(s)" in _project(run, corpus)


def _fallback_pixels(scan: Path, run: Path, vault: Path) -> None:
    with fitz.open(scan) as document:
        for page in range(1, _LONG + 1):
            pid = f"p{page:04d}"
            fallback = vault / "source" / "pages" / f"{pid}.png"
            assert fallback.read_bytes() == (run / "pages" / pid / "render/page.png").read_bytes()
            if page in (FAILED_PAGE, _REPEATED_FOLIO_PAGE, _LONG):
                color = fitz.csRGB if page == _LONG else fitz.csGRAY
                direct = document[page - 1].get_pixmap(dpi=72, colorspace=color)
                assert fitz.Pixmap(fallback).samples == direct.samples
    assert len(list((vault / "source/pages").glob("*.png"))) == _LONG
    assert (vault / "source" / scan.name).read_bytes() == scan.read_bytes()


def _ambiguous(run: Path, vault: Path, corpus: Path) -> None:
    assert read_structure(structure_path(run)).ambiguous_folios == ["18"]
    assert _END in _vault_text(vault)
    records = (
        BookReferences.model_validate_json(path.read_bytes())
        for path in corpus.glob("*.references.json")
    )
    (citation,) = [cite for record in records for cite in record.citations if cite.key == "page 18"]
    assert (citation.outcome, citation.reason, citation.target) == ("unresolved", "ambiguous", None)
    report = json.loads((vault / "metadata/validation.json").read_bytes())
    assert report["ok"] and report["references"]["ambiguous"] == {"page": 1}
    queue = read_queue(queue_path(run))
    write_queue(queue_path(run), queue.model_copy(update={"entries": []}))
    try:
        result = _book("validate", vault, code=1)
        assert QUEUE_STALE in result and MARKER_MISMATCH in result
    finally:
        write_queue(queue_path(run), queue)
    _book("validate", vault)


def _source_hit(staged: Path, vault: Path, source_name: str) -> None:
    records = chunk_corpus(staged, "fixed", 180, 0)
    assert annotate_page_metadata(records, staged) == 1
    index = LexicalIndex.build([record["text"] for record in records])
    [(ordinal, _score)] = index.search("Pyrochlore", k=1)
    hit = records[ordinal]
    assert hit["metadata"]["source_pdf"] == source_name
    assert hit["metadata"]["pages"] == [_LONG, _LONG]
    assert len(list((vault / "content").glob("*.md"))) > 1
    doc = staged / hit["doc_id"]
    citations = json.loads(doc.with_suffix(".citations.json").read_bytes())
    page = next(item for item in citations["pages"] if item["page"] == _LONG)
    text = doc.read_text()
    (block,) = [
        item for item in page["blocks"] if text[item["char_start"] : item["char_end"]] == _END
    ]
    assert hit["char_start"] <= block["char_start"] < hit["char_end"]
    assert block["block_id"].startswith(f"p{_LONG:04d}-")
    note = (vault / block["vault_file"]).read_text()
    assert block["vault_anchor"].startswith(f"p{_LONG:04d}-note-")
    assert f"^{block['vault_anchor']}\n" in note
    assert f"../source/pages/p{_LONG:04d}.png" in note
    x0, y0, x1, y1 = block["region"]
    pixels = images.read_pixels(vault / f"source/pages/p{_LONG:04d}.png")
    height, width = pixels.shape[:2]
    assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
    crop = pixels[int(y0) : int(y1), int(x0) : int(x1)]
    assert np.any(np.all(crop == (175, 94, 8), axis=2))


def _review_revision(run: Path, corpus: Path, prior_doc: str) -> None:
    before = _snapshot(corpus)
    path = queue_path(run)
    queue = read_queue(path)
    index = next(
        i
        for i, entry in enumerate(queue.entries)
        if entry.pdf_page == DISPUTE and entry.reason == "unresolved"
    )
    apply_review_decision(path, index, f"{ENTER_TEXT}:2")
    _book("run", "--resume", run)
    _settle(run)
    assert "1 revision(s)" in _project(run, corpus)
    after = _snapshot(corpus)
    (added,) = [name for name in after.keys() - before.keys() if name.endswith(".md")]
    assert json.loads(after[f"{added}.metadata.json"])["revision_of"] == prior_doc
    assert "The delta is 3.2 kilometres." in after[added].decode()
    changed = [name for name, old in before.items() if after[name] != old]
    (reference_sidecar,) = changed
    assert reference_sidecar.endswith(".references.json")
    assert after[reference_sidecar] == before[reference_sidecar].replace(
        f'"doc_id": "{prior_doc}"'.encode(), f'"doc_id": "{added}"'.encode()
    )
    assert "0 revision(s)" in _project(run, corpus)
    assert _snapshot(corpus) == after
