"""Unified mixed-corpus ingestion: text passthrough + reuse, PDF routing, skip diagnostics.

The PDF lane is exercised with an INJECTED extractor over dummy `.pdf` files, so these tests need
neither PyMuPDF nor real PDFs: `ingest_pdf_corpus` tolerates an unreadable source when an extractor
is supplied (diagnostics fall back gracefully).
"""

import json

import pytest

from llb.prep.corpus_ingest import CORPUS_MANIFEST, ingest_corpus

MD_DOC = "# Розділ\n\n" + ("Це достатньо довгий український документ. " * 20)
TXT_DOC = "Це текстовий документ про кругообіг води у природі. " * 20


def _manifest(out_dir):
    return json.loads((out_dir / CORPUS_MANIFEST).read_text(encoding="utf-8"))


def test_ingest_corpus_text_passthrough_and_reuse(tmp_path):
    root = tmp_path / "src"
    (root / "nested").mkdir(parents=True)
    (root / "a.md").write_text(MD_DOC, encoding="utf-8")
    (root / "nested" / "b.txt").write_text(TXT_DOC, encoding="utf-8")
    out = tmp_path / "out"

    result = ingest_corpus(root, out, min_chars=50)

    # both text docs pass through verbatim under their relative path
    assert result.n_docs == 2 and result.n_skipped == 0
    assert (out / "a.md").read_text(encoding="utf-8") == MD_DOC
    assert (out / "nested" / "b.txt").read_text(encoding="utf-8") == TXT_DOC
    manifest = _manifest(out)
    assert manifest["kind"] == "corpus" and manifest["n_docs"] == 2
    kinds = {item["source"]: item["kind"] for item in manifest["items"]}
    assert kinds == {"a.md": "text", "nested/b.txt": "text"}
    assert all(item["source_sha256"] for item in manifest["items"])

    # a rerun over the unchanged corpus reuses every document
    rerun = ingest_corpus(root, out, min_chars=50)
    assert rerun.n_reused == 2 and rerun.n_docs == 2
    assert all(item.reused for item in rerun.items)


def test_ingest_corpus_reconverts_changed_text(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    doc = root / "a.md"
    doc.write_text(MD_DOC, encoding="utf-8")
    out = tmp_path / "out"
    ingest_corpus(root, out, min_chars=50)

    doc.write_text(MD_DOC + "\n\nНовий абзац із додатковим змістом для тесту.", encoding="utf-8")
    rerun = ingest_corpus(root, out, min_chars=50)
    assert rerun.n_reused == 0
    assert (out / "a.md").read_text(encoding="utf-8").endswith("для тесту.")


def test_ingest_corpus_routes_pdf_through_injected_extractor(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "manual.pdf").write_bytes(b"%PDF-1.4 not-a-real-pdf")
    (root / "notes.md").write_text(MD_DOC, encoding="utf-8")
    out = tmp_path / "out"

    extracted_text = "Витягнутий текст із PDF документа. " * 10

    def fake_extractor(_pdf_path, _parser):
        return extracted_text

    result = ingest_corpus(root, out, min_chars=50, extractor=fake_extractor)

    manifest = _manifest(out)
    by_source = {item["source"]: item for item in manifest["items"]}
    assert by_source["manual.pdf"]["kind"] == "pdf"
    assert by_source["manual.pdf"]["status"] == "ok"
    assert by_source["manual.pdf"]["doc_id"].startswith("pdf-")
    assert by_source["notes.md"]["kind"] == "text"
    # the converted PDF markdown and the passthrough text both land in the corpus
    assert (out / by_source["manual.pdf"]["doc_id"]).is_file()
    assert (out / "notes.md").is_file()
    assert result.n_docs == 2


def test_ingest_corpus_skips_too_short_text(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "long.md").write_text(MD_DOC, encoding="utf-8")
    (root / "tiny.txt").write_text("замало", encoding="utf-8")
    out = tmp_path / "out"

    result = ingest_corpus(root, out, min_chars=100)

    assert result.n_docs == 1 and result.n_skipped == 1
    tiny = next(item for item in result.items if item.source == "tiny.txt")
    assert tiny.status == "too_short" and tiny.doc_id is None
    assert not (out / "tiny.txt").exists()


def test_ingest_corpus_excludes_out_dir_subtree(tmp_path):
    # default out dir is <root>/_md; a rerun must not re-ingest its own staged copies as new sources
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.md").write_text(MD_DOC, encoding="utf-8")

    first = ingest_corpus(root, min_chars=50)  # out_dir defaults to root/_md
    assert first.out_dir == root / "_md"
    assert first.n_docs == 1

    rerun = ingest_corpus(root, min_chars=50)
    # only the original source is listed (the staged _md/a.md is excluded), and it is reused
    assert [item.source for item in rerun.items] == ["a.md"]
    assert rerun.n_reused == 1


def test_ingest_corpus_empty_raises(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    (root / "note.rst").write_text("unsupported format", encoding="utf-8")
    with pytest.raises(ValueError, match="no .txt/.md/.pdf"):
        ingest_corpus(root, tmp_path / "out")
