"""Corpus projection: content, page citations, stable ids, and marker refusal."""

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llb.cli import app
from llb.core.contracts.data_prep.book_resolve import ResolvedSpan
from llb.finetune.dataset import export_finetune_set
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.goldset.validate import validate_items
from llb.core.contracts.data_prep.book_references import BookReferences
from llb.prep.book.project.build import project_corpus
from llb.prep.book.resolve.io import read_resolved, write_resolved
from llb.prep.corpus.fingerprints import corpus_doc_fingerprints
from llb.prep.corpus.ingest import ingest_corpus
from llb.prep.corpus.redistribution import RedistributionRefused
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.page_metadata import annotate_page_metadata
from tests.llb.prep.book.project_sample import _CHAPTER, _sha, sample_run

RUNNER = CliRunner()


def _docs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("book-*.md") if path.is_file())


def _text(root: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _docs(root))


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _admit(run: Path) -> None:
    path = run / "pages" / "p0002" / "resolved.json"
    page = read_resolved(path)
    body = "ALPHA 1 BETA"
    blocks = []
    for block in page.blocks:
        if "ALPHA" not in block.primary_text:
            blocks.append(block)
            continue
        blocks.append(
            block.model_copy(
                update={
                    "text": body,
                    "primary_text": body,
                    "spans": [
                        ResolvedSpan(
                            provenance="primary", text=body, start=0, end=len(body), adopted=True
                        )
                    ],
                }
            )
        )
    write_resolved(path, page.model_copy(update={"blocks": blocks}))


def _item(doc_id: str, phrase: str, corpus: Path) -> GoldItem:
    text = (corpus / doc_id).read_text(encoding="utf-8")
    start = text.index(phrase)
    return GoldItem(
        id="span-" + doc_id[:12],
        lang="en",
        question="What does the section say?",
        reference_answer=phrase,
        source_doc_id=doc_id,
        source_spans=[
            SourceSpan(doc_id=doc_id, char_start=start, char_end=start + len(phrase), text=phrase)
        ],
        provenance="human-authored",
        verified=True,
        split="tuning",
    )


def test_projection_is_content_with_exact_page_spans(tmp_path: Path) -> None:
    run = sample_run(tmp_path)
    out = tmp_path / "corpus"
    result = RUNNER.invoke(
        app,
        [
            "book-ocr",
            "project",
            str(run),
            "--out",
            str(out),
            "--licence",
            "redistributable",
            "--language",
            "en",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "TOK-BODY" not in result.output
    assert "ALPHA" not in result.output
    body = _text(out)
    assert "PN-PAGE-ONE" not in body
    assert "RUNNING-HEADER" not in body
    assert "Previous" not in body
    assert "Contents" not in body
    assert "[!" not in body
    assert "$$" in body and r"\tag{1-1}" in body
    assert "| a | b |" in body
    assert "{{llb-marker:unresolved:" in body
    assert _CHAPTER in body and "TOK-BODY-SEC" in body
    for path in _docs(out):
        text = path.read_text(encoding="utf-8")
        assert not text.startswith("---")
        _assert_sidecar(path, text)
        _assert_references(path, text)
        meta = json.loads(path.with_name(path.name + ".metadata.json").read_text(encoding="utf-8"))
        assert meta["language"] == "en"
        assert meta["source_system"] == "book-ocr"
        assert meta["licence"] == "redistributable"
        assert meta["payload_digest"] == _sha(run / "source" / "book.pdf")
        assert "revision_of" not in meta


def _assert_sidecar(path: Path, text: str) -> None:
    payload = json.loads(path.with_suffix(".citations.json").read_text(encoding="utf-8"))
    covered = ""
    cursor = 0
    assert payload["doc_id"] == path.name
    for page in payload["pages"]:
        assert page["char_start"] == cursor
        covered += text[page["char_start"] : page["char_end"]]
        cursor = page["char_end"]
        for block in page["blocks"]:
            assert text[block["char_start"] : block["char_end"]]
            assert block["vault_file"].startswith("content/") and block["vault_file"].endswith(
                ".md"
            )
            assert block["vault_anchor"]
            assert "region" in block
            for marker in block["markers"]:
                token = text[marker["char_start"] : marker["char_end"]]
                assert token.startswith("{{llb-marker:")
    assert covered == text


def _assert_references(path: Path, text: str) -> None:
    sidecar = path.with_name(path.name + ".references.json")
    record = BookReferences.model_validate_json(sidecar.read_text(encoding="utf-8"))
    assert record.doc_id == path.name
    assert record.hierarchy and record.hierarchy[0].level == "book"
    for item in record.citations:
        assert text[item.cue_start : item.cue_end].strip()
        if item.target is not None:
            target = path.with_name(item.target.doc_id).read_text(encoding="utf-8")
            assert target[item.target.char_start : item.target.char_end].strip()


def test_reprojection_is_identical_and_a_correction_revises(tmp_path: Path) -> None:
    run = sample_run(tmp_path)
    out = tmp_path / "corpus"
    project_corpus(run, out, "redistributable", language="en")
    first = _snapshot(out)
    again = project_corpus(run, out, "redistributable", language="en")
    assert again.revisions == 0
    assert _snapshot(out) == first
    staged = tmp_path / "staged"
    ingest_corpus(out, staged)
    fingerprints = corpus_doc_fingerprints(staged)
    ingest_corpus(out, staged)
    assert corpus_doc_fingerprints(staged) == fingerprints
    records = chunk_corpus(staged, "fixed", 180, 0)
    annotate_page_metadata(records, staged)
    assert records
    assert all(record["metadata"].get("pages") for record in records)
    marked = next(path for path in _docs(out) if "{{llb-marker:unresolved:" in path.read_text())
    kept = marked.read_bytes()
    _admit(run)
    revised = project_corpus(run, out, "redistributable", language="en")
    assert revised.revisions == 1
    assert marked.read_bytes() == kept
    added = [path for path in _docs(out) if path.relative_to(out).as_posix() not in first]
    assert len(added) == 1
    meta = json.loads(added[0].with_name(added[0].name + ".metadata.json").read_text())
    assert meta["revision_of"] == marked.name
    revised_text = added[0].read_text(encoding="utf-8")
    assert "ALPHA 1 BETA" in revised_text
    assert "p0002-b0001" not in revised_text
    assert "{{llb-marker:page-failed:p0003:" in revised_text
    ingested = tmp_path / "revised"
    ingest_corpus(out, ingested)
    assert (ingested / marked.name).is_file()
    good = _item(marked.name, "TOK-BODY-SEC", ingested)
    assert validate_items([good], ingested)["errors"] == []
    token = _marker_token(marked.read_text(encoding="utf-8"))
    refused = validate_items([_item(marked.name, token, ingested)], ingested)
    assert any("unresolved marker" in error for error in refused["errors"])


def _marker_token(text: str) -> str:
    start = text.index("{{llb-marker:unresolved:")
    return text[start : text.index("}}", start) + 2]


def test_local_only_export_refuses_and_missing_licence_refuses(tmp_path: Path) -> None:
    run = sample_run(tmp_path)
    out = tmp_path / "local"
    project_corpus(run, out, "local-only", language="en")
    staged = tmp_path / "staged"
    ingest_corpus(out, staged)
    doc = next(path.name for path in _docs(staged) if "TOK-BODY-SEC" in path.read_text())
    reference = staged / f"{doc}.references.json"
    assert reference.is_file()
    assert reference.read_bytes() == (out / reference.name).read_bytes()
    gold = tmp_path / "gold.jsonl"
    dump_goldset([_item(doc, "TOK-BODY-SEC", staged)], gold)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"run_id": "export", "config": {"corpus_root": str(staged)}}) + "\n",
        encoding="utf-8",
    )
    (bundle / "scores.jsonl").write_text(
        json.dumps({"item_id": "span-" + doc[:12], "objective_score": 1.0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RedistributionRefused, match=doc) as raised:
        export_finetune_set(
            run_dir=bundle, goldset_path=gold, out_dir=tmp_path / "export", corpus_root=staged
        )
    assert "local-only" in str(raised.value)
    export_dir = tmp_path / "export"
    assert not export_dir.exists()
    assert reference.is_file()
    missing = RUNNER.invoke(app, ["book-ocr", "project", str(run), "--out", str(tmp_path / "none")])
    assert missing.exit_code == 2
    with pytest.raises(ValueError, match="required"):
        project_corpus(run, tmp_path / "blank", "  ")
    with pytest.raises(ValueError, match="unknown redistribution class"):
        project_corpus(run, tmp_path / "other", "public-domain")


def test_a_removed_document_takes_its_reference_sidecar(tmp_path: Path) -> None:
    run = sample_run(tmp_path)
    out = tmp_path / "corpus"
    project_corpus(run, out, "redistributable", language="en")
    staged = tmp_path / "staged"
    ingest_corpus(out, staged)
    victim = next(iter(_docs(staged)))
    sidecar = staged / f"{victim.name}.references.json"
    citation = staged / f"{victim.stem}.citations.json"
    assert sidecar.is_file() and citation.is_file()
    slim = tmp_path / "slim"
    shutil.copytree(out, slim)
    (slim / victim.name).unlink()
    removed = ingest_corpus(slim, staged)
    assert victim.name in removed.removed_sources
    assert not victim.exists()
    assert not sidecar.exists()
    assert not citation.exists()
    kept = _docs(staged)
    assert kept
    assert all((staged / f"{path.name}.references.json").is_file() for path in kept)
