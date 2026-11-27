"""Vault, corpus, gold-span, and reviewer checks. The test names the seam that failed."""

import json
from collections.abc import Callable
from pathlib import Path

from llb.core.contracts.data_prep.book_references import BookReferences, PrintedCitation
from llb.goldset.schema import GoldItem, SourceSpan
from llb.goldset.validate import validate_items
from llb.prep.book.project.build import project_corpus
from llb.prep.book.render.build import RenderOptions, build_vault
from llb.prep.book.render.constants import CONTENT_DIRNAME, TOC_FILENAME
from llb.prep.book.render.names import anchor_map, book_slug
from llb.prep.book.review.build import build_review_queue
from llb.prep.book.review.constants import ENTER_TEXT
from llb.prep.book.review.decide import apply_review_decision
from llb.prep.book.review.io import queue_path
from llb.prep.book.runner import resume_run
from llb.prep.book.structure.build import build_structure
from llb.prep.book.structure.io import read_structure, structure_path
from llb.prep.book.validate.check import validate_book
from llb.prep.corpus.ingest import ingest_corpus
from llb.rag.chunking.corpus import chunk_corpus
from llb.rag.page_metadata import annotate_page_metadata
from tests.llb.prep.book.synthetic_book import DISPUTE, DISPUTE_TEXT, FIGURE_CAPTION, TABLE_CAPTION

_State = dict[str, object]
_Step = Callable[[_State], None]


def publish_steps() -> tuple[tuple[str, _Step], ...]:
    return (
        ("render", _render),
        ("validate", _validate),
        ("project", _project),
        ("references", _references),
        ("ingest-corpus", _ingest),
        ("validate-goldset", _goldset),
        ("review", _review),
    )


def _render(state: _State) -> None:
    resumed = _publish(_run(state))
    clean = _publish(_clean(state))
    assert _vault_bytes(resumed) == _vault_bytes(clean)
    text = _vault_text(resumed)
    assert "E = mc^2" in text and "\\tag{2-1}" in text
    table_head, table_tail = TABLE_CAPTION.split("2-1", 1)
    figure_head, figure_tail = FIGURE_CAPTION.split("2-1", 1)
    assert table_head in text and table_tail in text and "| Tank | 1200 |" in text
    assert figure_head in text and figure_tail in text and "![[" in text
    assert "[!unreadable]" in text
    assert "[!page-failed]" in text
    assert "{{llb-marker:" not in text
    state["vault"] = resumed
    state["vault-bytes"] = _vault_bytes(resumed)


def _validate(state: _State) -> None:
    report = validate_book(_vault(state), _run(state))
    assert report.ok, [(item.check, item.file, item.detail) for item in report.findings]


def _project(state: _State) -> None:
    out = _run(state).parent / "corpus"
    result = project_corpus(_run(state), out, "redistributable", language="en", slug="scan")
    assert result.documents
    text = _corpus_text(out)
    assert "{{llb-marker:unresolved:" in text
    assert "{{llb-marker:page-failed:" in text
    state["corpus"] = out
    state["corpus-bytes"] = _snapshot(out)


def _references(state: _State) -> None:
    corpus = _corpus(state)
    vault = _vault_text(_vault(state))
    anchors = anchor_map(read_structure(structure_path(_run(state))))
    resolved = 0
    for path in corpus.glob("*.references.json"):
        record = BookReferences.model_validate_json(path.read_text(encoding="utf-8"))
        doc = (corpus / record.doc_id).read_text(encoding="utf-8")
        resolved += _citations(corpus, doc, vault, anchors, record)
        _index(doc, record)
    assert resolved >= 2
    assert "Figure 9-9" in _corpus_text(corpus)


def _citations(
    corpus: Path, doc: str, vault: str, anchors: dict[str, str], record: BookReferences
) -> int:
    resolved = 0
    for item in record.citations:
        assert doc[item.cue_start : item.cue_end].strip()
        if item.outcome == "unresolved":
            assert item.reason in {"ambiguous", "gap", "absent"} and item.target is None
            continue
        resolved += _resolved_citation(corpus, vault, anchors, item)
    return resolved


def _resolved_citation(
    corpus: Path, vault: str, anchors: dict[str, str], item: PrintedCitation
) -> int:
    assert item.target is not None
    text = (corpus / item.target.doc_id).read_text(encoding="utf-8")
    assert text[item.target.char_start : item.target.char_end].strip()
    if item.kind == "index":
        return 1
    anchor = anchors.get(item.target.block_id)
    assert anchor and f"#^{anchor}" in vault
    return 1


def _index(doc: str, record: BookReferences) -> None:
    for entry in record.index:
        if doc[entry.term_start : entry.term_end] != "transfer":
            continue
        assert any(page.printed == "11" and page.pdf_page == DISPUTE for page in entry.pages)


def _ingest(state: _State) -> None:
    staged = _run(state).parent / "staged"
    ingest_corpus(_corpus(state), staged, min_chars=1, default_language="en")
    records = chunk_corpus(staged, "fixed", 180, 0)
    annotate_page_metadata(records, staged)
    assert records
    assert all(record["metadata"].get("pages") for record in records)
    state["staged"] = staged


def _goldset(state: _State) -> None:
    staged = _staged(state)
    marked = next(
        path for path in staged.glob("book-*.md") if "{{llb-marker:unresolved:" in path.read_text()
    )
    text = marked.read_text(encoding="utf-8")
    start = text.index("{{llb-marker:unresolved:")
    token = text[start : text.index("}}", start) + 2]
    item = GoldItem(
        id="open-span",
        lang="en",
        question="What is the open span?",
        reference_answer=token,
        source_doc_id=marked.name,
        source_spans=[
            SourceSpan(
                doc_id=marked.name, char_start=start, char_end=start + len(token), text=token
            )
        ],
        provenance="human-authored",
        verified=True,
        split="tuning",
    )
    report = validate_items([item], staged)
    assert any("unresolved marker" in error for error in report["errors"])
    state["marked"] = marked.name


def _review(state: _State) -> None:
    run = _run(state)
    index = int(state["entry"])
    apply_review_decision(queue_path(run), index, f"{ENTER_TEXT}:2")
    resume_run(run, registry=_registry(state))
    vault = build_vault(run, RenderOptions(slug=book_slug("scan.pdf")))
    validate_book(vault, run)
    after = _vault_bytes(vault)
    before = dict(state["vault-bytes"])
    changed = [name for name in before if before[name] != after.get(name)]
    content = [name for name in changed if name.startswith("content/")]
    assert len(content) == 1
    assert all(
        name.startswith(f"review/p{DISPUTE:04d}-") for name in changed if name not in content
    )
    assert "[!page-failed]" in _vault_text(vault)
    assert DISPUTE_TEXT in _vault_text(vault)
    corpus = _corpus(state)
    prior = dict(state["corpus-bytes"])
    revised = project_corpus(run, corpus, "redistributable", language="en", slug="scan")
    assert revised.revisions == 1
    added = [path for path in _docs(corpus) if path.relative_to(corpus).as_posix() not in prior]
    assert len(added) == 1
    meta = json.loads(
        added[0].with_name(added[0].name + ".metadata.json").read_text(encoding="utf-8")
    )
    assert meta["revision_of"] == state["marked"]
    assert DISPUTE_TEXT in added[0].read_text(encoding="utf-8")
    assert "{{llb-marker:page-failed:" in _corpus_text(corpus)
    ingested = run.parent / "revised"
    ingest_corpus(corpus, ingested, min_chars=1, default_language="en")
    assert (ingested / str(state["marked"])).is_file()
    assert (ingested / added[0].name).is_file()


def _publish(run: Path) -> Path:
    build_review_queue(run)
    build_structure(run)
    return build_vault(run, RenderOptions(slug=book_slug("scan.pdf")))


def _vault_bytes(vault: Path) -> dict[str, bytes]:
    # render.json records the absolute run directory, so that one field is set aside.
    skip = {"metadata/processing_report.md", "metadata/validation.json"}
    found = {}
    for path in sorted(vault.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(vault).as_posix()
        if rel in skip:
            continue
        data = path.read_bytes()
        if rel == "metadata/render.json":
            payload = json.loads(data)
            payload.pop("run_dir", None)
            data = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
        found[rel] = data
    return found


def _vault_text(vault: Path) -> str:
    notes = [vault / TOC_FILENAME, *sorted((vault / CONTENT_DIRNAME).glob("*.md"))]
    return "\n".join(path.read_text(encoding="utf-8") for path in notes if path.is_file())


def _corpus_text(root: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _docs(root))


def _docs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("book-*.md") if path.is_file())


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run(state: _State) -> Path:
    return state["run"]  # type: ignore[return-value]


def _clean(state: _State) -> Path:
    return state["clean"]  # type: ignore[return-value]


def _registry(state: _State) -> object:
    return state["registry"]


def _vault(state: _State) -> Path:
    return state["vault"]  # type: ignore[return-value]


def _corpus(state: _State) -> Path:
    return state["corpus"]  # type: ignore[return-value]


def _staged(state: _State) -> Path:
    return state["staged"]  # type: ignore[return-value]
