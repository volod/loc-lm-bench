"""External-service grounded-JSONL import (external-draft contract Artifact B).

Drives `import_external_draft` over tmp corpora + artifacts (no network): re-grounding, the
open-data sidecar gate, verbatim corpus copy, and item-provenance labels. One test imports the
committed `samples/external-drafts` fixture so it stays valid.
"""

import json

import pytest

from llb.goldset.schema import load_goldset
from llb.goldset.validate import validate_items
from llb.prep.external_draft import (
    GOLDSET_FILENAME,
    ITEM_PROVENANCE_FILENAME,
    PROVENANCE_FILENAME,
    import_external_draft,
)

DOC = "Начальник служби веде облік цінностей.\nАкт приймання складається у трьох примірниках."


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "doc-a.md").write_text(DOC, encoding="utf-8")
    return root


def _artifact(tmp_path, rows, name="grounded.jsonl"):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def _sidecar(tmp_path, classification="open", name="external_provenance.json"):
    path = tmp_path / name
    payload = {
        "service": "claude-projects",
        "service_model": "Claude Opus 4.8",
        "export_date": "2026-07-04",
        "data_classification": classification,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _rows():
    return [
        # exact verbatim substring, inline numeric label honored
        {
            "id": "ext-0001",
            "question": "Скільки примірників акта приймання складається?",
            "reference_answer": "у трьох примірниках",
            "source_doc_id": "doc-a.md",
            "quote": "у трьох примірниках",
            "question_type": "numeric",
            "difficulty": "easy",
        },
        # whitespace-flattened across the doc newline -> re-grounded (repaired), no inline labels
        {
            "id": "ext-0002",
            "question": "Що складає начальник служби після ведення обліку цінностей?",
            "reference_answer": "Акт приймання",
            "source_doc_id": "doc-a.md",
            "quote": "облік цінностей. Акт приймання",
        },
        # paraphrase not present in the doc -> dropped
        {
            "id": "ext-0003",
            "question": "У скількох примірниках складається акт?",
            "reference_answer": "сім примірників",
            "source_doc_id": "doc-a.md",
            "quote": "Акт складається у семи примірниках обовʼязково.",
        },
    ]


def test_import_grounds_repairs_and_drops(tmp_path, corpus):
    out = tmp_path / "bundle"
    result = import_external_draft(_artifact(tmp_path, _rows()), corpus, _sidecar(tmp_path), out)

    assert result.report.loaded == 3
    assert result.report.kept == 2
    assert [d["id"] for d in result.report.dropped] == ["ext-0003"]
    assert result.report.dropped[0]["reason"].startswith("quote is not a verbatim substring")
    assert [r["id"] for r in result.report.repaired] == ["ext-0002"]

    items = load_goldset(out / GOLDSET_FILENAME)
    assert {it.id for it in items} == {"ext-0001", "ext-0002"}
    for it in items:
        assert it.provenance == "frontier-drafted"
        assert it.verified is False
        span = it.source_spans[0]
        # span offsets round-trip to the exact doc substring (re-grounded, not the flattened quote)
        assert DOC[span.char_start : span.char_end] == span.text
    # the repaired item's stored span is the exact doc text (with the newline), not the flat quote
    repaired = next(it for it in items if it.id == "ext-0002")
    assert repaired.source_spans[0].text == "облік цінностей.\nАкт приймання"

    # emitted bundle passes validate-goldset; corpus copy is byte-identical
    assert validate_items(items, out / "corpus")["errors"] == []
    assert (out / "corpus" / "doc-a.md").read_text(encoding="utf-8") == DOC


def test_item_provenance_labels_inline_honored_else_classified(tmp_path, corpus):
    out = tmp_path / "bundle"
    import_external_draft(_artifact(tmp_path, _rows()), corpus, _sidecar(tmp_path), out)
    labels = {
        r["id"]: r
        for r in (
            json.loads(line)
            for line in (out / ITEM_PROVENANCE_FILENAME).read_text(encoding="utf-8").splitlines()
        )
    }
    assert labels["ext-0001"]["question_type"] == "numeric"  # inline label honored
    assert labels["ext-0001"]["difficulty"] == "easy"
    # ext-0002 had no inline labels -> classified deterministically (valid closed-set values)
    assert labels["ext-0002"]["question_type"] in {
        "factoid",
        "definition",
        "procedural",
        "numeric",
        "comparative",
    }
    assert labels["ext-0002"]["difficulty"] in {"easy", "medium", "hard"}


def test_provenance_records_service_and_classification(tmp_path, corpus):
    out = tmp_path / "bundle"
    import_external_draft(_artifact(tmp_path, _rows()), corpus, _sidecar(tmp_path), out)
    prov = json.loads((out / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert prov["service"] == "claude-projects"
    assert prov["service_model"] == "Claude Opus 4.8"
    assert prov["data_classification"] == "open"
    assert prov["provenance"] == "frontier-drafted" and prov["verified"] is False
    assert prov["question_type_distribution"]["numeric"] == 1


def test_unknown_source_doc_is_dropped(tmp_path, corpus):
    rows = [
        {
            "id": "ext-x",
            "question": "Питання про невідомий документ тут точно є?",
            "reference_answer": "невідомо",
            "source_doc_id": "missing.md",
            "quote": "у трьох примірниках",
        }
    ]
    out = tmp_path / "bundle"
    with pytest.raises(SystemExit, match="no verbatim-grounded items"):
        import_external_draft(_artifact(tmp_path, rows), corpus, _sidecar(tmp_path), out)
    assert not out.exists()  # nothing written when nothing imports


def test_missing_sidecar_aborts_and_writes_no_bundle(tmp_path, corpus):
    out = tmp_path / "bundle"
    with pytest.raises(SystemExit, match="required sidecar"):
        import_external_draft(_artifact(tmp_path, _rows()), corpus, tmp_path / "nope.json", out)
    assert not out.exists()


def test_non_open_sidecar_aborts_and_writes_no_bundle(tmp_path, corpus):
    out = tmp_path / "bundle"
    sidecar = _sidecar(tmp_path, classification="internal")
    with pytest.raises(SystemExit, match='data_classification: "open"'):
        import_external_draft(_artifact(tmp_path, _rows()), corpus, sidecar, out)
    assert not out.exists()


class _FakeRetriever:
    """Retrieves a chunk overlapping ext-0001's gold span; misses every other question."""

    def __init__(self):
        start = DOC.find("у трьох примірниках")
        self._hit_question = "Скільки примірників акта приймання складається?"
        self._hit_chunk = {
            "doc_id": "doc-a.md",
            "char_start": start,
            "char_end": start + 10,
            "text": DOC[start : start + 10],
        }

    def retrieve(self, question, k):
        if question == self._hit_question:
            return [self._hit_chunk]
        return [{"doc_id": "other.md", "char_start": 0, "char_end": 1, "text": ""}]


def test_import_annotates_retrieval_rank_with_an_index(tmp_path, corpus):
    """external-import-needle-parity: imported items reach the verify gate with the needle signal."""
    out = tmp_path / "bundle"
    result = import_external_draft(
        _artifact(tmp_path, _rows()),
        corpus,
        _sidecar(tmp_path),
        out,
        retriever=_FakeRetriever(),
    )
    rows = [
        json.loads(line)
        for line in (out / ITEM_PROVENANCE_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["id"]: row for row in rows}
    assert by_id["ext-0001"]["retrieval_rank"] == 1
    assert by_id["ext-0002"]["retrieval_rank"] is None  # a miss stays, rank null
    assert all(row["retrieval_k"] == 10 for row in rows)
    prov = json.loads((out / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert prov["needle_retrieval"]["enabled"] is True
    assert result.report.kept == 2


def test_import_drop_nonretrievable_is_explicit_opt_in(tmp_path, corpus):
    out = tmp_path / "bundle"
    result = import_external_draft(
        _artifact(tmp_path, _rows()),
        corpus,
        _sidecar(tmp_path),
        out,
        retriever=_FakeRetriever(),
        drop_nonretrievable_needles=True,
    )
    items = load_goldset(out / GOLDSET_FILENAME)
    assert [it.id for it in items] == ["ext-0001"]  # the retrieval miss was dropped
    dropped = {d["id"]: d["reason"] for d in result.report.dropped}
    assert "ext-0002" in dropped and "top-10" in dropped["ext-0002"]


def test_import_without_index_is_an_exact_no_op(tmp_path, corpus):
    out = tmp_path / "bundle"
    import_external_draft(_artifact(tmp_path, _rows()), corpus, _sidecar(tmp_path), out)
    rows = [
        json.loads(line)
        for line in (out / ITEM_PROVENANCE_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert all("retrieval_rank" not in row for row in rows)
    prov = json.loads((out / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert "needle_retrieval" not in prov


def test_drop_flag_requires_an_index(tmp_path, corpus):
    with pytest.raises(SystemExit, match="retrieval-index-dir"):
        import_external_draft(
            _artifact(tmp_path, _rows()),
            corpus,
            _sidecar(tmp_path),
            tmp_path / "bundle",
            drop_nonretrievable_needles=True,
        )


def test_committed_fixture_imports(tmp_path):
    from llb.core.paths import PROJECT_ROOT

    fixture = PROJECT_ROOT / "samples" / "external-drafts" / "claude-projects-open"
    corpus = PROJECT_ROOT / "samples" / "goldsets" / "ip_regulation_uk" / "corpus"
    out = tmp_path / "bundle"
    result = import_external_draft(
        fixture / "grounded_draft.jsonl",
        corpus,
        fixture / "external_provenance.json",
        out,
    )
    assert result.report.loaded == 5 and result.report.kept == 3
    assert len(result.report.dropped) == 2  # non-verbatim + unknown doc
    assert result.validation["errors"] == []
