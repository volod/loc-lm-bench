"""Tests for external draft retrieval."""

import json
import pytest
from llb.goldset.schema import load_goldset
from llb.prep.external_draft import (
    GOLDSET_FILENAME,
    ITEM_PROVENANCE_FILENAME,
    PROVENANCE_FILENAME,
    import_external_draft,
)
from test_external_draft import _FakeRetriever, _artifact, _rows, _sidecar, corpus as corpus


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
