"""Tests for curate grounded."""

import json
from llb.prep.curation import dispatcher as curation
from curation_helpers import DOC, _grounded_file, corpus as corpus


def test_write_curated_emits_artifact_and_report(tmp_path):
    out = tmp_path / "merged" / "cases.json"
    report = curation.CurationReport(kind="security")
    report.kept = 1
    report_path = curation.write_curated("security", [{"id": "x"}], out, report)
    assert json.loads(out.read_text(encoding="utf-8")) == [{"id": "x"}]
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["kind"] == "security" and persisted["kept"] == 1

    chains_out = tmp_path / "merged" / "chains.jsonl"
    curation.write_curated("chains", [{"chain_id": "c1"}, {"chain_id": "c2"}], chains_out, report)
    lines = chains_out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["chain_id"] for line in lines] == ["c1", "c2"]


def test_grounded_merge_reground_filter_dedup(tmp_path, corpus):
    a = _grounded_file(
        tmp_path,
        "claude.jsonl",
        [
            {
                "id": "ext-claude-0001",
                "question": "Хто призначається відповідальною особою за облік?",
                "source_doc_id": "doc-a.md",
                "quote": "Відповідальною особою призначається начальник служби",
            },
            # whitespace-flattened across the doc newline -> re-snapped to exact corpus text
            {
                "id": "ext-claude-0002",
                "question": "Про облік яких цінностей ідеться у загальних положеннях розділу?",
                "source_doc_id": "doc-a.md",
                "quote": "матеріальних цінностей. Відповідальною особою",
            },
            # quote not in the doc -> invalid, dropped
            {
                "id": "ext-claude-0003",
                "question": "У скількох примірниках складається акт приймання справ?",
                "source_doc_id": "doc-a.md",
                "quote": "у семи примірниках",
            },
        ],
    )
    b = _grounded_file(
        tmp_path,
        "gemini.jsonl",
        [
            # exact duplicate question of ext-claude-0001 -> dropped as exact-dup
            {
                "id": "ext-gemini-0001",
                "question": "Хто призначається відповідальною особою за облік?",
                "source_doc_id": "doc-a.md",
                "quote": "начальник служби",
            }
        ],
    )
    payload, report = curation.curate("grounded", [a, b], corpus_root=corpus)

    assert report.kept == 2
    assert {r["id"] for r in payload} == {"ext-claude-0001", "ext-claude-0002"}
    assert len(report.invalid) == 1 and "not a verbatim substring" in report.invalid[0]["reason"]
    assert len(report.exact_duplicates) == 1
    # the flattened quote was re-snapped to exact corpus text
    repaired = next(r for r in payload if r["id"] == "ext-claude-0002")
    assert repaired["quote"] in DOC


def test_grounded_is_a_curation_kind_and_writes_jsonl(tmp_path):
    assert "grounded" in curation.KINDS and "grounded" in curation.JSONL_KINDS
    out = tmp_path / "merged" / "grounded.jsonl"
    curation.write_curated(
        "grounded", [{"id": "g1"}, {"id": "g2"}], out, curation.CurationReport(kind="grounded")
    )
    lines = out.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["g1", "g2"]
