"""Source-span audit for external RAG answer logs (external-rag-source-mapping)."""

import csv
import json

import pytest

from llb.scoring.external_rag.run import score_external_rag_file
from llb.scoring.external_rag_source_map import (
    SourceMap,
    SourceMapEntry,
    load_source_map,
    map_source,
)
from llb.scoring.external_rag_sources import (
    audit_row_sources,
    summarize_source_audit,
)

GOLD_SPANS = [{"doc_id": "doc.txt", "char_start": 100, "char_end": 140, "text": "x" * 40}]


def _map(**entries) -> SourceMap:
    return SourceMap(
        by_article_id=entries.get("by_article_id", {}),
        by_url=entries.get("by_url", {}),
        by_title=entries.get("by_title", {}),
    )


# --- loading ---------------------------------------------------------------------------------


def test_load_source_map_reads_jsonl_json_and_csv(tmp_path):
    rows = [
        {"article_id": "a1", "doc_id": "doc.txt", "char_start": 100, "char_end": 150},
        {"url": "https://kb/a2", "doc_id": "other.txt"},
        {"article_title": "Стаття", "doc_id": "third.txt", "char_start": 0, "char_end": 10},
    ]
    jsonl = tmp_path / "map.jsonl"
    jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    as_json = tmp_path / "map.json"
    as_json.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    as_csv = tmp_path / "map.csv"
    with as_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["article_id", "url", "article_title", "doc_id", "char_start", "char_end"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    for path in (jsonl, as_json, as_csv):
        smap = load_source_map(path)
        assert smap.by_article_id["a1"].has_span
        assert smap.by_article_id["a1"].doc_id == "doc.txt"
        assert not smap.by_url["https://kb/a2"].has_span
        assert smap.by_title["Стаття"].char_end == 10


def test_load_source_map_rejects_bad_records(tmp_path):
    no_doc = tmp_path / "bad1.jsonl"
    no_doc.write_text(json.dumps({"article_id": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="doc_id"):
        load_source_map(no_doc)
    no_key = tmp_path / "bad2.jsonl"
    no_key.write_text(json.dumps({"doc_id": "d"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provider key"):
        load_source_map(no_key)


# --- key precedence ---------------------------------------------------------------------------


def test_map_source_key_precedence_id_over_url_over_title():
    smap = _map(
        by_article_id={"a1": SourceMapEntry("by-id.txt")},
        by_url={"https://kb/a1": SourceMapEntry("by-url.txt")},
        by_title={"Титул": SourceMapEntry("by-title.txt")},
    )
    source = {"article_id": "a1", "url": "https://kb/a1", "article_title": "Титул"}
    assert map_source(source, smap).doc_id == "by-id.txt"
    del source["article_id"]
    assert map_source(source, smap).doc_id == "by-url.txt"
    del source["url"]
    assert map_source(source, smap).doc_id == "by-title.txt"
    assert map_source({"article_id": "unknown"}, smap) is None


# --- per-row audit ----------------------------------------------------------------------------


def test_span_overlap_hit_uses_returned_order_rank():
    smap = _map(
        by_article_id={
            "miss": SourceMapEntry("doc.txt", 0, 50),  # same doc, no overlap
            "hit": SourceMapEntry("doc.txt", 120, 160),  # overlaps the gold span
        }
    )
    audit = audit_row_sources([{"article_id": "miss"}, {"article_id": "hit"}], GOLD_SPANS, smap)
    assert audit["source_hit"] == 1.0
    assert audit["source_first_hit_rank"] == 2  # rank counts the returned order
    assert audit["source_hit_weak"] == "false"
    assert audit["source_mapped_count"] == 2 and audit["source_unmapped_count"] == 0


def test_spanless_doc_match_is_weak_evidence():
    smap = _map(by_title={"Стаття": SourceMapEntry("doc.txt")})  # no char range
    audit = audit_row_sources([{"article_title": "Стаття"}], GOLD_SPANS, smap)
    assert audit["source_hit"] == 1.0
    assert audit["source_hit_weak"] == "true"  # doc-level only, never span proof
    assert audit["_source_strong_rank"] is None


def test_title_mapping_with_spans_is_span_proof():
    smap = _map(by_title={"Стаття": SourceMapEntry("doc.txt", 120, 160)})
    audit = audit_row_sources([{"article_title": "Стаття"}], GOLD_SPANS, smap)
    assert audit["source_hit"] == 1.0 and audit["source_hit_weak"] == "false"


def test_unmapped_sources_are_reported_separately():
    smap = _map(by_article_id={"a1": SourceMapEntry("doc.txt", 120, 160)})
    audit = audit_row_sources([{"article_id": "nowhere"}, {"article_id": "a1"}], GOLD_SPANS, smap)
    assert audit["source_unmapped_count"] == 1 and audit["source_mapped_count"] == 1
    assert audit["source_first_hit_rank"] == 2  # unmapped source keeps its position

    total_miss = audit_row_sources([{"article_id": "nowhere"}], GOLD_SPANS, smap)
    assert total_miss["source_hit"] == 0.0  # unmapped-only rows are misses, flagged by count
    assert total_miss["source_unmapped_count"] == 1


def test_row_without_sources_is_not_audited():
    audit = audit_row_sources([], GOLD_SPANS, _map())
    assert audit["source_hit"] == "" and audit["source_hit_weak"] == ""


# --- summary ----------------------------------------------------------------------------------


def test_summary_recall_and_mrr_count_span_proof_hits_only():
    rows = [
        {  # strong hit at rank 1
            "source_hit": 1.0,
            "source_hit_weak": "false",
            "_source_strong_rank": 1,
            "source_mapped_count": 1,
            "source_unmapped_count": 0,
        },
        {  # weak hit: excluded from recall/MRR
            "source_hit": 1.0,
            "source_hit_weak": "true",
            "_source_strong_rank": None,
            "source_mapped_count": 1,
            "source_unmapped_count": 1,
        },
        {  # not audited (no sources returned)
            "source_hit": "",
            "source_hit_weak": "",
            "_source_strong_rank": None,
            "source_mapped_count": 0,
            "source_unmapped_count": 0,
        },
    ]
    summary = summarize_source_audit(rows)
    assert summary["rows_audited"] == 2
    assert summary["source_recall_at_3"] == 0.5
    assert summary["source_mrr"] == 0.5
    assert summary["weak_hit_rows"] == 1
    assert summary["unmapped_sources"] == 1
    assert summary["unmapped_rate"] == pytest.approx(1 / 3, abs=1e-4)


# --- end to end -------------------------------------------------------------------------------


def test_score_external_rag_with_source_map_adds_audit_columns(tmp_path):
    answers = tmp_path / "answered.jsonl"
    answers.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "Що є столицею України?",
                "reference_answer": "Київ",
                "split": "final",
                "source_doc_id": "doc.txt",
                "source_spans": [
                    {"doc_id": "doc.txt", "char_start": 0, "char_end": 4, "text": "Київ"}
                ],
                "llm_answer": "Київ",
                "llm_sources": [
                    {"article_id": "a", "article_title": "столиця", "url": "/kb/a"},
                    {"article_id": "b"},
                ],
                "human_score_0_1": "1.0",
                "human_decision": "accept",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    source_map = tmp_path / "map.jsonl"
    source_map.write_text(
        json.dumps({"article_id": "a", "doc_id": "doc.txt", "char_start": 0, "char_end": 20})
        + "\n",
        encoding="utf-8",
    )

    result = score_external_rag_file(answers, source_map_path=source_map)

    rows = list(csv.DictReader(result.paths.csv.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["source_hit"] == "1.0"
    assert rows[0]["source_first_hit_rank"] == "1"
    assert rows[0]["source_hit_weak"] == "false"
    assert rows[0]["source_unmapped_count"] == "1"  # article b has no mapping
    audit = result.summary["source_audit"]
    assert audit["source_recall_at_3"] == 1.0 and audit["source_mrr"] == 1.0
    report = result.paths.report.read_text(encoding="utf-8")
    assert "## Source-span audit" in report
    assert "unmapped" in report


def test_score_external_rag_without_source_map_keeps_csv_shape(tmp_path):
    answers = tmp_path / "answered.jsonl"
    answers.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "q",
                "reference_answer": "a",
                "llm_answer": "a",
                "human_score_0_1": "1.0",
                "human_decision": "accept",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = score_external_rag_file(answers)
    header = result.paths.csv.read_text(encoding="utf-8").splitlines()[0]
    assert "source_hit" not in header  # audit columns appear only with --source-map
    assert "source_audit" not in result.summary
