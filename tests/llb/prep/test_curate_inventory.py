"""Tests for curate inventory."""

import json
from llb.prep.curation import dispatcher as curation
from llb.prep.curation.coverage_text import coverage_plan_to_text, write_coverage_plan_text
from curation_helpers import corpus as corpus


def test_inventory_merge_normalizes_types_and_grounds_quotes(tmp_path, corpus):
    inv1 = {
        "documents": [
            {
                "doc": "doc-a.md",
                "topics": ["облік цінностей"],
                "entities": [
                    {
                        "name": "начальник служби",
                        "type": "ROLE",
                        "mentions": 2,
                        "quote": "Відповідальною особою призначається начальник служби",
                    },
                ],
                "relations": [
                    {
                        "subject": "начальник служби",
                        "relation": "відповідає за",
                        "object": "облік",
                        "quote": "Відповідальною особою призначається начальник служби",
                    },
                ],
                "numeric_facts": [
                    {
                        "fact": "передача триває п'ять робочих днів",
                        "quote": "Передача здійснюється протягом п'яти робочих днів",
                    },
                ],
                "sensitive_topics": ["матеріальна відповідальність"],
            }
        ],
        "cross_document": [
            {"entity_or_topic": "акт приймання", "docs": ["doc-a.md"], "note": "спільна тема"}
        ],
    }
    inv2 = {
        "documents": [
            {
                "doc": "doc-a.md",
                "topics": ["Облік цінностей", "акти приймання"],  # first is a case-dup
                "entities": [
                    # same entity, higher mentions -> merged, mentions=max
                    {
                        "name": "Начальник служби",
                        "type": "PERSON",
                        "mentions": 5,
                        "quote": "Відповідальною особою призначається начальник служби",
                    },
                    # quote not in doc -> entity dropped
                    {
                        "name": "фантомна сутність",
                        "type": "ORG",
                        "mentions": 1,
                        "quote": "цитати немає",
                    },
                ],
                "relations": [],
                "numeric_facts": [],
                "sensitive_topics": [],
            },
            {
                "doc": "ghost.md",
                "topics": [],
                "entities": [],
                "relations": [],
                "numeric_facts": [],
                "sensitive_topics": [],
            },  # not in corpus -> invalid
        ],
        "cross_document": [
            {"entity_or_topic": "Акт приймання", "docs": ["doc-a.md", "doc-b.md"], "note": "x"}
        ],
    }
    p1 = tmp_path / "inv1.json"
    p2 = tmp_path / "inv2.json"
    p1.write_text(json.dumps(inv1, ensure_ascii=False), encoding="utf-8")
    p2.write_text(json.dumps(inv2, ensure_ascii=False), encoding="utf-8")

    payload, report = curation.curate("inventory", [p1, p2], corpus_root=corpus)
    assert len(payload["documents"]) == 1
    doc = payload["documents"][0]
    assert doc["topics"] == ["облік цінностей", "акти приймання"]
    # ROLE normalized to PERSON, so both inventories merged into one entity with mentions=max
    assert len(doc["entities"]) == 1
    assert doc["entities"][0]["type"] == "PERSON" and doc["entities"][0]["mentions"] == 5
    assert len(doc["relations"]) == 1 and len(doc["numeric_facts"]) == 1
    link = payload["cross_document"][0]
    assert link["docs"] == ["doc-a.md", "doc-b.md"]
    reasons = [r["reason"] for r in report.invalid]
    assert "document not in corpus" in reasons and "quote not found in document" in reasons


def test_inventory_accepts_array_of_response_objects(tmp_path, corpus):
    """NotebookLM continuation batches may be saved as [{response 1}, {response 2}, ...]."""
    batch1 = {
        "documents": [
            {
                "doc": "doc-a.md",
                "topics": ["облік цінностей"],
                "entities": [
                    {
                        "name": "головний бухгалтер",
                        "type": "PERSON",
                        "mentions": 2,
                        "quote": "Відповідальною особою призначається головний бухгалтер",
                    }
                ],
                "relations": [],
                "numeric_facts": [],
                "sensitive_topics": [],
            }
        ],
        "cross_document": [],
    }
    batch2 = {
        "documents": [
            {
                "doc": "doc-a.md",
                "topics": ["акти приймання"],
                "entities": [],
                "relations": [],
                "numeric_facts": [
                    {
                        "fact": "акт приймання складається у трьох примірниках",
                        "quote": "Акт приймання складається у трьох примірниках.",
                    }
                ],
                "sensitive_topics": ["матеріальна відповідальність"],
            }
        ],
        "cross_document": [
            {"entity_or_topic": "акт приймання", "docs": ["doc-a.md"], "note": "same doc"}
        ],
    }
    path = tmp_path / "notebooklm-inventory.json"
    path.write_text(json.dumps([batch1, batch2], ensure_ascii=False), encoding="utf-8")

    payload, report = curation.curate("inventory", [path], corpus_root=corpus)

    assert report.sources[str(path)] == 2
    assert report.loaded == 2
    assert len(payload["documents"]) == 1
    doc = payload["documents"][0]
    assert doc["topics"] == ["облік цінностей", "акти приймання"]
    assert len(doc["entities"]) == 1
    assert len(doc["numeric_facts"]) == 1
    assert doc["sensitive_topics"] == ["матеріальна відповідальність"]
    assert payload["cross_document"][0]["docs"] == ["doc-a.md"]


def test_coverage_plan_to_text_renders_notebooklm_source():
    coverage = {
        "documents": [
            {
                "doc": "doc-a.md",
                "topics": ["облік цінностей"],
                "entities": [
                    {
                        "name": "головний бухгалтер",
                        "type": "PERSON",
                        "mentions": 2,
                        "quote": "головний\nбухгалтер",
                    }
                ],
                "relations": [
                    {
                        "subject": "акт",
                        "relation": "складається",
                        "object": "у трьох примірниках",
                        "quote": "Акт приймання складається у трьох примірниках.",
                    }
                ],
                "numeric_facts": [
                    {
                        "fact": "передача протягом п'яти днів",
                        "quote": "протягом п'яти робочих днів",
                    }
                ],
                "sensitive_topics": ["матеріальна відповідальність"],
            }
        ],
        "cross_document": [
            {"entity_or_topic": "акт приймання", "docs": ["doc-a.md"], "note": "same doc"}
        ],
    }

    text = coverage_plan_to_text(coverage)

    assert "Coverage plan" in text
    assert "Document: doc-a.md" in text
    assert "Topics:\n- облік цінностей" in text
    assert 'quote: "головний бухгалтер"' in text
    assert "Cross-document links:" in text
    assert "docs: doc-a.md" in text


def test_write_coverage_plan_text_uses_default_txt_path(tmp_path):
    coverage = {
        "documents": [
            {
                "doc": "doc-a.md",
                "topics": ["облік цінностей"],
                "entities": [],
                "relations": [],
                "numeric_facts": [],
                "sensitive_topics": [],
            }
        ],
        "cross_document": [],
    }
    source = tmp_path / "coverage-doc-a.md.json"
    source.write_text(json.dumps(coverage, ensure_ascii=False), encoding="utf-8")

    result = write_coverage_plan_text(source)

    assert result.path == tmp_path / "coverage-doc-a.md.txt"
    assert result.documents == 1
    assert result.cross_document_links == 0
    assert "Document: doc-a.md" in result.path.read_text(encoding="utf-8")
