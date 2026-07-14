"""Tests for ontology retrieval draft."""

import json
from llb.goldset.schema import load_goldset
from llb.prep.ontology.artifacts.report import write_calibration_artifacts
from llb.prep.ontology.constants import (
    NEEDLE_GOLDSET_FILENAME,
)
from llb.prep.ontology.models import (
    DocExtraction,
    DocRecord,
    Entity,
    OntologyCandidate,
)
from llb.prep.ontology.needles import annotate_needle_retrieval
from tests.llb.prep.ontology._ontology_fixtures import DOC1, DOC2
from test_ontology_draft import FakeNeedleRetriever, _needle_item


def test_annotate_needle_retrieval_flags_rank_and_misses():
    hit_item = _needle_item("q1", "Де згадано Київ?", "a.md", DOC1, "Київ")
    miss_item = _needle_item("q2", "Де згадано Львів?", "b.md", DOC2, "Львів")
    retriever = FakeNeedleRetriever(
        {
            hit_item.question: [
                {"doc_id": "other.md", "char_start": 0, "char_end": 5, "text": "other"},
                {"doc_id": "a.md", "char_start": 0, "char_end": 6, "text": "# Київ"},
            ],
            miss_item.question: [
                {"doc_id": "other.md", "char_start": 0, "char_end": 5, "text": "other"},
            ],
        }
    )

    rows, report = annotate_needle_retrieval(
        [hit_item, miss_item], retriever, k=2, drop_nonretrievable=False
    )

    assert [row["retrieval_rank"] for row in rows] == [2, None]
    assert all(row["retrieval_k"] == 2 for row in rows)
    assert report["retrievable_items"] == 1
    assert report["missed_items"] == 1
    assert report["retrievable_fraction"] == 0.5
    assert report["missed_ids"] == ["q2"]


def test_annotate_needle_retrieval_can_drop_misses():
    hit_item = _needle_item("q1", "Де згадано Київ?", "a.md", DOC1, "Київ")
    miss_item = _needle_item("q2", "Де згадано Львів?", "b.md", DOC2, "Львів")
    retriever = FakeNeedleRetriever(
        {
            hit_item.question: [
                {"doc_id": "a.md", "char_start": 0, "char_end": 6, "text": "# Київ"},
            ]
        }
    )

    rows, report = annotate_needle_retrieval(
        [hit_item, miss_item], retriever, k=1, drop_nonretrievable=True
    )

    assert [row["id"] for row in rows] == ["q1"]
    assert report["dropped_items"] == 1


def test_calibration_artifacts_write_retrieval_ranked_needles(tmp_path):
    out = tmp_path / "bundle"
    corpus = out / "corpus"
    corpus.mkdir(parents=True)
    doc_id = "a.md"
    corpus_text = DOC1
    (corpus / doc_id).write_text(corpus_text, encoding="utf-8")
    citation = {
        "kind": "pdf-citations",
        "source": "source.pdf",
        "doc_id": doc_id,
        "parser": "test",
        "pages": [
            {
                "page": 1,
                "text_start": 0,
                "text_end": len(corpus_text),
                "parser": "test",
                "blocks": [],
            }
        ],
    }
    (corpus / "a.citations.json").write_text(
        json.dumps(citation, ensure_ascii=False), encoding="utf-8"
    )
    item = _needle_item("q1", "Де згадано Київ?", doc_id, corpus_text, "Київ")
    span = item.source_spans[0]
    extraction = DocExtraction(
        doc_id=doc_id,
        entities=[Entity(name="Київ", type="LOC", mentions=[span])],
    )
    retriever = FakeNeedleRetriever(
        {
            item.question: [
                {"doc_id": doc_id, "char_start": 0, "char_end": 6, "text": "# Київ"},
            ]
        }
    )

    report = write_calibration_artifacts(
        out,
        [DocRecord(doc_id=doc_id, text=corpus_text, sha256="x", n_chars=len(corpus_text))],
        [extraction],
        OntologyCandidate(),
        [item],
        elapsed_s=0.0,
        settings={},
        retrieval_store=retriever,
        retrieval_k=3,
    )

    raw_rows = [
        json.loads(line)
        for line in (out / NEEDLE_GOLDSET_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    assert raw_rows[0]["retrieval_rank"] == 1
    assert raw_rows[0]["retrieval_k"] == 3
    assert load_goldset(out / NEEDLE_GOLDSET_FILENAME)[0].id == "q1"
    assert report["citation_valid_needle_items"] == 1
    assert report["needle_items_written"] == 1
    assert report["retrieval_unique_needle_items"] == 1
    assert report["retrieval_unique_needle_fraction"] == 1.0
    assert report["gates"]["has_retrieval_unique_needles"] is True
