"""Tests for ontology refine yield."""

import json
from llb.goldset.schema import GoldItem
from llb.prep.ontology.artifacts.report import write_calibration_artifacts
from llb.prep.ontology.constants import (
    NEEDLE_GOLDSET_FILENAME,
    PROVENANCE_KIND,
)
from llb.prep.ontology.dedup import NearDuplicateFilter, load_prior_questions
from llb.prep.ontology.models import (
    DocExtraction,
    DocRecord,
    Entity,
    ItemLabels,
    OntologyCandidate,
)
from llb.prep.ontology.refine import refine_drafts_labeled
from ontology_yield_helpers import CHAIN_DOC, FakeEmbedder, FakeNeedleRetriever, _item, _span


def test_near_duplicate_filter_drops_paraphrase_of_prior_question():
    prior = ["Яка столиця України?"]
    items = [
        _item("dup", "Яка столиця України?"),  # exact prior -> dropped
        _item("keep", "Яка висота вежі?"),  # distinct -> kept
    ]
    kept, report = NearDuplicateFilter(prior, FakeEmbedder(), threshold=0.9).filter(items)
    assert [item.id for item in kept] == ["keep"]
    assert report["dropped"] == 1 and report["dropped_ids"] == ["dup"]
    assert report["prior_questions"] == 1


def test_near_duplicate_filter_no_prior_keeps_all():
    items = [_item("a", "q1"), _item("b", "q2")]
    kept, report = NearDuplicateFilter([], FakeEmbedder()).filter(items)
    assert kept == items and report["dropped"] == 0


def test_load_prior_questions_reads_prior_bundle_goldsets(tmp_path):
    bundle = tmp_path / "prior"
    bundle.mkdir()
    from llb.goldset.schema import dump_goldset

    dump_goldset(
        [_item("q1", "Питання одне?"), _item("q2", "Питання два?")], bundle / "goldset.jsonl"
    )
    assert load_prior_questions([bundle]) == ["Питання одне?", "Питання два?"]
    assert load_prior_questions([tmp_path / "missing"]) == []  # missing bundle skipped


def test_refine_labeled_tags_question_type_and_difficulty():
    docs = [DocRecord(doc_id="a.md", text=CHAIN_DOC, sha256="x", n_chars=len(CHAIN_DOC))]
    drafts = [
        {
            "doc_id": "a.md",
            "question": "Що таке ця організація?",
            "reference_answer": "Організацією є Gamma.",
            "answer_span": "Gamma",
            "difficulty": "easy",
        }
    ]
    items, labels = refine_drafts_labeled(docs, drafts)
    assert len(items) == 1
    label = labels[items[0].id]
    assert label.question_type == "definition"
    assert label.difficulty == "easy"  # carried from the seed via the draft dict


def test_calibration_report_labels_needles_and_reports_per_type_fraction(tmp_path):
    out = tmp_path / "bundle"
    corpus = out / "corpus"
    corpus.mkdir(parents=True)
    doc_id = "a.md"
    text = "Alpha керує Beta."
    (corpus / doc_id).write_text(text, encoding="utf-8")
    citation = {
        "kind": "pdf-citations",
        "source": "s.pdf",
        "doc_id": doc_id,
        "parser": "test",
        "pages": [
            {"page": 1, "text_start": 0, "text_end": len(text), "parser": "test", "blocks": []}
        ],
    }
    (corpus / "a.citations.json").write_text(
        json.dumps(citation, ensure_ascii=False), encoding="utf-8"
    )

    hit = GoldItem(
        id="q1",
        question="Хто керує Beta?",
        reference_answer="Alpha",
        source_doc_id=doc_id,
        source_spans=[_span(doc_id, 0, "Alpha")],
        provenance=PROVENANCE_KIND,
        split="final",
    )
    miss = GoldItem(
        id="q2",
        question="Що таке Beta?",
        reference_answer="Beta",
        source_doc_id=doc_id,
        source_spans=[_span(doc_id, text.index("Beta"), "Beta")],
        provenance=PROVENANCE_KIND,
        split="final",
    )
    labels = {
        "q1": ItemLabels(question_type="factoid", difficulty="easy"),
        "q2": ItemLabels(question_type="definition", difficulty="easy"),
    }
    extraction = DocExtraction(
        doc_id=doc_id,
        entities=[Entity(name="Alpha", type="ORG", mentions=[_span(doc_id, 0, "Alpha")])],
    )
    retriever = FakeNeedleRetriever(
        {hit.question: [{"doc_id": doc_id, "char_start": 0, "char_end": 5, "text": "Alpha"}]}
    )

    report = write_calibration_artifacts(
        out,
        [DocRecord(doc_id=doc_id, text=text, sha256="x", n_chars=len(text))],
        [extraction],
        OntologyCandidate(),
        [hit, miss],
        elapsed_s=0.0,
        settings={},
        retrieval_store=retriever,
        retrieval_k=3,
        item_labels=labels,
        coverage_matrix={"mode": "coverage-target"},
        dedup_report={"dropped": 2},
    )

    rows = [
        json.loads(line)
        for line in (out / NEEDLE_GOLDSET_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["id"]: row for row in rows}
    assert by_id["q1"]["question_type"] == "factoid"
    assert by_id["q2"]["question_type"] == "definition"

    assert report["question_type_distribution"] == {"definition": 1, "factoid": 1}
    assert report["coverage_matrix"] == {"mode": "coverage-target"}
    assert report["dedup"] == {"dropped": 2}
    per_type = report["retrieval_unique_needle_fraction_by_question_type"]
    assert per_type["factoid"]["retrievable_fraction"] == 1.0
    assert per_type["definition"]["retrievable_fraction"] == 0.0
