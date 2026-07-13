"""ontology-assisted gold-set drafting: the fake-endpoint full flow + needle/calibration artifacts.

No server, no provider key, no GPU: every LLM call is an injected fake, so the end-to-end bundle,
the PDF citation artifacts, needle retrieval annotation, and the calibration roll-up gates are
exercised deterministically. The per-stage units live in `test_ontology_extract.py`
(inventory/extraction/induction) and `test_ontology_coverage.py` (coverage/draft/refine/endpoint).

`DOC1`, `DOC2`, and `fake_endpoint` are re-exported here for `test_ontology_resume`.
"""

import json
import logging
from pathlib import Path

from llb.goldset.schema import GoldItem, SourceSpan, load_goldset
from llb.goldset.validate import validate_items
from llb.prep.ontology.artifacts.report import write_calibration_artifacts
from llb.prep.ontology.constants import (
    NEEDLE_GOLDSET_FILENAME,
    PDF_ONTOLOGY_REPORT_FILENAME,
    PROMPT_DICTIONARY_FILENAME,
    PROVENANCE_KIND,
)
from llb.prep.ontology.endpoint_config import EndpointCompleters, EndpointConfig, EndpointPlan
from llb.prep.ontology.models import (
    Claim,
    DocExtraction,
    DocRecord,
    Entity,
    OntologyCandidate,
)
from llb.prep.ontology.needles import annotate_needle_retrieval
from llb.prep.ontology.pipeline.run import draft_goldset

from tests.llb.prep.ontology._ontology_fixtures import DOC1, DOC2


class FakeNeedleRetriever:
    def __init__(self, hits_by_question: dict[str, list[dict[str, object]]]):
        self.hits_by_question = hits_by_question

    def retrieve(self, question: str, k: int) -> list[dict[str, object]]:
        return self.hits_by_question.get(question, [])[:k]


# --- stage 7: full flow over a fake local endpoint -------------------------------------------


def _extraction_json(prompt: str) -> str:
    if "столицею" in prompt:  # DOC1
        return json.dumps(
            {
                "entities": [
                    {"name": "Київ", "type": "LOC", "aliases": ["місто"], "mentions": ["Київ"]},
                    {"name": "Дніпро", "type": "LOC", "mentions": ["Дніпро"]},
                ],
                "claims": [{"text": "Київ є столицею", "evidence": "Київ є столицею України"}],
                "facts": [
                    {
                        "subject": "Київ",
                        "relation": "столиця",
                        "object": "України",
                        "evidence": "Київ є столицею України",
                    },
                    {
                        "subject": "Місто",
                        "relation": "розташоване",
                        "object": "Дніпро",
                        "evidence": "Місто розташоване на річці Дніпро",
                    },
                ],
            }
        )
    return json.dumps(
        {  # DOC2
            "entities": [{"name": "Львів", "type": "LOC", "mentions": ["Львів"]}],
            "events": [{"description": "заснування", "evidence": "Місто засноване у 1256 році"}],
            "facts": [
                {
                    "subject": "Львів",
                    "relation": "є",
                    "object": "культурним центром",
                    "evidence": "Львів є культурним центром заходу",
                },
                {
                    "subject": "Місто",
                    "relation": "засноване",
                    "object": "1256",
                    "evidence": "Місто засноване у 1256 році",
                },
            ],
        }
    )


def _draft_json(prompt: str) -> str:
    if "Сфокусуйся на факті:" in prompt:
        seg = prompt.split("Сфокусуйся на факті:")[1].split("\n")[0]
        subject = seg.split("|")[0].strip()
        obj = seg.rsplit("|", 1)[-1].strip().rstrip(".").strip()
        return json.dumps(
            {"question": f"Що відомо про {subject}?", "reference_answer": obj, "answer_span": obj}
        )
    if "Сфокусуйся на сутності:" in prompt:
        seg = prompt.split("Сфокусуйся на сутності:")[1].split("\n")[0]
        name = seg.split("(тип")[0].strip()
        return json.dumps(
            {"question": "Що згадано у документі?", "reference_answer": name, "answer_span": name}
        )
    return "{}"


def fake_endpoint(prompt: str) -> str:
    """One callable answering BOTH extraction and drafting prompts -- like a real local model."""
    if "будує онтологію" in prompt:
        return _extraction_json(prompt)
    if "укладач набору запитань" in prompt:
        return _draft_json(prompt)
    return "{}"


def _draft(corpus: Path, complete, **kwargs):
    config = EndpointConfig(kind="local", model="fake")
    return draft_goldset(
        corpus,
        EndpointPlan.single(config),
        completers=EndpointCompleters.single(complete),
        **kwargs,
    )


def _assert_items_unverified_grounded(result) -> None:
    # items: unverified, ontology-drafted, grounded, split-assigned
    assert len(result.items) > 0
    assert all(it.verified is False and it.provenance == PROVENANCE_KIND for it in result.items)
    assert all(it.split in ("calibration", "tuning", "final") for it in result.items)


def _assert_bundle_self_validates(out: Path) -> None:
    # the emitted bundle self-validates against its copied corpus
    loaded = load_goldset(out / "goldset.jsonl")
    report = validate_items(loaded, out / "corpus")
    assert report["errors"] == []


def _assert_ontology_artifacts(out: Path) -> None:
    # ontology + extraction artifacts written
    ontology = json.loads((out / "ontology.json").read_text(encoding="utf-8"))
    assert ontology["entity_types"] and ontology["relation_types"]
    assert (out / "extraction.jsonl").exists()


def _assert_provenance(out: Path, result) -> None:
    # provenance links endpoint / prompts / document hashes / cost
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["kind"] == PROVENANCE_KIND and prov["synthetic"] is False
    assert prov["endpoint"]["egress"] is False
    assert prov["endpoint"]["stages"]["extraction"]["kind"] == "local"
    assert set(prov["prompts"]) == {"extraction", "draft", "multi_hop"}
    assert prov["settings"]["extract_concurrency"] == 2
    assert {d["doc_id"] for d in prov["documents"]} == {"doc1.md", "doc2.md"}
    assert prov["stages"]["facts"] == 4 and prov["n_items"] == len(result.items)
    assert prov["stages"]["claims"] == 1 and prov["stages"]["events"] == 1  # seeded kinds counted


def _assert_ontology_report_and_gates(out: Path) -> None:
    report = json.loads((out / PDF_ONTOLOGY_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["grounded_facts"] == 4
    assert report["grounded_claims"] == 1 and report["grounded_events"] == 1
    assert report["dictionary_term_yield"] > 0
    assert (out / PROMPT_DICTIONARY_FILENAME).is_file()
    assert (out / NEEDLE_GOLDSET_FILENAME).is_file()
    # non-PDF corpus: grounded extractions + a non-empty gold set pass; the citation-needle gate is
    # not applicable (no page sidecars) and does not block.
    gates = report["gates"]
    assert gates["nonzero_grounded_extractions"] is True
    assert gates["nonzero_draft_items"] is True
    assert gates["pdf_citation_gate_applicable"] is False
    assert gates["passed"] is True


def test_full_flow_drafts_grounded_unverified_bundle(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.md").write_text(DOC1, encoding="utf-8")
    (corpus / "doc2.md").write_text(DOC2, encoding="utf-8")
    out = tmp_path / "bundle"

    result = _draft(
        corpus,
        fake_endpoint,
        max_items=20,
        out_dir=out,
        extract_concurrency=2,
    )

    _assert_items_unverified_grounded(result)
    _assert_bundle_self_validates(out)
    _assert_ontology_artifacts(out)
    _assert_provenance(out, result)
    _assert_ontology_report_and_gates(out)


def test_full_flow_writes_pdf_citation_artifacts_and_needles(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc_id = "pdf-test.md"
    body = (
        "Київська міська рада у 2024 році ухвалила рішення модернізувати трамвайні "
        "маршрути. Пасажиропотік зріс на 18 відсотків."
    )
    text = f"# Source PDF: source.pdf\n\n<!-- source_pdf: source.pdf page: 1 parser: test -->\n\n{body}\n"
    (corpus / doc_id).write_text(text, encoding="utf-8")
    text_start = text.index("Київська")
    citation = {
        "kind": "pdf-citations",
        "source": "source.pdf",
        "doc_id": doc_id,
        "parser": "test",
        "pages": [
            {
                "page": 1,
                "char_start": 0,
                "char_end": len(text),
                "text_start": text_start,
                "text_end": len(text),
                "n_chars": len(text) - text_start,
                "parser": "test",
                "blocks": [],
            }
        ],
    }
    (corpus / "pdf-test.citations.json").write_text(
        json.dumps(citation, ensure_ascii=False), encoding="utf-8"
    )
    out = tmp_path / "bundle"

    def pdf_endpoint(prompt: str) -> str:
        if "будує онтологію" in prompt:
            return json.dumps(
                {
                    "entities": [
                        {
                            "name": "Київська міська рада",
                            "type": "ORG",
                            "mentions": ["Київська міська рада"],
                        }
                    ],
                    "facts": [
                        {
                            "subject": "Київська міська рада",
                            "relation": "ухвалила рішення",
                            "object": "модернізувати трамвайні маршрути",
                            "evidence": (
                                "Київська міська рада у 2024 році ухвалила рішення "
                                "модернізувати трамвайні маршрути"
                            ),
                        }
                    ],
                }
            )
        if "укладач набору запитань" in prompt:
            return json.dumps(
                {
                    "question": "Яке рішення ухвалила міська рада?",
                    "reference_answer": "модернізувати трамвайні маршрути",
                    "answer_span": "модернізувати трамвайні маршрути",
                }
            )
        return "{}"

    result = _draft(
        corpus,
        pdf_endpoint,
        max_items=3,
        out_dir=out,
        doc_limit=1,
        extract_max_chars=500,
    )

    assert len(result.docs) == 1
    assert (out / "corpus" / "pdf-test.citations.json").is_file()
    report = json.loads((out / PDF_ONTOLOGY_REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["pdf_sidecar_docs"] == 1
    assert report["page_span_citation_coverage"]["coverage"] == 1.0
    assert report["item_page_span_citation_coverage"]["coverage"] == 1.0
    needles = load_goldset(out / NEEDLE_GOLDSET_FILENAME)
    assert len(needles) == report["citation_valid_needle_items"] >= 1
    # PDF corpus: the citation-needle gate is applicable and, with valid needles, the roll-up passes
    assert report["gates"]["pdf_citation_gate_applicable"] is True
    assert report["gates"]["has_citation_valid_needles"] is True
    assert report["gates"]["passed"] is True
    dictionary = [
        json.loads(line)
        for line in (out / PROMPT_DICTIONARY_FILENAME).read_text(encoding="utf-8").splitlines()
    ]
    entity = next(row for row in dictionary if row["term"] == "Київська міська рада")
    assert entity["examples"][0]["pdf_pages"][0]["source"] == "source.pdf"


def _grounded_span(doc_id: str, text: str, quote: str) -> SourceSpan:
    start = text.index(quote)
    return SourceSpan(doc_id=doc_id, char_start=start, char_end=start + len(quote), text=quote)


def _needle_item(item_id: str, question: str, doc_id: str, text: str, quote: str) -> GoldItem:
    span = _grounded_span(doc_id, text, quote)
    return GoldItem(
        id=item_id,
        question=question,
        reference_answer=quote,
        source_doc_id=doc_id,
        source_spans=[span],
        provenance=PROVENANCE_KIND,
        split="final",
    )


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


def test_calibration_gates_pass_without_sro_facts(tmp_path):
    # a corpus rich in entities/claims but with ZERO SRO facts still yields a usable gold set: the
    # roll-up passes on grounded extractions + a non-empty gold set, and does NOT require facts.
    out = tmp_path / "bundle"
    out.mkdir()
    span = _grounded_span("a.md", DOC1, "Київ")
    docs = [DocRecord(doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1))]
    extraction = DocExtraction(
        doc_id="a.md",
        entities=[Entity(name="Київ", type="LOC", mentions=[span])],
        claims=[Claim(text="Київ є столицею", evidence=span)],
    )
    item = GoldItem(
        id="q1",
        question="Що згадано у документі?",
        reference_answer="Київ",
        source_doc_id="a.md",
        source_spans=[span],
        provenance=PROVENANCE_KIND,
        split="final",
    )
    report = write_calibration_artifacts(
        out, docs, [extraction], OntologyCandidate(), [item], elapsed_s=0.0, settings={}
    )
    gates = report["gates"]
    assert gates["nonzero_grounded_facts"] is False  # no SRO facts...
    assert gates["nonzero_grounded_extractions"] is True  # ...but entities + claims are grounded
    assert gates["nonzero_draft_items"] is True
    assert gates["pdf_citation_gate_applicable"] is False
    assert gates["passed"] is True


def test_calibration_gates_fail_on_empty_draft(tmp_path):
    # no grounded evidence and no drafted items -> the roll-up fails so the operator does not accept
    out = tmp_path / "bundle"
    out.mkdir()
    docs = [DocRecord(doc_id="a.md", text=DOC1, sha256="x", n_chars=len(DOC1))]
    report = write_calibration_artifacts(
        out,
        docs,
        [DocExtraction(doc_id="a.md")],
        OntologyCandidate(),
        [],
        elapsed_s=0.0,
        settings={},
    )
    gates = report["gates"]
    assert gates["nonzero_grounded_extractions"] is False
    assert gates["nonzero_draft_items"] is False
    assert gates["passed"] is False


def test_pipeline_warns_and_names_the_blocking_gate(caplog):
    # the pipeline acts on the roll-up: a failing required gate is a WARNING that names it, so a
    # PDF run with no citation-valid needle is flagged (informational gates never appear here)
    from llb.prep.ontology.pipeline.bundle import _log_calibration_gates

    report = {
        "gates": {
            "nonzero_grounded_extractions": True,
            "nonzero_grounded_facts": False,  # informational: must NOT be named as a blocker
            "nonzero_draft_items": True,
            "has_citation_valid_needles": False,
            "pdf_citation_gate_applicable": True,
            "passed": False,
        }
    }
    with caplog.at_level(logging.WARNING):
        _log_calibration_gates(report, Path("/tmp/bundle"))
    assert "NOT passed" in caplog.text
    assert "has_citation_valid_needles" in caplog.text
    assert "nonzero_grounded_facts" not in caplog.text


def test_full_flow_does_not_write_when_write_false(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc1.md").write_text(DOC1, encoding="utf-8")
    out = tmp_path / "bundle"
    result = _draft(
        corpus,
        fake_endpoint,
        out_dir=out,
        write=False,
    )
    assert not out.exists() and len(result.items) >= 0
