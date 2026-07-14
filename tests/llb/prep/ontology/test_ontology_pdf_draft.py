"""Tests for ontology pdf draft."""

import json
from llb.goldset.schema import load_goldset
from llb.prep.ontology.constants import (
    NEEDLE_GOLDSET_FILENAME,
    PDF_ONTOLOGY_REPORT_FILENAME,
    PROMPT_DICTIONARY_FILENAME,
)
from test_ontology_draft import _draft


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
