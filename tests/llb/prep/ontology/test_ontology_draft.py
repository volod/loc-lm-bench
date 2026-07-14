"""ontology-assisted gold-set drafting: the fake-endpoint full flow + needle/calibration artifacts.

No server, no provider key, no GPU: every LLM call is an injected fake, so the end-to-end bundle,
the PDF citation artifacts, needle retrieval annotation, and the calibration roll-up gates are
exercised deterministically. The per-stage units live in `test_ontology_extract.py`
(inventory/extraction/induction) and `test_ontology_coverage.py` (coverage/draft/refine/endpoint).

`DOC1`, `DOC2`, and `fake_endpoint` are re-exported here for `test_ontology_resume`.
"""

import json
from pathlib import Path

from llb.goldset.schema import GoldItem, SourceSpan, load_goldset
from llb.goldset.validate import validate_items
from llb.prep.ontology.constants import (
    NEEDLE_GOLDSET_FILENAME,
    PDF_ONTOLOGY_REPORT_FILENAME,
    PROMPT_DICTIONARY_FILENAME,
    PROVENANCE_KIND,
)
from llb.prep.ontology.endpoint_config import EndpointCompleters, EndpointConfig, EndpointPlan
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
