"""Multi-hop draft widening: extraction reuse and one-ledger carry-forward."""

import hashlib
import json

import pytest

from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.prep.ontology.endpoint_config import (
    EndpointCompleters,
    EndpointConfig,
    EndpointPlan,
)
from llb.prep.ontology.models import DocRecord, ItemLabels
from llb.prep.ontology.pipeline.expansion import (
    carry_forward_multi_hop,
    prior_multihop_span_pairs,
    reused_extractions,
)
from llb.prep.ontology.pipeline.run import draft_goldset
from ontology_yield_helpers import CHAIN_DOC, _chain_endpoint


def _doc(text: str) -> DocRecord:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return DocRecord(doc_id="a.md", text=text, sha256=digest, n_chars=len(text))


def test_reused_extractions_require_the_same_corpus_and_exact_spans(tmp_path):
    text = "Alpha керує Beta."
    bundle = tmp_path / "source"
    bundle.mkdir()
    extraction = {
        "doc_id": "a.md",
        "entities": [],
        "events": [],
        "claims": [],
        "facts": [
            {
                "subject": "Alpha",
                "relation": "керує",
                "object": "Beta",
                "evidence": {
                    "doc_id": "a.md",
                    "char_start": 0,
                    "char_end": 5,
                    "text": "Alpha",
                },
            }
        ],
    }
    (bundle / "extraction.jsonl").write_text(
        json.dumps(extraction, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    doc = _doc(text)
    (bundle / "provenance.json").write_text(
        json.dumps(
            {"documents": [{"doc_id": doc.doc_id, "sha256": doc.sha256, "n_chars": doc.n_chars}]}
        )
        + "\n",
        encoding="utf-8",
    )

    assert reused_extractions(bundle, [doc])[0].doc_id == "a.md"
    with pytest.raises(ValueError, match="does not match"):
        reused_extractions(bundle, [_doc("Omega керує Beta.")])
    with pytest.raises(ValueError, match="does not match"):
        reused_extractions(
            bundle, [DocRecord(doc_id="b.md", text=text, sha256="x", n_chars=len(text))]
        )
    with pytest.raises(ValueError, match="does not match"):
        reused_extractions(bundle, [_doc(f"{text} Нова примітка.")])


def test_carry_forward_builds_one_collision_free_multihop_ledger(tmp_path):
    text = "Alpha керує Beta. Beta підтримує Gamma."
    bundle = tmp_path / "prior"
    bundle.mkdir()
    prior = GoldItem(
        id="shared",
        question="Який ланцюг поєднує Alpha та Gamma?",
        reference_answer="Alpha та Beta",
        source_doc_id="a.md",
        source_spans=[
            SourceSpan(doc_id="a.md", char_start=0, char_end=5, text="Alpha"),
            SourceSpan(
                doc_id="a.md",
                char_start=text.index("Beta"),
                char_end=text.index("Beta") + len("Beta"),
                text="Beta",
            ),
        ],
        provenance="ontology-drafted",
        split="final",
    )
    dump_goldset([prior], bundle / "goldset.jsonl")
    (bundle / "needle_items.jsonl").write_text(
        json.dumps({**prior.model_dump(), "question_type": "multi-hop", "difficulty": "hard"})
        + "\n",
        encoding="utf-8",
    )

    fresh = prior.model_copy(
        deep=True,
        update={"question": "Який інший зв'язок існує?", "reference_answer": "Alpha та Beta"},
    )
    labels = {"shared": ItemLabels(question_type="multi-hop", difficulty="hard")}
    items, merged_labels, report = carry_forward_multi_hop([fresh], labels, [bundle], [_doc(text)])

    assert len(items) == 2
    assert items[0].id == "shared"
    assert items[1].id == "shared-expansion-0"
    assert set(merged_labels) == {item.id for item in items}
    assert report == {
        "enabled": True,
        "sources": [{"bundle": str(bundle), "labeled_items": 1}],
        "carried_items": 1,
        "dropped_carried_exact_duplicates": 0,
        "new_items": 1,
        "combined_items": 2,
        "rewritten_new_ids": 1,
    }

    carried, _, repeated_report = carry_forward_multi_hop([], {}, [bundle, bundle], [_doc(text)])
    assert len(carried) == 1
    assert repeated_report["dropped_carried_exact_duplicates"] == 1


def test_span_pair_exclusion_skips_an_unlabeled_dedup_bundle(tmp_path):
    bundle = tmp_path / "ordinary"
    bundle.mkdir()
    dump_goldset([], bundle / "goldset.jsonl")

    assert prior_multihop_span_pairs([bundle]) == set()


def test_pipeline_reuses_extraction_and_skips_flat_drafting(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "chain.md").write_text(CHAIN_DOC, encoding="utf-8")
    source = tmp_path / "source"
    widened = tmp_path / "widened"
    config = EndpointConfig(kind="local", model="fake")
    plan = EndpointPlan.single(config)
    draft_goldset(
        corpus,
        plan,
        completers=EndpointCompleters.single(_chain_endpoint),
        max_items=1,
        out_dir=source,
    )

    def extraction_must_not_run(_prompt: str) -> str:
        raise AssertionError("the source bundle extraction must be reused")

    result = draft_goldset(
        corpus,
        plan,
        completers=EndpointCompleters(
            extraction=extraction_must_not_run,
            drafting=_chain_endpoint,
        ),
        out_dir=widened,
        reuse_extraction_bundle=source,
        multi_hop=True,
        multi_hop_only=True,
    )

    assert result.items
    assert result.seeds == []
    assert all(result.item_labels[item.id].question_type == "multi-hop" for item in result.items)
    provenance = json.loads((widened / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["settings"]["reuse_extraction_bundle"] == str(source)
    assert provenance["settings"]["multi_hop_only"] is True
    assert provenance["stages"]["draft_attempts"] == 0
