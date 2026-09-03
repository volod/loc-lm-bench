"""The `provenance.json` payload for one draft bundle: what produced it, from what.

Split from `bundle.py`, which writes the files; this module only builds the record. It answers
three questions a reader has months later: which prompt wording ran, what the drafted items look
like in aggregate, and which document -- and which upstream capture of it -- every span indexes
into.
"""

import hashlib

from llb.prep.corpus.fingerprints import corpus_version_binding
from llb.prep.corpus.governance import manifest_governance_by_doc
from llb.prep.corpus.governance_fields import ACQUIRED_GOVERNANCE_FIELDS
from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.ontology import (
    CorpusVersionRecord,
    OntologyProvenance,
    ProvenanceDocument,
)
from llb.prep.ontology.constants import (
    PROVENANCE_KIND,
    PROVENANCE_SCHEMA_ID,
    PROVENANCE_SCHEMA_VERSION,
)
from llb.prep.ontology.drafting.run import draft_prompt
from llb.prep.ontology.endpoints.config import EndpointPlan, endpoint_provenance
from llb.prep.ontology.extraction.run import extraction_prompt
from llb.prep.ontology.models import DraftSeed
from llb.prep.ontology.pipeline.settings import PipelineResult


def prompt_fingerprints() -> dict[str, str]:
    """sha256 of the exact template wording, so a run records WHICH prompts produced it."""
    placeholder_seed = DraftSeed(
        doc_id="<doc>",
        kind="fact",
        section_title="<section>",
        difficulty="medium",
        strata={},
        evidence={"doc_id": "<doc>", "char_start": 0, "char_end": 1, "text": "x"},  # type: ignore[arg-type]
    )
    from llb.prep.ontology.models import MultiHopSeed, MultiHopStep
    from llb.prep.ontology.drafting.multi_hop import multi_hop_prompt

    placeholder_step = MultiHopStep(
        subject="<a>",
        relation="<r>",
        object="<b>",
        section_title="<section>",
        evidence={"doc_id": "<doc>", "char_start": 0, "char_end": 1, "text": "x"},  # type: ignore[arg-type]
    )
    placeholder_chain = MultiHopSeed(
        steps=[placeholder_step, placeholder_step], bridge="<b>", start="<a>", end="<c>"
    )
    extract_tmpl = extraction_prompt("<doc>", "<text>")
    draft_tmpl = draft_prompt(placeholder_seed, "<context>")
    multi_hop_tmpl = multi_hop_prompt(placeholder_chain, "<context>")
    return {
        "extraction": hashlib.sha256(extract_tmpl.encode("utf-8")).hexdigest(),
        "draft": hashlib.sha256(draft_tmpl.encode("utf-8")).hexdigest(),
        "multi_hop": hashlib.sha256(multi_hop_tmpl.encode("utf-8")).hexdigest(),
    }


def label_counts(result: PipelineResult) -> dict[str, dict[str, int]]:
    """Question-type and difficulty distributions over the drafted items (from item labels)."""
    by_type: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for item in result.items:
        label = result.item_labels.get(item.id)
        qtype = label.question_type if label else "factoid"
        difficulty = label.difficulty if label else "medium"
        by_type[qtype] = by_type.get(qtype, 0) + 1
        by_difficulty[difficulty] = by_difficulty.get(difficulty, 0) + 1
    return {
        "question_type_distribution": dict(sorted(by_type.items())),
        "difficulty_distribution": dict(sorted(by_difficulty.items())),
    }


def document_rows(result: PipelineResult) -> list[ProvenanceDocument]:
    """One row per inventoried document: local identity plus the acquisition provenance it carries.

    Every acquired field is present on every row, `None` where the staged corpus recorded none, so
    a bundle drafted on an operator's own directory states that absence rather than leaving a
    reader to guess whether the question was asked.
    """
    governance = manifest_governance_by_doc(result.corpus_root)
    return [
        ProvenanceDocument(
            doc_id=doc.doc_id,
            sha256=doc.sha256,
            n_chars=doc.n_chars,
            **{
                field: (governance.get(doc.doc_id) or {}).get(field)
                for field in ACQUIRED_GOVERNANCE_FIELDS
            },
        )
        for doc in result.docs
    ]


def _stage_counts(result: PipelineResult, n_multi_hop: int) -> JsonObject:
    return {
        "documents": len(result.docs),
        "entities": sum(len(e.entities) for e in result.extractions),
        "events": sum(len(e.events) for e in result.extractions),
        "claims": sum(len(e.claims) for e in result.extractions),
        "facts": sum(len(e.facts) for e in result.extractions),
        "ontology_entity_types": len(result.ontology.entity_types),
        "ontology_relation_types": len(result.ontology.relation_types),
        "seeds": len(result.seeds),
        "draft_attempts": result.draft_attempts,
        "draft_parsed": result.draft_parsed,
        "draft_parse_rate": (
            result.draft_parsed / result.draft_attempts if result.draft_attempts else 0.0
        ),
        "multi_hop_items": n_multi_hop,
        "chains": len(result.chains),
        "items": len(result.items),
    }


def provenance_payload(
    result: PipelineResult, endpoints: EndpointPlan, seed: int, settings: dict[str, object]
) -> OntologyProvenance:
    """The bundle's provenance as its registered contract rather than an unchecked dictionary."""
    n_multi_hop = sum(
        1
        for item in result.items
        if (label := result.item_labels.get(item.id)) and label.question_type == "multi-hop"
    )
    return OntologyProvenance(
        schema_id=PROVENANCE_SCHEMA_ID,
        schema_version=PROVENANCE_SCHEMA_VERSION,
        kind=PROVENANCE_KIND,
        synthetic=False,  # drafted FROM a real corpus (vs planted synthetic docs)
        endpoint=endpoint_provenance(endpoints, result.endpoint_logs),
        prompts=prompt_fingerprints(),
        seed=seed,
        settings=settings,
        elapsed_s=round(result.elapsed_s, 3),
        corpus_version=CorpusVersionRecord.model_validate(
            dict(corpus_version_binding(result.corpus_root))
        ),
        documents=document_rows(result),
        stages=_stage_counts(result, n_multi_hop),
        labels=label_counts(result),
        ontology=result.ontology.model_dump(),
        n_items=len(result.items),
        cost=result.endpoint_logs.summary(),
        seed_coverage=result.coverage_report,
        dedup=result.dedup_report,
        multi_hop_carry_forward=result.carry_forward_report,
        applied_feedback=result.applied_feedback,
        multi_hop_path_strata=result.multi_hop_path_strata,
    )
