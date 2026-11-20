"""The redistribution gate on the ontology pipeline's frontier endpoints.

Explicit egress consent (`EndpointConfigBuilder`) says the operator accepts that corpus text
leaves the host; it cannot say the material is ENTITLED to leave. A frontier extraction or drafting
stage sends every inventoried document's text to the provider, so before the bundle directory or
any completer exists the inventoried documents -- after `doc_limit`, which is exactly the set the
pipeline will read -- are checked against the licences the staged corpus recorded. A local plan
sends nothing off the host and is not gated.
"""

from pathlib import Path

from llb.prep.corpus.redistribution import refuse_corpus_egress
from llb.prep.ontology.endpoints.config import EndpointPlan
from llb.rag.chunking.corpus import iter_docs

FRONTIER_DRAFT_EGRESS = "ontology drafting through a frontier endpoint"


def frontier_stage_names(endpoints: EndpointPlan) -> list[str]:
    """The pipeline stages whose model calls leave the host."""
    stages = {"extraction": endpoints.extraction, "drafting": endpoints.drafting}
    return [name for name, config in stages.items() if config.egress]


def refuse_inventoried_egress(
    corpus_root: Path | str, *, egress: str, doc_limit: int | None = None
) -> None:
    """Refuse when a document the pipeline inventories (after `doc_limit`) may not leave the host."""
    doc_ids = [doc_id for doc_id, _text in iter_docs(Path(corpus_root))]
    if doc_limit is not None:
        doc_ids = doc_ids[:doc_limit]
    refuse_corpus_egress(corpus_root, egress=egress, doc_ids=doc_ids)


def refuse_frontier_corpus_egress(
    endpoints: EndpointPlan, corpus_root: Path | str, *, doc_limit: int | None = None
) -> None:
    """Refuse a frontier plan over documents whose recorded licence forbids redistribution."""
    stages = frontier_stage_names(endpoints)
    if stages:
        refuse_inventoried_egress(
            corpus_root,
            egress=f"{FRONTIER_DRAFT_EGRESS} ({', '.join(stages)})",
            doc_limit=doc_limit,
        )
