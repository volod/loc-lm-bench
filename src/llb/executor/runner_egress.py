"""The redistribution gate on `run-eval`'s frontier scorer lane.

A frontier judge receives each case's question, answer, and retrieved contexts, so before the lane
records consent or issues a call, every document those records carry text from -- each retrieved
chunk's document and each gold item's source document -- is checked against its recorded licence.
Two sources answer that: the licence each retrieved chunk carried into the store, and the staged
corpus manifest the run is configured against. A document either source refuses stays refused, so a
store and corpus that disagree cannot launder a `local-only` passage into a `redistributable` one.
The local and human lanes send nothing off the host and are not gated.
"""

from typing import TYPE_CHECKING

from llb.prep.corpus.redistribution import (
    chunk_licences,
    corpus_licences,
    merge_licences,
    refuse_non_redistributable,
)

if TYPE_CHECKING:
    from llb.core.config import RunConfig
    from llb.executor.cases import CaseBatch

FRONTIER_SCORER_EGRESS = "frontier scorer (run-eval scorer_policy=frontier)"


def refuse_frontier_judge_egress(config: "RunConfig", batch: "CaseBatch") -> None:
    """Refuse the frontier lane when a judged case carries non-redistributable material."""
    if config.scorer_policy != "frontier":
        return
    chunks = [chunk for retrieved, _spans in batch.retrieval_pairs for chunk in retrieved]
    doc_ids = {str(chunk.get("doc_id", "")) for chunk in chunks}
    for item, _answer in batch.answers:
        doc_ids.add(item.source_doc_id)
        doc_ids.update(span.doc_id for span in item.source_spans)
    licences = merge_licences(chunk_licences(chunks), corpus_licences(config.corpus_root))
    refuse_non_redistributable(licences, egress=FRONTIER_SCORER_EGRESS, doc_ids=doc_ids - {""})
