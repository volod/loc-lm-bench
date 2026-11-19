"""The redistribution gate on auto-RAG's frontier verification lane.

The frontier verifier receives each drafted row's question, reference answer, and the corpus window
around its cited span. A drafted bundle carries its own corpus copy but not the staged manifest, so
the recorded terms come from the bundle's `provenance.json`, whose per-document rows hold the
acquired fields -- `licence` among them -- for every inventoried document
(`llb.prep.corpus.redistribution.bundle_licences`). A bundle without that record, or
whose rows carry no licence, has no recorded terms and verifies as before.
"""

from pathlib import Path

from llb.goldset.schema import GoldItem
from llb.prep.corpus.redistribution import bundle_licences, refuse_non_redistributable

FRONTIER_VERIFY_EGRESS = "auto-RAG frontier verification"


def refuse_frontier_verify_egress(bundle: Path, items: list[GoldItem]) -> None:
    """Refuse frontier verification of drafted rows citing non-redistributable documents."""
    doc_ids = {item.source_doc_id for item in items}
    doc_ids.update(span.doc_id for item in items for span in item.source_spans)
    refuse_non_redistributable(
        bundle_licences(bundle), egress=FRONTIER_VERIFY_EGRESS, doc_ids=doc_ids
    )
