"""Redistribution gate for the SFT/DPO fine-tune export.

The export reads a finalized run bundle and a gold set rather than a corpus, so the recorded terms
of the material it writes come from the corpus the run was evaluated against: an explicit
`corpus_root`, else the `config.corpus_root` the run manifest recorded. The material is every
document a selected record carries text from -- each item's source document and gold spans (the
question, reference answer, and fallback context), and every document a persisted retrieved span
stands for, collapsed duplicate occurrences included (the context preview and the rejected answer
that was generated from it).

A run that names no corpus has no recorded terms to enforce and exports as before. A run that
NAMES a corpus this host no longer holds is refused rather than exported unchecked: the gate cannot
tell a missing corpus from a permitted one, and guessing "permitted" is the silent failure it
exists to prevent. `--corpus` points the check at a relocated copy.
"""

from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.core.paths import resolve_project_path
from llb.goldset.schema import GoldItem
from llb.prep.corpus.redistribution import refuse_corpus_egress
from llb.rag.retrieval_records import record_documents

FINETUNE_EGRESS = "fine-tune dataset export"


def exported_doc_ids(items: list[GoldItem], retrieval: dict[str, JsonObject]) -> set[str]:
    """Every document whose text a fine-tune record for `items` can carry."""
    doc_ids: set[str] = set()
    for item in items:
        doc_ids.add(item.source_doc_id)
        doc_ids.update(span.doc_id for span in item.source_spans)
        for retrieved in (retrieval.get(item.id) or {}).get("retrieved") or []:
            doc_ids.update(record_documents(retrieved))
    return {doc_id for doc_id in doc_ids if doc_id}


def run_corpus_root(manifest: JsonObject, corpus_root: Path | str | None) -> Path | None:
    """The corpus whose recorded terms govern this run's export, or None when none is named."""
    if corpus_root is not None:
        return Path(corpus_root)
    recorded = (manifest.get("config") or {}).get("corpus_root")
    return resolve_project_path(str(recorded)) if recorded else None


def refuse_unredistributable_export(
    manifest: JsonObject,
    corpus_root: Path | str | None,
    items: list[GoldItem],
    retrieval: dict[str, JsonObject],
) -> None:
    """Refuse the whole export when any carried document's recorded licence forbids it."""
    root = run_corpus_root(manifest, corpus_root)
    if root is None:
        return
    if not root.is_dir():
        raise ValueError(
            f"[redistribution] {FINETUNE_EGRESS} cannot read the recorded terms of its material: "
            f"the run's corpus {root} is not on this host; pass --corpus <corpus-dir> to check it "
            "against a relocated copy"
        )
    refuse_corpus_egress(root, egress=FINETUNE_EGRESS, doc_ids=exported_doc_ids(items, retrieval))
