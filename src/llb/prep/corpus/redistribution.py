"""The redistribution gate: refuse an export or egress carrying material that may not leave the host.

`acl_label` answers who may READ a document inside a run. `licence` answers a different question --
whether the material may be copied off the host -- and it is the producing side's determination,
rendered into the acquired-corpus sidecar (`docs/design/acquired-corpus-projection.md`) and carried
here as a recorded fact. This module never makes that determination; it enforces the one it was
given, at every path where corpus-derived text leaves: the fine-tune and contrastive-pair exports,
and every opt-in frontier or hosted-API call.

Three rules shape it:

- **Refuse, never filter.** A refusal names EVERY offending document and stops the whole export
  before anything is written or sent. Dropping the offending rows instead would produce a partial
  artifact indistinguishable from a complete one -- the failure this gate exists to prevent.
- **Fail closed on an unknown class.** Only `redistributable` permits egress. `local-only` and any
  value this side does not recognise refuse: an answer this side cannot read is not a "yes".
- **No recorded terms, no change.** A document whose `licence` is absent (`None`) -- an operator's
  own corpus, or a corpus carrying no provenance at all -- is not gated, so a corpus without the
  field exports exactly as it did before the gate existed.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from llb.prep.corpus.governance import manifest_governance_by_doc
from llb.prep.ontology.constants import PROVENANCE_FILENAME

if TYPE_CHECKING:
    from llb.core.contracts.rag import ChunkRecord

LICENCE_FIELD = "licence"

# The redistribution classes this side reads. Only the first permits material to leave the host.
REDISTRIBUTABLE = "redistributable"
LOCAL_ONLY = "local-only"
REDISTRIBUTION_CLASSES = (REDISTRIBUTABLE, LOCAL_ONLY)

Licences = Mapping[str, str | None]


def permits_redistribution(licence: str | None) -> bool:
    """Whether a recorded licence lets the material leave the host (absent terms are not gated)."""
    return licence is None or licence == REDISTRIBUTABLE


@dataclass(frozen=True, order=True)
class RefusedDocument:
    """One document an egress path may not carry, with the licence that forbids it."""

    doc_id: str
    licence: str


class RedistributionRefused(ValueError):
    """An export or egress path refused material whose recorded terms forbid redistribution."""

    def __init__(self, egress: str, documents: tuple[RefusedDocument, ...]) -> None:
        self.egress = egress
        self.documents = documents
        super().__init__(_refusal_message(egress, documents))

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(document.doc_id for document in self.documents)


def _refusal_message(egress: str, documents: tuple[RefusedDocument, ...]) -> str:
    named = ", ".join(f"{doc.doc_id} (licence={doc.licence})" for doc in documents)
    return (
        f"[redistribution] {egress} refused: {len(documents)} document(s) carry a recorded "
        f"licence that forbids redistribution: {named}. The whole export was refused rather than "
        "filtered; remove the document(s) from the input, or obtain a redistributable "
        "determination from the producing service."
    )


def refuse_non_redistributable(
    licences: Licences, *, egress: str, doc_ids: Iterable[str] | None = None
) -> None:
    """Raise `RedistributionRefused` naming every in-scope document whose licence forbids egress.

    `doc_ids` limits the check to the documents the path actually carries; `None` checks every
    document in `licences`. A document absent from `licences` has no recorded terms and passes.
    """
    scope = set(licences) if doc_ids is None else set(doc_ids)
    refused = sorted(
        RefusedDocument(doc_id, str(licence))
        for doc_id in scope
        if not permits_redistribution(licence := licences.get(doc_id))
    )
    if refused:
        raise RedistributionRefused(egress, tuple(refused))


def _strictness(licence: str | None) -> int:
    """Order recorded terms so a merge keeps the strictest: refusing > redistributable > absent."""
    if not permits_redistribution(licence):
        return 2
    return 0 if licence is None else 1


def merge_licences(*sources: Licences) -> dict[str, str | None]:
    """Combine licence maps so a document refused by ANY source stays refused."""
    merged: dict[str, str | None] = {}
    for source in sources:
        for doc_id, licence in source.items():
            if doc_id not in merged or _strictness(licence) > _strictness(merged[doc_id]):
                merged[doc_id] = licence
    return merged


def corpus_licences(corpus_root: Path | str) -> dict[str, str | None]:
    """The recorded licence of every successfully staged document; empty without a manifest."""
    return {
        doc_id: row.get(LICENCE_FIELD)
        for doc_id, row in manifest_governance_by_doc(corpus_root).items()
    }


def bundle_licences(bundle_dir: Path | str) -> dict[str, str | None]:
    """The licence on each document row of a drafted bundle's `provenance.json`; empty without one.

    A drafted bundle carries a corpus COPY without the staged manifest, so its provenance record --
    whose per-document rows hold every acquired field -- is where its recorded terms travel.
    """
    path = Path(bundle_dir) / PROVENANCE_FILENAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("documents") if isinstance(payload, dict) else None
    return {
        str(row["doc_id"]): row.get(LICENCE_FIELD)
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and row.get("doc_id")
    }


def chunk_licences(chunks: Iterable["ChunkRecord"]) -> dict[str, str | None]:
    """The licence each chunk's document carried into the store, keyed by `doc_id`."""
    return merge_licences(
        *(
            {str(chunk.get("doc_id", "")): (chunk.get("metadata") or {}).get(LICENCE_FIELD)}
            for chunk in chunks
        )
    )


def refuse_corpus_egress(
    corpus_root: Path | str, *, egress: str, doc_ids: Iterable[str] | None = None
) -> None:
    """Refuse an egress path over a staged corpus (all documents, or only `doc_ids`)."""
    refuse_non_redistributable(corpus_licences(corpus_root), egress=egress, doc_ids=doc_ids)


def refuse_chunk_egress(chunks: Iterable["ChunkRecord"], *, egress: str) -> None:
    """Refuse an egress path whose payload is these chunks, by the licence each one carries."""
    refuse_non_redistributable(chunk_licences(chunks), egress=egress)
