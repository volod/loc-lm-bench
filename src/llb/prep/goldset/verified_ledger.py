"""Human-verified gold-set ledger (verified gold-set ledger).

The committed `samples/goldsets/ua_squad_postedited_v1` set is human-verified (every item is
`verified: true`). When a NEW gold set is generated -- ingested SQuAD drafts (the default
`ingest-uk-squad` development mode) or the frontier drafting `prepare-goldset` frontier drafts + synthetic
planted labels -- any draft whose id is ALREADY in that human-verified set has been reviewed, so
we adopt the verified item verbatim (its post-edited content + grounded spans) and it needs no
re-review. Drafts not in the ledger stay `verified: false` pending review. This is what lets the
default option emit a gold set that is already human-verified for its reviewed overlap, instead
of an all-unverified draft.

The join key is the item id. Ledger loading and matching are injectable and need no network or
GPU; adopted corpus files are copied explicitly so the resulting bundle remains self-contained.
"""

import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from llb.goldset.schema import GoldItem, load_goldset
from llb.core.paths import PROJECT_ROOT

# The committed human-verified fixture used as the default verification ledger.
DEFAULT_VERIFIED_GOLDSET = (
    PROJECT_ROOT / "samples" / "goldsets" / "ua_squad_postedited_v1" / "goldset.jsonl"
)


def verified_corpus_root(goldset_path: Path | str) -> Path:
    """The corpus dir that grounds a committed goldset (sibling `corpus/` of the JSONL)."""
    return Path(goldset_path).parent / "corpus"


@dataclass(frozen=True)
class VerificationLedger:
    """Reviewed items and the corpus files that ground them, keyed by stable ids."""

    items: dict[str, GoldItem]
    document_sources: dict[str, Path]


def load_verified_ledger(goldset_paths: Sequence[Path | str]) -> VerificationLedger:
    """Build an id -> verified GoldItem ledger from one or more committed gold sets.

    Only `verified: true` items enter the ledger; later paths win on an id collision.
    """
    items: dict[str, GoldItem] = {}
    document_sources: dict[str, Path] = {}
    for path in goldset_paths:
        goldset_path = Path(path)
        corpus_root = verified_corpus_root(goldset_path)
        for item in load_goldset(goldset_path):
            if item.verified:
                items[item.id] = item
                for doc_id in adopted_doc_ids(item):
                    document_sources[doc_id] = corpus_root / doc_id
    return VerificationLedger(items=items, document_sources=document_sources)


def adopted_doc_ids(item: GoldItem) -> set[str]:
    """Every corpus doc id a verified item depends on (its source doc + each span's doc)."""
    docs = {item.source_doc_id}
    docs.update(span.doc_id for span in item.source_spans)
    return docs


def apply_verified_ledger(
    items: Iterable[GoldItem], ledger: VerificationLedger
) -> tuple[list[GoldItem], dict[str, Path], int]:
    """Adopt the human-verified version of any draft whose id is in the ledger.

    Returns `(merged_items, documents_to_copy, n_verified)`. A draft whose id is in the ledger is
    REPLACED by the verified item (verified content + grounded spans); drafts not in the ledger
    pass through unchanged (`verified: false`). `documents_to_copy` maps the adopted items'
    corpus doc ids to their reviewed source files so the caller can preserve exact grounding.
    """
    merged: list[GoldItem] = []
    documents: dict[str, Path] = {}
    n_verified = 0
    for item in items:
        verified = ledger.items.get(item.id)
        if verified is not None:
            merged.append(verified)
            for doc_id in adopted_doc_ids(verified):
                documents[doc_id] = ledger.document_sources[doc_id]
            n_verified += 1
        else:
            merged.append(item)
    return merged, documents, n_verified


def copy_verified_documents(document_sources: dict[str, Path], corpus_root: Path | str) -> None:
    """Copy adopted canonical documents into the generated corpus."""
    destination_root = Path(corpus_root)
    for doc_id, source in sorted(document_sources.items()):
        if not source.is_file():
            raise FileNotFoundError(f"verified corpus document not found: {source}")
        destination = destination_root / doc_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
