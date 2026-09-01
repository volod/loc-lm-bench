"""The store a finished audit read, recorded as an IDENTITY rather than as a second manifest.

The `tree` block used to carry `doc_fingerprints` -- the store's whole `{doc_id: sha256}` map,
copied verbatim out of `store_meta.json`. It repeated in full both things the bundle record spent
four folds learning not to (`record_fold.py`): every document id, unfolded and un-interned, and a
64-hex digest per document that nothing ever compares for anything but equality.

**The per-document question is not the bundle's to answer.** `store_meta.json` holds the
authoritative map, and its consumer is the refresh diff (`llb.rag.refresh.diff`), which asks *which
documents changed* against the store itself -- never against an audit bundle. A bundle's copy is a
snapshot of that map taken at run time and can only ever be a worse answer to the same question.

What a bundle IS asked is one question with a yes/no answer: **is the store on disk still the store
this run read?** That is an equality test over the whole map, so it is answered by one digest over
the sorted pairs, in 64 bytes and independently of the corpus size. The digest is order-independent
by construction, so a store rebuilt in a different document order but over identical content still
reads as the same store.

Bundles on disk carry the old map, so `StoreIdentity.of` reads both forms and computes the digest
from a recorded map when that is all a bundle has -- which is why the verdict is identical through
either form and no schema version is needed to tell them apart. The two forms are self-describing:
`doc_fingerprints_digest` is present exactly when the map is not.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from llb.core.contracts.common import JsonObject

DIGEST_KEY = "doc_fingerprints_digest"
DOCUMENTS_KEY = "doc_fingerprints_documents"
# The form bundles on disk were written in. Read for the digest it implies, never per document.
MAP_KEY = "doc_fingerprints"

# Why a bundle cannot be placed against a store at all. Both halves are silences, not failures: a
# run below the semantic tier read no store, and a store built before `doc_fingerprints` records
# nothing to identify itself by.
NO_IDENTITY = (
    "no store identity recorded: this run read no store, or the store it read predates "
    "per-document fingerprints"
)


def fingerprint_digest(fingerprints: Mapping[str, str]) -> str:
    """A sha256 over the SORTED `(doc_id, fingerprint)` pairs of a store's manifest.

    Sorted, so the digest is a property of the mapping rather than of the order a store happened to
    write it in: a rebuild that visits the corpus in a different order over identical content is
    the same store and must read as one.
    """
    pairs = sorted((str(doc_id), str(value)) for doc_id, value in fingerprints.items())
    encoded = json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def identity_payload(fingerprints: Mapping[str, str]) -> JsonObject:
    """What the tree block records about the store it read, or nothing when there is nothing to say.

    A store with no fingerprints records no identity rather than the digest of an empty map: the
    two mean opposite things, and a digest that says "identical to every other fingerprintless
    store" would be a claim rather than a silence.
    """
    if not fingerprints:
        return {}
    return {DIGEST_KEY: fingerprint_digest(fingerprints), DOCUMENTS_KEY: len(fingerprints)}


@dataclass(frozen=True)
class StoreIdentity:
    """The store one bundle read, however that bundle happened to record it."""

    digest: str
    documents: int
    # True when the bundle carried the whole map and the digest was computed from it here. Kept so
    # a reading can say which form it read rather than implying every bundle carries a digest.
    from_recorded_map: bool

    @classmethod
    def of(cls, tree_meta: JsonObject) -> "StoreIdentity | None":
        """The identity a bundle's `tree` block carries, at either form, or None when it has none."""
        digest = tree_meta.get(DIGEST_KEY)
        if isinstance(digest, str) and digest:
            documents = tree_meta.get(DOCUMENTS_KEY)
            return cls(
                digest=digest,
                documents=documents if isinstance(documents, int) else 0,
                from_recorded_map=False,
            )
        recorded = tree_meta.get(MAP_KEY)
        if isinstance(recorded, dict) and recorded:
            return cls(
                digest=fingerprint_digest(recorded),
                documents=len(recorded),
                from_recorded_map=True,
            )
        return None


@dataclass(frozen=True)
class StoreChange:
    """Whether a store on disk is the store a bundle read, or why the two cannot be compared."""

    comparable: bool
    changed: bool
    detail: str

    def payload(self) -> JsonObject:
        return {
            "comparable": self.comparable,
            "changed": self.changed if self.comparable else None,
            "detail": self.detail,
        }


def compare_store(
    tree_meta: JsonObject, fingerprints: Mapping[str, str], *, label: str = "the store"
) -> StoreChange:
    """Place a bundle against a store on disk: the same store, a different one, or unanswerable."""
    identity = StoreIdentity.of(tree_meta)
    if identity is None:
        return StoreChange(comparable=False, changed=False, detail=NO_IDENTITY)
    if not fingerprints:
        return StoreChange(
            comparable=False,
            changed=False,
            detail=f"{label} records no per-document fingerprints to compare against",
        )
    if fingerprint_digest(fingerprints) == identity.digest:
        return StoreChange(
            comparable=True,
            changed=False,
            detail=f"{label} is the one this run read ({identity.documents} documents)",
        )
    return StoreChange(
        comparable=True,
        changed=True,
        detail=(
            f"{label} is NOT the one this run read: {identity.documents} documents recorded, "
            f"{len(fingerprints)} on disk now -- this bundle's readings are about the store it "
            "held, not this one"
        ),
    )


def identity_entry(
    summary: JsonObject,
    fingerprints: Mapping[str, str],
    *,
    label: str = "the store",
    reference: str | None = None,
    reference_source: str | None = None,
) -> JsonObject:
    """One bundle placed against one store, in the shape the stage re-read carries it in."""
    tree_meta = summary.get("tree")
    payload = compare_store(
        tree_meta if isinstance(tree_meta, dict) else {}, fingerprints, label=label
    ).payload()
    if reference is not None:
        payload["reference"] = reference
    if reference_source is not None:
        payload["reference_source"] = reference_source
    return payload
