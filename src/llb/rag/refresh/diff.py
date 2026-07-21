"""Manifest diff: the per-document change set between an indexed store and the current corpus.

Both sides are `doc_id -> fingerprint` maps (see `corpus_doc_fingerprints` for the vector store
and the graph meta's `doc_fingerprints` for the graph store). The diff is pure and deterministic:
sorted doc-id lists per class, so refresh plans, logs, and reports are stable.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ManifestDiff:
    """Sorted per-class doc ids from diffing indexed vs current per-doc fingerprints."""

    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def changed(self) -> set[str]:
        """Doc ids that need re-chunk/re-embed/re-extract (added + modified)."""
        return set(self.added) | set(self.modified)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    def summary(self) -> str:
        return (
            f"{len(self.added)} added, {len(self.modified)} modified, "
            f"{len(self.deleted)} deleted, {len(self.unchanged)} unchanged"
        )

    def counts(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "modified": len(self.modified),
            "deleted": len(self.deleted),
            "unchanged": len(self.unchanged),
        }


def diff_fingerprints(indexed: dict[str, str], current: dict[str, str]) -> ManifestDiff:
    """Classify every doc id across the indexed and current fingerprint maps."""
    added = sorted(doc_id for doc_id in current if doc_id not in indexed)
    deleted = sorted(doc_id for doc_id in indexed if doc_id not in current)
    common = set(indexed) & set(current)
    modified = sorted(doc_id for doc_id in common if indexed[doc_id] != current[doc_id])
    unchanged = sorted(common - set(modified))
    return ManifestDiff(added=added, modified=modified, deleted=deleted, unchanged=unchanged)
