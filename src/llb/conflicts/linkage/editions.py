"""Edition groups: the fit's identity clusters, ordered by the governance fields.

A cluster says which documents the model put in one identity. An edition GROUP is that cluster read
as a document's history -- the members in edition order, with the newest named -- which is the unit
a supersession decision is taken on. The ordering is `compare_editions`, the same function the
tiers already attach to every finding, so a group and a finding cannot disagree about which side is
newer.

Naming the newest is not deciding anything about it: the group is a proposal, and nothing here
retires, rewrites, or removes a document.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llb.conflicts.governance.editions import (
    BASIS_EFFECTIVE_DATE,
    BASIS_VERSION,
    SIDE_A,
    compare_editions,
    edition_key,
)
from llb.core.contracts.common import JsonObject
from llb.linkage.model import LinkageCluster

EDITION_ID_PREFIX = "E"


@dataclass(frozen=True)
class EditionGroup:
    """One proposed edition group: its members oldest first, and the newest one when it orders."""

    edition_id: str
    doc_ids: tuple[str, ...]
    current: tuple[str, ...]
    basis: str | None

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def payload(self) -> JsonObject:
        return {
            "edition_id": self.edition_id,
            "size": self.size,
            "doc_ids": list(self.doc_ids),
            "current": list(self.current),
            "basis": self.basis,
        }


def _sort_key(governance: JsonObject, doc_id: str) -> tuple[Any, ...]:
    """Oldest-first ordering key: `effective_date` decides, `version` breaks its ties.

    The two fields in that precedence are `compare_editions`'s rule; a record carrying neither
    sorts before the ones that do, so an undated document never displaces a dated one from the top.
    """
    date = edition_key(governance, BASIS_EFFECTIVE_DATE)
    version = edition_key(governance, BASIS_VERSION)
    return (date is not None, date or (), version is not None, version or (), doc_id)


def _current(
    members: Sequence[str], governance: dict[str, JsonObject]
) -> tuple[tuple[str, ...], str | None]:
    """The members no other member is newer than, and the field that decided the ordering.

    A LIST rather than one name, because two byte-identical re-uploads carrying the same date are
    one edition held twice -- naming either of them as the current one would invent a precedence
    the governance fields do not record. `basis` is None when no pair in the group could be ordered
    at all, which is the corpus ingested without dates rather than a tie inside a dated group.
    """
    basis: str | None = None
    current: list[str] = []
    for member in members:
        newer_exists = False
        for other in members:
            if other == member:
                continue
            staleness = compare_editions(governance[other], governance[member])
            basis = basis or staleness.basis
            newer_exists = newer_exists or staleness.newer_side == SIDE_A
        if not newer_exists:
            current.append(member)
    return tuple(current), basis


def edition_groups(
    clusters: Sequence[LinkageCluster], governance: dict[str, JsonObject]
) -> tuple[EditionGroup, ...]:
    """Every multi-document cluster as an edition group, largest first.

    Singletons are dropped: a document in an identity of one has no edition history to read, and
    listing every unclustered document would bury the groups a reviewer opened the file for.
    """
    groups: list[EditionGroup] = []
    multi = [cluster for cluster in clusters if cluster.size > 1]
    for index, cluster in enumerate(multi, start=1):
        members = tuple(
            sorted(cluster.record_ids, key=lambda doc_id: _sort_key(governance[doc_id], doc_id))
        )
        current, basis = _current(members, governance)
        groups.append(
            EditionGroup(
                edition_id=f"{EDITION_ID_PREFIX}{index}",
                doc_ids=members,
                current=current,
                basis=basis,
            )
        )
    return tuple(groups)


def editions_by_document(groups: Sequence[EditionGroup]) -> dict[str, str]:
    """Which edition group each grouped document belongs to, for a per-document lookup."""
    return {doc_id: group.edition_id for group in groups for doc_id in group.doc_ids}
