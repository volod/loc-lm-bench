"""Re-cluster ONE fit's scored pairs at any threshold, without Splink and without re-fitting.

`engine.run_linkage` resolves the specification's own `match_threshold` through Splink's
clustering. A consumer that has to price several candidate cuts -- read a threshold off a curve,
or measure what each one would do downstream -- needs the same connected-components resolution at
an arbitrary cut, and re-fitting per cut would change the model as well as the threshold, which is
exactly the confound the reading is meant to avoid.

So this is deliberately the same rule Splink applies: an edge for every pair at or above the cut,
connected components over those edges, and the smallest member id as the cluster id. One fit, many
cuts, and the pairs the cuts are read from are the ones already published in `pairs.jsonl`.
"""

from collections.abc import Iterable, Sequence

from llb.linkage.model import LinkageCluster, LinkagePair, sorted_clusters


def cluster_pairs(
    record_ids: Iterable[str], pairs: Sequence[LinkagePair], threshold: float
) -> tuple[LinkageCluster, ...]:
    """Connected components of the pairs at or above `threshold`, over every record.

    Every record id is returned in exactly one cluster, singletons included -- a record no pair
    reached is an identity of one, not a record the decision is silent about.
    """
    parent = {record_id: record_id for record_id in record_ids}
    for pair in pairs:
        if pair.match_probability >= threshold:
            _union(parent, pair.left_id, pair.right_id)
    members: dict[str, list[str]] = {}
    for record_id in parent:
        members.setdefault(_root(parent, record_id), []).append(record_id)
    return sorted_clusters(
        LinkageCluster(cluster_id=cluster_id, record_ids=tuple(sorted(ids)))
        for cluster_id, ids in members.items()
    )


def _root(parent: dict[str, str], key: str) -> str:
    """The component representative of `key`, compressing the path it was found through."""
    if key not in parent:
        raise KeyError(f"pair references record {key!r}, which is not in the record table")
    while parent[key] != key:
        parent[key] = parent[parent[key]]
        key = parent[key]
    return key


def _union(parent: dict[str, str], left: str, right: str) -> None:
    """Join two components, keeping the SMALLER id as the representative.

    Splink names a cluster after its smallest member, and a reader comparing a re-clustered cut
    against a published `clusters.jsonl` should not have to translate ids to do it.
    """
    left_root, right_root = _root(parent, left), _root(parent, right)
    if left_root == right_root:
        return
    low, high = sorted((left_root, right_root))
    parent[high] = low
