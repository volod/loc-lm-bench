"""The semantic prefix tree returns exactly what an all-pairs scan would, having skipped most of it.

The tree's whole value is skipping work, so the load-bearing test is not "did it find most pairs"
but "is its output identical to brute force" -- pruning uses the angular triangle inequality, which
makes exactness provable rather than probabilistic. These tests assert that equality across
clustered, uniform, and degenerate inputs, and also assert that real pruning happened, so a tree
that silently degraded into an exhaustive scan would fail rather than quietly pass.
"""

import itertools
import math
import random

import pytest

from llb.conflicts.tree import SemanticPrefixTree
from llb.conflicts.vectorops import VectorSet, angular_distance
from llb.rag.refresh.diff import ManifestDiff
from llb.conflicts.tree_refresh import refresh_tree, tree_is_reusable, tree_meta


def brute_force(vectors: VectorSet, threshold: float):
    return sorted(
        (i, j)
        for i, j in itertools.combinations(range(len(vectors)), 2)
        if vectors.similarity(i, j) >= threshold
    )


def clustered(n: int, dim: int, clusters: int, spread: float, seed: int) -> VectorSet:
    rng = random.Random(seed)
    centers = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(clusters)]
    rows = []
    for index in range(n):
        center = centers[index % clusters]
        rows.append([value + rng.gauss(0, spread) for value in center])
    return VectorSet(rows)


@pytest.mark.parametrize(
    ("n", "dim", "clusters", "spread", "threshold"),
    [
        (120, 16, 8, 0.25, 0.9),
        (200, 32, 12, 0.35, 0.8),
        (150, 8, 5, 0.5, 0.7),
        (64, 24, 3, 0.15, 0.95),
    ],
)
def test_matching_pairs_equal_brute_force(n, dim, clusters, spread, threshold):
    vectors = clustered(n, dim, clusters, spread, seed=n + dim)
    tree = SemanticPrefixTree.build(vectors, leaf_size=16)
    assert [(a, b) for a, b, _ in tree.matching_pairs(vectors, threshold)] == brute_force(
        vectors, threshold
    )


def test_pruning_actually_skips_work():
    vectors = clustered(300, 32, 15, 0.2, seed=3)
    tree = SemanticPrefixTree.build(vectors, leaf_size=16)
    exhaustive = 300 * 299 // 2
    assert len(tree.candidate_pairs(0.9)) < exhaustive * 0.5


def test_candidate_pairs_superset_of_matches():
    vectors = clustered(80, 12, 6, 0.3, seed=11)
    tree = SemanticPrefixTree.build(vectors, leaf_size=8)
    candidates = set(tree.candidate_pairs(0.85))
    assert {(a, b) for a, b, _ in tree.matching_pairs(vectors, 0.85)} <= candidates


def test_identical_vectors_do_not_recurse_forever():
    vectors = VectorSet([[1.0, 0.0, 0.0]] * 50)
    tree = SemanticPrefixTree.build(vectors, leaf_size=4)
    assert len(tree.matching_pairs(vectors, 0.99)) == 50 * 49 // 2


def test_single_and_empty_inputs():
    empty = SemanticPrefixTree.build(VectorSet([]), leaf_size=4)
    assert empty.matching_pairs(VectorSet([]), 0.9) == []
    assert empty.stats()["n_vectors"] == 0
    one = VectorSet([[0.0, 1.0]])
    assert SemanticPrefixTree.build(one, leaf_size=4).matching_pairs(one, 0.5) == []


def test_build_is_deterministic():
    vectors = clustered(100, 16, 7, 0.3, seed=5)
    first = SemanticPrefixTree.build(vectors, leaf_size=8).payload()
    second = SemanticPrefixTree.build(vectors, leaf_size=8).payload()
    assert first == second


def test_leaf_size_respected_when_separable():
    vectors = clustered(120, 16, 10, 0.2, seed=9)
    tree = SemanticPrefixTree.build(vectors, leaf_size=8)
    oversized = [leaf for leaf in tree.leaves() if len(leaf.members) > 8]
    assert not oversized


def test_radius_bounds_every_member():
    vectors = clustered(60, 10, 4, 0.4, seed=13)
    tree = SemanticPrefixTree.build(vectors, leaf_size=8)
    for node in tree.nodes.values():
        for member in node.members:
            distance = angular_distance(
                sum(x * y for x, y in zip(node.centroid, vectors.row(member)))
            )
            assert distance <= node.radius + 1e-9


def test_save_load_round_trip(tmp_path):
    vectors = clustered(90, 16, 6, 0.3, seed=21)
    tree = SemanticPrefixTree.build(vectors, leaf_size=8)
    path = tmp_path / "tree" / "tree.json"
    tree.save(path)
    reloaded = SemanticPrefixTree.load(path)
    assert reloaded.matching_pairs(vectors, 0.85) == tree.matching_pairs(vectors, 0.85)


def test_load_rejects_a_foreign_version(tmp_path):
    path = tmp_path / "tree.json"
    path.write_text('{"version": "other", "root_id": 0, "leaf_size": 4, "nodes": []}')
    with pytest.raises(SystemExit, match="rebuild"):
        SemanticPrefixTree.load(path)


def test_pure_python_and_numpy_paths_agree():
    rows = [[random.Random(i).gauss(0, 1) for _ in range(12)] for i in range(40)]
    accelerated = VectorSet(rows, use_numpy=True)
    plain = VectorSet(rows, use_numpy=False)
    fast = SemanticPrefixTree.build(accelerated, leaf_size=8).matching_pairs(accelerated, 0.5)
    slow = SemanticPrefixTree.build(plain, leaf_size=8).matching_pairs(plain, 0.5)
    assert [(a, b) for a, b, _ in fast] == [(a, b) for a, b, _ in slow]


def _chunks(doc_ids):
    return [
        {"doc_id": doc_id, "chunk_id": f"{doc_id}#{i}", "char_start": 0, "char_end": 1, "text": ""}
        for i, doc_id in enumerate(doc_ids)
    ]


def test_refresh_matches_a_rebuild_on_the_same_state():
    """An incrementally refreshed tree answers queries identically to one rebuilt from scratch."""
    doc_ids = [f"doc-{i // 4}.md" for i in range(48)]
    before = clustered(48, 16, 6, 0.3, seed=31)
    tree = SemanticPrefixTree.build(before, leaf_size=8)

    after = clustered(48, 16, 6, 0.3, seed=31)
    changed = "doc-2.md"
    diff = ManifestDiff(added=[], modified=[changed], deleted=[], unchanged=[])
    refreshed, outcome = refresh_tree(tree, diff, _chunks(doc_ids), after)

    rebuilt = SemanticPrefixTree.build(after, leaf_size=8)
    assert [(a, b) for a, b, _ in refreshed.matching_pairs(after, 0.85)] == [
        (a, b) for a, b, _ in rebuilt.matching_pairs(after, 0.85)
    ]
    assert outcome.inserted == 4


def test_refresh_touches_only_affected_branches():
    doc_ids = [f"doc-{i // 4}.md" for i in range(48)]
    vectors = clustered(48, 16, 6, 0.3, seed=37)
    tree = SemanticPrefixTree.build(vectors, leaf_size=4)
    total_nodes = len(tree.nodes)
    diff = ManifestDiff(added=[], modified=["doc-1.md"], deleted=[], unchanged=[])
    _, outcome = refresh_tree(tree, diff, _chunks(doc_ids), vectors)
    assert not outcome.rebuilt
    assert outcome.touched_nodes < total_nodes


def test_refresh_rebuilds_when_most_of_the_corpus_changed():
    doc_ids = [f"doc-{i // 4}.md" for i in range(48)]
    vectors = clustered(48, 16, 6, 0.3, seed=41)
    tree = SemanticPrefixTree.build(vectors, leaf_size=8)
    diff = ManifestDiff(added=[], modified=sorted({d for d in doc_ids}), deleted=[], unchanged=[])
    _, outcome = refresh_tree(tree, diff, _chunks(doc_ids), vectors)
    assert outcome.rebuilt


def test_refresh_is_a_no_op_without_changes():
    vectors = clustered(20, 8, 3, 0.3, seed=43)
    tree = SemanticPrefixTree.build(vectors, leaf_size=8)
    before = tree.payload()
    _, outcome = refresh_tree(
        tree, ManifestDiff(unchanged=["a.md"]), _chunks(["a.md"] * 20), vectors
    )
    assert (outcome.inserted, outcome.removed, outcome.rebuilt) == (0, 0, False)
    assert tree.payload() == before


def test_tree_meta_pins_the_encoder():
    vectors = clustered(20, 8, 3, 0.3, seed=47)
    tree = SemanticPrefixTree.build(vectors, leaf_size=8)
    meta = tree_meta(
        tree,
        embedding_model="intfloat/multilingual-e5-base",
        dim=8,
        corpus_fingerprint="abc",
        doc_fingerprints={"a.md": "x"},
        cos_threshold=0.9,
    )
    assert tree_is_reusable(meta, "intfloat/multilingual-e5-base", 8)
    assert not tree_is_reusable(meta, "some/other-encoder", 8)
    assert not tree_is_reusable(meta, "intfloat/multilingual-e5-base", 768)
    assert math.isfinite(float(meta["max_radius_rad"]))


def test_centering_restores_isotropy_on_an_anisotropic_space():
    """Encoder spaces put every vector in one cone; centering is what makes a threshold mean something."""
    rng = random.Random(101)
    # A shared offset of 9 against unit noise over 16 dims puts unrelated pairs near cosine 0.83
    # -- the value actually measured between unrelated chunks in multilingual-E5 space.
    bias = [9.0] + [0.0] * 15
    rows = [[b + rng.gauss(0, 1) for b in bias] for _ in range(300)]
    raw = VectorSet(rows)
    centered = raw.centered()

    pairs = [(rng.randrange(300), rng.randrange(300)) for _ in range(1500)]
    raw_median = sorted(raw.similarity(a, b) for a, b in pairs if a != b)[750]
    centered_median = sorted(centered.similarity(a, b) for a, b in pairs if a != b)[750]
    assert raw_median > 0.7, "the biased space really is anisotropic"
    assert abs(centered_median) < 0.2, "centering pulls unrelated pairs back toward zero"


def test_pairs_above_equals_the_tree_and_a_brute_force_scan():
    """Three independent implementations of the same exact question must agree."""
    vectors = clustered(140, 16, 8, 0.3, seed=57)
    tree = SemanticPrefixTree.build(vectors, leaf_size=16)
    scan = [(a, b) for a, b, _ in vectors.pairs_above(0.85)]
    assert scan == [(a, b) for a, b, _ in tree.matching_pairs(vectors, 0.85)]
    assert scan == brute_force(vectors, 0.85)


def test_pairs_above_numpy_and_pure_python_agree():
    rows = [[random.Random(i * 7).gauss(0, 1) for _ in range(10)] for i in range(60)]
    fast = VectorSet(rows, use_numpy=True).pairs_above(0.4)
    slow = VectorSet(rows, use_numpy=False).pairs_above(0.4)
    assert [(a, b) for a, b, _ in fast] == [(a, b) for a, b, _ in slow]


def test_centering_an_empty_set_is_safe():
    assert len(VectorSet([]).centered()) == 0
