"""The PCA blocker is an exact filter: it prunes work but never true cosine matches."""

import itertools
import json
import random

import pytest

from llb.conflicts.audit import AuditParams, run_audit
from llb.conflicts.semantic_tree.projected_index import prepare_projected_index
from llb.conflicts.semantic_tree.projection import (
    PCAProjection,
    euclidean_threshold,
    fit_pca_projection,
)
from llb.conflicts.store_access import StoreView
from llb.conflicts.semantic_tree.tree import SemanticPrefixTree
from llb.conflicts.semantic_tree.vectorops import VectorSet
from llb.conflicts.tiers import semantic_run

from tests.llb.conflicts.conflict_helpers import FAKE_COS_THRESHOLD, FIXTURE_CORPUS, fake_store_view


def _vectors(n: int = 180, dim: int = 40, seed: int = 73) -> VectorSet:
    rng = random.Random(seed)
    centers = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(9)]
    return VectorSet(
        [
            [value + rng.gauss(0, 0.25) for value in centers[index % len(centers)]]
            for index in range(n)
        ]
    )


def _projection(vectors: VectorSet, dims: int = 8) -> PCAProjection:
    return fit_pca_projection(
        vectors,
        dims,
        embedding_model="test-encoder",
        centered=False,
        source_fingerprint="fixture",
    )


def test_projection_never_expands_pairwise_distance():
    vectors = _vectors()
    projected = _projection(vectors).transform(vectors)
    for left, right in itertools.combinations(range(len(vectors)), 2):
        full_distance = euclidean_threshold(vectors.similarity(left, right))
        assert projected.distance(left, right) <= full_distance + 1e-10


def test_projected_rows_are_not_renormalized():
    vectors = _vectors(n=40)
    projected = _projection(vectors, dims=5).transform(vectors)
    norms = [sum(value * value for value in projected.row(i)) ** 0.5 for i in range(len(projected))]
    assert max(norms) - min(norms) > 0.1
    assert any(abs(norm - 1.0) > 0.1 for norm in norms)


@pytest.mark.parametrize("dims", [4, 8, 16])
def test_projected_filter_is_a_superset_and_final_matches_are_exact(dims):
    vectors = _vectors()
    threshold = 0.88
    projected = _projection(vectors, dims=dims).transform(vectors)
    tree = SemanticPrefixTree.build(projected, leaf_size=12)
    candidates = tree.candidate_pairs_within(euclidean_threshold(threshold))
    brute = vectors.pairs_above(threshold)
    assert {(left, right) for left, right, _ in brute} <= set(candidates)
    confirmed = vectors.pairs_above_candidates(candidates, threshold)
    assert [(left, right) for left, right, _ in confirmed] == [
        (left, right) for left, right, _ in brute
    ]
    assert [score for _, _, score in confirmed] == pytest.approx(
        [score for _, _, score in brute], abs=1e-14
    )


def test_projection_fingerprint_detects_tampering(tmp_path):
    projection = _projection(_vectors(n=40), dims=6)
    path = tmp_path / "projection.json"
    projection.save(path)
    assert PCAProjection.load(path) == projection
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mean"][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        PCAProjection.load(path)


def test_projected_index_persists_and_reuses_matching_artifacts(tmp_path):
    base = fake_store_view()
    store = StoreView(tmp_path, base.chunks, base.vectors, base.meta)
    first = prepare_projected_index(store, store.vectors, dims=8, leaf_size=8, centered=False)
    second = prepare_projected_index(store, store.vectors, dims=8, leaf_size=8, centered=False)
    assert first.meta["index_action"] == "built"
    assert second.meta["index_action"] == "reused"
    assert second.tree.payload() == first.tree.payload()
    assert (tmp_path / "semantic_tree" / "projection.json").is_file()
    assert (tmp_path / "semantic_tree" / "tree.json").is_file()


@pytest.mark.parametrize(
    ("meta_key", "foreign_value"),
    [
        ("embedding_model", "some/other-encoder"),
        ("corpus_fingerprint", "different-corpus-generation"),
    ],
)
def test_projected_index_rebuilds_foreign_source_artifacts(tmp_path, meta_key, foreign_value):
    base = fake_store_view()
    original = StoreView(tmp_path, base.chunks, base.vectors, base.meta)
    first = prepare_projected_index(original, original.vectors, dims=8, leaf_size=8, centered=False)

    changed_meta = {**base.meta, meta_key: foreign_value}
    changed = StoreView(tmp_path, base.chunks, base.vectors, changed_meta)
    replaced = prepare_projected_index(
        changed, changed.vectors, dims=8, leaf_size=8, centered=False
    )
    reused = prepare_projected_index(changed, changed.vectors, dims=8, leaf_size=8, centered=False)
    projection = json.loads(
        (tmp_path / "semantic_tree" / "projection.json").read_text(encoding="utf-8")
    )

    assert first.meta["index_action"] == "built"
    assert replaced.meta["index_action"] == "built"
    assert replaced.meta["source_fingerprint"] != first.meta["source_fingerprint"]
    assert projection["fitted_source_fingerprint"] == replaced.meta["source_fingerprint"]
    assert reused.meta["index_action"] == "reused"


def test_full_space_path_builds_fresh_geometry_for_each_encoder(tmp_path, monkeypatch):
    base = fake_store_view()
    builds = 0
    real_build_tree = semantic_run.build_tree

    def recording_build(vectors, *, leaf_size):
        nonlocal builds
        builds += 1
        return real_build_tree(vectors, leaf_size=leaf_size)

    monkeypatch.setattr(semantic_run, "build_tree", recording_build)
    first_store = StoreView(tmp_path, base.chunks, base.vectors, base.meta)
    second_store = StoreView(
        tmp_path,
        base.chunks,
        base.vectors,
        {**base.meta, "embedding_model": "some/other-encoder"},
    )

    first = run_audit(FIXTURE_CORPUS, AuditParams(effort="semantic"), store=first_store)
    second = run_audit(FIXTURE_CORPUS, AuditParams(effort="semantic"), store=second_store)

    assert builds == 2
    assert first.tree_meta["embedding_model"] == "fake-hashed-bow"
    assert second.tree_meta["embedding_model"] == "some/other-encoder"
    assert not (tmp_path / "semantic_tree").exists()


def test_audit_projected_output_equals_unprojected_scan(tmp_path):
    base = fake_store_view()
    store = StoreView(tmp_path, base.chunks, base.vectors, base.meta)
    unprojected = run_audit(
        FIXTURE_CORPUS,
        AuditParams(effort="semantic", cos_threshold=FAKE_COS_THRESHOLD),
        store=store,
    )
    projected = run_audit(
        FIXTURE_CORPUS,
        AuditParams(effort="semantic", cos_threshold=FAKE_COS_THRESHOLD, project_dims=8),
        store=store,
    )
    assert [finding.payload() for finding in projected.findings] == [
        finding.payload() for finding in unprojected.findings
    ]
    semantic = projected.tiers[-1]
    assert semantic.extra["blocking"] == "pca-euclidean"
    assert semantic.extra["projected_pruning_fraction"] >= 0.0
