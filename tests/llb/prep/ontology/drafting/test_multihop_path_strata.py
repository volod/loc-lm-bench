"""Relation/document-stratified multi-hop path selection and relative review headroom."""

import pytest

from llb.goldset.schema import SourceSpan
from llb.prep.ontology.models import MultiHopSeed, MultiHopStep
from llb.prep.ontology.drafting.path_strata import PathStratumTargets, select_stratified_paths
from llb.prep.ontology.pipeline.expansion_audit import minimum_combined_items


def _path(
    path_id: int,
    relations: tuple[str, str],
    documents: tuple[str, str],
) -> MultiHopSeed:
    steps = [
        MultiHopStep(
            subject=f"s{path_id}-{index}",
            relation=relation,
            object=f"o{path_id}-{index}",
            section_title="S",
            evidence=SourceSpan(
                doc_id=doc_id,
                char_start=path_id * 10 + index,
                char_end=path_id * 10 + index + 1,
                text="x",
            ),
        )
        for index, (relation, doc_id) in enumerate(zip(relations, documents))
    ]
    return MultiHopSeed(
        steps=steps,
        bridge=f"b{path_id}",
        start=f"a{path_id}",
        end=f"c{path_id}",
    )


def test_path_stratification_covers_relation_mode_and_document_targets_deterministically():
    candidates = [
        _path(0, ("r1", "r2"), ("a.md", "a.md")),
        _path(1, ("r1", "r2"), ("a.md", "b.md")),
        _path(2, ("r3", "r4"), ("b.md", "b.md")),
        _path(3, ("r5", "r6"), ("c.md", "c.md")),
    ]
    targets = PathStratumTargets(
        relation_pair=1,
        document_mode=1,
        source_document=1,
    )

    first, report = select_stratified_paths(
        candidates,
        max_paths=3,
        targets=targets,
        source_documents=["a.md", "b.md", "c.md"],
    )
    second, repeated = select_stratified_paths(
        candidates,
        max_paths=3,
        targets=targets,
        source_documents=["a.md", "b.md", "c.md"],
    )

    assert first == second
    assert report == repeated
    assert report["all_requested_covered_or_exhausted"] is True
    assert report["document_modes"]["cross-document"]["selected"] == 1
    assert all(row["status"] == "covered" for row in report["source_documents"].values())


def test_path_stratification_marks_unavailable_strata_exhausted():
    selected, report = select_stratified_paths(
        [_path(0, ("r1", "r2"), ("a.md", "a.md"))],
        max_paths=4,
        targets=PathStratumTargets(
            relation_pair=1,
            document_mode=1,
            source_document=1,
        ),
        source_documents=["a.md", "missing.md"],
    )

    assert len(selected) == 1
    assert report["document_modes"]["cross-document"]["status"] == "exhausted"
    assert report["source_documents"]["missing.md"]["status"] == "exhausted"
    assert report["all_requested_covered_or_exhausted"] is True


def test_path_stratification_does_not_call_a_known_stratum_exhausted_by_budget():
    _, report = select_stratified_paths(
        [
            _path(0, ("r1", "r2"), ("a.md", "a.md")),
            _path(1, ("r3", "r4"), ("b.md", "b.md")),
        ],
        max_paths=1,
        targets=PathStratumTargets(
            relation_pair=1,
            document_mode=1,
            source_document=1,
        ),
        source_documents=["a.md", "b.md"],
    )

    assert report["relation_pairs"]["r3 -> r4"]["status"] == "budget-exhausted"
    assert report["all_requested_covered_or_exhausted"] is False


def test_relative_headroom_scales_with_the_carried_ledger():
    assert minimum_combined_items(10, 0.15) == 12
    assert minimum_combined_items(100, 0.15) == 115
    with pytest.raises(ValueError, match="between zero and one"):
        minimum_combined_items(10, 1.1)
