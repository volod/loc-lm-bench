"""Sweep engine over an injected store, its published bundle, and the per-constant verdict."""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.eval.restoration_sweep.grid import (
    LANE_CLEAN,
    LANE_NORMALIZE,
    LANE_OFF,
    SWEEP_VARIANT_CLASSES,
    policy_grid,
    run_restoration_sweep,
)
from llb.eval.restoration_sweep.audit import AuditCounts
from llb.eval.restoration_sweep.lanes import LaneReading, SweepResult
from llb.eval.restoration_sweep.report import render_report
from llb.eval.restoration_sweep.run import run_and_publish_sweep
from llb.eval.restoration_sweep.verdict import (
    CONSTANT_CUTOFF,
    CONSTANT_RANK,
    CONSTANT_SURFACE,
    VERDICT_ADOPT,
    VERDICT_EXPOSE,
    VERDICT_PIN,
    constant_verdicts,
    recommended_policy,
)
from llb.rag.query_prep.restore_policy import DEFAULT_RESTORATION_POLICY, RestorationPolicy

CHUNK_TEXT = "герцогство було засновано коли норманів очолював граф рожер у місті"
QUESTION = "Коли засновано герцогство норманів?"


class FakeStore:
    """A lexical-overlap store: the gold chunk comes back when the query shares a rare token."""

    def __init__(self) -> None:
        self.chunk = {
            "doc_id": "doc",
            "char_start": 0,
            "char_end": len(CHUNK_TEXT),
            "text": CHUNK_TEXT,
            "rank": 1,
            "retrieval_score": 1.0,
        }
        self.chunks = [self.chunk]
        self.queries: list[tuple[str, str]] = []

    def retrieve_queries(self, dense_query, lexical_query, k, chunk_filter=None):
        self.queries.append((dense_query, lexical_query))
        indexed = set(CHUNK_TEXT.split())
        hit = any(token in indexed for token in lexical_query.casefold().split())
        return [dict(self.chunk)] if hit else []


def _goldset(path: Path) -> Path:
    items = [
        {
            "id": f"q{index}",
            "lang": "uk",
            "question": QUESTION,
            "reference_answer": "у 1071 році",
            "source_doc_id": "doc",
            "source_spans": [
                {"doc_id": "doc", "char_start": 0, "char_end": len(CHUNK_TEXT), "text": CHUNK_TEXT}
            ],
            "provenance": "human-authored",
            "verified": True,
            "split": "final",
        }
        for index in (1, 2)
    ]
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8"
    )
    return path


@pytest.fixture
def sweep_config(tmp_path, monkeypatch) -> RunConfig:
    store = FakeStore()
    monkeypatch.setattr("llb.executor.runner_retrieval._load_store", lambda config: store)
    return RunConfig(
        data_dir=tmp_path / "data",
        goldset_path=_goldset(tmp_path / "goldset.jsonl"),
        corpus_root=tmp_path,
        top_k=5,
    )


def test_every_setting_is_measured_on_every_class_against_shared_reference_lanes(sweep_config):
    policies = policy_grid([0, 1], [3], ["morphology"])
    result = run_restoration_sweep(sweep_config, policies=policies)
    assert result.policies == policies
    assert len(result.settings) == len(policies) * len(SWEEP_VARIANT_CLASSES)
    for lane in (LANE_CLEAN, LANE_OFF, LANE_NORMALIZE):
        assert result.pooled(lane).n == 2 * len(SWEEP_VARIANT_CLASSES)
    # every lane is paired item for item, which is what the delta reading depends on
    lengths = {reading.n for reading in (*result.settings, *result.references)}
    assert lengths == {2}


def test_the_sweep_is_deterministic_and_labels_every_correction_it_made(sweep_config):
    policies = (DEFAULT_RESTORATION_POLICY,)
    first = run_restoration_sweep(sweep_config, policies=policies)
    second = run_restoration_sweep(sweep_config, policies=policies)
    assert [record.as_row() for record in first.edits] == [
        record.as_row() for record in second.edits
    ]
    pooled = first.pooled(DEFAULT_RESTORATION_POLICY.label)
    assert pooled.counts.corrections == len(first.edits)
    assert pooled.counts.corrections == (
        pooled.counts.correct + pooled.counts.wrong + pooled.counts.unaligned
    )


def test_the_published_bundle_carries_the_report_the_rows_and_the_audit(sweep_config):
    run = run_and_publish_sweep(
        sweep_config, policies=policy_grid([0, 1], [3], ["morphology"]), resamples=50
    )
    assert set(run.paths) == {"report", "settings", "edit_audit", "metadata"}
    report = Path(run.paths["report"]).read_text(encoding="utf-8")
    assert "## Verdict per constant" in report
    assert DEFAULT_RESTORATION_POLICY.label in report
    rows = [
        json.loads(line)
        for line in Path(run.paths["settings"]).read_text(encoding="utf-8").splitlines()
    ]
    assert {row["setting"] for row in rows} >= {LANE_CLEAN, LANE_OFF, LANE_NORMALIZE}
    assert {verdict.constant for verdict in run.verdicts} == {
        CONSTANT_SURFACE,
        CONSTANT_CUTOFF,
        CONSTANT_RANK,
    }
    with pytest.raises(FileExistsError):
        from llb.eval.restoration_sweep.report import write_sweep_artifacts

        write_sweep_artifacts(run.result, run.verdicts, run.out_dir, {})


def _reading(lane: str, hits: tuple[float, ...], counts: AuditCounts) -> LaneReading:
    return LaneReading(
        lane=lane,
        variant_class=SWEEP_VARIANT_CLASSES[0],
        hits=hits,
        reciprocal_ranks=hits,
        counts=counts,
    )


def _result(alternative: RestorationPolicy, hits: tuple[float, ...], counts: AuditCounts):
    default_counts = AuditCounts(corrections=10, correct=9, wrong=1, opportunities=10, restored=9)
    return SweepResult(
        policies=(DEFAULT_RESTORATION_POLICY, alternative),
        variant_classes=(SWEEP_VARIANT_CLASSES[0],),
        settings=(
            _reading(DEFAULT_RESTORATION_POLICY.label, (0.0,) * 10, default_counts),
            _reading(alternative.label, hits, counts),
        ),
        references=(),
        edits=(),
        item_ids=tuple(f"q{index}" for index in range(10)),
        top_k=5,
    )


def test_an_alternative_that_retrieves_more_at_no_precision_cost_moves_the_default():
    alternative = RestorationPolicy(ambiguous_token_max_chars=3)
    result = _result(
        alternative,
        (1.0,) * 10,
        AuditCounts(corrections=12, correct=11, wrong=1, opportunities=12, restored=11),
    )
    verdicts = {v.constant: v for v in constant_verdicts(result, resamples=200, seed=13)}
    assert verdicts[CONSTANT_CUTOFF].verdict == VERDICT_ADOPT
    assert recommended_policy(list(verdicts.values())) == alternative
    assert verdicts[CONSTANT_SURFACE].verdict == VERDICT_PIN


def test_an_alternative_that_buys_recall_with_wrong_corrections_stays_a_knob():
    alternative = RestorationPolicy(surface_max_distance=1)
    result = _result(
        alternative,
        (1.0,) * 10,
        AuditCounts(corrections=12, correct=6, wrong=6, opportunities=12, restored=6),
    )
    verdicts = {v.constant: v for v in constant_verdicts(result, resamples=200, seed=13)}
    assert verdicts[CONSTANT_SURFACE].verdict == VERDICT_EXPOSE
    assert recommended_policy(list(verdicts.values())) == DEFAULT_RESTORATION_POLICY
    assert "query_prep_surface_max_distance" in verdicts[CONSTANT_SURFACE].rationale


def test_an_alternative_that_retrieves_no_more_pins_the_conservative_default():
    alternative = RestorationPolicy(rank_order="context")
    result = _result(
        alternative,
        (0.0,) * 10,
        AuditCounts(corrections=10, correct=8, wrong=2, opportunities=10, restored=8),
    )
    verdicts = {v.constant: v for v in constant_verdicts(result, resamples=200, seed=13)}
    assert verdicts[CONSTANT_RANK].verdict == VERDICT_PIN
    report = render_report(
        result,
        list(verdicts.values()),
        {
            "goldset": "gs",
            "split": "final",
            "embedding_model": "e5",
            "seed": 13,
            "typo_rate": 0.08,
            "lane": "normalize,typos",
            "typo_guard": True,
        },
    )
    assert "**pin**" in report
    assert "unchanged from the shipped default" in report
