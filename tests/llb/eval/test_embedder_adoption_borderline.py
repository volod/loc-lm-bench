"""adoption-single-sweep-carries-no-borderline-signal -- is ONE model's verdict settled?

The documented per-model recipe is a SINGLE `compare-embedder-adoption` run, so the sweep has to
say for itself whether the cell its verdict names sits on the cut -- an operator must not have to
assemble a five-model roster to find out. These tests cover the annotation as the sweep produces
it: persisted per cell, rendered in the report and the terminal summary, and qualifying the verdict
reason, plus the invariant that it equals what the roster re-derives from the same run bundles.

Pure and file-driven, like the rest of the lane: fake per-case rows and an injected lane runner, so
the whole vertical runs in the lightweight CI install.
"""

import json
from pathlib import Path

from llb.core.config import RunConfig
from llb.eval.embedder_adoption.models import (
    DECISION_EXTEND_BAR,
    DECISION_KEEP_BAR,
    EmbedderLane,
)
from llb.eval.embedder_adoption.cells import build_cells
from llb.eval.embedder_adoption.screen_data import cell_item_deltas
from llb.eval.embedder_adoption.compare import compare_cells
from llb.eval.embedder_adoption.report import (
    format_report,
    format_summary,
)
from llb.eval.embedder_adoption.stability import row_stability
from llb.eval.embedder_adoption.run import run_adoption_bar_sweep
from llb.eval.embedder_adoption.models import METRIC_OBJECTIVE, CellSpec, ItemDeltas
from llb.goldset.schema import GoldItem

BASELINE = "intfloat/multilingual-e5-base"
CANDIDATE = "BAAI/bge-m3"


def _row(item_id: str, objective: float, rank: int | None = 1, hit: float = 1.0) -> dict:
    return {
        "item_id": item_id,
        "split": "final",
        "status": "ok",
        "objective_score": objective,
        "token_f1": objective,
        "exact": 0.0,
        "contains": 1.0 if objective > 0 else 0.0,
        "retrieval_hit": hit,
        "first_hit_rank": rank,
    }


def _ids(n: int) -> list[str]:
    return [f"q{i}" for i in range(n)]


def _sweep(cells: dict[str, tuple[list[dict], list[dict]]], resamples: int = 200):
    specs = {"k10": CellSpec(10, None), "k3": CellSpec(3, None), "k3+rerank": CellSpec(3, "x")}
    return compare_cells(
        [
            (specs[label], {BASELINE: base, CANDIDATE: cand})
            for label, (base, cand) in cells.items()
        ],
        {},
        baseline=BASELINE,
        candidate=CANDIDATE,
        resamples=resamples,
    )


def _gold_item(item_id: str) -> GoldItem:
    return GoldItem(
        id=item_id,
        lang="uk",
        question=f"питання {item_id}",
        reference_answer="відповідь",
        source_doc_id="doc",
        source_spans=[{"doc_id": "doc", "char_start": 0, "char_end": 9, "text": "відповідь"}],
        provenance="human-authored",
        verified=True,
        split="final",
    )


def _lanes(tmp_path: Path) -> list[EmbedderLane]:
    return [EmbedderLane(BASELINE, tmp_path / "e5"), EmbedderLane(CANDIDATE, tmp_path / "bge")]


def _near_miss_pair(ids: list[str], wins: int, losses: int) -> tuple[list[dict], list[dict]]:
    """A cell whose objective delta lands close to the cut: `wins` items up, `losses` items down.

    The exact sign-flip tail puts 7/1 at p=0.0352 (separates at 90%, not 95%) and 6/0 at p=0.0156
    (separates at 95%, not 97.5% because that level also needs seven discordant items).
    """
    up, down = set(ids[:wins]), set(ids[wins : wins + losses])
    return (
        [_row(i, 1.0 if i in down else 0.0, rank=2) for i in ids],
        [_row(i, 1.0 if i in up else 0.0, rank=2) for i in ids],
    )


def test_every_cell_carries_the_stability_of_its_own_reading():
    """The one-model sweep says how settled its verdict is, without assembling a roster."""
    ids = _ids(30)
    report = _sweep(
        {
            "k10": ([_row(i, 1.0, rank=3) for i in ids], [_row(i, 1.0, rank=1) for i in ids]),
            "k3": ([_row(i, 0.0, rank=None, hit=0.0) for i in ids], [_row(i, 1.0) for i in ids]),
        },
        resamples=2000,
    )
    settled = {cell["label"]: cell["stability"] for cell in report["cells"]}
    assert settled["k10"]["reading"] == "rank_only"  # ranks earlier, answers identically
    assert settled["k3"]["reading"] == "answer"
    assert settled["k3"]["p_positive"] == 1.0
    assert all(entry["borderline"] is False for entry in settled.values())


def test_the_persisted_stability_is_measured_from_the_sweeps_own_resample_draw():
    """Not a second draw beside the intervals: the same one, so the annotation matches the row."""
    ids = _ids(30)
    rows = {"k3": _near_miss_pair(ids, wins=7, losses=1)}
    report = _sweep(rows, resamples=2000)
    cell = report["cells"][0]
    deltas = ItemDeltas(
        item_ids=sorted(ids),
        objective=[
            c[METRIC_OBJECTIVE] - b[METRIC_OBJECTIVE]
            for b, c in zip(
                sorted(rows["k3"][0], key=lambda r: r["item_id"]),
                sorted(rows["k3"][1], key=lambda r: r["item_id"]),
            )
        ],
        reciprocal_rank=[0.0] * len(ids),
    )
    assert cell["stability"] == row_stability(deltas, resamples=2000, seed=report["seed"])


def test_a_knife_edge_negative_is_marked_and_the_keep_reason_says_so():
    """The point of the task: `keep_bar` must not print the same for a miss and a near-miss."""
    ids = _ids(30)
    report = _sweep(
        {
            # The rank premise holds so the sweep can decide at all, and the objective just misses.
            "k10": ([_row(i, 1.0, rank=3) for i in ids], [_row(i, 1.0, rank=1) for i in ids]),
            "k3": _near_miss_pair(ids, wins=7, losses=1),
        },
        resamples=2000,
    )
    borderline = next(cell for cell in report["cells"] if cell["label"] == "k3")
    assert borderline["stability"]["borderline"] is True
    assert borderline["stability"]["side"] == "below"
    verdict = report["verdict"]
    assert verdict["decision"] == DECISION_KEEP_BAR  # the decision itself never moves
    assert verdict["borderline_cells"] == ["k3"]
    assert "BORDERLINE" in verdict["reason"] and "too close to call" in verdict["reason"]
    assert "0.90 convention would read it `answer`" in verdict["reason"]
    text = format_report(report)
    assert "### How close each cell sits to the cut" in text
    assert "p_positive" in text and "neither (borderline)" in text
    assert text.isascii()
    assert "(borderline)" in format_summary(report)


def test_an_extend_bar_resting_on_a_near_miss_positive_is_qualified_too():
    """The two-sided half: the cell that SCOPES the second bar can itself be on the cut."""
    ids = _ids(30)
    report = _sweep({"k3": _near_miss_pair(ids, wins=6, losses=0)}, resamples=2000)
    cell = report["cells"][0]
    assert cell["stability"]["side"] == "above"
    verdict = report["verdict"]
    assert verdict["decision"] == DECISION_EXTEND_BAR
    assert verdict["answer_cells"] == ["k3"] and verdict["borderline_cells"] == ["k3"]
    assert "0.975 convention would read it `insufficient_evidence`" in verdict["reason"]


def test_a_settled_verdict_reason_carries_no_borderline_clause():
    ids = _ids(30)
    report = _sweep(
        {"k3": ([_row(i, 0.0, rank=None, hit=0.0) for i in ids], [_row(i, 1.0) for i in ids])},
        resamples=2000,
    )
    assert report["verdict"]["decision"] == DECISION_EXTEND_BAR
    assert report["verdict"]["borderline_cells"] == []
    assert "BORDERLINE" not in report["verdict"]["reason"]


def test_a_sweep_that_drew_no_resamples_carries_no_exceedance_probability():
    """`p_positive` is a share OF resamples; a persisted 0.0 would read as a confident negative."""
    ids = _ids(6)
    report = _sweep({"k3": ([_row(i, 0.0) for i in ids], [_row(i, 1.0) for i in ids])}, resamples=0)
    assert "stability" not in report["cells"][0]
    assert report["verdict"]["borderline_cells"] == []
    assert format_report(report).count("How close each cell sits") == 0


def test_the_persisted_stability_equals_what_a_roster_re_derives_from_the_same_bundles(
    tmp_path: Path,
):
    """The gate on the whole annotation: one-model run and roster must not read a cell differently.

    The sweep measures stability from the vectors it holds in memory; the roster and the screen
    re-derive the per-item deltas from the `run-eval` bundles the sweep names. Both orders are the
    sorted shared item ids and both draws are `(n, resamples, seed)`, so they must agree exactly --
    if they ever drift, an operator's own run would contradict the roster table.
    """

    ids = _ids(30)
    near_miss = set(ids[:7]), set(ids[7:8])
    goldset = tmp_path / "goldset.jsonl"
    goldset.write_text(
        "".join(_gold_item(i).model_dump_json(exclude_none=True) + "\n" for i in ids),
        encoding="utf-8",
    )

    def fake_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scores = run_dir / "scores.jsonl"
        won = near_miss[0] if config.embedding_model == CANDIDATE else near_miss[1]
        scores.write_text(
            "".join(
                json.dumps(_row(item.id, 1.0 if item.id in won else 0.0, rank=2)) + "\n"
                for item in items
            ),
            encoding="utf-8",
        )
        return scores

    run = run_adoption_bar_sweep(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        build_cells([10], [None]),
        _lanes(tmp_path),
        out_dir=tmp_path / "adoption",
        resamples=2000,
        run_lane=fake_lane,
    )
    cell = run.report["cells"][0]
    re_derived = row_stability(
        cell_item_deltas(run.report, cell["label"]),
        resamples=run.report["resamples"],
        confidence=run.report["confidence"],
        seed=run.report["seed"],
    )
    assert cell["stability"] == re_derived
    assert cell["stability"]["borderline"] is True  # a cell worth checking the agreement on
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["cells"][0]["stability"] == re_derived
