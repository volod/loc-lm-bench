"""embedder-first-hit-rank-adoption-bar -- does an encoder's RANK gain reach the answer?

Pure and file-driven: the comparison consumes canonical per-case rows, and the orchestration takes
an injected lane runner, so the whole vertical runs in the lightweight CI install (no FAISS, no
backend, no GPU). Only the CLI layers real stores and `run-eval` on top.
"""

import json
from pathlib import Path

import pytest

from llb.core.config import RunConfig
from llb.eval.embedder_adoption import (
    DECISION_EXTEND_BAR,
    DECISION_KEEP_BAR,
    DECISION_NO_EVIDENCE,
    CellSpec,
    EmbedderLane,
    build_cells,
    cell_config,
    compare_cells,
    format_report,
    format_summary,
    parse_rerankers,
    parse_top_ks,
    run_adoption_bar_sweep,
    with_reciprocal_rank,
)
from llb.eval.embedder_adoption.models import METRIC_OBJECTIVE, METRIC_RECIPROCAL_RANK
from llb.goldset.schema import GoldItem
from llb.rag.rerank import DEFAULT_RERANKER

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


# --- the grid -----------------------------------------------------------------------------


def test_the_grid_is_the_product_of_the_two_knobs_that_make_rank_bind():
    cells = build_cells(parse_top_ks("10,3"), parse_rerankers("off,on"))
    assert [cell.label for cell in cells] == ["k10", "k10+rerank", "k3", "k3+rerank"]
    assert [cell.reranker for cell in cells] == [None, DEFAULT_RERANKER, None, DEFAULT_RERANKER]


def test_the_grid_selections_are_deduplicated_and_validated():
    assert parse_top_ks("10, 3 ,10") == [10, 3]
    assert parse_rerankers("off,off,cross/encoder-x") == [None, "cross/encoder-x"]
    with pytest.raises(ValueError, match="must be an integer"):
        parse_top_ks("ten")
    with pytest.raises(ValueError, match="at least 1"):
        parse_top_ks("0")
    with pytest.raises(ValueError, match="at least one top_k"):
        parse_top_ks(" , ")


def test_a_cell_config_carries_the_encoder_its_store_was_built_with(tmp_path: Path):
    """The store root moves WITH the encoder: `index_dir()` resolves from `data_dir`."""
    base = RunConfig(data_dir=tmp_path, top_k=5, reranker="stale/cross-encoder")
    cfg = cell_config(base, CellSpec(3, None), EmbedderLane(CANDIDATE, tmp_path / "bge"))
    assert cfg.embedding_model == CANDIDATE
    assert cfg.index_dir() == tmp_path / "bge" / "llb" / "rag"
    assert cfg.top_k == 3
    assert cfg.reranker is None  # the off half of the grid must be able to CLEAR the knob
    assert cfg.run_name == f"embedder-adoption-k3-{CANDIDATE}"


# --- the derived first-hit-rank column ----------------------------------------------------


def test_reciprocal_rank_is_derived_from_the_bundles_first_hit_rank():
    rows = with_reciprocal_rank([_row("q0", 1.0, rank=1), _row("q1", 1.0, rank=4)])
    assert [row[METRIC_RECIPROCAL_RANK] for row in rows] == [1.0, 0.25]


def test_an_item_whose_context_carried_no_gold_span_has_reciprocal_rank_zero():
    """No hit is rank 0.0, the same convention `llb.rag.retrieval.reciprocal_rank` uses."""
    rows = with_reciprocal_rank([_row("q0", 0.0, rank=None, hit=0.0)])
    assert rows[0][METRIC_RECIPROCAL_RANK] == 0.0


# --- the per-cell comparison and the bar verdict -------------------------------------------


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


def _ids(n: int) -> list[str]:
    return [f"q{i}" for i in range(n)]


def test_a_rank_gain_that_reaches_the_answer_in_one_cell_extends_the_bar():
    """The only outcome that justifies a second bar -- and it is SCOPED to the cell that showed it."""
    ids = _ids(12)
    report = _sweep(
        {
            # k=10: the candidate ranks earlier, and both encoders answer identically.
            "k10": (
                [_row(i, 1.0, rank=3) for i in ids],
                [_row(i, 1.0, rank=1) for i in ids],
            ),
            # k=3: the baseline's rank-3 evidence falls off the budget and its answers collapse.
            "k3": (
                [_row(i, 0.0, rank=None, hit=0.0) for i in ids],
                [_row(i, 1.0, rank=1) for i in ids],
            ),
        }
    )
    verdict = report["verdict"]
    assert verdict["decision"] == DECISION_EXTEND_BAR
    assert verdict["answer_cells"] == ["k3"]
    assert verdict["rank_cells"] == ["k10", "k3"]
    assert "k3" in verdict["reason"]


def test_a_rank_gain_that_never_reaches_the_answer_keeps_recall_as_the_sole_bar():
    """The measured negative: the encoder does rank better and the answers do not move."""
    ids = _ids(12)
    report = _sweep(
        {
            "k10": (
                [_row(i, 1.0, rank=3) for i in ids],
                [_row(i, 1.0, rank=1) for i in ids],
            ),
            "k3": (
                [_row(i, 1.0, rank=2) for i in ids],
                [_row(i, 1.0, rank=1) for i in ids],
            ),
        }
    )
    verdict = report["verdict"]
    assert verdict["decision"] == DECISION_KEEP_BAR
    assert verdict["answer_cells"] == []
    assert verdict["rank_cells"] == ["k10", "k3"]
    assert "recall@k stays the sole adoption bar" in verdict["reason"]


def test_a_sweep_where_the_rank_gain_never_reproduces_decides_nothing():
    """An unmet premise is not a `keep` -- the sweep never tested the question."""
    ids = _ids(12)
    report = _sweep(
        {"k10": ([_row(i, 1.0, rank=1) for i in ids], [_row(i, 1.0, rank=1) for i in ids])}
    )
    verdict = report["verdict"]
    assert verdict["decision"] == DECISION_NO_EVIDENCE
    assert verdict["rank_cells"] == []
    assert "never tested" in verdict["reason"]


def test_a_one_item_answer_gain_does_not_clear_the_interval():
    """The bar reads the paired INTERVAL, never the point estimate."""
    ids = _ids(12)
    report = _sweep(
        {
            "k3": (
                [_row(i, 0.0, rank=2) for i in ids],
                [_row(i, 1.0 if i == "q0" else 0.0, rank=1) for i in ids],
            )
        }
    )
    assert report["cells"][0]["paired"][METRIC_OBJECTIVE]["delta"]["mean"] > 0.0
    assert report["verdict"]["decision"] == DECISION_KEEP_BAR


def test_every_cell_is_paired_on_the_same_items_and_the_same_resample_draw():
    """Common random numbers across cells: the cells are comparable to each other, not just within."""
    ids = _ids(10)
    rows = [_row(i, 1.0, rank=1) for i in ids]
    report = _sweep({"k10": (rows, rows), "k3": (rows, rows)})
    assert report["item_ids"] == sorted(ids)
    assert all(cell["n"] == 10 for cell in report["cells"])
    intervals = [cell["paired"][METRIC_OBJECTIVE]["delta"] for cell in report["cells"]]
    assert intervals[0] == intervals[1]


def test_a_cell_that_scored_a_different_item_set_is_not_a_comparison():
    ids = _ids(4)
    with pytest.raises(ValueError, match="different item sets"):
        _sweep(
            {
                "k10": ([_row(i, 1.0) for i in ids], [_row(i, 1.0) for i in ids]),
                "k3": ([_row(i, 1.0) for i in ids], [_row(i, 1.0) for i in ids[:3]]),
            }
        )


def test_a_cell_missing_an_encoder_fails_loudly():
    ids = _ids(4)
    with pytest.raises(ValueError, match="did not score"):
        compare_cells(
            [(CellSpec(10, None), {BASELINE: [_row(i, 1.0) for i in ids]})],
            {},
            baseline=BASELINE,
            candidate=CANDIDATE,
            resamples=0,
        )


def test_an_empty_sweep_is_refused():
    with pytest.raises(ValueError, match="at least one cell"):
        compare_cells([], {}, baseline=BASELINE, candidate=CANDIDATE)


# --- reporting ----------------------------------------------------------------------------


def test_report_renders_ascii_tables_with_the_verdict_and_every_cell():
    ids = _ids(6)
    report = _sweep(
        {
            "k10": ([_row(i, 1.0, rank=3) for i in ids], [_row(i, 1.0, rank=1) for i in ids]),
            "k3": ([_row(i, 0.0, rank=None, hit=0.0) for i in ids], [_row(i, 1.0) for i in ids]),
        }
    )
    text = format_report(report, metadata={"model": "m", "backend": "ollama"})
    assert "# Embedder adoption bar" in text
    assert "### Answer-side delta per cell" in text
    assert "### Per-encoder means" in text
    assert all(label in text for label in ("k10", "k3"))
    assert DECISION_EXTEND_BAR in text
    assert text.isascii()
    summary = format_summary(report)
    assert "d objective" in summary and summary.isascii()


# --- orchestration ------------------------------------------------------------------------


def _gold_item(item_id: str, verified: bool = True) -> GoldItem:
    return GoldItem(
        id=item_id,
        lang="uk",
        question=f"питання {item_id}",
        reference_answer="відповідь",
        source_doc_id="doc",
        source_spans=[{"doc_id": "doc", "char_start": 0, "char_end": 9, "text": "відповідь"}],
        provenance="human-authored",
        verified=verified,
        split="final",
    )


def _write_goldset(goldset: Path, verified: bool = True) -> None:
    goldset.write_text(
        "".join(
            _gold_item(item_id, verified).model_dump_json(exclude_none=True) + "\n"
            for item_id in ("q1", "q2")
        ),
        encoding="utf-8",
    )


def _recording_lane(tmp_path: Path, seen: list[tuple[str, str, int, str | None]]):
    """A fake lane runner whose candidate encoder ranks first and answers better at a small k."""

    def fake_lane(config: RunConfig, items: list[GoldItem], split: str) -> Path:
        seen.append((config.embedding_model, str(config.data_dir), config.top_k, config.reranker))
        run_dir = tmp_path / "run-eval" / f"{config.run_name}-{split}"
        run_dir.mkdir(parents=True, exist_ok=True)
        scores = run_dir / "scores.jsonl"
        is_candidate = config.embedding_model == CANDIDATE
        rank = 1 if is_candidate else 3
        objective = 1.0 if is_candidate or config.top_k >= rank else 0.0
        scores.write_text(
            "".join(json.dumps(_row(item.id, objective, rank)) + "\n" for item in items),
            encoding="utf-8",
        )
        return scores

    return fake_lane


def _lanes(tmp_path: Path) -> list[EmbedderLane]:
    return [
        EmbedderLane(BASELINE, tmp_path / "e5"),
        EmbedderLane(CANDIDATE, tmp_path / "bge"),
    ]


def test_every_cell_scores_the_same_items_under_both_encoders_and_persists(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_goldset(goldset)
    seen: list[tuple[str, str, int, str | None]] = []

    run = run_adoption_bar_sweep(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        build_cells([10, 2], [None]),
        _lanes(tmp_path),
        out_dir=tmp_path / "adoption",
        resamples=50,
        run_lane=_recording_lane(tmp_path, seen),
    )

    assert [entry[0] for entry in seen] == [BASELINE, CANDIDATE] * 2
    assert [entry[2] for entry in seen] == [10, 10, 2, 2]
    assert {entry[1] for entry in seen} == {str(tmp_path / "e5"), str(tmp_path / "bge")}
    assert run.report["item_ids"] == ["q1", "q2"]
    assert [cell["label"] for cell in run.report["cells"]] == ["k10", "k2"]
    assert run.report["cells"][0]["lanes"][CANDIDATE]["run_dirs"] == [
        str(tmp_path / "run-eval" / f"embedder-adoption-k10-{CANDIDATE}-final")
    ]
    persisted = json.loads(Path(run.paths["comparison"]).read_text(encoding="utf-8"))
    assert persisted["metadata"]["grounding"] == "verified"
    assert persisted["metadata"]["split"] == "final"
    assert Path(run.paths["report"]).read_text(encoding="utf-8").startswith("# Embedder adoption")


def test_the_sweep_compares_exactly_two_encoders(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    _write_goldset(goldset)
    cfg = RunConfig(data_dir=tmp_path, goldset_path=goldset)
    with pytest.raises(ValueError, match="exactly two encoders"):
        run_adoption_bar_sweep(cfg, build_cells([10], [None]), _lanes(tmp_path)[:1])
    with pytest.raises(ValueError, match="must differ"):
        run_adoption_bar_sweep(
            cfg,
            build_cells([10], [None]),
            [EmbedderLane(BASELINE, tmp_path / "a"), EmbedderLane(BASELINE, tmp_path / "b")],
        )


def test_a_drafted_ledger_is_scorable_only_on_request_and_says_so_in_every_artifact(
    tmp_path: Path,
):
    goldset = tmp_path / "goldset.jsonl"
    _write_goldset(goldset, verified=False)
    cfg = RunConfig(data_dir=tmp_path, goldset_path=goldset)
    cells = build_cells([10], [None])
    with pytest.raises(SystemExit, match="no verified"):
        run_adoption_bar_sweep(cfg, cells, _lanes(tmp_path), run_lane=_recording_lane(tmp_path, []))

    run = run_adoption_bar_sweep(
        cfg,
        cells,
        _lanes(tmp_path),
        out_dir=tmp_path / "adoption",
        resamples=0,
        verified_only=False,
        run_lane=_recording_lane(tmp_path, []),
    )
    assert "grounding: `drafted`" in Path(run.paths["report"]).read_text(encoding="utf-8")


def test_several_splits_pool_into_one_compared_item_set(tmp_path: Path):
    goldset = tmp_path / "goldset.jsonl"
    items = [_gold_item("q1"), _gold_item("q2")]
    items[1].split = "tuning"
    goldset.write_text(
        "".join(item.model_dump_json(exclude_none=True) + "\n" for item in items),
        encoding="utf-8",
    )
    run = run_adoption_bar_sweep(
        RunConfig(data_dir=tmp_path, goldset_path=goldset),
        build_cells([10], [None]),
        _lanes(tmp_path),
        splits=["final", "tuning"],
        out_dir=tmp_path / "adoption",
        resamples=0,
        run_lane=_recording_lane(tmp_path, []),
    )
    assert run.report["item_ids"] == ["q1", "q2"]
    assert len(run.report["cells"][0]["lanes"][BASELINE]["run_dirs"]) == 2
