"""embedder-adoption-bar-reranker-model-dependence -- is the reranker gain predictable in advance?

Pure and file-driven: the input is N finished `AdoptionBarReport`s plus a declared profile per
model, so the whole roster reading is unit-tested with dict reports -- no backend, store, or GPU.
"""

from llb.eval.embedder_adoption.compare import compare_cells
from llb.eval.embedder_adoption.roster import compare_roster
from llb.eval.embedder_adoption.models import CellSpec

BASELINE = "intfloat/multilingual-e5-base"

CANDIDATE = "BAAI/bge-m3"

FOCUS = "k10+rerank"

OTHER = "k10"


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


def _ids(n: int = 12) -> list[str]:
    return [f"q{i}" for i in range(n)]


def _answer_pair(ids):
    """Candidate answers better AND ranks earlier -> READING_ANSWER."""
    return ([_row(i, 0.0, rank=None, hit=0.0) for i in ids], [_row(i, 1.0, rank=1) for i in ids])


def _rank_only_pair(ids):
    """Candidate ranks earlier, answers identically -> READING_RANK_ONLY."""
    return ([_row(i, 1.0, rank=3) for i in ids], [_row(i, 1.0, rank=1) for i in ids])


def _report(*, model: str, focus_answers: bool, ids=None):
    """A finished sweep whose FOCUS cell either reaches the answer or is rank-only."""
    ids = ids or _ids()
    focus = _answer_pair(ids) if focus_answers else _rank_only_pair(ids)
    other = _rank_only_pair(ids)
    report = compare_cells(
        [
            (CellSpec(10, None), {BASELINE: other[0], CANDIDATE: other[1]}),
            (CellSpec(10, "x"), {BASELINE: focus[0], CANDIDATE: focus[1]}),
        ],
        {},
        baseline=BASELINE,
        candidate=CANDIDATE,
        resamples=200,
    )
    report["metadata"] = {"model": model, "goldset": "gs.jsonl", "corpus": "c", "split": "final"}
    return report


def _roster(spec: list[tuple[str, bool]], profiles=None, **kwargs):
    return compare_roster(
        [_report(model=m, focus_answers=a) for m, a in spec],
        profiles,
        focus_cell=kwargs.pop("focus_cell", FOCUS),
        **kwargs,
    )
