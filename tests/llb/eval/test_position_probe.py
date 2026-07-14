"""rerank-context-order: lost-in-the-middle probe (`llb probe-context-position`).

Pure: a fake store supplies the gold chunk + real-distractor pool and a fake chat answers
correctly ONLY when the gold chunk leads the prompt, so probe construction, per-position
scoring, the recommendation rule, and the artifacts are provable without a backend or GPU.
"""

from pathlib import Path

import pytest

from llb.core.contracts import ChatMessage, ChunkRecord
from llb.eval.position_probe import (
    POSITION_HEAD,
    POSITION_MIDDLE,
    POSITION_TAIL,
    POSITIONS,
    PositionSummary,
    ProbeCase,
    assemble_context_chunks,
    build_probe_cases,
    position_index,
    recommend_order,
    run_probe,
)
from llb.eval.position_probe_report import write_probe
from llb.goldset.schema import GoldItem

GOLD_TEXT = "Київ є столицею України."


def _gold_item(item_id: str) -> GoldItem:
    return GoldItem(
        id=item_id,
        lang="uk",
        question=f"Питання {item_id}: яка столиця України?",
        reference_answer="Київ",
        source_doc_id="kyiv.txt",
        source_spans=[
            {"doc_id": "kyiv.txt", "char_start": 0, "char_end": len(GOLD_TEXT), "text": GOLD_TEXT}
        ],
        provenance="public-reused",
        verified=True,
        split="final",
    )


def _gold_chunk() -> ChunkRecord:
    return {"doc_id": "kyiv.txt", "char_start": 0, "char_end": len(GOLD_TEXT), "text": GOLD_TEXT}


def _distractor(i: int) -> ChunkRecord:
    return {
        "doc_id": "other.txt",
        "char_start": i * 100,
        "char_end": i * 100 + 20,
        "text": f"Нерелевантний фрагмент {i}.",
    }


class FakeStore:
    """Returns the same candidate pool for every question (gold at a mid rank)."""

    def __init__(self, candidates: list[ChunkRecord]) -> None:
        self.candidates = candidates
        self.requested: list[int] = []

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]:
        self.requested.append(k)
        return self.candidates[:k]


def head_only_chat(messages: list[ChatMessage]) -> tuple[str, str | None]:
    """Answers correctly only when the gold chunk is the FIRST context block."""
    user = messages[-1]["content"]
    return ("Київ" if user.find(GOLD_TEXT) < user.find("Нерелевантний") else "Не знаю", None)


def test_position_index_slots():
    assert position_index(POSITION_HEAD, 5) == 0
    assert position_index(POSITION_MIDDLE, 5) == 2
    assert position_index(POSITION_TAIL, 5) == 4
    with pytest.raises(ValueError, match="position"):
        position_index("edge", 5)


def test_build_probe_cases_selects_gold_and_distractors():
    pool = [_distractor(1), _gold_chunk(), _distractor(2), _distractor(3)]
    cases, skipped = build_probe_cases([_gold_item("q1")], FakeStore(pool), k=3)
    assert len(cases) == 1 and not any(skipped.values())
    assert cases[0].gold_chunk["text"] == GOLD_TEXT
    assert [c["text"] for c in cases[0].distractors] == [
        "Нерелевантний фрагмент 1.",
        "Нерелевантний фрагмент 2.",
    ]


def test_build_probe_cases_counts_skip_reasons_and_rejects_tiny_k():
    no_gold = FakeStore([_distractor(1), _distractor(2), _distractor(3)])
    cases, skipped = build_probe_cases([_gold_item("q1")], no_gold, k=3)
    assert not cases and skipped["gold_not_retrieved"] == 1

    too_few = FakeStore([_gold_chunk(), _distractor(1)])
    cases, skipped = build_probe_cases([_gold_item("q1")], too_few, k=3)
    assert not cases and skipped["too_few_distractors"] == 1

    with pytest.raises(ValueError, match=">= 3"):
        build_probe_cases([_gold_item("q1")], no_gold, k=2)


def test_assemble_context_places_gold_exactly():
    case = ProbeCase(
        item=_gold_item("q1"),
        gold_chunk=_gold_chunk(),
        distractors=[_distractor(1), _distractor(2)],
    )
    for position in POSITIONS:
        chunks = assemble_context_chunks(case, position, k=3)
        assert len(chunks) == 3
        assert chunks[position_index(position, 3)]["text"] == GOLD_TEXT


def test_run_probe_scores_positions_and_recommends_rank(tmp_path: Path):
    pool = [_distractor(1), _gold_chunk(), _distractor(2), _distractor(3)]
    items = [_gold_item("q1"), _gold_item("q2")]
    report = run_probe(items, FakeStore(pool), head_only_chat, model="m", backend="ollama", k=3)
    assert report.n_items == 2 and len(report.rows) == 6  # 2 items x 3 positions
    by_pos = {p.position: p for p in report.positions}
    assert by_pos[POSITION_HEAD].mean_score == 1.0
    assert by_pos[POSITION_MIDDLE].mean_score == 0.0
    assert by_pos[POSITION_TAIL].mean_score == 0.0
    assert report.recommendation == "rank"  # head wins -> best-first
    assert by_pos[POSITION_HEAD].ci is not None  # >= 2 scores -> bootstrap CI

    paths = write_probe(report, tmp_path / "probe")
    assert Path(paths["report"]).is_file() and Path(paths["cases"]).is_file()
    rendered = Path(paths["report"]).read_text(encoding="utf-8")
    assert "**rank**" in rendered and "| head |" in rendered
    assert len(Path(paths["cases"]).read_text(encoding="utf-8").splitlines()) == 6


def test_recommend_reverse_rank_when_tail_wins():
    positions = [
        PositionSummary(POSITION_HEAD, 4, 0.2, (0.1, 0.3)),
        PositionSummary(POSITION_MIDDLE, 4, 0.1, (0.0, 0.2)),
        PositionSummary(POSITION_TAIL, 4, 0.9, (0.8, 1.0)),
    ]
    order, note = recommend_order(positions)
    assert order == "reverse_rank"
    assert "CIs overlap" not in note

    overlapping = [
        PositionSummary(POSITION_HEAD, 4, 0.5, (0.3, 0.7)),
        PositionSummary(POSITION_MIDDLE, 4, 0.4, (0.2, 0.6)),
        PositionSummary(POSITION_TAIL, 4, 0.6, (0.4, 0.8)),
    ]
    order, note = recommend_order(overlapping)
    assert order == "reverse_rank" and "CIs overlap" in note
