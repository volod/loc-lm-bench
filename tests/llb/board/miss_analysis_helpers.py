"""Miss analysis (miss-analysis-recommendations): classifier, clusters, recommendations, probes.

Drives `llb.board.miss_analysis` + `llb.board.miss_probe` over a synthetic scored bundle that
contains exactly one case of every miss class plus a clean hit, so class separation (zero
cross-class leakage), numeric evidence in every recommendation line, and probe reuse/resume are
all provable without a backend, store, or GPU.
"""

import json
from pathlib import Path


from llb.board.miss_analysis.classify import analyze_run
from llb.board.miss_analysis.model import (
    MissAnalysis,
)
from llb.core.config import RunConfig
from llb.goldset.schema import GoldItem

DOC_A = "doc_a.txt"
DOC_B = "doc_b.txt"
RUN_ID = "cafe01234567"


def _item(item_id: str, question: str, doc_id: str = DOC_A) -> GoldItem:
    return GoldItem(
        id=item_id,
        lang="uk",
        question=question,
        reference_answer="еталон",
        source_doc_id=doc_id,
        source_spans=[{"doc_id": doc_id, "char_start": 10, "char_end": 16, "text": "еталон"}],
        provenance="sample-generated",
        verified=True,
        split="final",
    )


def _goldset() -> list[GoldItem]:
    return [
        _item("m-retr", "Коли ухвалили закон про авторське право?"),
        _item("m-gen", "Хто підписав закон про авторське право?"),
        _item("m-refuse", "Що каже закон про авторське право?"),
        _item("m-empty", "Де зареєстрrelated закон?"),
        _item("m-judge", "Скільки статей у законі про авторське право?"),
        _item("hit", "Яка столиця України?", doc_id=DOC_B),
    ]


def _hit_retrieval(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "retrieved": [
            {"doc_id": DOC_A, "char_start": 0, "char_end": 40, "rank": 1, "text_preview": "x"}
        ],
        "gold_spans": [{"doc_id": DOC_A, "char_start": 10, "char_end": 16, "text": "еталон"}],
    }


def _miss_retrieval(item_id: str) -> dict:
    return {
        "item_id": item_id,
        "retrieved": [
            {"doc_id": DOC_B, "char_start": 0, "char_end": 40, "rank": 1, "text_preview": "y"}
        ],
        "gold_spans": [{"doc_id": DOC_A, "char_start": 10, "char_end": 16, "text": "еталон"}],
    }


def _score_row(item_id: str, status: str, objective: float, hit: float, **extra) -> dict:
    row = {
        "item_id": item_id,
        "split": "final",
        "status": status,
        "objective_score": objective,
        "token_f1": objective,
        "exact": 0.0,
        "contains": 0.0,
        "retrieval_hit": hit,
        "first_hit_rank": 1 if hit else None,
        "tokens_per_s": 10.0,
        "latency_s": 0.5,
        "completion_tokens": 12,
        "answer_preview": "відповідь",
    }
    row.update(extra)
    return row


def _bundle_config(tmp_path: Path, **overrides) -> RunConfig:
    goldset_path = tmp_path / "goldset.jsonl"
    return RunConfig(
        data_dir=tmp_path,
        model="fake-uk",
        backend="ollama",
        top_k=5,
        run_name="rag-eval",
        goldset_path=goldset_path,
        **overrides,
    )


def _write_bundle(
    tmp_path: Path,
    rows: list[dict],
    retrieval_rows: list[dict] | None,
    *,
    name: str = "20260101T000000.000000Z-" + RUN_ID,
    objective: float = 0.3,
) -> Path:
    config = _bundle_config(tmp_path)
    run_dir = tmp_path / "run-eval" / name
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": RUN_ID,
        "run_name": config.run_name,
        "split": "final",
        "config": config.fingerprint(),
        "metrics": {"objective_score": objective, "reliability": 1.0, "tokens_per_s": 10.0},
        "n_cases": len(rows),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "scores.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    if retrieval_rows is not None:
        (run_dir / "retrieval.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in retrieval_rows),
            encoding="utf-8",
        )
    return run_dir


def _all_class_rows() -> tuple[list[dict], list[dict]]:
    """One case per miss class + one clean hit; retrieval records agree with the story."""
    rows = [
        _score_row("m-retr", "ok", 0.1, 0.0),
        _score_row("m-gen", "ok", 0.2, 1.0),
        _score_row("m-refuse", "refusal", 0.0, 1.0),
        _score_row("m-empty", "empty", 0.0, 1.0),
        _score_row("m-judge", "ok", 0.1, 1.0, judge_score=0.9),
        _score_row("hit", "ok", 1.0, 1.0),
    ]
    retrieval = [
        _miss_retrieval("m-retr"),
        _hit_retrieval("m-gen"),
        _hit_retrieval("m-refuse"),
        _hit_retrieval("m-empty"),
        _hit_retrieval("m-judge"),
        _hit_retrieval("hit"),
    ]
    return rows, retrieval


def _analyze(tmp_path: Path, **kwargs) -> MissAnalysis:
    rows, retrieval = _all_class_rows()
    run_dir = _write_bundle(tmp_path, rows, retrieval)
    return analyze_run(run_dir, _goldset(), **kwargs)


# --------------------------------------------------------------------------- classification


# --------------------------------------------------------------------------- clusters + labels


# --------------------------------------------------------------------------- recommendations


# --------------------------------------------------------------------------- artifacts + report


# --------------------------------------------------------------------------- retrieval.jsonl


# --------------------------------------------------------------------------- probe mode
