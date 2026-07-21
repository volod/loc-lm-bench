"""Adaptive retrieval validation behavior."""

from pathlib import Path

from llb.auto_rag import retrieval


class _Store:
    def __init__(self, label: str):
        self.label = label

    def save(self, path: Path) -> None:
        path.mkdir(parents=True)
        (path / "selected.txt").write_text(self.label, encoding="utf-8")


def test_retrieval_searches_repairs_only_after_baseline_miss(tmp_path, monkeypatch) -> None:
    scores = iter(
        [
            {"n": 3, "k": 10, "recall_at_k": 0.6, "mrr": 0.4},
            {"n": 3, "k": 10, "recall_at_k": 1.0, "mrr": 0.8},
            {"n": 3, "k": 10, "recall_at_k": 0.7, "mrr": 0.5},
            {"n": 3, "k": 10, "recall_at_k": 0.8, "mrr": 0.6},
        ]
    )

    def fake_evaluate(corpus, items, candidate, k, embedder):
        del corpus, items, k, embedder
        return _Store(candidate.strategy + str(candidate.chunk_size)), next(scores)

    monkeypatch.setattr(retrieval, "load_goldset", lambda _path: [object()])
    monkeypatch.setattr(retrieval, "_evaluate", fake_evaluate)
    result = retrieval.validate_and_repair_retrieval(
        tmp_path / "corpus",
        tmp_path / "goldset.jsonl",
        tmp_path / "stage",
        k=10,
        recall_gate=0.8,
    )
    assert result["repaired"] is True
    assert result["selected"]["chunk_size"] == 400
    assert len(result["attempts"]) == 4
    assert (tmp_path / "stage/index/selected.txt").read_text(encoding="utf-8") == "recursive400"
