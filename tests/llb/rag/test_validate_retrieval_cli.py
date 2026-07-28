"""CLI wiring for retrieval validation against an explicitly selected corpus."""

from pathlib import Path

from typer.testing import CliRunner

from llb.main import app


def test_validate_retrieval_forwards_corpus_root(monkeypatch, tmp_path: Path) -> None:
    observed = []
    monkeypatch.setattr(
        "llb.cli.rag.validate.run_retrieval_validation",
        lambda request: observed.append(request),
    )
    corpus = tmp_path / "corpus"
    result = CliRunner().invoke(
        app,
        [
            "validate-retrieval",
            "--corpus-root",
            str(corpus),
            "--goldset",
            str(tmp_path / "goldset.jsonl"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(observed) == 1
    assert observed[0].corpus_root == corpus
