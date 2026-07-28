"""CLI wiring for a run against an explicitly selected corpus."""

from pathlib import Path

from typer.testing import CliRunner

from llb.main import app


def test_run_eval_forwards_corpus_root(monkeypatch, tmp_path: Path) -> None:
    observed = []
    monkeypatch.setattr(
        "llb.cli.eval.run.execute_eval",
        lambda config, **_kwargs: observed.append(config),
    )
    corpus = tmp_path / "corpus"
    result = CliRunner().invoke(app, ["run-eval", "--corpus-root", str(corpus), "--limit", "1"])

    assert result.exit_code == 0, result.output
    assert len(observed) == 1
    assert observed[0].corpus_root == corpus
