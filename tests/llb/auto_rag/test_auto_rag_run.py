"""Resumable auto-RAG stage graph tests with injected lightweight stages."""

import json
from pathlib import Path

import pytest

from llb.auto_rag.models import STAGES, AutoRagSettings
from llb.auto_rag.run import run_auto_rag


def _settings(tmp_path: Path, run_id: str = "test-run") -> AutoRagSettings:
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    candidates = tmp_path / "models.yaml"
    candidates.write_text("models: []\n", encoding="utf-8")
    return AutoRagSettings(
        corpus=corpus,
        data_dir=tmp_path,
        run_id=run_id,
        draft_model="fake-12b",
        candidates=candidates,
    )


def _fake_stages(calls: dict[str, int]):
    def runner(name: str):
        def run(settings, outputs):
            del settings
            calls[name] = calls.get(name, 0) + 1
            assert list(outputs) == list(STAGES[: STAGES.index(name)])
            if name == "recommendation":
                return {"recommendation": "rec.yaml", "report": "report.md"}
            return {"artifact": f"{name}.json"}

        return run

    return {name: runner(name) for name in STAGES}


@pytest.mark.parametrize("interrupted_after", STAGES)
def test_resume_after_every_stage_boundary(tmp_path: Path, interrupted_after: str) -> None:
    settings = _settings(tmp_path, interrupted_after)
    calls: dict[str, int] = {}
    stages = _fake_stages(calls)

    def interrupt(stage: str, _result: dict) -> None:
        if stage == interrupted_after:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_auto_rag(settings, stages=stages, after_stage=interrupt)
    result = run_auto_rag(settings, stages=stages)

    assert result.resumed is True
    assert result.completed == STAGES
    assert calls == {stage: 1 for stage in STAGES}
    events = [
        json.loads(line)
        for line in (settings.run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["stage"] == interrupted_after and row["status"] == "completed" for row in events)


def test_resume_refuses_changed_settings(tmp_path: Path) -> None:
    first = _settings(tmp_path)
    run_auto_rag(first, stages=_fake_stages({}))
    changed = AutoRagSettings(
        **{
            **first.manifest_payload(),
            "corpus": first.corpus,
            "data_dir": first.data_dir,
            "candidates": first.candidates,
            "trials": 99,
        }
    )
    with pytest.raises(ValueError, match="settings differ"):
        run_auto_rag(changed, stages=_fake_stages({}))
