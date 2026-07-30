"""Focused compact-vs-cap lane: active gate, paired evidence, and persistence."""

import json
from pathlib import Path

from llb.bench.agentic.context_budget import fixed_budget
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_compact_vs_cap import run_compact_vs_cap
from llb.bench.agentic_compact_vs_cap_report import (
    METHOD,
    VERDICT_INACTIVE,
    VERDICT_PREFER_COMPACT,
)
from llb.bench.agentic_context_report import (
    METRIC_COMPLETION,
    METRIC_TOTAL_MODEL_INPUT_TOKENS,
)

BIG_HIT = "дані про бюджет громади " * 250


def long_tasks(n: int = 8) -> list[AgenticTask]:
    return [
        AgenticTask(
            id=f"long-{i}",
            prompt="Знайди дані, повторно перевір їх і повідом готово.",
            setup={"corpus": {"d1": BIG_HIT}},
            success=[{"kind": "answer_contains", "value": "готово"}],
        )
        for i in range(n)
    ]


def active_complete(prompt: str) -> str:
    if "Стисло підсумуй" in prompt:
        return "пошук перевірено"
    if "підсумок попередніх кроків" in prompt:
        return "готово"
    return '{"name":"search","arguments":{"query":"дані"}}'


def test_active_lane_pairs_compact_directly_against_cap_and_persists(tmp_path: Path):
    run = run_compact_vs_cap(
        long_tasks(),
        model="fake",
        backend="fake",
        complete=active_complete,
        max_steps=8,
        budget=fixed_budget(5000),
        observation_cap_chars=400,
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    assert run.n_compactions >= len(long_tasks())
    assert run.n_compacted_episodes == len(long_tasks())
    assert run.verdict == VERDICT_PREFER_COMPACT
    assert run.paired[METRIC_COMPLETION]["delta"]["mean"] > 0
    assert METRIC_TOTAL_MODEL_INPUT_TOKENS in run.paired
    assert "including summarizer" not in run.table  # table reports the metric by its contract name
    assert "d(total-model-input-tok)" in run.table

    root = tmp_path / METHOD
    manifests = sorted(root.glob("*/manifest.json"))
    assert len(manifests) == 3
    summary = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in manifests
        if json.loads(path.read_text(encoding="utf-8"))["run_name"].endswith("summary")
    )
    assert summary["config"]["n_compactions"] == run.n_compactions
    assert summary["config"]["verdict"] == VERDICT_PREFER_COMPACT


def test_lane_calls_out_an_inactive_task_shape():
    run = run_compact_vs_cap(
        long_tasks(4),
        model="fake",
        backend="fake",
        complete=lambda _prompt: "готово",
        budget=fixed_budget(50_000),
        persist=False,
    )
    assert run.n_compactions == 0
    assert run.verdict == VERDICT_INACTIVE
    assert "tighten" in run.reason
