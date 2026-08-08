"""Constant sweep for agent context-policy knobs -- trim arithmetic and pin/expose verdicts."""

import itertools

from llb.bench.agentic.context import (
    DEFAULT_KEEP_LAST_N,
    DEFAULT_OBSERVATION_CAP_CHARS,
    OBSERVATION_HEAD_SHARE,
    POLICY_KEEP_LAST_N,
    POLICY_OBSERVATION_CAP,
    ContextPolicy,
    trim_observation,
)
from llb.bench.agentic.context_budget import fixed_budget
from llb.bench.agentic.model import AgenticTask
from llb.bench.agentic_context_sweep import run_constant_sweep
from llb.bench.agentic_context_sweep_model import (
    AXIS_CAP,
    AXIS_HEAD,
    AXIS_KEEP,
    VERDICT_EXPOSE,
    VERDICT_INAPPLICABLE,
    VERDICT_PIN,
    SettingReport,
    SweepSetting,
    default_grid,
)
from llb.bench.agentic_context_sweep_verdict import decide_axis_verdict
from llb.bench.agentic_context_report import PolicyReport
from llb.bench.common import category_result
from llb.core.contracts.benchmarks import AgenticCaseRow
from llb.scoring.aggregate import TIER_AGENTIC


def _scripted(outputs):
    it = itertools.cycle(outputs)
    return lambda _prompt: next(it)


def _task(task_id: str, observation: str) -> AgenticTask:
    return AgenticTask(
        task_id,
        "знайди",
        setup={"corpus": {"d1": observation}},
        success=[{"kind": "answer_contains", "value": "готово"}],
    )


def test_trim_observation_head_share_controls_span_lengths():
    """CI covers the trim's span arithmetic: head_share moves the cut, total keep is fixed."""
    blob = ("H" * 100) + ("M" * 100) + ("T" * 100)
    head_heavy, _ = trim_observation(blob, 50, head_share=0.8, aggregate_safe=False)
    balanced, _ = trim_observation(blob, 50, head_share=0.5, aggregate_safe=False)
    # 0.8 of 50 = 40 head / 10 tail; 0.5 of 50 = 25 / 25.
    assert head_heavy.startswith("H" * 40) and head_heavy.endswith("T" * 10)
    assert balanced.startswith("H" * 25) and balanced.endswith("T" * 25)
    assert "обрізано" in head_heavy and "обрізано" in balanced


def test_context_policy_rejects_out_of_range_head_share():
    try:
        ContextPolicy(name=POLICY_OBSERVATION_CAP, observation_head_share=1.0)
    except ValueError as exc:
        assert "observation_head_share" in str(exc)
    else:
        raise AssertionError("expected ValueError for head_share=1.0")


def test_default_grid_covers_three_axes_with_shipped_cells():
    grid = default_grid()
    assert {s.axis for s in grid} == {AXIS_CAP, AXIS_HEAD, AXIS_KEEP}
    shipped = {s.axis: s for s in grid if s.is_shipped}
    assert shipped[AXIS_CAP].overrides["observation_cap_chars"] == DEFAULT_OBSERVATION_CAP_CHARS
    assert shipped[AXIS_HEAD].overrides["observation_head_share"] == OBSERVATION_HEAD_SHARE
    assert shipped[AXIS_KEEP].overrides["keep_last_n"] == DEFAULT_KEEP_LAST_N


def _fake_report(
    *,
    policy: str,
    success: list[float],
    prompt_tokens: list[float],
    overflows: int,
) -> PolicyReport:
    rows: list[AgenticCaseRow] = [
        {
            "item_id": f"t{i}",
            "success": s,
            "max_prompt_tokens": prompt_tokens[i],
            "n_steps": 1.0,
            "n_tool_calls": 1.0,
            "observation_bytes": 0.0,
            "n_compactions": 0.0,
            "n_trimmed_observations": 0.0,
            "status": "context_overflow" if i < overflows else "completed",
        }
        for i, s in enumerate(success)
    ]
    result = category_result(
        model=policy,
        backend="fake",
        tier=TIER_AGENTIC,
        case_objectives=success,
        reliability=1.0,
    )
    return PolicyReport(
        policy=policy,
        result=result,
        rows=rows,
        episodes=[],
        case_success=success,
        reliability=1.0,
        completion_ci=None,
        mean_steps=1.0,
        mean_tool_calls=1.0,
        n_context_overflow=overflows,
    )


def _cell(
    axis: str,
    label: str,
    *,
    shipped: bool,
    success: list[float],
    prompts: list[float],
    overflows: int,
    policy: str = POLICY_OBSERVATION_CAP,
) -> SettingReport:
    return SettingReport(
        setting=SweepSetting(
            axis=axis,
            label=label,
            policy_name=policy,
            overrides={},
            is_shipped=shipped,
        ),
        report=_fake_report(
            policy=policy, success=success, prompt_tokens=prompts, overflows=overflows
        ),
        paired={},
    )


def test_decide_axis_verdict_pins_when_grid_is_flat():
    # Ten identical successes so the evidence gate can reach separated if anything differed;
    # here every cell is identical, so the verdict must pin.
    success = [1.0] * 10
    prompts = [100.0] * 10
    cells = [
        _cell(AXIS_CAP, "cap=400", shipped=False, success=success, prompts=prompts, overflows=0),
        _cell(AXIS_CAP, "cap=800", shipped=True, success=success, prompts=prompts, overflows=0),
        _cell(AXIS_CAP, "cap=1600", shipped=False, success=success, prompts=prompts, overflows=0),
    ]
    from llb.bench.agentic_context_sweep_verdict import pair_against_shipped

    pair_against_shipped(cells)
    verdict = decide_axis_verdict(AXIS_CAP, cells)
    assert verdict.verdict == VERDICT_PIN


def test_decide_axis_verdict_marks_keep_last_n_inapplicable_on_uniform_overflow():
    success = [0.0] * 10
    prompts = [3000.0] * 10
    cells = [
        _cell(
            AXIS_KEEP,
            "keep=1",
            shipped=False,
            success=success,
            prompts=prompts,
            overflows=10,
            policy=POLICY_KEEP_LAST_N,
        ),
        _cell(
            AXIS_KEEP,
            "keep=2",
            shipped=False,
            success=success,
            prompts=prompts,
            overflows=10,
            policy=POLICY_KEEP_LAST_N,
        ),
        _cell(
            AXIS_KEEP,
            "keep=3",
            shipped=True,
            success=success,
            prompts=prompts,
            overflows=10,
            policy=POLICY_KEEP_LAST_N,
        ),
    ]
    verdict = decide_axis_verdict(AXIS_KEEP, cells)
    assert verdict.verdict == VERDICT_INAPPLICABLE
    assert "long-transcript" in verdict.reason


def test_decide_axis_verdict_pins_when_alternative_is_worse():
    # Shipped succeeds; tighter cap fails half the set with a CI clear of zero -- pin, naming the
    # worse cell rather than claiming the grid is flat.
    shipped_success = [1.0] * 12
    worse_success = [1.0] * 6 + [0.0] * 6
    prompts = [100.0] * 12
    cells = [
        _cell(
            AXIS_CAP,
            "cap=400",
            shipped=False,
            success=worse_success,
            prompts=prompts,
            overflows=0,
        ),
        _cell(
            AXIS_CAP,
            "cap=800",
            shipped=True,
            success=shipped_success,
            prompts=prompts,
            overflows=0,
        ),
    ]
    from llb.bench.agentic_context_sweep_verdict import pair_against_shipped

    pair_against_shipped(cells)
    verdict = decide_axis_verdict(AXIS_CAP, cells)
    assert verdict.verdict == VERDICT_PIN
    assert "worse cells" in verdict.reason
    assert "cap=400" in verdict.reason


def test_decide_axis_verdict_exposes_when_completion_separates_favorably():
    # Shipped fails every task; alternative succeeds every task -- clear separated expose.
    shipped_success = [0.0] * 12
    alt_success = [1.0] * 12
    prompts = [100.0] * 12
    cells = [
        _cell(
            AXIS_CAP, "cap=400", shipped=False, success=alt_success, prompts=prompts, overflows=0
        ),
        _cell(
            AXIS_CAP,
            "cap=800",
            shipped=True,
            success=shipped_success,
            prompts=prompts,
            overflows=0,
        ),
    ]
    from llb.bench.agentic_context_sweep_verdict import pair_against_shipped

    pair_against_shipped(cells)
    verdict = decide_axis_verdict(AXIS_CAP, cells)
    assert verdict.verdict == VERDICT_EXPOSE
    assert "cap=400" in verdict.reason


def test_run_constant_sweep_over_fake_endpoint_persists_verdicts(tmp_path):
    """End-to-end over the fake complete: unique cells run once, summary carries verdicts."""
    big = "hit " * 500
    tasks = [_task("a", big), _task("b", big)]
    # Always finish after one search so every policy completes without needing real reasoning.
    complete = _scripted(
        [
            '{"name":"search","arguments":{"query":"q"}}',
            '{"name":"finish","arguments":{"answer":"готово"}}',
        ]
    )
    # Tiny grid: one axis, two cells, so the fake endpoint stays cheap.
    settings = [
        SweepSetting(
            axis=AXIS_CAP,
            label="cap=50",
            policy_name=POLICY_OBSERVATION_CAP,
            overrides={
                "observation_cap_chars": 50,
                "observation_head_share": OBSERVATION_HEAD_SHARE,
                "keep_last_n": DEFAULT_KEEP_LAST_N,
            },
            is_shipped=False,
        ),
        SweepSetting(
            axis=AXIS_CAP,
            label="cap=800",
            policy_name=POLICY_OBSERVATION_CAP,
            overrides={
                "observation_cap_chars": DEFAULT_OBSERVATION_CAP_CHARS,
                "observation_head_share": OBSERVATION_HEAD_SHARE,
                "keep_last_n": DEFAULT_KEEP_LAST_N,
            },
            is_shipped=True,
        ),
        SweepSetting(
            axis=AXIS_HEAD,
            label="head=0.6",
            policy_name=POLICY_OBSERVATION_CAP,
            overrides={
                "observation_cap_chars": DEFAULT_OBSERVATION_CAP_CHARS,
                "observation_head_share": OBSERVATION_HEAD_SHARE,
                "keep_last_n": DEFAULT_KEEP_LAST_N,
            },
            is_shipped=True,
        ),
        SweepSetting(
            axis=AXIS_KEEP,
            label="keep=3",
            policy_name=POLICY_KEEP_LAST_N,
            overrides={
                "observation_cap_chars": DEFAULT_OBSERVATION_CAP_CHARS,
                "observation_head_share": OBSERVATION_HEAD_SHARE,
                "keep_last_n": DEFAULT_KEEP_LAST_N,
            },
            is_shipped=True,
        ),
    ]
    run = run_constant_sweep(
        tasks,
        model="fake",
        backend="fake",
        complete=complete,
        settings=settings,
        max_steps=4,
        budget=fixed_budget(50_000),
        data_dir=tmp_path,
    )
    assert len(run.settings) == 4
    assert {v.axis for v in run.verdicts} == {AXIS_CAP, AXIS_HEAD, AXIS_KEEP}
    assert "verdicts:" in run.table
    # Shared shipped observation_cap cell must not re-run: only three unique episode batches
    # (cap=50, cap=800/head=0.6, keep=3), and the summary + per-setting bundles land under the
    # method root.
    sweep_root = tmp_path / "agentic-context-sweep"
    assert sweep_root.is_dir()
    bundles = list(sweep_root.iterdir())
    assert len(bundles) >= 4  # 3 settings that persist + summary (cap=800 shared twice as cells)
