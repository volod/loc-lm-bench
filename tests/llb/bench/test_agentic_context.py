"""Agent context-management policies: the four policy rows, the per-step budget guard, the
telemetry, and the paired comparison -- all over the fake `complete` seam with no GPU."""

import itertools
import json
from pathlib import Path

import pytest

from llb.bench.agentic.context import (
    CONTEXT_POLICIES,
    POLICY_COMPACT,
    POLICY_FULL,
    POLICY_KEEP_LAST_N,
    POLICY_OBSERVATION_CAP,
    ContextPolicy,
    ContextState,
    compact_state,
    policy_history_lines,
    trim_observation,
)
from llb.bench.agentic.context_budget import (
    ContextBudget,
    fixed_budget,
    prompt_tokens,
    unbounded_budget,
)
from llb.bench.agentic.episode import (
    build_agent_prompt,
    build_agent_prompt_lines,
    run_episode,
    step_prompt,
)
from llb.bench.agentic.model import (
    STATUS_COMPLETED,
    STATUS_CONTEXT_OVERFLOW,
    STATUS_INCOMPLETE,
    AgenticTask,
)
from llb.bench.agentic_context import run_agentic_context, task_set_digest
from llb.bench.agentic_context_report import (
    METHOD,
    METRIC_COMPLETION,
    METRIC_PROMPT_TOKENS,
    completion_reading,
)
from llb.bench.tool_world import tool_catalog
from llb.board.agentic_context import agentic_context_comparison
from llb.rag.fusion_evidence.evidence_gate import READING_FLAT, READING_SEPARATED

# A corpus document that `search` matches and returns WHOLE -- one deliberately large observation
# is the exact shape that grows every later prompt for the rest of an episode.
BIG = "дані про бюджет громади " * 250


def scripted_cycle(outputs):
    """A fake endpoint that cycles a fixed script (episodes are longer than the script)."""
    it = itertools.cycle(outputs)
    return lambda _prompt: next(it)


def search_task(task_id: str = "t") -> AgenticTask:
    return AgenticTask(
        task_id,
        "знайди дані",
        setup={"corpus": {"d1": BIG}},
        success=[{"kind": "answer_contains", "value": "готово"}],
    )


# --- observation trimming -----------------------------------------------------------------


def test_trim_observation_keeps_head_and_tail_with_an_explicit_marker():
    trimmed, was_trimmed = trim_observation("A" * 100 + "Z" * 100, 50)
    assert was_trimmed is True
    assert len(trimmed) > 50  # the marker is ADDED, never counted against the kept content
    assert "обрізано 150 символів" in trimmed
    # Aggregate header sits above the body (chars-only for a non-search blob).
    assert trimmed.startswith("[агрегат: chars=200]")
    body = trimmed.split("\n", 1)[1]
    assert body.startswith("A") and body.endswith("Z")


def test_trim_observation_leaves_a_short_observation_untouched():
    assert trim_observation("коротко", 800) == ("коротко", False)
    assert trim_observation("будь-що", 0) == ("будь-що", False)  # cap 0 disables trimming


def test_trim_observation_can_disable_aggregate_header():
    trimmed, was_trimmed = trim_observation("A" * 100 + "Z" * 100, 50, aggregate_safe=False)
    assert was_trimmed is True
    assert not trimmed.startswith("[агрегат:")
    assert trimmed.startswith("A") and trimmed.endswith("Z")


def test_observation_cap_trims_search_hits_with_aggregate_header():
    from llb.bench.agentic.context_aggregate import extract_aggregate_facts

    # Middle docs carry the count; a head/tail trim without a header would lose them.
    hits = "\n".join(f"[d{i}] body-{i}-" + ("x" * 40) for i in range(10))
    facts = extract_aggregate_facts(hits)
    assert facts["hits"] == 10 and facts["doc_ids"][0] == "d0"
    trimmed, was_trimmed = trim_observation(hits, 120)
    assert was_trimmed is True
    assert trimmed.startswith("[агрегат: hits=10 chars=")
    assert "docs=d0,d1,d2,d3,d4,d5,d6,d7,d8,d9]" in trimmed.split("\n", 1)[0]


# --- policy history assembly --------------------------------------------------------------


def state_with(n_entries: int, observation: str = "ok") -> ContextState:
    state = ContextState()
    policy = ContextPolicy(name=POLICY_FULL)
    for i in range(n_entries):
        state.record(policy, "db_get", {"key": f"k{i}"}, observation)
    return state


def test_full_policy_prompt_is_byte_identical_to_the_pre_policy_loop():
    task, catalog = search_task(), tool_catalog()
    state = state_with(3)
    policy_prompt = build_agent_prompt_lines(
        task, catalog, policy_history_lines(ContextPolicy(name=POLICY_FULL), state)
    )
    assert policy_prompt == build_agent_prompt(task, catalog, state.entries)


def test_keep_last_n_drops_older_steps_and_announces_the_drop():
    lines = policy_history_lines(
        ContextPolicy(name=POLICY_KEEP_LAST_N, keep_last_n=2), state_with(5)
    )
    assert len(lines) == 3  # the marker plus the two survivors
    assert "опущено попередніх кроків: 3" in lines[0]
    assert '"k3"' in lines[1] and '"k4"' in lines[2]


def test_observation_cap_trims_only_the_oversized_observations():
    state = ContextState()
    policy = ContextPolicy(name=POLICY_OBSERVATION_CAP, observation_cap_chars=100)
    state.record(policy, "db_get", {"key": "small"}, "ok")
    state.record(policy, "search", {"query": "q"}, BIG)
    lines = policy_history_lines(policy, state)
    assert "обрізано" not in lines[0] and "обрізано" in lines[1]
    assert state.telemetry.n_trimmed_observations == 1
    assert state.telemetry.observation_bytes > len(BIG)  # counted BEFORE the trim


def test_compact_folds_older_entries_and_keeps_the_full_executed_record():
    state, policy = state_with(3), ContextPolicy(name=POLICY_COMPACT, compact_keep_recent=1)
    assert compact_state(policy, state, lambda older: f"стисло {len(older)}") is True
    assert state.summary == "стисло 2" and len(state.entries) == 1
    assert len(state.executed) == 3  # a policy changes what the model SEES, not what ran
    assert state.telemetry.n_compactions == 1
    # the summary marker carries the folded step count itself -- no separate "dropped" line
    lines = policy_history_lines(policy, state)
    assert lines[0] == "- [підсумок попередніх кроків (2): стисло 2]"
    assert not any("опущено" in line for line in lines)


def test_compact_injects_search_aggregate_facts_into_the_summary():
    state = ContextState()
    policy = ContextPolicy(name=POLICY_COMPACT, compact_keep_recent=1)
    hits = "\n".join(f"[d{i}] body" for i in range(4))
    state.record(policy, "search", {"query": "q"}, hits)
    state.record(policy, "db_get", {"key": "k"}, "ok")
    assert compact_state(policy, state, lambda _older: "модель забула число") is True
    assert "hits=4" in state.summary and "docs=d0,d1,d2,d3" in state.summary
    assert "модель забула число" in state.summary


def test_compact_trims_live_observations_with_aggregate_headers():
    """Live compact steps share observation_cap trimming so a fat hit list does not re-blow."""
    from llb.bench.agentic.context import summary_hit_count

    state = ContextState()
    policy = ContextPolicy(name=POLICY_COMPACT, observation_cap_chars=100)
    hits = "\n".join(f"[d{i}] body-{i}-" + ("x" * 40) for i in range(6))
    state.record(policy, "search", {"query": "q"}, hits)
    assert state.telemetry.n_trimmed_observations == 1
    lines = policy_history_lines(policy, state)
    assert "hits=6" in lines[0] and "обрізано" in lines[0]
    assert summary_hit_count("") is None


def test_compact_finish_cue_when_summary_already_has_hit_facts():
    """After folding a search, the next prompt must steer the model to finish, not search again."""
    state = ContextState()
    policy = ContextPolicy(name=POLICY_COMPACT, compact_keep_recent=1)
    hits = "\n".join(f"[d{i}] body" for i in range(3))
    state.record(policy, "search", {"query": "q"}, hits)
    assert compact_state(policy, state, lambda _older: "модель забула число") is True
    lines = policy_history_lines(policy, state)
    assert any("підсумок попередніх кроків" in line for line in lines)
    cue = next(line for line in lines if "підказка" in line)
    assert "hits=3" in cue and "finish" in cue


def test_compact_count_episode_finishes_from_live_aggregate_header():
    """With live trimming, compact recovers a count the way observation_cap does: header -> finish."""
    corpus = {f"d{i}": f"тема {i} згадка" for i in range(4)}
    hits_blob = "\n".join(f"[{doc}] {text}" for doc, text in corpus.items())
    # Oversized observation so the trim path fires; the scripted model finishes when it sees hits=.
    fat = hits_blob + (" padding" * 200)
    task = AgenticTask(
        "search-count-000",
        "скільки документів згадують тему",
        setup={"corpus": {"blob": fat}},
        success=[{"kind": "answer_contains", "value": "1"}],
    )

    def complete(prompt: str) -> str:
        if "Стисло підсумуй" in prompt:
            return "знайдено документи"
        if "hits=" in prompt:
            # Prefer the machine header (or finish cue) over another search.
            return '{"name":"finish","arguments":{"answer":"1"}}'
        return '{"name":"search","arguments":{"query":"тема"}}'

    episode = run_episode(
        task,
        complete,
        budget=fixed_budget(12_000),
        policy=ContextPolicy(name=POLICY_COMPACT, observation_cap_chars=200, compact_share=0.5),
        max_steps=6,
    )
    assert episode.status == STATUS_COMPLETED
    assert episode.success is True
    assert episode.n_steps <= 3
    assert episode.telemetry.n_trimmed_observations >= 1


def test_compact_folds_the_whole_transcript_when_there_is_no_older_step():
    state, policy = state_with(1), ContextPolicy(name=POLICY_COMPACT, compact_keep_recent=1)
    assert compact_state(policy, state, lambda older: "стисло") is True
    assert state.entries == [] and state.summary == "стисло"


def test_compact_is_a_no_op_on_an_empty_transcript_or_an_empty_summary():
    policy = ContextPolicy(name=POLICY_COMPACT)
    assert compact_state(policy, ContextState(), lambda _older: "щось") is False
    assert compact_state(policy, state_with(3), lambda _older: "  ") is False


def test_unknown_policy_name_is_refused():
    with pytest.raises(ValueError, match="unknown context policy"):
        ContextPolicy(name="whatever")


# --- the per-step budget guard ------------------------------------------------------------


def test_budget_guard_ends_the_episode_as_context_overflow_without_sending_the_prompt():
    sent: list[str] = []

    def complete(prompt: str) -> str:
        sent.append(prompt)
        return '{"name":"search","arguments":{"query":"дані"}}'

    episode = run_episode(search_task(), complete, budget=fixed_budget(4000))
    assert episode.status == STATUS_CONTEXT_OVERFLOW
    # step 1 fit and ran; step 2 (carrying the 5k-char observation) was refused, never sent
    assert len(sent) == 1 and episode.n_steps == 1
    assert episode.telemetry.max_prompt_chars > 4000  # the refused prompt is still observable


def test_an_unbounded_budget_refuses_nothing():
    budget = unbounded_budget()
    assert budget.bounded is False and budget.fits(10**9) is True
    assert budget.compaction_trigger_chars(0.5) == 0
    episode = run_episode(
        search_task(),
        scripted_cycle(['{"name":"search","arguments":{"query":"д"}}', "готово"]),
        budget=budget,
    )
    assert episode.status == STATUS_COMPLETED and episode.success is True


def test_fixed_budget_reports_its_trigger_share_and_degrades_to_unbounded_at_zero():
    assert fixed_budget(1000).compaction_trigger_chars(0.5) == 500
    assert fixed_budget(0).bounded is False


def test_prompt_tokens_uses_the_shared_chars_per_token_conversion():
    from llb.optimize.tuning_space import CHARS_PER_TOKEN

    assert prompt_tokens(3000) == int(3000 / CHARS_PER_TOKEN)


def test_observation_cap_survives_a_budget_that_overflows_full():
    """The whole point of the lane: the same task, the same budget, a different policy outcome."""
    script = ['{"name":"search","arguments":{"query":"дані"}}', "готово"]
    budget = fixed_budget(4000)
    assert run_episode(search_task(), scripted_cycle(script), budget=budget).status == (
        STATUS_CONTEXT_OVERFLOW
    )
    capped = run_episode(
        search_task(),
        scripted_cycle(script),
        budget=budget,
        policy=ContextPolicy(name=POLICY_OBSERVATION_CAP, observation_cap_chars=200),
    )
    assert capped.status == STATUS_COMPLETED and capped.success is True


def test_the_summarize_call_is_itself_capped_so_it_cannot_overflow():
    """The summarizer's input IS the transcript that blew the step prompt -- it must be trimmed."""
    from llb.bench.agentic.context import summarize_entries

    seen: list[str] = []
    entries = [("search", {"query": "дані"}, BIG)]
    summarize_entries(lambda p: seen.append(p) or "стисло", entries, 500)
    assert len(seen[0]) < len(BIG) and "обрізано" in seen[0]
    assert "[агрегат: chars=" in seen[0]  # per-entry header before the outer trim
    seen.clear()
    summarize_entries(lambda p: seen.append(p) or "стисло", entries, 0)  # 0 = no cap
    assert "обрізано" not in seen[0]
    assert "[агрегат: chars=" in seen[0]


def test_summarize_search_hits_carry_hit_count_in_the_compaction_prompt():
    from llb.bench.agentic.context import summarize_entries

    hits = "\n".join(f"[doc-{i}] text-{i}" for i in range(5))
    seen: list[str] = []
    summarize_entries(lambda p: seen.append(p) or "стисло", [("search", {"query": "q"}, hits)], 0)
    assert "hits=5" in seen[0] and "docs=doc-0,doc-1,doc-2,doc-3,doc-4" in seen[0]


def test_repeated_compaction_carries_the_prior_summary_and_aggregate_facts():
    from llb.bench.agentic.context import summarize_entries

    hits = "\n".join(f"[doc-{i}] text-{i}" for i in range(3))
    seen: list[str] = []
    telemetry = ContextState().telemetry
    summarize_entries(
        lambda prompt: seen.append(prompt) or "оновлений підсумок",
        [("search", {"query": "q"}, hits)],
        2000,
        prior_summary="[агрегат: hits=2 chars=10 docs=old-a,old-b]. старий підсумок",
        telemetry=telemetry,
    )
    assert "попередній підсумок" in seen[0] and "старий підсумок" in seen[0]
    assert telemetry.compaction_prompt_chars == len(seen[0])
    assert telemetry.model_input_prompt_chars == len(seen[0])

    state = ContextState(summary="[агрегат: hits=2 chars=10 docs=old-a,old-b]")
    policy = ContextPolicy(name=POLICY_COMPACT)
    state.record(policy, "search", {"query": "q"}, hits)
    assert compact_state(policy, state, lambda _older: "оновлений підсумок") is True
    assert "docs=old-a,old-b" in state.summary
    assert "docs=doc-0,doc-1,doc-2" in state.summary


def test_compact_preserves_typed_memory_and_cues_finish_after_workflow_completion():
    state = ContextState()
    policy = ContextPolicy(name=POLICY_COMPACT, compact_keep_recent=1)
    state.record(policy, "advance", {"token": "t0"}, "[memory: final_code=MEM-001]")
    state.record(policy, "advance", {"token": "t1"}, "[workflow complete]")
    assert compact_state(policy, state, lambda _older: "етап виконано") is True
    assert "[memory: final_code=MEM-001]" in state.summary
    lines = policy_history_lines(policy, state)
    assert any('finish з answer="MEM-001"' in line for line in lines)


@pytest.mark.parametrize("summary_input_cap", ("window", "trigger"))
def test_a_compacting_episode_never_sends_an_oversized_summarize_call(summary_input_cap: str):
    """Both summarize-input bounds must fit: the wider one still reserves the elision marker."""
    budget, prompts = fixed_budget(6000), []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return (
            "стисло"
            if "Стисло підсумуй" in prompt
            else '{"name":"search","arguments":{"query":"дані"}}'
        )

    run_episode(
        search_task(),
        complete,
        budget=budget,
        # Cap above BIG so live trimming does not prevent the compact trigger this test measures.
        policy=ContextPolicy(
            name=POLICY_COMPACT,
            compact_share=0.5,
            observation_cap_chars=100_000,
            summary_input_cap=summary_input_cap,
        ),
    )
    summarize_calls = [p for p in prompts if "Стисло підсумуй" in p]
    assert summarize_calls and all(budget.fits(len(p)) for p in summarize_calls)


def test_compact_calls_the_model_for_a_summary_once_the_trigger_is_crossed():
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        if "Стисло підсумуй" in prompt:
            return "агент знайшов d1"
        return '{"name":"search","arguments":{"query":"дані"}}' if len(prompts) < 3 else "готово"

    episode = run_episode(
        search_task(),
        complete,
        budget=fixed_budget(8000),
        policy=ContextPolicy(
            name=POLICY_COMPACT,
            compact_share=0.5,
            compact_keep_recent=1,
            observation_cap_chars=100_000,
        ),
    )
    assert episode.telemetry.n_compactions >= 1
    assert episode.status == STATUS_COMPLETED
    assert any("Стисло підсумуй" in p for p in prompts)


def test_compact_uses_full_budget_as_hysteresis_after_first_summary():
    state = ContextState(summary="remembered code")
    policy = ContextPolicy(
        name=POLICY_COMPACT,
        compact_share=0.5,
        observation_cap_chars=100_000,
    )
    state.record(policy, "read_file", {"path": "stage.txt"}, "x" * 2500)
    calls: list[str] = []
    prompt = step_prompt(
        search_task(),
        tool_catalog(),
        policy,
        state,
        fixed_budget(8000),
        lambda text: calls.append(text) or "updated",
    )
    assert 4000 < len(prompt) < 8000
    assert calls == []
    assert state.telemetry.n_compactions == 0


# --- telemetry ----------------------------------------------------------------------------


def test_case_row_carries_the_context_columns():
    from llb.bench.agentic.episode import _row

    task = search_task()
    episode = run_episode(
        task,
        scripted_cycle(['{"name":"search","arguments":{"query":"д"}}', "готово"]),
        policy=ContextPolicy(name=POLICY_OBSERVATION_CAP, observation_cap_chars=100),
    )
    row = _row(task, episode)
    assert row["max_prompt_tokens"] > 0 and row["total_prompt_tokens"] >= row["max_prompt_tokens"]
    assert row["total_model_input_tokens"] > 0
    assert row["n_model_calls"] == episode.n_steps + episode.telemetry.n_compactions
    assert row["observation_bytes"] > 0
    assert row["n_trimmed_observations"] == 1 and row["n_compactions"] == 0
    json.dumps(row)  # the row stays persistable


def test_a_baseline_episode_reproduces_the_recorded_row_shape():
    """The policy seam adds NOTHING to the baseline path beyond the additive telemetry columns."""
    from llb.bench.agentic.episode import _row

    task = AgenticTask("a", "p", success=[{"kind": "answer_contains", "value": "x"}])
    row = _row(task, run_episode(task, lambda _p: "x"))
    assert row["status"] == STATUS_COMPLETED and row["n_steps"] == 1 and row["success"] == 1.0
    assert row["n_compactions"] == 0 and row["n_trimmed_observations"] == 0


# --- the comparison lane ------------------------------------------------------------------


def comparison_tasks(n: int = 8) -> list[AgenticTask]:
    return [search_task(f"t{i}") for i in range(n)]


def comparison_complete():
    return scripted_cycle(
        [
            '{"name":"search","arguments":{"query":"дані"}}',
            '{"name":"db_get","arguments":{"key":"nope"}}',
            "готово",
        ]
    )


def test_run_agentic_context_runs_every_policy_on_the_identical_task_set():
    run = run_agentic_context(
        comparison_tasks(),
        model="m",
        backend="ollama",
        complete=comparison_complete(),
        budget=fixed_budget(6000),
        persist=False,
    )
    assert [r.policy for r in run.reports] == list(CONTEXT_POLICIES)
    assert all(len(r.case_success) == 8 for r in run.reports)
    assert run.max_prompt_chars == 6000
    assert task_set_digest(comparison_tasks()) == run.task_set_digest


def test_kind_table_splits_count_vs_locate_and_scores_pre_header_delta():
    from llb.bench.agentic_context_report import (
        aggregate_safe_verdict,
        format_kind_table,
        kind_completion,
    )

    # Mix generator ids so the kind split is exercised; scripted complete finishes every task.
    tasks = [
        AgenticTask(
            "search-count-000",
            "скільки?",
            setup={"corpus": {"d1": "foo"}},
            success=[{"kind": "answer_contains", "value": "готово"}],
        ),
        AgenticTask(
            "search-count-001",
            "скільки?",
            setup={"corpus": {"d1": "foo"}},
            success=[{"kind": "answer_contains", "value": "готово"}],
        ),
        AgenticTask(
            "search-locate-000",
            "де?",
            setup={"corpus": {"d1": "foo"}},
            success=[{"kind": "answer_contains", "value": "готово"}],
        ),
        AgenticTask(
            "seed-other",
            "інше",
            setup={"corpus": {"d1": "foo"}},
            success=[{"kind": "answer_contains", "value": "готово"}],
        ),
    ]
    run = run_agentic_context(
        tasks,
        model="m",
        backend="ollama",
        complete=comparison_complete(),
        policies=["full", "observation_cap", "compact"],
        budget=fixed_budget(8000),
        persist=False,
    )
    assert "by task kind:" in run.kind_table
    assert "count" in run.kind_table and "locate" in run.kind_table
    assert "vs-pre-header" in run.kind_table
    assert run.aggregate_safe_verdict
    # Scripted complete answers "готово" so count completion is 1.0 vs pre-header 0.0 -> recovered.
    cap = next(r for r in run.reports if r.policy == "observation_cap")
    assert kind_completion(cap, "count") == 1.0
    assert "recovered" in aggregate_safe_verdict(run.reports)
    assert format_kind_table(run.reports).startswith("by task kind:")


def test_every_non_baseline_policy_carries_a_paired_delta_against_full():
    run = run_agentic_context(
        comparison_tasks(),
        model="m",
        backend="ollama",
        complete=comparison_complete(),
        budget=fixed_budget(6000),
        persist=False,
    )
    baseline = next(r for r in run.reports if r.policy == POLICY_FULL)
    assert baseline.paired == {}  # the baseline is not paired against itself
    for report in run.reports:
        if report.policy == POLICY_FULL:
            continue
        assert set(report.paired) == {
            METRIC_COMPLETION,
            "n_steps",
            "n_tool_calls",
            METRIC_PROMPT_TOKENS,
        }
        delta = report.paired[METRIC_COMPLETION]["delta"]
        assert delta["lo"] <= delta["mean"] <= delta["hi"]
        assert completion_reading(report) in (
            READING_SEPARATED,
            READING_FLAT,
            "insufficient_evidence",
        )


def test_no_episode_in_any_policy_sends_a_prompt_over_the_resolved_window():
    budget = fixed_budget(6000)
    run = run_agentic_context(
        comparison_tasks(),
        model="m",
        backend="ollama",
        complete=comparison_complete(),
        budget=budget,
        persist=False,
    )
    for report in run.reports:
        for episode in report.episodes:
            # every prompt but a refused final one was sent, and a refused one ends the episode
            sent = episode.telemetry.prompt_chars[
                : -1 if episode.status == STATUS_CONTEXT_OVERFLOW else None
            ]
            assert all(budget.fits(chars) for chars in sent)


def test_an_identical_policy_pair_reads_flat():
    """Two runs of the same behavior must not manufacture a separation out of resample noise."""
    run = run_agentic_context(
        comparison_tasks(),
        model="m",
        backend="ollama",
        complete=comparison_complete(),
        policies=[POLICY_FULL, POLICY_KEEP_LAST_N],
        policy_overrides={"keep_last_n": 99},  # keeps everything -> identical to `full`
        budget=fixed_budget(6000),
        persist=False,
    )
    twin = next(r for r in run.reports if r.policy == POLICY_KEEP_LAST_N)
    assert completion_reading(twin) == READING_FLAT
    assert twin.paired[METRIC_COMPLETION]["delta"] == {"mean": 0.0, "lo": 0.0, "hi": 0.0}


def test_the_recommendation_states_a_flat_reading_rather_than_ranking_noise():
    run = run_agentic_context(
        comparison_tasks(),
        model="m",
        backend="ollama",
        complete=comparison_complete(),
        policies=[POLICY_FULL, POLICY_KEEP_LAST_N],
        policy_overrides={"keep_last_n": 99},
        budget=fixed_budget(6000),
        persist=False,
    )
    assert "не відокремилася" in run.recommendation
    assert "full" in run.recommendation


def test_an_empty_task_set_is_refused():
    with pytest.raises(SystemExit, match="no agentic tasks"):
        run_agentic_context([], model="m", backend="ollama", complete=lambda _p: "x")


def test_an_unknown_policy_is_refused_before_any_model_call():
    with pytest.raises(SystemExit, match="unknown context policies"):
        run_agentic_context(
            comparison_tasks(2),
            model="m",
            backend="ollama",
            complete=lambda _p: pytest.fail("no model call may happen"),
            policies=["full", "nonsense"],
        )


# --- persistence + the board ---------------------------------------------------------------


def test_each_policy_persists_its_own_tagged_bundle_that_the_board_reloads(tmp_path: Path):
    run = run_agentic_context(
        comparison_tasks(),
        model="m",
        backend="ollama",
        complete=comparison_complete(),
        budget=fixed_budget(6000),
        data_dir=tmp_path,
        mirror=lambda *_: None,
    )
    for report in run.reports:
        assert report.paths is not None
        manifest = json.loads(Path(report.paths["manifest"]).read_text(encoding="utf-8"))
        config = manifest["config"]
        assert config["policy"] == report.policy and config["category"] == METHOD
        assert config["task_set_digest"] == run.task_set_digest
        assert config["max_prompt_chars"] == 6000
        assert "n_context_overflow" in config and "mean_max_prompt_tokens" in config
        assert (config["paired_vs_full"] is None) == (report.policy == POLICY_FULL)

    rows, table, policies = agentic_context_comparison(tmp_path, "m")
    assert sorted(policies) == sorted(CONTEXT_POLICIES)
    assert len(rows) == len(CONTEXT_POLICIES) and "full" in table


def test_the_board_ignores_a_model_with_no_context_policy_runs(tmp_path: Path):
    assert agentic_context_comparison(tmp_path, "absent") == ([], "", [])


# --- the guard resolves from a run config --------------------------------------------------


def test_resolve_context_budget_bounds_the_prompt_from_an_explicit_context_budget():
    from llb.bench.agentic.context_budget import resolve_context_budget
    from llb.core.config import RunConfig
    from llb.optimize.tuning_space import CHARS_PER_TOKEN, PROMPT_HEADROOM_TOKENS

    config = RunConfig().with_overrides(model="unlisted-model", context_budget=4096)
    budget = resolve_context_budget(config, model_spec=None, vram_mib=0, ram_mib=0)
    assert budget.bounded is True
    usable = 4096 - PROMPT_HEADROOM_TOKENS - config.max_tokens
    assert budget.max_prompt_chars == int(usable * CHARS_PER_TOKEN)
    # the reported budget and the predicate that refuses a prompt agree at the boundary
    assert budget.fits(budget.max_prompt_chars) is True
    assert budget.fits(budget.max_prompt_chars + int(2 * CHARS_PER_TOKEN)) is False


def test_resolve_context_budget_cannot_bound_an_unknown_model():
    from llb.bench.agentic.context_budget import resolve_context_budget
    from llb.core.config import RunConfig

    budget = resolve_context_budget(
        RunConfig().with_overrides(model="unlisted-model"), model_spec=None, vram_mib=0, ram_mib=0
    )
    assert budget.bounded is False and budget.fits(10**9) is True


def test_a_budget_is_a_plain_predicate_seam():
    budget = ContextBudget(max_prompt_chars=10, fits=lambda chars: chars < 5)
    assert budget.bounded is True and budget.fits(4) and not budget.fits(6)


def test_incomplete_status_still_reports_when_the_step_budget_runs_out():
    episode = run_episode(
        AgenticTask("t", "p", success=[{"kind": "answer_contains", "value": "x"}]),
        lambda _p: '{"name":"db_get","arguments":{"key":"k"}}',
        max_steps=3,
    )
    assert episode.status == STATUS_INCOMPLETE and episode.n_steps == 3
    assert len(episode.telemetry.prompt_chars) == 3
