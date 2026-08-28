"""Composed agent operating profile: the four field states, the consistency guard, the staleness
demotion, and the replay commands."""

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llb.board.agent_profile.compose import build_agent_profile
from llb.board.agent_profile.model import (
    FIELD_ADAPTER,
    FIELD_BACKEND,
    FIELD_CONTEXT_BUDGET,
    FIELD_CONTEXT_ORDER,
    FIELD_CONTEXT_POLICY,
    FIELD_LOOP_POLICY,
    FIELD_MODEL,
    FIELD_PROMPT_SYSTEM,
    FIELD_RERANKER,
    FIELD_TOP_K,
    PROFILE_FIELD_SPECS,
    STATE_DEMOTED,
    STATE_MEASURED,
    STATE_REFUSED,
    STATE_UNMEASURED,
)
from llb.board.agent_profile.persist import write_profile
from llb.board.agent_profile.render import format_profile_md, profile_payload
from llb.board.agent_profile.replay import replay_commands
from llb.finetune.registry.io import registry_path
from llb.finetune.registry.model import EVENT_REGISTER

from tests.llb.board.agent_profile.conftest import (
    CORPUS,
    MODEL,
    STORE,
    write_context_policy,
    write_loop_policy,
    write_probe,
    write_reranker,
)

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)


def _profile(tmp_path, rec=None, index_dir=None):
    return build_agent_profile(tmp_path, rec, now=NOW, index_dir=index_dir)


def _register_adapter(tmp_path: Path, *, base_model: str = MODEL, fingerprint=None) -> str:
    """Append one register event so the registry resolves an adapter for `base_model`."""
    adapter_id = "db80e8440b7d0000"
    path = registry_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event": EVENT_REGISTER,
        "adapter_id": adapter_id,
        "base_model": base_model,
        "adapter_label": "ua-sft",
        "adapter_dir": str(tmp_path / "adapters" / adapter_id),
        "dataset_digest": "d" * 64,
        "goldset_digest": "g" * 64,
        "corpus_digest": "c" * 64,
        "goldset_path": str(tmp_path / "goldset.jsonl"),
        "corpus_root": str(tmp_path / "corpus"),
        "retrieval_fingerprint": fingerprint if fingerprint is not None else dict(STORE),
        "index_dir": str(tmp_path / "llb" / "rag"),
        "created_at": "2026-08-01T00:00:00Z",
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")
    return adapter_id


# --- the empty profile -------------------------------------------------------------------------


def test_profile_without_any_bundle_is_entirely_unmeasured_and_still_valid(tmp_path):
    profile = _profile(tmp_path)

    assert [item.name for item in profile.fields] == [s.name for s in PROFILE_FIELD_SPECS]
    assert {item.state for item in profile.fields} == {STATE_UNMEASURED}
    payload = profile_payload(profile)
    assert payload["states"]["unmeasured"] == len(PROFILE_FIELD_SPECS)
    assert payload["anchor"]["resolved"] is False
    # Every field still carries its lane, so the artifact says WHICH run would close each gap.
    assert all(entry["lane"] for entry in payload["fields"].values())
    assert "unmeasured" in format_profile_md(profile)


def test_an_unmeasured_field_never_carries_a_value(tmp_path, recommendation, index_dir):
    profile = _profile(tmp_path, recommendation, index_dir)

    for item in profile.fields:
        assert (item.value is None) == (item.state == STATE_UNMEASURED)


# --- populated fields --------------------------------------------------------------------------


def test_every_populated_field_resolves_its_evidence_and_keeps_its_lane_verdict(
    tmp_path, recommendation, index_dir
):
    write_reranker(tmp_path)
    write_probe(tmp_path)
    write_loop_policy(tmp_path)
    write_context_policy(tmp_path)

    profile = _profile(tmp_path, recommendation, index_dir)

    populated = [item for item in profile.fields if item.state != STATE_UNMEASURED]
    assert {item.name for item in populated} >= {
        FIELD_MODEL,
        FIELD_BACKEND,
        FIELD_TOP_K,
        FIELD_CONTEXT_BUDGET,
        FIELD_RERANKER,
        FIELD_CONTEXT_ORDER,
        FIELD_CONTEXT_POLICY,
        FIELD_LOOP_POLICY,
    }
    for item in populated:
        assert item.evidence_path and Path(item.evidence_path).exists(), item.name
        assert item.verdict, item.name
        assert item.measured_at, item.name

    # Each verdict is the lane artifact's own, not a re-derivation.
    assert profile.by_name(FIELD_RERANKER).verdict == "retain"
    assert profile.by_name(FIELD_LOOP_POLICY).verdict == "flat"
    assert profile.by_name(FIELD_CONTEXT_POLICY).verdict == "separated"
    assert profile.by_name(FIELD_TOP_K).value == 3
    assert profile.by_name(FIELD_CONTEXT_ORDER).value == "rank"


def test_freshness_is_reported_per_field_in_days(tmp_path, recommendation, index_dir):
    write_context_policy(tmp_path)
    profile = _profile(tmp_path, recommendation, index_dir)

    freshness = profile_payload(profile)["fields"][FIELD_CONTEXT_POLICY]["freshness"]
    assert freshness["measured_at"] == "2026-08-20T00:00:00+00:00"
    assert freshness["age_days"] == pytest.approx(8.5, abs=0.01)


def test_an_empty_registry_reads_as_a_measured_no_adapter(tmp_path, recommendation, index_dir):
    registry_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    registry_path(tmp_path).write_text("", encoding="utf-8")

    adapter = _profile(tmp_path, recommendation, index_dir).by_name(FIELD_ADAPTER)

    assert (adapter.state, adapter.value) == (STATE_MEASURED, "none")


# --- the consistency guard ---------------------------------------------------------------------


def test_a_field_measured_on_another_model_is_refused_and_names_the_axis(
    tmp_path, recommendation, index_dir
):
    write_loop_policy(tmp_path, model="some-other-model")

    item = _profile(tmp_path, recommendation, index_dir).by_name(FIELD_LOOP_POLICY)

    assert item.state == STATE_REFUSED
    assert "some-other-model" in " ".join(item.notes)
    assert MODEL in " ".join(item.notes)


def test_a_field_measured_on_another_corpus_is_refused(tmp_path, recommendation, index_dir):
    write_reranker(tmp_path, corpus="/corpora/somewhere-else")

    item = _profile(tmp_path, recommendation, index_dir).by_name(FIELD_RERANKER)

    assert item.state == STATE_REFUSED
    assert "/corpora/somewhere-else" in " ".join(item.notes)
    assert CORPUS in " ".join(item.notes)


def test_a_probe_taken_against_a_re_chunked_store_is_refused(tmp_path, recommendation, index_dir):
    """A context_order recommendation is only about the store whose distractors produced it."""
    write_probe(tmp_path, fingerprint={**STORE, "chunk_size": 512})

    item = _profile(tmp_path, recommendation, index_dir).by_name(FIELD_CONTEXT_ORDER)

    assert item.state == STATE_REFUSED
    assert "chunk_size" in " ".join(item.notes)
    assert "512" in " ".join(item.notes) and "800" in " ".join(item.notes)


def test_a_matching_lane_is_not_refused(tmp_path, recommendation, index_dir):
    write_reranker(tmp_path)
    write_loop_policy(tmp_path)

    profile = _profile(tmp_path, recommendation, index_dir)

    assert profile.by_name(FIELD_RERANKER).state == STATE_MEASURED
    assert profile.by_name(FIELD_LOOP_POLICY).state == STATE_MEASURED


def test_without_an_anchor_nothing_is_refused_but_the_gap_is_stated(tmp_path, index_dir):
    write_loop_policy(tmp_path, model="some-other-model")

    item = _profile(tmp_path, None, index_dir).by_name(FIELD_LOOP_POLICY)

    assert item.state == STATE_MEASURED
    assert any("consistency unchecked" in note for note in item.notes)


# --- the staleness demotion --------------------------------------------------------------------


def test_a_changed_store_fingerprint_demotes_every_store_dependent_field_and_names_the_knob(
    tmp_path, recommendation
):
    write_reranker(tmp_path)
    write_probe(tmp_path)
    write_loop_policy(tmp_path)
    write_context_policy(tmp_path)
    moved = tmp_path / "moved-store"
    moved.mkdir()
    (moved / "store_meta.json").write_text(
        json.dumps(
            {
                "strategy": "semantic",
                "size": 704,
                "overlap": 120,
                "mode": "flat",
                "embedding_model": STORE["embedding_model"],
            }
        ),
        encoding="utf-8",
    )

    profile = build_agent_profile(tmp_path, recommendation, now=NOW, index_dir=moved)

    assert profile.by_name(FIELD_MODEL).state == STATE_DEMOTED
    assert profile.by_name(FIELD_TOP_K).state == STATE_DEMOTED
    assert profile.by_name(FIELD_RERANKER).state == STATE_DEMOTED
    assert profile.by_name(FIELD_CONTEXT_ORDER).state == STATE_DEMOTED
    # Not store-dependent: the loop policy and the prompt budget survive a re-chunked store.
    assert profile.by_name(FIELD_LOOP_POLICY).state == STATE_MEASURED
    assert profile.by_name(FIELD_CONTEXT_BUDGET).state == STATE_MEASURED
    named = " ".join(profile.by_name(FIELD_TOP_K).notes)
    assert "strategy" in named and "recursive" in named and "semantic" in named
    assert "chunk_size" in named and "800" in named and "704" in named
    assert {c["knob"] for c in profile.store_drift} == {"strategy", "chunk_size"}


def test_an_unmeasured_field_is_not_demoted_by_store_drift(tmp_path, recommendation):
    moved = tmp_path / "moved-store"
    moved.mkdir()
    (moved / "store_meta.json").write_text(json.dumps({"strategy": "semantic"}), encoding="utf-8")

    item = build_agent_profile(tmp_path, recommendation, now=NOW, index_dir=moved).by_name(
        FIELD_CONTEXT_ORDER
    )

    assert item.state == STATE_UNMEASURED
    assert not any("fingerprint changed" in note for note in item.notes)


def test_a_stale_adapter_demotes_every_adapter_dependent_field(tmp_path, recommendation, index_dir):
    write_loop_policy(tmp_path)
    write_context_policy(tmp_path)
    write_reranker(tmp_path)
    # A goldset/corpus the registry recorded but that no longer exists reads as `unknown`, so the
    # stale verdict has to come from a fingerprint axis that actually moved.
    _register_adapter(tmp_path, fingerprint={**STORE, "chunk_size": 512})

    profile = _profile(tmp_path, recommendation, index_dir)

    adapter = profile.by_name(FIELD_ADAPTER)
    assert adapter.verdict == "stale"
    assert adapter.state == STATE_DEMOTED
    assert profile.by_name(FIELD_MODEL).state == STATE_DEMOTED
    assert profile.by_name(FIELD_LOOP_POLICY).state == STATE_DEMOTED
    assert profile.by_name(FIELD_CONTEXT_POLICY).state == STATE_DEMOTED
    # The reranker does not depend on the adapter, so it keeps standing.
    assert profile.by_name(FIELD_RERANKER).state == STATE_MEASURED
    assert any("chunk_size" in note for note in profile.by_name(FIELD_MODEL).notes)


# --- replay ------------------------------------------------------------------------------------


def test_measured_values_replay_as_flags_that_reproduce_the_configuration(
    tmp_path, recommendation, index_dir
):
    write_reranker(tmp_path)
    write_probe(tmp_path)
    write_loop_policy(tmp_path)
    write_context_policy(tmp_path)

    profile = _profile(tmp_path, recommendation, index_dir)
    commands = replay_commands(profile)

    run_eval = next(c for c in commands if c.startswith("llb run-eval"))
    assert f"--model {MODEL}" in run_eval
    assert "--backend ollama" in run_eval
    assert "--top-k 3" in run_eval
    assert "--context-budget 8192" in run_eval
    assert "--reranker BAAI/bge-reranker-v2-m3" in run_eval
    assert "--context-order rank" in run_eval
    bench = next(c for c in commands if c.startswith("llb bench-agentic "))
    assert "--context-policy observation_cap" in bench and "--max-steps 6" in bench
    assert "--malformed-policy answer" in bench and "--repeated-call-policy allow" in bench
    loop = next(c for c in commands if c.startswith("llb bench-agentic-loop"))
    assert "--agent-max-steps 6" in loop and "--agent-repeated-call-policy allow" in loop


def test_a_recommended_loop_cell_replays_as_a_scored_run(tmp_path, recommendation, index_dir):
    """The whole recommended cell reaches `bench-agentic`, not only the step budget."""
    write_loop_policy(
        tmp_path,
        max_steps=10,
        malformed_call_policy="repair_once",
        repeated_call_policy="noop",
        repeat_feedback_variant="uk",
    )

    profile = _profile(tmp_path, recommendation, index_dir)
    bench = next(c for c in replay_commands(profile) if c.startswith("llb bench-agentic "))

    assert "--max-steps 10" in bench
    assert "--malformed-policy repair_once" in bench
    assert "--repeated-call-policy noop" in bench
    assert "--repeat-feedback uk" in bench


def test_the_repeat_wording_is_pinned_only_where_it_reaches_a_prompt(
    tmp_path, recommendation, index_dir
):
    """Under `allow` no repeat is ever suppressed, so pinning its wording would state nothing."""
    write_loop_policy(tmp_path, repeated_call_policy="allow", repeat_feedback_variant="uk")

    profile = _profile(tmp_path, recommendation, index_dir)
    bench = next(c for c in replay_commands(profile) if c.startswith("llb bench-agentic "))

    assert "--repeat-feedback" not in bench


def test_bench_agentic_replay_flags_are_real_command_options(tmp_path, recommendation, index_dir):
    """The flags are not decoration: `bench-agentic` accepts every one of them."""
    from llb.board.agent_profile.replay import bench_agentic_flags
    from llb.cli.app import app

    write_loop_policy(tmp_path, repeated_call_policy="noop")
    write_context_policy(tmp_path)
    profile = _profile(tmp_path, recommendation, index_dir)

    command = next(c for c in app.registered_commands if c.name == "bench-agentic")
    accepted = {
        f"--{name.replace('_', '-')}" for name in inspect.signature(command.callback).parameters
    }

    emitted = {flag for flag in bench_agentic_flags(profile) if flag.startswith("--")}
    assert emitted and emitted <= accepted


def test_replay_flags_parse_back_into_the_recommended_run_config(
    tmp_path, recommendation, index_dir
):
    """The run-eval flags are not decoration: they map onto real configuration fields."""
    from llb.cli.eval.run_config import _CONFIG_OPTIONS, config_overrides
    from llb.core.config import RunConfig

    write_reranker(tmp_path)
    write_probe(tmp_path)
    profile = _profile(tmp_path, recommendation, index_dir)
    flags = replay_commands(profile)[0].removeprefix("llb run-eval ").split(" ")
    parsed = dict(zip(flags[::2], flags[1::2]))
    named = {flag.removeprefix("--").replace("-", "_"): value for flag, value in parsed.items()}
    named["top_k"] = int(named["top_k"])
    named["context_budget"] = int(named["context_budget"])
    # `config_overrides` reads the command's whole `locals()`, so every option it knows must exist.
    options = {option: None for option in (*_CONFIG_OPTIONS, "query_prep")} | named

    config = RunConfig().with_overrides(**config_overrides(options))

    assert config.model == MODEL
    assert config.backend == "ollama"
    assert config.top_k == 3
    assert config.context_budget == 8192
    assert config.context_order == "rank"
    assert config.reranker == "BAAI/bge-reranker-v2-m3"


def test_a_demoted_model_suppresses_every_replay_command(tmp_path, recommendation):
    write_loop_policy(tmp_path)
    moved = tmp_path / "moved-store"
    moved.mkdir()
    (moved / "store_meta.json").write_text(json.dumps({"strategy": "semantic"}), encoding="utf-8")

    profile = build_agent_profile(tmp_path, recommendation, now=NOW, index_dir=moved)

    assert replay_commands(profile) == []
    assert profile_payload(profile)["replay"]["run_eval"] == []


def test_a_measured_none_contributes_no_flag(tmp_path, recommendation, index_dir):
    registry_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    registry_path(tmp_path).write_text("", encoding="utf-8")

    profile = _profile(tmp_path, recommendation, index_dir)
    run_eval = replay_commands(profile)[0]

    assert profile.by_name(FIELD_ADAPTER).value == "none"
    assert "--adapter" not in run_eval


def test_omitted_fields_are_listed_with_their_state(tmp_path, recommendation, index_dir):
    profile = _profile(tmp_path, recommendation, index_dir)

    omitted = profile_payload(profile)["replay"]["omitted"]

    assert {entry["field"] for entry in omitted} >= {FIELD_PROMPT_SYSTEM, FIELD_CONTEXT_ORDER}
    assert all(entry["state"] != STATE_MEASURED for entry in omitted)


# --- the artifact ------------------------------------------------------------------------------


def test_the_bundle_holds_the_json_and_the_rationale(tmp_path, recommendation, index_dir):
    write_context_policy(tmp_path)
    profile = _profile(tmp_path, recommendation, index_dir)

    paths = write_profile(profile, tmp_path)

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert set(payload["fields"]) == {spec.name for spec in PROFILE_FIELD_SPECS}
    assert paths["json"].parent.parent.name == "agent-profile"
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "# Agent operating profile" in markdown
    assert MODEL in markdown
