"""CI coverage for aggregate-safe observation facts (no model / GPU)."""

from llb.bench.agentic.context_aggregate import (
    extract_aggregate_facts,
    format_aggregate_header,
    task_kind,
    with_aggregate_header,
)
from llb.bench.tool_world import OBS_NO_RESULTS


def test_extract_facts_from_search_hit_list():
    observation = "[doc-a] alpha\n[doc-b] beta\n[doc-c] gamma"
    facts = extract_aggregate_facts(observation)
    assert facts["hits"] == 3
    assert facts["doc_ids"] == ["doc-a", "doc-b", "doc-c"]
    assert facts["chars"] == len(observation)
    assert facts["is_search_hits"] is True


def test_extract_facts_from_no_results_marker():
    facts = extract_aggregate_facts(OBS_NO_RESULTS)
    assert facts["hits"] == 0 and facts["doc_ids"] == [] and facts["is_search_hits"] is True
    assert "hits=0" in format_aggregate_header(OBS_NO_RESULTS)
    assert "docs=-" in format_aggregate_header(OBS_NO_RESULTS)


def test_non_search_observation_gets_chars_only_header():
    header = format_aggregate_header("просто значення")
    assert header == "[агрегат: chars=15]"
    assert extract_aggregate_facts("просто значення")["is_search_hits"] is False


def test_with_aggregate_header_is_idempotent():
    obs = "[d1] text"
    once = with_aggregate_header(obs)
    assert once.startswith("[агрегат: hits=1")
    assert with_aggregate_header(once) == once


def test_task_kind_from_generator_ids():
    assert task_kind("search-count-003") == "count"
    assert task_kind("search-locate-001") == "locate"
    assert task_kind("seed-db-get") == "other"
