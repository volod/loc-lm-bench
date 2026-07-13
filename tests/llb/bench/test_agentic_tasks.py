"""agentic benchmark real-UA-corpus agentic search-task generation (count + locate), deterministic."""

from llb.bench import agentic_tasks as at
from llb.bench.agentic.episode import run_episode
from llb.bench.agentic.model import AgenticTask

CORPUS = {
    "energy": "Звіт про відновлювану енергетику України. Сонячні станції зросли.",
    "budget": "Річний бюджет міста зріс. Інвестиції у відновлювану енергетику також.",
    "transport": "Громадський транспорт міста модернізовано новими тролейбусами.",
}


def test_search_count_task_answer_is_doc_frequency():
    task = at.search_count_task(CORPUS, "відновлювану", 0)
    # appears in energy + budget -> 2
    assert task["success"][0]["value"] == "2"
    assert task["success"][0]["kind"] == at.ASSERT_ANSWER_CONTAINS


def test_search_locate_only_for_unique_terms():
    # "тролейбусами" appears in exactly one doc -> locate task
    located = at.search_locate_task(CORPUS, "тролейбусами", 0)
    assert located is not None and located["success"][0]["value"] == "transport"
    # an ambiguous term (2 docs) -> no locate task
    assert at.search_locate_task(CORPUS, "відновлювану", 0) is None


def test_derive_queries_splits_count_and_locate():
    count_terms, locate_terms = at.derive_queries(CORPUS, top_k=20)
    assert count_terms  # frequent content terms
    # every locate term appears in exactly one doc
    for term in locate_terms:
        assert len(at._docs_containing(CORPUS, term)) == 1


def test_build_search_tasks_round_trip_into_agentic_tasks():
    raw = at.build_search_tasks(CORPUS, top_k=10)
    assert raw
    tasks = [AgenticTask.from_record(r) for r in raw]
    assert all(t.setup.get("corpus") for t in tasks)  # the real corpus travels in the task


def test_generated_count_task_is_solvable_by_a_good_agent():
    raw = at.search_count_task(CORPUS, "відновлювану", 0)
    task = AgenticTask.from_record(raw)
    # a good agent: search, then finish reporting the count (2)
    scripted = iter(
        [
            '{"name":"search","arguments":{"query":"відновлювану"}}',
            '{"name":"finish","arguments":{"answer":"Знайдено 2 документи."}}',
        ]
    )
    episode = run_episode(task, lambda _p: next(scripted), max_steps=4)
    assert episode.success is True


def test_build_search_tasks_with_explicit_queries():
    tasks = at.build_search_tasks(CORPUS, queries=["бюджет"], kinds=(at.KIND_COUNT,))
    assert len(tasks) == 1 and tasks[0]["success"][0]["value"] == "1"


def test_build_from_corpus(tmp_path):
    (tmp_path / "a.md").write_text(CORPUS["energy"], encoding="utf-8")
    (tmp_path / "b.md").write_text(CORPUS["transport"], encoding="utf-8")
    tasks = at.build_from_corpus(tmp_path, top_k=5)
    assert tasks and all("search" in t["id"] for t in tasks)
