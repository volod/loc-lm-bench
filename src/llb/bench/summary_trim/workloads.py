"""The workload families the trim strategy is compared across, and the oracle that walks each one.

A trim strategy is only readable where a fold ELIDES, and whether a fold elides is a property of
the workload's transcript shape rather than of the strategy. So the adoption study is organized by
WORKLOAD: each one names a task builder, the geometry that makes it fold, and the oracle that walks
it with no model -- which is what lets every cell predeclare its fold count, offered transcript, and
elision before a GPU is warmed.

The four declared workloads are the ones a compact session actually meets, plus the middle-critical
stratum set the entry-aware recovery was first measured on (`agentic_memory_window_elision_tasks`),
carried here so the recovery is RE-MEASURED under the promoted policy field rather than cited.
"""

import json
from collections.abc import Callable
from typing import Any, cast

from llb.bench.agentic.context_summary import is_summary_prompt
from llb.bench.agentic.model import AgenticTask
from llb.bench.memory.boundary.probe import ORACLE_SUMMARY, oracle_controller
from llb.bench.memory.transcript import build_memory_dependent_tasks
from llb.bench.memory.window_elision.tasks import STRATA, build_window_elision_stratum_tasks
from llb.bench.summary_trim.tasks import build_aggregate_search_tasks
from llb.bench.tool_world import FINISH, SEARCH

# What a workload's `task_builder` may name.
BUILDER_MEMORY_CHAIN = "memory_chain"
BUILDER_AGGREGATE_SEARCH = "aggregate_search"
BUILDER_EVIDENCE_STRATA = "evidence_strata"
TASK_BUILDERS: tuple[str, ...] = (
    BUILDER_MEMORY_CHAIN,
    BUILDER_AGGREGATE_SEARCH,
    BUILDER_EVIDENCE_STRATA,
)

# The oracle answers the summarize call with a FIXED summary, exactly as the policy-change replay
# does: a probe that let the summary vary would measure the summarizer, not the geometry.
Oracle = Callable[[str], str]


def build_workload_tasks(workload: dict[str, object]) -> list[dict[str, Any]]:
    """The task records one workload declares, as its builder produces them.

    `n_tasks` is the builder's own count, which is PER STRATUM for `evidence_strata` (three strata
    must stay equal-sized for the comparison to be balanced) and a total for the other two.
    """
    builder = str(workload["task_builder"])
    shape = cast(dict[str, object], workload["task_shape"])
    if builder == BUILDER_MEMORY_CHAIN:
        return build_memory_dependent_tasks(
            n_tasks=int(cast(int, workload["n_tasks"])),
            depth=int(cast(int, shape["depth"])),
            pad_chars=int(cast(int, shape["pad_chars"])),
        )
    if builder == BUILDER_AGGREGATE_SEARCH:
        return build_aggregate_search_tasks(
            n_tasks=int(cast(int, workload["n_tasks"])),
            n_queries=int(cast(int, shape["n_queries"])),
            n_docs=int(cast(int, shape["n_docs"])),
            doc_chars=int(cast(int, shape["doc_chars"])),
        )
    if builder == BUILDER_EVIDENCE_STRATA:
        stages = {
            key: int(cast(int, value))
            for key, value in cast(dict[str, object], shape["fact_stages"]).items()
        }
        if set(stages) != set(STRATA):
            raise ValueError(f"evidence-strata fact stages must name exactly {STRATA!r}")
        return build_window_elision_stratum_tasks(
            n_tasks_per_stratum=int(cast(int, workload["n_tasks"])),
            fact_stages=stages,
            depth=int(cast(int, shape["depth"])),
            pad_chars=int(cast(int, shape["pad_chars"])),
        )
    raise ValueError(f"unknown workload task builder {builder!r}; choose from {TASK_BUILDERS}")


def workload_tasks(workload: dict[str, object]) -> list[AgenticTask]:
    """Typed tasks the agent loop consumes, in the builder's own deterministic order."""
    return [AgenticTask.from_record(record) for record in build_workload_tasks(workload)]


def workload_oracle(workload: dict[str, object], record: dict[str, Any]) -> Oracle:
    """The controller that walks ONE task of this workload perfectly, with no model."""
    builder = str(workload["task_builder"])
    if builder == BUILDER_AGGREGATE_SEARCH:
        return _aggregate_search_oracle(record)
    if builder == BUILDER_EVIDENCE_STRATA:
        return _evidence_strata_oracle(record)
    # The memory chain plants its answer as a typed `[memory: ...]` marker, which the shared
    # workflow oracle already reads back; answering the summarize call is what makes a compact
    # walk deterministic.
    return _memory_oracle


def workload_case_metadata(workload: dict[str, object]) -> dict[str, dict[str, object]]:
    """Per-case fields a reading needs beside the outcome -- today, the evidence stratum."""
    if str(workload["task_builder"]) != BUILDER_EVIDENCE_STRATA:
        return {}
    return {
        str(record["id"]): {
            "evidence_stratum": record["elision_stratum"],
            "fact_stage": record["fact_stage"],
        }
        for record in build_workload_tasks(workload)
    }


def _memory_oracle(prompt: str) -> str:
    return ORACLE_SUMMARY if is_summary_prompt(prompt) else oracle_controller(prompt)


def _evidence_strata_oracle(record: dict[str, Any]) -> Oracle:
    """Walk the token chain, then finish with the stratum task's OWN planted answer code.

    The stratum tasks deliberately plant an `answer_code` rather than the typed `[memory: ...]`
    marker -- that is what makes them evidence about elision rather than about marker preservation
    -- so the shared workflow oracle, which reads the marker back, would finish them empty.
    """
    answer = str(record["answer_code"])

    def complete(prompt: str) -> str:
        if is_summary_prompt(prompt):
            return ORACLE_SUMMARY
        reply = oracle_controller(prompt)
        call = json.loads(reply)
        if call["name"] != FINISH:
            return reply
        return json.dumps({"name": FINISH, "arguments": {"answer": answer}}, ensure_ascii=False)

    return complete


def _aggregate_search_oracle(record: dict[str, Any]) -> Oracle:
    """Search every planted term in the declared order, then finish with the scored count."""
    terms = cast(list[str], record["queries"])
    answer = str(record["answer"])
    issued = 0

    def complete(prompt: str) -> str:
        nonlocal issued
        if is_summary_prompt(prompt):
            return ORACLE_SUMMARY
        if issued < len(terms):
            query = terms[issued]
            issued += 1
            return json.dumps({"name": SEARCH, "arguments": {"query": query}}, ensure_ascii=False)
        return json.dumps({"name": FINISH, "arguments": {"answer": answer}}, ensure_ascii=False)

    return complete
