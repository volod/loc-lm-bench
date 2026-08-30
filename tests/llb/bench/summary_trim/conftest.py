"""One injected oracle that can walk the whole adoption design, shared by both test modules.

The runner drives five workloads with three different task builders, so a fake `complete` that
answers them all is what stands in for a GPU here. It is a fixture rather than a helper because two
modules now need it: the study's own contracts, and the per-family guard fit that runs a walk
control before the arms.
"""

import pytest

from llb.bench.summary_trim.design import load_summary_trim_design, workloads
from llb.bench.summary_trim.run import FamilyRun, run_summary_trim_family
from llb.bench.summary_trim.workloads import build_workload_tasks, workload_oracle

# The step prompt carries this header only once the episode has a history to render, so a prompt
# WITHOUT it is the first step of a fresh episode.
_HISTORY_HEADER = "Виконані кроки:"


@pytest.fixture(scope="module")
def adoption_design() -> dict[str, object]:
    return load_summary_trim_design()


@pytest.fixture
def oracle_complete(adoption_design: dict[str, object]):
    """One fake `complete` that plays whichever declared task the loop is currently walking.

    An oracle is per EPISODE -- the aggregate-search one counts the queries it has issued -- and
    the runner walks each task more than once, so a fresh oracle is taken when the prompt shows no
    history rather than when the task changes.
    """
    oracles = {
        record["prompt"][:60]: (workload, record)
        for workload in workloads(adoption_design)
        for record in build_workload_tasks(workload)
    }
    state: dict[str, object] = {"oracle": None}

    def complete(prompt: str) -> str:
        key = next((candidate for candidate in oracles if candidate in prompt), None)
        if key is not None and _HISTORY_HEADER not in prompt:
            state["oracle"] = workload_oracle(*oracles[key])
        oracle = state["oracle"]
        if oracle is None:
            return '{"name": "finish", "arguments": {"answer": ""}}'
        return oracle(prompt)  # type: ignore[operator]

    return complete


@pytest.fixture
def oracle_family(adoption_design: dict[str, object], oracle_complete):
    """Run one family of the committed design end to end over the injected oracles."""

    def run(name: str, *, order_offset: int = 0) -> FamilyRun:
        return run_summary_trim_family(
            adoption_design,
            {"model_family": name, "model": name, "backend": "ollama"},
            complete=oracle_complete,
            order_offset=order_offset,
        )

    return run
