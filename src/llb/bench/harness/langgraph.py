"""agentic harness comparison LangGraph agentic harness -- the pure loop compiled as a LangGraph app.

Mirrors `llb.eval.multi_hop.build_multi_hop_graph`: the controller->execute->controller cycle is
decomposed into an `agent` node (the model proposes one tool call), a `tool` node (the
deterministic `ToolWorld` executes it), and two conditional edges that route exactly like the
pure loop -- so the compiled graph produces the SAME `Episode` as `run_episode`. The node closures
and routers are pure and unit-testable WITHOUT langgraph (same convention as `multi_hop`); only
`build_agentic_graph` imports langgraph (the `[eval]` extra), so the base install stays light.

Context policy and budget ride through the same `step_prompt` / `ContextState` seam the loop uses,
so a measured policy win transfers to this harness rather than only living on `loop`.
"""

from typing import Any, Callable, cast

from typing_extensions import TypedDict

from llb.bench.agentic.context import ContextState
from llb.bench.agentic.context_policy import ContextPolicy
from llb.bench.agentic.context_budget import ContextBudget, unbounded_budget
from llb.bench.agentic.episode_prompt import step_prompt
from llb.bench.agentic.model import (
    DEFAULT_MAX_STEPS,
    STATUS_COMPLETED,
    STATUS_CONTEXT_OVERFLOW,
    STATUS_INCOMPLETE,
    AgenticTask,
    Episode,
)
from llb.bench.agentic.success import check_success
from llb.bench.common import LLMComplete
from llb.bench.tool_world import FINISH, ToolWorld
from llb.core.contracts.benchmarks import ToolDef
from llb.scoring.tooling.tool_calls import parse_tool_call

# Routing labels (the conditional-edge targets).
ROUTE_TOOL = "tool"
ROUTE_AGENT = "agent"
ROUTE_END = "end"


class AgenticGraphState(TypedDict, total=False):
    task: AgenticTask  # immutable: the goal + initial setup + success assertions
    max_steps: int
    world: ToolWorld  # the mutable env the tool node executes against
    context: ContextState  # policy-managed transcript + prompt telemetry
    answer: str
    finished: bool  # the agent emitted `finish` or answered in prose
    overflow: bool  # a step prompt did not fit the resolved window
    n_steps: int  # model calls in the trajectory
    n_tool_calls: int  # sandbox tools executed
    pending_name: str  # the tool the agent just proposed (consumed by the tool node)
    pending_args: dict[str, Any]


def make_agent_node(
    complete: LLMComplete,
    catalog: dict[str, ToolDef],
    policy: ContextPolicy,
    budget: ContextBudget,
) -> Callable[[AgenticGraphState], AgenticGraphState]:
    """Closure: the model proposes one tool call (or finishes / answers in prose).

    Same step semantics as the loop body: a non-tool reply is the final prose answer, `finish`
    ends with its `answer`, any other call is staged in `pending_*` for the tool node. The step
    prompt is assembled under `policy` and refused under `budget` exactly as `run_episode` does."""

    def agent(state: AgenticGraphState) -> AgenticGraphState:
        task = state["task"]
        context = state["context"]
        prompt = step_prompt(task, catalog, policy, context, budget, complete)
        context.telemetry.prompt_chars.append(len(prompt))
        if not budget.fits(len(prompt)):
            return {"finished": True, "overflow": True, "answer": ""}
        context.telemetry.model_input_prompt_chars += len(prompt)
        raw = complete(prompt)
        n_steps = state.get("n_steps", 0) + 1
        call = parse_tool_call(raw)
        if call is None:  # answered in prose -> final answer
            return {"answer": raw.strip(), "finished": True, "n_steps": n_steps}
        if call.name == FINISH:
            return {
                "answer": str(call.arguments.get("answer", "")),
                "finished": True,
                "n_steps": n_steps,
            }
        return {
            "finished": False,
            "n_steps": n_steps,
            "pending_name": call.name,
            "pending_args": call.arguments,
        }

    return agent


def make_tool_node(
    policy: ContextPolicy,
) -> Callable[[AgenticGraphState], AgenticGraphState]:
    """Closure: execute the agent's pending tool against the world and record the observation."""

    def tool(state: AgenticGraphState) -> AgenticGraphState:
        world = state["world"]
        context = state["context"]
        name = state.get("pending_name", "")
        args = state.get("pending_args", {})
        observation = world.execute(name, args)
        context.record(policy, name, args, observation)
        return {
            "context": context,
            "world": world,
            "n_tool_calls": state.get("n_tool_calls", 0) + 1,
        }

    return tool


def route_after_agent(state: AgenticGraphState) -> str:
    """Finish/prose/overflow ends the episode; otherwise execute the proposed tool."""
    return ROUTE_END if state.get("finished") else ROUTE_TOOL


def route_after_tool(state: AgenticGraphState) -> str:
    """Hop back to the agent only while the step budget remains (matches the loop's range bound)."""
    if state.get("n_steps", 0) < state.get("max_steps", DEFAULT_MAX_STEPS):
        return ROUTE_AGENT
    return ROUTE_END


def episode_from_state(task: AgenticTask, state: AgenticGraphState) -> Episode:
    """Adapt a terminal graph state into the canonical `Episode` (success re-checked objectively)."""
    world = state.get("world") or ToolWorld.from_setup(task.setup)
    context = state.get("context") or ContextState()
    answer = state.get("answer", "")
    finished = bool(state.get("finished"))
    overflow = bool(state.get("overflow"))
    if overflow:
        status = STATUS_CONTEXT_OVERFLOW
    elif finished:
        status = STATUS_COMPLETED
    else:
        status = STATUS_INCOMPLETE
    return Episode(
        success=check_success(task, world, answer),
        status=status,
        n_steps=state.get("n_steps", 0),
        n_tool_calls=state.get("n_tool_calls", 0),
        answer=answer,
        world=world,
        transcript=list(context.executed),
        telemetry=context.telemetry,
        context_policy_supported=True,
    )


def _initial_state(task: AgenticTask, max_steps: int) -> AgenticGraphState:
    return {
        "task": task,
        "world": ToolWorld.from_setup(task.setup),
        "max_steps": max_steps,
        "context": ContextState(),
        "n_steps": 0,
        "n_tool_calls": 0,
        "overflow": False,
    }


def _resolve_policy_budget(
    policy: ContextPolicy | None, budget: ContextBudget | None
) -> tuple[ContextPolicy, ContextBudget]:
    return (
        policy if policy is not None else ContextPolicy(),
        budget if budget is not None else unbounded_budget(),
    )


def build_agentic_graph(
    complete: LLMComplete,
    catalog: dict[str, ToolDef],
    *,
    policy: ContextPolicy | None = None,
    budget: ContextBudget | None = None,
) -> Any:
    """Compile the agent -> {tool -> agent | END} LangGraph app. Needs the `[eval]` extra."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise SystemExit(
            'ERROR: the langgraph harness needs the [eval] extra. Run: uv pip install -e ".[eval]"'
        ) from exc
    resolved_policy, resolved_budget = _resolve_policy_budget(policy, budget)
    graph = StateGraph(AgenticGraphState)
    # LangGraph's callable overloads cannot express partial TypedDict state updates.
    graph.add_node(
        ROUTE_AGENT, cast(Any, make_agent_node(complete, catalog, resolved_policy, resolved_budget))
    )
    graph.add_node(ROUTE_TOOL, cast(Any, make_tool_node(resolved_policy)))
    graph.add_edge(START, ROUTE_AGENT)
    graph.add_conditional_edges(
        ROUTE_AGENT, cast(Any, route_after_agent), {ROUTE_TOOL: ROUTE_TOOL, ROUTE_END: END}
    )
    graph.add_conditional_edges(
        ROUTE_TOOL, cast(Any, route_after_tool), {ROUTE_AGENT: ROUTE_AGENT, ROUTE_END: END}
    )
    return graph.compile()


def run_agentic_case(app: Any, task: AgenticTask, max_steps: int) -> AgenticGraphState:
    """Invoke a compiled agentic graph for one task; returns the terminal state.

    The recursion limit is sized to the step budget (each step is an agent+tool pair) so a
    long-but-valid trajectory is never cut short by LangGraph's default guard."""
    return cast(
        AgenticGraphState,
        app.invoke(_initial_state(task, max_steps), {"recursion_limit": 2 * max_steps + 5}),
    )


def langgraph_harness(
    task: AgenticTask,
    complete: LLMComplete,
    catalog: dict[str, ToolDef],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    policy: ContextPolicy | None = None,
    budget: ContextBudget | None = None,
) -> Episode:
    """The `Harness`: compile the agentic graph and drive one task to a canonical `Episode`."""
    app = build_agentic_graph(complete, catalog, policy=policy, budget=budget)
    state = run_agentic_case(app, task, max_steps)
    return episode_from_state(task, state)


def step_graph_pure(
    task: AgenticTask,
    complete: LLMComplete,
    catalog: dict[str, ToolDef],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    policy: ContextPolicy | None = None,
    budget: ContextBudget | None = None,
) -> Episode:
    """Drive the SAME pure node closures + routers WITHOUT langgraph (for CI equivalence tests).

    This executes the agent/tool nodes exactly as the compiled graph's conditional edges would,
    so a test can assert the LangGraph harness reproduces `run_episode` even when langgraph is not
    installed. It is NOT the production path (`langgraph_harness` is); it shares the same nodes.
    """
    resolved_policy, resolved_budget = _resolve_policy_budget(policy, budget)
    agent = make_agent_node(complete, catalog, resolved_policy, resolved_budget)
    tool = make_tool_node(resolved_policy)
    state: AgenticGraphState = _initial_state(task, max_steps)
    for _ in range(2 * max_steps + 1):
        state.update(agent(state))
        if route_after_agent(state) == ROUTE_END:
            break
        state.update(tool(state))
        if route_after_tool(state) == ROUTE_END:
            break
    return episode_from_state(task, state)
