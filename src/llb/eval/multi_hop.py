"""Multi-hop retrieval evaluation template -- text analysis.

Iterative retrieve -> reason -> (optionally retrieve again) for questions whose answer must
chain several facts. After each retrieval, a CONTROLLER node decides whether the gathered
context is sufficient (stop and answer) or another sub-query is needed (hop again), bounded by
`max_hops`. An answer node then synthesizes the final answer over everything gathered.

This is the SUBSTRATE for the agentic benchmark: that benchmark extends the controller
to emit tool calls and adds an in-sandbox tool-execution node, but the retrieve/controller/
answer loop + conditional routing defined here is the fixed shape it grows from.

Same node-closure convention as `graph.py`: the controller parser, message builders, node
closures, and the router are pure and unit-testable WITHOUT langgraph; only
`build_multi_hop_graph` imports it (the `[eval]` extra). Trajectory length (`n_hops`) and the
model-call/token counts are recorded on the state as the efficiency signal agentic benchmark ranks on.
"""

from typing import Any, Callable, cast

from typing_extensions import TypedDict

from llb.core.contracts import ChunkRecord, SourceSpanRecord, UsageRecord
from llb.eval.common import RETRIEVAL_MISS, classify_response
from llb.eval.multi_hop_prompts import (
    CONTINUE,
    DEFAULT_MAX_HOPS,
    STOP,
    _chunk_key,
    build_answer_messages,
    build_controller_messages,
    parse_controller,
)

# Controller decisions.

# Controller protocol markers (UA). The controller replies with one of these on its first line.


class MultiHopState(TypedDict, total=False):
    question: str
    gold_spans: list[SourceSpanRecord]
    subquery: str
    hop: int
    max_hops: int
    decision: str
    gathered: list[ChunkRecord]
    seen_keys: list[str]
    answer: str
    status: str
    error: str | None
    usage: UsageRecord
    n_hops: int
    n_model_calls: int
    total_completion_tokens: int


def make_retrieve_node(store: Any, k: int) -> Callable[[MultiHopState], MultiHopState]:
    """Closure: retrieve for the current sub-query (the question on hop 0); merge NEW chunks
    into `gathered`, deduped across hops, and advance the hop counter."""

    def retrieve(state: MultiHopState) -> MultiHopState:
        query = state.get("subquery") or state["question"]
        chunks = store.retrieve(query, k)
        gathered = list(state.get("gathered", []))
        seen = set(state.get("seen_keys", []))
        for chunk in chunks:
            key = _chunk_key(chunk)
            if key not in seen:
                seen.add(key)
                gathered.append(chunk)
        return {
            "gathered": gathered,
            "seen_keys": sorted(seen),
            "hop": state.get("hop", 0) + 1,
            "n_hops": state.get("hop", 0) + 1,
        }

    return retrieve


def make_controller_node(
    launcher: Any, max_tokens: int, temperature: float, timeout: float
) -> Callable[[MultiHopState], MultiHopState]:
    """Closure: ask the model whether to stop or hop again, and for the next sub-query."""

    def controller(state: MultiHopState) -> MultiHopState:
        result = launcher.chat(
            build_controller_messages(state["question"], state.get("gathered", [])),
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        calls = state.get("n_model_calls", 0) + 1
        completion = state.get("total_completion_tokens", 0) + (result.completion_tokens or 0)
        if result.error:  # a failed controller turn ends the loop; answer node will short-circuit
            return {
                "decision": STOP,
                "subquery": "",
                "n_model_calls": calls,
                "total_completion_tokens": completion,
            }
        decision, subquery = parse_controller(result.text)
        return {
            "decision": decision,
            "subquery": subquery,
            "n_model_calls": calls,
            "total_completion_tokens": completion,
        }

    return controller


def route_after_controller(state: MultiHopState) -> str:
    """Conditional edge: hop again only when the controller asked to continue AND the hop
    budget is not exhausted; otherwise synthesize the answer."""
    if state.get("decision") == CONTINUE and state.get("hop", 0) < state.get(
        "max_hops", DEFAULT_MAX_HOPS
    ):
        return "retrieve"
    return "answer"


def make_answer_node(
    launcher: Any, max_tokens: int, temperature: float, timeout: float
) -> Callable[[MultiHopState], MultiHopState]:
    """Closure: synthesize the final answer over everything gathered; classify the response."""

    def answer(state: MultiHopState) -> MultiHopState:
        gathered = state.get("gathered", [])
        if not gathered:
            return {"answer": "", "status": RETRIEVAL_MISS}  # nothing retrieved across all hops
        result = launcher.chat(
            build_answer_messages(state["question"], gathered),
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        completion = state.get("total_completion_tokens", 0) + (result.completion_tokens or 0)
        return {
            "answer": result.text or "",
            "status": classify_response(result.text, result.error),
            "error": result.error,
            "n_model_calls": state.get("n_model_calls", 0) + 1,
            "total_completion_tokens": completion,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": completion,
                "latency_s": result.latency_s,
                "tokens_per_s": result.tokens_per_s(),
            },
        }

    return answer


def build_multi_hop_graph(
    store: Any,
    launcher: Any,
    k: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Any:
    """Compile the retrieve -> controller -> {retrieve | answer} LangGraph app. Needs `[eval]`."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise SystemExit(
            'ERROR: the eval graph needs the [eval] extra. Run: uv pip install -e ".[eval]"'
        ) from exc
    graph = StateGraph(MultiHopState)
    # LangGraph's callable overloads cannot express partial TypedDict state updates.
    graph.add_node("retrieve", cast(Any, make_retrieve_node(store, k)))
    graph.add_node(
        "controller", cast(Any, make_controller_node(launcher, max_tokens, temperature, timeout))
    )
    graph.add_node(
        "answer", cast(Any, make_answer_node(launcher, max_tokens, temperature, timeout))
    )
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "controller")
    graph.add_conditional_edges(
        "controller",
        cast(Any, route_after_controller),
        {"retrieve": "retrieve", "answer": "answer"},
    )
    graph.add_edge("answer", END)
    return graph.compile()


def run_case(
    app: Any,
    question: str,
    gold_spans: list[SourceSpanRecord],
    max_hops: int = DEFAULT_MAX_HOPS,
) -> MultiHopState:
    """Invoke a compiled multi-hop graph for one item; returns the terminal state."""
    return cast(
        MultiHopState,
        app.invoke({"question": question, "gold_spans": gold_spans, "max_hops": max_hops}),
    )
