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

from llb.contracts import ChatMessage, ChunkRecord, SourceSpanRecord, UsageRecord
from llb.eval.common import RETRIEVAL_MISS, classify_response, format_context

# Controller decisions.
CONTINUE = "continue"
STOP = "stop"

# Controller protocol markers (UA). The controller replies with one of these on its first line.
DONE_MARKER = "ГОТОВО"  # enough gathered -> synthesize the answer
NEXT_MARKER = "ДАЛІ:"  # need more -> the text after the colon is the next sub-query

DEFAULT_MAX_HOPS = 3

CONTROLLER_SYSTEM_PROMPT = (
    "Ти плануєш багатокроковий пошук відповіді. Тобі дано питання та вже зібрані факти. "
    f"Якщо зібраних фактів достатньо, щоб відповісти, напиши рівно: {DONE_MARKER}. "
    f"Якщо ще чогось бракує, напиши: {NEXT_MARKER} <наступний підзапит для пошуку>. "
    "Постав лише ОДИН наступний підзапит і нічого більше."
)

ANSWER_SYSTEM_PROMPT = (
    "Ти відповідаєш на питання, спираючись ВИКЛЮЧНО на зібрані факти. "
    "Якщо їх недостатньо, скажи, що інформації недостатньо. "
    "Відповідай стисло українською мовою."
)


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


def parse_controller(text: str) -> tuple[str, str]:
    """Map a controller reply to (decision, next_subquery).

    `DONE_MARKER` -> (STOP, ""); `NEXT_MARKER <q>` -> (CONTINUE, q). Anything else (including an
    empty reply) is treated as STOP so a malformed controller turn ends the loop safely rather
    than spinning; `max_hops` is the hard bound regardless.
    """
    stripped = (text or "").strip()
    if not stripped:
        return STOP, ""
    upper = stripped.upper()
    idx = upper.find(NEXT_MARKER)
    if idx >= 0:
        remainder = stripped[idx + len(NEXT_MARKER) :].strip().splitlines()
        subquery = remainder[0].strip() if remainder else ""
        if subquery:
            return CONTINUE, subquery
    return STOP, ""


def _chunk_key(chunk: ChunkRecord) -> str:
    """Stable identity for de-duplicating chunks gathered across hops."""
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"{chunk.get('doc_id', '?')}:{chunk.get('char_start')}:{chunk.get('char_end')}"


def build_controller_messages(question: str, gathered: list[ChunkRecord]) -> list[ChatMessage]:
    facts = format_context(gathered) if gathered else "(поки що нічого не знайдено)"
    user = f"Питання: {question}\n\nЗібрані факти:\n<<<\n{facts}\n>>>"
    return [
        {"role": "system", "content": CONTROLLER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_answer_messages(question: str, gathered: list[ChunkRecord]) -> list[ChatMessage]:
    facts = format_context(gathered)
    user = f"Зібрані факти:\n<<<\n{facts}\n>>>\n\nПитання: {question}\n\nВідповідь:"
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


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
