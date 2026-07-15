"""Map-reduce (long-document) evaluation template -- text analysis.

For a document too long to answer within a single context window: SPLIT it into overlapping
segments, MAP a partial answer over each segment independently, then REDUCE the partial
answers into one final answer. This is the substrate for the spec's long-doc comprehension
sub-task (Appendix D text analysis) and any summarization-style flow over oversized inputs.

Same node-closure shape as the single-call template (`graph.py`): the segmenter, the message
builders, and the node closures are pure and unit-testable WITHOUT langgraph; only
`build_map_reduce_graph` imports it (the `[eval]` extra). The shared status taxonomy and
`classify_response` come from `llb.eval.common`.

Trajectory cost (number of model calls = `n_segments` map calls + 1 reduce call, plus total
completion tokens) is recorded on the state as an efficiency signal, mirroring how the agentic
template records hop count.
"""

from typing import Any, Callable, cast

from typing_extensions import TypedDict

from llb.core.contracts.common import UsageRecord
from llb.eval.common import EMPTY, classify_response
from llb.eval.map_reduce_prompts import (
    build_map_messages,
    build_reduce_messages,
    is_no_info,
    map_text_prompt,
    reduce_text_prompt,
    split_document,
)

# A segment that yields no usable answer should say exactly this, so the reduce step can drop it.


class MapReduceState(TypedDict, total=False):
    question: str
    document: str
    segments: list[str]
    partials: list[str]
    answer: str
    status: str
    error: str | None
    usage: UsageRecord
    n_segments: int
    n_model_calls: int
    total_completion_tokens: int


def make_split_node(max_chars: int, overlap: int) -> Callable[[MapReduceState], MapReduceState]:
    """Closure: split the document into segments; flag an empty document as a terminal EMPTY."""

    def split(state: MapReduceState) -> MapReduceState:
        segments = split_document(state.get("document", ""), max_chars, overlap)
        update: MapReduceState = {"segments": segments, "n_segments": len(segments)}
        if not segments:
            update["status"] = EMPTY
        return update

    return split


def make_map_node(
    launcher: Any, max_tokens: int, temperature: float, timeout: float
) -> Callable[[MapReduceState], MapReduceState]:
    """Closure: run one map call per segment; collect the non-empty partial answers."""

    def map_segments(state: MapReduceState) -> MapReduceState:
        if state.get("status") == EMPTY:
            return {"partials": []}  # short-circuit; status already terminal
        partials: list[str] = []
        calls = 0
        completion = 0
        last_error: str | None = None
        for segment in state.get("segments", []):
            result = launcher.chat(
                build_map_messages(state["question"], segment),
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
            calls += 1
            completion += result.completion_tokens or 0
            if result.error:
                last_error = result.error
                continue
            text = result.text or ""
            if not is_no_info(text):
                partials.append(text.strip())
        update: MapReduceState = {
            "partials": partials,
            "n_model_calls": calls,
            "total_completion_tokens": completion,
        }
        if not partials and last_error:
            update["status"] = last_error  # every segment failed transport -> propagate
            update["error"] = last_error
        return update

    return map_segments


def make_reduce_node(
    launcher: Any, max_tokens: int, temperature: float, timeout: float
) -> Callable[[MapReduceState], MapReduceState]:
    """Closure: reduce the partial answers into one final answer; classify the response."""

    def reduce(state: MapReduceState) -> MapReduceState:
        if state.get("status") in (EMPTY, "timeout", "backend_error"):
            return {"answer": ""}  # short-circuit; status already terminal
        partials = state.get("partials", [])
        if not partials:
            return {"answer": "", "status": EMPTY}  # nothing recovered from any segment
        result = launcher.chat(
            build_reduce_messages(state["question"], partials),
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

    return reduce


def build_map_reduce_graph(
    launcher: Any,
    max_chars: int,
    overlap: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Any:
    """Compile the split -> map -> reduce LangGraph app. Needs the `[eval]` extra."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise SystemExit(
            'ERROR: the eval graph needs the [eval] extra. Run: uv pip install -e ".[eval]"'
        ) from exc
    graph = StateGraph(MapReduceState)
    # LangGraph's callable overloads cannot express partial TypedDict state updates.
    graph.add_node("split", cast(Any, make_split_node(max_chars, overlap)))
    graph.add_node("map", cast(Any, make_map_node(launcher, max_tokens, temperature, timeout)))
    graph.add_node(
        "reduce", cast(Any, make_reduce_node(launcher, max_tokens, temperature, timeout))
    )
    graph.add_edge(START, "split")
    graph.add_edge("split", "map")
    graph.add_edge("map", "reduce")
    graph.add_edge("reduce", END)
    return graph.compile()


def run_case(app: Any, question: str, document: str) -> MapReduceState:
    """Invoke a compiled map-reduce graph for one long-doc item; returns the terminal state."""
    return cast(MapReduceState, app.invoke({"question": question, "document": document}))


# --- text-prompt driver (the category suite `complete` substrate; no langgraph / launcher) --------------


def run_map_reduce_text(
    complete: Callable[[str], str],
    question: str,
    document: str,
    *,
    max_chars: int = 1200,
    overlap: int = 120,
) -> str:
    """Drive the split -> map -> reduce template with a `complete: str->str` callable (no langgraph).

    Splits the document, maps a partial per segment (dropping the `NO_INFO` ones), and reduces the
    survivors into one answer. A single surviving partial is returned directly (no reduce call); no
    survivors -> empty answer. This is the long-doc comprehension substrate for `bench.text_analysis`.
    """
    segments = split_document(document, max_chars, overlap)
    if not segments:
        return ""
    partials = [complete(map_text_prompt(question, seg)) for seg in segments]
    survivors = [p.strip() for p in partials if not is_no_info(p)]
    if not survivors:
        return ""
    if len(survivors) == 1:
        return survivors[0]
    return complete(reduce_text_prompt(question, survivors)).strip()
