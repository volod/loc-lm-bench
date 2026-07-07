"""Single-call RAG evaluation graph.

The flow is retrieve -> generate, the first of the three DRY LangGraph templates (the
map-reduce and multi-hop templates follow the same node-closure shape -- see `map_reduce.py`
and `multi_hop.py`). The node functions are plain closures over a `RagState` dict, so the
retrieval, prompt-building, and failure-classification logic is unit-testable WITHOUT
langgraph installed; only `build_rag_graph` imports it (the `[eval]` extra).

The shared status taxonomy, refusal markers, `classify_response`, and `format_context` live
in `llb.eval.common`; see that module for the failure-taxonomy contract.
"""

from typing import Any, Callable, cast

from typing_extensions import TypedDict

from llb.core.contracts import ChatMessage, ChunkRecord, SourceSpanRecord, UsageRecord
from llb.eval import common as eval_common
from llb.prompts import PromptAugmentation, render_chat, render_text

__all__ = [
    "RagState",
    "SYSTEM_PROMPT",
    "build_messages",
    "build_rag_graph",
    "make_generate_node",
    "make_retrieve_node",
    "run_case",
]

SYSTEM_PROMPT = render_text("eval.rag.system")


class RagState(TypedDict, total=False):
    question: str
    gold_spans: list[SourceSpanRecord]
    retrieved: list[ChunkRecord]
    context: str
    answer: str
    status: str
    error: str | None
    usage: UsageRecord


def build_messages(
    question: str, context: str, prompt_package: Any | None = None
) -> list[ChatMessage]:
    augmentation: PromptAugmentation | None = None
    if prompt_package is not None:
        augmentation = PromptAugmentation(system_prefix=str(prompt_package.system_prompt))
        extra = str(prompt_package.additional_prompt).strip()
        if extra:
            context = render_text(
                "eval.rag.package_context",
                {"additional_prompt": extra, "context": context},
            )
    return render_chat(
        "eval.rag.chat",
        {"context": context, "question": question},
        augmentation=augmentation,
    )


def make_retrieve_node(store: Any, k: int) -> Callable[[RagState], RagState]:
    """Closure: retrieve top-k chunks; flag retrieval_miss when nothing comes back."""

    def retrieve(state: RagState) -> RagState:
        chunks = store.retrieve(state["question"], k)
        update: RagState = {"retrieved": chunks, "context": eval_common.format_context(chunks)}
        if not chunks:
            update["status"] = eval_common.RETRIEVAL_MISS
        return update

    return retrieve


def make_generate_node(
    launcher: Any,
    max_tokens: int,
    temperature: float,
    timeout: float,
    prompt_package: Any | None = None,
) -> Callable[[RagState], RagState]:
    """Closure: call the backend on the retrieved context; classify the response."""

    def generate(state: RagState) -> RagState:
        if state.get("status") == eval_common.RETRIEVAL_MISS:
            return {"answer": "", "usage": {}}  # short-circuit; status already terminal
        messages = build_messages(state["question"], state.get("context", ""), prompt_package)
        result = launcher.chat(
            messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout
        )
        return {
            "answer": result.text or "",
            "status": eval_common.classify_response(result.text, result.error),
            "error": result.error,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_s": result.latency_s,
                "tokens_per_s": result.tokens_per_s(),
            },
        }

    return generate


def build_rag_graph(
    store: Any,
    launcher: Any,
    k: int,
    max_tokens: int,
    temperature: float,
    timeout: float,
    prompt_package: Any | None = None,
) -> Any:
    """Compile the retrieve -> generate LangGraph app. Needs the `[eval]` extra."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise SystemExit(
            'ERROR: the eval graph needs the [eval] extra. Run: uv pip install -e ".[eval]"'
        ) from exc
    graph = StateGraph(RagState)
    # LangGraph's callable overloads cannot express partial TypedDict state updates.
    graph.add_node("retrieve", cast(Any, make_retrieve_node(store, k)))
    graph.add_node(
        "generate",
        cast(Any, make_generate_node(launcher, max_tokens, temperature, timeout, prompt_package)),
    )
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_case(app: Any, question: str, gold_spans: list[SourceSpanRecord]) -> RagState:
    """Invoke a compiled graph for one gold item; returns the terminal state."""
    return cast(RagState, app.invoke({"question": question, "gold_spans": gold_spans}))
