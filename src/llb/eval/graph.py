"""Single-call RAG evaluation graph.

The flow is retrieve -> generate, the first of the three DRY LangGraph templates (the
map-reduce and multi-hop templates follow the same node-closure shape -- see `map_reduce.py`
and `multi_hop.py`). The node functions are plain closures over a `RagState` dict, so the
retrieval, prompt-building, and failure-classification logic is unit-testable WITHOUT
langgraph installed; only `build_rag_graph` imports it (the `[eval]` extra).

The shared status taxonomy, refusal markers, `classify_response`, and `format_context` live
in `llb.eval.common`; see that module for the failure-taxonomy contract.
"""

import time
from typing import Any, Callable, cast

from typing_extensions import TypedDict

from llb.core.contracts.common import ChatMessage, UsageRecord
from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.eval import common as eval_common
from llb.prompts.engine import PromptAugmentation
from llb.prompts.registry import render_chat, render_text

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
    # Per-stage wall-clock (rerank-context-order): retrieval always, rerank when a reranking
    # store is wired. Generation latency stays in `usage` (from the backend's ChatResult).
    retrieve_latency_s: float
    rerank_latency_s: float
    # Query-side processing lane (uk-query-processing): the processed query actually retrieved
    # with and the number of transformations applied. The raw query stays in `question`, so both
    # forms are recoverable per case; absent when the lane is off.
    query_processed: str
    query_corrections: int
    query_hypothetical_answer: str
    query_decomposition: str
    query_subqueries: list[str]


# Generation prompt ids: the baseline RAG chat and the cited-answer variant that requires `[i]`
# chunk citations for factual claims (groundedness-citation-metrics).
CHAT_TEMPLATE = "eval.rag.chat"
CITED_ANSWER_TEMPLATE = "eval.rag.cited_answer"


def build_messages(
    question: str, context: str, prompt_package: Any | None = None, cited: bool = False
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
        CITED_ANSWER_TEMPLATE if cited else CHAT_TEMPLATE,
        {"context": context, "question": question},
        augmentation=augmentation,
    )


def make_retrieve_node(
    store: Any,
    k: int,
    context_order: str = eval_common.ORDER_RANK,
    query_prep: Any | None = None,
    chunk_filter: Any | None = None,
) -> Callable[[RagState], RagState]:
    """Closure: retrieve top-k chunks; flag retrieval_miss when nothing comes back.

    `context_order` is the rerank-context-order policy applied when the kept chunks are laid
    into the prompt; `retrieved` stays in rank order so the source-span metrics are unaffected.
    A reranking store (`llb.rag.rerank.RerankingRetriever`) exposes its per-stage wall-clock,
    recorded as `retrieve_latency_s` / `rerank_latency_s`.

    `query_prep` (`llb.rag.query_prep.pipeline.QueryPrep`) is the opt-in query-side lane: when set, the
    question is processed BEFORE retrieval (the raw question stays in state for generation), and
    the processed form + correction count are recorded (uk-query-processing).
    """

    def retrieve(state: RagState) -> RagState:
        question = state["question"]
        prep_update: RagState = {}
        if query_prep is not None:
            result = query_prep.process(question)
            prep_update = cast(RagState, result.provenance())
        started = time.perf_counter()
        if query_prep is not None:
            from llb.rag.query_prep.retrieval import retrieve_prepared

            chunks = retrieve_prepared(store, result, k, chunk_filter=chunk_filter)
        elif chunk_filter is None:
            chunks = store.retrieve(question, k)
        else:
            chunks = store.retrieve(question, k, chunk_filter=chunk_filter)
        total_s = time.perf_counter() - started
        update: RagState = {
            "retrieved": chunks,
            "context": eval_common.format_context(chunks, order=context_order),
            **prep_update,
        }
        stage = getattr(store, "stage_latency", None)
        if isinstance(stage, dict) and "rerank_s" in stage:
            update["retrieve_latency_s"] = float(stage.get("retrieve_s", 0.0))
            update["rerank_latency_s"] = float(stage["rerank_s"])
        else:
            update["retrieve_latency_s"] = total_s
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
    cited: bool = False,
) -> Callable[[RagState], RagState]:
    """Closure: call the backend on the retrieved context; classify the response."""

    def generate(state: RagState) -> RagState:
        if state.get("status") == eval_common.RETRIEVAL_MISS:
            return {"answer": "", "usage": {}}  # short-circuit; status already terminal
        messages = build_messages(
            state["question"], state.get("context", ""), prompt_package, cited=cited
        )
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
    context_order: str = eval_common.ORDER_RANK,
    query_prep: Any | None = None,
    chunk_filter: Any | None = None,
    cited: bool = False,
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
    graph.add_node(
        "retrieve",
        cast(Any, make_retrieve_node(store, k, context_order, query_prep, chunk_filter)),
    )
    graph.add_node(
        "generate",
        cast(
            Any,
            make_generate_node(launcher, max_tokens, temperature, timeout, prompt_package, cited),
        ),
    )
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_case(app: Any, question: str, gold_spans: list[SourceSpanRecord]) -> RagState:
    """Invoke a compiled graph for one gold item; returns the terminal state."""
    return cast(RagState, app.invoke({"question": question, "gold_spans": gold_spans}))
