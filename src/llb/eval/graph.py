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

from llb.backends.base import ChatResult
from llb.core.contracts.common import ChatMessage
from llb.core.contracts.rag import SourceSpanRecord
from llb.eval import common as eval_common
from llb.eval.answer_envelope import boundary as envelope_boundary
from llb.eval.answer_envelope import lane as envelope_lane
from llb.eval.answer_envelope.models import envelope_prompt_values
from llb.eval.graph_contracts import ContextSource, RagState
from llb.eval.table_headers import HeaderRestorer, prompt_context
from llb.prompts.engine import PromptAugmentation
from llb.prompts.registry import render_chat, render_text

__all__ = [
    "SYSTEM_PROMPT",
    "build_messages",
    "build_rag_graph",
    "generation_template",
    "make_generate_node",
    "make_retrieve_node",
    "run_case",
]

SYSTEM_PROMPT = render_text("eval.rag.system")


# Generation prompt ids: the baseline RAG chat, the cited-answer variant that requires `[i]`
# chunk citations for factual claims (groundedness-citation-metrics), and the closed-book variant
# that lays NO context into the prompt at all (rag-vs-long-context-ablation).
CHAT_TEMPLATE = "eval.rag.chat"
CITED_ANSWER_TEMPLATE = "eval.rag.cited_answer"
CLOSED_BOOK_TEMPLATE = "eval.rag.closed_book"
# The declared-answer variant (typed-rag-answer-envelope): the model returns the typed contract
# instead of prose, and the answer-side signals are read from its fields.
ENVELOPE_TEMPLATE = "eval.rag.envelope"


# A context source REPLACES store retrieval for a diagnostic context lane. It returns the same
# partial state update the retrieve node would (`retrieved` / `context`, plus an optional terminal
# `status`), so the graph, the per-case scoring, and the run-bundle shape are all unchanged.
def generation_template(cited: bool = False, answer_format: str = envelope_lane.FREE_TEXT) -> str:
    """The generation prompt id for this run's answer style.

    The declared answer format supersedes the citation style: the envelope asks for citations as a
    typed field, so the `[i]`-in-prose instruction would only be a second, weaker way to ask.
    """
    if answer_format == envelope_lane.ENVELOPE:
        return ENVELOPE_TEMPLATE
    return CITED_ANSWER_TEMPLATE if cited else CHAT_TEMPLATE


def build_messages(
    question: str,
    context: str,
    prompt_package: Any | None = None,
    cited: bool = False,
    template_id: str | None = None,
    answer_format: str = envelope_lane.FREE_TEXT,
) -> list[ChatMessage]:
    """Render the generation prompt. `template_id` overrides the `cited` style selection."""
    augmentation: PromptAugmentation | None = None
    if prompt_package is not None:
        augmentation = PromptAugmentation(system_prefix=str(prompt_package.system_prompt))
        extra = str(prompt_package.additional_prompt).strip()
        if extra:
            context = render_text(
                "eval.rag.package_context",
                {"additional_prompt": extra, "context": context},
            )
    selected = template_id or generation_template(cited, answer_format)
    values: dict[str, Any] = {"context": context, "question": question}
    if selected == ENVELOPE_TEMPLATE:
        values.update(envelope_prompt_values())
    return render_chat(selected, values, augmentation=augmentation)


def make_retrieve_node(
    store: Any,
    k: int,
    context_order: str = eval_common.ORDER_RANK,
    query_prep: Any | None = None,
    chunk_filter: Any | None = None,
    context_source: ContextSource | None = None,
    header_restorer: HeaderRestorer | None = None,
) -> Callable[[RagState], RagState]:
    """Closure: retrieve top-k chunks; flag retrieval_miss when nothing comes back.

    `context_order` is the rerank-context-order policy applied when the kept chunks are laid
    into the prompt; `retrieved` stays in rank order so the source-span metrics are unaffected.
    A reranking store (`llb.rag.rerank.RerankingRetriever`) exposes its per-stage wall-clock,
    recorded as `retrieve_latency_s` / `rerank_latency_s`.

    `query_prep` (`llb.rag.query_prep.pipeline.QueryPrep`) is the opt-in query-side lane: when set, the
    question is processed BEFORE retrieval (the raw question stays in state for generation), and
    the processed form + correction count are recorded (uk-query-processing).

    `context_source` replaces store retrieval entirely for a diagnostic context lane
    (rag-vs-long-context-ablation): the store, `k`, the ordering policy, and the query-prep lane
    all belong to retrieval, so a lane that does not retrieve simply supplies its own update.

    `header_restorer` (`llb.eval.table_headers`) is the opt-in prompt-side context-assembly step
    that gives a table row block back its column names (table-header-context-restoration). It
    rewrites the PROMPT copies only -- `retrieved` keeps the stored records -- so the source-span
    metrics cannot move with it. Its accounting is recorded on every retrieving case, restorer or
    not, so an off lane and an on lane carry the same column and stay comparable.
    """

    def retrieve(state: RagState) -> RagState:
        if context_source is not None:
            return context_source(state)
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
        total_s = time.perf_counter() - started  # retrieval only; assembly is not retrieval
        update: RagState = {
            "retrieved": chunks,
            **prompt_context(chunks, context_order, header_restorer),
            **_stage_latency(store, total_s),
            **prep_update,
        }
        if not chunks:
            update["status"] = eval_common.RETRIEVAL_MISS
        return update

    return retrieve


def _stage_latency(store: Any, total_s: float) -> RagState:
    """Per-stage wall-clock of one retrieval: a reranking store splits it, others report one total."""
    stage = getattr(store, "stage_latency", None)
    if isinstance(stage, dict) and "rerank_s" in stage:
        return {
            "retrieve_latency_s": float(stage.get("retrieve_s", 0.0)),
            "rerank_latency_s": float(stage["rerank_s"]),
        }
    return {"retrieve_latency_s": total_s}


def make_generate_node(
    launcher: Any,
    max_tokens: int,
    temperature: float,
    timeout: float,
    prompt_package: Any | None = None,
    cited: bool = False,
    template_id: str | None = None,
    answer_format: str = envelope_lane.FREE_TEXT,
) -> Callable[[RagState], RagState]:
    """Closure: call the backend on the retrieved context; classify the response.

    `template_id` overrides the generation prompt, which is how the closed-book context lane asks
    the model to answer from its own weights instead of from an (empty) context block.

    `answer_format` selects the ANSWER CONTRACT (typed-rag-answer-envelope). `free_text` is the
    unchanged path -- same prompt, same single call, same state keys, so a run with the envelope
    off records exactly what it recorded before this seam existed. `envelope` asks for the typed
    contract and parses it at this boundary, spending at most one repair reprompt.
    """

    def chat(messages: list[ChatMessage]) -> ChatResult:
        return cast(
            ChatResult,
            launcher.chat(
                messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout
            ),
        )

    def generate(state: RagState) -> RagState:
        if state.get("status") in eval_common.PRE_GENERATION_STATUSES:
            return {"answer": "", "usage": {}}  # short-circuit; status already terminal
        messages = build_messages(
            state["question"],
            state.get("context", ""),
            prompt_package,
            cited=cited,
            template_id=template_id,
            answer_format=answer_format,
        )
        result = chat(messages)
        if answer_format == envelope_lane.ENVELOPE:
            return envelope_lane.envelope_state(
                envelope_boundary.complete_envelope(chat, messages, result)
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
    context_source: ContextSource | None = None,
    template_id: str | None = None,
    header_restorer: HeaderRestorer | None = None,
    answer_format: str = envelope_lane.FREE_TEXT,
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
        cast(
            Any,
            make_retrieve_node(
                store,
                k,
                context_order,
                query_prep,
                chunk_filter,
                context_source=context_source,
                header_restorer=header_restorer,
            ),
        ),
    )
    graph.add_node(
        "generate",
        cast(
            Any,
            make_generate_node(
                launcher,
                max_tokens,
                temperature,
                timeout,
                prompt_package,
                cited,
                template_id=template_id,
                answer_format=answer_format,
            ),
        ),
    )
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_case(app: Any, question: str, gold_spans: list[SourceSpanRecord]) -> RagState:
    """Invoke a compiled graph for one gold item; returns the terminal state."""
    return cast(RagState, app.invoke({"question": question, "gold_spans": gold_spans}))
