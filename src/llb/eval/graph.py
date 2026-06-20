"""Single-call RAG evaluation graph + typed failure taxonomy.

The flow is retrieve -> generate, wired as a LangGraph so every eval flow shares one
uniform pattern (the design standardizes all eval flows on LangGraph; map-reduce and
multi-hop templates follow the same shape later). The node functions are plain closures
over a `RagState` dict, so the retrieval, prompt-building, and failure-classification
logic is unit-testable WITHOUT langgraph installed; only `build_rag_graph` imports it
(the `[eval]` extra).

Failure taxonomy (design "distinct typed cases"): each case ends in exactly one status --
ok / empty / malformed / refusal / timeout / backend_error / retrieval_miss -- recorded
separately, never collapsed into a single "reliability failure".
"""

from typing import Callable, TypedDict

# Terminal case statuses.
OK = "ok"
EMPTY = "empty"
MALFORMED = "malformed"
REFUSAL = "refusal"
RETRIEVAL_MISS = "retrieval_miss"
# transport tokens (timeout / backend_error) are passed through from ChatResult.error.

# Markers a model uses when it declines to answer (UA + EN).
_REFUSAL_MARKERS = (
    "не можу відповісти",
    "не маю можливості",
    "не можу допомогти",
    "вибачте, але я",
    "i cannot answer",
    "i can't answer",
    "i'm unable to",
    "as an ai",
)

SYSTEM_PROMPT = (
    "Ти асистент, який відповідає виключно на основі наданого контексту. "
    "Якщо відповіді немає в контексті, скажи, що інформації недостатньо. "
    "Відповідай стисло українською мовою."
)


class RagState(TypedDict, total=False):
    question: str
    gold_spans: list[dict]
    retrieved: list[dict]
    context: str
    answer: str
    status: str
    error: str | None
    usage: dict


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as a delimited, numbered block (corpus is untrusted input)."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] ({chunk.get('doc_id', '?')})\n{chunk.get('text', '').strip()}")
    return "\n\n".join(parts)


def build_messages(question: str, context: str) -> list[dict]:
    user = (
        f"Контекст:\n<<<\n{context}\n>>>\n\n"
        f"Питання: {question}\n\nВідповідь:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def classify_response(text: str, error: str | None, expect_json: bool = False) -> str:
    """Map a raw model response to a terminal status."""
    if error:
        return error  # "timeout" | "backend_error", passed through verbatim
    if text is None or not text.strip():
        return EMPTY
    stripped = text.strip()
    low = stripped.lower()
    if any(marker in low for marker in _REFUSAL_MARKERS):
        return REFUSAL
    if expect_json:
        import json

        try:
            json.loads(stripped)
        except (ValueError, TypeError):
            return MALFORMED
    return OK


def make_retrieve_node(store, k: int) -> Callable[[RagState], dict]:
    """Closure: retrieve top-k chunks; flag retrieval_miss when nothing comes back."""

    def retrieve(state: RagState) -> dict:
        chunks = store.retrieve(state["question"], k)
        update = {"retrieved": chunks, "context": format_context(chunks)}
        if not chunks:
            update["status"] = RETRIEVAL_MISS
        return update

    return retrieve


def make_generate_node(launcher, max_tokens: int, temperature: float,
                       timeout: float) -> Callable[[RagState], dict]:
    """Closure: call the backend on the retrieved context; classify the response."""

    def generate(state: RagState) -> dict:
        if state.get("status") == RETRIEVAL_MISS:
            return {"answer": "", "usage": {}}  # short-circuit; status already terminal
        messages = build_messages(state["question"], state.get("context", ""))
        result = launcher.chat(
            messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout
        )
        return {
            "answer": result.text or "",
            "status": classify_response(result.text, result.error),
            "error": result.error,
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_s": result.latency_s,
                "tokens_per_s": result.tokens_per_s(),
            },
        }

    return generate


def build_rag_graph(store, launcher, k: int, max_tokens: int, temperature: float,
                    timeout: float):
    """Compile the retrieve -> generate LangGraph app. Needs the `[eval]` extra."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise SystemExit(
            'ERROR: the eval graph needs the [eval] extra. Run: uv pip install -e ".[eval]"'
        ) from exc
    graph = StateGraph(RagState)
    graph.add_node("retrieve", make_retrieve_node(store, k))
    graph.add_node("generate", make_generate_node(launcher, max_tokens, temperature, timeout))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_case(app, question: str, gold_spans: list[dict]) -> RagState:
    """Invoke a compiled graph for one gold item; returns the terminal state."""
    return app.invoke({"question": question, "gold_spans": gold_spans})
