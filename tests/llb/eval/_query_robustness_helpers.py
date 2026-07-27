"""Deterministic variants, fake end-to-end lanes, aggregation, and probe persistence."""

from llb.backends.base import ChatResult
from llb.eval import graph as eval_graph
from llb.goldset.schema import GoldItem


def _item() -> GoldItem:
    return GoldItem(
        id="q1",
        lang="uk",
        question="Який закон?",
        reference_answer="відповідь",
        source_doc_id="doc",
        source_spans=[{"doc_id": "doc", "char_start": 0, "char_end": 9, "text": "відповідь"}],
        provenance="human-authored",
        verified=True,
        split="final",
    )


APOSTROPHE_QUESTION = "Який закон про пам'ять?"


class FakeStore:
    def __init__(self, *questions: str) -> None:
        self.questions = {question.casefold() for question in (questions or (_item().question,))}
        self.chunk = {
            "doc_id": "doc",
            "char_start": 0,
            "char_end": 20,
            "text": "який закон відповідь",
            "rank": 1,
            "retrieval_score": 1.0,
        }
        self.chunks = [self.chunk]

    def retrieve(self, question: str, k: int) -> list[dict[str, object]]:
        return [self.chunk] if question.casefold() in self.questions else []


class FakeEndpoint:
    def chat(self, messages, max_tokens, temperature, timeout) -> ChatResult:
        return ChatResult(text="відповідь", latency_s=0.01)


class FakeGraphApp:
    """Run the production graph's pure nodes without the optional LangGraph package."""

    def __init__(self, retrieve, generate) -> None:
        self.retrieve = retrieve
        self.generate = generate

    def invoke(self, state):
        retrieved = {**state, **self.retrieve(state)}
        return {**retrieved, **self.generate(retrieved)}


def build_fake_graph(
    store,
    launcher,
    k,
    max_tokens,
    temperature,
    timeout,
    prompt_package=None,
    context_order="rank",
    query_prep=None,
    chunk_filter=None,
    cited=False,
):
    retrieve = eval_graph.make_retrieve_node(store, k, context_order, query_prep, chunk_filter)
    generate = eval_graph.make_generate_node(
        launcher, max_tokens, temperature, timeout, prompt_package, cited
    )
    return FakeGraphApp(retrieve, generate)
