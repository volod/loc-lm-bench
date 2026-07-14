"""Walking-skeleton end-to-end (RAG core acceptance), driven by fakes.

Exercises the full vertical -- retrieve -> generate -> classify -> score -> aggregate ->
persist -- without FAISS, langgraph, Ollama, or a GPU, by injecting a fake store, a fake
launcher, and a runner_fn that composes the real eval-graph node closures sequentially.
"""

import json
from pathlib import Path


from llb.backends.base import BackendLauncher, ChatResult
from llb.core.config import RunConfig
from llb.eval import common
from llb.eval import graph
from llb.executor.runner import run_eval
from llb.goldset.schema import GoldItem
from llb.prompt_system.template import PromptPackage, TemplateFields

DOC = "Київ є столицею України. Дніпро тече через місто."


def gold_item(item_id, question, reference, answer_text, split="final"):
    start = DOC.find(answer_text)
    return GoldItem(
        id=item_id,
        lang="uk",
        question=question,
        reference_answer=reference,
        source_doc_id="kyiv.txt",
        source_spans=[
            {
                "doc_id": "kyiv.txt",
                "char_start": start,
                "char_end": start + len(answer_text),
                "text": answer_text,
            }
        ],
        provenance="public-reused",
        verified=True,
        split=split,
    )


class FakeStore:
    """Returns preset chunks per question (doc_id + offsets drive retrieval scoring)."""

    def __init__(self, by_question):
        self._by_question = by_question

    def retrieve(self, question, k):
        return self._by_question.get(question, [])[:k]


class FakeLauncher(BackendLauncher):
    def __init__(self, responder):
        super().__init__(model="fake-uk", meta={"backend": "fake"})
        self._responder = responder

    def chat(self, messages, max_tokens, temperature, timeout):
        return self._responder(messages)


def _runner_fn(store, launcher, cfg):
    retrieve = graph.make_retrieve_node(store, cfg.top_k)
    generate = graph.make_generate_node(
        launcher, cfg.max_tokens, cfg.temperature, cfg.request_timeout_s
    )

    def run(item):
        state = {
            "question": item.question,
            "gold_spans": [s.model_dump() for s in item.source_spans],
        }
        state.update(retrieve(state))
        state.update(generate(state))
        return state

    return run


def test_walking_skeleton_end_to_end(tmp_path):
    hit_q = "Яка столиця України?"
    miss_q = "Що тече через місто?"
    items = [
        gold_item("uk-1", hit_q, "Київ", "Київ"),
        gold_item("uk-2", miss_q, "Дніпро", "Дніпро"),
    ]
    store = FakeStore(
        {
            # uk-1: chunk overlaps the gold span (doc kyiv.txt, 0..4) -> hit + correct answer
            hit_q: [{"doc_id": "kyiv.txt", "char_start": 0, "char_end": 24, "text": DOC[:24]}],
            # uk-2: chunk is a different doc -> retrieval miss
            miss_q: [{"doc_id": "other.txt", "char_start": 0, "char_end": 30, "text": "noise"}],
        }
    )

    def responder(messages):
        content = messages[-1]["content"]
        if "столиц" in content:  # only uk-1's context/question mentions the capital
            return ChatResult(text="Київ", completion_tokens=2, latency_s=0.4)
        return ChatResult(text="", completion_tokens=0, latency_s=0.2)  # -> empty

    launcher = FakeLauncher(responder)
    cfg = RunConfig(data_dir=tmp_path, run_name="skeleton-test", top_k=3, model="fake-uk")

    result = run_eval(
        cfg,
        items=items,
        store=store,
        launcher=launcher,
        runner_fn=_runner_fn(store, launcher, cfg),
        mirror=lambda *a: None,
        emit=False,
    )

    # One ranked row, ranked #1, both cases counted.
    rows = result["rows"]
    assert len(rows) == 1 and rows[0]["rank"] == 1 and rows[0]["n_cases"] == 2
    assert rows[0]["model"] == "fake-uk"

    # objective = mean(f1=1.0 for uk-1, f1=0.0 for uk-2) = 0.5; reliability 0.5 (one empty)
    assert result["metrics"]["objective_score"] == 0.5
    assert result["metrics"]["reliability"] == 0.5
    assert result["manifest"].split == "final"

    # retrieval: uk-1 hits, uk-2 misses -> recall 0.5
    assert result["retrieval"]["recall_at_k"] == 0.5

    # canonical record on disk: scores.jsonl (single format, independent of installed extras)
    run_dir = cfg.run_dir(result["run_timestamp"])
    assert run_dir == Path(result["paths"]["manifest"]).parent
    assert (run_dir / "manifest.json").exists()
    assert any(run_dir.glob("scores.*"))


def test_build_messages_applies_prompt_system_package():
    pkg = PromptPackage(
        system_prompt="SYS PROMPT",
        additional_prompt="AUGMENTED KNOWLEDGE",
        fields=TemplateFields(),
        dropped_context={"budget_tokens": 10, "used_tokens": 1, "sections": []},
    )
    messages = graph.build_messages("Питання?", "BASE CONTEXT", pkg)

    assert messages[0]["content"].startswith("SYS PROMPT")
    assert graph.SYSTEM_PROMPT in messages[0]["content"]
    assert "AUGMENTED KNOWLEDGE" in messages[1]["content"]
    assert "BASE CONTEXT" in messages[1]["content"]


def test_run_eval_persists_prompt_system_provenance(tmp_path):
    q = "Яка столиця України?"
    items = [gold_item("uk-1", q, "Київ", "Київ")]
    cfg = RunConfig(data_dir=tmp_path, run_name="prompt-system", model="fake-uk")
    provenance = {
        "prompt_system_id": "ps-test",
        "corpus_digest": "corpus",
        "mapping_digest": "mapping",
        "template_revision": "template",
        "tokenizer": "char-ratio",
        "context_window": 4096,
        "prompt_budget_tokens": 3000,
    }
    result = run_eval(
        cfg,
        items=items,
        launcher=FakeLauncher(lambda messages: ChatResult(text="Київ")),
        runner_fn=lambda item: {
            "answer": "Київ",
            "status": common.OK,
            "retrieved": [],
            "usage": {},
        },
        prompt_system_provenance=provenance,
        mirror=lambda *a: None,
        emit=False,
    )

    manifest = result["manifest"]
    assert manifest.config["prompt_system"] == "ps-test"
    assert manifest.prompt_system_provenance == provenance
    persisted = json.loads(Path(result["paths"]["manifest"]).read_text(encoding="utf-8"))
    assert persisted["config"]["prompt_system"] == "ps-test"
    assert persisted["prompt_system_provenance"] == provenance
