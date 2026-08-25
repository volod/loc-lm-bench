"""Walking-skeleton end-to-end (RAG core acceptance), driven by fakes.

Exercises the full vertical -- retrieve -> generate -> classify -> score -> aggregate ->
persist -- without FAISS, langgraph, Ollama, or a GPU, by injecting a fake store, a fake
launcher, and a runner_fn that composes the real eval-graph node closures sequentially.
"""

import json
from pathlib import Path

import pytest


from llb.backends.base import BackendLauncher, ChatResult
from llb.core.config import RunConfig
from llb.eval import common
from llb.eval import graph
from llb.executor.runner import ITEM_GROUNDING_DRAFTED, run_eval
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


def test_drafted_items_are_skipped_unless_asked_for_and_then_marked_in_the_manifest(tmp_path):
    """A drafted (unverified) ledger scores only on request, and says so in its own bundle."""
    q = "Яка столиця України?"
    items = [gold_item("uk-1", q, "Київ", "Київ")]
    items[0].verified = False
    cfg = RunConfig(data_dir=tmp_path, run_name="drafted", model="fake-uk")
    kwargs = {
        "items": items,
        "launcher": FakeLauncher(lambda messages: ChatResult(text="Київ")),
        "runner_fn": lambda item: {
            "answer": "Київ",
            "status": common.OK,
            "retrieved": [],
            "usage": {},
        },
        "mirror": lambda *a: None,
        "emit": False,
    }
    with pytest.raises(SystemExit, match="no verified"):
        run_eval(cfg, **kwargs)

    result = run_eval(cfg, verified_only=False, **kwargs)
    assert result["manifest"].config["item_grounding"] == ITEM_GROUNDING_DRAFTED
    persisted = json.loads(Path(result["paths"]["manifest"]).read_text(encoding="utf-8"))
    assert persisted["config"]["item_grounding"] == ITEM_GROUNDING_DRAFTED


def test_response_guard_flags_reach_the_persisted_bundle(tmp_path):
    """Leaked reasoning and off-language answers are named per-case, not scored as content.

    The three fake generations are the shapes measured on this host (see the roster suppression
    verdicts in the backend-telemetry docs): a bounded-budget leak with no terminator, a leak whose
    bare `</think>` survived the chat template, and a clean Ukrainian answer. All three are `ok` by
    status -- which is exactly why the guard exists.
    """
    leak_q = "Що таке авторське право?"
    tail_q = "Скільки років діють майнові права?"
    clean_q = "Яка столиця України?"
    items = [
        gold_item("uk-1", leak_q, "Київ", "Київ"),
        gold_item("uk-2", tail_q, "Київ", "Київ"),
        gold_item("uk-3", clean_q, "Київ", "Київ"),
    ]
    answers = {
        leak_q: "Okay, I need to explain what copyright is. First, I need to check the context.",
        tail_q: "Okay, let's see. The context states the term of the rights.\n</think>\n\nКиїв",
        clean_q: "Київ",
    }
    cfg = RunConfig(data_dir=tmp_path, run_name="guard-test", model="fake-uk")

    def runner_fn(item):
        return {
            "answer": answers[item.question],
            "status": common.OK,
            "retrieved": [],
            "usage": {"completion_tokens": 32},
        }

    result = run_eval(
        cfg,
        items=items,
        launcher=FakeLauncher(lambda messages: ChatResult(text="")),
        runner_fn=runner_fn,
        mirror=lambda *a: None,
        emit=False,
    )

    run_dir = cfg.run_dir(result["run_timestamp"])
    rows = {
        json.loads(line)["item_id"]: json.loads(line)
        for line in (run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    # Every case stays `ok`: the guard is additive and never rewrites the failure taxonomy.
    assert {row["status"] for row in rows.values()} == {common.OK}

    assert rows["uk-1"]["reasoning_leak"] and rows["uk-1"]["reasoning_leak_marker"] == "okay,"
    assert rows["uk-1"]["reasoning_leak_chars"] == len(answers[leak_q])
    assert rows["uk-1"]["answer_language"] == "en" and rows["uk-1"]["language_mismatch"]

    assert rows["uk-2"]["reasoning_leak"] and rows["uk-2"]["reasoning_leak_marker"] == "</think>"
    assert rows["uk-2"]["reasoning_leak_chars"] < len(answers[tail_q])

    assert not rows["uk-3"]["reasoning_leak"]
    assert rows["uk-3"]["reasoning_leak_chars"] == 0
    assert rows["uk-3"]["answer_language"] == "uk"
    assert not rows["uk-3"]["language_mismatch"]

    # The run-level rates share `reliability`'s denominator, and both reach the manifest on disk.
    metrics = result["metrics"]
    assert metrics["reliability"] == 1.0
    assert metrics["reasoning_leak_rate"] == round(2 / 3, 4)
    assert metrics["language_mismatch_rate"] == round(2 / 3, 4)
    assert metrics["mean_reasoning_leak_chars"] > 0
    persisted = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["metrics"]
    assert persisted["reasoning_leak_rate"] == metrics["reasoning_leak_rate"]
    assert persisted["language_mismatch_rate"] == metrics["language_mismatch_rate"]


def test_suppress_reasoning_prompt_appends_the_instruction_to_the_system_message():
    """The prompt-level lever is off by default and composes with a prompt package."""
    plain = graph.build_messages("Питання?", "CTX")
    guarded = graph.build_messages("Питання?", "CTX", suppress_reasoning=True)
    instruction = graph.render_text(graph.NO_REASONING_TEMPLATE)

    assert instruction not in plain[0]["content"]
    assert guarded[0]["content"].startswith(graph.SYSTEM_PROMPT)
    assert guarded[0]["content"].endswith(instruction)
    assert plain[1] == guarded[1]  # the user message is untouched

    pkg = PromptPackage(
        system_prompt="SYS PROMPT",
        additional_prompt="",
        fields=TemplateFields(),
        dropped_context={"budget_tokens": 10, "used_tokens": 1, "sections": []},
    )
    both = graph.build_messages("Питання?", "CTX", pkg, suppress_reasoning=True)
    assert both[0]["content"].startswith("SYS PROMPT")
    assert both[0]["content"].endswith(instruction)
