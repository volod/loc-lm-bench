"""Context-policy benchmark -- context assembly per policy + scored run over a fake endpoint."""

import re

import pytest

from llb.bench import chain_context as cc
from llb.goldset.chains import ChainItem, ChainStep
from llb.scoring.aggregate import TIER_CHAIN_CONTEXT


def _chunk(text, doc_id="d.md", start=0):
    return {"doc_id": doc_id, "char_start": start, "char_end": start + len(text), "text": text}


class FakeRetriever:
    """Returns a fixed chunk list; records the queries it was asked (retrieval is held constant)."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.queries: list[str] = []

    def retrieve(self, question, k):
        self.queries.append(question)
        return list(self.chunks)[:k]


def _two_step_chain(chain_id="chain-0"):
    return ChainItem(
        chain_id=chain_id,
        steps=[
            ChainStep(
                order=1,
                question="Q1?",
                reference_answer="Факт один.",
                source_doc_id="d.md",
                source_spans=[
                    {"doc_id": "d.md", "char_start": 0, "char_end": 10, "text": "Факт один."}
                ],
            ),
            ChainStep(
                order=2,
                question="Q2?",
                reference_answer="Факт два.",
                source_doc_id="d.md",
                source_spans=[
                    {"doc_id": "d.md", "char_start": 10, "char_end": 19, "text": "Факт два."}
                ],
                dependency_note="крок 1",
            ),
        ],
    )


def _three_step_chain(chain_id="chain-3"):
    steps = [
        ChainStep(
            order=i + 1,
            question=f"Q{i + 1}?",
            reference_answer=f"Відповідь {i + 1}.",
            source_doc_id="d.md",
            source_spans=[{"doc_id": "d.md", "char_start": i, "char_end": i + 3, "text": "abc"}],
            dependency_note="dep" if i > 0 else "",
        )
        for i in range(3)
    ]
    return ChainItem(chain_id=chain_id, steps=steps)


# --- context assembly (the four policies) --------------------------------------------------


def test_fresh_has_no_prior_block():
    retrieved = [_chunk("контекст")]
    tid1, p1 = cc.assemble_prompt(
        cc.POLICY_FRESH,
        index=0,
        n_steps=2,
        question="Q1?",
        retrieved=retrieved,
        prior_qa=[],
        running_summary="",
    )
    tid2, p2 = cc.assemble_prompt(
        cc.POLICY_FRESH,
        index=1,
        n_steps=2,
        question="Q2?",
        retrieved=retrieved,
        prior_qa=[("Q1?", "A1")],
        running_summary="",
    )
    assert tid1 == tid2 == cc.INSTRUCTION_DEFAULT
    # even on step 2 with prior answers, fresh carries NO history into the prompt
    assert "Попередні кроки" not in p2
    assert "A1" not in p2
    assert "Питання: Q2?" in p2


def test_history_accumulates_prior_qa():
    _, prompt = cc.assemble_prompt(
        cc.POLICY_HISTORY,
        index=1,
        n_steps=2,
        question="Q2?",
        retrieved=[_chunk("c")],
        prior_qa=[("Q1?", "A1")],
        running_summary="",
    )
    assert "Попередні кроки:" in prompt
    assert "Крок 1 — Питання: Q1?" in prompt
    assert "Крок 1 — Відповідь: A1" in prompt
    assert prompt.index("Попередні кроки") < prompt.index("Питання: Q2?")


def test_summary_uses_running_summary_not_transcript():
    _, prompt = cc.assemble_prompt(
        cc.POLICY_SUMMARY,
        index=1,
        n_steps=2,
        question="Q2?",
        retrieved=[_chunk("c")],
        prior_qa=[("Q1?", "A1")],
        running_summary="Короткий підсумок.",
    )
    assert "Підсумок попередніх кроків:" in prompt
    assert "Короткий підсумок." in prompt
    assert "Крок 1 — Відповідь" not in prompt  # summary replaces the full transcript


def test_roles_rotate_system_prompt():
    # 3-step chain: librarian -> analyst -> answerer
    assert cc.role_for_step(0, 3) == cc.ROLE_LIBRARIAN
    assert cc.role_for_step(1, 3) == cc.ROLE_ANALYST
    assert cc.role_for_step(2, 3) == cc.ROLE_ANSWERER
    # 2-step chain skips the analyst: librarian -> answerer
    assert cc.role_for_step(0, 2) == cc.ROLE_LIBRARIAN
    assert cc.role_for_step(1, 2) == cc.ROLE_ANSWERER
    tid0, p0 = cc.assemble_prompt(
        cc.POLICY_ROLES,
        index=0,
        n_steps=2,
        question="Q1?",
        retrieved=[_chunk("c")],
        prior_qa=[],
        running_summary="",
    )
    tid1, p1 = cc.assemble_prompt(
        cc.POLICY_ROLES,
        index=1,
        n_steps=2,
        question="Q2?",
        retrieved=[_chunk("c")],
        prior_qa=[("Q1?", "A1")],
        running_summary="",
    )
    assert tid0 == cc.ROLE_LIBRARIAN and tid1 == cc.ROLE_ANSWERER
    assert "бібліотекар" in p0
    assert "відповідач" in p1
    assert "Попередні кроки:" in p1  # roles also carry the accumulated history


def test_empty_retrieval_marks_no_context():
    _, prompt = cc.assemble_prompt(
        cc.POLICY_FRESH,
        index=0,
        n_steps=1,
        question="Q?",
        retrieved=[],
        prior_qa=[],
        running_summary="",
    )
    assert cc._NO_CONTEXT in prompt


def test_prompt_system_ids_per_policy():
    assert cc.prompt_system_ids(cc.POLICY_ROLES) == [
        cc.ROLE_LIBRARIAN,
        cc.ROLE_ANALYST,
        cc.ROLE_ANSWERER,
    ]
    for policy in (cc.POLICY_FRESH, cc.POLICY_HISTORY, cc.POLICY_SUMMARY):
        assert cc.prompt_system_ids(policy) == [cc.INSTRUCTION_DEFAULT]


# --- summary threading ---------------------------------------------------------------------


def test_summary_policy_threads_a_fresh_summary_each_step():
    """The running summary is refreshed by a model call after each step and fed to the next."""
    chain = _three_step_chain()
    retriever = FakeRetriever([_chunk("контекст")])
    seen_summaries: list[str] = []

    def complete(prompt):
        if prompt.startswith("Стисло підсумуй"):
            return "SUMMARY-CALL"
        # record whether a summary was present in the step prompt
        m = re.search(r"Підсумок попередніх кроків:\n(.+)\n", prompt)
        seen_summaries.append(m.group(1) if m else "")
        return "відповідь"

    cc.run_policy(
        [chain],
        cc.POLICY_SUMMARY,
        model="m",
        backend="ollama",
        retriever=retriever,
        complete=complete,
        k=2,
    )
    # step 1 has no summary; steps 2 and 3 see the model-written summary
    assert seen_summaries[0] == ""
    assert seen_summaries[1] == "SUMMARY-CALL"
    assert seen_summaries[2] == "SUMMARY-CALL"


# --- scored run ----------------------------------------------------------------------------


def _grounded_complete(chains):
    """A fake model that answers with the reference ONLY when prior context is present, so
    fresh scores lower than the carryover policies on step >= 2 (ranking must discriminate)."""
    ref_by_q = {step.question: step.reference_answer for c in chains for step in c.steps}
    first_questions = {c.steps[0].question for c in chains}

    def complete(prompt):
        if prompt.startswith("Стисло підсумуй"):
            return "підсумок"
        m = re.search(r"Питання: (.+?)\n", prompt)
        question = m.group(1) if m else ""
        has_prior = "Попередні кроки:" in prompt or "Підсумок попередніх кроків:" in prompt
        # answers correctly on a first step (no dependency) or whenever prior context is carried;
        # a later step WITHOUT carryover (fresh) cannot recover the dependent answer
        if question in first_questions or has_prior:
            return ref_by_q.get(question, "не знаю")
        return "не знаю"

    return complete


def test_run_chain_context_ranks_policies_and_records_provenance(tmp_path):
    chains = [_two_step_chain("chain-0"), _two_step_chain("chain-1")]
    retriever = FakeRetriever([_chunk("контекст")])
    run = cc.run_chain_context(
        chains,
        model="cand",
        backend="ollama",
        retriever=retriever,
        complete=_grounded_complete(chains),
        data_dir=tmp_path,
    )
    # one report + one ranked row per policy
    assert {r.policy for r in run.reports} == set(cc.CONTEXT_POLICIES)
    assert len(run.board) == len(cc.CONTEXT_POLICIES)
    # each report scores every step, and final objectives = one per chain
    for report in run.reports:
        assert len(report.step_rows) == 4  # 2 chains x 2 steps
        assert len(report.final_objectives) == 2
        assert report.final_ci is not None  # >= 2 points -> a bootstrap CI exists
    # carryover policies beat fresh on the final step (which needs step-1 context)
    final = {r.policy: sum(r.final_objectives) / len(r.final_objectives) for r in run.reports}
    assert final[cc.POLICY_HISTORY] > final[cc.POLICY_FRESH]
    # the winning row is a carryover policy, and the recommendation names it
    assert run.board[0]["model"] != cc.POLICY_FRESH
    assert run.board[0]["model"] in run.recommendation
    # persisted provenance records policy, prompt-system ids, and the chain digest
    import json
    from pathlib import Path

    for report in run.reports:
        manifest = json.loads(Path(report.paths["manifest"]).read_text(encoding="utf-8"))
        config = manifest["config"]
        assert config["tier"] == TIER_CHAIN_CONTEXT
        assert config["policy"] == report.policy
        assert config["prompt_system_ids"] == cc.prompt_system_ids(report.policy)
        assert config["chain_set_digest"] == run.chain_digest


def test_run_chain_context_rejects_unknown_policy(tmp_path):
    with pytest.raises(SystemExit):
        cc.run_chain_context(
            [_two_step_chain()],
            model="m",
            backend="ollama",
            retriever=FakeRetriever([_chunk("c")]),
            complete=lambda p: "x",
            policies=["fresh", "bogus"],
            persist=False,
        )


def test_run_chain_context_empty_chains_raises():
    with pytest.raises(SystemExit):
        cc.run_chain_context(
            [],
            model="m",
            backend="ollama",
            retriever=FakeRetriever([]),
            complete=lambda p: "x",
            persist=False,
        )


def test_data_verified_requires_verification_ref(tmp_path):
    with pytest.raises(ValueError):
        cc.run_chain_context(
            [_two_step_chain()],
            model="m",
            backend="ollama",
            retriever=FakeRetriever([_chunk("c")]),
            complete=lambda p: "x",
            persist=False,
            data_verified=True,
            verification_ref=None,
        )


def test_chain_set_digest_is_content_sensitive():
    a = [_two_step_chain("chain-0")]
    b = [_two_step_chain("chain-9")]
    assert cc.chain_set_digest(a) != cc.chain_set_digest(b)
    assert cc.chain_set_digest(a) == cc.chain_set_digest([_two_step_chain("chain-0")])


# --- board + recommend loading -------------------------------------------------------------


def test_board_and_recommend_load_persisted_policies(tmp_path):
    from llb.board.chain_context import chain_context_comparison, load_chain_context_records
    from llb.board.recommend import format_chain_context_section_md, latest_chain_context

    chains = [_two_step_chain("chain-0"), _two_step_chain("chain-1")]
    cc.run_chain_context(
        chains,
        model="cand",
        backend="ollama",
        retriever=FakeRetriever([_chunk("контекст")]),
        complete=_grounded_complete(chains),
        data_dir=tmp_path,
    )
    records = load_chain_context_records(tmp_path)
    assert {(r.model, r.policy) for r in records} == {("cand", p) for p in cc.CONTEXT_POLICIES}
    rows, table, policies = chain_context_comparison(tmp_path, "cand")
    assert {row["model"] for row in rows} == set(cc.CONTEXT_POLICIES)
    assert sorted(policies) == sorted(cc.CONTEXT_POLICIES)
    assert "policy:" in table  # ranking-policy note rendered
    # fresh cannot be the top row (it loses the dependent final step)
    top = next(row for row in rows if row["rank"] == 1)
    assert top["model"] != cc.POLICY_FRESH

    section = format_chain_context_section_md(latest_chain_context(tmp_path))
    assert "## Context policy" in section
    assert "cand" in section


def test_recommend_section_empty_without_bundles(tmp_path):
    from llb.board.recommend import format_chain_context_section_md, latest_chain_context

    assert latest_chain_context(tmp_path) is None
    assert format_chain_context_section_md(None) == ""
