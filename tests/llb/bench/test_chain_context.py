"""Context-policy benchmark -- context assembly per policy + scored run over a fake endpoint."""

import re


from llb.bench import chain_context_policy as policy
from llb.goldset.chains import ChainItem, ChainStep


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
    tid1, p1 = policy.assemble_prompt(
        policy.POLICY_FRESH,
        index=0,
        n_steps=2,
        question="Q1?",
        retrieved=retrieved,
        prior_qa=[],
        running_summary="",
    )
    tid2, p2 = policy.assemble_prompt(
        policy.POLICY_FRESH,
        index=1,
        n_steps=2,
        question="Q2?",
        retrieved=retrieved,
        prior_qa=[("Q1?", "A1")],
        running_summary="",
    )
    assert tid1 == tid2 == policy.INSTRUCTION_DEFAULT
    # even on step 2 with prior answers, fresh carries NO history into the prompt
    assert "Попередні кроки" not in p2
    assert "A1" not in p2
    assert "Питання: Q2?" in p2


def test_history_accumulates_prior_qa():
    _, prompt = policy.assemble_prompt(
        policy.POLICY_HISTORY,
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
    _, prompt = policy.assemble_prompt(
        policy.POLICY_SUMMARY,
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
    assert policy.role_for_step(0, 3) == policy.ROLE_LIBRARIAN
    assert policy.role_for_step(1, 3) == policy.ROLE_ANALYST
    assert policy.role_for_step(2, 3) == policy.ROLE_ANSWERER
    # 2-step chain skips the analyst: librarian -> answerer
    assert policy.role_for_step(0, 2) == policy.ROLE_LIBRARIAN
    assert policy.role_for_step(1, 2) == policy.ROLE_ANSWERER
    tid0, p0 = policy.assemble_prompt(
        policy.POLICY_ROLES,
        index=0,
        n_steps=2,
        question="Q1?",
        retrieved=[_chunk("c")],
        prior_qa=[],
        running_summary="",
    )
    tid1, p1 = policy.assemble_prompt(
        policy.POLICY_ROLES,
        index=1,
        n_steps=2,
        question="Q2?",
        retrieved=[_chunk("c")],
        prior_qa=[("Q1?", "A1")],
        running_summary="",
    )
    assert tid0 == policy.ROLE_LIBRARIAN and tid1 == policy.ROLE_ANSWERER
    assert "бібліотекар" in p0
    assert "відповідач" in p1
    assert "Попередні кроки:" in p1  # roles also carry the accumulated history


def test_empty_retrieval_marks_no_context():
    _, prompt = policy.assemble_prompt(
        policy.POLICY_FRESH,
        index=0,
        n_steps=1,
        question="Q?",
        retrieved=[],
        prior_qa=[],
        running_summary="",
    )
    assert policy._NO_CONTEXT in prompt


def test_prompt_system_ids_per_policy():
    assert policy.prompt_system_ids(policy.POLICY_ROLES) == [
        policy.ROLE_LIBRARIAN,
        policy.ROLE_ANALYST,
        policy.ROLE_ANSWERER,
    ]
    for policy_name in (policy.POLICY_FRESH, policy.POLICY_HISTORY, policy.POLICY_SUMMARY):
        assert policy.prompt_system_ids(policy_name) == [policy.INSTRUCTION_DEFAULT]


# --- summary threading ---------------------------------------------------------------------


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


# --- board + recommend loading -------------------------------------------------------------
