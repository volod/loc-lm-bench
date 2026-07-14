"""Tests for chain context summary."""

import re
from llb.bench import chain_context_policy as policy
from test_chain_context import FakeRetriever, _chunk, _three_step_chain


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

    policy.run_policy(
        [chain],
        policy.POLICY_SUMMARY,
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
