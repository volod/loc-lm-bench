"""Turn the profile's measured fields back into the commands that reproduce them.

A recommendation an operator has to hand-translate into flags is a recommendation they will
mistype. Only `measured` fields contribute: a demoted, refused, or unmeasured field is listed as
omitted with its state, so the replay command is exactly the configuration the evidence supports
and nothing else.
"""

from llb.board.agent_profile.model import (
    FIELD_ADAPTER,
    FIELD_BACKEND,
    FIELD_CONTEXT_BUDGET,
    FIELD_CONTEXT_ORDER,
    FIELD_CONTEXT_POLICY,
    FIELD_LOOP_POLICY,
    FIELD_MODEL,
    FIELD_PROMPT_SYSTEM,
    FIELD_RERANKER,
    FIELD_TOP_K,
    STATE_MEASURED,
    AgentProfile,
)
from llb.board.agent_profile.sources_agent import ADAPTER_NONE
from llb.core.contracts.common import JsonObject
from llb.rag.rerank_bakeoff.models import ROW_NO_RERANK

# `llb bench-agentic-loop` validates that the grid contains the mandatory legacy cell, so a pinned
# replay of a recommended cell has to carry the baseline alongside it.
BASELINE_MAX_STEPS = 6
BASELINE_MALFORMED_POLICY = "answer"
BASELINE_REPEATED_CALL_POLICY = "allow"

# The run-eval option name for each retrieval-side field it accepts.
_RUN_EVAL_FLAGS = {
    FIELD_MODEL: "--model",
    FIELD_BACKEND: "--backend",
    FIELD_TOP_K: "--top-k",
    FIELD_CONTEXT_BUDGET: "--context-budget",
    FIELD_RERANKER: "--reranker",
    FIELD_CONTEXT_ORDER: "--context-order",
    FIELD_PROMPT_SYSTEM: "--prompt-system",
    FIELD_ADAPTER: "--adapter",
}
# A field whose measured answer is "nothing here" contributes no flag: the run-eval default IS off.
_OFF_VALUES = {FIELD_RERANKER: ROW_NO_RERANK, FIELD_ADAPTER: ADAPTER_NONE}


def _measured(profile: AgentProfile, name: str) -> object | None:
    item = profile.by_name(name)
    if item.state != STATE_MEASURED or item.value is None:
        return None
    return None if _OFF_VALUES.get(name) == item.value else item.value


def _replayable(profile: AgentProfile) -> bool:
    """No measured model means no replay at all.

    A command that pins `--top-k` but not the model does not reproduce the recommended
    configuration -- it reproduces whatever the caller's config already said, wearing one knob from
    this profile. That is exactly the silent mixing the profile exists to prevent.
    """
    return _measured(profile, FIELD_MODEL) is not None


def run_eval_flags(profile: AgentProfile) -> list[str]:
    """`llb run-eval` flags for every measured retrieval-side field."""
    if not _replayable(profile):
        return []
    flags: list[str] = []
    for name, flag in _RUN_EVAL_FLAGS.items():
        value = _measured(profile, name)
        if value is not None:
            flags += [flag, str(value)]
    return flags


def bench_agentic_flags(profile: AgentProfile) -> list[str]:
    """`llb bench-agentic` flags: the served model plus the context policy and step budget."""
    if not _replayable(profile):
        return []
    flags: list[str] = []
    for name, flag in ((FIELD_MODEL, "--model"), (FIELD_BACKEND, "--backend")):
        value = _measured(profile, name)
        if value is not None:
            flags += [flag, str(value)]
    policy = _measured(profile, FIELD_CONTEXT_POLICY)
    if policy is not None:
        flags += ["--context-policy", str(policy)]
    loop = _measured(profile, FIELD_LOOP_POLICY)
    if isinstance(loop, dict) and loop.get("max_steps") is not None:
        flags += ["--max-steps", str(loop["max_steps"])]
    return flags


def bench_agentic_loop_flags(profile: AgentProfile) -> list[str]:
    """`llb bench-agentic-loop` flags pinning the recommended cell against its mandatory baseline.

    The sweep refuses a grid without the legacy cell, so the recommended values are emitted as a
    two-point grid whenever they differ from it -- which is also what confirms the recommendation
    rather than merely restating it.
    """
    loop = _measured(profile, FIELD_LOOP_POLICY)
    if not isinstance(loop, dict) or not _replayable(profile):
        return []
    flags: list[str] = []
    for name, flag in ((FIELD_MODEL, "--model"), (FIELD_BACKEND, "--backend")):
        value = _measured(profile, name)
        if value is not None:
            flags += [flag, str(value)]
    axes = (
        ("--agent-max-steps", loop.get("max_steps"), BASELINE_MAX_STEPS),
        ("--agent-malformed-policy", loop.get("malformed_call_policy"), BASELINE_MALFORMED_POLICY),
        (
            "--agent-repeated-call-policy",
            loop.get("repeated_call_policy"),
            BASELINE_REPEATED_CALL_POLICY,
        ),
    )
    for flag, value, baseline in axes:
        if value is None:
            continue
        points = [str(baseline)] if str(value) == str(baseline) else [str(baseline), str(value)]
        flags += [flag, ",".join(points)]
    return flags


def replay_block(profile: AgentProfile) -> JsonObject:
    """Every replay command plus the fields that could not contribute one, with their state."""
    return {
        "run_eval": run_eval_flags(profile),
        "bench_agentic": bench_agentic_flags(profile),
        "bench_agentic_loop": bench_agentic_loop_flags(profile),
        "omitted": [
            {"field": item.name, "state": item.state}
            for item in profile.fields
            if item.state != STATE_MEASURED
        ],
    }


def replay_commands(profile: AgentProfile) -> list[str]:
    """The three replay commands as copy-pasteable one-liners; a command with no flags is dropped."""
    block = replay_block(profile)
    commands = [
        ("llb run-eval", block["run_eval"]),
        ("llb bench-agentic", block["bench_agentic"]),
        ("llb bench-agentic-loop", block["bench_agentic_loop"]),
    ]
    return [f"{name} {' '.join(flags)}" for name, flags in commands if flags]
