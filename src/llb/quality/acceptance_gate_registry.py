"""Declared classification of experiment count controls."""

from dataclasses import dataclass

MAKE_CONFIG = "make/config.mk"


@dataclass(frozen=True)
class GateDeclaration:
    gate_id: str
    location: str
    symbol: str
    classification: str
    default: str
    justification: str
    decision_role: str


def _make(
    gate_id: str,
    symbol: str,
    classification: str,
    default: str,
    justification: str,
    decision_role: str,
) -> GateDeclaration:
    return GateDeclaration(
        gate_id,
        MAKE_CONFIG,
        symbol,
        classification,
        default,
        justification,
        decision_role,
    )


DECLARATIONS = (
    _make(
        "goldset-output-cap",
        "GOLDSET_N",
        "resource_cap",
        "250",
        "bounds generated development data and provider work",
        "not an acceptance decision",
    ),
    _make(
        "draft-output-cap",
        "DRAFT_MAX_ITEMS",
        "resource_cap",
        "60",
        "bounds drafting calls and artifact size",
        "not an acceptance decision",
    ),
    _make(
        "draft-compare-call-budget",
        "DRAFT_COMPARE_SEEDS",
        "resource_cap",
        "20",
        "bounds exact paired provider calls",
        "reported comparison size, not a pass threshold",
    ),
    _make(
        "frontier-probe-call-budget",
        "FRONTIER_UA_PROBE_SEEDS",
        "resource_cap",
        "12",
        "bounds paid provider calls",
        "probe display sample, not a pass threshold",
    ),
    _make(
        "local-compare-call-budget",
        "LOCAL_DRAFT_COMPARE_SEEDS",
        "resource_cap",
        "12",
        "bounds local comparison runtime",
        "reported comparison size, not a pass threshold",
    ),
    _make(
        "cutoff-fit-search-budget",
        "KNOWLEDGE_CUTOFF_TRIALS",
        "resource_cap",
        "200",
        "bounds optimizer work",
        "search effort does not establish inferential support",
    ),
    _make(
        "pipeline-finalist-cap",
        "PIPELINE_TOP_N",
        "resource_cap",
        "2",
        "bounds expensive finalist evaluation",
        "ranking breadth, not evidence strength",
    ),
    _make(
        "pipeline-search-budget",
        "PIPELINE_TRIALS",
        "resource_cap",
        "20",
        "bounds optimizer work",
        "search effort does not establish inferential support",
    ),
    _make(
        "joint-search-budget",
        "JOINT_SEARCH_TRIALS",
        "resource_cap",
        "20",
        "bounds optimizer work",
        "successor to the retired UA roster target",
    ),
    _make(
        "joint-search-finalist-floor",
        "JOINT_SEARCH_MIN_FINALISTS",
        "safety_cap",
        "2",
        "keeps a comparative final stage",
        "structural guard, not statistical evidence",
    ),
    _make(
        "auto-rag-search-budget",
        "AUTO_RAG_TRIALS",
        "resource_cap",
        "20",
        "bounds optimizer work",
        "search effort does not establish inferential support",
    ),
    _make(
        "auto-rag-finalist-floor",
        "AUTO_RAG_MIN_FINALISTS",
        "safety_cap",
        "2",
        "keeps a comparative final stage",
        "structural guard, not statistical evidence",
    ),
    _make(
        "recommend-nonempty-floor",
        "RECOMMEND_MIN_CASES",
        "safety_cap",
        "1",
        "rejects empty recommendation evidence",
        "structural existence check only",
    ),
    _make(
        "verification-sample-precision",
        "VERIFY_N",
        "inferential_gate",
        "derived",
        "finite-population confidence/precision plan",
        "explicit row override is recorded",
    ),
    _make(
        "chain-verification-sample-precision",
        "CHAIN_VERIFY_N",
        "inferential_gate",
        "derived",
        "reuses the verification confidence/precision plan",
        "explicit row override is recorded",
    ),
    _make(
        "chain-review-retention",
        "CHAIN_MIN_ACCEPTED",
        "inferential_gate",
        "derived",
        "relative retention of the reviewed sample",
        "explicit count override is recorded",
    ),
    _make(
        "quickstart-verification-sample-precision",
        "QUICKSTART_DRAFT_VERIFY_N",
        "inferential_gate",
        "derived",
        "scales with the drafted population",
        "explicit row override is recorded",
    ),
    GateDeclaration(
        "draft-sample-disabled-sentinel",
        "src/llb/cli/prep/draft.py",
        "verification_sample_size",
        "safety_cap",
        "0",
        "zero disables optional worksheet creation",
        "not a quality threshold",
    ),
    GateDeclaration(
        "verification-cli-sample-precision",
        "src/llb/goldset/verify/cli.py",
        "size",
        "inferential_gate",
        "derived",
        "finite-population confidence/precision plan",
        "explicit row override is recorded",
    ),
    GateDeclaration(
        "chain-cli-review-retention",
        "src/llb/goldset/promote_chains.py",
        "min_chains",
        "inferential_gate",
        "derived",
        "relative retention of the reviewed sample",
        "explicit count override is recorded",
    ),
    GateDeclaration(
        "screen-search-budget",
        "src/llb/cli/eval/screen.py",
        "trials",
        "resource_cap",
        "20",
        "bounds stage-one optimizer work",
        "not a quality threshold",
    ),
    GateDeclaration(
        "sweep-search-budget",
        "src/llb/cli/models/sweep.py",
        "trials",
        "resource_cap",
        "30",
        "bounds stage-one optimizer work",
        "not a quality threshold",
    ),
    GateDeclaration(
        "joint-cli-search-budget",
        "src/llb/cli/models/joint_search.py",
        "trials",
        "resource_cap",
        "20",
        "bounds stage-one optimizer work",
        "not a quality threshold",
    ),
    GateDeclaration(
        "auto-rag-cli-search-budget",
        "src/llb/cli/auto_rag.py",
        "trials",
        "resource_cap",
        "20",
        "bounds optimizer work",
        "not a quality threshold",
    ),
    GateDeclaration(
        "finetune-search-budget",
        "src/llb/cli/finetune/training.py",
        "max_trials",
        "resource_cap",
        "8",
        "bounds training time and energy",
        "not a quality threshold",
    ),
    GateDeclaration(
        "cutoff-cli-search-budget",
        "src/llb/cli/bench/knowledge_cutoff.py",
        "optuna_trials",
        "resource_cap",
        "200",
        "bounds decay-fit optimizer work",
        "not a quality threshold",
    ),
    GateDeclaration(
        "cutoff-ua-cli-search-budget",
        "src/llb/cli/bench/knowledge_cutoff_ua.py",
        "optuna_trials",
        "resource_cap",
        "200",
        "bounds decay-fit optimizer work",
        "not a quality threshold",
    ),
)
