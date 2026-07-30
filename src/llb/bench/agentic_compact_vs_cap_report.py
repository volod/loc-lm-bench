"""Report contract and verdict for active compact versus observation_cap evidence."""

from dataclasses import dataclass

from llb.bench.agentic_context_report import (
    METRIC_COMPLETION,
    METRIC_TOTAL_MODEL_INPUT_TOKENS,
    PolicyReport,
)
from llb.rag.fusion_evidence.evidence_gate import (
    DEFAULT_CONFIDENCE,
    READING_FLAT,
    READING_SEPARATED,
    apply_evidence_gate,
)
from llb.rag.fusion_evidence.paired import PairedComparison

METHOD = "agentic-compact-vs-cap"
VERDICT_PREFER_COMPACT = "prefer_compact"
VERDICT_PREFER_CAP = "prefer_cap"
VERDICT_STILL_TIED = "still_tied"
VERDICT_INACTIVE = "inactive"

PAIRED_METRICS = (METRIC_COMPLETION, METRIC_TOTAL_MODEL_INPUT_TOKENS, "n_model_calls")


@dataclass(slots=True)
class CompactVsCapRun:
    """Focused comparison result with an explicit active-compaction gate."""

    model: str
    backend: str
    cap: PolicyReport
    compact: PolicyReport
    paired: dict[str, PairedComparison]
    verdict: str
    reason: str
    table: str
    task_set_digest: str
    max_prompt_chars: int
    n_compactions: int
    n_compacted_episodes: int


def _reading(comparison: PairedComparison) -> str:
    delta = comparison["delta"]
    raw = READING_SEPARATED if delta["lo"] > 0.0 or delta["hi"] < 0.0 else READING_FLAT
    return apply_evidence_gate(
        raw,
        discordant=comparison["wins"] + comparison["losses"],
        confidence=DEFAULT_CONFIDENCE,
    )


def _delta(comparison: PairedComparison) -> str:
    value = comparison["delta"]
    return f"{value['mean']:+.3f} [{value['lo']:+.3f}, {value['hi']:+.3f}]"


def decide_verdict(
    cap: PolicyReport,
    compact: PolicyReport,
    paired: dict[str, PairedComparison],
) -> tuple[str, str]:
    """Prefer a policy only on active, paired completion/cost evidence."""
    n_compactions = int(sum(compact.vector("n_compactions")))
    if n_compactions == 0:
        return (
            VERDICT_INACTIVE,
            "compact never fired; lengthen the transcript or tighten the prompt budget",
        )

    completion = paired[METRIC_COMPLETION]
    completion_reading = _reading(completion)
    if completion_reading == READING_SEPARATED:
        if completion["delta"]["mean"] > 0:
            return (
                VERDICT_PREFER_COMPACT,
                f"compact separates on completion ({_delta(completion)}) after compaction fired",
            )
        return (
            VERDICT_PREFER_CAP,
            f"compact separates worse on completion ({_delta(completion)})",
        )
    if compact.n_context_overflow > cap.n_context_overflow:
        return (
            VERDICT_PREFER_CAP,
            "completion is not separated and compact causes more context overflows",
        )

    cost = paired[METRIC_TOTAL_MODEL_INPUT_TOKENS]
    if _reading(cost) == READING_SEPARATED:
        if cost["delta"]["mean"] < 0:
            return (
                VERDICT_PREFER_COMPACT,
                "completion is not separated and compact uses fewer total model-input tokens "
                f"including summarizer prompts ({_delta(cost)})",
            )
        return (
            VERDICT_PREFER_CAP,
            "completion is not separated and compact uses more total model-input tokens "
            f"including summarizer prompts ({_delta(cost)})",
        )
    return (
        VERDICT_STILL_TIED,
        "compaction fired, but neither completion nor total model-input cost separates",
    )


def format_table(
    cap: PolicyReport,
    compact: PolicyReport,
    paired: dict[str, PairedComparison],
    *,
    verdict: str,
    reason: str,
) -> str:
    """Render policy means, paired deltas, and the evidence verdict."""
    lines = [
        (
            f"{'policy':<16} {'completion':>10} {'steps':>7} {'model-calls':>11} "
            f"{'input-tok':>10} {'compact':>8} {'overflow':>9}"
        ),
        "-" * 82,
    ]
    for report in (cap, compact):
        lines.append(
            f"{report.policy:<16} {report.result.objective_score:>10.3f} "
            f"{report.mean_steps:>7.2f} {report.metric_mean('n_model_calls'):>11.2f} "
            f"{report.metric_mean(METRIC_TOTAL_MODEL_INPUT_TOKENS):>10.1f} "
            f"{int(sum(report.vector('n_compactions'))):>8d} "
            f"{report.n_context_overflow:>9d}"
        )
    lines.extend(
        [
            "",
            "paired compact - observation_cap:",
            f"  d(completion)={_delta(paired[METRIC_COMPLETION])}",
            f"  d(total-model-input-tok)={_delta(paired[METRIC_TOTAL_MODEL_INPUT_TOKENS])}",
            f"  d(model-calls)={_delta(paired['n_model_calls'])}",
            f"verdict: [{verdict}] {reason}",
        ]
    )
    return "\n".join(lines)
