"""M7.2 judge diagnostic observability -- classify and count zero-valued judge scores.

A gated judge can return zero for very different reasons, and conflating them hides whether a
benchmark dipped because the CANDIDATE failed (it produced no answer) or because the LOCAL JUDGE
failed (it could not emit the strict JSON the metric engine expects, or its endpoint was
unreachable). This module turns the per-record outcome into a small, manifest-friendly summary so
that distinction is recorded ALONGSIDE the objective headline (never folded into it).

Pure + dependency-free: it classifies from `(answer, score)` plus any precise per-record reason the
real judge path surfaced, so the wiring is provable from fake scorers with no DeepEval / endpoint.
"""

from collections import Counter

from llb.contracts import JudgeDiagnostics, JudgeInputRecord, JudgeScore

# Reason codes for a zero-valued judge score.
JUDGE_DIAG_OK = "ok"  # not a zero -> a real, well-formed score
JUDGE_DIAG_EMPTY_ANSWER = "empty_answer"  # the CANDIDATE produced no answer to judge
JUDGE_DIAG_MALFORMED_JSON = "malformed_judge_json"  # the JUDGE did not emit strict JSON
JUDGE_DIAG_TRANSPORT_ERROR = "judge_transport_error"  # the JUDGE endpoint was unreachable
JUDGE_DIAG_ZERO = "zero_score"  # non-empty answer scored zero, reason not otherwise classified

ZERO_REASONS = (
    JUDGE_DIAG_EMPTY_ANSWER,
    JUDGE_DIAG_MALFORMED_JSON,
    JUDGE_DIAG_TRANSPORT_ERROR,
    JUDGE_DIAG_ZERO,
)

# Substrings that mark a judge exception as a transport failure rather than a format failure.
_TRANSPORT_HINTS = (
    "connection",
    "connecterror",
    "timeout",
    "apiconnection",
    "apitimeout",
    "transport",
    "socket",
    "httpx",
    "unreachable",
)


def classify_judge_exception(exc: BaseException) -> str:
    """Map a judge-measurement exception to a transport vs malformed-JSON reason code."""
    text = f"{type(exc).__name__} {exc}".lower()
    if any(hint in text for hint in _TRANSPORT_HINTS):
        return JUDGE_DIAG_TRANSPORT_ERROR
    return JUDGE_DIAG_MALFORMED_JSON


def _is_zero(score: JudgeScore) -> bool:
    return float(score.get("faithfulness", 0.0) or 0.0) == 0.0 and (
        float(score.get("answer_relevancy", 0.0) or 0.0) == 0.0
    )


def classify_record(answer: str, score: JudgeScore, reason: str | None = None) -> str:
    """Reason for one record's judge score.

    An empty candidate answer is always `empty_answer`. Otherwise a precise `reason` surfaced by
    the real judge path (malformed JSON / transport error) wins; failing that, a zero score is
    `zero_score` and any non-zero score is `ok`."""
    if not str(answer).strip():
        return JUDGE_DIAG_EMPTY_ANSWER
    if reason in (JUDGE_DIAG_MALFORMED_JSON, JUDGE_DIAG_TRANSPORT_ERROR):
        return reason
    return JUDGE_DIAG_ZERO if _is_zero(score) else JUDGE_DIAG_OK


def summarize_judge_diagnostics(
    records: list[JudgeInputRecord],
    scores: list[JudgeScore],
    reasons: list[str | None] | None = None,
) -> JudgeDiagnostics:
    """Aggregate per-record classifications into the manifest-friendly `JudgeDiagnostics`.

    `reasons[i]` is the precise reason the real judge path recorded for record `i` (or None); when
    absent, the classification falls back to inference from `(answer, score)`, so a fake scorer
    still yields a faithful empty-answer / zero-score breakdown."""
    counts: Counter[str] = Counter()
    for index, (record, score) in enumerate(zip(records, scores)):
        reason = reasons[index] if reasons is not None and index < len(reasons) else None
        counts[classify_record(str(record.get("answer", "")), score, reason)] += 1
    n_zero = sum(counts[reason] for reason in ZERO_REASONS)
    zero_reasons = {reason: counts[reason] for reason in ZERO_REASONS if counts[reason]}
    return {
        "n": len(scores),
        "n_ok": counts[JUDGE_DIAG_OK],
        "n_zero": n_zero,
        "reasons": zero_reasons,
    }
