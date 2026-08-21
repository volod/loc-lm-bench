"""Verify what a constructed control edit actually IS, instead of assuming it is unrelated.

The second-generation counterfactual controls carry exact construction provenance: which span
changed, from what, to what. Provenance proves the edit happened; it cannot prove the edited passage
is unrelated to its source. A changed quantity or a flipped modality is exactly the positive
contradiction the detector must keep, so a control family built from those edits may be a bank of
planted POSITIVES wearing a null's clothes.

This lane asks the strongest host-fit Ukrainian model for the relation between each original and its
edit, on the same prompt the claim tier uses, and declares the family eligible as a null only when
essentially all of them come back non-conflicting.
"""

from dataclasses import dataclass

from llb.conflicts.claim.prompt import AdjudicationError, adjudication_prompt, parse_adjudication
from llb.conflicts.constants import REL_COMPLEMENTARY
from llb.conflicts.interval_stats import wilson_interval
from llb.conflicts.null_research.geometry import CorpusGeometry
from llb.core.contracts.common import JsonObject
from llb.prep.frontier.telemetry import LLMComplete

MAX_CONFLICTING_CONTROL_FRACTION = 0.05
DEFAULT_ROLE_SAMPLE_PER_TYPE = 12


@dataclass(frozen=True)
class RoleSample:
    """One traced edit resolved back to the exact original and edited passage."""

    dataset: str
    edit_type: str
    original: str
    edited: str
    trace: JsonObject


def _edited_text(original: str, trace: JsonObject) -> str:
    start, end = int(trace["char_start"]), int(trace["char_end"])
    return original[:start] + str(trace["after"]) + original[end:]


def sample_traces(
    traces: list[JsonObject],
    corpora: dict[str, CorpusGeometry],
    *,
    per_type: int,
) -> list[RoleSample]:
    """Take a deterministic, evenly spaced slice of each dataset's edits per edit type."""
    grouped: dict[tuple[str, str], list[JsonObject]] = {}
    for trace in traces:
        key = (str(trace["dataset"]), str(trace["edit_type"]))
        grouped.setdefault(key, []).append(trace)
    samples: list[RoleSample] = []
    for (dataset, edit_type), rows in sorted(grouped.items()):
        if dataset not in corpora:
            continue
        step = max(1, len(rows) // per_type)
        for trace in rows[::step][:per_type]:
            original = corpora[dataset].chunks[int(trace["source_ordinal"])]["text"]
            samples.append(
                RoleSample(
                    dataset=dataset,
                    edit_type=edit_type,
                    original=original,
                    edited=_edited_text(original, trace),
                    trace=trace,
                )
            )
    return samples


def _role_verdict(sample: RoleSample, complete: LLMComplete) -> JsonObject:
    try:
        relation: str | None = parse_adjudication(
            complete(adjudication_prompt(sample.original, sample.edited))
        )["relation"]
        parsed = True
    except (AdjudicationError, RuntimeError):
        relation, parsed = None, False
    return {
        "dataset": sample.dataset,
        "edit_type": sample.edit_type,
        "doc_id": sample.trace["doc_id"],
        "source_ordinal": sample.trace["source_ordinal"],
        "before": sample.trace["before"],
        "after": sample.trace["after"],
        "relation": relation,
        "parsed": parsed,
        "conflicting": bool(parsed and relation != REL_COMPLEMENTARY),
    }


def _family_payload(rows: list[JsonObject]) -> JsonObject:
    conflicting = sum(bool(row["conflicting"]) for row in rows)
    lower, upper = wilson_interval(conflicting, len(rows))
    return {
        "verified_rows": len(rows),
        "conflicting_rows": conflicting,
        "conflicting_fraction": round(conflicting / len(rows), 6) if rows else 1.0,
        "conflicting_wilson_95": [round(lower, 6), round(upper, 6)],
        "relations": {
            relation: sum(row["relation"] == relation for row in rows)
            for relation in sorted({str(row["relation"]) for row in rows if row["parsed"]})
        },
        "eligible_as_null": bool(rows) and lower <= MAX_CONFLICTING_CONTROL_FRACTION,
    }


def verify_control_roles(
    traces: list[JsonObject],
    corpora: dict[str, CorpusGeometry],
    complete: LLMComplete,
    *,
    per_type: int = DEFAULT_ROLE_SAMPLE_PER_TYPE,
) -> JsonObject:
    """Adjudicate original-versus-edit relations and decide whether the family is a null at all."""
    verdicts = [
        _role_verdict(sample, complete)
        for sample in sample_traces(traces, corpora, per_type=per_type)
    ]
    by_type = {
        edit_type: _family_payload(
            [verdict for verdict in verdicts if verdict["edit_type"] == edit_type]
        )
        for edit_type in sorted({str(verdict["edit_type"]) for verdict in verdicts})
    }
    conflicting_total = sum(bool(verdict["conflicting"]) for verdict in verdicts)
    lower, _ = wilson_interval(conflicting_total, len(verdicts)) if verdicts else (1.0, 1.0)
    return {
        "method": "verified_counterfactual_role",
        "verifier": "host-fit UA relation adjudicator on the claim-tier prompt",
        "max_conflicting_control_fraction": MAX_CONFLICTING_CONTROL_FRACTION,
        "verified_rows": len(verdicts),
        "conflicting_rows": conflicting_total,
        "conflicting_fraction": round(conflicting_total / len(verdicts), 6) if verdicts else 1.0,
        "by_edit_type": by_type,
        "verdicts": verdicts,
        "gates": {
            "any_family_eligible": any(
                bool(payload["eligible_as_null"]) for payload in by_type.values()
            ),
            "family_eligible": bool(verdicts) and lower <= MAX_CONFLICTING_CONTROL_FRACTION,
        },
    }
