"""Curate merged security-case drafts (external-draft contract Artifact C).

Merges exported case arrays from any number of services/batches, validates each record against
the committed `SecurityCase` schema (closed families, detector kinds), enforces the probe-role
consistency rules the prompts state (a benign control never expects refusal; a leak detector
needs markers), grounds `attrs.grounding_hint` in the staged corpus when given, deduplicates
prompts across services WITHOUT touching intentional twins (bias pairs, cross-language groups),
and drops orphaned bias-pair variants so decision-consistency scoring always sees complete pairs.
"""

import logging
from pathlib import Path
from typing import Any

from llb.prep.frontier import ground_span
from llb.prep.curation.common import (
    CurationReport,
    DEFAULT_DEDUP_THRESHOLD,
    QuestionEmbedder,
    drop_exact_duplicates,
    drop_near_duplicates,
    load_json_documents,
    load_jsonl_rows,
    normalize_text,
    question_too_vague,
    unique_ids,
)

_LOG = logging.getLogger(__name__)


def _load_cases(inputs: list[Path], report: CurationReport) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in inputs:
        source = str(path)
        n_before = len(cases)
        for row in load_jsonl_rows(load_json_documents(path)):
            if isinstance(row, dict):
                row = dict(row)
                row["_source"] = source
                cases.append(row)
        report.sources[source] = len(cases) - n_before
    return cases


def _case_id(case: dict[str, Any], index: int) -> str:
    return str(case.get("id") or f"case-{index:04d}")


def _validate(case: dict[str, Any], case_id: str, report: CurationReport) -> bool:
    from llb.scoring.security import SecurityCase

    source = case["_source"]
    try:
        parsed = SecurityCase.from_record({k: v for k, v in case.items() if k != "_source"})
    except (KeyError, ValueError, TypeError) as exc:
        report.reject_invalid(case_id, source, f"schema: {exc}")
        return False
    if parsed.detector_kind == "leak" and not parsed.markers:
        report.reject_invalid(case_id, source, "leak detector requires non-empty markers")
        return False
    if parsed.benign and parsed.expect_refusal:
        report.reject_invalid(case_id, source, "benign control must not expect refusal")
        return False
    if question_too_vague(parsed.prompt):
        report.reject_flabby(case_id, source, "prompt too short or vague")
        return False
    return True


def _ground_hint(
    case: dict[str, Any], case_id: str, corpus_texts: dict[str, str] | None, report: CurationReport
) -> bool:
    """Require `attrs.grounding_hint` to be exact corpus text (repaired when near-verbatim)."""
    if corpus_texts is None:
        return True
    attrs = case.get("attrs") or {}
    hint = str(attrs.get("grounding_hint") or "").strip()
    if not hint:
        report.reject_flabby(case_id, case["_source"], "missing attrs.grounding_hint")
        return False
    for text in corpus_texts.values():
        grounded = ground_span(text, hint)
        if grounded is not None:
            _start, exact = grounded
            if exact != hint:
                report.note_repair(case_id, case["_source"], "grounding_hint re-snapped")
                attrs["grounding_hint"] = exact
                case["attrs"] = attrs
            return True
    report.reject_invalid(case_id, case["_source"], "grounding_hint not found in corpus")
    return False


def _protection_key(case: dict[str, Any]) -> str:
    """Cases sharing a bias pair or cross-language group are intentional twins, never dedup them."""
    attrs = case.get("attrs") or {}
    pair = str(attrs.get("pair_id") or "")
    xlang = str(case.get("xlang_group") or attrs.get("xlang_group") or "")
    return pair or xlang


def _drop_orphan_pair_variants(
    cases: list[dict[str, Any]], report: CurationReport
) -> list[dict[str, Any]]:
    """A bias pair needs >= 2 surviving variants; a lone survivor cannot be scored for
    consistency, so it is dropped with a reason (runs AFTER all other filters/dedup)."""
    counts: dict[str, int] = {}
    for case in cases:
        pair = str((case.get("attrs") or {}).get("pair_id") or "")
        if pair:
            counts[pair] = counts.get(pair, 0) + 1
    kept: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        pair = str((case.get("attrs") or {}).get("pair_id") or "")
        if pair and counts[pair] < 2:
            report.reject_invalid(
                _case_id(case, i), case["_source"], f"incomplete bias pair {pair}"
            )
            continue
        kept.append(case)
    return kept


def curate_security(
    inputs: list[Path],
    *,
    corpus_texts: dict[str, str] | None = None,
    embedder: QuestionEmbedder | None = None,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    prior_questions: list[str] | None = None,
) -> tuple[list[dict[str, Any]], CurationReport]:
    """Merge + validate + ground + dedup security cases; returns (merged array, report)."""
    report = CurationReport(kind="security")
    cases = _load_cases(inputs, report)
    report.loaded = len(cases)

    valid: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        case_id = _case_id(case, i)
        if not _validate(case, case_id, report):
            continue
        if not _ground_hint(case, case_id, corpus_texts, report):
            continue
        valid.append(case)

    ids = [_case_id(c, i) for i, c in enumerate(valid)]
    sources = [c["_source"] for c in valid]
    keep = drop_exact_duplicates(
        [normalize_text(str(c.get("prompt", ""))) for c in valid], report, ids, sources
    )
    valid = [valid[i] for i in keep]
    keep = drop_near_duplicates(
        [str(c.get("prompt", "")) for c in valid],
        embedder,
        dedup_threshold,
        report,
        [_case_id(c, i) for i, c in enumerate(valid)],
        [c["_source"] for c in valid],
        protected_groups=[_protection_key(c) for c in valid],
        prior_texts=prior_questions,
    )
    valid = [valid[i] for i in keep]
    valid = _drop_orphan_pair_variants(valid, report)

    final_ids = unique_ids(
        [_case_id(c, i) for i, c in enumerate(valid)], report, [c["_source"] for c in valid]
    )
    out: list[dict[str, Any]] = []
    for case, case_id in zip(valid, final_ids):
        cleaned = {k: v for k, v in case.items() if k != "_source"}
        cleaned["id"] = case_id
        out.append(cleaned)
    report.kept = len(out)
    _LOG.info(
        "[curate] security: kept %d/%d (%d invalid, %d flabby, %d exact-dup, %d near-dup)",
        report.kept,
        report.loaded,
        len(report.invalid),
        len(report.flabby),
        len(report.exact_duplicates),
        len(report.near_duplicates),
    )
    return out, report
