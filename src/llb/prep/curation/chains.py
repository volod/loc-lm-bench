"""Curate merged chain-of-questions drafts (external-draft contract Artifact D, provisional).

The external form remains separate from canonical ``ChainItem`` rows because services provide
quotes rather than trusted offsets. Curation merges exports, re-grounds step quotes to exact corpus
text, drops structurally broken chains, and deduplicates chains that walk the same question
sequence.
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
from llb.prep.ontology.refine import is_circular

_LOG = logging.getLogger(__name__)

MIN_CHAIN_STEPS = 2
MAX_CHAIN_STEPS = 4


def _load_chains(inputs: list[Path], report: CurationReport) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    for path in inputs:
        source = str(path)
        n_before = len(chains)
        for row in load_jsonl_rows(load_json_documents(path)):
            if isinstance(row, dict):
                row = dict(row)
                row["_source"] = source
                chains.append(row)
        report.sources[source] = len(chains) - n_before
    return chains


def _chain_id(chain: dict[str, Any], index: int) -> str:
    return str(chain.get("chain_id") or f"chain-{index:04d}")


def _sorted_steps(chain: dict[str, Any]) -> list[dict[str, Any]] | None:
    steps = chain.get("steps")
    if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
        return None
    return sorted(steps, key=lambda s: int(s.get("order") or 0))


def _validate_steps(
    chain_id: str, source: str, steps: list[dict[str, Any]], report: CurationReport
) -> bool:
    if not (MIN_CHAIN_STEPS <= len(steps) <= MAX_CHAIN_STEPS):
        report.reject_invalid(chain_id, source, f"chain must have 2-4 steps, has {len(steps)}")
        return False
    orders = [int(s.get("order") or 0) for s in steps]
    if orders != list(range(1, len(steps) + 1)):
        report.reject_invalid(chain_id, source, f"step orders must be 1..n, got {orders}")
        return False
    for step in steps:
        if not str(step.get("question") or "").strip():
            report.reject_invalid(chain_id, source, f"step {step.get('order')}: empty question")
            return False
        if not str(step.get("quote") or "").strip():
            report.reject_invalid(chain_id, source, f"step {step.get('order')}: empty quote")
            return False
        if not str(step.get("source_doc_id") or "").strip():
            report.reject_invalid(chain_id, source, f"step {step.get('order')}: no source doc")
            return False
    return True


def _ground_steps(
    chain_id: str,
    source: str,
    steps: list[dict[str, Any]],
    corpus_texts: dict[str, str] | None,
    report: CurationReport,
) -> bool:
    if corpus_texts is None:
        return True
    for step in steps:
        doc_id = str(step.get("source_doc_id"))
        text = corpus_texts.get(doc_id)
        if text is None:
            report.reject_invalid(
                chain_id, source, f"step {step.get('order')}: unknown doc {doc_id}"
            )
            return False
        grounded = ground_span(text, str(step.get("quote", "")))
        if grounded is None:
            report.reject_invalid(
                chain_id, source, f"step {step.get('order')}: quote not found in {doc_id}"
            )
            return False
        _start, exact = grounded
        if exact != step.get("quote"):
            report.note_repair(chain_id, source, f"step {step.get('order')}: quote re-snapped")
            step["quote"] = exact
        if not str(step.get("reference_answer") or "").strip():
            report.note_repair(
                chain_id, source, f"step {step.get('order')}: reference_answer set from quote"
            )
            step["reference_answer"] = exact
    return True


def _is_flabby_chain(
    chain_id: str, source: str, steps: list[dict[str, Any]], report: CurationReport
) -> bool:
    quotes = [normalize_text(str(s.get("quote", ""))) for s in steps]
    if len(set(quotes)) != len(quotes):
        report.reject_flabby(chain_id, source, "steps reuse the same passage")
        return True
    for step in steps:
        question = str(step.get("question", ""))
        answer = str(step.get("reference_answer", ""))
        if question_too_vague(question):
            report.reject_flabby(
                chain_id, source, f"step {step.get('order')}: question too short or vague"
            )
            return True
        if is_circular(question, answer, answer):
            report.reject_flabby(
                chain_id, source, f"step {step.get('order')}: question leaks its answer"
            )
            return True
    final_answer = normalize_text(str(steps[-1].get("reference_answer", "")))
    if final_answer and final_answer in quotes[0]:
        report.reject_flabby(chain_id, source, "final answer findable from step-1 passage")
        return True
    return False


def _chain_is_valid(
    chain: dict[str, Any],
    chain_id: str,
    corpus_texts: dict[str, str] | None,
    report: CurationReport,
) -> bool:
    """Structural checks + step grounding + flabbiness; sorts `chain['steps']` in place."""
    source = chain["_source"]
    steps = _sorted_steps(chain)
    if steps is None:
        report.reject_invalid(chain_id, source, "steps must be a list of objects")
        return False
    if not _validate_steps(chain_id, source, steps, report):
        return False
    if not _ground_steps(chain_id, source, steps, corpus_texts, report):
        return False
    if _is_flabby_chain(chain_id, source, steps, report):
        return False
    chain["steps"] = steps
    return True


def _chain_signature(chain: dict[str, Any]) -> str:
    return " | ".join(normalize_text(str(s.get("question", ""))) for s in chain["steps"])


def _dedup_chains(
    valid: list[dict[str, Any]],
    *,
    embedder: QuestionEmbedder | None,
    dedup_threshold: float,
    prior_questions: list[str] | None,
    report: CurationReport,
) -> list[dict[str, Any]]:
    """Exact question-sequence dedup, then embedding near-dup filtering."""
    keep = drop_exact_duplicates(
        [_chain_signature(c) for c in valid],
        report,
        [_chain_id(c, i) for i, c in enumerate(valid)],
        [c["_source"] for c in valid],
    )
    valid = [valid[i] for i in keep]
    keep = drop_near_duplicates(
        [" ".join(str(s.get("question", "")) for s in c["steps"]) for c in valid],
        embedder,
        dedup_threshold,
        report,
        [_chain_id(c, i) for i, c in enumerate(valid)],
        [c["_source"] for c in valid],
        prior_texts=prior_questions,
    )
    return [valid[i] for i in keep]


def curate_chains(
    inputs: list[Path],
    *,
    corpus_texts: dict[str, str] | None = None,
    embedder: QuestionEmbedder | None = None,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    prior_questions: list[str] | None = None,
) -> tuple[list[dict[str, Any]], CurationReport]:
    """Merge + validate + ground + dedup chain drafts; returns (chain rows, report)."""
    report = CurationReport(kind="chains")
    chains = _load_chains(inputs, report)
    report.loaded = len(chains)

    valid = [
        chain
        for i, chain in enumerate(chains)
        if _chain_is_valid(chain, _chain_id(chain, i), corpus_texts, report)
    ]
    valid = _dedup_chains(
        valid,
        embedder=embedder,
        dedup_threshold=dedup_threshold,
        prior_questions=prior_questions,
        report=report,
    )

    final_ids = unique_ids(
        [_chain_id(c, i) for i, c in enumerate(valid)], report, [c["_source"] for c in valid]
    )
    out: list[dict[str, Any]] = []
    for chain, chain_id in zip(valid, final_ids):
        cleaned = {k: v for k, v in chain.items() if k != "_source"}
        cleaned["chain_id"] = chain_id
        out.append(cleaned)
    report.kept = len(out)
    _LOG.info(
        "[curate] chains: kept %d/%d (%d invalid, %d flabby, %d exact-dup, %d near-dup)",
        report.kept,
        report.loaded,
        len(report.invalid),
        len(report.flabby),
        len(report.exact_duplicates),
        len(report.near_duplicates),
    )
    return out, report
