"""Review-card rendering + command parsing for the interactive verification session.

The pure presentation half of `verify_session.py`: parse one prompt line into a `Command`, render
a sampled item as a terminal review card (standard QA layout or the dense chain layout), and the
`?` help text. No I/O, no session state, so it is fully unit-testable. `verify_session.py` (the
session loop) builds on these; both share the worksheet schema from `verify.py`.
"""

import json

from llb.goldset.span_occurrences import SPAN_OCCURRENCES_COL
from llb.goldset.verify_base import CHECK_COLS, KIND_CHAINS
from llb.goldset.verify_commands import check_label
from llb.goldset.verify_card_text import (
    _ANSI_ANSWER,
    _ANSI_BOLD,
    _ANSI_DEPENDENCY,
    _ANSI_QUESTION,
    _ANSI_SOURCE,
    _color,
    _context_excerpt,
    _indent,
    _truncate,
)
from llb.goldset.verify_card_translation import is_translation_row, translation_card_lines

_CHAIN_SEPARATOR = "+" * 64
_CHAIN_DEFAULT_WIDTH = 120
_CHAIN_MIN_VALUE_WIDTH = 24

# The four checks, in card order, mapped to the keystroke that marks them. Lowercase = PASS,
# uppercase = FAIL. `planted` only applies to synthetic items (blank/N/A for real ones).

# Command kinds returned by `parse_command`.

# Command keys mirror the external-RAG review session (`llb.scoring.external_rag_session`) where
# the two tools do the same thing: o=note and w=edit-the-answer are shared aliases, and the
# navigation row (Enter/n, b, u, j<N>, ?, q) is identical. The decision keys necessarily differ:
# here a/r/p mark CHECKS (answerable/reference/planted), so accept/reject are y/x.


def _field(row: dict[str, str], name: str, blank: str) -> str:
    value = (row.get(name) or "").strip()
    return value if value else blank


def _is_synthetic_row(row: dict[str, str]) -> bool:
    return (row.get("synthetic") or "").strip().lower() == "true"


def _is_chain_row(row: dict[str, str]) -> bool:
    return (row.get("item_kind") or "").strip() == KIND_CHAINS


def _chain_field(
    label: str,
    value: object,
    *,
    width: int,
    color: str = "",
    use_color: bool = False,
    blank: str = "(none)",
) -> str:
    available = max(_CHAIN_MIN_VALUE_WIDTH, width - len(label) - 1)
    line = f"{label} {_truncate(value, available, blank=blank)}"
    return _color(line, color, use_color and bool(color))


def _format_chain_steps(
    row: dict[str, str], *, color: bool = False, width: int = _CHAIN_DEFAULT_WIDTH
) -> list[str]:
    raw = (row.get("chain_steps") or "").strip()
    if not raw:
        return []
    try:
        steps = json.loads(raw)
    except json.JSONDecodeError:
        return ["== chain_steps: (invalid JSON)"]
    if not isinstance(steps, list):
        return ["== chain_steps: (invalid JSON)"]
    valid_steps = [step for step in steps if isinstance(step, dict)]
    lines: list[str] = []
    for index, step in enumerate(valid_steps, start=1):
        order = str(step.get("order", ""))
        page = str(step.get("page_citation") or "(none)")
        doc = _truncate(step.get("span_doc_id"), max(_CHAIN_MIN_VALUE_WIDTH, width // 2))
        header = f"STEP {order or index}/{len(valid_steps)} | doc={doc} | page={page}"
        lines.append(_color(header, _ANSI_BOLD, color))
        lines.append(
            _chain_field(
                "Q:", step.get("question"), width=width, color=_ANSI_QUESTION, use_color=color
            )
        )
        lines.append(
            _chain_field(
                "A:",
                step.get("reference_answer"),
                width=width,
                color=_ANSI_ANSWER,
                use_color=color,
            )
        )
        lines.append(
            _chain_field(
                "SOURCE:",
                step.get("span_text"),
                width=width,
                color=_ANSI_SOURCE,
                use_color=color,
                blank="(missing)",
            )
        )
        dependency = str(step.get("dependency_note") or "").strip()
        if dependency:
            lines.append(
                _chain_field(
                    "DEPENDENCY:",
                    dependency,
                    width=width,
                    color=_ANSI_DEPENDENCY,
                    use_color=color,
                )
            )
        context = str(step.get("context") or "").strip()
        if context:
            available = max(_CHAIN_MIN_VALUE_WIDTH, width - len("CONTEXT:") - 1)
            lines.append(f"CONTEXT: {_context_excerpt(context, available)}")
    return lines


def _chain_checks(row: dict[str, str]) -> str:
    checks = [
        f"{col.removeprefix('chk_')}={_field(row, col, '(unchecked)')}"
        for col in CHECK_COLS
        if col != "chk_planted"
    ]
    return "CHECKS: " + " | ".join(checks)


def _chain_card_lines(
    row: dict[str, str], position: int, total: int, decided: int, *, color: bool, width: int
) -> list[str]:
    """Dense one-line-per-field chain layout."""
    decision = _field(row, "decision", "(undecided)")
    lines = [
        _CHAIN_SEPARATOR,
        f"CHAIN {position}/{total} | id={row.get('item_id', '')} | decided={decided} "
        f"remaining={total - decided} | decision={decision}",
        "REVIEW: compare A with SOURCE; confirm Q is answered and later steps add context.",
    ]
    lines.extend(_format_chain_steps(row, color=color, width=width))
    lines.append(_chain_checks(row))
    return lines


def _standard_card_lines(row: dict[str, str], position: int, total: int, decided: int) -> list[str]:
    """Classic multi-line QA card: header, meta, question/answer/span, context, checks."""
    rank = (row.get("retrieval_rank") or "").strip() or "(none)"
    page = (row.get("page_citation") or "").strip() or "(none)"
    lines = [
        "===== goldset verification review =====",
        f"item {position}/{total} (decided {decided}, remaining {total - decided})",
        f"== id: {row.get('item_id', '')}",
        f"== meta: stratum={row.get('stratum', '')} synthetic={row.get('synthetic', '')} "
        f"retrieval_rank={rank}",
    ]
    reviewer = (row.get("reviewer_id") or "").strip()
    if reviewer:
        lines.append(f"== reviewer: {reviewer}")
    priors = (row.get("prior_decisions") or "").strip()
    if priors:
        lines.append(f"== prior_decisions: {priors} -- decide independently, then compare")
    lines += [
        "",
        f"== question: {row.get('question', '')}",
        f"== reference_answer: {row.get('reference_answer', '')}",
        f"== span: doc={row.get('span_doc_id', '')} page={page}",
    ]
    occurrences = (row.get(SPAN_OCCURRENCES_COL) or "").strip()
    if occurrences and occurrences != "1":
        lines.append(
            f"== ambiguous evidence: this span text appears in {occurrences} places in the corpus "
            "-- decide whether the question is uniquely answerable"
        )
    lines += [
        "== context (cited span between >>> <<<)",
        _indent(row.get("context", "") or "(missing)"),
        "== checks:",
    ]
    synthetic = _is_synthetic_row(row)
    for col in CHECK_COLS:
        if col == "chk_planted" and not synthetic:
            continue
        lines.append(
            f"    {col:<15}: {_field(row, col, '(unchecked)')}  -- "
            f"{check_label(col, row.get('review_profile', ''))}"
        )
    lines.append(f"== decision: {_field(row, 'decision', '(undecided)')}")
    return lines


def _card_annotation_lines(row: dict[str, str], show_crosscheck: bool) -> list[str]:
    """Optional trailing lines: edit, reject code, note, cross-check summary."""
    lines: list[str] = []
    for column, label in (
        ("edited_answer", "edited_answer"),
        ("reject_code", "reject_code"),
        ("human_note", "human_note"),
    ):
        value = (row.get(column) or "").strip()
        if value:
            lines.append(f"== {label}: {value}")
    if show_crosscheck:
        cc = "  ".join(
            f"{key}={_field(row, f'cc_{key}', '?')}"
            for key in ("grounded", "non_circular", "supported", "answerable")
        )
        lines.append(f"== crosscheck: {cc}  note={_field(row, 'cc_note', '')}")
    return lines


def format_card(
    row: dict[str, str],
    position: int,
    total: int,
    decided: int,
    *,
    show_crosscheck: bool = False,
    color: bool = False,
    width: int = _CHAIN_DEFAULT_WIDTH,
) -> str:
    """Render a review card; chain rows use a dense one-line-per-field terminal layout."""
    if is_translation_row(row):
        lines = translation_card_lines(row, position, total, decided)
    elif _is_chain_row(row):
        lines = _chain_card_lines(row, position, total, decided, color=color, width=width)
    else:
        lines = _standard_card_lines(row, position, total, decided)
    lines.extend(_card_annotation_lines(row, show_crosscheck))
    return "\n".join(lines)
