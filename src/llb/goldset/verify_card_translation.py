"""Specialized card for source-aligned English/Ukrainian translation review."""

from llb.goldset.verify_base import CHECK_COLS
from llb.goldset.verify_commands import TRANSLATION_PROFILE, check_label


def is_translation_row(row: dict[str, str]) -> bool:
    return row.get("review_profile", "") == TRANSLATION_PROFILE


def translation_card_lines(
    row: dict[str, str], position: int, total: int, decided: int
) -> list[str]:
    """Render both language fields without irrelevant gold-set span metadata."""
    lines = [
        "===== knowledge-cutoff translation review =====",
        f"item {position}/{total} (decided {decided}, remaining {total - decided})",
        f"== id: {row.get('item_id', '')} month={row.get('stratum', '')}",
        "",
        f"== English question: {row.get('question', '')}",
        f"== Ukrainian question: {row.get('reference_answer', '')}",
        "== ordered English/Ukrainian choices:",
        row.get("context", "") or "(missing)",
        "== checks:",
    ]
    for column in CHECK_COLS:
        value = (row.get(column) or "").strip() or "(unchecked)"
        lines.append(f"    {column:<15}: {value}  -- {check_label(column, TRANSLATION_PROFILE)}")
    decision = (row.get("decision") or "").strip() or "(undecided)"
    lines.append(f"== decision: {decision}")
    return lines
