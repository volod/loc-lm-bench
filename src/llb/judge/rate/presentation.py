"""Calibration-rater cards, help, completion panel, and final summary."""

from collections.abc import Sequence
from pathlib import Path

from llb.judge.rate.commands import RATING_ANCHORS, RATING_MAX, RATING_MIN
from llb.judge.rate.state import rated_count, rating_histogram


def summary_lines(rows: Sequence[dict[str, str]], path: Path) -> list[str]:
    total = len(rows)
    rated = rated_count(rows)
    answered = sum(1 for row in rows if (row.get("human_answer") or "").strip())
    histogram = rating_histogram(rows)
    spread = "  ".join(
        f"{rating}:{histogram[rating]}" for rating in range(RATING_MIN, RATING_MAX + 1)
    )
    lines = [
        f"[calibration] saved {path}",
        f"[calibration] progress : {rated}/{total} rated, {total - rated} remaining, "
        f"{answered} with your own answer",
        f"[calibration] ratings  : {spread}  (1=wrong .. 5=fully correct)",
    ]
    if rated < total:
        lines.append(
            "[calibration] resume   : re-run `make calibration-rate` "
            "(continues at the first unrated item)"
        )
    lines.append(f"[calibration] score    : make calibration-score RATINGS={path}")
    return lines


def completion_panel(rows: Sequence[dict[str, str]], total: int) -> str:
    answered = sum(1 for row in rows if (row.get("human_answer") or "").strip())
    histogram = rating_histogram(rows)
    spread = "  ".join(
        f"{rating}:{histogram[rating]}" for rating in range(RATING_MIN, RATING_MAX + 1)
    )
    return "\n".join(
        [
            f"===== all {total} items rated ({answered} with your own answer) =====",
            f"  rating spread: {spread}   (1=wrong .. 5=fully correct)",
            "  review/change: p = last item, j <N> = jump to item N, u = next unrated",
            "  finish: press Enter or q to save + quit (then run make calibration-score)",
        ]
    )


def format_card(
    row: dict[str, str],
    position: int,
    total: int,
    rated: int,
    *,
    show_judge: bool = False,
) -> str:
    """Render a calibration item without revealing the judge unless requested."""
    lines = [
        f"item {position}/{total} (rated {rated}, remaining {total - rated})",
        f"  item_id    : {row.get('item_id', '')}",
        f"  provenance : {_field(row, 'provenance', '(unknown)')}",
        f"  question   : {row.get('question', '')}",
        f"  reference  : {row.get('reference_answer', '')}",
        f"  model      : {_field(row, 'model_answer', '(no answer)')}",
        f"  your answer: {_field(row, 'human_answer', '(none)')}",
        f"  your rating: {_field(row, 'human_rating', '(unrated)')}",
    ]
    note = (row.get("human_note") or "").strip()
    if note:
        lines.append(f"  your note  : {note}")
    if show_judge:
        lines.append(f"  judge      : {_field(row, 'judge_rating', '(none)')}")
    return "\n".join(lines)


def help_text() -> str:
    anchors = "; ".join(
        f"{rating} = {RATING_ANCHORS[rating]}" for rating in range(RATING_MIN, RATING_MAX + 1)
    )
    return "\n".join(
        [
            "commands:",
            f"  {RATING_MIN}-{RATING_MAX}  set rating + advance     a     author/edit your answer",
            "  n/Enter  next (no change)         note  edit a note",
            "  p/b      previous                 c     clear this rating",
            "  j <N>    jump to item N           u     jump to next unrated",
            "  ?/h      this help                q     save + quit",
            "  arrow keys: up/left = previous, down/right = next",
            f"rating anchors: {anchors}",
            "rate INDEPENDENTLY -- author your own answer first; do not anchor to the judge.",
        ]
    )


def _field(row: dict[str, str], name: str, blank: str) -> str:
    value = (row.get(name) or "").strip()
    return value if value else blank
