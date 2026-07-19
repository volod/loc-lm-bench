"""Judge-calibration CSV adapter."""

from pathlib import Path

from llb.judge.calibration_worksheet import load_worksheet
from llb.judge.rate.state import clear_rating, save_human_columns, set_rating
from llb.review.core import ReviewAction, ReviewAdapter, ReviewRecord
from llb.review.presentation import fields_section

_ACTIONS = tuple(
    ReviewAction(str(value), f"Rate {value}", str(value), "positive" if value >= 4 else "warning")
    for value in range(1, 6)
) + (ReviewAction("c", "Clear", "clear", "neutral"),)


class JudgeCalibrationAdapter(ReviewAdapter):
    """Preserve the calibration worksheet's human-column merge semantics."""

    kind = "judge-calibration"

    def __init__(self, worksheet: Path | str) -> None:
        self.path = Path(worksheet)
        self.rows, self.fieldnames = load_worksheet(self.path)

    @property
    def actions(self) -> tuple[ReviewAction, ...]:
        return _ACTIONS

    def __len__(self) -> int:
        return len(self.rows)

    def record(self, index: int) -> ReviewRecord:
        row = self.rows[index]
        item_id = row.get("item_id") or str(index + 1)
        return ReviewRecord(
            key=item_id,
            title=f"calibration: {item_id}",
            sections=(
                fields_section(
                    "Record content",
                    row,
                    ("question", "reference_answer", "model_answer", "human_answer"),
                    "data",
                ),
                fields_section("Evidence", row, ("provenance",), "evidence"),
                fields_section(
                    "Metadata",
                    row,
                    ("split", "human_rating", "human_note", "human_status"),
                    "metadata",
                ),
            ),
            stratum=row.get("split") or "calibration",
            verdict=(row.get("human_rating") or "").strip(),
        )

    def apply(self, index: int, action: str) -> None:
        row = self.rows[index]
        if action == "clear":
            clear_rating(row)
        elif action.isdigit() and 1 <= int(action) <= 5:
            set_rating(row, int(action))
        else:
            raise ValueError(f"unsupported {self.kind} action: {action}")
        save_human_columns(self.path, self.rows, self.fieldnames)
