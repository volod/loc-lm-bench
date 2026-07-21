"""Headless Textual interaction and theme-role coverage."""

import asyncio
from pathlib import Path

from llb.review.core import ReviewAction, ReviewAdapter, ReviewRecord, ReviewSection
from llb.review.workbench import ReviewWorkbench


class _Adapter(ReviewAdapter):
    kind = "test"
    path = Path("ledger")
    actions = (ReviewAction("y", "Accept", "accept"),)

    def __init__(self) -> None:
        self.verdicts = ["accept", "", ""]

    def __len__(self) -> int:
        return len(self.verdicts)

    def record(self, index: int) -> ReviewRecord:
        return ReviewRecord(
            str(index),
            str(index),
            (
                ReviewSection("Record content", f"data {index}", "data"),
                ReviewSection("Evidence", f"evidence {index}", "evidence"),
                ReviewSection("Metadata", f"metadata {index}", "metadata"),
            ),
            "even" if index % 2 == 0 else "odd",
            self.verdicts[index],
        )

    def apply(self, index: int, action: str) -> None:
        self.verdicts[index] = action


def test_pilot_navigation_verdict_resume_and_color_roles() -> None:
    adapter = _Adapter()

    app = ReviewWorkbench(adapter)

    async def drive() -> None:
        async with app.run_test() as pilot:
            assert app.navigator.index == 1
            await pilot.press("y")
            assert adapter.verdicts[1] == "accept"
            assert app.navigator.index == 2
            await pilot.press("left")
            assert app.navigator.index == 1
            progress = app.query_one("#progress")
            assert "dataset 2/3 reviewed" in str(progress.render())
            content = app.query_one("#record-content")
            evidence = app.query_one("#evidence")
            actions = app.query_one("#actions")
            assert content.styles.background != evidence.styles.background
            assert content.styles.background != actions.styles.background

    asyncio.run(drive())
