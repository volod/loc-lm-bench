"""Shared navigation and progress behavior."""

from pathlib import Path

from llb.review.core import ReviewAction, ReviewAdapter, ReviewNavigator, ReviewRecord


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
            (),
            "even" if index % 2 == 0 else "odd",
            self.verdicts[index],
        )

    def apply(self, index: int, action: str) -> None:
        self.verdicts[index] = action


def test_navigation_resumes_and_wraps_to_pending() -> None:
    adapter = _Adapter()
    navigator = ReviewNavigator(adapter)
    assert navigator.index == 1
    navigator.next()
    adapter.apply(2, "accept")
    assert navigator.advance_after_verdict() == 1
    progress = adapter.progress(1)
    assert (progress.reviewed, progress.total) == (2, 3)
    assert (progress.stratum_reviewed, progress.stratum_total) == (0, 1)
