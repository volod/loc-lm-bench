"""Public screening task and report contracts."""

from typing_extensions import NotRequired, TypedDict


class ScreenTaskResult(TypedDict):
    task: str
    metric: str
    score: float


class ScreenReport(TypedDict):
    model: str
    backend: str
    track: str
    requested_tasks: list[str]
    results: list[ScreenTaskResult]
    covered: list[str]
    missing: list[str]
    complete: bool
    # Examples per task the screen was run with (`--limit`), or None for the full task. A capped
    # smoke report is a different measurement from a full one, so the cache reader compares it
    # rather than handing a two-example screen to a decision that asked for the whole track.
    limit: NotRequired[int | None]
