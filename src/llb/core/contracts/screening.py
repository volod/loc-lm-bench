"""Public screening task and report contracts."""

from typing_extensions import TypedDict


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
