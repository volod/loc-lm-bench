"""Unified Textual workbench over the persistence-neutral review core."""

from pathlib import Path
from typing import Literal

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, ScrollableContainer
from textual.widgets import Button, Footer, Header, Static

from llb.review.core import ActionTone, ReviewAdapter, ReviewNavigator, ReviewSection
from llb.review.registry import open_review

_BUTTON_VARIANTS: dict[ActionTone, Literal["default", "primary", "success", "warning", "error"]] = {
    "positive": "success",
    "warning": "warning",
    "negative": "error",
    "neutral": "default",
}


class ReviewWorkbench(App[None]):
    """One keyboard and color language for every human review gate."""

    TITLE = "LLB review workbench"
    CSS = """
    Screen { layout: vertical; background: #10151c; color: #d9e2ec; }
    #progress { height: 3; padding: 1 2; background: #16324f; color: #b9e6ff; }
    #panes { height: 1fr; padding: 1; }
    .data-pane { min-height: 8; margin-bottom: 1; padding: 1 2; border: round #477998; }
    #record-content { background: #182631; color: #e8f1f7; }
    #evidence { background: #192d2a; color: #d8f3e8; border: round #4b8f8c; }
    #metadata { background: #29243a; color: #ddd4f5; border: round #7469a8; }
    #actions { height: auto; min-height: 5; padding: 1; background: #332913;
               grid-size: 6; grid-gutter: 0 1; }
    #actions Button { width: 1fr; min-width: 12; text-style: bold; }
    #status { height: 2; padding: 0 2; background: #332913; color: #ffe3a3; }
    """
    BINDINGS = [
        Binding("right,n", "next", "Next"),
        Binding("left,b", "previous", "Previous"),
        Binding("u", "pending", "Next pending"),
        Binding("q", "quit", "Save and quit"),
    ]

    def __init__(self, adapter: ReviewAdapter, *, start: int | None = None) -> None:
        super().__init__()
        self.adapter = adapter
        self.navigator = ReviewNavigator(adapter, start)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="progress", markup=False)
        with ScrollableContainer(id="panes"):
            yield Static("", id="record-content", classes="data-pane", markup=False)
            yield Static("", id="evidence", classes="data-pane", markup=False)
            yield Static("", id="metadata", classes="data-pane", markup=False)
        with Grid(id="actions"):
            for index, action in enumerate(self.adapter.actions):
                yield Button(
                    f"[{action.key}] {action.label}",
                    id=f"review-action-{index}",
                    variant=_BUTTON_VARIANTS[action.tone],
                )
        yield Static("", id="status", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self._render_record()

    def on_key(self, event: object) -> None:
        character = getattr(event, "character", None)
        key = (
            character
            if isinstance(character, str) and len(character) == 1
            else getattr(event, "key", "")
        )
        action = next((item for item in self.adapter.actions if item.key == key), None)
        if action is None:
            return
        getattr(event, "prevent_default")()
        getattr(event, "stop")()
        self._apply(action.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        prefix = "review-action-"
        button_id = event.button.id or ""
        if not button_id.startswith(prefix):
            return
        action = self.adapter.actions[int(button_id.removeprefix(prefix))]
        self._apply(action.value)

    def action_next(self) -> None:
        self.navigator.next()
        self._render_record()

    def action_previous(self) -> None:
        self.navigator.previous()
        self._render_record()

    def action_pending(self) -> None:
        self.navigator.next_pending()
        self._render_record()

    def _apply(self, value: str) -> None:
        try:
            self.adapter.apply(self.navigator.index, value)
        except (OSError, ValueError) as exc:
            self.query_one("#status", Static).update(f"[blocked] {exc}")
            return
        self.query_one("#status", Static).update(f"[saved] {value} -> {self.adapter.path}")
        if value != "clear":
            self.navigator.advance_after_verdict()
        self._render_record()

    def _render_record(self) -> None:
        index = self.navigator.index
        record = self.adapter.record(index)
        progress = self.adapter.progress(index)
        self.sub_title = f"{self.adapter.kind} - {record.key}"
        self.query_one("#progress", Static).update(
            f"dataset {progress.reviewed}/{progress.total} reviewed | "
            f"record {progress.position}/{progress.total} | "
            f"stratum {progress.stratum}: "
            f"{progress.stratum_reviewed}/{progress.stratum_total} reviewed"
        )
        by_role = {section.role: section for section in record.sections}
        self._update_section("#record-content", by_role.get("data"))
        self._update_section("#evidence", by_role.get("evidence"))
        self._update_section("#metadata", by_role.get("metadata"))

    def _update_section(self, selector: str, section: ReviewSection | None) -> None:
        text = "(none)" if section is None else f"{section.title}\n\n{section.text}"
        self.query_one(selector, Static).update(text)


def run_workbench(
    path_or_adapter: Path | str | ReviewAdapter,
    *,
    start: int | None = None,
) -> ReviewAdapter:
    """Detect a ledger when needed, run the app, and return its live adapter."""
    adapter = (
        path_or_adapter
        if isinstance(path_or_adapter, ReviewAdapter)
        else open_review(path_or_adapter)
    )
    ReviewWorkbench(adapter, start=start).run()
    adapter.finish()
    return adapter
