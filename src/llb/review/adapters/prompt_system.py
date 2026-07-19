"""Prompt-system candidate JSON adapter."""

from dataclasses import asdict
from pathlib import Path

from llb.prompt_system.pipeline import CANDIDATES_FILE
from llb.prompt_system.review import (
    STATUS_APPROVED,
    STATUS_PENDING,
    STATUS_PINNED,
    STATUS_REJECTED,
    approve,
    load_candidates,
    pin,
    reject,
    save_candidates,
)
from llb.review.core import ReviewAction, ReviewAdapter, ReviewRecord
from llb.review.presentation import fields_section, json_section

_ACTIONS = (
    ReviewAction("a", "Approve", STATUS_APPROVED, "positive"),
    ReviewAction("p", "Pin", STATUS_PINNED, "positive"),
    ReviewAction("r", "Reject", STATUS_REJECTED, "negative"),
    ReviewAction("c", "Clear", STATUS_PENDING, "neutral"),
)


class PromptSystemAdapter(ReviewAdapter):
    """Persist candidate status through the existing pretty-JSON writer."""

    kind = "prompt-system"

    def __init__(self, run: Path | str) -> None:
        value = Path(run)
        self.path = value / CANDIDATES_FILE if value.is_dir() else value
        self.candidates = load_candidates(self.path)

    @property
    def actions(self) -> tuple[ReviewAction, ...]:
        return _ACTIONS

    def __len__(self) -> int:
        return len(self.candidates)

    def record(self, index: int) -> ReviewRecord:
        candidate = self.candidates[index]
        fields = asdict(candidate.fields)
        stratum = f"tree-depth-{fields.get('knowledge_tree_depth', 0)}"
        verdict = "" if candidate.status == STATUS_PENDING else candidate.status
        return ReviewRecord(
            key=candidate.prompt_system_id,
            title=f"prompt system: {candidate.prompt_system_id}",
            sections=(
                fields_section(
                    "Record content",
                    {
                        "system_prompt": candidate.system_prompt,
                        "additional_prompt": candidate.additional_prompt,
                    },
                    ("system_prompt", "additional_prompt"),
                    "data",
                ),
                json_section(
                    "Evidence",
                    {
                        "dropped_context": candidate.dropped_context,
                        "knowledge_tree": candidate.knowledge_tree,
                    },
                    "evidence",
                ),
                json_section(
                    "Metadata",
                    {
                        "fields": fields,
                        "used_tokens": candidate.used_tokens,
                        "status": candidate.status,
                        "note": candidate.note,
                    },
                    "metadata",
                ),
            ),
            stratum=stratum,
            verdict=verdict,
        )

    def apply(self, index: int, action: str) -> None:
        candidate = self.candidates[index]
        if action == STATUS_APPROVED:
            approve(candidate)
        elif action == STATUS_PINNED:
            pin(candidate)
        elif action == STATUS_REJECTED:
            reject(candidate)
        elif action == STATUS_PENDING:
            candidate.status = STATUS_PENDING
            candidate.note = ""
        else:
            raise ValueError(f"unsupported {self.kind} action: {action}")
        save_candidates(self.candidates, self.path)
