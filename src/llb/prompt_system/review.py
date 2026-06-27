"""RAG prompt-system comparison human review loop -- approve / revise / pin / reject prompt-system candidates.

Exposes each generated candidate with its rendered prompts, context-budget breakdown, and
dropped-context report, plus the editable template fields, so an operator can ACCEPT a candidate,
REVISE its fields and re-render (re-fitting the budget), PIN a preferred candidate, or REJECT one
before benchmarking. Candidates persist to JSON so a review session survives across invocations.
Pure + deterministic -- the rendering it calls is the same the benchmark uses.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

from llb.prompt_system.budget import ContextBudget, DroppedContextReport, Tokenizer
from llb.prompt_system.corpus import CorpusPackage
from llb.prompt_system.manifest import prompt_system_id
from llb.prompt_system.template import PromptPackage, TemplateFields, render_package

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REVISED = "revised"
STATUS_PINNED = "pinned"
STATUS_REJECTED = "rejected"
REVIEW_STATUSES = (
    STATUS_PENDING,
    STATUS_APPROVED,
    STATUS_REVISED,
    STATUS_PINNED,
    STATUS_REJECTED,
)


@dataclass(slots=True)
class PromptCandidate:
    """One reviewable prompt-system candidate (rendered + budgeted + status-tracked)."""

    prompt_system_id: str
    fields: TemplateFields
    system_prompt: str
    additional_prompt: str
    dropped_context: DroppedContextReport
    used_tokens: int
    status: str = STATUS_PENDING
    note: str = ""

    def package(self) -> PromptPackage:
        """Reconstruct the harness-usable `PromptPackage` from this candidate."""
        return PromptPackage(
            system_prompt=self.system_prompt,
            additional_prompt=self.additional_prompt,
            fields=self.fields,
            dropped_context=self.dropped_context,
            used_tokens=self.used_tokens,
        )


def make_candidate(
    corpus: CorpusPackage,
    fields: TemplateFields,
    budget: ContextBudget,
    tokenizer: Tokenizer,
) -> PromptCandidate:
    """Render one candidate from template fields, fitting the attached context to the budget."""
    package = render_package(corpus, fields, budget, tokenizer)
    return PromptCandidate(
        prompt_system_id=prompt_system_id(corpus, fields),
        fields=fields,
        system_prompt=package.system_prompt,
        additional_prompt=package.additional_prompt,
        dropped_context=package.dropped_context,
        used_tokens=package.used_tokens,
    )


def approve(candidate: PromptCandidate, note: str = "") -> PromptCandidate:
    candidate.status = STATUS_APPROVED
    candidate.note = note
    return candidate


def reject(candidate: PromptCandidate, note: str = "") -> PromptCandidate:
    candidate.status = STATUS_REJECTED
    candidate.note = note
    return candidate


def pin(candidate: PromptCandidate, note: str = "") -> PromptCandidate:
    candidate.status = STATUS_PINNED
    candidate.note = note
    return candidate


def revise(
    candidate: PromptCandidate,
    new_fields: TemplateFields,
    corpus: CorpusPackage,
    budget: ContextBudget,
    tokenizer: Tokenizer,
    note: str = "",
) -> PromptCandidate:
    """Re-render the candidate with edited template fields (a new id; status=revised)."""
    revised = make_candidate(corpus, new_fields, budget, tokenizer)
    revised.status = STATUS_REVISED
    revised.note = note
    return revised


def candidate_to_dict(candidate: PromptCandidate) -> dict[str, object]:
    data = asdict(candidate)
    return data


def candidate_from_dict(data: dict[str, Any]) -> PromptCandidate:
    return PromptCandidate(
        prompt_system_id=str(data["prompt_system_id"]),
        fields=TemplateFields(**data["fields"]),
        system_prompt=str(data["system_prompt"]),
        additional_prompt=str(data["additional_prompt"]),
        dropped_context=data["dropped_context"],
        used_tokens=int(data["used_tokens"]),
        status=str(data.get("status", STATUS_PENDING)),
        note=str(data.get("note", "")),
    )


def save_candidates(candidates: list[PromptCandidate], path: Path | str) -> None:
    payload = [candidate_to_dict(c) for c in candidates]
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_candidates(path: Path | str) -> list[PromptCandidate]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [candidate_from_dict(item) for item in raw]


@dataclass(slots=True)
class ReviewSummary:
    """A compact roll-up of a review session by status (for the operator-facing CLI)."""

    n: int
    by_status: dict[str, int] = field(default_factory=dict)
    pinned: list[str] = field(default_factory=list)


def summarize_review(candidates: list[PromptCandidate]) -> ReviewSummary:
    by_status: dict[str, int] = {}
    pinned: list[str] = []
    for candidate in candidates:
        by_status[candidate.status] = by_status.get(candidate.status, 0) + 1
        if candidate.status == STATUS_PINNED:
            pinned.append(candidate.prompt_system_id)
    return ReviewSummary(n=len(candidates), by_status=by_status, pinned=pinned)
