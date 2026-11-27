"""One structured edit a model may propose. The guard decides whether it is applied."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Proposal:
    """Offsets are into the block's primary reading. `visual` claims the crop shows the glyph."""

    start: int
    end: int
    replacement: str
    confusion_class: str
    evidence: str
    visual: bool


@dataclass(frozen=True, slots=True)
class BlockPrompt:
    """The crop and the bounded text around one block. Neighbour pages are text only."""

    page_id: str
    block_id: str
    text: str
    before: str
    after: str
    previous_page: str
    next_page: str
    crop_png: bytes


@dataclass(frozen=True, slots=True)
class ProposalResult:
    """Proposals from one reply. `unparsed` means the reply was not a structured edit list."""

    proposals: tuple[Proposal, ...] = ()
    unparsed: bool = False


class Proposer(Protocol):
    """Ask a model about one block. A transport failure raises; a bad reply does not."""

    def propose(self, prompt: BlockPrompt) -> ProposalResult: ...
