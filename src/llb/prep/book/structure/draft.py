"""The mutable node the hierarchy is built with, before it becomes the contract model."""

from dataclasses import dataclass, field

from llb.core.contracts.data_prep.book_structure import NODE_EVIDENCE, NodeEvidence, NodeLevel

RANK: dict[str, int] = {
    "book": 0,
    "part": 1,
    "chapter": 2,
    "section": 3,
    "subsection": 4,
    "page-range": 5,
}
_LEVELS: dict[str, NodeLevel] = {
    "book": "book",
    "part": "part",
    "chapter": "chapter",
    "section": "section",
    "subsection": "subsection",
    "page-range": "page-range",
}
_EVIDENCE: dict[str, NodeEvidence] = {name: name for name in NODE_EVIDENCE}


@dataclass
class Draft:
    """One hierarchy node while its range and its evidence are still being filled in."""

    level: str
    number: str
    title: str
    pdf_start: int
    pdf_end: int
    block_id: str | None
    evidence: list[str]
    fallback: bool
    children: list["Draft"] = field(default_factory=list)


def add_evidence(node: Draft, name: str) -> None:
    """Record one piece of evidence once."""
    if name not in node.evidence:
        node.evidence.append(name)


def level_of(name: str) -> NodeLevel:
    """The contract level. An unknown name stays a chapter rather than failing the file."""
    return _LEVELS.get(name, "chapter")


def evidence_of(name: str) -> NodeEvidence:
    """The contract evidence tag. An unknown name is recorded as fallback."""
    return _EVIDENCE.get(name, "fallback")
