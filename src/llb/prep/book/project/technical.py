"""Keep a technical body's printed label with it across projection parts."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from llb.core.contracts.data_prep.book_structure import DocumentStructure, ItemKind, NumberedItem
from llb.prep.book.render.catalog import canonical
from llb.prep.book.render.constants import FURNITURE_ROLES
from llb.prep.book.render.types import RenderBlock
from llb.prep.book.structure.text import norm_space


@dataclass(frozen=True, slots=True)
class TechnicalIdentity:
    kind: ItemKind
    defining_block_id: str


def technical_identities(
    blocks: list[RenderBlock], structure: DocumentStructure
) -> dict[str, TechnicalIdentity]:
    """Use numbered definitions and attach bodies only where the layout or number agrees."""
    pages: dict[int, list[RenderBlock]] = defaultdict(list)
    for block in blocks:
        if block.role not in FURNITURE_ROLES:
            pages[block.pdf_page].append(block)
    definitions = {
        item.block_id: item for item in structure.inventory if item.evidence != "page-footer"
    }
    identities = {
        block_id: TechnicalIdentity(item.kind, block_id) for block_id, item in definitions.items()
    }
    for page in pages.values():
        _assign_bodies(page, definitions, identities)
    return identities


def _assign_bodies(
    page: list[RenderBlock],
    definitions: dict[str, NumberedItem],
    identities: dict[str, TechnicalIdentity],
) -> None:
    bodies: dict[str, TechnicalIdentity] = {}
    for index, block in enumerate(page):
        kind = _body_kind(block)
        if kind is None or block.block_id in definitions:
            continue
        defining = _defining_block(page, index, kind, definitions)
        bodies[block.block_id] = TechnicalIdentity(kind, defining or block.block_id)
    claims = Counter(identity.defining_block_id for identity in bodies.values())
    for block_id, identity in bodies.items():
        # Two grids or displays claiming one label are independent blocks until disambiguated.
        identities[block_id] = (
            identity
            if claims[identity.defining_block_id] == 1
            else TechnicalIdentity(identity.kind, block_id)
        )


def _body_kind(block: RenderBlock) -> ItemKind | None:
    if block.role == "table":
        return "table"
    if block.role == "formula":
        return "equation"
    if block.role == "figure":
        return "figure"
    return None


def _defining_block(
    page: list[RenderBlock], index: int, kind: ItemKind, definitions: dict[str, NumberedItem]
) -> str | None:
    block = page[index]
    candidates = [
        (neighbor, definitions[neighbor.block_id])
        for neighbor in page
        if neighbor.block_id in definitions and definitions[neighbor.block_id].kind == kind
    ]
    adjacent = {neighbor.block_id for neighbor in page[max(0, index - 1) : index + 2]}
    matches = _matching_labels(block, kind, candidates, adjacent)
    return matches[0].block_id if len(matches) == 1 else None


def _matching_labels(
    block: RenderBlock,
    kind: ItemKind,
    candidates: list[tuple[RenderBlock, NumberedItem]],
    adjacent: set[str],
) -> list[RenderBlock]:
    if kind == "equation" and block.formula and block.formula.equation_number:
        number = canonical(block.formula.equation_number)
        return [neighbor for neighbor, item in candidates if canonical(item.number) == number]
    if kind == "table" and block.table and block.table.caption:
        caption = norm_space(block.table.caption)
        return [neighbor for neighbor, _ in candidates if norm_space(neighbor.text) == caption]
    return [
        neighbor
        for neighbor, _ in candidates
        if neighbor.block_id in adjacent and _same_column(block, neighbor)
    ]


def _same_column(left: RenderBlock, right: RenderBlock) -> bool:
    a, b = left.region, right.region
    return a is not None and b is not None and max(a.x0, b.x0) < min(a.x1, b.x1)
