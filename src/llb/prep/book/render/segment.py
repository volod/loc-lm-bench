"""Turn reading order into section files, splitting only where a boundary allows.

A formula, a table, and a figure with its caption stay whole. A paragraph is
split only when that paragraph alone is longer than the cap. Furniture on a
page stays with the section that owns the page's text.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

from llb.core.contracts.data_prep.book_structure import DocumentStructure, StructureNode
from llb.prep.book.render.constants import FURNITURE_ROLES
from llb.prep.book.render.join import paragraph_runs, stitch_columns
from llb.prep.book.render.names import structure_roles
from llb.prep.book.render.prose import join_lines, passage
from llb.prep.book.render.types import RenderBlock, Unit

Href = Callable[[int], str]


@dataclass
class Span:
    """One contiguous run of blocks owned by one structure node."""

    node: StructureNode
    depth: int
    blocks: list[RenderBlock]


def prepare(blocks: list[RenderBlock], structure: DocumentStructure) -> list[RenderBlock]:
    """Relabel body lines that structure already called a heading, header, or page number."""
    roles = structure_roles(structure)
    prepared: list[RenderBlock] = []
    for block in blocks:
        role = roles.get(block.block_id, "")
        if role and block.role in ("body", "unknown"):
            prepared.append(replace(block, role=role))
        else:
            prepared.append(block)
    return prepared


def section_spans(blocks: list[RenderBlock], structure: DocumentStructure) -> list[Span]:
    """Each block belongs to the deepest node that has started by that block."""
    nodes = _walk(structure.hierarchy)
    depth = _depths(structure.hierarchy)
    index = {
        block.block_id: (block.pdf_page, block.reading_order) for block in blocks if block.block_id
    }
    owners = [_owner(block, nodes, depth, index) for block in blocks]
    owners = _move_furniture(blocks, owners)
    return _spans(blocks, owners, depth)


def pack_span(span: Span, cap: int, href: Href) -> list[list[Unit]]:
    """The span's units, packed into parts that stay under `cap` when they can."""
    return pack(_pairs(stitch_columns(_units(span.blocks, href))), cap)


def pack(units: list[Unit], cap: int) -> list[list[Unit]]:
    """Greedy parts. An atomic unit that does not fit starts a part and stays whole."""
    parts: list[list[Unit]] = []
    current: list[Unit] = []
    size = 0
    for unit in units:
        current, size = _place(parts, current, size, unit, cap)
    if current:
        parts.append(current)
    return parts


def _place(
    parts: list[list[Unit]], current: list[Unit], size: int, unit: Unit, cap: int
) -> tuple[list[Unit], int]:
    if unit.splittable and len(unit.text) > cap:
        if current:
            parts.append(current)
        pieces = _split(unit, cap)
        parts.extend([piece] for piece in pieces[:-1])
        return [pieces[-1]], len(pieces[-1].text)
    if current and size + len(unit.text) > cap:
        parts.append(current)
        current, size = [], 0
    current.append(unit)
    return current, size + len(unit.text)


def _split(unit: Unit, cap: int) -> list[Unit]:
    pieces = _cuts(unit.text.strip(), cap)
    return [
        Unit("paragraph", unit.blocks, piece, False, index == 0)
        for index, piece in enumerate(pieces)
    ]


def _cuts(text: str, cap: int) -> list[str]:
    pieces: list[str] = []
    rest = text
    while len(rest) > cap:
        cut = rest.rfind(" ", 0, cap + 1)
        if cut <= 0:
            cut = cap
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces or [text]


def _units(blocks: list[RenderBlock], href: Href) -> list[Unit]:
    return [_unit(run, href) for run in paragraph_runs(blocks)]


def _unit(blocks: list[RenderBlock], href: Href) -> Unit:
    role = blocks[0].role
    if role == "body":
        return Unit("paragraph", tuple(blocks), join_lines(blocks, href), True)
    text = passage(blocks[0], href)
    return Unit(role, tuple(blocks), text, False)


def _pairs(units: list[Unit]) -> list[Unit]:
    bound: list[Unit] = []
    index = 0
    while index < len(units):
        nxt = units[index + 1] if index + 1 < len(units) else None
        if nxt is not None and _is_pair(units[index], nxt):
            bound.append(_merge(units[index], nxt))
            index += 2
        else:
            bound.append(units[index])
            index += 1
    return bound


def _is_pair(left: Unit, right: Unit) -> bool:
    if not left.blocks or not right.blocks:
        return False
    if left.blocks[0].pdf_page != right.blocks[0].pdf_page:
        return False
    return {left.role, right.role} == {"figure", "caption"}


def _merge(left: Unit, right: Unit) -> Unit:
    text = "\n\n".join(piece for piece in (left.text, right.text) if piece)
    return Unit("pair", left.blocks + right.blocks, text, False)


def _owner(
    block: RenderBlock,
    nodes: list[StructureNode],
    depth: dict[str, int],
    index: dict[str, tuple[int, int]],
) -> StructureNode:
    if block.failed:
        return _deepest(nodes, depth, block.pdf_page)
    started = [node for node in nodes if _covers(node, block, index)]
    if not started:
        return nodes[0]
    return max(started, key=lambda node: _rank(node, depth, index))


def _covers(node: StructureNode, block: RenderBlock, index: dict[str, tuple[int, int]]) -> bool:
    if block.pdf_page < node.pdf_start or block.pdf_page > node.pdf_end:
        return False
    if not node.block_id or node.block_id not in index:
        return True
    return (block.pdf_page, block.reading_order) >= index[node.block_id]


def _deepest(nodes: list[StructureNode], depth: dict[str, int], pdf_page: int) -> StructureNode:
    covering = [node for node in nodes if node.pdf_start <= pdf_page <= node.pdf_end]
    if not covering:
        return nodes[0]
    return max(covering, key=lambda node: _rank(node, depth, {}))


def _rank(
    node: StructureNode, depth: dict[str, int], index: dict[str, tuple[int, int]]
) -> tuple[int, int, int]:
    """Deeper wins. A later sibling wins a tie, so its heading takes the blocks after it."""
    start = index.get(node.block_id or "", (node.pdf_start, -1))
    return (depth[node.node_id], start[0], start[1])


def _move_furniture(blocks: list[RenderBlock], owners: list[StructureNode]) -> list[StructureNode]:
    result = list(owners)
    groups: dict[int, list[int]] = {}
    for index, block in enumerate(blocks):
        groups.setdefault(block.pdf_page, []).append(index)
    for indexes in groups.values():
        _attach(blocks, result, indexes)
    return result


def _attach(blocks: list[RenderBlock], owners: list[StructureNode], indexes: list[int]) -> None:
    content = [index for index in indexes if _content(blocks[index])]
    if not content:
        return
    for index in indexes:
        if _content(blocks[index]):
            continue
        earlier = [item for item in content if item <= index]
        owners[index] = owners[earlier[-1] if earlier else content[0]]


def _content(block: RenderBlock) -> bool:
    return block.role not in FURNITURE_ROLES and not block.failed


def _spans(
    blocks: list[RenderBlock], owners: list[StructureNode], depth: dict[str, int]
) -> list[Span]:
    spans: list[Span] = []
    for block, owner in zip(blocks, owners, strict=True):
        if spans and spans[-1].node.node_id == owner.node_id:
            spans[-1].blocks.append(block)
        else:
            spans.append(Span(owner, depth[owner.node_id], [block]))
    return spans


def _walk(node: StructureNode) -> list[StructureNode]:
    found = [node]
    for child in node.children:
        found.extend(_walk(child))
    return found


def _depths(node: StructureNode) -> dict[str, int]:
    found: dict[str, int] = {}

    def visit(current: StructureNode, depth: int) -> None:
        found[current.node_id] = depth
        for child in current.children:
            visit(child, depth + 1)

    visit(node, 0)
    return found
