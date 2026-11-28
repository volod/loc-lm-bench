"""Build a structural graph from a book's printed spans and resolved sidecars."""

from collections import defaultdict

from llb.core.contracts.data_prep.book_structure import ITEM_KINDS
from llb.graph.book_inputs import BlockKey, BookDocument, BookSources, PrintedBlock
from llb.graph.model import UNKNOWN_CONFIDENCE, GraphEdge, GraphMention, GraphNode, KnowledgeGraph

SECTION = "section"
PAGE = "page"
INDEX_TERM = "index-term"
CONTAINS = "contains"
CITES = "cites"
INDEXES = "indexes"

type NodeKey = tuple[str, str | int | BlockKey]
type SpanKey = tuple[str, int, int]


def _span_key(mention: GraphMention) -> SpanKey:
    return mention["doc_id"], mention["char_start"], mention["char_end"]


class _BookGraph:
    def __init__(self) -> None:
        self.graph = KnowledgeGraph()
        self.nodes: dict[NodeKey, GraphNode] = {}
        self.mentions: dict[int, set[SpanKey]] = defaultdict(set)
        self.edges: set[tuple[int, int, str, SpanKey]] = set()

    def node(self, key: NodeKey, name: str, evidence: GraphMention) -> GraphNode:
        if key not in self.nodes:
            node = GraphNode(len(self.graph.nodes), name, key[0], UNKNOWN_CONFIDENCE)
            self.nodes[key] = node
            self.graph.nodes.append(node)
        node = self.nodes[key]
        self.mention(node, evidence)
        return node

    def mention(self, node: GraphNode, evidence: GraphMention) -> None:
        span = _span_key(evidence)
        if span not in self.mentions[node.node_id]:
            self.mentions[node.node_id].add(span)
            node.mentions.append(evidence)

    def link(
        self, source: GraphNode, target: GraphNode, relation: str, evidence: GraphMention
    ) -> None:
        key = source.node_id, target.node_id, relation, _span_key(evidence)
        if source is not target and key not in self.edges:
            self.edges.add(key)
            self.graph.edges.append(
                GraphEdge(len(self.graph.edges), source.node_id, target.node_id, relation, evidence)
            )


def build_book_reference_graph(sources: BookSources) -> KnowledgeGraph:
    """Only hierarchy, physical pages, defining blocks, and explicit printed references."""
    built = _BookGraph()
    blocks_by_doc: dict[str, list[PrintedBlock]] = defaultdict(list)
    for block in sources.blocks.values():
        blocks_by_doc[block.key[0]].append(block)
    sections = _sections(built, sources)
    technical = _technical(built, sources)
    owners = {
        doc_id: _containment(built, doc, blocks_by_doc[doc_id], technical)
        for doc_id, doc in sources.documents.items()
    }
    for doc_id, doc in sources.documents.items():
        _references(built, sources, doc, blocks_by_doc[doc_id], owners[doc_id], sections, technical)
    return built.graph


def _sections(built: _BookGraph, sources: BookSources) -> dict[BlockKey, GraphNode]:
    places = {
        place.heading_block_id: place
        for doc in sources.documents.values()
        for place in doc.references.hierarchy
        if place.level != "book" and place.heading_block_id
    }
    headings = {}
    for block in sources.blocks.values():
        if place := places.get(block.key[1]):
            name = " ".join(part for part in (place.number, place.title) if part)
            headings[block.key] = built.node((SECTION, place.node_id), name, block.mention)
    return headings


def _technical(built: _BookGraph, sources: BookSources) -> dict[BlockKey, GraphNode]:
    nodes = {}
    for doc in sources.documents.values():
        for item in doc.references.technical_blocks:
            part = sources.target(item.span)
            label = sources.target(item.definition)
            if part.key[0] != doc.source.doc_id or part.page != label.page:
                raise ValueError(f"{doc.source.doc_id}: technical block and its label disagree")
            key = item.kind, label.key
            name = label.mention["text"].splitlines()[0]
            built.node(key, name, label.mention)
            nodes[part.key] = built.node(key, name, part.mention)
    return nodes


def _page_node(built: _BookGraph, block: PrintedBlock) -> GraphNode:
    return built.node((PAGE, block.page), f"PDF page {block.page}", block.mention)


def _containment(
    built: _BookGraph,
    doc: BookDocument,
    blocks: list[PrintedBlock],
    technical: dict[BlockKey, GraphNode],
) -> GraphNode | None:
    path = doc.references.hierarchy
    parents = [built.nodes.get((SECTION, place.node_id)) for place in path]
    for parent, child in zip(parents, parents[1:]):
        if parent is not None and child is not None:
            built.link(parent, child, CONTAINS, child.mentions[0])
    section = next((parent for parent in reversed(parents) if parent is not None), None)
    pages: dict[int, GraphNode] = {}
    for block in blocks:
        if block.page not in pages:
            page = pages[block.page] = _page_node(built, block)
            if section is not None:
                built.link(section, page, CONTAINS, block.mention)
        if block.key in technical:
            built.link(pages[block.page], technical[block.key], CONTAINS, block.mention)
    return section


def _references(
    built: _BookGraph,
    sources: BookSources,
    doc: BookDocument,
    blocks: list[PrintedBlock],
    section: GraphNode | None,
    sections: dict[BlockKey, GraphNode],
    technical: dict[BlockKey, GraphNode],
) -> None:
    for citation in doc.references.citations:
        cue = doc.mention(citation.cue_start, citation.cue_end)
        if citation.target is None:
            continue
        target = sources.target(citation.target)
        if citation.kind == "index" or _inside(cue, target.mention):
            continue
        destination = _destination(built, target, citation.kind, sections, technical)
        _alias(destination, citation.key)
        _cite(built, blocks, section, destination, cue)
    _index_links(built, doc)


def _cite(
    built: _BookGraph,
    blocks: list[PrintedBlock],
    section: GraphNode | None,
    destination: GraphNode,
    cue: GraphMention,
) -> None:
    # A reference can straddle OCR blocks. Its first character still identifies the page.
    start = cue["char_start"]
    citing = next((block for block in blocks if block.key[2] <= start < block.key[3]), None)
    if citing is None:
        raise ValueError(f"{cue['doc_id']}: citation cue has no printed source block")
    origin = section if section is not None else _page_node(built, citing)
    built.mention(origin, citing.mention)
    _alias(origin, citing.mention["text"])
    built.link(origin, destination, CITES, cue)


def _destination(
    built: _BookGraph,
    target: PrintedBlock,
    kind: str,
    sections: dict[BlockKey, GraphNode],
    technical: dict[BlockKey, GraphNode],
) -> GraphNode:
    if kind == PAGE:
        return _page_node(built, target)
    if kind in ("chapter", SECTION) and target.key in sections:
        return sections[target.key]
    if kind in ITEM_KINDS:
        if target.key in technical and technical[target.key].type == kind:
            return technical[target.key]
        return built.node((kind, target.key), target.mention["text"], target.mention)
    raise ValueError(f"{target.key[0]}: invalid {kind} citation target")


def _index_links(built: _BookGraph, doc: BookDocument) -> None:
    for entry in doc.references.index:
        term = doc.mention(entry.term_start, entry.term_end)
        key = term["text"].casefold()
        node = built.node((INDEX_TERM, key), term["text"], term)
        for page in entry.pages:
            evidence = doc.mention(page.char_start, page.char_end)
            target = built.nodes.get((PAGE, page.pdf_page))
            if target is None:
                raise ValueError(
                    f"{doc.source.doc_id}: indexed PDF page {page.pdf_page} has no text"
                )
            _alias(target, f"page {page.printed}")
            built.link(node, target, INDEXES, evidence)


def _alias(node: GraphNode, text: str) -> None:
    if text != node.name and text not in node.aliases:
        node.aliases.append(text)


def _inside(inner: GraphMention, outer: GraphMention) -> bool:
    return (
        inner["doc_id"] == outer["doc_id"]
        and outer["char_start"] <= inner["char_start"] < inner["char_end"] <= outer["char_end"]
    )
