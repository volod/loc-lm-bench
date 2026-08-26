"""The one name/relation key the extraction, the graph, and the axiom layer all merge on.

Two stages that disagree about when two surfaces are "the same thing" report about different
graphs. `graph/build.py` keys a node on the folded entity name, so an axiom that says "one subject
has at most one object" has to fold subjects the same way, or it would report a violation the
graph never merged (or miss one it did).
"""


def normalize_name(name: str) -> str:
    """Case- and whitespace-insensitive key for an entity name or a fact endpoint."""
    return " ".join(name.split()).casefold()


def normalize_relation(relation: str) -> str:
    """The same folding for a relation surface, so an axiom matches the extractor's wording."""
    return " ".join(relation.split()).casefold()
