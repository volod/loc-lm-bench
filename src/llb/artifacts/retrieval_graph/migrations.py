"""One-step transformations from an older retrieval or graph record to the current one.

Each restates what the OLD reader was already doing, so a migrated record and the record a
current writer would produce say the same thing. Nothing here invents a value: a field the old
form could not carry stays absent, and only a field the reader supplied from a constant is stated.
"""

from llb.core.contracts.retrieval_graph.stores import STORE_META_SCHEMA_ID

# The value `llb.rag.refresh.store_refresh` and `llb.cli.rag.duplicate_residue` read out of a meta
# that does not state it. Restating it is the whole migration: the store behaved this way, the
# record simply did not say so.
LEGACY_COLLAPSE_DUPLICATES = True
LEGACY_DUPLICATE_TIER = "exact"


def store_meta_v1_to_v2(record: dict[str, object]) -> dict[str, object]:
    """State the duplicate-collapse knobs a reader defaulted, and declare no index members.

    A store meta written before duplicate collapse shipped leaves both knobs out; one written
    after states them and is carried through untouched. `index_members` is empty either way --
    an older generation never recorded which opaque files it was built with, and an empty list
    means "this generation does not state its index members", never "it has none".
    """
    migrated = {
        **record,
        "schema_id": STORE_META_SCHEMA_ID,
        "schema_version": "2.0.0",
        "index_members": [],
    }
    migrated.setdefault("collapse_duplicates", LEGACY_COLLAPSE_DUPLICATES)
    migrated.setdefault("duplicate_tier", LEGACY_DUPLICATE_TIER)
    return migrated
