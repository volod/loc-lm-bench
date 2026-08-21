"""Resolve graph entity nodes that denote one entity but do not share a normalized name.

The graph builder keys a node on `_norm(name)`, a full-string equality test: in Ukrainian a
surname alone, a full name, an initialed form, and an inflected case form key differently, so one
entity becomes several nodes and its mentions, degree, and community membership split across them.

This package proposes canonical node clusters as an OVERLAY beside the built graph and measures
what applying one would do to the graph lane. It never rewrites the stored graph -- see the
confidence contract in `docs/design/spec.md#entity-resolution-and-record-linkage`.
"""
