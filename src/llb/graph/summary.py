"""Optional LLM community summaries -- recorded as a TAGGED DIAGNOSTIC, never span-scored.

A `global_community` answer is built from offset-bearing member spans (so it scores on the span
metric). An LLM-written one-paragraph summary of a community is a useful narrative artifact, but it
is an UN-GROUNDED abstraction -- it has no source span -- so it must never enter the metric. This
mirrors the `--score-semantic` discipline (recorded but not ranked): the summary is attached to the
store as a diagnostic (`community_summaries`) and persisted to its own file, and the retrieval path
(`GraphStore.retrieve`) never returns it.

`complete` is the injectable M4.4 endpoint callable, so this is unit-tested with a fake endpoint
and never needs a server.
"""

import logging

from llb.graph.model import KnowledgeGraph
from llb.prep.frontier import LLMComplete

_LOG = logging.getLogger(__name__)

# A community with fewer member nodes than this is too small to carry a corpus-level narrative.
MIN_COMMUNITY_SIZE = 3
# Cap how many member surface forms / facts feed one summary prompt (keeps it bounded + cheap).
MAX_SUMMARY_MEMBERS = 12


def community_summary_prompt(names: list[str], relations: list[str]) -> str:
    """Ask for a short Ukrainian narrative summary of one community's entities + relations."""
    entities = ", ".join(names)
    facts = "; ".join(relations) if relations else "(без явних відношень)"
    return (
        "Ти аналітик. Нижче -- спільнота пов'язаних сутностей з україномовного корпусу.\n"
        f"Сутності: {entities}.\n"
        f"Відношення: {facts}.\n"
        "Стисло (1-2 речення) опиши спільну тему цієї спільноти українською. "
        "Не вигадуй фактів поза наведеними."
    )


def summarize_communities(
    graph: KnowledgeGraph,
    complete: LLMComplete,
    *,
    min_size: int = MIN_COMMUNITY_SIZE,
    max_members: int = MAX_SUMMARY_MEMBERS,
) -> dict[str, str]:
    """Generate a diagnostic summary per sizable community. Keys are community ids (as strings,
    for JSON). A failed call is skipped, never fatal (the summary is optional + diagnostic)."""
    by_id = graph.node_by_id()
    summaries: dict[str, str] = {}
    for community_id, member_ids in sorted(graph.community_members().items()):
        if len(member_ids) < min_size:
            continue
        names = [by_id[nid].name for nid in member_ids[:max_members]]
        relations = [
            f"{by_id[e.src].name} -{e.relation}-> {by_id[e.dst].name}"
            for e in graph.edges
            if e.src in set(member_ids) and e.dst in set(member_ids)
        ][:max_members]
        try:
            text = complete(community_summary_prompt(names, relations)).strip()
        except Exception as exc:  # diagnostic only -> skip on any endpoint error
            _LOG.warning("[graph] community %s summary skipped: %s", community_id, exc)
            continue
        if text:
            summaries[str(community_id)] = text
    _LOG.info(
        "[graph] generated %d diagnostic community summaries (not span-scored)", len(summaries)
    )
    return summaries
