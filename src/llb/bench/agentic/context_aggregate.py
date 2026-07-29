"""Aggregate-safe facts for agent observation trimming.

A positional head-and-tail trim destroys counts that sit in the middle of a search hit list --
exactly the facts a `search-count` task needs. Compaction faces the same loss when a free-text
summary is not asked for a total. This module extracts machine-computed facts (hit count, total
length, matched doc ids) from a tool observation so a trim or a compaction prompt can PREPEND
them and keep count questions answerable after the body is shortened.
"""

import re

from typing_extensions import TypedDict

from llb.bench.tool_world import OBS_NO_RESULTS

# Search hits are rendered as `[doc_id] text` lines (`ToolWorld._search`).
_HIT_LINE = re.compile(r"^\[([^\]]+)\] ", re.MULTILINE)

# ASCII-safe UA marker: the model must see that these numbers are machine facts, not truncated
# body text that might itself have been elided.
_HEADER = "[агрегат: hits={hits} chars={chars} docs={docs}]"
_HEADER_CHARS_ONLY = "[агрегат: chars={chars}]"


class AggregateFacts(TypedDict):
    """Facts a positional trim would destroy on a search (or long) observation."""

    chars: int
    hits: int
    doc_ids: list[str]
    is_search_hits: bool


def extract_aggregate_facts(observation: str) -> AggregateFacts:
    """Parse hit lines / no-results marker; always report total char length."""
    doc_ids = _HIT_LINE.findall(observation)
    stripped = observation.strip()
    is_search = bool(doc_ids) or stripped == OBS_NO_RESULTS
    return {
        "chars": len(observation),
        "hits": len(doc_ids),
        "doc_ids": list(doc_ids),
        "is_search_hits": is_search,
    }


def format_aggregate_header(observation: str) -> str:
    """One-line header naming the facts a middle-of-list trim would drop."""
    facts = extract_aggregate_facts(observation)
    if facts["is_search_hits"]:
        docs = ",".join(facts["doc_ids"]) if facts["doc_ids"] else "-"
        return _HEADER.format(hits=facts["hits"], chars=facts["chars"], docs=docs)
    return _HEADER_CHARS_ONLY.format(chars=facts["chars"])


def with_aggregate_header(observation: str) -> str:
    """Prepend the aggregate header when it is not already the leading line."""
    if observation.startswith("[агрегат:"):
        return observation
    return f"{format_aggregate_header(observation)}\n{observation}"


def task_kind(item_id: str) -> str:
    """Map a task id to the generator's kind label (`count` / `locate` / `other`)."""
    low = item_id.casefold()
    if "search-count" in low or low.startswith("count"):
        return "count"
    if "search-locate" in low or "locate" in low:
        return "locate"
    return "other"
