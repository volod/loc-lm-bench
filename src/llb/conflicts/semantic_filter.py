"""Select claim-bearing chunks before semantic calibration and candidate generation.

The filter is corpus-relative. In particular, publication metadata is not recognized from words
such as "publication" or "bulletin": repeated blocks are grouped by their normalized deepest
Markdown heading and confirmed from the tokens and numeric fields they share across documents.
This keeps the detector useful across languages and source systems without teaching it a growing
list of metadata labels.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import re

from llb.conflicts.constants import (
    MIN_CLAIM_TOKENS,
    MIN_METADATA_BLOCK_DOCUMENTS,
    MIN_METADATA_NUMERIC_TOKEN_FRACTION,
    MIN_METADATA_SHARED_COVERAGE,
    MIN_METADATA_SHARED_TOKENS,
)
from llb.core.contracts.rag import ChunkRecord
from llb.rag.lexical import tokenize

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def claim_token_count(text: str) -> int:
    """Content tokens in text, ignoring PDF page/provenance comments."""
    return len(tokenize(_HTML_COMMENT.sub(" ", text)))


@dataclass(frozen=True)
class ContentSelection:
    """Comparable ordinals plus disjoint exclusion-reason counts."""

    ordinals: set[int]
    front_matter: int
    low_content: int
    metadata_blocks: int

    def stats(self) -> dict[str, int]:
        return {
            "excluded_front_matter_chunks": self.front_matter,
            "excluded_low_content_chunks": self.low_content,
            "excluded_metadata_block_chunks": self.metadata_blocks,
        }


def _deepest_heading_key(chunk: ChunkRecord) -> tuple[str, ...] | None:
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        return None
    headers = metadata.get("headers")
    if not isinstance(headers, dict):
        return None
    levels = [
        (int(name[1:]), value)
        for name, value in headers.items()
        if isinstance(name, str)
        and name.startswith("h")
        and name[1:].isdigit()
        and isinstance(value, str)
    ]
    if not levels:
        return None
    key = tuple(tokenize(max(levels)[1]))
    return key or None


def _numeric_fraction(tokens: list[str]) -> float:
    return sum(token.isdigit() for token in tokens) / len(tokens) if tokens else 0.0


def repeated_metadata_ordinals(chunks: list[ChunkRecord], candidates: set[int]) -> set[int]:
    """Repeated structured metadata blocks among otherwise claim-sized body chunks.

    A heading must occur in multiple documents and at most once in every participating document.
    Pairwise token coverage supplies the near-identical-block check; numeric density distinguishes
    variable publication/registry records from repeated claim prose under ordinary shared section
    names. Both signals are structural and derived from the current corpus.
    """
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    token_lists: dict[int, list[str]] = {}
    token_sets: dict[int, set[str]] = {}
    for ordinal in sorted(candidates):
        key = _deepest_heading_key(chunks[ordinal])
        if key is None:
            continue
        groups[key].append(ordinal)
        tokens = tokenize(chunks[ordinal]["text"])
        token_lists[ordinal] = tokens
        token_sets[ordinal] = set(tokens)

    excluded: set[int] = set()
    for ordinals in groups.values():
        per_doc = Counter(chunks[ordinal]["doc_id"] for ordinal in ordinals)
        if len(per_doc) < MIN_METADATA_BLOCK_DOCUMENTS or max(per_doc.values()) > 1:
            continue
        for position, left in enumerate(ordinals):
            left_tokens = token_lists[left]
            if _numeric_fraction(left_tokens) < MIN_METADATA_NUMERIC_TOKEN_FRACTION:
                continue
            for right in ordinals[position + 1 :]:
                right_tokens = token_lists[right]
                if _numeric_fraction(right_tokens) < MIN_METADATA_NUMERIC_TOKEN_FRACTION:
                    continue
                shared = token_sets[left] & token_sets[right]
                if len(shared) < MIN_METADATA_SHARED_TOKENS:
                    continue
                left_coverage = len(shared) / len(token_sets[left])
                right_coverage = len(shared) / len(token_sets[right])
                if min(left_coverage, right_coverage) < MIN_METADATA_SHARED_COVERAGE:
                    continue
                excluded.update((left, right))
    return excluded


def select_content_chunks(
    chunks: list[ChunkRecord],
    body_offsets: dict[str, int],
    *,
    min_tokens: int = MIN_CLAIM_TOKENS,
) -> ContentSelection:
    """Select semantic-comparable chunks and account for one exclusion reason per chunk."""
    front_matter: set[int] = set()
    low_content: set[int] = set()
    candidates: set[int] = set()
    for ordinal, chunk in enumerate(chunks):
        if int(chunk["char_end"]) <= body_offsets.get(chunk["doc_id"], 0):
            front_matter.add(ordinal)
        elif claim_token_count(chunk["text"]) < min_tokens:
            low_content.add(ordinal)
        else:
            candidates.add(ordinal)
    metadata_blocks = repeated_metadata_ordinals(chunks, candidates)
    return ContentSelection(
        ordinals=candidates - metadata_blocks,
        front_matter=len(front_matter),
        low_content=len(low_content),
        metadata_blocks=len(metadata_blocks),
    )
