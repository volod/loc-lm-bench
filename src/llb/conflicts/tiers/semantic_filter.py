"""Select claim-bearing chunks before semantic calibration and candidate generation.

The filter is corpus-relative. In particular, publication metadata is not recognized from words
such as "publication" or "bulletin": repeated blocks are grouped by their normalized deepest
Markdown heading and confirmed from the tokens and numeric fields they share across documents.
This keeps the detector useful across languages and source systems without teaching it a growing
list of metadata labels.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import re

from llb.conflicts.constants import (
    MIN_CLAIM_TOKENS,
    MIN_METADATA_BLOCK_DOCUMENTS,
    MIN_METADATA_NUMERIC_TOKEN_FRACTION,
    MIN_METADATA_SHARED_COVERAGE,
    MIN_METADATA_SHARED_TOKENS,
)
from llb.core.contracts.rag import ChunkRecord
from llb.rag.vector_store.lexical import tokenize

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)


def claim_token_count(text: str) -> int:
    """Content tokens in text, ignoring PDF page/provenance comments."""
    return len(tokenize(_HTML_COMMENT.sub(" ", text)))


@dataclass(frozen=True)
class ContentSelection:
    """Comparable ordinals plus the disjoint exclusion sets -- one reason per excluded chunk.

    The reasons are kept as ORDINAL SETS rather than as counts because the chunk each ordinal names
    still exists here and nowhere downstream: the per-document exclusion record
    (`document_exclusions.py`) is folded from them, and a count cannot say which document.
    """

    ordinals: set[int]
    front_matter: set[int]
    low_content: set[int]
    metadata_blocks: set[int]
    # Claim tokens per low-content ordinal. The floor is the ONLY one of the three exclusions a
    # `--min-claim-tokens` change moves, so this is what turns "lower the floor" into a floor VALUE.
    low_content_tokens: dict[int, int] = field(default_factory=dict)

    def stats(self) -> dict[str, int]:
        return {
            "excluded_front_matter_chunks": len(self.front_matter),
            "excluded_low_content_chunks": len(self.low_content),
            "excluded_metadata_block_chunks": len(self.metadata_blocks),
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


def _metadata_groups(
    chunks: list[ChunkRecord], candidates: set[int]
) -> tuple[
    dict[tuple[str, ...], list[int]],
    dict[int, list[str]],
    dict[int, set[str]],
]:
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
    return groups, token_lists, token_sets


def _is_cross_document_group(chunks: list[ChunkRecord], ordinals: list[int]) -> bool:
    per_doc = Counter(chunks[ordinal]["doc_id"] for ordinal in ordinals)
    return len(per_doc) >= MIN_METADATA_BLOCK_DOCUMENTS and max(per_doc.values()) <= 1


def _is_metadata_pair(
    left: int,
    right: int,
    token_lists: dict[int, list[str]],
    token_sets: dict[int, set[str]],
) -> bool:
    if _numeric_fraction(token_lists[left]) < MIN_METADATA_NUMERIC_TOKEN_FRACTION:
        return False
    if _numeric_fraction(token_lists[right]) < MIN_METADATA_NUMERIC_TOKEN_FRACTION:
        return False
    shared = token_sets[left] & token_sets[right]
    if len(shared) < MIN_METADATA_SHARED_TOKENS:
        return False
    coverage = min(len(shared) / len(token_sets[ordinal]) for ordinal in (left, right))
    return coverage >= MIN_METADATA_SHARED_COVERAGE


def _metadata_ordinals_in_group(
    ordinals: list[int],
    token_lists: dict[int, list[str]],
    token_sets: dict[int, set[str]],
) -> set[int]:
    excluded: set[int] = set()
    for position, left in enumerate(ordinals):
        for right in ordinals[position + 1 :]:
            if _is_metadata_pair(left, right, token_lists, token_sets):
                excluded.update((left, right))
    return excluded


def repeated_metadata_ordinals(chunks: list[ChunkRecord], candidates: set[int]) -> set[int]:
    """Repeated structured metadata blocks among otherwise claim-sized body chunks.

    A heading must occur in multiple documents and at most once in every participating document.
    Pairwise token coverage supplies the near-identical-block check; numeric density distinguishes
    variable publication/registry records from repeated claim prose under ordinary shared section
    names. Both signals are structural and derived from the current corpus.
    """
    groups, token_lists, token_sets = _metadata_groups(chunks, candidates)
    excluded: set[int] = set()
    for ordinals in groups.values():
        if _is_cross_document_group(chunks, ordinals):
            excluded.update(_metadata_ordinals_in_group(ordinals, token_lists, token_sets))
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
    low_content_tokens: dict[int, int] = {}
    candidates: set[int] = set()
    for ordinal, chunk in enumerate(chunks):
        if int(chunk["char_end"]) <= body_offsets.get(chunk["doc_id"], 0):
            front_matter.add(ordinal)
            continue
        tokens = claim_token_count(chunk["text"])
        if tokens < min_tokens:
            low_content.add(ordinal)
            low_content_tokens[ordinal] = tokens
        else:
            candidates.add(ordinal)
    metadata_blocks = repeated_metadata_ordinals(chunks, candidates)
    return ContentSelection(
        ordinals=candidates - metadata_blocks,
        front_matter=front_matter,
        low_content=low_content,
        metadata_blocks=metadata_blocks,
        low_content_tokens=low_content_tokens,
    )
