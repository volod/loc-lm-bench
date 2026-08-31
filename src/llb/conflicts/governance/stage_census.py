"""Count the lost orderable pairs each attribution stage accounts for.

The headline still names one pair. This census answers the sizing question beside it: how many of
all lost orderable pairs reached the named stage, and how many stopped elsewhere. It stays off the
quadratic pair space by grouping documents by their first stopped stage. For each group, the exact
pair count is the orderable count before removing that group minus the count after removing it.
Both counts use key multisets, so the work is bounded by the number of stages times the documents,
plus one pass over the returned document pairs.
"""

from collections.abc import Collection, Sequence

from llb.conflicts.bundle.document_chunks import DocumentChunks
from llb.conflicts.governance.editions import compare_editions
from llb.conflicts.governance.orderability import orderable_pair_count
from llb.conflicts.governance.stage_rule import (
    PAIR_STAGE_ORDER,
    REPORT_STAGE_ORDER,
    STAGE_EFFORT,
    document_stage,
    stage_of,
)
from llb.core.contracts.common import JsonObject


def _possible_by_stage(
    documents: Sequence[tuple[str, JsonObject]], chunks: DocumentChunks | None
) -> dict[str, int]:
    """Orderable corpus pairs by their first stopped stage, before returned pairs are removed."""
    if chunks is None:
        return {STAGE_EFFORT: orderable_pair_count([governance for _, governance in documents])}

    groups: dict[str, list[tuple[str, JsonObject]]] = {stage: [] for stage in PAIR_STAGE_ORDER}
    for document in documents:
        groups[document_stage(document[0], chunks)].append(document)

    remaining = list(documents)
    remaining_count = orderable_pair_count([governance for _, governance in remaining])
    counts: dict[str, int] = {}
    for stage in PAIR_STAGE_ORDER:
        if not groups[stage]:
            continue
        following = [
            document for document in remaining if document_stage(document[0], chunks) != stage
        ]
        following_count = orderable_pair_count([governance for _, governance in following])
        counts[stage] = remaining_count - following_count
        remaining, remaining_count = following, following_count
    return counts


def lost_pairs_by_stage(
    documents: Sequence[tuple[str, JsonObject]],
    returned_pairs: Collection[tuple[str, str]],
    chunks: DocumentChunks | None,
) -> dict[str, int]:
    """Exact lost-orderable-pair counts, with zero stages omitted in report order.

    Corpus-pair totals are counted from document classes. Returned pairs are already a bounded
    run output and are the only pairs visited individually; each orderable one is subtracted from
    the stage it survived.
    """
    counts = _possible_by_stage(documents, chunks)
    governance = {doc_id: value for doc_id, value in documents}
    for left, right in returned_pairs:
        if left not in governance or right not in governance:
            continue
        if compare_editions(governance[left], governance[right]).newer_side is None:
            continue
        stage = STAGE_EFFORT if chunks is None else stage_of((left, right), chunks)[0]
        counts[stage] = counts.get(stage, 0) - 1
    order = (STAGE_EFFORT, *REPORT_STAGE_ORDER)
    return {stage: counts[stage] for stage in order if counts.get(stage, 0) > 0}
