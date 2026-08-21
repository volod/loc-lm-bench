"""The document record table the edition fit reads, and the specification it is linked under.

The features are the ones the conflict audit already computes. `shingles` supplies both overlap
measures the lexical tier decides on, `block_shingles` is that tier's own inverted index expressed
as an exploding blocking rule, and the governance fields are the ones `compare_editions` orders an
edition pair by. Nothing here re-derives a signal the audit did not already have -- what the fit
adds is a price for combining them.
"""

from collections.abc import Sequence

from llb.conflicts.corpus import CorpusDoc
from llb.conflicts.linkage.constants import (
    BLOCK_KEY_COLUMN,
    BLOCK_KEY_VALUE,
    BLOCK_SHINGLES_COLUMN,
    DATE_GAP_DAYS,
    DOC_ID_COLUMN,
    EFFECTIVE_DATE_COLUMN,
    LADDER_STEP,
    LEVEL_PROBABILITY_FLOOR,
    RETAIN_COLUMNS,
    SHINGLES_COLUMN,
    SOURCE_SYSTEM_COLUMN,
    TITLE_COLUMN,
    TITLE_SIMILARITY_THRESHOLDS,
)
from llb.core.contracts.common import JsonObject
from llb.linkage.constants import (
    KIND_DATE_DIFFERENCE,
    KIND_EXACT,
    KIND_JARO_WINKLER,
    KIND_SET_OVERLAP,
)
from llb.linkage.comparison_spec import ComparisonSpec
from llb.linkage.spec import BlockingRule, LinkageSpec

_HEADING = "#"


def document_title(doc: CorpusDoc) -> str:
    """The document's first heading, case-folded and whitespace-collapsed.

    Case-folded because a re-issue that differs only in case is exactly the edition the hash tier
    calls a normalized duplicate -- comparing the titles verbatim would score that pair as a title
    DISAGREEMENT, which is the opposite of what it is evidence of.
    """
    for line in doc.body.splitlines():
        stripped = line.strip()
        if stripped.startswith(_HEADING):
            return " ".join(stripped.lstrip(_HEADING).split()).casefold()
    return ""


def _hashes(values: set[int]) -> list[str]:
    """Shingle hashes as text, sorted, so the same document always materialises the same array."""
    return sorted(format(value, "x") for value in values)


def discriminative(doc_shingles: list[set[int]], max_doc_frequency: float) -> list[set[int]]:
    """Each document's shingles minus the ones too common to block on.

    The same rule `candidate_pairs` applies, applied per document instead of over the inverted
    index, so an exploding blocking rule over these arrays generates exactly the pairs that
    function returns -- one candidate list, not two that have to be kept in step.
    """
    frequency: dict[int, int] = {}
    for shingle_set in doc_shingles:
        for shingle in shingle_set:
            frequency[shingle] = frequency.get(shingle, 0) + 1
    limit = max(2, int(max_doc_frequency * len(doc_shingles)))
    return [{s for s in shingle_set if frequency[s] <= limit} for shingle_set in doc_shingles]


def build_records(
    docs: Sequence[CorpusDoc], doc_shingles: Sequence[set[int]], max_doc_frequency: float
) -> list[JsonObject]:
    """One record per corpus document, in corpus order, with both shingle arrays materialised."""
    blocking = discriminative(list(doc_shingles), max_doc_frequency)
    return [
        {
            DOC_ID_COLUMN: doc.doc_id,
            SHINGLES_COLUMN: _hashes(whole),
            BLOCK_SHINGLES_COLUMN: _hashes(block),
            TITLE_COLUMN: document_title(doc),
            SOURCE_SYSTEM_COLUMN: str(doc.governance.get("source_system") or ""),
            EFFECTIVE_DATE_COLUMN: str(doc.governance.get("effective_date") or ""),
            BLOCK_KEY_COLUMN: BLOCK_KEY_VALUE,
        }
        for doc, whole, block in zip(docs, doc_shingles, blocking)
    ]


def _ladder(cutoff: float) -> tuple[float, ...]:
    """The tier's own cutoff, then one rung below it (never below zero, never a repeat)."""
    lower = round(cutoff - LADDER_STEP, 4)
    return (cutoff,) if lower <= 0.0 else (cutoff, lower)


def build_edition_spec(
    *,
    jaccard_threshold: float,
    containment_threshold: float,
    match_threshold: float,
    random_match_probability: float,
) -> LinkageSpec:
    """The document-edition comparison specification at this run's own lexical cutoffs.

    The top rung of each overlap ladder IS the cutoff the lexical tier decides on, so the fit and
    the thresholds are read off the same two numbers and a difference between them is a difference
    in how the evidence is COMBINED rather than in what was measured.
    """
    spec = LinkageSpec(
        comparisons=(
            ComparisonSpec(
                column=SHINGLES_COLUMN,
                kind=KIND_SET_OVERLAP,
                thresholds=_ladder(jaccard_threshold),
                containment_thresholds=_ladder(containment_threshold),
            ),
            ComparisonSpec(
                column=TITLE_COLUMN,
                kind=KIND_JARO_WINKLER,
                thresholds=TITLE_SIMILARITY_THRESHOLDS,
            ),
            ComparisonSpec(column=SOURCE_SYSTEM_COLUMN, kind=KIND_EXACT),
            ComparisonSpec(
                column=EFFECTIVE_DATE_COLUMN,
                kind=KIND_DATE_DIFFERENCE,
                thresholds=DATE_GAP_DAYS,
            ),
        ),
        blocking_rules=(
            BlockingRule((BLOCK_SHINGLES_COLUMN,), arrays_to_explode=(BLOCK_SHINGLES_COLUMN,)),
        ),
        # One pass over every pair. An edition changes its date and often its source system, so
        # blocking the training pass on either would hold fixed the very field an edition varies
        # and hide the pairs the fit exists to price.
        training_rules=(BlockingRule((BLOCK_KEY_COLUMN,)),),
        retain_columns=RETAIN_COLUMNS,
        match_threshold=match_threshold,
        random_match_probability=random_match_probability,
        min_level_probability=LEVEL_PROBABILITY_FLOOR,
        # The shingle arrays are the comparison's input, not provenance: carried into the pair
        # table they would make every scored row hold two whole shingle sets.
        retain_matching_columns=False,
    )
    spec.validate()
    return spec
